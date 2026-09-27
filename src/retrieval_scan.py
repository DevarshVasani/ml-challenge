"""One-pass, full-background exact-key delta retrieval for a held-out split."""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_experiments import DATA, ROOT, SOURCES, gt, load_pairs, manifest, norm, score, sha, sim, suffix_key, write_json, rows

MAX_POSTINGS = 20


def collect_inputs(split):
    m=manifest(split,verify=False)
    query={}
    seeds=defaultdict(list)
    for r in rows(m,["source1_entity_id","candidate_entity_id","candidate_source","name_a","address_a","country_a",
                     "name_b","address_b","country_b","name_tfidf_score","address_tfidf_score"]):
        q=r["source1_entity_id"]
        query[q]=(r["name_a"],r["address_a"],r["country_a"])
        ns=sim(r["name_a"],r["name_b"])
        ads=sim(r["address_a"],r["address_b"])
        agreement=.65*ns+.35*ads
        if ns>=.70 or (ns>=.45 and ads>=.65):
            item=(agreement,r["candidate_entity_id"],r["name_b"],r["address_b"])
            current=seeds[q]
            if item not in current:
                current.append(item)
                current.sort(key=lambda x:(-x[0],x[1]))
                del current[2:]
    if set(query)!=set(m["query_ids"]):
        raise ValueError("query text coverage incomplete")
    return m,query,seeds


def keys_for(name,address,kind):
    if kind=="query_folded":
        k=norm(name)
        return [("name",k)] if len(k)>=8 and len(k.split())>=2 else []
    if kind=="query_suffix":
        k=suffix_key(name)
        return [("suffix",k)] if k else []
    if kind=="seed_name":
        k=norm(name)
        return [("name",k)] if len(k)>=8 and len(k.split())>=2 else []
    if kind=="seed_address":
        k=norm(address)
        return [("address",k)] if len(k)>=16 and len(k.split())>=4 else []
    raise ValueError(kind)


def scan(split,out,limit_rows=0):
    start=time.time()
    m,query,seeds=collect_inputs(split)
    requested=defaultdict(set)
    for q,(name,address,_) in query.items():
        for kind in ("query_folded","query_suffix"):
            for key in keys_for(name,address,kind): requested[key].add((q,kind))
        for _,sid,sname,saddress in seeds[q]:
            for kind in ("seed_name","seed_address"):
                for key in keys_for(sname,saddress,kind): requested[key].add((q,kind))
    postings=defaultdict(list); frequency=defaultdict(int)
    source_rows={}; visited={}
    for source in ("S2","S3"):
        path=SOURCES/f"train_source{source[-1]}.tsv"
        count=0
        with open(path,newline="") as f:
            for r in csv.DictReader(f,delimiter="\t"):
                count+=1
                if limit_rows and count>limit_rows: break
                name=norm(r["business_name"]);address=norm(r["business_address"])
                for key in set((("name",name),("suffix",suffix_key(r["business_name"])),("address",address))):
                    if key not in requested: continue
                    frequency[key]+=1
                    if len(postings[key])<MAX_POSTINGS+1:
                        postings[key].append((source,r["entity_id"]))
                        source_rows[(source,r["entity_id"])]=(r["business_name"],r["business_address"],r["country"])
        visited[source]=count if not limit_rows else min(count,limit_rows)
    additions=defaultdict(lambda:defaultdict(set))
    for key,targets in requested.items():
        if frequency[key]>MAX_POSTINGS: continue
        for q,kind in targets:
            for source,cid in postings[key]: additions[kind][q].add((source,cid))
    return m,query,seeds,additions,source_rows,{"seconds":time.time()-start,"visited":visited,
        "peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "requested_keys":len(requested),"overfull_keys":sum(v>MAX_POSTINGS for v in frequency.values()),
        "postings_stored":sum(map(len,postings.values())),"source_identities":{
            s:{"size":(SOURCES/f"train_source{s[-1]}.tsv").stat().st_size,
               "sha256":sha(SOURCES/f"train_source{s[-1]}.tsv") if not limit_rows else None}
            for s in ("S2","S3")}}


def evaluate(split,out,m,additions,baseline,chosen=None):
    truth=gt(m["query_ids"])
    results={}; baseline_pairs=sum(map(len,baseline.values()))
    arms={"query_folded":["query_folded"],"query_suffix":["query_suffix"],
          "seed":["seed_name","seed_address"],"combination":["query_folded","query_suffix","seed_name","seed_address"]}
    for arm,kinds in arms.items():
        if split=="final" and arm!=chosen:
            continue
        merged={q:set(baseline[q]) for q in m["query_ids"]}
        for kind in kinds:
            for q,values in additions[kind].items(): merged[q].update(x[1] for x in values)
        metric=score(m,merged,truth)
        metric["additional_pairs"]=sum(map(len,merged.values()))-baseline_pairs
        metric["mean_candidate_growth_ratio"]=metric["candidate_count"]["mean"]/(baseline_pairs/len(merged))
        results[arm]=metric
    return results


def export(split,out,m,query,additions,source_rows,baseline,chosen,scan_info):
    kinds={"query_folded":["query_folded"],"query_suffix":["query_suffix"],
           "seed":["seed_name","seed_address"],"combination":["query_folded","query_suffix","seed_name","seed_address"],
           "baseline":[]}[chosen]
    by_pair=defaultdict(set)
    for kind in kinds:
        for q,ids in additions[kind].items():
            for source,cid in ids:
                if cid not in baseline[q]:by_pair[(q,source,cid)].add(kind)
    truth=gt(m["query_ids"])
    metadata=json.loads((DATA/"query_manifest.json").read_text())["query_metadata"]
    records=[]
    for (q,source,cid),origins in sorted(by_pair.items()):
        an,aa,ac=query[q];bn,ba,bc=source_rows[(source,cid)]
        records.append({"source1_entity_id":q,"candidate_entity_id":cid,"candidate_source":source,
                        "name_a":an,"address_a":aa,"country_a":ac,"name_b":bn,"address_b":ba,"country_b":bc,
                        "source1_fold":int(metadata[q]["fold"]),
                        "candidate_fold":int(metadata[q]["fold"]) if cid in truth[q] else None,
                        "label":int(cid in truth[q]),"retrieval_provenance":"experiment:"+",".join(sorted(origins)),
                        "name_tfidf_score":None,"address_tfidf_score":None,"name_rank":None,"address_rank":None,
                        "retrieved_by_name":int(bool(origins&{"query_folded","query_suffix","seed_name"})),
                        "retrieved_by_address":int("seed_address" in origins),"retrieval_score":None,
                        "retrieved_by_exact":1,"frequent_key_expansion":0})
    shard=out/f"{split}_delta_pairs.parquet"
    tmp=shard.with_suffix(".tmp")
    schema=pq.ParquetFile(DATA/m["shard_order"][0]).schema_arrow.remove_metadata()
    pq.write_table(pa.Table.from_pylist(records,schema=schema),tmp,compression="zstd")
    os.replace(tmp,shard)
    chosen_config={"policy":chosen,"max_postings":MAX_POSTINGS,"seed_rule":"name token Jaccard >= .70 or name >= .45 and address >= .65; top two by .65 name + .35 address", "query_keys":"accent-folded alphanumeric tokens; trailing legal suffix removal", "source_background":"complete train S2/S3 TSV", "source_identities":scan_info["source_identities"]}
    ph=__import__("hashlib").sha256(json.dumps(chosen_config,sort_keys=True).encode()).hexdigest()
    write_json(out/f"{split}_delta_manifest.json",{"complete":True,"split":split,"query_ids":m["query_ids"],
        "covered_query_count":len(m["query_ids"]),"zero_addition_query_count":len(m["query_ids"])-len({q for q,_,_ in by_pair}),
        "row_count":len(records),"shard":shard.name,"sha256":sha(shard),"parent_manifest":f"{split}_pairs_manifest.json",
        "parent_manifest_sha256":sha(DATA/f"{split}_pairs_manifest.json"),"policy_hash":ph,
        "config":chosen_config,"scan":scan_info})
    return ph,len(records)


def main():
    p=argparse.ArgumentParser();p.add_argument("split",choices=["threshold","final"])
    p.add_argument("--out",type=Path,required=True);p.add_argument("--limit-rows",type=int,default=0)
    p.add_argument("--chosen",choices=["baseline","query_folded","query_suffix","seed","combination"])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    state=args.out/f"{args.split}_{'pilot' if args.limit_rows else 'full'}_scan_state.json"
    write_json(state,{"completion":"in_progress","split":args.split,"limit_rows_per_source":args.limit_rows,
                      "chosen":args.chosen,"query_ids":manifest(args.split,verify=False)["query_ids"]})
    m,query,seeds,additions,source_rows,info=scan(args.split,args.out,args.limit_rows)
    if args.limit_rows:
        write_json(args.out/f"{args.split}_pilot.json",info)
        write_json(state,{"completion":"complete","split":args.split,"limit_rows_per_source":args.limit_rows,
                          "query_ids":m["query_ids"],"scan":info})
        return
    _,baseline,_,_=load_pairs(args.split,verify=False,text=False)
    if args.split=="final" and not args.chosen:
        raise ValueError("final evaluation requires a frozen --chosen policy")
    metrics=evaluate(args.split,args.out,m,additions,baseline,args.chosen)
    write_json(args.out/f"{args.split}_arms.json",{"scan":info,"arms":metrics,"seed_coverage_queries":sum(bool(seeds[q]) for q in m["query_ids"])})
    if args.chosen:
        export(args.split,args.out,m,query,additions,source_rows,baseline,args.chosen,info)
    write_json(state,{"completion":"complete","split":args.split,"limit_rows_per_source":0,
                      "chosen":args.chosen,"query_ids":m["query_ids"],"scan":info})


if __name__=="__main__":main()
