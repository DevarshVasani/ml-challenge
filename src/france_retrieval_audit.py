"""Prepare deterministic French test probes when test retrieval indexes are absent."""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

from .retrieval_experiments import ROOT, norm, write_json
from .text_utils import normalize_text


def perturb(name,address):
    out=[]
    if any(c in "éèêëàâäîïôöùûüçÉÈÊËÀÂÄÎÏÔÖÙÛÜÇ" for c in name):
        folded=norm(name)
        if normalize_text(folded)!=normalize_text(name):out.append(("accent_fold_name",folded,address))
    parts=[x.strip() for x in address.split(",")]
    if len(parts)>1:
        altered=", ".join(parts[:-1])
        if normalize_text(altered)!=normalize_text(address):out.append(("omit_last_address_component",name,altered))
        altered=", ".join(reversed(parts))
        if normalize_text(altered)!=normalize_text(address):out.append(("reverse_address_components",name,altered))
    return out


def main(out):
    root=ROOT.parent/"student_resource"/"dataset"/"test"
    buckets=defaultdict(list); counts=Counter()
    for source in ("S2","S3"):
        with open(root/f"test_source{source[-1]}.tsv",newline="") as f:
            for r in csv.DictReader(f,delimiter="\t"):
                if r["country"]!="France":continue
                name,address=r["business_name"],r["business_address"]
                length="short" if len(name)<16 else "medium" if len(name)<40 else "long"
                missing="missing_address" if not address else "has_address"
                stratum=(source,length,missing);counts[stratum]+=1
                h=int(hashlib.sha256(r["entity_id"].encode()).hexdigest()[:16],16)
                heap=buckets[stratum]
                item=(-h,r["entity_id"],name,address)
                if len(heap)<50:heapq.heappush(heap,item)
                elif item>heap[0]:heapq.heapreplace(heap,item)
    s1_heap=[];s1_count=0
    with open(root/"test_source1.tsv",newline="") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            if r["country"]!="France":continue
            s1_count+=1
            h=int(hashlib.sha256(r["entity_id"].encode()).hexdigest()[:16],16)
            item=(-h,r["entity_id"],r["business_name"],r["business_address"])
            if len(s1_heap)<30:heapq.heappush(s1_heap,item)
            elif item>s1_heap[0]:heapq.heapreplace(s1_heap,item)
    sample=[];probes=[]
    for stratum,heap in sorted(buckets.items()):
        for _,sid,name,address in sorted(heap,reverse=True):
            sample.append({"target_source":stratum[0],"target_entity_id":sid,"business_name":name,
                           "business_address":address,"stratum":"/".join(stratum)})
            for kind,n,a in perturb(name,address):
                probes.append({"synthetic_query_id":"FR-PROBE-"+str(len(probes)+1).zfill(5),
                               "target_source":stratum[0],"known_target_id":sid,"perturbation":kind,
                               "business_name":n,"business_address":a,"country":"France"})
    s1_audit=[{"entity_id":sid,"business_name":name,"business_address":address,
               "normalized_name":normalize_text(name),"normalized_address":normalize_text(address)}
              for _,sid,name,address in sorted(s1_heap,reverse=True)]
    write_json(out/"france_probe_inputs.json",{"sample":sample,"probes":probes,"actual_s1_qualitative_sample":s1_audit})
    probe_tsv=out/"france_probe_queries.tsv"
    with probe_tsv.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["entity_id","business_name","business_address","country"],delimiter="\t")
        writer.writeheader()
        for r in probes:
            writer.writerow({"entity_id":r["synthetic_query_id"],"business_name":r["business_name"],
                             "business_address":r["business_address"],"country":"France"})
    index_config={"top_k":100,"index_backend":"disk_backed","index_workers":1,"sparse_threads":1,
                  "query_batch_size":128,"source_read_chunk_rows":25000,"resume":True,
                  "memory_budget_mb":4096,"max_vocabulary":500000,
                  "sources":{s:{"path":str(root/f"test_source{s[-1]}.tsv"),
                                 "fields":["business_name","business_address"]} for s in ("S2","S3")},
                  "output_dir":str(out/"test_candidate_index")}
    query_config={"s1":str(probe_tsv),"index_root":str(out/"test_candidate_index"),
                  "indexes":{f"{s}:{field}":f"{s}_{field}" for s in ("S2","S3") for field in ("business_name","business_address")},
                  "output_dir":str(out/"france_probe_candidates"),"top_k":100,"sparse_threads":1,
                  "query_batch_size":128,"shard_target_rows":75000,"index_workers":1,"resume":True,
                  "strict_country":False}
    write_json(out/"france_test_index_config.json",index_config)
    write_json(out/"france_probe_query_config.json",query_config)
    write_json(out/"france_robustness.json",{"status":"pending_test_index","sample_records":len(sample),
        "synthetic_probe_count":len(probes),"actual_s1_france_records":s1_count,
        "actual_s1_qualitative_sample_count":len(s1_audit),
        "strata_population":{"/".join(k):v for k,v in sorted(counts.items())},
        "known_target_recovery":None,"actual_france_entity_recall":None,
        "reason":"No complete test-source retrieval indexes or source store were present. Probes retain targets in the full test background for a future run.",
        "pending_commands":[
            f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.build_candidate_index --config {out/'france_test_index_config.json'} --execute",
            f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.query_candidate_index --config {out/'france_probe_query_config.json'} --execute"],
        "next_validation":"Join complete candidate shards to france_probe_inputs.json by synthetic_query_id and compare candidate_entity_id with known_target_id; retain full test background."})


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);main(a.out)
