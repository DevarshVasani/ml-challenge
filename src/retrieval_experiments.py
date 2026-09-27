"""Bounded diagnostics and exact-key retrieval experiments on the published pair shards.

Run with ``python -m src.retrieval_experiments --help`` from the repository root.
Only threshold labels inform policy selection. Final failure identities are never emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-data"
SOURCES = ROOT.parent / "student_resource" / "dataset" / "train"
COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source", "label",
        "name_a", "address_a", "country_a", "name_b", "address_b", "country_b",
        "name_tfidf_score", "address_tfidf_score", "name_rank", "address_rank",
        "retrieved_by_exact"]
SUFFIX = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
          "co", "company", "plc", "pvt", "private", "gmbh", "sarl", "sa", "bv"}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(split, verify=True):
    m = json.loads((DATA / f"{split}_pairs_manifest.json").read_text())
    if m["completion"] != "complete" or len(m["query_ids"]) != len(set(m["query_ids"])):
        raise ValueError(f"{split}: incomplete or duplicate queries")
    if set(m["shard_order"]) != set(m["row_counts"]) or set(m["shard_order"]) != set(m["generated_file_checksums"]):
        raise ValueError(f"{split}: shard accounting differs")
    for rel in m["shard_order"]:
        p = DATA / rel
        if not p.is_file() or pq.ParquetFile(p).metadata.num_rows != m["row_counts"][rel]:
            raise ValueError(f"{split}: missing or truncated {rel}")
        if verify and sha(p) != m["generated_file_checksums"][rel]:
            raise ValueError(f"{split}: checksum mismatch {rel}")
    return m


def rows(m, columns):
    for rel in m["shard_order"]:
        pf = pq.ParquetFile(DATA / rel)
        if not set(columns) <= set(pf.schema_arrow.names):
            raise ValueError(f"missing columns in {rel}")
        for batch in pf.iter_batches(batch_size=8192, columns=columns):
            yield from batch.to_pylist()


def gt(ids):
    wanted = set(ids)
    out = {}
    with open(SOURCES / "train_ground_truth.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            q = r["source1_entity_id"]
            if q in wanted:
                out[q] = set(filter(None, r["matched_entity_ids"].split(",")))
    if set(out) != wanted:
        raise ValueError(f"ground truth missing {len(wanted-set(out))} queries")
    return out


def score(m, pairs, truth):
    counts, hit, total, oracle = [], 0, 0, []
    zero, full = 0, 0
    by_country = defaultdict(lambda: [0, 0, 0])
    by_source = defaultdict(lambda: [0, 0])
    meta = json.loads((DATA / "query_manifest.json").read_text())["query_metadata"]
    for q in m["query_ids"]:
        got = pairs.get(q, set())
        true = truth[q]
        h, n = len(got & true), len(true)
        if int(meta[q]["match_count"]) != n:
            raise ValueError(f"manifest match count differs: {q}")
        val = 1.0 if n == 0 else 5*h/(4*h+n)
        oracle.append(val)
        counts.append(len(got))
        hit += h; total += n
        zero += bool(n and not h)
        full += bool(h == n)
        c = by_country[meta[q]["country"]]
        c[0] += 1; c[1] += val; c[2] += n-h
        for source in ("S2", "S3"):
            t = {x for x in true if x.startswith(source + "-")}
            s = by_source[source]
            s[0] += len(got & t); s[1] += len(t)
    sorted_counts = sorted(counts)
    nons = [v for q, v in zip(m["query_ids"], oracle) if truth[q]]
    return {"queries": len(oracle), "non_singletons": len(nons),
            "singletons": len(oracle)-len(nons), "oracle_macro_f05": statistics.mean(oracle),
            "non_singleton_oracle": statistics.mean(nons), "positive_pair_recall": hit/total,
            "zero_positive_queries": zero, "zero_positive_rate": zero/len(nons),
            "full_positive_coverage_rate": full/len(oracle),
            "candidate_count": {"mean": statistics.mean(counts), "p50": sorted_counts[len(counts)//2],
                                "p95": sorted_counts[int(.95*(len(counts)-1))], "max": max(counts)},
            "by_country": {k: {"queries": v[0], "oracle": v[1]/v[0], "missed_edges": v[2]} for k,v in sorted(by_country.items())},
            "by_source": {k: {"hits": v[0], "total": v[1], "recall": v[0]/v[1] if v[1] else 1} for k,v in by_source.items()}}


def load_pairs(split, verify=True, text=False):
    m = manifest(split, verify)
    query_ids = set(m["query_ids"])
    pairs = defaultdict(set)
    qtext = {}
    candidate_text = {}
    columns = COLS if text else COLS[:4]
    for r in rows(m, columns):
        q, cid, source = r["source1_entity_id"], r["candidate_entity_id"], r["candidate_source"]
        if q not in query_ids or source not in ("S2", "S3") or not cid.startswith(source + "-"):
            raise ValueError(f"invalid pair endpoint: {q}, {source}, {cid}")
        if split != "train" and r["label"] not in (0, 1):
            raise ValueError("missing evaluation label")
        pairs[q].add((source,cid))
        if text:
            qtext[q] = (r["name_a"], r["address_a"], r["country_a"])
            candidate_text[(source,cid)] = (r["name_b"], r["address_b"], r["country_b"])
    return m, {q:{x[1] for x in pairs[q]} for q in m["query_ids"]}, qtext, candidate_text


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", s))


def suffix_key(s):
    t = norm(s).split()
    while t and t[-1] in SUFFIX:
        t.pop()
    return " ".join(t) if len(t) >= 2 and len("".join(t)) >= 8 else ""


def sim(a,b):
    x,y=set(norm(a).split()),set(norm(b).split())
    return len(x&y)/len(x|y) if x and y else 0.0


def category(a,b):
    na,aa=a[0],a[1]; nb,ab=b[0],b[1]
    ns,ads=sim(na,nb),sim(aa,ab)
    if na.casefold()!=nb.casefold() and norm(na)==norm(nb) and norm(na):
        c="accent/transliteration"
    elif not na or not nb or not aa or not ab:
        c="missing fields"
    elif norm(na)==norm(nb) or suffix_key(na)==suffix_key(nb) and suffix_key(na):
        c="cutoff/crowding or suffix"
    elif ns >= .5 and ads >= .5:
        c="abbreviation/suffix/punctuation"
    elif ns >= .5 or ads >= .5:
        c="complementary descriptions"
    elif ns < .2 and ads < .2:
        c="weak overlap in both fields"
    else:
        c="unknown"
    return c,ns,ads


def selected_text(ids):
    wanted={"S1":set(),"S2":set(),"S3":set()}
    for x in ids:
        wanted[x.split("-",1)[0]].add(x)
    out={}
    for source in wanted:
        if not wanted[source]: continue
        with open(SOURCES/f"train_source{source[-1]}.tsv",newline="") as f:
            for r in csv.DictReader(f,delimiter="\t"):
                if r["entity_id"] in wanted[source]:
                    out[r["entity_id"]]=(r["business_name"],r["business_address"],r["country"])
    if len(out)!=len(ids):
        raise ValueError(f"selected text missing {len(ids)-len(out)} IDs")
    return out


def baseline(out):
    started=time.time()
    results={}; threshold=None
    for split in ("threshold","final"):
        m,pairs,_,_=load_pairs(split, text=False)
        truth=gt(m["query_ids"])
        results[split]=score(m,pairs,truth)
        if split=="threshold": threshold=(m,pairs,truth)
    stated={"threshold":.9873426133521909,"final":.9903324939612327}
    for split in stated:
        if abs(results[split]["oracle_macro_f05"]-stated[split])>1e-8:
            raise ValueError(f"{split} baseline mismatch: {results[split]['oracle_macro_f05']}")
    results["elapsed_seconds"]=time.time()-started
    write_json(out/"baseline_metrics.json",results)
    return threshold


def failures(out, threshold):
    m,pairs,truth=threshold
    missed=[]
    for q in m["query_ids"]:
        n=len(truth[q]);h=len(truth[q]&pairs[q]);loss=0 if n==0 else 1-5*h/(4*h+n)
        for cid in sorted(truth[q]-pairs[q]):
            missed.append((q,cid,n,h,loss))
    text=selected_text({q for q,*_ in missed}|{cid for _,cid,*_ in missed})
    meta=json.loads((DATA/"query_manifest.json").read_text())["query_metadata"]
    records=[]
    for q,cid,n,h,loss in missed:
        a,b=text[q],text[cid]
        cat,ns,ads=category(a,b)
        records.append({"source1_entity_id":q,"candidate_entity_id":cid,"candidate_source":cid.split("-",1)[0],
                        "country":meta[q]["country"],"name_a":a[0],"address_a":a[1],"name_b":b[0],"address_b":b[1],
                        "match_count":n,"retrieved_positive_count":h,"query_oracle_loss":loss,
                        "name_token_jaccard":ns,"address_token_jaccard":ads,"hypothesis":cat})
    pq.write_table(pa.Table.from_pylist(records),out/"failure_analysis.parquet")
    counts=Counter(r["hypothesis"] for r in records)
    unique={r["source1_entity_id"] for r in records}
    summary=[f"# Threshold failure analysis\n",f"{len(records)} missed edges in {len(unique)} queries. Category labels are hypotheses.\n",
             "Query loss is counted once per query in the breakdown below.\n"]
    groups=defaultdict(lambda: [set(),0])
    for r in records:
        for key in ("country:"+r["country"],"source:"+r["candidate_source"],
                    "cardinality:"+("1" if r["match_count"]==1 else "2-3" if r["match_count"]<=3 else "4+"),
                    "hypothesis:"+r["hypothesis"]):
            groups[key][0].add(r["source1_entity_id"])
    qloss={r["source1_entity_id"]:r["query_oracle_loss"] for r in records}
    for key,(qs,_) in sorted(groups.items()):
        summary.append(f"- {key}: {len(qs)} affected queries; sum of their losses {sum(qloss[q] for q in qs):.6f}\n")
    summary.append("\nMissed edges by hypothesis: "+json.dumps(counts,sort_keys=True)+"\n")
    (out/"failure_summary.md").write_text("".join(summary))
    return records


def training_failures(out):
    m=manifest("train",verify=True)
    cols=["source1_entity_id","candidate_entity_id","candidate_source","name_a","address_a","country_a",
          "name_b","address_b","country_b","positive_injected_for_training","label"]
    found=set(); records=[]
    for r in rows(m,cols):
        if r["positive_injected_for_training"] not in (0,1):
            raise ValueError("missing training injection flag")
        if r["positive_injected_for_training"]!=1: continue
        if r["label"]!=1:
            raise ValueError("injected row is not labeled positive")
        key=(r["source1_entity_id"],r["candidate_entity_id"])
        if key in found: continue
        found.add(key)
        cat,ns,ads=category((r["name_a"],r["address_a"]),(r["name_b"],r["address_b"]))
        records.append({**{k:r[k] for k in cols if k not in ("positive_injected_for_training","label")},
                        "name_token_jaccard":ns,"address_token_jaccard":ads,"hypothesis":cat})
    target=out/"training_injected_failures.parquet"
    pq.write_table(pa.Table.from_pylist(records),target,compression="zstd")
    # Balanced deterministic example set; category/country/source strata are retained.
    groups=defaultdict(list)
    for r in records:
        groups[(r["country_a"],r["candidate_source"],r["hypothesis"])].append(r)
    for group in groups.values():
        group.sort(key=lambda r:hashlib.sha256((r["source1_entity_id"]+r["candidate_entity_id"]).encode()).hexdigest())
    sample=[]
    while len(sample)<100 and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(sample)<100:sample.append(groups[key].pop())
    pq.write_table(pa.Table.from_pylist(sample),out/"training_failure_sample.parquet",compression="zstd")
    write_json(out/"training_failure_summary.json",{"injected_positive_pairs":len(records),
        "sample_rows":len(sample),"category_counts":dict(Counter(r["hypothesis"] for r in records)),
        "country_counts":dict(Counter(r["country_a"] for r in records)),
        "source_counts":dict(Counter(r["candidate_source"] for r in records)),
        "interpretation":"Injected positives are retrieval misses, not a complete retrieval candidate set."})


def calibrate_seeds(out):
    m=manifest("train",verify=False)
    selected=defaultdict(list)
    injected=0
    columns=["source1_entity_id","candidate_entity_id","name_a","address_a","name_b","address_b",
             "label","positive_injected_for_training"]
    for r in rows(m,columns):
        if r["positive_injected_for_training"] not in (0,1):
            raise ValueError("missing injection flag")
        if r["positive_injected_for_training"]:
            injected+=1
            continue
        ns=sim(r["name_a"],r["name_b"]);ads=sim(r["address_a"],r["address_b"])
        if ns>=.70 or (ns>=.45 and ads>=.65):
            item=(.65*ns+.35*ads,r["candidate_entity_id"],int(r["label"]))
            group=selected[r["source1_entity_id"]]
            group.append(item)
            group.sort(key=lambda x:(-x[0],x[1]))
            del group[2:]
    values=[x for group in selected.values() for x in group]
    write_json(out/"training_seed_calibration.json",{
        "rule":"name token Jaccard >= .70 or name >= .45 and address >= .65; top two by .65 name + .35 address",
        "training_queries":len(m["query_ids"]),"eligible_queries":len(selected),
        "selected_seed_rows":len(values),"selected_seed_labeled_positive":sum(x[2] for x in values),
        "selected_seed_precision_in_sampled_train_pairs":sum(x[2] for x in values)/len(values) if values else None,
        "injected_positives_excluded":injected,
        "limitation":"Training negatives are sampled; precision is not a full-background estimate."})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["baseline","analyze","training","calibrate_seeds"])
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="training": training_failures(args.out)
    elif args.command=="calibrate_seeds":calibrate_seeds(args.out)
    else:
        threshold=baseline(args.out)
        if args.command=="analyze": failures(args.out,threshold)


if __name__=="__main__": main()
