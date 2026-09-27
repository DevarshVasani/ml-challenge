"""Validate a held-out delta and publish baseline UNION delta shard references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .retrieval_experiments import DATA, gt, load_pairs, manifest, score, sha, write_json


def union_pairs(baseline, delta):
    merged={q:set(ids) for q,ids in baseline.items()}
    for q,cid in delta:
        if q not in merged:
            raise ValueError(f"delta query outside parent: {q}")
        if cid in merged[q]:
            raise ValueError(f"delta duplicates baseline pair: {q}/{cid}")
        merged[q].add(cid)
    return merged


def validate(split,out):
    out=Path(out)
    base=manifest(split,verify=True)
    dm=json.loads((out/f"{split}_delta_manifest.json").read_text())
    shard=out/dm["shard"]
    if not dm["complete"] or dm["query_ids"]!=base["query_ids"]:
        raise ValueError("delta query coverage incomplete or wrong order")
    if dm["parent_manifest_sha256"]!=sha(DATA/f"{split}_pairs_manifest.json"):
        raise ValueError("parent baseline changed")
    if dm["sha256"]!=sha(shard) or dm["row_count"]!=pq.ParquetFile(shard).metadata.num_rows:
        raise ValueError("delta checksum or row count differs")
    _,baseline,_,_=load_pairs(split,verify=False)
    table=pq.read_table(shard,columns=["source1_entity_id","candidate_entity_id","candidate_source","label"])
    seen=set(); delta=[]
    truth=gt(base["query_ids"])
    for r in table.to_pylist():
        q,cid,source=r["source1_entity_id"],r["candidate_entity_id"],r["candidate_source"]
        key=(q,source,cid)
        if key in seen or source not in ("S2","S3") or not cid.startswith(source+"-"):
            raise ValueError("duplicate or invalid delta pair")
        if r["label"]!=int(cid in truth[q]):
            raise ValueError("held-out label differs from ground truth")
        seen.add(key);delta.append((q,cid))
    merged=union_pairs(baseline,delta)
    if len({q for q,_ in delta})+dm["zero_addition_query_count"]!=len(base["query_ids"]):
        raise ValueError("zero-addition coverage differs")
    result={"completion":"complete","split":split,"policy_hash":dm["policy_hash"],
            "semantics":"baseline UNION delta; score each distinct (query, candidate_source, candidate_entity_id) once",
            "baseline_manifest":str(DATA/f"{split}_pairs_manifest.json"),
            "baseline_shards":[str(DATA/p) for p in base["shard_order"]],"delta_shard":str(shard),
            "row_count_delta":len(delta),"query_ids":base["query_ids"],"metrics":score(base,merged,truth)}
    write_json(out/f"{split}_merged_manifest.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["threshold","final"])
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();r=validate(a.split,a.out)
    print(json.dumps({k:r[k] for k in ("split","row_count_delta","policy_hash")},indent=2))
