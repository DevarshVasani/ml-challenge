"""Attach seed and query-variant lineage to an already validated delta shard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_experiments import norm, sha, write_json
from .retrieval_scan import collect_inputs, keys_for


def main(split,out):
    out=Path(out)
    dm=json.loads((out/f"{split}_delta_manifest.json").read_text())
    if not dm["complete"] or sha(out/dm["shard"])!=dm["sha256"]:
        raise ValueError("delta is incomplete or changed")
    _,_,seeds=collect_inputs(split)
    records=[]
    table=pq.read_table(out/dm["shard"],columns=["source1_entity_id","candidate_entity_id","candidate_source",
                                              "name_b","address_b","retrieval_provenance"])
    for r in table.to_pylist():
        q=r["source1_entity_id"]
        kinds=set(r["retrieval_provenance"].split(":",1)[1].split(","))
        used=[]
        for _,sid,name,address in seeds[q]:
            if "seed_name" in kinds and ("name",norm(r["name_b"])) in keys_for(name,address,"seed_name"):
                used.append(sid)
            elif "seed_address" in kinds and ("address",norm(r["address_b"])) in keys_for(name,address,"seed_address"):
                used.append(sid)
        if kinds&{"seed_name","seed_address"} and not used:
            raise ValueError(f"missing seed lineage for {q}/{r['candidate_entity_id']}")
        records.append({"source1_entity_id":q,"candidate_entity_id":r["candidate_entity_id"],
                        "candidate_source":r["candidate_source"],"seed_entity_ids":sorted(set(used)),
                        "variant_ids":[q+":"+kind for kind in sorted(kinds&{"query_folded","query_suffix"})],
                        "retrieval_channels":sorted(kinds)})
    path=out/f"{split}_delta_provenance.parquet";tmp=path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records),tmp,compression="zstd");os.replace(tmp,path)
    write_json(out/f"{split}_provenance_manifest.json",{"complete":True,"split":split,"row_count":len(records),
        "shard":path.name,"sha256":sha(path),"parent_delta_manifest":f"{split}_delta_manifest.json",
        "parent_delta_sha256":sha(out/f"{split}_delta_manifest.json"),"policy_hash":dm["policy_hash"]})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["threshold","final"])
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();main(a.split,a.out)
