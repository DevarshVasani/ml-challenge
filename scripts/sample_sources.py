"""Deterministic 1% unlabeled source sample, stratified by observable fields."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def script(value):
    for c in value:
        if 0x0900<=ord(c)<=0x097f:return "Devanagari"
    for c in value:
        if 0x0600<=ord(c)<=0x06ff:return "Arabic"
    return "Latin/other"


def main():
    p=argparse.ArgumentParser();p.add_argument("--split",choices=["train","test"],required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]/"student_resource"/"dataset"/a.split
    records=[];seen=Counter();selected=Counter()
    for source in ("S2","S3"):
        path=root/f"{a.split}_source{source[-1]}.tsv"
        with path.open(newline="") as f:
            for row in csv.DictReader(f,delimiter="\t"):
                length=len(row["business_name"])+len(row["business_address"])
                bucket="short" if length<50 else "medium" if length<120 else "long"
                group=(source,row["country"],script(row["business_name"]),bucket)
                seen[group]+=1
                key="|".join(group)+(":"+row["entity_id"])
                if int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),"big")%100:continue
                selected[group]+=1
                records.append({"source":source,"entity_id":row["entity_id"],"business_name":row["business_name"],
                                "business_address":row["business_address"],"country":row["country"],"script":group[2],"length_bucket":bucket})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records),a.output,compression="zstd")
    report={"split":a.split,"rows":len(records),"sampling":"blake2b hash 1% within source/country/script/length strata",
            "strata":[{"source":k[0],"country":k[1],"script":k[2],"length_bucket":k[3],"population":n,"sample":selected[k]} for k,n in sorted(seen.items())]}
    a.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"rows":len(records),"strata":len(seen)}))

if __name__=="__main__":main()
