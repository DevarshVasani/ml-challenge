import csv
from pathlib import Path

import pyarrow.parquet as pq

from src import reverse_retrieval as rr


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=["entity_id","business_name","business_address","country"],delimiter="\t")
        writer.writeheader();writer.writerows(rows)


def test_reverse_full_background_string_ids_and_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(rr,"DATASET",tmp_path/"dataset")
    base=tmp_path/"dataset"/"train"
    _write(base/"train_source1.tsv",[
        {"entity_id":"S1-0001","business_name":"Cedar Grove Bakery","business_address":"12 Pine Road","country":"US"},
        {"entity_id":"S1-2","business_name":"Harbor Dental Clinic","business_address":"42 Lake Street","country":"US"},
        {"entity_id":"S1-3","business_name":"Lotus Traders","business_address":"9 Market Road","country":"India"},
    ])
    _write(base/"train_source2.tsv",[
        {"entity_id":"S2-shared","business_name":"Cedar Grove Bakery","business_address":"12 Pine Road","country":"US"},
        {"entity_id":"S2-other","business_name":"Harbor Dental Clinic","business_address":"42 Lake Street","country":"US"},
    ])
    _write(base/"train_source3.tsv",[
        {"entity_id":"S3-shared","business_name":"Cedar Grove Bakery","business_address":"12 Pine Road","country":"US"},
    ])
    root=tmp_path/"index";rr.build(root,"train",chunk=1)
    rr.retrieve(root,"train",tmp_path/"resumed",k=2,batch_size=1,limit=2)
    rr.retrieve(root,"train",tmp_path/"resumed",k=2,batch_size=1)
    rr.retrieve(root,"train",tmp_path/"straight",k=2,batch_size=1)
    def pairs(folder):
        rows=[]
        for path in sorted(folder.glob("*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
        keys=[(r["source"],r["source_record_id"],r["s1_id"]) for r in rows]
        assert len(keys)==len(set(keys))
        return set(keys)
    assert pairs(tmp_path/"resumed")==pairs(tmp_path/"straight")
    assert ("S2","S2-shared","S1-0001") in pairs(tmp_path/"resumed")
    assert ("S3","S3-shared","S1-0001") in pairs(tmp_path/"resumed")
