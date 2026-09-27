"""Prepare unlabeled source rows for a bounded production-code retrieval replay."""
import csv
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval_experiments import gt, load_pairs, selected_text, write_json


def main():
    root = Path("artifacts/handoff-retrieval-v1")
    ids = (root / "threshold_query_ids_50.txt").read_text().splitlines()
    manifest, _, _, _ = load_pairs("threshold", verify=False)
    if len(ids) != 50 or not set(ids) <= set(manifest["query_ids"]):
        raise ValueError("handoff query selection invalid")
    truth = gt(ids)
    endpoint_ids = sorted({cid for q in ids for cid in truth[q]})
    content = selected_text(endpoint_ids)
    rows = [{"source": cid.split("-", 1)[0], "entity_id": cid,
             "business_name": content[cid][0], "business_address": content[cid][1],
             "country": content[cid][2]} for cid in endpoint_ids]
    if len(rows) != len(endpoint_ids):
        raise ValueError("duplicate source endpoint")
    target = root / "source_endpoints_50_queries.parquet"
    if not target.exists() or pq.read_table(target).to_pylist() != rows:
        pq.write_table(pa.Table.from_pylist(rows), target, compression="zstd")
    for source in ("S2", "S3"):
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=["entity_id", "business_name", "business_address", "country"], delimiter="\t")
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows if r["source"] == source)
        path = root / f"{source}_source_endpoints.tsv"
        content = handle.getvalue()
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    write_json(root / "source_subset_manifest.json", {"queries": 50, "source_records": len(rows),
                                                        "S2": sum(r["source"] == "S2" for r in rows),
                                                        "S3": sum(r["source"] == "S3" for r in rows),
                                                        "positive_only_diagnostic": True})
    print(json.dumps({"source_records": len(rows), "sample": str(target)}))


if __name__ == "__main__":
    main()
