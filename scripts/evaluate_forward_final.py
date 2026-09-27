"""One aggregate-only final evaluation after the forward budget is frozen on threshold."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import pyarrow as pa

from src.forward_fts import ForwardIndex
from src.retrieval_experiments import gt, load_pairs, score, selected_text, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, choices=[5, 10, 20, 50], required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/forward-fts-v2/frozen_final_aggregate.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("frozen final forward policy was already evaluated")
    start = time.time()
    union = Path("artifacts/reverse-v1/train_union")
    union_meta = json.loads((union / "manifest.json").read_text())
    if not union_meta["complete"] or union_meta["source_groups"] != 10_320_219:
        raise ValueError("complete train reverse union is required")
    manifest, baseline, _, _ = load_pairs("final", verify=False)
    truth = gt(manifest["query_ids"])
    qtext = selected_text(set(manifest["query_ids"]))
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.register("wanted", pa.table({"s1_id": manifest["query_ids"]}))
    candidates = {q: set() for q in manifest["query_ids"]}
    parquet_glob = str((union / "bucket-*.parquet").resolve()).replace("'", "''")
    for batch in db.execute(f"SELECT r.s1_id,r.source_record_id FROM read_parquet('{parquet_glob}') r INNER JOIN wanted w ON r.s1_id=w.s1_id").fetch_record_batch(100_000):
        for q, cid in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            candidates[q].add(cid)
    db.close()
    index = ForwardIndex(Path("artifacts/forward-fts-v2"), "train")
    for q in manifest["query_ids"]:
        name, address, country = qtext[q]
        for hit in index.query(name, address, country, 50):
            if (hit["name_rank"] is not None and hit["name_rank"] <= args.k) or (
                    hit["address_rank"] is not None and hit["address_rank"] <= args.k):
                candidates[q].add(hit["source_record_id"])
    index.close()
    historical = {q: candidates[q] | baseline[q] for q in manifest["query_ids"]}
    output = {"policy": {"forward_fts_k_per_field": args.k,
                         "reverse": "lexical50+dense20", "split": "final"},
              "full_background_complete_validation": True,
              "forward_reverse": score(manifest, candidates, truth),
              "historical_baseline_plus_forward_reverse_diagnostic": score(manifest, historical, truth),
              "inputs_sha256": {"train_reverse_union": sha(union / "manifest.json")},
              "seconds": time.time() - start}
    write_json(args.output, output)
    print(json.dumps({"forward_reverse_oracle": output["forward_reverse"]["oracle_macro_f05"],
                      "forward_reverse_recall": output["forward_reverse"]["positive_pair_recall"],
                      "candidate_count_mean": output["forward_reverse"]["candidate_count"]["mean"],
                      "seconds": output["seconds"]}))


if __name__ == "__main__":
    main()
