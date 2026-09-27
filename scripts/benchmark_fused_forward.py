"""Threshold-only gate for a fused, full-background historical forward search."""

import gc
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from src.candidate_index import SqliteExactMatchIndex
from src.fused_forward import GpuGroupedForwardIndex
from src.retrieval_experiments import gt, load_pairs, score, selected_text, sha, write_json


def main() -> None:
    started = time.time()
    manifest, baseline, _, _ = load_pairs("threshold", verify=False)
    truth = gt(manifest["query_ids"])
    text = selected_text(set(manifest["query_ids"]))
    query = pd.DataFrame([{"entity_id": q, "business_name": text[q][0],
                           "business_address": text[q][1]} for q in manifest["query_ids"]])
    hits = {q: set() for q in manifest["query_ids"]}
    reverse = {q: set() for q in manifest["query_ids"]}
    for path, k in (("artifacts/reverse-v1/pilot_v5_all/positive_ranks.parquet", 50),
                    ("artifacts/dense-v1/pilot/positive_ranks.parquet", 20)):
        for row in pq.read_table(path, columns=["s1_id", "source_record_id", "rank"]).to_pylist():
            if row["rank"] is not None and row["rank"] <= k:
                reverse[row["s1_id"]].add(row["source_record_id"])
    root = Path("artifacts/candidate-index")
    query_hash = hashlib.sha256(json.dumps(manifest["query_ids"]).encode()).hexdigest()
    idf_by_field = {"business_name": 6.0, "business_address": 4.0}
    result = {"version": "gpu-idf-pruned-forward-threshold-gate-v3", "split": "threshold",
              "queries": len(query), "k_per_field": 128, "batch_size": 256,
              "min_idf_by_field": idf_by_field,
              "candidate_index_manifest_sha256": sha(root / "manifest.json"), "channels": {}}
    for source in ("S2", "S3"):
        for field in ("business_name", "business_address"):
            channel = f"{source}_{field}"
            min_idf = idf_by_field[field]
            checkpoint_path = Path("artifacts/gpu-forward-v6") / f"threshold_{channel}.json"
            checkpoint_identity = {"channel": channel, "query_ids_sha256": query_hash,
                                   "index_metadata_sha256": sha(root / channel / "disk_metadata.json"),
                                   "k_per_field": 128, "batch_size": 256,
                                   "group_shards": 64, "local_k": 512, "min_idf": min_idf}
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text())
                if checkpoint.get("identity") != checkpoint_identity:
                    raise ValueError(f"incompatible fused forward checkpoint: {checkpoint_path}")
                for qid, candidate in checkpoint["positive_pairs"]:
                    hits[qid].add(candidate)
                result["channels"][channel] = checkpoint["metrics"]
                print(json.dumps({"resumed_channel": channel}), flush=True)
                continue
            began = time.time()
            index = GpuGroupedForwardIndex(root / channel, group_shards=64, local_k=512,
                                           min_idf=min_idf)
            built = time.time() - began
            pairs = 0
            channel_hits = set()
            for start in range(0, len(query), 256):
                batch = query.iloc[start:start + 256]
                output = index.query(batch, candidate_source=source, top_k=128, batch_size=256)
                pairs += len(output)
                for qid, candidate in zip(output["source1_entity_id"], output["candidate_entity_id"]):
                    if candidate in truth[qid]:
                        channel_hits.add((qid, candidate))
            exact = SqliteExactMatchIndex(root / channel / "exact_sqlite.db", field=field,
                                          candidate_source=source)
            exact_rows = exact.query(query, field=field)
            for qid, candidate in zip(exact_rows["source1_entity_id"], exact_rows["candidate_entity_id"]):
                if candidate in truth[qid]:
                    channel_hits.add((qid, candidate))
            exact.close()
            for qid, candidate in channel_hits:
                hits[qid].add(candidate)
            result["channels"][channel] = {"seconds": time.time() - began,
                                            "matrix_setup_seconds": built,
                                            "tfidf_pairs_before_union": pairs,
                                            "channel_positive_hits": len(channel_hits),
                                            "resident_matrix_bytes": sum(int(s.get("resident_bytes", 0)) for s in index.disk.shard_specs)}
            write_json(checkpoint_path, {"identity": checkpoint_identity,
                                         "metrics": result["channels"][channel],
                                         "positive_pairs": sorted(channel_hits)})
            write_json("artifacts/gpu-forward-v6/threshold_progress.json", result)
            print(json.dumps({"channel": channel, **result["channels"][channel]}), flush=True)
            index.close()
            del index, exact_rows
            gc.collect()
    fused_reverse = {q: hits[q] | reverse[q] for q in hits}
    historical = {q: baseline[q] | reverse[q] for q in hits}
    result["historical_forward_reverse_oracle"] = score(manifest, historical, truth)["oracle_macro_f05"]
    result["fused_forward_reverse_oracle"] = score(manifest, fused_reverse, truth)["oracle_macro_f05"]
    result["historical_positive_edges_not_recovered"] = sum(
        len((baseline[q] & truth[q]) - fused_reverse[q]) for q in hits)
    result["seconds_total"] = time.time() - started
    result["positive_only_oracle_diagnostic"] = True
    write_json("artifacts/gpu-forward-v6/threshold_gate.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
