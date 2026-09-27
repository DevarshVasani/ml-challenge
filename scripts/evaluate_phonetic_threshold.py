"""Threshold-only positive-endpoint pilot for the phonetic name channel."""
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

from src.phonetic_retrieval import PhoneticIndex
from src.retrieval_experiments import gt, load_pairs, score, write_json


def main():
    manifest, baseline, _, _ = load_pairs("threshold", verify=False)
    truth = gt(manifest["query_ids"])
    lex = {(r["s1_id"], r["source_record_id"]): r["rank"] for r in pq.read_table(
        "artifacts/reverse-v1/pilot_v5_all/positive_ranks.parquet").to_pylist()}
    dense = {(r["s1_id"], r["source_record_id"]): r["rank"] for r in pq.read_table(
        "artifacts/dense-v1/pilot/positive_ranks.parquet").to_pylist()}
    existing = {q: set(baseline[q]) for q in manifest["query_ids"]}
    for q in manifest["query_ids"]:
        for cid in truth[q]:
            if (lex.get((q, cid)) is not None and lex[q, cid] <= 50) or (dense.get((q, cid)) is not None and dense[q, cid] <= 20):
                existing[q].add(cid)
    missing = pq.read_table("artifacts/reverse-v1/threshold_all_missing_positives.parquet").to_pylist()
    index = PhoneticIndex(Path("artifacts/phonetic-v2"), "train")
    cached = {}
    for r in missing:
        key = r["candidate_entity_id"]
        if key not in cached:
            cached[key] = [sid for sid, _ in index.query(r["name_b"], r["address_b"], r["country"], 50)]
    index.close()
    ranks = [{"s1_id": r["source1_entity_id"], "source_record_id": r["candidate_entity_id"],
              "rank": cached[r["candidate_entity_id"]].index(r["source1_entity_id"]) + 1
              if r["source1_entity_id"] in cached[r["candidate_entity_id"]] else None}
             for r in missing]
    pq.write_table(pa.Table.from_pylist(ranks), "artifacts/phonetic-v2/threshold_positive_ranks.parquet", compression="zstd")
    output = {"baseline_union": score(manifest, existing, truth), "positive_only_diagnostic": True,
              "source_endpoints_queried": len(cached), "policies": {}}
    output["baseline_union"].pop("candidate_count")
    for k in (5, 10, 20, 50):
        merged = {q: set(ids) for q, ids in existing.items()}
        gains = defaultdict(int)
        for r in missing:
            q, cid = r["source1_entity_id"], r["candidate_entity_id"]
            if cid not in merged[q] and q in cached[cid][:k]:
                merged[q].add(cid)
                gains[r["country"]] += 1
        metrics = score(manifest, merged, truth)
        metrics.pop("candidate_count")
        output["policies"][str(k)] = {"metrics": metrics, "new_positive_edges": sum(gains.values()),
                                        "new_by_country": dict(gains)}
    target = Path("artifacts/phonetic-v2/threshold_pilot.json")
    write_json(target, output)
    print(json.dumps({"baseline_oracle": output["baseline_union"]["oracle_macro_f05"],
                      "policies": {k: {"oracle": v["metrics"]["oracle_macro_f05"], "gains": v["new_positive_edges"]}
                                   for k, v in output["policies"].items()}}))


if __name__ == "__main__":
    main()
