"""Publish the completed 50-query production-code retrieval replay."""
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval_experiments import gt, score, sha, write_json


ROOT = Path("artifacts/handoff-retrieval-v1")


def main():
    ids = (ROOT / "threshold_query_ids_50.txt").read_text().splitlines()
    truth = gt(ids)
    source_rows = pq.read_table(ROOT / "source_endpoints_50_queries.parquet", columns=["source", "entity_id"]).to_pylist()
    sources = {(r["source"], r["entity_id"]) for r in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("duplicate source record in handoff sample")
    pairs = {}
    input_meta = {}
    for channel in ("lexical", "dense"):
        directory = ROOT / f"{channel}_candidates"
        meta = json.loads((directory / "manifest.json").read_text())
        if not meta["complete"] or sum(s["source_rows"] for s in meta["shards"].values()) != len(sources):
            raise ValueError(f"{channel} sample is incomplete")
        input_meta[channel] = {"manifest_sha256": sha(directory / "manifest.json"), "shards": {}}
        for name, shard in meta["shards"].items():
            path = directory / name
            if sha(path) != shard["sha256"]:
                raise ValueError(f"corrupt {channel} shard {name}")
            input_meta[channel]["shards"][name] = shard["sha256"]
            for r in pq.read_table(path).to_pylist():
                key = (r["source"], r["source_record_id"], r["s1_id"])
                if key[:2] not in sources:
                    raise ValueError("candidate outside source sample")
                item = pairs.setdefault(key, {"source": key[0], "source_record_id": key[1], "s1_id": key[2],
                                              "dense_score": None, "dense_rank": None,
                                              "name_score": None, "name_rank": None,
                                              "address_score": None, "address_rank": None})
                for field in ("dense_score", "dense_rank") if channel == "dense" else ("name_score", "name_rank", "address_score", "address_rank"):
                    item[field] = r[field]
    grouped = defaultdict(int)
    for source, cid, _ in pairs:
        grouped[source, cid] += 1
    if set(grouped) != sources:
        raise ValueError("candidate output did not close every source group")
    ordered_pairs = [pairs[k] for k in sorted(pairs)]
    target = ROOT / "subset_union.parquet"
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(ordered_pairs), tmp, compression="zstd")
    os.replace(tmp, target)
    nested = {}
    for n in (5, 10, 20, 50):
        selected = ids[:n]
        candidate_sets = {q: set() for q in selected}
        for _, cid, q in pairs:
            if q in candidate_sets:
                candidate_sets[q].add(cid)
        result = score({"query_ids": selected}, candidate_sets, truth)
        result.pop("candidate_count")  # Only gold-endpoint source records were sampled.
        nested[str(n)] = {"queries": n, "gold_positive_edges": sum(len(truth[q]) for q in selected),
                          "oracle_macro_f05": result["oracle_macro_f05"],
                          "positive_pair_recall": result["positive_pair_recall"],
                          "zero_positive_queries": result["zero_positive_queries"]}
    replay = json.loads((ROOT / "threshold_subset_report.json").read_text())
    for n, result in nested.items():
        expected = replay["nested_subsets"][n]["reverse_production"]
        for field, original in (("oracle_macro_f05", expected["oracle_macro_f05"]),
                                ("positive_pair_recall", expected["positive_pair_recall"])):
            if abs(result[field] - original) > 1e-12:
                raise ValueError(f"fresh subset differs from cached full-background ranks: {n}/{field}")
    counts = sorted(grouped.values())
    report = {"complete": True, "positive_only_diagnostic": True,
              "production_policy": "reverse lexical 50 + exact dense 20",
              "source_groups": len(sources), "unique_pairs": len(pairs),
              "candidate_count_on_selected_positive_endpoints": {
                  "mean": statistics.mean(counts), "p95": counts[int(.95 * (len(counts)-1))],
                  "p99": counts[int(.99 * (len(counts)-1))], "max": max(counts)},
              "nested_subsets": nested, "union_sha256": sha(target), "input_manifests": input_meta}
    write_json(ROOT / "fresh_subset_report.json", report)
    print(json.dumps({"source_groups": len(sources), "unique_pairs": len(pairs), "nested_subsets": nested}, indent=2))


if __name__ == "__main__":
    main()
