"""Full-threshold, full-background S1-forward FTS pilot with reverse union."""
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

from src.forward_fts import ForwardIndex
from src.retrieval_experiments import gt, load_pairs, score, selected_text, write_json


def main():
    started = time.time()
    manifest, baseline, _, _ = load_pairs("threshold", verify=False)
    truth = gt(manifest["query_ids"])
    qtext = selected_text(set(manifest["query_ids"]))
    lexical = {(r["s1_id"], r["source_record_id"]): r["rank"] for r in pq.read_table(
        "artifacts/reverse-v1/pilot_v5_all/positive_ranks.parquet").to_pylist()}
    dense = {(r["s1_id"], r["source_record_id"]): r["rank"] for r in pq.read_table(
        "artifacts/dense-v1/pilot/positive_ranks.parquet").to_pylist()}
    index = ForwardIndex(Path("artifacts/forward-fts-v2"), "train")
    hits = {}
    rates = {}
    for pos, q in enumerate(manifest["query_ids"], 1):
        name, address, country = qtext[q]
        hits[q] = index.query(name, address, country, 50)
        if pos in (100, 500, 2500):
            rates[str(pos)] = {"queries_per_second": pos / (time.time() - started), "elapsed": time.time() - started}
    index.close()
    positive_ranks = []
    for q in manifest["query_ids"]:
        by_id = {r["source_record_id"]: r for r in hits[q]}
        for cid in sorted(truth[q]):
            row = by_id.get(cid, {})
            positive_ranks.append({"s1_id": q, "source_record_id": cid,
                                   "name_rank": row.get("name_rank"),
                                   "address_rank": row.get("address_rank")})
    pq.write_table(pa.Table.from_pylist(positive_ranks),
                   "artifacts/forward-fts-v2/threshold_positive_ranks.parquet", compression="zstd")
    output = {"positive_only_diagnostic": False, "full_s1_background": True,
              "queries": len(manifest["query_ids"]), "seconds_total": time.time() - started,
              "rates_including_setup": rates, "policies": {}}
    for k in (5, 10, 20, 50):
        forward = {q: {r["source_record_id"] for r in hits[q] if
                       (r["name_rank"] is not None and r["name_rank"] <= k) or
                       (r["address_rank"] is not None and r["address_rank"] <= k)}
                   for q in manifest["query_ids"]}
        reverse = {q: {cid for cid in truth[q] if
                       (lexical.get((q, cid)) is not None and lexical[q, cid] <= 50) or
                       (dense.get((q, cid)) is not None and dense[q, cid] <= 20)}
                   for q in manifest["query_ids"]}
        deployable = {q: forward[q] | reverse[q] for q in manifest["query_ids"]}
        historical = {q: deployable[q] | baseline[q] for q in manifest["query_ids"]}
        output["policies"][str(k)] = {
            "forward_only": score(manifest, forward, truth),
            "forward_plus_reverse": score(manifest, deployable, truth),
            "historical_forward_plus_new_forward_plus_reverse_diagnostic": score(manifest, historical, truth),
            "forward_pair_count": sum(map(len, forward.values())),
        }
    write_json("artifacts/forward-fts-v2/threshold_pilot.json", output)
    print(json.dumps({k: {"deployable_oracle": v["forward_plus_reverse"]["oracle_macro_f05"],
                          "deployable_recall": v["forward_plus_reverse"]["positive_pair_recall"],
                          "forward_mean": v["forward_only"]["candidate_count"]["mean"]}
                      for k, v in output["policies"].items()}, indent=2))


if __name__ == "__main__":
    main()
