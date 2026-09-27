"""Replay the frozen production rank policy on nested threshold query subsets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.retrieval_experiments import gt, load_pairs, score, sha, write_json


ROOT = Path("artifacts/handoff-retrieval-v1")
LEX = Path("artifacts/reverse-v1/pilot_v5_all/positive_ranks.parquet")
DENSE = Path("artifacts/dense-v1/pilot/positive_ranks.parquet")
POLICY = Path("artifacts/reverse-v1/frozen_policy.json")


def ranks(path: Path) -> dict[tuple[str, str], int | None]:
    rows = pq.read_table(path, columns=["s1_id", "source_record_id", "rank"]).to_pylist()
    result = {(r["s1_id"], r["source_record_id"]): r["rank"] for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate positive rank pair in {path}")
    return result


def metric(ids, truth, baseline, lexical, dense, lex_k, dense_k, include_forward=False):
    pairs = {q: set(baseline[q]) if include_forward else set() for q in ids}
    for q in ids:
        for cid in truth[q]:
            lr, dr = lexical.get((q, cid)), dense.get((q, cid))
            if (lr is not None and lr <= lex_k) or (dr is not None and dr <= dense_k):
                pairs[q].add(cid)
    # score() consumes only query_ids and the original frozen query metadata.
    out = score({"query_ids": ids}, pairs, truth)
    out.pop("candidate_count")  # Positive-only rank replay has no arbitrary-record volume.
    return out


def main():
    m, baseline, _, _ = load_pairs("threshold", verify=False)
    truth = gt(m["query_ids"])
    lexical, dense = ranks(LEX), ranks(DENSE)
    expected = {(q, cid) for q in m["query_ids"] for cid in truth[q]}
    if set(lexical) != expected or set(dense) != expected:
        raise ValueError("positive-rank files do not cover every frozen threshold edge")
    ordered = sorted(m["query_ids"], key=lambda q: hashlib.sha256(("handoff-retrieval-v1:" + q).encode()).hexdigest())
    selected = ordered[:50]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "threshold_query_ids_50.txt").write_text("\n".join(selected) + "\n")
    policy = json.loads(POLICY.read_text())
    if policy["lexical"]["top_k"] != 50 or policy["dense"]["top_k"] != 20:
        raise ValueError("frozen production policy differs from rank replay")
    output = {"selection": "first 50 SHA-256 ordered frozen threshold query IDs, seed handoff-retrieval-v1",
              "positive_only_diagnostic": True,
              "full_s1_background": True,
              "production_policy": "reverse lexical 50 + exact dense 20",
              "inputs_sha256": {"lexical_ranks": sha(LEX), "dense_ranks": sha(DENSE), "policy": sha(POLICY)},
              "nested_subsets": {}, "candidate_budget_sensitivity": {}}
    for n in (5, 10, 20, 50):
        ids = selected[:n]
        reverse = metric(ids, truth, baseline, lexical, dense, 50, 20)
        forward_union = metric(ids, truth, baseline, lexical, dense, 50, 20, include_forward=True)
        total = sum(len(truth[q]) for q in ids)
        output["nested_subsets"][str(n)] = {"queries": n, "gold_positive_edges": total,
                                             "reverse_production": reverse,
                                             "historical_forward_plus_reverse_diagnostic": forward_union}
    for k in (5, 10, 20, 50):
        output["candidate_budget_sensitivity"][str(k)] = {
            "threshold_2500_reverse": metric(m["query_ids"], truth, baseline, lexical, dense, k, min(k, 20)),
            "threshold_2500_historical_forward_union_diagnostic": metric(m["query_ids"], truth, baseline, lexical, dense, k, min(k, 20), include_forward=True),
        }
    write_json(ROOT / "threshold_subset_report.json", output)
    print(json.dumps({n: {"oracle": v["reverse_production"]["oracle_macro_f05"],
                          "recall": v["reverse_production"]["positive_pair_recall"],
                          "gold": v["gold_positive_edges"]}
                      for n, v in output["nested_subsets"].items()}, indent=2))


if __name__ == "__main__":
    main()
