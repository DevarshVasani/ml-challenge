"""Threshold-only gate/oracle audit for Member D.

This module deliberately scopes evaluation truth through the supplied query
manifest.  That prevents a threshold-only gate from being accidentally scored
against the frozen final partition.

It also answers the neural-budget question: how many additional labelled gold
edges would be lost if low-tree-score neural rows were hard rejected.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .member_d_contracts import (
    FORBIDDEN_PRODUCTION_COLUMNS,
    PAIR_KEY,
    TREE_SCORE,
    derive_candidate_source,
    validate_pair_keys,
)
from .neural_contracts import sha256_file

ROUTE_ALIASES = {
    "neural": "neural_fusion",
    "neural_fusion": "neural_fusion",
    "tree": "tree_only",
    "tree_only": "tree_only",
    "discard": "discard",
}


def _read(path: str | Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p, columns=columns)
    frame = pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)
    return frame if columns is None else frame[columns]


def _query_ids(path: str | Path, *, refuse_final: bool = True) -> tuple[list[str], dict[str, str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    id_col = "source1_entity_id" if "source1_entity_id" in frame.columns else "entity_id"
    ids = frame[id_col].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Query manifest contains duplicate S1 IDs")
    splits = {}
    if "split" in frame.columns:
        splits = dict(zip(ids, frame["split"].astype(str)))
        if refuse_final and any(v.lower() == "final" for v in splits.values()):
            raise ValueError(
                "Refusing to tune/audit cutoffs on a final query manifest. "
                "Use the frozen threshold manifest."
            )
    return ids, splits


def _truth(path: str | Path, query_ids: Iterable[str]) -> dict[str, list[str]]:
    wanted = set(map(str, query_ids))
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Truth table missing columns: {sorted(required - set(frame.columns))}")
    raw = {}
    for row in frame.itertuples(index=False):
        sid = str(row.source1_entity_id)
        if sid in wanted:
            raw[sid] = [
                value.strip()
                for value in str(row.matched_entity_ids or "").split(",")
                if value.strip()
            ]
    return {sid: raw.get(sid, []) for sid in wanted}


def _normalize_route(value: object) -> str:
    route = str(value).strip().lower()
    if route not in ROUTE_ALIASES:
        raise ValueError(f"Unsupported gate route: {value!r}")
    return ROUTE_ALIASES[route]


def _gold_metrics(
    truth: dict[str, list[str]],
    surviving_keys: set[tuple[str, str, str]],
) -> dict[str, Any]:
    total_gold = 0
    survived_gold = 0
    recalls: list[float] = []
    full = partial = zero = 0
    source_total = {"S2": 0, "S3": 0}
    source_survived = {"S2": 0, "S3": 0}

    for sid, gold in truth.items():
        if not gold:
            continue
        got = 0
        for cid in gold:
            source = derive_candidate_source(cid)
            total_gold += 1
            source_total[source] += 1
            if (source, cid, sid) in surviving_keys:
                got += 1
                survived_gold += 1
                source_survived[source] += 1

        recall = got / len(gold)
        recalls.append(recall)
        if got == len(gold):
            full += 1
        elif got == 0:
            zero += 1
        else:
            partial += 1

    by_source = {}
    for source in ("S2", "S3"):
        denom = source_total[source]
        by_source[source] = {
            "gold_edge_count": denom,
            "survived_gold_edge_count": source_survived[source],
            "pair_recall": source_survived[source] / denom if denom else "N/A",
        }

    return {
        "gold_bearing_s1_count": len(recalls),
        "gold_edge_count": total_gold,
        "survived_gold_edge_count": survived_gold,
        "missing_gold_edge_count": total_gold - survived_gold,
        "micro_oracle_recall": survived_gold / total_gold if total_gold else "N/A",
        "macro_oracle_recall": sum(recalls) / len(recalls) if recalls else "N/A",
        "full_recall_s1": full,
        "partial_miss_s1": partial,
        "zero_recall_s1": zero,
        "by_source": by_source,
    }


def audit_gate(
    gate_path: str | Path,
    query_manifest_path: str | Path,
    truth_path: str | Path,
    *,
    cutoffs: Iterable[float] = (0.0001, 0.001, 0.01, 0.05),
    pairs_per_second: float | None = None,
    production_top3_pairs: int | None = None,
    refuse_final: bool = True,
) -> dict[str, Any]:
    gate = _read(gate_path)
    validate_pair_keys(gate)

    forbidden = sorted(set(gate.columns) & FORBIDDEN_PRODUCTION_COLUMNS)
    if forbidden:
        raise ValueError(
            "Gate contains forbidden/truth-derived production columns: "
            f"{forbidden}. Use the leakage-safe gate artifact."
        )
    if TREE_SCORE not in gate.columns:
        raise ValueError(f"Gate is missing {TREE_SCORE}")
    if "route" not in gate.columns and "decision_route" not in gate.columns:
        raise ValueError("Gate requires route or decision_route")

    query_ids, splits = _query_ids(query_manifest_path, refuse_final=refuse_final)
    query_set = set(query_ids)
    unknown_s1 = sorted(set(gate["source1_entity_id"].astype(str)) - query_set)
    if unknown_s1:
        raise ValueError(
            "Gate contains S1 IDs outside the query manifest: "
            f"{unknown_s1[:5]}"
        )

    truth = _truth(truth_path, query_ids)
    route_col = "decision_route" if "decision_route" in gate.columns else "route"
    routes = gate[route_col].map(_normalize_route)
    tree_score = pd.to_numeric(gate[TREE_SCORE], errors="raise")
    if tree_score.isna().any() or ((tree_score < 0) | (tree_score > 1)).any():
        raise ValueError("tree_score must be finite and in [0,1]")

    key_tuples = list(
        map(
            tuple,
            gate[PAIR_KEY].astype(str).itertuples(index=False, name=None),
        )
    )
    all_keys = set(key_tuples)
    active_mask = routes != "discard"
    active_keys = {
        key for key, active in zip(key_tuples, active_mask.tolist()) if active
    }

    raw = _gold_metrics(truth, all_keys)
    post_gate = _gold_metrics(truth, active_keys)
    base_retrieved = post_gate["survived_gold_edge_count"]

    cutoff_rows = []
    neural_mask = routes == "neural_fusion"
    neural_total = int(neural_mask.sum())

    for cutoff in map(float, cutoffs):
        keep_neural = neural_mask & (tree_score >= cutoff)
        survive = (~(routes == "discard")) & (~neural_mask | keep_neural)
        surviving = {
            key for key, keep in zip(key_tuples, survive.tolist()) if keep
        }
        metrics = _gold_metrics(truth, surviving)
        neural_kept = int(keep_neural.sum())
        retained_fraction = neural_kept / neural_total if neural_total else 0.0
        projected_pairs = (
            int(round(production_top3_pairs * retained_fraction))
            if production_top3_pairs is not None
            else None
        )
        projected_hours = (
            projected_pairs / float(pairs_per_second) / 3600.0
            if projected_pairs is not None and pairs_per_second
            else None
        )
        cutoff_rows.append(
            {
                "tree_reject_below": cutoff,
                "neural_rows_kept": neural_kept,
                "neural_rows_total": neural_total,
                "neural_retained_fraction": retained_fraction,
                "additional_gold_edges_lost_vs_gate": (
                    base_retrieved - metrics["survived_gold_edge_count"]
                ),
                "survived_gold_edge_count": metrics["survived_gold_edge_count"],
                "micro_oracle_recall": metrics["micro_oracle_recall"],
                "macro_oracle_recall": metrics["macro_oracle_recall"],
                "projected_production_neural_pairs": projected_pairs,
                "projected_hours": projected_hours,
            }
        )

    split_values = sorted(set(splits.values())) if splits else []
    return {
        "artifact": {
            "gate_path": str(gate_path),
            "gate_sha256": sha256_file(gate_path),
            "query_manifest_path": str(query_manifest_path),
            "query_manifest_sha256": sha256_file(query_manifest_path),
            "truth_path": str(truth_path),
            "truth_sha256": sha256_file(truth_path),
        },
        "scope": {
            "query_count": len(query_ids),
            "split_values": split_values,
            "singleton_query_count": sum(not truth[sid] for sid in query_ids),
            "gold_bearing_query_count": sum(bool(truth[sid]) for sid in query_ids),
        },
        "gate": {
            "row_count": len(gate),
            "route_counts": {
                str(k): int(v)
                for k, v in routes.value_counts(dropna=False).to_dict().items()
            },
            "raw_candidate_oracle": raw,
            "post_gate_oracle": post_gate,
            "additional_gate_gold_loss": (
                raw["survived_gold_edge_count"]
                - post_gate["survived_gold_edge_count"]
            ),
        },
        "cutoff_audit": cutoff_rows,
        "projection": {
            "pairs_per_second": pairs_per_second,
            "production_top3_pairs": production_top3_pairs,
            "note": (
                "Projection only. Retained fractions are measured on the supplied "
                "threshold gate and may not transfer to source-complete production."
            ),
        },
        "selection_eligible": False,
        "selection_ineligible_reason": (
            "Gate/cutoff audit is diagnostic until C supplies leakage-safe provenance "
            "and source-complete competition features."
        ),
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "gate_audit.json"
    cutoff_path = out / "gate_cutoffs.tsv"
    summary_path = out / "gate_summary.tsv"

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(report["cutoff_audit"]).to_csv(
        cutoff_path, sep="\t", index=False
    )

    raw = report["gate"]["raw_candidate_oracle"]
    post = report["gate"]["post_gate_oracle"]
    pd.DataFrame(
        [
            {
                "query_count": report["scope"]["query_count"],
                "gate_rows": report["gate"]["row_count"],
                "raw_gold_recall": raw["micro_oracle_recall"],
                "post_gate_gold_recall": post["micro_oracle_recall"],
                "post_gate_macro_recall": post["macro_oracle_recall"],
                "raw_missing_gold_edges": raw["missing_gold_edge_count"],
                "gate_added_missing_gold_edges": report["gate"][
                    "additional_gate_gold_loss"
                ],
            }
        ]
    ).to_csv(summary_path, sep="\t", index=False)

    return {
        "json": str(json_path),
        "cutoffs_tsv": str(cutoff_path),
        "summary_tsv": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cutoffs",
        default="0.0001,0.001,0.01,0.05",
        help="comma-separated tree_score hard-reject cutoffs",
    )
    parser.add_argument("--pairs-per-second", type=float, default=None)
    parser.add_argument("--production-top3-pairs", type=int, default=None)
    parser.add_argument(
        "--allow-final",
        action="store_true",
        help="Only for structural diagnostics. Never use this for tuning.",
    )
    args = parser.parse_args()

    report = audit_gate(
        args.gate,
        args.query_manifest,
        args.truth,
        cutoffs=[float(x) for x in args.cutoffs.split(",") if x.strip()],
        pairs_per_second=args.pairs_per_second,
        production_top3_pairs=args.production_top3_pairs,
        refuse_final=not args.allow_final,
    )
    paths = write_report(report, args.output_dir)
    print(json.dumps({"report": report, "outputs": paths}, indent=2, default=str))


if __name__ == "__main__":
    main()
