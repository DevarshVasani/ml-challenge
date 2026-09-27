"""Unified evaluation pipeline for Member D.

Evaluates any combination of raw candidates, post-gate candidates,
scored pairs, and final predictions against ground truth.

Always evaluates over the FULL query manifest (including zero-candidate
queries).  Every S1 query is in the denominator.

Usage
-----
    python -m src.evaluate_pipeline --config configs/member_d/evaluate_threshold.json

Config JSON keys (all optional except query_manifest and ground_truth_tsv)
--------------------------------------------------------------------------
    query_manifest          str   path to threshold_queries.tsv (or similar)
    ground_truth_tsv        str   path to evaluation_truth.tsv
    s1_metadata_tsv         str   path to S1 metadata with country column
    raw_candidate_shards    str   path/dir to raw candidate parquet/TSV shards
    post_gate_candidate_shards str  (optional)
    scored_pair_shards      str   (optional) joined-score shards with decision_route
    final_prediction_tsv    str   (optional) TSV with source1_entity_id, matched_entity_ids
    source_registry_paths   dict  {"s2": "...", "s3": "..."}  (optional)
    output_json             str   where to write JSON result
    experiments_csv_row     bool  if true, append one row to reports/experiments.csv
    run_label               str   label for experiments.csv row
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .evaluate import evaluate_predictions
from .evaluation_slices import evaluate_fake_pattern_slices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_query_manifest(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for col in ("entity_id", "source1_entity_id"):
            if col in (reader.fieldnames or []):
                return [row[col].strip() for row in reader if row[col].strip()]
    raise ValueError(f"Query manifest {path} has no entity_id or source1_entity_id column")


def _load_ground_truth(path: Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1 = row.get("source1_entity_id", "").strip()
            raw = row.get("matched_entity_ids", "").strip()
            ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
            if s1:
                truth[s1] = ids
    return truth


def _load_s1_metadata(path: Path | None) -> dict[str, str]:
    """Returns {s1_id: country}."""
    if path is None or not path.exists():
        return {}
    meta: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                meta[eid] = row.get("country", "UNKNOWN").strip()
    return meta


def _load_prediction_tsv(path: Path) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1 = row.get("source1_entity_id", "").strip()
            raw = row.get("matched_entity_ids", "").strip()
            ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
            if s1:
                preds[s1] = ids
    return preds


def _load_candidate_shard_dir(path: Path) -> pd.DataFrame | None:
    """Load all parquet/TSV shards under a directory into one frame."""
    if path is None or not path.exists():
        return None
    frames: list[pd.DataFrame] = []
    for f in sorted(path.rglob("*.parquet")):
        frames.append(pd.read_parquet(f))
    for f in sorted(path.rglob("*.tsv")):
        frames.append(pd.read_csv(f, sep="\t", dtype=object, keep_default_na=False))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _p95(values: Iterable[float]) -> float:
    lst = sorted(values)
    if not lst:
        return 0.0
    idx = int(math.ceil(0.95 * len(lst))) - 1
    return lst[max(0, idx)]


# ---------------------------------------------------------------------------
# Candidate-level metrics
# ---------------------------------------------------------------------------


def _candidate_metrics(
    cand_df: pd.DataFrame,
    s1_ids: list[str],
    truth: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute candidate-level retrieval metrics."""
    if cand_df is None or cand_df.empty:
        return {}

    # Build mapping s1 -> set of candidate ids
    cand_map: dict[str, set[str]] = defaultdict(set)
    if "source1_entity_id" in cand_df.columns and "candidate_entity_id" in cand_df.columns:
        for _, row in cand_df.iterrows():
            cand_map[str(row["source1_entity_id"])].add(str(row["candidate_entity_id"]))

    counts_per_s1 = [len(cand_map.get(s1, set())) for s1 in s1_ids]
    zero_cand = sum(1 for c in counts_per_s1 if c == 0)

    # Gold edge retrieval
    total_gold = sum(len(truth.get(s1, [])) for s1 in s1_ids)
    retrieved_gold = sum(
        len(set(truth.get(s1, [])) & cand_map.get(s1, set()))
        for s1 in s1_ids
    )
    missed_gold = total_gold - retrieved_gold
    pair_recall = retrieved_gold / total_gold if total_gold > 0 else float("nan")

    # Oracle F0.5: assume we predict exactly the true positives in candidates
    oracle_preds = {s1: list(set(truth.get(s1, [])) & cand_map.get(s1, set())) for s1 in s1_ids}
    oracle_truth = {s1: truth.get(s1, []) for s1 in s1_ids}
    oracle_result = evaluate_predictions(oracle_truth, oracle_preds)

    return {
        "total_pairs": int(len(cand_df)),
        "mean_candidates_per_s1": float(sum(counts_per_s1) / len(s1_ids)) if s1_ids else 0.0,
        "p95_candidates_per_s1": float(_p95(counts_per_s1)),
        "max_candidates_per_s1": int(max(counts_per_s1, default=0)),
        "zero_candidate_s1_count": zero_cand,
        "total_gold_edges": total_gold,
        "retrieved_gold_edges": retrieved_gold,
        "missed_gold_edges": missed_gold,
        "pair_recall": pair_recall,
        "oracle_macro_f05": oracle_result["macro_f05"],
    }


# ---------------------------------------------------------------------------
# Route breakdown
# ---------------------------------------------------------------------------


def _route_breakdown(
    scored_df: pd.DataFrame,
    s1_ids: list[str],
    truth: dict[str, list[str]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Breakdown by decision_route if the column exists."""
    if scored_df is None or "decision_route" not in scored_df.columns:
        return {}
    result: dict[str, Any] = {}
    for route in scored_df["decision_route"].unique():
        subset = scored_df[scored_df["decision_route"] == route]
        pair_count = len(subset)
        s1_count = subset["source1_entity_id"].nunique() if "source1_entity_id" in subset else 0
        result[str(route)] = {"pair_count": pair_count, "s1_count": s1_count}
    return result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Run full evaluation and return results dict."""
    t0 = time.time()

    query_path = Path(config["query_manifest"])
    truth_path = Path(config["ground_truth_tsv"])
    s1_meta_path = Path(config["s1_metadata_tsv"]) if config.get("s1_metadata_tsv") else None
    output_json = Path(config["output_json"]) if config.get("output_json") else None
    experiments_csv = config.get("experiments_csv_row", False)
    run_label = config.get("run_label", "unlabeled")

    s1_ids = _load_query_manifest(query_path)
    truth = _load_ground_truth(truth_path)
    # Ensure every query S1 is in truth (singletons have empty list)
    full_truth = {s1: truth.get(s1, []) for s1 in s1_ids}
    country_map = _load_s1_metadata(s1_meta_path)

    results: dict[str, Any] = {
        "run_label": run_label,
        "s1_count": len(s1_ids),
        "config": config,
    }

    # ---- raw candidates ----
    raw_path = config.get("raw_candidate_shards")
    if raw_path:
        raw_df = _load_candidate_shard_dir(Path(raw_path))
        results["raw_candidates"] = _candidate_metrics(raw_df, s1_ids, full_truth)

    # ---- post-gate candidates ----
    gate_path = config.get("post_gate_candidate_shards")
    if gate_path:
        gate_df = _load_candidate_shard_dir(Path(gate_path))
        results["post_gate_candidates"] = _candidate_metrics(gate_df, s1_ids, full_truth)

    # ---- scored pairs ----
    scored_path = config.get("scored_pair_shards")
    scored_df: pd.DataFrame | None = None
    if scored_path:
        scored_df = _load_candidate_shard_dir(Path(scored_path))
        results["route_breakdown"] = _route_breakdown(scored_df, s1_ids, full_truth)
        
        # New fake-pattern evaluation by partition
        split_path = Path("configs/member_d/split.tsv")
        if not split_path.exists():
            split_path = Path("configs/member_d/selection_split.tsv")
        
        if split_path.exists() and "final_prediction_tsv" in config and config["final_prediction_tsv"]:
            try:
                split_df = pd.read_csv(split_path, sep="	", dtype=str, keep_default_na=False)
                if "member_d_partition" in split_df.columns:
                    s1_to_partition = dict(zip(split_df["source1_entity_id"], split_df["member_d_partition"]))
                else:
                    s1_to_partition = {}
            except Exception:
                s1_to_partition = {}
                
            # Build evaluation frame
            pred_path = Path(config["final_prediction_tsv"])
            if pred_path.exists():
                preds_dict = _load_prediction_tsv(pred_path)
                
                # Add label and predicted_match to scored_df
                def _get_label(row):
                    s1 = str(row.get("source1_entity_id", ""))
                    c = str(row.get("candidate_entity_id", ""))
                    return 1 if c in full_truth.get(s1, []) else 0
                    
                def _get_pred(row):
                    s1 = str(row.get("source1_entity_id", ""))
                    c = str(row.get("candidate_entity_id", ""))
                    return 1 if c in preds_dict.get(s1, []) else 0
                    
                eval_df = scored_df.copy()
                eval_df["label"] = eval_df.apply(_get_label, axis=1)
                eval_df["predicted_match"] = eval_df.apply(_get_pred, axis=1)
                
                # Add country to eval_df for route breakdown
                if country_map:
                    eval_df["country"] = eval_df["source1_entity_id"].apply(lambda s1: country_map.get(str(s1), "UNKNOWN"))
                
                fake_results = {}
                partitions = ["fusion_fit", "calibration", "selection", "UNKNOWN"]
                
                for part in partitions:
                    part_s1s = {s1 for s1, p in s1_to_partition.items() if p == part}
                    if not part_s1s and part != "UNKNOWN":
                        continue
                    part_mask = eval_df["source1_entity_id"].astype(str).isin(part_s1s)
                    part_df = eval_df[part_mask]
                    if part_df.empty:
                        continue
                        
                    part_metrics = evaluate_fake_pattern_slices(
                        part_df,
                        label_col="label",
                        prediction_col="predicted_match",
                        score_col="match_probability" if "match_probability" in part_df.columns else None
                    )
                    
                    # Route breakdown
                    if "decision_route" in part_df.columns:
                        for route in part_df["decision_route"].unique():
                            route_df = part_df[part_df["decision_route"] == route]
                            part_metrics[f"route_{route}"] = evaluate_fake_pattern_slices(
                                route_df,
                                label_col="label",
                                prediction_col="predicted_match",
                                score_col="match_probability" if "match_probability" in route_df.columns else None
                            )
                            
                    # Source breakdown
                    if "candidate_source" in part_df.columns:
                        for src in part_df["candidate_source"].unique():
                            src_df = part_df[part_df["candidate_source"] == src]
                            part_metrics[f"source_{src}"] = evaluate_fake_pattern_slices(
                                src_df,
                                label_col="label",
                                prediction_col="predicted_match",
                                score_col="match_probability" if "match_probability" in src_df.columns else None
                            )
                            
                    # Country breakdown
                    if "country" in part_df.columns:
                        for ctry in part_df["country"].unique():
                            ctry_df = part_df[part_df["country"] == ctry]
                            part_metrics[f"country_{ctry}"] = evaluate_fake_pattern_slices(
                                ctry_df,
                                label_col="label",
                                prediction_col="predicted_match",
                                score_col="match_probability" if "match_probability" in ctry_df.columns else None
                            )
                            
                    fake_results[part] = part_metrics
                    
                if fake_results:
                    results["fake_patterns"] = fake_results

    # ---- final predictions ----
    pred_path = config.get("final_prediction_tsv")
    if pred_path:
        preds = _load_prediction_tsv(Path(pred_path))
        # Ensure every S1 present (singletons default to empty)
        full_preds = {s1: preds.get(s1, []) for s1 in s1_ids}
        eval_result = evaluate_predictions(full_truth, full_preds)
        results["predictions"] = {
            "macro_f05": eval_result["macro_f05"],
            "singleton_accuracy": eval_result.get("singleton_accuracy", float("nan")),
            "non_singleton_macro_f05": eval_result.get("non_singleton_macro_f05", float("nan")),
            "num_entities": eval_result["num_entities"],
            "num_singletons": eval_result.get("num_singletons", 0),
        }

        # Country breakdown
        if country_map:
            by_country: dict[str, dict[str, Any]] = defaultdict(lambda: {"truth": {}, "preds": {}})
            country_truth: dict[str, dict[str, list[str]]] = defaultdict(dict)
            country_preds: dict[str, dict[str, list[str]]] = defaultdict(dict)
            for s1 in s1_ids:
                c = country_map.get(s1, "UNKNOWN")
                country_truth[c][s1] = full_truth[s1]
                country_preds[c][s1] = full_preds[s1]
            country_results: dict[str, Any] = {}
            for c, ct in country_truth.items():
                cr = evaluate_predictions(ct, country_preds.get(c, {}))
                country_results[c] = {
                    "macro_f05": cr["macro_f05"],
                    "num_entities": cr["num_entities"],
                }
            results["by_country"] = country_results

        # Source breakdown (S2/S3 edges)
        if scored_df is not None and "candidate_source" in scored_df.columns:
            by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for _, row in scored_df.iterrows():
                src = str(row.get("candidate_source", ""))
                if src:
                    by_source[src]["pair_count"] += 1
            results["by_source"] = {k: dict(v) for k, v in by_source.items()}

    results["elapsed_s"] = round(time.time() - t0, 2)

    # ---- write JSON ----
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"[evaluate_pipeline] Written: {output_json}")

    # ---- append to experiments.csv ----
    if experiments_csv:
        exp_path = Path("reports/experiments.csv")
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        pred_r = results.get("predictions", {})
        raw_r = results.get("raw_candidates", {})
        row = {
            "run_label": run_label,
            "macro_f05": pred_r.get("macro_f05", "N/A"),
            "singleton_accuracy": pred_r.get("singleton_accuracy", "N/A"),
            "non_singleton_macro_f05": pred_r.get("non_singleton_macro_f05", "N/A"),
            "oracle_macro_f05": raw_r.get("oracle_macro_f05", "N/A"),
            "pair_recall": raw_r.get("pair_recall", "N/A"),
            "s1_count": results["s1_count"],
            "elapsed_s": results["elapsed_s"],
            "output_json": str(output_json) if output_json else "N/A",
        }
        write_header = not exp_path.exists()
        with exp_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ---- stdout summary ----
    print(f"\n{'='*60}")
    print(f"[evaluate_pipeline] {run_label}  |  S1 count: {results['s1_count']}")
    if "predictions" in results:
        p = results["predictions"]
        print(f"  macro F0.5          : {p.get('macro_f05', 'N/A'):.4f}")
        print(f"  singleton accuracy  : {p.get('singleton_accuracy', 'N/A'):.4f}")
        print(f"  non-singleton F0.5  : {p.get('non_singleton_macro_f05', 'N/A'):.4f}")
    if "raw_candidates" in results:
        r = results["raw_candidates"]
        print(f"  oracle F0.5         : {r.get('oracle_macro_f05', 'N/A'):.4f}")
        print(f"  pair recall         : {r.get('pair_recall', 'N/A'):.4f}")
        print(f"  zero-cand queries   : {r.get('zero_candidate_s1_count', 'N/A')}")
    print(f"  elapsed             : {results['elapsed_s']}s")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Member D evaluation pipeline.")
    parser.add_argument("--config", required=True, help="Path to evaluate JSON config")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
