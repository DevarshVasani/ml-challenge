"""Canonical Member D evaluation command for raw/post-gate/final stages."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import compute_confusion_components, evaluate_predictions
from .evaluation_slices import evaluate_fake_pattern_slices
from .member_d_contracts import DECISION_ROUTE, MATCH_PROBABILITY, derive_candidate_source


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        files = sorted(list(p.rglob("*.parquet")) + list(p.rglob("*.tsv")))
        return pd.concat([_read_table(x) for x in files], ignore_index=True) if files else pd.DataFrame()
    if p.suffix.lower() in {".parquet", ".pq"}: return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _query_manifest(path: str | Path) -> tuple[list[str], dict[str, str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    id_col = "source1_entity_id" if "source1_entity_id" in df.columns else "entity_id"
    ids = df[id_col].astype(str).tolist()
    if len(ids) != len(set(ids)): raise ValueError("Duplicate S1 IDs in query manifest")
    countries = dict(zip(ids, df["country"].astype(str))) if "country" in df.columns else {}
    return ids, countries


def _truth(path: str | Path, query_ids: list[str]) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    raw = {str(r.source1_entity_id): [x.strip() for x in str(r.matched_entity_ids).split(",") if x.strip()] for r in df.itertuples(index=False)}
    return {s: raw.get(s, []) for s in query_ids}


def _predictions(path: str | Path, query_ids: list[str]) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    raw = {str(r.source1_entity_id): [x.strip() for x in str(r.matched_entity_ids).split(",") if x.strip()] for r in df.itertuples(index=False)}
    unknown = set(raw) - set(query_ids)
    if unknown: raise ValueError(f"Predictions contain unknown S1 IDs: {sorted(unknown)[:5]}")
    return {s: raw.get(s, []) for s in query_ids}


def _candidate_metrics(frame: pd.DataFrame, query_ids: list[str], truth: dict[str, list[str]]) -> dict[str, Any]:
    cand_map: dict[str, set[str]] = defaultdict(set)
    if not frame.empty:
        for r in frame.itertuples(index=False): cand_map[str(r.source1_entity_id)].add(str(r.candidate_entity_id))
    counts = [len(cand_map[s]) for s in query_ids]
    retrieved = 0; total = 0; partial = 0; complete_miss = 0
    for s in query_ids:
        gold = set(truth[s]); got = gold & cand_map[s]; total += len(gold); retrieved += len(got)
        if gold and 0 < len(got) < len(gold): partial += 1
        if gold and len(got) == 0: complete_miss += 1
    oracle = {s: sorted(set(truth[s]) & cand_map[s]) for s in query_ids}
    vals = sorted(counts)
    p95 = vals[max(0, math.ceil(.95 * len(vals)) - 1)] if vals else 0
    return {
        "query_count": len(query_ids),
        "candidate_pair_count": int(len(frame)),
        "zero_candidate_query_count": int(sum(x == 0 for x in counts)),
        "gold_edge_count": int(total),
        "retrieved_gold_edge_count": int(retrieved),
        "missed_gold_edge_count": int(total - retrieved),
        "candidate_pair_recall": float(retrieved / total) if total else "N/A",
        "oracle_macro_f05": float(evaluate_predictions(truth, oracle)["macro_f05"]),
        "partial_miss_query_count": partial,
        "complete_miss_non_singleton_count": complete_miss,
        "mean_candidates_per_s1": float(sum(counts) / len(counts)) if counts else 0.0,
        "p95_candidates_per_s1": int(p95),
        "max_candidates_per_s1": int(max(counts, default=0)),
    }


def _edge_metrics(truth: dict[str, list[str]], preds: dict[str, list[str]]) -> dict[str, Any]:
    tp = fp = fn = 0
    for s, gold in truth.items():
        a, b, c = compute_confusion_components(gold, preds.get(s, [])); tp += a; fp += b; fn += c
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "pair_precision": float(tp / (tp + fp)) if tp + fp else "N/A",
        "pair_recall": float(tp / (tp + fn)) if tp + fn else "N/A",
        "predicted_edge_count": int(tp + fp),
        "predicted_nonempty_s1_count": int(sum(bool(preds.get(s, [])) for s in truth)),
    }


def _source_metrics(truth: dict[str, list[str]], preds: dict[str, list[str]], source: str) -> dict[str, Any]:
    prefix = f"{source}-"
    t = {s: [x for x in xs if x.startswith(prefix)] for s, xs in truth.items()}
    p = {s: [x for x in preds.get(s, []) if x.startswith(prefix)] for s in truth}
    return _edge_metrics(t, p)


def _route_metrics(scored: pd.DataFrame, truth: dict[str, list[str]], preds: dict[str, list[str]]) -> dict[str, Any]:
    if scored.empty or DECISION_ROUTE not in scored.columns: return {}
    out = {}
    total_gold = sum(len(v) for v in truth.values())
    for route, grp in scored.groupby(DECISION_ROUTE, sort=False):
        selected = []
        for r in grp.itertuples(index=False):
            if str(r.candidate_entity_id) in preds.get(str(r.source1_entity_id), []): selected.append((str(r.source1_entity_id), str(r.candidate_entity_id)))
        tp = sum(cid in truth[s] for s, cid in selected); fp = len(selected) - tp
        out[str(route)] = {
            "pair_count": int(len(grp)), "selected_edge_count": len(selected), "tp": int(tp), "fp": int(fp),
            "pair_precision": float(tp / len(selected)) if selected else "N/A",
            "pair_recall_against_all_gold": float(tp / total_gold) if total_gold else "N/A",
        }
    return out


def _fake_slices(scored: pd.DataFrame, truth: dict[str, list[str]], preds: dict[str, list[str]], split_path: str | None, countries: dict[str, str]) -> dict[str, Any]:
    needed = {"source1_entity_id", "candidate_entity_id"}
    if scored.empty or not needed.issubset(scored.columns): return {}
    frame = scored.copy()
    frame["label"] = [int(str(r.candidate_entity_id) in truth.get(str(r.source1_entity_id), [])) for r in frame.itertuples(index=False)]
    frame["predicted_match"] = [int(str(r.candidate_entity_id) in preds.get(str(r.source1_entity_id), [])) for r in frame.itertuples(index=False)]
    if countries: frame["country"] = frame.source1_entity_id.astype(str).map(countries).fillna("UNKNOWN")
    partition = {}
    if split_path and Path(split_path).exists():
        sdf = pd.read_csv(split_path, sep="\t", dtype=str, keep_default_na=False); partition = dict(zip(sdf.source1_entity_id, sdf.member_d_partition))
        frame["member_d_partition"] = frame.source1_entity_id.astype(str).map(partition).fillna("UNKNOWN")
    groups = {"all": frame}
    if partition:
        for p in ("fusion_fit", "calibration", "selection"): groups[p] = frame[frame.member_d_partition == p]
    result = {}
    for name, sub in groups.items():
        if sub.empty: continue
        base = evaluate_fake_pattern_slices(sub, score_col=MATCH_PROBABILITY if MATCH_PROBABILITY in sub.columns else None)
        if DECISION_ROUTE in sub.columns:
            base["by_route"] = {str(k): evaluate_fake_pattern_slices(g, score_col=MATCH_PROBABILITY if MATCH_PROBABILITY in g.columns else None) for k, g in sub.groupby(DECISION_ROUTE)}
        if "candidate_source" in sub.columns:
            base["by_source"] = {str(k): evaluate_fake_pattern_slices(g, score_col=MATCH_PROBABILITY if MATCH_PROBABILITY in g.columns else None) for k, g in sub.groupby("candidate_source")}
        if "country" in sub.columns:
            base["by_country"] = {str(k): evaluate_fake_pattern_slices(g, score_col=MATCH_PROBABILITY if MATCH_PROBABILITY in g.columns else None) for k, g in sub.groupby("country")}
        result[name] = base
    return result


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    query_ids, countries = _query_manifest(config["query_manifest"]); truth = _truth(config["ground_truth_tsv"], query_ids)
    result: dict[str, Any] = {"run_label": config.get("run_label", "member_d"), "query_count": len(query_ids)}
    if config.get("raw_candidate_shards"): result["raw_candidates"] = _candidate_metrics(_read_table(config["raw_candidate_shards"]), query_ids, truth)
    if config.get("post_gate_candidate_shards"): result["post_gate_candidates"] = _candidate_metrics(_read_table(config["post_gate_candidate_shards"]), query_ids, truth)
    scored = _read_table(config["scored_pair_shards"]) if config.get("scored_pair_shards") else pd.DataFrame()
    if config.get("final_prediction_tsv"):
        preds = _predictions(config["final_prediction_tsv"], query_ids); canonical = evaluate_predictions(truth, preds); edges = _edge_metrics(truth, preds)
        result["predictions"] = {k: v for k, v in canonical.items() if k != "per_entity_scores"} | edges
        result["by_source"] = {s: _source_metrics(truth, preds, s) for s in ("S2", "S3")}
        if countries:
            by_country = {}
            for country in sorted(set(countries.values())):
                ids = [s for s in query_ids if countries.get(s) == country]; t = {s: truth[s] for s in ids}; p = {s: preds[s] for s in ids}
                m = evaluate_predictions(t, p); by_country[country] = {k: v for k, v in m.items() if k != "per_entity_scores"} | _edge_metrics(t, p)
            result["by_country"] = by_country
        result["by_route"] = _route_metrics(scored, truth, preds)
        result["fake_patterns"] = _fake_slices(scored, truth, preds, config.get("member_d_split_tsv"), countries)
    out_json = Path(config.get("output_json", "reports/member_d/evaluation.json")); out_json.parent.mkdir(parents=True, exist_ok=True); out_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {
        "run_label": result["run_label"],
        "macro_f05": result.get("predictions", {}).get("macro_f05", "N/A"),
        "singleton_accuracy": result.get("predictions", {}).get("singleton_accuracy", "N/A"),
        "pair_precision": result.get("predictions", {}).get("pair_precision", "N/A"),
        "pair_recall": result.get("predictions", {}).get("pair_recall", "N/A"),
        "raw_oracle": result.get("raw_candidates", {}).get("oracle_macro_f05", "N/A"),
        "post_gate_oracle": result.get("post_gate_candidates", {}).get("oracle_macro_f05", "N/A"),
    }
    out_tsv = Path(config.get("output_summary_tsv", out_json.with_name("evaluation_summary.tsv"))); pd.DataFrame([summary]).to_csv(out_tsv, sep="\t", index=False)
    return result


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); a = p.parse_args(); run_evaluation(json.loads(Path(a.config).read_text(encoding="utf-8")))


if __name__ == "__main__": main()
