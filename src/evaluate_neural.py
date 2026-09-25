"""Threshold tuning and frozen-subset evaluation for neural predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .neural_data import parse_ground_truth, read_text_tsv
from .neural_models import evaluate_scores, tune_threshold
from .neural_contracts import atomic_write_json, sha256_file


def _read_many(raw, *, score: bool = False) -> pd.DataFrame:
    values = list(map(str, raw)) if isinstance(raw, list) else ([str(raw)] if raw else [])
    paths: list[str] = []
    pattern = "scores-*.parquet" if score else "*.parquet"
    for value in values:
        paths.extend(str(path) for path in sorted(Path(value).glob(pattern))) if Path(value).is_dir() else paths.append(value)
    frames = [pd.read_parquet(path) if Path(path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False) for path in paths]
    if frames:
        return pd.concat(frames, ignore_index=True)
    columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score"] if score else ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    return pd.DataFrame(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate neural pair probabilities with official macro-F0.5 behavior.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if not args.execute:
        print(json.dumps({"command": "evaluate_neural", "execute_required": True, "threshold_subset": "deferred", "final_subset": "deferred"}, indent=2))
        return
    prediction_manifest = config.get("prediction_manifest")
    if prediction_manifest:
        manifest_paths = prediction_manifest if isinstance(prediction_manifest, list) else [prediction_manifest]
        for path in manifest_paths:
            manifest_value = json.loads(Path(path).read_text(encoding="utf-8"))
            if manifest_value.get("completion") != "complete":
                raise ValueError(f"prediction manifest is incomplete: {path}")
            for name, expected in manifest_value.get("generated_file_checksums", {}).items():
                shard = Path(path).parent / name
                if not shard.is_file() or sha256_file(shard) != expected:
                    raise ValueError(f"prediction shard is missing or changed: {shard}")
    scores = _read_many(config["scores"], score=True)
    scores["score"] = scores["score"].astype(float)
    candidates = None
    if config.get("candidates"):
        candidates = _read_many(config["candidates"])
    gt = parse_ground_truth(config["ground_truth"])
    manifest = json.loads(Path(config["query_manifest"]).read_text(encoding="utf-8"))
    query_sets = manifest["query_ids"]
    tuning_candidates = candidates[candidates["source1_entity_id"].astype(str).isin(set(query_sets["threshold"]))] if candidates is not None else None
    final_candidates = candidates[candidates["source1_entity_id"].astype(str).isin(set(query_sets["final"]))] if candidates is not None else None
    tuning = tune_threshold(scores, gt, query_sets["threshold"], candidates=tuning_candidates, thresholds=config.get("thresholds"))
    final = evaluate_scores(scores, gt, query_sets["final"], tuning["threshold"], candidate_pairs=final_candidates, query_metadata=manifest.get("query_metadata"))
    result = {"threshold_tuning": tuning, "final_comparison": final, "threshold_frozen_before_final": True}
    atomic_write_json(config.get("output", "neural_evaluation.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
