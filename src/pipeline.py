"""End-to-end cached training and test inference pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .candidates import candidate_diagnostics, generate_candidates, load_ground_truth, load_source, read_tsv, save_candidates
from .export import candidate_table_to_mapping, export_all, load_test_s1_ids, validate_submission_files
from .features import build_pair_features
from .predict_pair_model import predict_pair_model
from .split import create_folds, load_folds_tsv, save_folds_tsv
from .train_pair_model import train_pair_model


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _resolve(data_dir: Path, split: str, name: str) -> Path:
    return data_dir / split / name


def _cached(path: Path, force: bool) -> bool:
    return path.exists() and not force


def run_pipeline(data_dir: str | Path, work_dir: str | Path, output_dir: str | Path, n_splits: int = 3, top_k_name: int = 50, top_k_address: int = 50, batch_size: int = 2048, seed: int = 42, force: bool = False, positive_weight: str | float = "none") -> dict[str, object]:
    started = time.perf_counter()
    data_dir, work_dir, output_dir = Path(data_dir), Path(work_dir), Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    train = {name: _resolve(data_dir, "train", name) for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv")}
    test = {name: _resolve(data_dir, "test", name) for name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")}
    for path in [*train.values(), *test.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    folds_path = work_dir / "folds.tsv"
    if not _cached(folds_path, force):
        entity_to_fold, summary = create_folds(str(train["train_source1.tsv"]), str(train["train_ground_truth.tsv"]), str(train["train_source2.tsv"]), str(train["train_source3.tsv"]), n_splits, seed)
        save_folds_tsv(entity_to_fold, str(folds_path)); _write_json(work_dir / "fold_summary.json", summary)
    folds = load_folds_tsv(str(folds_path))

    train_candidates_path = work_dir / "train_candidates.tsv"; train_report_path = work_dir / "train_candidate_report.json"
    if not _cached(train_candidates_path, force):
        train_s1, train_s2, train_s3 = load_source(train["train_source1.tsv"]), load_source(train["train_source2.tsv"]), load_source(train["train_source3.tsv"])
        started_candidates = time.perf_counter(); train_candidates = generate_candidates(train_s1, train_s2, train_s3, top_k_name, top_k_address, batch_size); save_candidates(train_candidates, train_candidates_path)
        report = candidate_diagnostics(train_candidates, train_s1, train_s2, train_s3, load_ground_truth(train["train_ground_truth.tsv"]), time.perf_counter() - started_candidates); _write_json(train_report_path, report)
    elif not train_report_path.exists():
        train_s1, train_s2, train_s3 = load_source(train["train_source1.tsv"]), load_source(train["train_source2.tsv"]), load_source(train["train_source3.tsv"])
        _write_json(train_report_path, candidate_diagnostics(pd.read_csv(train_candidates_path, sep="\t", dtype=str), train_s1, train_s2, train_s3, load_ground_truth(train["train_ground_truth.tsv"])))

    train_features_path = work_dir / "train_features.parquet"
    feature_runtime = None
    if not _cached(train_features_path, force):
        feature_started = time.perf_counter(); features = build_pair_features(str(train["train_source1.tsv"]), str(train["train_source2.tsv"]), str(train["train_source3.tsv"]), str(train_candidates_path), str(train["train_ground_truth.tsv"]), str(folds_path)); features.to_parquet(train_features_path, index=False); feature_runtime = time.perf_counter() - feature_started
    model_dir = work_dir / "model"
    required_model = [model_dir / name for name in ("final_model.joblib", "feature_columns.json", "training_config.json", "oof_predictions.parquet", "oof_predictions.tsv", "best_threshold.json", "metrics.json")]
    if force or not all(path.exists() for path in required_model):
        train_pair_model(str(train_features_path), str(train["train_ground_truth.tsv"]), str(model_dir), positive_weight, str(train["train_source1.tsv"]), seed, feature_runtime)

    test_candidates_path = work_dir / "test_candidates.tsv"
    if not _cached(test_candidates_path, force):
        test_s1, test_s2, test_s3 = load_source(test["test_source1.tsv"]), load_source(test["test_source2.tsv"]), load_source(test["test_source3.tsv"])
        save_candidates(generate_candidates(test_s1, test_s2, test_s3, top_k_name, top_k_address, batch_size), test_candidates_path)
    test_features_path = work_dir / "test_features.parquet"
    if not _cached(test_features_path, force):
        features = build_pair_features(str(test["test_source1.tsv"]), str(test["test_source2.tsv"]), str(test["test_source3.tsv"]), str(test_candidates_path)); features.to_parquet(test_features_path, index=False)
    score_dir = work_dir / "test_scores"
    if force or not (score_dir / "predictions.json").exists() or not (score_dir / "test_pair_scores.parquet").exists():
        test_s1_ids = load_test_s1_ids(str(test["test_source1.tsv"])); predict_pair_model(str(test_features_path), str(model_dir), str(score_dir), test_s1_ids)
    predictions = json.loads((score_dir / "predictions.json").read_text(encoding="utf-8"))
    test_candidates = pd.read_csv(test_candidates_path, sep="\t", dtype=str, keep_default_na=False)
    candidate_mapping = candidate_table_to_mapping(test_candidates, load_test_s1_ids(str(test["test_source1.tsv"])))
    matching_path, candidate_path = export_all(load_test_s1_ids(str(test["test_source1.tsv"])), predictions, candidate_mapping, str(output_dir))
    validate_submission_files(matching_path, candidate_path, str(test["test_source1.tsv"]), str(test["test_source2.tsv"]), str(test["test_source3.tsv"]))
    validator = data_dir.parent / "utils" / "validate_submission.py"
    if validator.exists():
        result = subprocess.run([sys.executable, str(validator), "--matching", matching_path, "--candidate", candidate_path, "--test-dir", str(data_dir / "test")], capture_output=True, text=True)
        if result.returncode != 0 or "PASS" not in result.stdout:
            raise RuntimeError(f"Official submission validator failed:\n{result.stdout}\n{result.stderr}")
    return {"matching_results": matching_path, "candidate_pairs": candidate_path, "model_dir": str(model_dir), "runtime_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete entity-resolution pipeline.")
    parser.add_argument("--data-dir", required=True); parser.add_argument("--work-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--n-splits", type=int, default=3); parser.add_argument("--top-k-name", type=int, default=50); parser.add_argument("--top-k-address", type=int, default=50); parser.add_argument("--batch-size", type=int, default=2048); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--positive-weight", default="none"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    setting: str | float = args.positive_weight if args.positive_weight in {"none", "auto"} else float(args.positive_weight)
    result = run_pipeline(args.data_dir, args.work_dir, args.output_dir, args.n_splits, args.top_k_name, args.top_k_address, args.batch_size, args.seed, args.force, setting)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
