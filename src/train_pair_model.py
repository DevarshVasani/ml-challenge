"""Leakage-safe OOF training, threshold selection, and final model artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from .evaluate import evaluate_predictions
from .split import load_ground_truth


PAIR_ID_COLUMNS = {"source1_entity_id", "candidate_entity_id"}
NON_FEATURE_COLUMNS = PAIR_ID_COLUMNS | {"label", "fold", "source1_fold", "candidate_fold"}
AUTO_POSITIVE_WEIGHT_MIN = 1.0
AUTO_POSITIVE_WEIGHT_MAX = 20.0


def identify_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and (pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]))
    ]
    if not columns:
        raise ValueError("No numeric pair feature columns were found")
    return columns


def _feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature table is missing model columns: {missing}")
    values = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _pair_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("source1_entity_id", sort=False)["candidate_entity_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def _new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=0.5, random_state=42)


def _positive_multiplier(y_train: np.ndarray, setting: str | float) -> float:
    if isinstance(setting, str):
        mode = setting.casefold()
        if mode == "none":
            return 1.0
        if mode != "auto":
            raise ValueError("positive_weight must be none, auto, or a positive float")
        positives = int(np.sum(y_train == 1)); negatives = int(np.sum(y_train == 0))
        raw = negatives / positives if positives else 1.0
        return float(np.clip(raw, AUTO_POSITIVE_WEIGHT_MIN, AUTO_POSITIVE_WEIGHT_MAX))
    value = float(setting)
    if value <= 0:
        raise ValueError("positive_weight float must be greater than zero")
    return value


def _training_weights(frame: pd.DataFrame, y: np.ndarray, positive_weight: str | float) -> tuple[np.ndarray, float]:
    multiplier = _positive_multiplier(y, positive_weight)
    weights = _pair_weights(frame)
    return weights * np.where(y == 1, multiplier, 1.0), multiplier


def _fit_classifier(x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray, classifier_factory: Callable[[], object] | None = None) -> object:
    classifier: object = DummyClassifier(strategy="prior") if np.unique(y_train).size < 2 else (classifier_factory() if classifier_factory else _new_classifier())
    classifier.fit(x_train, y_train, sample_weight=weights)
    return classifier


def _positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(getattr(model, "classes_", []))
    return np.asarray(probabilities)[:, classes.index(1)] if 1 in classes else np.zeros(len(x), dtype=float)


def deduplicate_pair_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per pair, preferring the highest retrieval score."""
    required = PAIR_ID_COLUMNS - set(frame.columns)
    if required:
        raise ValueError(f"Training feature table is missing pair IDs: {sorted(required)}")
    if not frame.duplicated(list(PAIR_ID_COLUMNS)).any():
        return frame.reset_index(drop=True)
    result = frame.copy()
    result["__row_order"] = np.arange(len(result))
    if "retrieval_score" in result.columns:
        result["__retrieval_score"] = pd.to_numeric(result["retrieval_score"], errors="coerce").fillna(-np.inf)
        result = result.sort_values(["__retrieval_score", "__row_order"], ascending=[False, True], kind="stable")
    result = result.drop_duplicates(list(PAIR_ID_COLUMNS), keep="first").sort_values("__row_order", kind="stable")
    return result.drop(columns=[c for c in ("__row_order", "__retrieval_score") if c in result]).reset_index(drop=True)


def _with_endpoint_folds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "source1_fold" not in result.columns or "candidate_fold" not in result.columns:
        if "fold" not in result.columns:
            raise ValueError("Training feature table needs source1_fold/candidate_fold or legacy fold")
        result["source1_fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
        result["candidate_fold"] = result["source1_fold"]
    result["source1_fold"] = pd.to_numeric(result["source1_fold"], errors="raise").astype(int)
    result["candidate_fold"] = pd.to_numeric(result["candidate_fold"], errors="raise").astype(int)
    return result


def generate_oof_predictions(frame: pd.DataFrame, feature_columns: Sequence[str] | None = None, classifier_factory: Callable[[], object] | None = None, positive_weight: str | float = "none") -> tuple[pd.DataFrame, list[str]]:
    frame = deduplicate_pair_rows(_with_endpoint_folds(frame))
    required = {"source1_entity_id", "candidate_entity_id", "source1_fold", "candidate_fold", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training feature table is missing columns: {sorted(missing)}")
    feature_columns = list(feature_columns or identify_feature_columns(frame))
    x = _feature_matrix(frame, feature_columns)
    y = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
    source_folds = frame["source1_fold"].to_numpy(dtype=int)
    candidate_folds = frame["candidate_fold"].to_numpy(dtype=int)
    oof_scores = np.full(len(frame), np.nan, dtype=float)
    unique_folds = sorted(set(source_folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required for out-of-fold predictions")
    fold_multipliers: dict[str, float] = {}
    for fold in unique_folds:
        validation = source_folds == fold
        training = (source_folds != fold) & (candidate_folds != fold)
        if not validation.any() or not training.any():
            raise ValueError(f"Fold {fold} cannot be evaluated with an empty train or validation partition")
        weights, multiplier = _training_weights(frame.loc[training], y[training], positive_weight)
        fold_multipliers[str(fold)] = multiplier
        model = _fit_classifier(x[training], y[training], weights, classifier_factory)
        oof_scores[validation] = _positive_probability(model, x[validation])
    if not np.isfinite(oof_scores).all():
        raise RuntimeError("OOF prediction generation left non-finite scores")
    output = frame[["source1_entity_id", "candidate_entity_id", "source1_fold", "candidate_fold", "label"]].copy()
    output["score"] = oof_scores
    output.attrs["positive_weight_multipliers"] = fold_multipliers
    return output, feature_columns


cross_validated_predictions = generate_oof_predictions


def predictions_from_threshold(pair_scores: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    required = {"source1_entity_id", "candidate_entity_id", "score"}
    missing = required - set(pair_scores.columns)
    if missing:
        raise ValueError(f"Pair scores are missing columns: {sorted(missing)}")
    frame = pair_scores.copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    frame["__order"] = np.arange(len(frame))
    frame = frame.sort_values(["score", "__order"], ascending=[False, True], kind="stable").drop_duplicates(["source1_entity_id", "candidate_entity_id"], keep="first").sort_values("__order", kind="stable")
    predictions: dict[str, list[str]] = {}
    for row in frame.loc[frame["score"] >= float(threshold)].itertuples(index=False):
        predictions.setdefault(str(row.source1_entity_id), []).append(str(row.candidate_entity_id))
    return {key: list(dict.fromkeys(values)) for key, values in predictions.items()}


pair_scores_to_predictions = predictions_from_threshold
threshold_to_predictions = predictions_from_threshold


def search_threshold(oof_predictions: pd.DataFrame, ground_truth: Mapping[str, Iterable[str]], thresholds: Sequence[float] | None = None) -> tuple[float, dict[str, object]]:
    if thresholds is not None:
        grid = sorted(set(float(value) for value in thresholds))
    else:
        coarse = np.linspace(0.0, 1.0, 101).tolist()
        coarse_results = [(threshold, evaluate_predictions(dict(ground_truth), predictions_from_threshold(oof_predictions, threshold))) for threshold in coarse]
        best_coarse = max(coarse_results, key=lambda item: (float(item[1]["macro_f05"]), item[0]))[0]
        fine = np.linspace(max(0.0, best_coarse - 0.05), min(1.0, best_coarse + 0.05), 101).tolist()
        grid = sorted(set(coarse + fine))
    if not grid:
        raise ValueError("Threshold grid is empty")
    results = [(threshold, evaluate_predictions(dict(ground_truth), predictions_from_threshold(oof_predictions, threshold))) for threshold in grid]
    return max(results, key=lambda item: (float(item[1]["macro_f05"]), item[0]))


def _fit_final_model(frame: pd.DataFrame, feature_columns: Sequence[str], positive_weight: str | float = "none") -> tuple[object, float]:
    x = _feature_matrix(frame, feature_columns)
    y = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
    weights, multiplier = _training_weights(frame, y, positive_weight)
    return _fit_classifier(x, y, weights), multiplier


def _country_metrics(oof: pd.DataFrame, ground_truth: Mapping[str, Iterable[str]], threshold: float, s1_path: str | os.PathLike[str] | None) -> dict[str, float]:
    if not s1_path:
        return {}
    source = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    if "country" not in source.columns:
        return {}
    country_by_id = source.set_index("entity_id")["country"].to_dict()
    predictions = predictions_from_threshold(oof, threshold)
    result: dict[str, float] = {}
    for country in sorted({str(country_by_id.get(sid, "UNKNOWN")) for sid in ground_truth}):
        ids = {sid for sid in ground_truth if str(country_by_id.get(sid, "UNKNOWN")) == country}
        if ids:
            result[country] = float(evaluate_predictions({sid: ground_truth[sid] for sid in ids}, {sid: predictions.get(sid, []) for sid in ids})["macro_f05"])
    return result


def train_pair_model(feature_path: str | os.PathLike[str], ground_truth_path: str | os.PathLike[str], output_dir: str | os.PathLike[str], positive_weight: str | float = "none", s1_path: str | os.PathLike[str] | None = None, seed: int = 42, feature_generation_runtime_seconds: float | None = None) -> dict[str, object]:
    started = time.perf_counter()
    frame = pd.read_parquet(feature_path)
    frame = deduplicate_pair_rows(_with_endpoint_folds(frame))
    ground_truth = load_ground_truth(str(ground_truth_path))
    oof, feature_columns = generate_oof_predictions(frame, positive_weight=positive_weight)
    threshold, selected = search_threshold(oof, ground_truth)
    predictions = predictions_from_threshold(oof, threshold)
    positive_count = sum(len(values) for values in predictions.values())
    metrics = {key: value for key, value in selected.items() if key != "per_entity_scores"}
    metrics.update({
        "selected_threshold": float(threshold), "num_positive_predictions": int(positive_count),
        "positive_prediction_percentage": 100.0 * positive_count / len(oof) if len(oof) else 0.0,
        "num_positive_entities": len(predictions), "positive_entity_percentage": 100.0 * len(predictions) / len(ground_truth) if ground_truth else 0.0,
        "per_country_macro_f05": _country_metrics(oof, ground_truth, threshold, s1_path),
        "positive_weight": positive_weight, "auto_positive_weight_range": [AUTO_POSITIVE_WEIGHT_MIN, AUTO_POSITIVE_WEIGHT_MAX],
        "oof_positive_weight_multipliers": oof.attrs.get("positive_weight_multipliers", {}),
        "runtime_seconds": time.perf_counter() - started,
        "feature_generation_runtime_seconds": feature_generation_runtime_seconds,
    })
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    final_model, final_multiplier = _fit_final_model(frame, feature_columns, positive_weight)
    metrics["final_positive_weight_multiplier"] = final_multiplier
    joblib.dump(final_model, output_path / "final_model.joblib")
    (output_path / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (output_path / "training_config.json").write_text(json.dumps({"positive_weight": positive_weight, "seed": seed, "feature_columns": feature_columns}, indent=2), encoding="utf-8")
    oof.to_parquet(output_path / "oof_predictions.parquet", index=False); oof.to_csv(output_path / "oof_predictions.tsv", sep="\t", index=False)
    (output_path / "best_threshold.json").write_text(json.dumps({"threshold": float(threshold)}, indent=2), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "threshold": threshold, "predictions": predictions, "feature_columns": feature_columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pair-level entity-resolution baseline.")
    parser.add_argument("--features", required=True); parser.add_argument("--ground-truth", "--gt", dest="ground_truth", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-weight", default="none", help="none, auto, or a positive multiplier")
    parser.add_argument("--s1", default=None); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setting: str | float = args.positive_weight
    if setting not in {"none", "auto"}:
        setting = float(setting)
    result = train_pair_model(args.features, args.ground_truth, args.output_dir, setting, args.s1, args.seed)
    metrics = result["metrics"]
    print(f"Macro F0.5: {metrics['macro_f05']:.6f}")
    print(f"Singleton accuracy: {metrics['singleton_accuracy']:.6f}")
    print(f"Non-singleton macro F0.5: {metrics['non_singleton_macro_f05']:.6f}")
    print(f"Selected threshold: {metrics['selected_threshold']:.4f}")


if __name__ == "__main__":
    main()
