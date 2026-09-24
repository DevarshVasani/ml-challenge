"""Cross-validated baseline model for pair-level entity resolution features."""

from __future__ import annotations

import argparse
import json
import os
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
NON_FEATURE_COLUMNS = PAIR_ID_COLUMNS | {"label", "fold"}


def identify_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric model columns, excluding IDs, labels, and fold numbers."""
    columns: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric pair feature columns were found")
    return columns


def _feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature table is missing model columns: {missing}")
    values = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def _pair_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("source1_entity_id", sort=False)["candidate_entity_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def _new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=42,
    )


def _fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    classifier_factory: Callable[[], object] | None = None,
) -> object:
    # A fold containing one class is valid for a small/synthetic dataset.  A
    # prior-only model keeps the OOF contract intact instead of failing in fit.
    if np.unique(y_train).size < 2:
        classifier: object = DummyClassifier(strategy="prior")
    else:
        classifier = classifier_factory() if classifier_factory else _new_classifier()
    classifier.fit(x_train, y_train, sample_weight=weights)
    return classifier


def _positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return np.asarray(probabilities)[:, classes.index(1)]
    return np.zeros(len(x), dtype=float)


def generate_oof_predictions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    classifier_factory: Callable[[], object] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fit one model per held-out fold and return predictions in input order."""
    required = {"source1_entity_id", "candidate_entity_id", "fold", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training feature table is missing columns: {sorted(missing)}")
    feature_columns = list(feature_columns or identify_feature_columns(frame))
    x = _feature_matrix(frame, feature_columns)
    y = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
    folds = pd.to_numeric(frame["fold"], errors="raise").to_numpy(dtype=int)
    oof_scores = np.full(len(frame), np.nan, dtype=float)
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("At least two folds are required for out-of-fold predictions")
    weights = _pair_weights(frame)
    for fold in unique_folds:
        validation = folds == fold
        training = ~validation
        if not validation.any() or not training.any():
            raise ValueError(f"Fold {fold} cannot be evaluated with an empty train or validation partition")
        model = _fit_classifier(x[training], y[training], weights[training], classifier_factory)
        oof_scores[validation] = _positive_probability(model, x[validation])
    if not np.isfinite(oof_scores).all():
        raise RuntimeError("OOF prediction generation left non-finite scores")
    output = frame[["source1_entity_id", "candidate_entity_id", "fold", "label"]].copy()
    output["score"] = oof_scores
    return output, feature_columns


# Alias useful to callers using the terminology in the task brief.
cross_validated_predictions = generate_oof_predictions


def predictions_from_threshold(
    pair_scores: pd.DataFrame,
    threshold: float,
) -> dict[str, list[str]]:
    """Convert pair probabilities to zero-, one-, or many-match predictions."""
    required = {"source1_entity_id", "candidate_entity_id", "score"}
    missing = required - set(pair_scores.columns)
    if missing:
        raise ValueError(f"Pair scores are missing columns: {sorted(missing)}")
    predictions: dict[str, list[str]] = {}
    for row in pair_scores.loc[pair_scores["score"] >= threshold].itertuples(index=False):
        predictions.setdefault(str(row.source1_entity_id), []).append(str(row.candidate_entity_id))
    return predictions


# Additional descriptive alias for downstream scripts.
pair_scores_to_predictions = predictions_from_threshold
threshold_to_predictions = predictions_from_threshold


def search_threshold(
    oof_predictions: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, dict[str, object]]:
    """Select the threshold with the best repository-defined macro F0.5."""
    grid = list(thresholds) if thresholds is not None else [0.0, *np.linspace(0.05, 0.95, 19).tolist(), 1.0]
    results: list[tuple[float, dict[str, object]]] = []
    for threshold in sorted(set(float(value) for value in grid)):
        predictions = predictions_from_threshold(oof_predictions, threshold)
        metrics = evaluate_predictions(dict(ground_truth), predictions)
        results.append((threshold, metrics))
    # Higher threshold wins an exact tie because it is the safer deterministic
    # choice for this precision-weighted metric.
    return max(results, key=lambda item: (float(item[1]["macro_f05"]), item[0]))


def _fit_final_model(frame: pd.DataFrame, feature_columns: Sequence[str]) -> object:
    x = _feature_matrix(frame, feature_columns)
    y = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
    return _fit_classifier(x, y, _pair_weights(frame))


def train_pair_model(
    feature_path: str | os.PathLike[str],
    ground_truth_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Train, evaluate, and save the baseline artifacts."""
    frame = pd.read_parquet(feature_path)
    ground_truth = load_ground_truth(str(ground_truth_path))
    oof, feature_columns = generate_oof_predictions(frame)
    threshold, selected = search_threshold(oof, ground_truth)
    predictions = predictions_from_threshold(oof, threshold)
    positive_count = int((oof["score"] >= threshold).sum())
    metrics = {key: value for key, value in selected.items() if key != "per_entity_scores"}
    metrics.update(
        {
            "selected_threshold": float(threshold),
            "num_positive_predictions": positive_count,
            "positive_prediction_percentage": (100.0 * positive_count / len(oof)) if len(oof) else 0.0,
            "num_positive_entities": len(predictions),
            "positive_entity_percentage": (100.0 * len(predictions) / len(ground_truth)) if ground_truth else 0.0,
        }
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_model = _fit_final_model(frame, feature_columns)
    joblib.dump(final_model, output_path / "final_model.joblib")
    (output_path / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    oof.to_parquet(output_path / "oof_predictions.parquet", index=False)
    oof.to_csv(output_path / "oof_predictions.tsv", sep="\t", index=False)
    (output_path / "best_threshold.json").write_text(
        json.dumps({"threshold": float(threshold)}, indent=2), encoding="utf-8"
    )
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "threshold": threshold, "predictions": predictions, "feature_columns": feature_columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pair-level entity-resolution baseline.")
    parser.add_argument("--features", required=True, help="Training pair feature Parquet")
    parser.add_argument("--ground-truth", "--gt", dest="ground_truth", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = train_pair_model(args.features, args.ground_truth, args.output_dir)
    metrics = result["metrics"]
    print(f"Macro F0.5: {metrics['macro_f05']:.6f}")
    print(f"Singleton accuracy: {metrics['singleton_accuracy']:.6f}")
    print(f"Non-singleton macro F0.5: {metrics['non_singleton_macro_f05']:.6f}")
    print(f"Selected threshold: {metrics['selected_threshold']:.4f}")
    print(
        f"Positive predictions: {metrics['num_positive_predictions']} "
        f"({metrics['positive_prediction_percentage']:.2f}%)"
    )


if __name__ == "__main__":
    main()
