"""Score test pair features with saved training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import pandas as pd

from .train_pair_model import _feature_matrix, predictions_from_threshold
from .model_config import CANDIDATE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION, BaselineModelConfig


def _read_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)


def predict_pair_model(test_features: str | Path | pd.DataFrame, model_dir: str | Path, output_dir: str | Path, s1_ids: Iterable[str] | None = None, threshold: float | None = None, feature_schema_version: str | None = None, candidate_schema_version: str | None = None) -> dict[str, object]:
    model_dir = Path(model_dir); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_dir / "final_model.joblib")
    feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))

    from .member_d_contracts import validate_feature_allowlist
    if "is_injected_positive" in feature_columns or "positive_injected_for_training" in feature_columns:
        raise ValueError("Tree model is tainted with injected positive features.")
    validate_feature_allowlist(feature_columns)

    config_path = model_dir / "training_config.json"
    if config_path.exists():
        config = BaselineModelConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
        if list(config.feature_columns) != list(feature_columns):
            raise ValueError("Model feature column order does not match training_config.json")
        if config.feature_schema_version != (feature_schema_version or FEATURE_SCHEMA_VERSION):
            raise ValueError("Feature schema version does not match the trained model")
        if config.candidate_schema_version != (candidate_schema_version or CANDIDATE_SCHEMA_VERSION):
            raise ValueError("Candidate schema version does not match the trained model")
    elif feature_schema_version not in (None, FEATURE_SCHEMA_VERSION) or candidate_schema_version not in (None, CANDIDATE_SCHEMA_VERSION):
        raise ValueError("Schema version verification requires training_config.json")
    frame = _read_features(test_features) if not isinstance(test_features, pd.DataFrame) else test_features.copy()
    required = {"source1_entity_id", "candidate_entity_id"}
    missing_ids = required - set(frame.columns)
    if missing_ids:
        raise ValueError(f"Test feature table is missing columns: {sorted(missing_ids)}")
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Test feature table is missing trained feature columns: {missing_features}")
    # Labels and endpoint folds are deliberately ignored even if accidentally present.
    scores = _feature_matrix(frame, feature_columns)
    # §1.2  Preserve candidate_source so downstream D code can join on the full
    # 3-part pair key. Derive from entity_id prefix when the input feature table
    # lacks the column (old inputs); fail on ambiguous IDs.
    _preserve = ["source1_entity_id", "candidate_entity_id"]
    if "candidate_source" in frame.columns:
        _preserve.append("candidate_source")
    else:
        from .member_d_contracts import derive_candidate_source
        frame = frame.copy()
        frame["candidate_source"] = (
            frame["candidate_entity_id"].astype(str).map(derive_candidate_source)
        )
        _preserve.append("candidate_source")
    frame = frame[_preserve].copy()
    frame["score"] = model.predict_proba(scores)[:, list(model.classes_).index(1)] if 1 in list(getattr(model, "classes_", [])) else 0.0
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    frame["__order"] = range(len(frame))
    frame = frame.sort_values(["source1_entity_id", "candidate_entity_id", "score", "__order"], ascending=[True, True, False, True], kind="stable").drop_duplicates(["source1_entity_id", "candidate_entity_id"], keep="first").sort_values("__order", kind="stable").drop(columns="__order")
    frame.to_parquet(output_dir / "test_pair_scores.parquet", index=False)
    frame.to_csv(output_dir / "test_pair_scores.tsv", sep="\t", index=False)
    if threshold is None:
        threshold = float(json.loads((model_dir / "best_threshold.json").read_text(encoding="utf-8"))["threshold"])
    predictions = predictions_from_threshold(frame, threshold)
    if s1_ids is not None:
        predictions = {str(s1): predictions.get(str(s1), []) for s1 in s1_ids}
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    
    import hashlib
    manifest = {
        "tree_feature_contract_validated": True,
        "tree_feature_contract_validation_version": "member_d_v1",
        "tree_feature_columns_sha256": hashlib.sha256(json.dumps(feature_columns).encode()).hexdigest(),
        "tree_training_git_sha": config.git_commit if config_path.exists() and hasattr(config, 'git_commit') else None,
        "tree_feature_schema_version": config.feature_schema_version if config_path.exists() and hasattr(config, 'feature_schema_version') else None,
    }
    if "is_injected_positive" in feature_columns:
        manifest["invalid_for_selection"] = True
        manifest["invalid_reason"] = "tree training feature leakage: is_injected_positive"
        manifest["feature_contract_valid"] = False
    else:
        manifest["feature_contract_valid"] = True
    (output_dir / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"scores": frame, "predictions": predictions, "threshold": float(threshold)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score test pair features with a saved pair model.")
    parser.add_argument("--features", required=True); parser.add_argument("--model-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--s1-ids", default=None)
    args = parser.parse_args()
    ids = None
    if args.s1_ids:
        ids = pd.read_csv(args.s1_ids, sep="\t", dtype=str, keep_default_na=False)["entity_id"].tolist()
    result = predict_pair_model(args.features, args.model_dir, args.output_dir, ids)
    print(f"Scored {len(result['scores'])} test pairs at threshold {result['threshold']:.6f}")


if __name__ == "__main__":
    main()
