"""Score pair features with saved tree model while preserving Member D contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from .member_d_contracts import (
    PAIR_KEY,
    derive_candidate_source,
    feature_columns_sha256,
    validate_feature_allowlist,
)
from .model_config import CANDIDATE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION, BaselineModelConfig
from .train_pair_model import _feature_matrix, predictions_from_threshold


def _read_features(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _load_training_provenance(model_dir: Path) -> dict:
    for name in ("training_provenance.json", "model_manifest.json"):
        p = model_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def predict_pair_model(
    test_features: str | Path | pd.DataFrame,
    model_dir: str | Path,
    output_dir: str | Path,
    s1_ids: Iterable[str] | None = None,
    threshold: float | None = None,
    feature_schema_version: str | None = None,
    candidate_schema_version: str | None = None,
    *,
    tree_model_version: str | None = None,
    training_git_sha: str | None = None,
    candidate_version: str | None = None,
) -> dict[str, object]:
    model_dir = Path(model_dir); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_dir / "final_model.joblib")
    feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    validate_feature_allowlist(feature_columns)

    config_path = model_dir / "training_config.json"
    config = None
    if config_path.exists():
        config = BaselineModelConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
        if list(config.feature_columns) != list(feature_columns):
            raise ValueError("Model feature column order does not match training_config.json")
        if config.feature_schema_version != (feature_schema_version or FEATURE_SCHEMA_VERSION):
            raise ValueError("Feature schema version does not match the trained model")
        if config.candidate_schema_version != (candidate_schema_version or CANDIDATE_SCHEMA_VERSION):
            raise ValueError("Candidate schema version does not match the trained model")

    frame = _read_features(test_features) if not isinstance(test_features, pd.DataFrame) else test_features.copy()
    required = {"source1_entity_id", "candidate_entity_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Test feature table is missing columns: {sorted(required - set(frame.columns))}")
    missing_features = [c for c in feature_columns if c not in frame.columns]
    if missing_features:
        raise ValueError(f"Test feature table is missing trained feature columns: {missing_features}")
    if "candidate_source" not in frame.columns:
        frame["candidate_source"] = frame["candidate_entity_id"].astype(str).map(derive_candidate_source)

    scores = _feature_matrix(frame, feature_columns)
    out = frame[PAIR_KEY].copy()
    classes = list(getattr(model, "classes_", []))
    out["score"] = model.predict_proba(scores)[:, classes.index(1)] if 1 in classes else 0.0
    out["score"] = pd.to_numeric(out["score"], errors="raise")
    if out.duplicated(PAIR_KEY).any():
        # Deterministic duplicate resolution using highest score, then input order.
        out["__order"] = range(len(out))
        out = out.sort_values([*PAIR_KEY, "score", "__order"], ascending=[True, True, True, False, True], kind="stable").drop_duplicates(PAIR_KEY, keep="first").sort_values("__order").drop(columns="__order")

    out.to_parquet(output_dir / "test_pair_scores.parquet", index=False)
    out.to_csv(output_dir / "test_pair_scores.tsv", sep="\t", index=False)

    if threshold is None:
        threshold = float(json.loads((model_dir / "best_threshold.json").read_text(encoding="utf-8"))["threshold"])
    predictions = predictions_from_threshold(out[["source1_entity_id", "candidate_entity_id", "score"]], float(threshold))
    if s1_ids is not None:
        predictions = {str(s): predictions.get(str(s), []) for s in s1_ids}
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")

    upstream = _load_training_provenance(model_dir)
    train_sha = training_git_sha or upstream.get("training_git_sha") or upstream.get("git_commit")
    model_version = tree_model_version or upstream.get("tree_model_version") or model_dir.name
    manifest = {
        "tree_model_version": model_version,
        "training_git_sha": train_sha,
        "feature_columns": feature_columns,
        "feature_columns_sha256": feature_columns_sha256(feature_columns),
        "feature_schema_version": config.feature_schema_version if config is not None else feature_schema_version or FEATURE_SCHEMA_VERSION,
        "candidate_schema_version": config.candidate_schema_version if config is not None else candidate_schema_version or CANDIDATE_SCHEMA_VERSION,
        "candidate_version": candidate_version or upstream.get("candidate_version"),
        "tree_feature_contract_validated": True,
        "tree_feature_contract_validation_version": "member_d_v1",
        "feature_contract_valid": True,
        "provenance_complete": bool(train_sha and model_version),
        "invalid_for_selection": False,
    }
    (output_dir / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"scores": out, "predictions": predictions, "threshold": float(threshold), "manifest": manifest}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True); p.add_argument("--model-dir", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--s1-ids"); p.add_argument("--tree-model-version"); p.add_argument("--training-git-sha"); p.add_argument("--candidate-version")
    args = p.parse_args()
    ids = None
    if args.s1_ids:
        ids = pd.read_csv(args.s1_ids, sep="\t", dtype=str, keep_default_na=False)["entity_id"].tolist()
    predict_pair_model(args.features, args.model_dir, args.output_dir, ids, tree_model_version=args.tree_model_version, training_git_sha=args.training_git_sha, candidate_version=args.candidate_version)


if __name__ == "__main__": main()
