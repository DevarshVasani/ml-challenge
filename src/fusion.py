"""Leakage-safe route-specific logistic fusion for Member D."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .member_d_contracts import (
    DECISION_ROUTE, MATCH_PROBABILITY, NEURAL_LOGIT, NEURAL_PROBABILITY, PAIR_KEY,
    ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY, TREE_SCORE,
    validate_feature_allowlist, validate_joined_scores,
)
from .member_d_manifest import MemberDManifest, write_manifest
from .neural_contracts import sha256_file

DEFAULT_NEURAL_FEATURES = [
    TREE_SCORE, "tree_logit", "neural_logit_feature",
    "competing_s1_count", "competing_s1_score_gap", "competing_best_second_gap",
    "competing_near_tie_count", "is_top_competing_s1",
    "postcode_match_house_mismatch", "legal_form_agreement", "house_number_match",
    "name_tokens_extra_in_b", "name_tokens_missing_from_b",
    "source_s2_indicator", "source_s3_indicator",
]
DEFAULT_TREE_FEATURES = [f for f in DEFAULT_NEURAL_FEATURES if f not in {"neural_logit_feature"}]


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path); return pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _join_labels(scores: pd.DataFrame, labels_path: str | Path) -> pd.DataFrame:
    labels = _read(labels_path)
    needed = set(PAIR_KEY + ["label"])
    if not needed.issubset(labels.columns): raise ValueError(f"Label table missing: {sorted(needed - set(labels.columns))}")
    if labels.duplicated(PAIR_KEY).any(): raise ValueError("Duplicate keyed labels")
    extra = set(map(tuple, labels[PAIR_KEY].astype(str).to_numpy())) - set(map(tuple, scores[PAIR_KEY].astype(str).to_numpy()))
    if extra: raise ValueError(f"Labels contain unknown pairs: {sorted(extra)[:5]}")
    return scores.merge(labels[PAIR_KEY + ["label"]], on=PAIR_KEY, how="left", validate="1:1")


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    tree = pd.to_numeric(out[TREE_SCORE], errors="raise").clip(1e-6, 1 - 1e-6)
    out["tree_logit"] = np.log(tree / (1 - tree))
    if NEURAL_LOGIT in out.columns:
        out["neural_logit_feature"] = pd.to_numeric(out[NEURAL_LOGIT], errors="coerce")
    elif NEURAL_PROBABILITY in out.columns:
        p = pd.to_numeric(out[NEURAL_PROBABILITY], errors="coerce").clip(1e-6, 1 - 1e-6)
        out["neural_logit_feature"] = np.log(p / (1 - p))
    out["source_s2_indicator"] = (out["candidate_source"].astype(str) == "S2").astype(int)
    out["source_s3_indicator"] = (out["candidate_source"].astype(str) == "S3").astype(int)
    return out


def _matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [f for f in features if f not in frame.columns]
    if missing: raise ValueError(f"Fusion feature columns missing: {missing}")
    x = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        bad = x.columns[x.isna().any()].tolist(); raise ValueError(f"Fusion features contain missing/non-numeric values: {bad}")
    return x.to_numpy(float)


def fit_fusion(
    joined_scores_path: str | Path,
    labels_path: str | Path,
    partition_split_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    scores = _read(joined_scores_path); validate_joined_scores(scores)
    split = pd.read_csv(partition_split_path, sep="\t", dtype=str, keep_default_na=False)
    fit_ids = set(split.loc[split.member_d_partition == "fusion_fit", "source1_entity_id"])
    labeled = _join_labels(scores[scores.source1_entity_id.astype(str).isin(fit_ids)].copy(), labels_path)
    if labeled["label"].isna().any(): raise ValueError("Missing keyed labels for fusion_fit rows")
    labeled = _derived(labeled)

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    features_by_route = {
        ROUTE_NEURAL_FUSION: config.get("neural_features", DEFAULT_NEURAL_FEATURES),
        ROUTE_TREE_ONLY: config.get("tree_features", DEFAULT_TREE_FEATURES),
    }
    manifests = {}
    for route, features in features_by_route.items():
        validate_feature_allowlist(features)
        rows = labeled[labeled[DECISION_ROUTE].astype(str) == route].copy()
        if rows.empty: continue
        y = pd.to_numeric(rows.label, errors="raise").astype(int).to_numpy()
        if len(np.unique(y)) < 2: raise ValueError(f"Fusion route {route} has only one label class")
        model = LogisticRegression(C=float(config.get("C", 1.0)), max_iter=int(config.get("max_iter", 1000)), class_weight=config.get("class_weight", "balanced"), solver="lbfgs")
        model.fit(_matrix(rows, features), y)
        route_dir = out / route; route_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, route_dir / "fusion_model.joblib")
        manifest = {
            "route": route, "features": list(features), "training_partition": "fusion_fit",
            "training_row_count": len(rows), "positive_count": int(y.sum()), "version": str(config.get("version", "v1")),
            "joined_scores_sha256": sha256_file(joined_scores_path), "labels_sha256": sha256_file(labels_path), "split_sha256": sha256_file(partition_split_path),
        }
        (route_dir / "fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8"); manifests[route] = manifest
    return {"manifests": manifests, "model_dir": str(out)}


def apply_fusion(scores: pd.DataFrame, fusion_model_dir: str | Path) -> pd.DataFrame:
    validate_joined_scores(scores); out = _derived(scores); out["fusion_score"] = np.nan
    root = Path(fusion_model_dir)
    for route in (ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY):
        mask = out[DECISION_ROUTE].astype(str) == route
        if not mask.any(): continue
        manifest_path = root / route / "fusion_manifest.json"; model_path = root / route / "fusion_model.joblib"
        if not manifest_path.exists() or not model_path.exists(): raise FileNotFoundError(f"Missing fusion artifact for route {route}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")); model = joblib.load(model_path)
        x = _matrix(out.loc[mask], manifest["features"]); classes = list(model.classes_)
        out.loc[mask, "fusion_score"] = model.predict_proba(x)[:, classes.index(1)]
    active = out[DECISION_ROUTE].isin([ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY])
    if out.loc[active, "fusion_score"].isna().any(): raise ValueError("Some active-route rows did not receive fusion scores")
    return out
