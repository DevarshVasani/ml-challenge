"""Leakage-safe route-specific Platt/isotonic calibration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .member_d_contracts import DECISION_ROUTE, MATCH_PROBABILITY, PAIR_KEY, ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY, validate_joined_scores
from .neural_contracts import sha256_file


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path); return pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _join_labels(scores: pd.DataFrame, labels_path: str | Path) -> pd.DataFrame:
    labels = _read(labels_path)
    if not set(PAIR_KEY + ["label"]).issubset(labels.columns): raise ValueError("Calibration labels require PAIR_KEY + label")
    if labels.duplicated(PAIR_KEY).any(): raise ValueError("Duplicate calibration labels")
    return scores.merge(labels[PAIR_KEY + ["label"]], on=PAIR_KEY, how="left", validate="1:1")


def fit_calibration(
    fused_scores_path: str | Path,
    labels_path: str | Path,
    partition_split_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    scores = _read(fused_scores_path)
    # validate the production columns by temporarily ignoring fusion_score, which is allowed.
    validate_joined_scores(scores)
    if "fusion_score" not in scores.columns: raise ValueError("Run fusion before calibration")
    split = pd.read_csv(partition_split_path, sep="\t", dtype=str, keep_default_na=False)
    ids = set(split.loc[split.member_d_partition == "calibration", "source1_entity_id"])
    rows = _join_labels(scores[scores.source1_entity_id.astype(str).isin(ids)].copy(), labels_path)
    if rows.label.isna().any(): raise ValueError("Missing calibration labels")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for route in (ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY):
        part = rows[rows[DECISION_ROUTE].astype(str) == route].copy()
        if part.empty: continue
        x = pd.to_numeric(part.fusion_score, errors="raise").to_numpy(float); y = pd.to_numeric(part.label, errors="raise").astype(int).to_numpy()
        if len(np.unique(y)) < 2: raise ValueError(f"Calibration route {route} has one label class")
        method = str(config.get("method", "platt"))
        if method == "isotonic" and min((y == 0).sum(), (y == 1).sum()) >= int(config.get("min_isotonic_per_class", 500)):
            model = IsotonicRegression(out_of_bounds="clip").fit(x, y); actual = "isotonic"
        else:
            model = LogisticRegression(C=1.0, max_iter=1000).fit(x.reshape(-1, 1), y); actual = "platt"
        route_dir = out / route; route_dir.mkdir(parents=True, exist_ok=True); joblib.dump(model, route_dir / "calibrator.joblib")
        manifest = {
            "route": route, "method": actual, "training_partition": "calibration", "training_row_count": len(part),
            "positive_count": int(y.sum()), "version": str(config.get("version", "v1")), "test_prior_correction": False,
            "fused_scores_sha256": sha256_file(fused_scores_path), "labels_sha256": sha256_file(labels_path), "split_sha256": sha256_file(partition_split_path),
        }
        (route_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8"); manifests[route] = manifest
    return {"calibration_dir": str(out), "manifests": manifests}


def apply_calibration(scores: pd.DataFrame, calibrator_dir: str | Path) -> pd.DataFrame:
    if "fusion_score" not in scores.columns: raise ValueError("Missing fusion_score")
    out = scores.copy(); out[MATCH_PROBABILITY] = np.nan; root = Path(calibrator_dir)
    for route in (ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY):
        mask = out[DECISION_ROUTE].astype(str) == route
        if not mask.any(): continue
        manifest = json.loads((root / route / "calibration_manifest.json").read_text(encoding="utf-8")); model = joblib.load(root / route / "calibrator.joblib")
        x = pd.to_numeric(out.loc[mask, "fusion_score"], errors="raise").to_numpy(float)
        if manifest["method"] == "isotonic": p = model.predict(x)
        else: p = model.predict_proba(x.reshape(-1, 1))[:, list(model.classes_).index(1)]
        out.loc[mask, MATCH_PROBABILITY] = p
    active = out[DECISION_ROUTE].isin([ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY])
    if out.loc[active, MATCH_PROBABILITY].isna().any(): raise ValueError("Active rows missing calibrated probability")
    return out
