"""Fusion model fitting for Member D (§6.1).

Fits a regularized LogisticRegression on fusion_fit partition rows only.
Trains SEPARATE models for neural_fusion and tree_only routes.
Never applies the neural-fusion model to rows with has_neural_score=0.

Feature allowlist is explicit; no auto-discovery of numeric columns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .member_d_contracts import (
    PAIR_KEY,
    ROUTE_COL_DECISION,
    ROUTE_COL_HAS,
    ROUTE_NEURAL_FUSION,
    ROUTE_TREE_ONLY,
    SCORE_COL_NEURAL,
    SCORE_COL_TREE,
    FORBIDDEN_PRODUCTION_COLUMNS,
)

try:
    import joblib
except ImportError as _e:
    raise ImportError("joblib is required for fusion model fitting") from _e


# ---------------------------------------------------------------------------
# Default feature allowlist
# ---------------------------------------------------------------------------

# [Member D Plan 2: Fake-Pattern Features Review]
# The following Member C string features have shown strong diagnostic 
# performance on fake patterns (branch words, legal forms, house numbers).
# They are explicitly preserved here for review when the fusion feature 
# allowlist is finalized. Do NOT automatically include them without
# validating on D's fusion_fit/calibration splits:
# - postcode_match_house_mismatch
# - legal_form_agreement
# - house_number_match
# - name_tokens_extra_in_b
# - name_tokens_missing_from_b


DEFAULT_NEURAL_FEATURES: list[str] = [
    SCORE_COL_TREE,
    SCORE_COL_NEURAL,
    "tree_logit",
    "neural_logit",
    "source_s2_indicator",
    "source_s3_indicator",
    "route_indicator",
    "source_competitor_count",
    "tree_source_rank",
    "tree_gap_to_best",
    "tree_best_second_gap",
    "tree_near_tie_count",
    "is_tree_best_owner",
]

DEFAULT_TREE_FEATURES: list[str] = [
    SCORE_COL_TREE,
    "tree_logit",
    "source_s2_indicator",
    "source_s3_indicator",
    "source_competitor_count",
    "tree_source_rank",
    "tree_gap_to_best",
    "tree_best_second_gap",
    "tree_near_tie_count",
    "is_tree_best_owner",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add logit transforms and indicator columns."""
    out = df.copy()
    if SCORE_COL_TREE in out.columns:
        t = pd.to_numeric(out[SCORE_COL_TREE], errors="coerce").fillna(0.5)
        out["tree_logit"] = _logit(t.to_numpy())
    if SCORE_COL_NEURAL in out.columns:
        n = pd.to_numeric(out[SCORE_COL_NEURAL], errors="coerce")
        mask = n.notna()
        logit_col = np.full(len(out), np.nan)
        logit_col[mask] = _logit(n[mask].to_numpy())
        out["neural_logit"] = logit_col
    if "candidate_source" in out.columns:
        out["source_s2_indicator"] = (out["candidate_source"].astype(str) == "S2").astype("int8")
        out["source_s3_indicator"] = (out["candidate_source"].astype(str) == "S3").astype("int8")
    if ROUTE_COL_HAS in out.columns:
        out["route_indicator"] = pd.to_numeric(out[ROUTE_COL_HAS], errors="coerce").fillna(0)
    return out


def _build_feature_matrix(df: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """Build float matrix for the given feature list.  Missing columns -> zeros."""
    cols = []
    for feat in feature_list:
        if feat in df.columns:
            cols.append(pd.to_numeric(df[feat], errors="coerce").fillna(0.0).to_numpy())
        else:
            cols.append(np.zeros(len(df), dtype="float64"))
    return np.column_stack(cols)


def _sha256_df(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(df.to_csv(index=False).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_fusion(
    joined_scores_path: str | Path,
    partition_split_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit fusion model(s) using only fusion_fit partition rows.

    Parameters
    ----------
    joined_scores_path : path to joined_scores.parquet (output of score_join)
    partition_split_path : path to configs/member_d/selection_split.tsv
    config : dict with optional keys:
        neural_features     list[str]  default DEFAULT_NEURAL_FEATURES
        tree_features       list[str]  default DEFAULT_TREE_FEATURES
        C                   float      LogisticRegression regularisation (default 1.0)
        max_iter            int        default 1000
        solver              str        default 'lbfgs'
        class_weight        str|None   default 'balanced'
        version             str        model version tag
        git_commit          str|None
    output_dir : where to write model artefacts

    Returns
    -------
    dict with keys: model_dir, neural_row_count, tree_row_count, version
    """
    from sklearn.linear_model import LogisticRegression

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    neural_features: list[str] = config.get("neural_features", DEFAULT_NEURAL_FEATURES)
    tree_features: list[str] = config.get("tree_features", DEFAULT_TREE_FEATURES)

    # Validate allowlist: no forbidden columns
    bad = FORBIDDEN_PRODUCTION_COLUMNS & (set(neural_features) | set(tree_features))
    if bad:
        raise ValueError(f"Feature allowlist contains forbidden columns: {sorted(bad)}")

    C = float(config.get("C", 1.0))
    max_iter = int(config.get("max_iter", 1000))
    solver = str(config.get("solver", "lbfgs"))
    class_weight = config.get("class_weight", "balanced")
    version = str(config.get("version", "v1"))

    # ---- load data ----
    scores = pd.read_parquet(joined_scores_path)

    # Validate no forbidden columns
    forbidden_present = FORBIDDEN_PRODUCTION_COLUMNS & set(scores.columns)
    if forbidden_present:
        raise ValueError(
            f"joined_scores contains forbidden columns: {sorted(forbidden_present)}"
        )

    if "label" not in scores.columns:
        raise ValueError(
            "joined_scores must contain a 'label' column for fusion_fit. "
            "Do not include this in production score artifacts."
        )

    # ---- load partition ----
    split_df = pd.read_csv(partition_split_path, sep="\t", dtype=str, keep_default_na=False)
    fit_ids = set(
        split_df.loc[split_df["member_d_partition"] == "fusion_fit", "source1_entity_id"]
    )
    fit_rows = scores[scores["source1_entity_id"].isin(fit_ids)].copy()
    if fit_rows.empty:
        raise ValueError("No fusion_fit rows found after partition filter.")

    # ---- derive features ----
    fit_rows = _add_derived_features(fit_rows)

    # ---- split by route ----
    has_neural_col = ROUTE_COL_HAS if ROUTE_COL_HAS in fit_rows.columns else None
    if has_neural_col:
        neural_mask = pd.to_numeric(fit_rows[has_neural_col], errors="coerce").fillna(0) == 1
    else:
        neural_mask = pd.Series(False, index=fit_rows.index)

    neural_rows = fit_rows[neural_mask]
    tree_rows = fit_rows[~neural_mask]

    manifests: dict[str, Any] = {}

    def _fit_one(subset: pd.DataFrame, features: list[str], route_name: str) -> None:
        if subset.empty:
            print(f"[fusion] No {route_name} rows; skipping.")
            return
        X = _build_feature_matrix(subset, features)
        y = pd.to_numeric(subset["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        model = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver, class_weight=class_weight,
        )
        model.fit(X, y)
        route_dir = output_dir / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, route_dir / "fusion_model.joblib")
        manifest = {
            "route": route_name,
            "features": features,
            "C": C, "max_iter": max_iter, "solver": solver, "class_weight": class_weight,
            "version": version,
            "training_row_count": len(subset),
            "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "git_commit": config.get("git_commit"),
        }
        (route_dir / "fusion_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        manifests[route_name] = manifest
        print(f"[fusion] {route_name}: {len(subset)} rows fitted -> {route_dir}")

    _fit_one(neural_rows, neural_features, ROUTE_NEURAL_FUSION)
    _fit_one(tree_rows, tree_features, ROUTE_TREE_ONLY)

    result = {
        "model_dir": str(output_dir),
        "neural_row_count": len(neural_rows),
        "tree_row_count": len(tree_rows),
        "version": version,
        "manifests": manifests,
    }
    (output_dir / "fusion_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def apply_fusion(
    scores: pd.DataFrame,
    fusion_model_dir: str | Path,
) -> pd.DataFrame:
    """Apply fitted fusion models to produce raw (uncalibrated) fusion scores.

    Applies the neural_fusion model to neural-scored rows and the tree_only
    model to tree-only rows.  Never applies the neural model to tree-only rows.

    Returns frame with new column 'fusion_score' (prob of match, pre-calibration).
    """
    fusion_model_dir = Path(fusion_model_dir)
    out = scores.copy()
    out["fusion_score"] = float("nan")

    has_neural_col = ROUTE_COL_HAS if ROUTE_COL_HAS in out.columns else None

    for route_name, mask_fn in [
        (ROUTE_NEURAL_FUSION, lambda df: (
            pd.to_numeric(df.get(ROUTE_COL_HAS, pd.Series(0, index=df.index)), errors="coerce") == 1
        )),
        (ROUTE_TREE_ONLY, lambda df: (
            pd.to_numeric(df.get(ROUTE_COL_HAS, pd.Series(1, index=df.index)), errors="coerce") == 0
        )),
    ]:
        route_dir = fusion_model_dir / route_name
        model_path = route_dir / "fusion_model.joblib"
        manifest_path = route_dir / "fusion_manifest.json"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = manifest["features"]

        subset = out[mask_fn(out)]
        if subset.empty:
            continue

        subset_derived = _add_derived_features(subset)
        X = _build_feature_matrix(subset_derived, features)
        if 1 in list(getattr(model, "classes_", [])):
            proba = model.predict_proba(X)[:, list(model.classes_).index(1)]
        else:
            proba = np.zeros(len(X))
        out.loc[subset.index, "fusion_score"] = proba

    return out
