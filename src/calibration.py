"""Per-route calibration for Member D (§6.2).

Uses only the calibration partition.
Calibrates neural_fusion and tree_only routes separately.
Preferred method: Platt (logistic) calibration.
Isotonic allowed only when sample size is clearly sufficient.

Test-prior correction is DISABLED.
No unlabeled-France prior fitting.

Output: match_probability column + decision_route.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .member_d_contracts import (
    ROUTE_COL_DECISION,
    ROUTE_COL_HAS,
    ROUTE_NEURAL_FUSION,
    ROUTE_TREE_ONLY,
    SCORE_COL_FUSED,
)

try:
    import joblib
except ImportError as _e:
    raise ImportError("joblib is required for calibration") from _e

_MIN_ISOTONIC_ROWS = 500  # isotonic requires at least this many rows per class


# ---------------------------------------------------------------------------
# Platt / logistic calibration wrapper
# ---------------------------------------------------------------------------


def _fit_platt(X: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X.reshape(-1, 1), y)
    return model


def _apply_platt(model: Any, X: np.ndarray) -> np.ndarray:
    if 1 in list(getattr(model, "classes_", [])):
        return model.predict_proba(X.reshape(-1, 1))[:, list(model.classes_).index(1)]
    return np.zeros(len(X))


def _fit_isotonic(X: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(X, y)
    return model


def _apply_isotonic(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# ---------------------------------------------------------------------------
# Core per-route calibration
# ---------------------------------------------------------------------------


def _calibrate_route(
    subset: pd.DataFrame,
    score_col: str,
    route_name: str,
    method: str,
    output_dir: Path,
    version: str,
    git_commit: str | None,
) -> dict[str, Any]:
    """Fit and persist a calibrator for one route."""
    X = pd.to_numeric(subset[score_col], errors="coerce").fillna(0.5).to_numpy()
    y = pd.to_numeric(subset["label"], errors="coerce").fillna(0).astype(int).to_numpy()

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {str(k): int(v) for k, v in zip(unique, counts)}

    if method == "isotonic":
        min_count = int(min(counts)) if len(counts) > 1 else 0
        if min_count < _MIN_ISOTONIC_ROWS:
            print(
                f"[calibration] {route_name}: isotonic requested but min class count "
                f"{min_count} < {_MIN_ISOTONIC_ROWS}; falling back to platt."
            )
            method = "platt"

    if method == "isotonic":
        model = _fit_isotonic(X, y)
        model_type = "isotonic"
    else:
        model = _fit_platt(X, y)
        model_type = "platt"

    route_dir = output_dir / route_name
    route_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, route_dir / "calibrator.joblib")

    # Diagnostics: calibration curve
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    pred_proba = _apply_platt(model, X) if model_type == "platt" else _apply_isotonic(model, X)
    diag_bins = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (X >= lo) & (X < hi)
        if mask.sum() > 0:
            diag_bins.append({
                "bin_low": round(float(lo), 2),
                "bin_high": round(float(hi), 2),
                "count": int(mask.sum()),
                "mean_raw": float(X[mask].mean()),
                "mean_calibrated": float(pred_proba[mask].mean()),
                "fraction_positive": float(y[mask].mean()),
            })

    manifest: dict[str, Any] = {
        "route": route_name,
        "method": model_type,
        "score_col": score_col,
        "training_row_count": len(subset),
        "class_counts": class_counts,
        "version": version,
        "git_commit": git_commit,
        "diagnostics_calibration_bins": diag_bins,
        "test_prior_correction": False,
    }
    (route_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[calibration] {route_name}: {len(subset)} rows, method={model_type}, "
        f"class_counts={class_counts}"
    )
    return manifest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_calibration(
    joined_scores_path: str | Path,
    fusion_model_dir: str | Path,
    partition_split_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit per-route calibrators using only the calibration partition.

    Parameters
    ----------
    joined_scores_path : path to joined_scores.parquet with fusion_score column
    fusion_model_dir : path to output of fit_fusion (for score_col resolution)
    partition_split_path : path to selection_split.tsv
    config : dict with optional keys:
        method      'platt' (default) or 'isotonic'
        score_col   column to calibrate (default 'fusion_score')
        version     str
        git_commit  str | None
    output_dir : where to write calibrators

    Returns
    -------
    dict with route manifests
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = config.get("method", "platt")
    score_col = config.get("score_col", "fusion_score")
    version = str(config.get("version", "v1"))
    git_commit = config.get("git_commit")

    scores = pd.read_parquet(joined_scores_path)

    if "label" not in scores.columns:
        raise ValueError("joined_scores must contain 'label' column for calibration.")
    if score_col not in scores.columns:
        raise ValueError(
            f"Score column '{score_col}' not found. Run fit_fusion first to produce it."
        )

    # Load calibration partition
    split_df = pd.read_csv(partition_split_path, sep="\t", dtype=str, keep_default_na=False)
    cal_ids = set(
        split_df.loc[split_df["member_d_partition"] == "calibration", "source1_entity_id"]
    )
    cal_rows = scores[scores["source1_entity_id"].isin(cal_ids)].copy()
    if cal_rows.empty:
        raise ValueError("No calibration rows found after partition filter.")

    # Split by route
    if ROUTE_COL_HAS in cal_rows.columns:
        neural_mask = pd.to_numeric(cal_rows[ROUTE_COL_HAS], errors="coerce").fillna(0) == 1
    else:
        neural_mask = pd.Series(False, index=cal_rows.index)

    manifests: dict[str, Any] = {}
    for route_name, subset in [
        (ROUTE_NEURAL_FUSION, cal_rows[neural_mask]),
        (ROUTE_TREE_ONLY, cal_rows[~neural_mask]),
    ]:
        if not subset.empty:
            manifests[route_name] = _calibrate_route(
                subset, score_col, route_name, method, output_dir, version, git_commit
            )

    summary = {"calibration_dir": str(output_dir), "manifests": manifests}
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def apply_calibration(
    scores_frame: pd.DataFrame,
    calibrator_dir: str | Path,
    score_col: str = "fusion_score",
) -> pd.DataFrame:
    """Apply fitted route-specific calibrators to produce match_probability.

    Parameters
    ----------
    scores_frame : DataFrame with score_col and decision_route columns
    calibrator_dir : directory written by fit_calibration
    score_col : column containing raw fusion scores

    Returns
    -------
    DataFrame with new column 'match_probability' and preserved 'decision_route'.
    """
    calibrator_dir = Path(calibrator_dir)
    out = scores_frame.copy()
    out[SCORE_COL_FUSED] = float("nan")

    route_col = ROUTE_COL_DECISION if ROUTE_COL_DECISION in out.columns else None

    for route_name, mask_fn in [
        (ROUTE_NEURAL_FUSION, lambda df: (
            pd.to_numeric(df.get(ROUTE_COL_HAS, pd.Series(0, index=df.index)), errors="coerce") == 1
        )),
        (ROUTE_TREE_ONLY, lambda df: (
            pd.to_numeric(df.get(ROUTE_COL_HAS, pd.Series(1, index=df.index)), errors="coerce") == 0
        )),
    ]:
        route_dir = calibrator_dir / route_name
        cal_path = route_dir / "calibrator.joblib"
        manifest_path = route_dir / "calibration_manifest.json"
        if not cal_path.exists():
            continue
        calibrator = joblib.load(cal_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        method = manifest.get("method", "platt")

        subset_mask = mask_fn(out)
        subset = out[subset_mask]
        if subset.empty:
            continue

        X = pd.to_numeric(subset[score_col], errors="coerce").fillna(0.5).to_numpy()
        if method == "isotonic":
            calibrated = _apply_isotonic(calibrator, X)
        else:
            calibrated = _apply_platt(calibrator, X)

        out.loc[subset.index, SCORE_COL_FUSED] = calibrated

    return out
