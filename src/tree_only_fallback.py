"""Explicit independently versioned Member D tree-only fallback path."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calibration import apply_calibration, fit_calibration
from .decoder import select_decoder
from .member_d_contracts import DECISION_ROUTE, ROUTE_TREE_ONLY, TREE_SCORE


def run_tree_only_fallback(
    joined_scores_path: str | Path,
    labels_path: str | Path,
    split_path: str | Path,
    truth: dict[str, list[str]],
    selection_s1_ids: list[str],
    calibration_config: dict[str, Any],
    decoder_config: dict[str, Any],
    output_dir: str | Path,
    ownership_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(joined_scores_path)
    tree = scores[scores[DECISION_ROUTE].astype(str) == ROUTE_TREE_ONLY].copy()
    if tree.empty: raise ValueError("Tree-only fallback has no tree_only rows")
    tree["fusion_score"] = pd.to_numeric(tree[TREE_SCORE], errors="raise")
    raw_path = out / "tree_only_raw.parquet"; tree.to_parquet(raw_path, index=False)

    cal_dir = out / "calibration"
    fit_calibration(raw_path, labels_path, split_path, calibration_config, cal_dir)
    calibrated = apply_calibration(tree, cal_dir)
    calibrated_path = out / "calibrated_scores.parquet"; calibrated.to_parquet(calibrated_path, index=False)

    split = pd.read_csv(split_path, sep="\t", dtype=str, keep_default_na=False)
    selection_ids = set(split.loc[split.member_d_partition == "selection", "source1_entity_id"])
    selection_frame = calibrated[calibrated.source1_entity_id.astype(str).isin(selection_ids)].copy()
    result = select_decoder(
        selection_frame,
        truth,
        selection_s1_ids,
        thresholds=decoder_config.get("thresholds"),
        ownership_audit=ownership_audit,
        enable_expected_f05=False,
        output_path=out / "decoder.json",
    )
    (out / "fallback_manifest.json").write_text(json.dumps({
        "version": "fallback_tree_only_v1",
        "route": ROUTE_TREE_ONLY,
        "calibrated_scores": str(calibrated_path),
        "decoder": result,
    }, indent=2) + "\n", encoding="utf-8")
    return result


def apply_tree_only_fallback(
    production_tree_scores: pd.DataFrame,
    calibrator_dir: str | Path,
    decoder_config_path: str | Path,
) -> pd.DataFrame:
    """Apply the frozen tree-only fallback to unlabeled production rows."""
    frame = production_tree_scores.copy()
    frame = frame[frame[DECISION_ROUTE].astype(str) == ROUTE_TREE_ONLY].copy()
    frame["fusion_score"] = pd.to_numeric(frame[TREE_SCORE], errors="raise")
    frame = apply_calibration(frame, calibrator_dir)
    decoder = json.loads(Path(decoder_config_path).read_text(encoding="utf-8"))
    threshold = decoder.get("threshold")
    if decoder.get("decoder_name") not in {"threshold", "threshold_best_owner"} or threshold is None:
        raise ValueError("Fallback requires a frozen threshold-based decoder")
    frame["selected"] = (pd.to_numeric(frame["match_probability"], errors="raise") >= float(threshold)).astype("int8")
    return frame
