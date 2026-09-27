"""Explicit independently versioned Member D tree-only fallback path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calibration import apply_calibration, fit_calibration
from .decoder import select_decoder
from .member_d_contracts import (
    DECISION_ROUTE,
    HAS_NEURAL_SCORE,
    NEURAL_LOGIT,
    NEURAL_PROBABILITY,
    NEURAL_REQUESTED,
    ROUTE_DISCARD,
    ROUTE_TREE_ONLY,
    TREE_SCORE,
    validate_joined_scores,
)
from .member_d_manifest import MemberDManifest, write_manifest
from .neural_contracts import sha256_file
from .ownership import apply_best_owner


def _force_tree_only(scores: pd.DataFrame) -> pd.DataFrame:
    """Use tree evidence for every non-discard decision pair.

    A fallback must not keep only rows that happened to be routed tree_only
    in the primary system; it must remain usable when neural inference is
    unavailable for the entire decision set.
    """
    frame = scores.copy()
    if DECISION_ROUTE in frame.columns:
        frame = frame[
            frame[DECISION_ROUTE].astype(str) != ROUTE_DISCARD
        ].copy()

    frame[DECISION_ROUTE] = ROUTE_TREE_ONLY
    frame[NEURAL_REQUESTED] = 0
    frame = frame.drop(
        columns=[NEURAL_LOGIT, NEURAL_PROBABILITY],
        errors="ignore",
    )
    frame[HAS_NEURAL_SCORE] = 0
    validate_joined_scores(frame)
    return frame


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
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(joined_scores_path)
    tree = _force_tree_only(scores)
    if tree.empty:
        raise ValueError("Tree-only fallback has no decision rows")

    tree["fusion_score"] = pd.to_numeric(
        tree[TREE_SCORE], errors="raise"
    )
    raw_path = out / "tree_only_raw.parquet"
    tree.to_parquet(raw_path, index=False)

    calibration_dir = out / "calibration"
    fit_calibration(
        raw_path,
        labels_path,
        split_path,
        calibration_config,
        calibration_dir,
    )
    calibrated = apply_calibration(tree, calibration_dir)
    calibrated_path = out / "calibrated_scores.parquet"
    calibrated.to_parquet(calibrated_path, index=False)

    split = pd.read_csv(
        split_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    selection_ids = set(
        split.loc[
            split["member_d_partition"] == "selection",
            "source1_entity_id",
        ].astype(str)
    )
    selection_frame = calibrated[
        calibrated["source1_entity_id"]
        .astype(str)
        .isin(selection_ids)
    ].copy()

    result = select_decoder(
        selection_frame,
        truth,
        selection_s1_ids,
        thresholds=decoder_config.get("thresholds"),
        ownership_audit=ownership_audit,
        enable_expected_f05=False,
        output_path=out / "decoder.json",
    )

    manifest = MemberDManifest(
        artifact_kind="tree_only_fallback",
        version=str(
            calibration_config.get(
                "fallback_version",
                "fallback_tree_only_v1",
            )
        ),
        completion_state="complete",
        row_count=len(calibrated),
        calibration_version=str(
            calibration_config.get("version", "v1")
        ),
        decoder_version=result["decoder_name"],
        files={
            raw_path.name: sha256_file(raw_path),
            calibrated_path.name: sha256_file(calibrated_path),
            "decoder.json": sha256_file(out / "decoder.json"),
        },
        metadata={
            "route": ROUTE_TREE_ONLY,
            "selection_macro_f05": result["macro_f05"],
            "threshold": result["threshold"],
            "joined_scores_sha256": sha256_file(
                joined_scores_path
            ),
            "labels_sha256": sha256_file(labels_path),
            "split_sha256": sha256_file(split_path),
        },
    )
    write_manifest(out / "fallback_manifest.json", manifest)

    return {
        **result,
        "manifest": str(out / "fallback_manifest.json"),
        "calibrated_scores": str(calibrated_path),
    }


def apply_tree_only_fallback(
    production_scores: pd.DataFrame,
    calibrator_dir: str | Path,
    decoder_config_path: str | Path,
    *,
    ownership_audit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    frame = _force_tree_only(production_scores)
    frame["fusion_score"] = pd.to_numeric(
        frame[TREE_SCORE], errors="raise"
    )
    frame = apply_calibration(frame, calibrator_dir)

    decoder = json.loads(
        Path(decoder_config_path).read_text(encoding="utf-8")
    )
    threshold = decoder.get("threshold")
    decoder_name = decoder.get("decoder_name")
    if (
        decoder_name not in {"threshold", "threshold_best_owner"}
        or threshold is None
    ):
        raise ValueError(
            "Fallback requires a frozen threshold-based decoder"
        )

    if decoder_name == "threshold":
        frame["selected"] = (
            pd.to_numeric(
                frame["match_probability"], errors="raise"
            )
            >= float(threshold)
        ).astype("int8")
        return frame

    if ownership_audit is None:
        raise ValueError(
            "threshold_best_owner fallback requires ownership_audit"
        )
    owned = apply_best_owner(
        frame,
        threshold=float(threshold),
        ownership_audit=ownership_audit,
    )
    owned["selected"] = owned["ownership_selected"].astype("int8")
    return owned
