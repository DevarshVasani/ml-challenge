"""Fit Member D's real threshold-selection stack on source-complete C features.

This stage is selection-eligible for the frozen historical threshold split.
It is NOT the final production/test run.

Partition discipline:
- fusion_fit: fit route-specific fusion
- calibration: fit route-specific calibration
- selection: choose decoder/threshold

Ownership is included only if a supplied ownership audit explicitly declares a
full-training scope and ownership_enabled=true. The existing
evaluation_truth_only audit is insufficient for a production ownership freeze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calibration import apply_calibration, fit_calibration
from .decoder import select_decoder
from .fusion import apply_fusion, fit_fusion
from .neural_contracts import sha256_file


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _truth_for_ids(
    truth_path: str | Path,
    wanted_ids: list[str],
) -> dict[str, list[str]]:
    wanted = set(map(str, wanted_ids))
    frame = pd.read_csv(
        truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    result = {sid: [] for sid in wanted}
    for row in frame.itertuples(index=False):
        sid = str(row.source1_entity_id)
        if sid not in wanted:
            continue
        result[sid] = [
            value.strip()
            for value in str(row.matched_entity_ids or "").split(",")
            if value.strip()
        ]
    return result


def _full_training_ownership_audit(
    path: str | Path | None,
) -> tuple[dict[str, Any] | None, str]:
    if not path:
        return None, "No full-training ownership audit supplied"
    p = Path(path)
    if not p.exists():
        return None, f"Ownership audit not found: {p}"
    audit = _load_json(p)
    scope = str(audit.get("audit_scope", "")).strip().lower()
    accepted_scopes = {
        "full_training",
        "full_training_truth",
        "training_truth_full",
    }
    if scope not in accepted_scopes:
        return (
            None,
            "Ownership audit scope is not full-training; "
            f"received {scope or '<missing>'}",
        )
    if audit.get("ownership_enabled") is not True:
        return None, "Full-training ownership audit did not enable ownership"
    if int(audit.get("multi_owner_violation_count", 1)) != 0:
        return None, "Full-training ownership audit has multi-owner violations"
    return audit, "Full-training ownership audit accepted"


def run_threshold_selection(
    joined_scores_path: str | Path,
    labels_path: str | Path,
    split_path: str | Path,
    truth_path: str | Path,
    fusion_config_path: str | Path,
    calibration_config_path: str | Path,
    decoder_config_path: str | Path,
    output_dir: str | Path,
    *,
    ownership_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fusion_cfg = _load_json(fusion_config_path)
    calibration_cfg = _load_json(calibration_config_path)
    decoder_cfg = _load_json(decoder_config_path)

    fusion_dir = out / "fusion"
    calibration_dir = out / "calibration"

    fusion_result = fit_fusion(
        joined_scores_path,
        labels_path,
        split_path,
        fusion_cfg,
        fusion_dir,
    )

    joined = pd.read_parquet(joined_scores_path)
    fused = apply_fusion(joined, fusion_dir)
    fused_path = out / "fused_scores.parquet"
    fused.to_parquet(fused_path, index=False)

    calibration_result = fit_calibration(
        fused_path,
        labels_path,
        split_path,
        calibration_cfg,
        calibration_dir,
    )

    calibrated = apply_calibration(fused, calibration_dir)
    calibrated_path = out / "calibrated_scores.parquet"
    calibrated.to_parquet(calibrated_path, index=False)

    split = pd.read_csv(
        split_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    selection_ids = split.loc[
        split["member_d_partition"] == "selection",
        "source1_entity_id",
    ].astype(str).tolist()
    if not selection_ids:
        raise ValueError("Member D selection partition is empty")

    selection = calibrated[
        calibrated["source1_entity_id"].astype(str).isin(selection_ids)
    ].copy()
    truth = _truth_for_ids(truth_path, selection_ids)

    ownership_audit, ownership_status = _full_training_ownership_audit(
        ownership_audit_path
    )

    decoder_path = out / "selection_decoder.json"
    decoder_result = select_decoder(
        selection,
        truth,
        selection_ids,
        thresholds=list(map(float, decoder_cfg["thresholds"])),
        ownership_audit=ownership_audit,
        enable_expected_f05=bool(
            decoder_cfg.get("enable_expected_f05", False)
        ),
        output_path=decoder_path,
    )
    decoder_result.update(
        {
            "selection_eligible": True,
            "production_ready": False,
            "selection_scope": "historical_threshold",
            "production_blocker": (
                "Await B/C/A production candidate/tree/route/neural artifacts "
                "before final production assembly."
            ),
            "ownership_production_eligible": ownership_audit is not None,
            "ownership_status": ownership_status,
        }
    )
    decoder_path.write_text(
        json.dumps(decoder_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "artifact_kind": "member_d_source_complete_threshold_selection",
        "version": "source_complete_threshold_selection_v1",
        "selection_eligible": True,
        "production_ready": False,
        "selection_scope": "historical_threshold",
        "joined_scores_sha256": sha256_file(joined_scores_path),
        "labels_sha256": sha256_file(labels_path),
        "split_sha256": sha256_file(split_path),
        "truth_sha256": sha256_file(truth_path),
        "fusion_config_sha256": sha256_file(fusion_config_path),
        "calibration_config_sha256": sha256_file(
            calibration_config_path
        ),
        "decoder_config_sha256": sha256_file(decoder_config_path),
        "fusion": fusion_result,
        "calibration": calibration_result,
        "decoder": decoder_result,
        "outputs": {
            "fused_scores": str(fused_path),
            "calibrated_scores": str(calibrated_path),
            "selection_decoder": str(decoder_path),
        },
    }
    summary_path = out / "selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["selection_summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined-scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--calibration-config", required=True)
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ownership-audit")
    args = parser.parse_args()

    result = run_threshold_selection(
        args.joined_scores,
        args.labels,
        args.split,
        args.truth,
        args.fusion_config,
        args.calibration_config,
        args.decoder_config,
        args.output_dir,
        ownership_audit_path=args.ownership_audit,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
