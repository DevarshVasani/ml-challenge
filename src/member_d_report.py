"""Member D comparison report writer with Plan-2/3 diagnostics and eligibility gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

COLUMNS = [
    "candidate_version",
    "tree_version",
    "neural_version",
    "fusion_version",
    "calibration_version",
    "decoder_version",
    "tree_training_git_sha",
    "tree_feature_schema_version",
    "tree_feature_columns_sha256",
    "feature_contract_valid",
    "macro_f05",
    "singleton_accuracy",
    "pair_precision",
    "pair_recall",
    "raw_oracle",
    "post_gate_oracle",
    "candidate_pair_count",
    "US_macro_f05",
    "India_macro_f05",
    "S2_pair_precision",
    "S2_pair_recall",
    "S3_pair_precision",
    "S3_pair_recall",
    # Plan 2 guard rails.
    "fake_pattern_negative_count",
    "fake_pattern_catch_rate",
    "fake_pattern_false_accept_count",
    "legit_variation_positive_count",
    "legit_variation_keep_rate",
    "legit_variation_false_reject_count",
    "runtime_seconds",
    "peak_ram_gb",
    "artifact_path",
    "eligible_for_selection",
]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {column: row.get(column, "N/A") for column in COLUMNS}
    valid = (
        out["feature_contract_valid"] is True
        or str(out["feature_contract_valid"]).lower() == "true"
    )
    out["eligible_for_selection"] = bool(
        valid and not row.get("invalid_for_selection", False)
    )
    return out


def write_comparison_report(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_row(row) for row in rows]

    tsv_path = out / "comparison.tsv"
    json_path = out / "comparison.json"
    pd.DataFrame(
        normalized, columns=COLUMNS
    ).to_csv(tsv_path, sep="\t", index=False)
    json_path.write_text(
        json.dumps(normalized, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "tsv": str(tsv_path),
        "json": str(json_path),
    }
