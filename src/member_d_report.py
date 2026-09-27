"""Member D comparison report writer with Plan-3 provenance eligibility gating."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import pandas as pd

COLUMNS = [
    "candidate_version", "tree_version", "neural_version", "fusion_version", "calibration_version", "decoder_version",
    "tree_training_git_sha", "tree_feature_schema_version", "tree_feature_columns_sha256", "feature_contract_valid",
    "macro_f05", "singleton_accuracy", "pair_precision", "pair_recall", "raw_oracle", "post_gate_oracle", "candidate_pair_count",
    "US_macro_f05", "India_macro_f05", "S2_pair_precision", "S2_pair_recall", "S3_pair_precision", "S3_pair_recall",
    "runtime_seconds", "peak_ram_gb", "artifact_path", "eligible_for_selection",
]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {c: row.get(c, "N/A") for c in COLUMNS}
    valid = out["feature_contract_valid"] is True or str(out["feature_contract_valid"]).lower() == "true"
    out["eligible_for_selection"] = bool(valid and not row.get("invalid_for_selection", False))
    return out


def write_comparison_report(rows: Sequence[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); normalized = [normalize_row(r) for r in rows]
    tsv = out / "comparison.tsv"; js = out / "comparison.json"
    pd.DataFrame(normalized, columns=COLUMNS).to_csv(tsv, sep="\t", index=False)
    js.write_text(json.dumps(normalized, indent=2, default=str) + "\n", encoding="utf-8")
    return {"tsv": str(tsv), "json": str(js)}
