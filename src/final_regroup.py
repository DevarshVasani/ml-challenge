"""Regroup source-record ownership decisions back to complete S1 edge sets."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .member_d_contracts import PAIR_KEY, validate_pair_keys


def regroup_selected_edges(ownership_rows: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    if ownership_rows.empty:
        out = pd.DataFrame(columns=PAIR_KEY + ["match_probability", "decision_route", "selected"])
    else:
        required = set(PAIR_KEY) | {"ownership_selected"}
        if not required.issubset(ownership_rows.columns): raise ValueError(f"Ownership rows missing {sorted(required - set(ownership_rows.columns))}")
        selected = ownership_rows[ownership_rows.ownership_selected.astype(bool)].copy()
        if len(selected): validate_pair_keys(selected)
        keep = [c for c in PAIR_KEY + ["match_probability", "decision_route"] if c in selected.columns]
        out = selected[keep].copy(); out["selected"] = 1
        out = out.sort_values(["source1_entity_id", "candidate_source", "candidate_entity_id"], kind="stable").reset_index(drop=True)
    if output_path:
        p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(p, index=False)
    return out
