"""Gold ownership audit and deterministic best-owner selection."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .member_d_contracts import MATCH_PROBABILITY, PAIR_KEY, SOURCE_GROUP_KEY, TREE_SCORE, derive_candidate_source, validate_pair_keys


def audit_gold_ownership(ground_truth_path: str | Path, *, first_n_examples: int = 10) -> dict[str, Any]:
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    with Path(ground_truth_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (row.get("source1_entity_id") or "").strip()
            for cid in [x.strip() for x in (row.get("matched_entity_ids") or "").split(",") if x.strip()]:
                owners[(derive_candidate_source(cid), cid)].add(s1)
    violations = [(key, sorted(v)) for key, v in owners.items() if len(v) > 1]
    return {
        "source_record_count": len(owners),
        "records_with_one_owner": sum(len(v) == 1 for v in owners.values()),
        "records_with_multiple_owners": len(violations),
        "multi_owner_violation_count": len(violations),
        "example_violations": [
            {"candidate_source": k[0], "candidate_entity_id": k[1], "owner_s1_ids": v}
            for k, v in violations[:first_n_examples]
        ],
        "ownership_enabled": len(violations) == 0,
    }


def apply_best_owner(
    frame: pd.DataFrame,
    probability_col: str = MATCH_PROBABILITY,
    threshold: float = 0.5,
    *,
    ownership_audit: dict[str, Any] | None = None,
    require_source_complete_flag: bool = True,
) -> pd.DataFrame:
    validate_pair_keys(frame)
    if probability_col not in frame.columns or TREE_SCORE not in frame.columns:
        raise ValueError(f"Ownership requires {probability_col} and {TREE_SCORE}")
    if ownership_audit is not None and not ownership_audit.get("ownership_enabled", False):
        raise ValueError("Ownership cannot be enabled: gold audit did not pass")
    if require_source_complete_flag:
        if "source_group_complete" not in frame.columns or not frame["source_group_complete"].astype(bool).all():
            raise ValueError("Ownership requires source_group_complete=true on every row")

    rows = []
    for key, group in frame.groupby(SOURCE_GROUP_KEY, sort=False):
        probs = pd.to_numeric(group[probability_col], errors="raise")
        survivors = group.loc[probs >= threshold].copy()
        if survivors.empty:
            row = group.iloc[0].to_dict(); row["ownership_selected"] = False; row["chosen_owner"] = ""; rows.append(row); continue
        survivors["__p"] = pd.to_numeric(survivors[probability_col], errors="raise")
        survivors["__t"] = pd.to_numeric(survivors[TREE_SCORE], errors="raise")
        survivors["__s"] = survivors["source1_entity_id"].astype(str)
        best = survivors.sort_values(["__p", "__t", "__s"], ascending=[False, False, True], kind="stable").iloc[0]
        row = best.drop(labels=["__p", "__t", "__s"], errors="ignore").to_dict(); row["ownership_selected"] = True; row["chosen_owner"] = str(best["source1_entity_id"]); rows.append(row)
    return pd.DataFrame(rows)
