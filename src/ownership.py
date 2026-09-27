"""Gold ownership audit and best-owner selector (Member D, §4).

audit_gold_ownership  – inspect ground truth for S2/S3 records with multiple
                         S1 owners; must pass before ownership can be enabled.
apply_best_owner      – for each complete source-record group choose at most
                         one S1 owner deterministically.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .member_d_contracts import (
    OWNERSHIP_GROUP_KEY,
    PAIR_KEY,
    SCORE_COL_FUSED,
    SCORE_COL_TREE,
)


# ---------------------------------------------------------------------------
# Gold ownership audit
# ---------------------------------------------------------------------------


def audit_gold_ownership(
    ground_truth_path: str | Path,
    *,
    first_n_examples: int = 10,
) -> dict[str, Any]:
    """Inspect ground truth for source-record ownership violations.

    For every gold S2/S3 record (identified by source-qualified key from the
    matched_entity_ids column), collect all S1 IDs that claim it as a match.

    A "violation" occurs when a single source record has more than one S1 owner.

    Ownership may be enabled **only** when violation_count == 0, or when an
    explicit contest rule describes how multi-owner truth should be handled.

    Parameters
    ----------
    ground_truth_path : path to evaluation_truth.tsv
        Expected columns: source1_entity_id, matched_entity_ids (comma-separated)
    first_n_examples : int
        How many violating examples to include in the report.

    Returns
    -------
    dict with keys:
        total_unique_records        int
        records_zero_owners         int  (should be 0 if truth is well-formed)
        records_one_owner           int
        records_multiple_owners     int
        violation_count             int
        first_n_violating_examples  list[dict]
        ownership_enabled           bool   True only when violation_count == 0
    """
    # source_record_id -> set of S1 IDs that claim it
    owners: dict[str, set[str]] = defaultdict(set)

    with Path(ground_truth_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1 = row.get("source1_entity_id", "").strip()
            raw = row.get("matched_entity_ids", "").strip()
            if not s1 or not raw:
                continue
            for cand_id in (x.strip() for x in raw.split(",") if x.strip()):
                owners[cand_id].add(s1)

    total = len(owners)
    records_zero = 0  # should not happen if truth is self-consistent
    records_one = 0
    records_multiple = 0
    violating_examples: list[dict[str, Any]] = []

    for cand_id, s1_set in owners.items():
        n = len(s1_set)
        if n == 0:
            records_zero += 1
        elif n == 1:
            records_one += 1
        else:
            records_multiple += 1
            if len(violating_examples) < first_n_examples:
                violating_examples.append({
                    "candidate_entity_id": cand_id,
                    "owner_s1_ids": sorted(s1_set),
                    "owner_count": n,
                })

    violation_count = records_multiple

    report: dict[str, Any] = {
        "total_unique_records": total,
        "records_zero_owners": records_zero,
        "records_one_owner": records_one,
        "records_multiple_owners": records_multiple,
        "violation_count": violation_count,
        "first_n_violating_examples": violating_examples,
        "ownership_enabled": violation_count == 0,
    }
    return report


# ---------------------------------------------------------------------------
# Best-owner selector
# ---------------------------------------------------------------------------


def apply_best_owner(
    frame: pd.DataFrame,
    probability_col: str = SCORE_COL_FUSED,
    threshold: float = 0.5,
    *,
    ownership_audit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """For each complete (candidate_source, candidate_entity_id) source-record group,
    choose at most one S1 owner deterministically.

    Parameters
    ----------
    frame : DataFrame
        Must contain PAIR_KEY columns, probability_col, and tree_score.
        The frame must represent COMPLETE source-record groups (all shards merged).
    probability_col : str
        Column containing calibrated match_probability.
    threshold : float
        Only S1 owners at or above this probability are considered.
    ownership_audit : dict or None
        If provided, raises ValueError when ownership_enabled is False.

    Returns
    -------
    DataFrame with one row per retained source record (with selected S1 or
    unmatched flag).  Columns include all original columns plus
    ``ownership_selected`` (bool) and ``ownership_selected_s1`` (str or '').

    Algorithm
    ---------
    1. Keep only S1 owners with probability >= threshold.
    2. If none remain for a source record, mark unmatched.
    3. Otherwise choose exactly ONE S1 by:
       a. Higher probability
       b. Then higher tree_score
       c. Then lexicographically smaller source1_entity_id

    NOTE: Ownership is many-to-one (source record -> one S1).
          An S1 may receive any number of source records.
    NOTE: Never constrains an S1 to one match.
    """
    if ownership_audit is not None and not ownership_audit.get("ownership_enabled", False):
        raise ValueError(
            "Ownership audit has not passed (violation_count > 0). "
            "Run audit_gold_ownership and resolve violations before applying ownership."
        )

    required = set(PAIR_KEY) | {probability_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"apply_best_owner: frame is missing columns: {sorted(missing)}")

    has_tree = SCORE_COL_TREE in frame.columns
    prob = pd.to_numeric(frame[probability_col], errors="coerce").fillna(-1.0)
    tree = pd.to_numeric(frame[SCORE_COL_TREE], errors="coerce").fillna(-1.0) if has_tree else pd.Series(0.0, index=frame.index)

    output_rows: list[dict] = []
    for (src, cand_id), grp in frame.groupby(OWNERSHIP_GROUP_KEY, sort=False):
        grp_prob = prob.loc[grp.index]
        candidates = grp.loc[grp_prob >= threshold].copy()
        if candidates.empty:
            # unmatched
            row = grp.iloc[0].to_dict()
            row["ownership_selected"] = False
            row["ownership_selected_s1"] = ""
            output_rows.append(row)
        else:
            cand_prob = prob.loc[candidates.index]
            cand_tree = tree.loc[candidates.index]
            cand_s1 = candidates["source1_entity_id"].astype(str)
            # sort: highest prob, then highest tree, then lex-smallest S1
            order = pd.DataFrame({
                "prob": cand_prob,
                "tree": cand_tree,
                "s1_neg": cand_s1,
            })
            order = order.sort_values(
                ["prob", "tree", "s1_neg"],
                ascending=[False, False, True],
                kind="stable",
            )
            best_idx = order.index[0]
            best_s1 = str(candidates.loc[best_idx, "source1_entity_id"])
            row = candidates.loc[best_idx].to_dict()
            row["ownership_selected"] = True
            row["ownership_selected_s1"] = best_s1
            output_rows.append(row)

    return pd.DataFrame(output_rows).reset_index(drop=True)
