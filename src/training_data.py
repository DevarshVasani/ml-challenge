"""Deterministic, endpoint-safe training row selection."""

from __future__ import annotations

import pandas as pd


def select_hard_training_rows(frame: pd.DataFrame, negatives_per_s1: int = 10, score_column: str = "retrieval_score") -> pd.DataFrame:
    if negatives_per_s1 < 0:
        raise ValueError("negatives_per_s1 must be non-negative")
    required = {"source1_entity_id", "candidate_entity_id", "label", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training rows are missing columns: {missing}")
    work = frame.copy()
    labels = pd.to_numeric(work["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("label must contain only 0 or 1")
    work["__label"] = labels.astype(int)
    work["__score"] = pd.to_numeric(work[score_column], errors="coerce").fillna(float("-inf"))
    positives = work.loc[work["__label"].eq(1)]
    negatives = work.loc[work["__label"].eq(0)].sort_values(
        ["source1_entity_id", "__score", "candidate_entity_id"], ascending=[True, False, True], kind="stable"
    ).groupby("source1_entity_id", sort=False).head(negatives_per_s1)
    selected = pd.concat([positives, negatives], ignore_index=True)
    return selected.sort_values(
        ["source1_entity_id", "__label", "__score", "candidate_entity_id"], ascending=[True, False, False, True], kind="stable"
    ).drop(columns=["__label", "__score"]).reset_index(drop=True)


def training_rows_for_fold(frame: pd.DataFrame, validation_fold: int, negatives_per_s1: int = 10) -> pd.DataFrame:
    missing = sorted({"source1_fold", "candidate_fold"} - set(frame.columns))
    if missing:
        raise ValueError(f"Fold-aware training rows are missing columns: {missing}")
    source = pd.to_numeric(frame["source1_fold"], errors="raise")
    candidate = pd.to_numeric(frame["candidate_fold"], errors="raise")
    eligible = frame.loc[(source != validation_fold) & (candidate != validation_fold)].copy()
    return select_hard_training_rows(eligible, negatives_per_s1=negatives_per_s1)

