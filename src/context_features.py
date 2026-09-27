"""Candidate-set context features computed jointly across S2 and S3."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rank(frame: pd.DataFrame, score: str) -> pd.Series:
    """Vectorized replacement for the old per-group Python loop.

    Sorts the whole frame once by (source1_entity_id, score desc,
    candidate_entity_id asc) then uses groupby().cumcount() to derive rank
    entirely at the pandas/C level -- no per-group Python iteration.
    """
    working = frame[["source1_entity_id", "candidate_entity_id"]].copy()
    working["__score"] = pd.to_numeric(frame[score], errors="coerce").fillna(-np.inf)
    ordered = working.sort_values(
        ["source1_entity_id", "__score", "candidate_entity_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    rank = ordered.groupby("source1_entity_id", sort=False).cumcount() + 1
    return rank.reindex(frame.index).astype("int16")


def add_candidate_context_features(frame: pd.DataFrame, score_margin: float = 0.05) -> pd.DataFrame:
    """Add deterministic per-S1 retrieval context while preserving row order."""
    required = {"source1_entity_id", "candidate_entity_id", "candidate_source", "retrieval_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Context features require columns: {missing}")
    if score_margin < 0:
        raise ValueError("score_margin must be non-negative")
    output = frame.copy()
    output["retrieval_score"] = pd.to_numeric(output["retrieval_score"], errors="coerce")
    is_injected = output["retrieval_provenance"].eq("training_injected_positive") if "retrieval_provenance" in output else pd.Series(False, index=output.index)
    bad = output["retrieval_score"].isna() | ~np.isfinite(output["retrieval_score"].fillna(0))
    if (bad & ~is_injected).any():
        raise ValueError("retrieval_score is non-finite on rows that are not training_injected_positive")
    # Injected positives get a neutral placeholder score so groupby/rank logic
    # below does not crash; they are clearly flagged via retrieval_provenance
    # and downstream consumers should treat them as "not naturally retrieved".
    output.loc[bad, "retrieval_score"] = -1.0
    grouped = output.groupby("source1_entity_id", sort=False)
    output["candidate_count"] = grouped["candidate_entity_id"].transform("size").astype("int32")
    output["retrieval_score_rank"] = _rank(output, "retrieval_score")
    output["name_score_rank"] = _rank(output, "name_tfidf_score") if "name_tfidf_score" in output else output["retrieval_score_rank"]
    output["address_score_rank"] = _rank(output, "address_tfidf_score") if "address_tfidf_score" in output else output["retrieval_score_rank"]
    best = grouped["retrieval_score"].transform("max")
    output["best_retrieval_score"] = best.astype("float32")
    output["retrieval_score_gap"] = (best - output["retrieval_score"]).astype("float32")
    second = grouped["retrieval_score"].transform(lambda values: values.nlargest(2).iloc[-1] if len(values) > 1 else values.iloc[0])
    output["best_second_retrieval_score_gap"] = (best - second).clip(lower=0).astype("float32")
    output["near_tie_count"] = grouped["retrieval_score"].transform(lambda values: int((values >= values.max() - score_margin).sum())).astype("int16")
    by_name = pd.to_numeric(output["retrieved_by_name"], errors="coerce") if "retrieved_by_name" in output else pd.Series(0, index=output.index)
    by_address = pd.to_numeric(output["retrieved_by_address"], errors="coerce") if "retrieved_by_address" in output else pd.Series(0, index=output.index)
    output["retrieved_by_both"] = (by_name.fillna(0).eq(1) & by_address.fillna(0).eq(1)).astype("int8")
    output["retrieved_by_name_and_address"] = output["retrieved_by_both"]
    for source, column in (("S2", "is_strongest_s2_candidate"), ("S3", "is_strongest_s3_candidate")):
        subset = output.loc[output["candidate_source"].eq(source)]
        result = pd.Series(0, index=output.index, dtype="int8")
        if not subset.empty:
            ordered = subset.sort_values(
                ["source1_entity_id", "retrieval_score", "candidate_entity_id"],
                ascending=[True, False, True],
                kind="stable",
            )
            is_first = ordered.groupby("source1_entity_id", sort=False).cumcount().eq(0)
            result.loc[ordered.index[is_first]] = 1
        output[column] = result
    output["strongest_s2_candidate"] = output["is_strongest_s2_candidate"]
    output["strongest_s3_candidate"] = output["is_strongest_s3_candidate"]
    return output

def add_competition_features(frame: pd.DataFrame, score_margin: float = 0.05) -> pd.DataFrame:
    """Add per-candidate-record competition features: grouped by the S2/S3 record,
    not the S1 query. Answers: 'of all S1 entities competing to own this candidate
    record, how strong is this particular S1 relative to its rivals?'

    This is the opposite grouping from add_candidate_context_features, which groups
    by source1_entity_id. Both directions matter: a candidate can look like a great
    match for its S1 query while simultaneously being an even better match for a
    different, competing S1 query.
    """
    required = {"source1_entity_id", "candidate_entity_id", "candidate_source", "retrieval_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Competition features require columns: {missing}")
    if score_margin < 0:
        raise ValueError("score_margin must be non-negative")

    output = frame.copy()
    output["retrieval_score"] = pd.to_numeric(output["retrieval_score"], errors="coerce")

    # Rows injected only for training (true matches retrieval never found) have no
    # real retrieval_score by design. They did not participate in the actual
    # retrieval competition, so exclude them from the competition statistics
    # rather than rejecting the whole batch.
    is_injected = output["retrieval_provenance"].eq("training_injected_positive") if "retrieval_provenance" in output else pd.Series(False, index=output.index)
    natural = output["retrieval_score"].notna() & np.isfinite(output["retrieval_score"].fillna(0))
    if (~natural & ~is_injected).any():
        raise ValueError("retrieval_score is non-finite on rows that are not training_injected_positive")

    output["is_injected_positive"] = is_injected.astype("int8")

    # Group by the candidate record itself (source + id), not the S1 query.
    group_key = list(zip(output["candidate_source"], output["candidate_entity_id"]))
    output["_competition_group"] = group_key

    scored = output.loc[natural]
    grouped = scored.groupby("_competition_group", sort=False)

    competing_s1_count = grouped["source1_entity_id"].transform("nunique").astype("int32")
    best = grouped["retrieval_score"].transform("max").astype("float32")
    second = grouped["retrieval_score"].transform(
        lambda values: values.nlargest(2).iloc[-1] if len(values) > 1 else values.iloc[0]
    ).astype("float32")
    near_tie_count = grouped["retrieval_score"].transform(
        lambda values: int((values >= values.max() - score_margin).sum())
    ).astype("int16")
    rank = grouped["retrieval_score"].transform(lambda values: values.rank(method="first", ascending=False)).astype("int16")

    for col, series, fill in (
        ("competing_s1_count", competing_s1_count, 0),
        ("best_competing_s1_score", best, np.nan),
        ("competing_best_second_gap", (best - second).clip(lower=0), 0.0),
        ("competing_near_tie_count", near_tie_count, 0),
        ("competing_s1_rank", rank, 0),
    ):
        output[col] = fill
        output.loc[scored.index, col] = series

    output["competing_s1_score_gap"] = np.nan
    output.loc[scored.index, "competing_s1_score_gap"] = (best - scored["retrieval_score"]).astype("float32")

    output["is_top_competing_s1"] = 0
    output.loc[scored.index, "is_top_competing_s1"] = (output.loc[scored.index, "competing_s1_rank"] == 1).astype("int8")

    output = output.drop(columns=["_competition_group"])
    return output
