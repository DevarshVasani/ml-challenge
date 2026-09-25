"""Candidate-set context features computed jointly across S2 and S3."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rank(frame: pd.DataFrame, score: str) -> pd.Series:
    result = pd.Series(index=frame.index, dtype="int64")
    for _, group in frame.groupby("source1_entity_id", sort=False):
        ordered = group.assign(__score=pd.to_numeric(group[score], errors="coerce").fillna(-np.inf))
        ordered = ordered.sort_values(["__score", "candidate_entity_id"], ascending=[False, True], kind="stable")
        result.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return result.astype("int16")


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
    if output["retrieval_score"].isna().any() or not np.isfinite(output["retrieval_score"]).all():
        raise ValueError("retrieval_score must be finite")
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
        output[column] = pd.Series(0, index=output.index, dtype="int8")
        for _, group in output.loc[output["candidate_source"].eq(source)].groupby("source1_entity_id", sort=False):
            ordered = group.sort_values(["retrieval_score", "candidate_entity_id"], ascending=[False, True], kind="stable")
            output.loc[ordered.index[0], column] = 1
        output[column] = output[column].astype("int8")
    output["strongest_s2_candidate"] = output["is_strongest_s2_candidate"]
    output["strongest_s3_candidate"] = output["is_strongest_s3_candidate"]
    return output
