"""Member D contracts shared by evaluation, fusion, calibration and export.

The canonical pair key is source-qualified and must be used for every join:
(candidate_source, candidate_entity_id, source1_entity_id).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

PAIR_KEY = ["candidate_source", "candidate_entity_id", "source1_entity_id"]
SOURCE_GROUP_KEY = ["candidate_source", "candidate_entity_id"]

TREE_SCORE = "tree_score"
NEURAL_LOGIT = "neural_logit"
NEURAL_PROBABILITY = "neural_probability"
MATCH_PROBABILITY = "match_probability"

NEURAL_REQUESTED = "neural_requested"
HAS_NEURAL_SCORE = "has_neural_score"
DECISION_ROUTE = "decision_route"

ROUTE_NEURAL_FUSION = "neural_fusion"
ROUTE_TREE_ONLY = "tree_only"
ROUTE_DISCARD = "discard"
VALID_DECISION_ROUTES = frozenset({ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY, ROUTE_DISCARD})

VALID_SOURCE_PREFIXES = {"S2": "S2-", "S3": "S3-"}

FORBIDDEN_TRAINING_COLUMNS = frozenset({
    "label",
    "fold",
    "source1_fold",
    "candidate_fold",
    "member_d_partition",
    "positive_injected_for_training",
    "is_injected_positive",
})
FORBIDDEN_PRODUCTION_COLUMNS = FORBIDDEN_TRAINING_COLUMNS
FORBIDDEN_PREFIXES = ("truth_", "gold_", "target_", "ground_truth_")

TREE_PROVENANCE_FIELDS = (
    "tree_model_version",
    "training_git_sha",
    "feature_columns_sha256",
    "feature_schema_version",
)


def feature_columns_sha256(columns: Sequence[str]) -> str:
    payload = json.dumps(list(columns), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_candidate_source(entity_id: str) -> str:
    value = str(entity_id).strip()
    for source, prefix in VALID_SOURCE_PREFIXES.items():
        if value.startswith(prefix):
            return source
    raise ValueError(f"Cannot derive candidate_source from {entity_id!r}; expected S2-* or S3-*.")


def validate_pair_keys(frame: pd.DataFrame) -> None:
    missing = [c for c in PAIR_KEY if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing pair-key columns: {missing}")
    for col in PAIR_KEY:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} contains null/empty IDs")
    bad_source = ~frame["candidate_source"].astype(str).isin(VALID_SOURCE_PREFIXES)
    if bad_source.any():
        raise ValueError(f"Invalid candidate_source values: {sorted(frame.loc[bad_source, 'candidate_source'].astype(str).unique())}")
    expected = frame["candidate_source"].astype(str).map(VALID_SOURCE_PREFIXES)
    ids = frame["candidate_entity_id"].astype(str)
    mismatch = ~pd.Series([eid.startswith(pref) for eid, pref in zip(ids, expected)], index=frame.index)
    if mismatch.any():
        sample = frame.loc[mismatch, ["candidate_source", "candidate_entity_id"]].head(5).to_dict("records")
        raise ValueError(f"candidate_source/id prefix mismatch: {sample}")
    if frame.duplicated(PAIR_KEY).any():
        raise ValueError("Duplicate canonical pair key detected")


def _finite(series: pd.Series, name: str, *, probability: bool, nullable: bool = False) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    if not nullable and numeric.isna().any():
        raise ValueError(f"{name} contains missing/non-numeric values")
    values = numeric.dropna()
    if not values.map(math.isfinite).all():
        raise ValueError(f"{name} contains non-finite values")
    if probability and ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} must be in [0, 1]")


def validate_tree_scores(frame: pd.DataFrame) -> None:
    validate_pair_keys(frame)
    if TREE_SCORE not in frame.columns:
        raise ValueError(f"Missing {TREE_SCORE}")
    _finite(frame[TREE_SCORE], TREE_SCORE, probability=True)


def validate_neural_scores(frame: pd.DataFrame) -> None:
    validate_pair_keys(frame)
    if NEURAL_LOGIT not in frame.columns and NEURAL_PROBABILITY not in frame.columns:
        raise ValueError(f"Neural artifact must contain {NEURAL_LOGIT} or {NEURAL_PROBABILITY}")
    if NEURAL_LOGIT in frame.columns:
        _finite(frame[NEURAL_LOGIT], NEURAL_LOGIT, probability=False, nullable=True)
    if NEURAL_PROBABILITY in frame.columns:
        _finite(frame[NEURAL_PROBABILITY], NEURAL_PROBABILITY, probability=True, nullable=True)


def validate_feature_allowlist(feature_columns: Iterable[str]) -> None:
    features = [str(x) for x in feature_columns]
    forbidden = sorted(set(features) & FORBIDDEN_TRAINING_COLUMNS)
    prefixed = sorted(f for f in features if f.startswith(FORBIDDEN_PREFIXES))
    if forbidden or prefixed:
        raise ValueError(f"Feature allowlist contains forbidden/truth-derived fields: {sorted(set(forbidden + prefixed))}")


def validate_tree_provenance(manifest: Mapping[str, Any], feature_columns: Sequence[str] | None = None) -> None:
    missing = [f for f in TREE_PROVENANCE_FIELDS if not manifest.get(f)]
    if missing:
        raise ValueError(f"Tree provenance is incomplete: {missing}")
    actual_features = list(feature_columns if feature_columns is not None else manifest.get("feature_columns", []))
    if not actual_features:
        raise ValueError("Tree provenance must provide the actual feature list")
    validate_feature_allowlist(actual_features)
    expected_hash = feature_columns_sha256(actual_features)
    if str(manifest["feature_columns_sha256"]) != expected_hash:
        raise ValueError("Tree feature_columns_sha256 does not match actual feature list")
    if manifest.get("feature_contract_valid") is False:
        raise ValueError("Tree artifact is marked feature_contract_valid=false")
    if manifest.get("invalid_for_selection"):
        raise ValueError(f"Tree artifact invalid for selection: {manifest.get('invalid_reason', 'unspecified')}")


def validate_joined_scores(frame: pd.DataFrame) -> None:
    validate_pair_keys(frame)
    forbidden = sorted(set(frame.columns) & FORBIDDEN_PRODUCTION_COLUMNS)
    if forbidden:
        raise ValueError(f"Production joined-score artifact contains forbidden fields: {forbidden}")
    if TREE_SCORE not in frame.columns:
        raise ValueError(f"Missing {TREE_SCORE}")
    _finite(frame[TREE_SCORE], TREE_SCORE, probability=True)

    if NEURAL_REQUESTED not in frame.columns or HAS_NEURAL_SCORE not in frame.columns or DECISION_ROUTE not in frame.columns:
        raise ValueError("Joined scores require neural_requested, has_neural_score and decision_route")
    requested = pd.to_numeric(frame[NEURAL_REQUESTED], errors="raise").astype(int)
    has = pd.to_numeric(frame[HAS_NEURAL_SCORE], errors="raise").astype(int)
    if (~requested.isin([0, 1])).any() or (~has.isin([0, 1])).any():
        raise ValueError("neural_requested/has_neural_score must be 0/1")

    neural_present = pd.Series(False, index=frame.index)
    if NEURAL_LOGIT in frame.columns:
        _finite(frame[NEURAL_LOGIT], NEURAL_LOGIT, probability=False, nullable=True)
        neural_present |= pd.to_numeric(frame[NEURAL_LOGIT], errors="coerce").notna()
    if NEURAL_PROBABILITY in frame.columns:
        _finite(frame[NEURAL_PROBABILITY], NEURAL_PROBABILITY, probability=True, nullable=True)
        neural_present |= pd.to_numeric(frame[NEURAL_PROBABILITY], errors="coerce").notna()
    if ((requested == 1) & ~neural_present).any():
        raise ValueError("neural_requested=1 with missing neural score")
    if (has.astype(bool) != neural_present).any():
        raise ValueError("has_neural_score disagrees with neural score presence")

    routes = frame[DECISION_ROUTE].astype(str)
    invalid_routes = sorted(set(routes) - VALID_DECISION_ROUTES)
    if invalid_routes:
        raise ValueError(f"Invalid decision routes: {invalid_routes}")
    if ((routes == ROUTE_NEURAL_FUSION) & (has != 1)).any():
        raise ValueError("neural_fusion rows require a neural score")
    if ((routes == ROUTE_TREE_ONLY) & (has != 0)).any():
        raise ValueError("tree_only rows must not contain a neural score")

    if MATCH_PROBABILITY in frame.columns:
        _finite(frame[MATCH_PROBABILITY], MATCH_PROBABILITY, probability=True)


def validate_decisions(frame: pd.DataFrame) -> None:
    validate_joined_scores(frame)
    if "selected" in frame.columns:
        selected = pd.to_numeric(frame["selected"], errors="raise").astype(int)
        if (~selected.isin([0, 1])).any():
            raise ValueError("selected must be 0/1")
