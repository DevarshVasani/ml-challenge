"""Validation and metadata helpers for the streaming candidate artifact."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


CANDIDATE_SCHEMA_VERSION = "candidate-artifact-v1"
REQUIRED_CANDIDATE_COLUMNS = (
    "source1_entity_id", "candidate_entity_id", "candidate_source",
    "name_tfidf_score", "address_tfidf_score", "name_rank", "address_rank",
    "retrieved_by_name", "retrieved_by_address", "retrieval_score",
)
MANIFEST_FIELDS = (
    "schema_version", "split", "git_commit", "seed", "top_k_name",
    "top_k_address", "minimum_score_settings", "number_of_buckets",
    "partition_columns", "row_count", "created_at",
)


def stable_bucket(entity_id: str, seed: int, number_of_buckets: int) -> int:
    """Return the documented stable bucket for a Source 1 ID."""
    if number_of_buckets <= 0:
        raise ValueError("number_of_buckets must be positive")
    digest = hashlib.sha256(f"{seed}:{entity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % number_of_buckets


bucket_for_entity = stable_bucket


def validate_candidate_frame(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless *frame* satisfies the row contract."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("candidate artifact must be a pandas DataFrame")
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate artifact is missing columns: {missing}")
    if frame[["source1_entity_id", "candidate_entity_id"]].isna().any().any():
        raise ValueError("Candidate IDs may not be null")
    s1 = frame["source1_entity_id"].astype(str)
    candidate = frame["candidate_entity_id"].astype(str)
    if (~s1.str.startswith("S1-")).any():
        raise ValueError("All source1_entity_id values must start with 'S1-'")
    sources = frame["candidate_source"].astype(str)
    if (~sources.isin(["S2", "S3"])).any():
        raise ValueError("candidate_source must contain only S2 or S3")
    expected_prefix = sources.map({"S2": "S2-", "S3": "S3-"})
    if (~candidate.str[:3].eq(expected_prefix)).any():
        raise ValueError("candidate_entity_id prefix must agree with candidate_source")
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise ValueError("Candidate artifact contains duplicate S1/candidate pairs")
    for column in ("name_tfidf_score", "address_tfidf_score", "retrieval_score"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"{column} must contain finite numeric values")
    for column in ("name_rank", "address_rank"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain non-negative integers")
    for column in ("retrieved_by_name", "retrieved_by_address"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~values.isin([0, 1])).any():
            raise ValueError(f"{column} must contain only 0 or 1")


def validate_candidate_partition(frame: pd.DataFrame) -> None:
    validate_candidate_frame(frame)


validate_candidate_artifact = validate_candidate_frame


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("candidate manifest must be a mapping")
    missing = [field for field in MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"Candidate manifest is missing fields: {missing}")
    if manifest["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported candidate schema version: {manifest['schema_version']}")
    if not isinstance(manifest["split"], str) or not manifest["split"]:
        raise ValueError("manifest split must be a non-empty string")
    if not isinstance(manifest["number_of_buckets"], int) or manifest["number_of_buckets"] <= 0:
        raise ValueError("manifest number_of_buckets must be a positive integer")
    if not isinstance(manifest["partition_columns"], list):
        raise ValueError("manifest partition_columns must be a list")
    if not isinstance(manifest["row_count"], int) or manifest["row_count"] < 0:
        raise ValueError("manifest row_count must be a non-negative integer")


def validate_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(value)
    return value
