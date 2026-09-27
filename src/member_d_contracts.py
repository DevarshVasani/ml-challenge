"""Member D pipeline contracts, constants, and validators.

All new D code must import from here rather than repeating these definitions.
The canonical 3-part pair key is (candidate_source, candidate_entity_id, source1_entity_id).
Never join score frames by row order; always use the pair key.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical keys
# ---------------------------------------------------------------------------

PAIR_KEY: list[str] = ["candidate_source", "candidate_entity_id", "source1_entity_id"]
OWNERSHIP_GROUP_KEY: list[str] = ["candidate_source", "candidate_entity_id"]

# ---------------------------------------------------------------------------
# Score / route column names
# ---------------------------------------------------------------------------

SCORE_COL_TREE = "tree_score"
SCORE_COL_NEURAL = "neural_score"
SCORE_COL_FUSED = "match_probability"

ROUTE_COL_REQUESTED = "neural_requested"
ROUTE_COL_HAS = "has_neural_score"
ROUTE_COL_DECISION = "decision_route"

ROUTE_NEURAL_FUSION = "neural_fusion"
ROUTE_TREE_ONLY = "tree_only"
DECISION_ROUTES: frozenset[str] = frozenset({ROUTE_NEURAL_FUSION, ROUTE_TREE_ONLY})

# ---------------------------------------------------------------------------
# Valid source prefixes
# ---------------------------------------------------------------------------

VALID_SOURCE_PREFIXES: dict[str, str] = {"S2": "S2-", "S3": "S3-"}
VALID_SOURCES: frozenset[str] = frozenset(VALID_SOURCE_PREFIXES)

# ---------------------------------------------------------------------------
# Provenance fields required in every D artifact manifest
# ---------------------------------------------------------------------------

PROVENANCE_FIELDS: list[str] = [
    "candidate_version",
    "tree_model_version",
    "neural_checkpoint_version",
    "fusion_version",
    "calibration_version",
    "decoder_version",
    "query_split_hash",
]

# ---------------------------------------------------------------------------
# Columns that must NEVER appear in production score artifacts
# ---------------------------------------------------------------------------

FORBIDDEN_PRODUCTION_COLUMNS: frozenset[str] = frozenset({
    "label",
    "positive_injected_for_training",
    "source1_fold",
    "candidate_fold",
})

# ---------------------------------------------------------------------------
# Columns that must NEVER appear in training/feature lists
# ---------------------------------------------------------------------------

FORBIDDEN_TRAINING_COLUMNS: frozenset[str] = frozenset({
    "positive_injected_for_training",
    "is_injected_positive",
    "label",
    "fold",
    "source1_fold",
    "candidate_fold",
    "member_d_partition",
})


def validate_feature_allowlist(feature_columns: list[str]) -> None:
    """Validate that the final feature list contains no forbidden features."""
    forbidden = FORBIDDEN_TRAINING_COLUMNS & set(feature_columns)
    if forbidden:
        raise ValueError(f"Feature allowlist validation failed. Forbidden features found: {sorted(forbidden)}")

# ---------------------------------------------------------------------------
# Source derivation
# ---------------------------------------------------------------------------


def derive_candidate_source(entity_id: str) -> str:
    """Derive candidate_source from a validated entity_id prefix.

    Only S2-* and S3-* prefixes are accepted.  Raises ValueError for any
    entity_id that is ambiguous, empty, or uses an unknown prefix.
    """
    if not entity_id or not isinstance(entity_id, str):
        raise ValueError(f"entity_id must be a non-empty string, got {entity_id!r}")
    entity_id = entity_id.strip()
    for source, prefix in VALID_SOURCE_PREFIXES.items():
        if entity_id.startswith(prefix):
            return source
    raise ValueError(
        f"Cannot derive candidate_source from {entity_id!r}; "
        f"expected one of {sorted(VALID_SOURCE_PREFIXES.values())}"
    )


def validate_source_prefix_agreement(frame: pd.DataFrame) -> None:
    """Raise ValueError if candidate_source does not agree with candidate_entity_id prefix."""
    candidate_ids = frame["candidate_entity_id"].astype(str)
    sources = frame["candidate_source"].astype(str)
    expected_prefix = sources.map(VALID_SOURCE_PREFIXES)
    # rows with unknown source values
    unknown_source_mask = expected_prefix.isna()
    if unknown_source_mask.any():
        bad = sources[unknown_source_mask].unique().tolist()
        raise ValueError(f"candidate_source contains unknown values: {bad}")
    mismatch = ~candidate_ids.str[:3].eq(expected_prefix)
    if mismatch.any():
        bad_rows = frame.loc[mismatch, ["candidate_source", "candidate_entity_id"]].head(5)
        raise ValueError(
            f"candidate_source prefix does not agree with candidate_entity_id prefix:\n{bad_rows}"
        )


# ---------------------------------------------------------------------------
# Core score-frame validator
# ---------------------------------------------------------------------------


def validate_score_frame(
    frame: pd.DataFrame,
    *,
    require_tree_score: bool = True,
    check_routes: bool = True,
) -> None:
    """Validate a D-layer score frame.

    Enforced rules
    --------------
    - Pair key (candidate_source, candidate_entity_id, source1_entity_id) unique.
    - All ID columns non-empty strings.
    - candidate_source prefix matches S2-/S3- prefix of candidate_entity_id.
    - tree_score finite in [0, 1] when require_tree_score=True.
    - neural_score finite in [0, 1] when present and not null.
    - neural_requested == 1 with missing / null neural_score -> hard ValueError.
    - decision_route in DECISION_ROUTES when present.
    - No FORBIDDEN_PRODUCTION_COLUMNS in frame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("score frame must be a pandas DataFrame")

    # --- forbidden columns ---
    forbidden = FORBIDDEN_PRODUCTION_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(
            f"Production score artifact must not contain: {sorted(forbidden)}"
        )

    # --- pair key columns present ---
    missing_key_cols = [c for c in PAIR_KEY if c not in frame.columns]
    if missing_key_cols:
        raise ValueError(f"Score frame is missing pair key columns: {missing_key_cols}")

    # --- ID columns non-empty ---
    for col in PAIR_KEY:
        series = frame[col].astype(str).str.strip()
        if series.eq("").any() or frame[col].isna().any():
            raise ValueError(f"{col} contains empty or null values")

    # --- candidate_source prefix agreement ---
    validate_source_prefix_agreement(frame)

    # --- pair key uniqueness ---
    if frame.duplicated(PAIR_KEY).any():
        raise ValueError(
            "Score frame contains duplicate pair keys "
            "(candidate_source, candidate_entity_id, source1_entity_id)"
        )

    # --- tree_score ---
    if require_tree_score:
        if SCORE_COL_TREE not in frame.columns:
            raise ValueError(f"Score frame is missing required column: {SCORE_COL_TREE}")
        _validate_score_column(frame, SCORE_COL_TREE, nullable=False)

    # --- neural_score ---
    if SCORE_COL_NEURAL in frame.columns:
        _validate_score_column(frame, SCORE_COL_NEURAL, nullable=True)

    # --- route checks ---
    if check_routes and ROUTE_COL_REQUESTED in frame.columns:
        neural_req = pd.to_numeric(frame[ROUTE_COL_REQUESTED], errors="coerce")
        if SCORE_COL_NEURAL not in frame.columns:
            has_neural = pd.Series(0, index=frame.index)
        else:
            has_neural = frame[SCORE_COL_NEURAL].notna().astype(int)
        violation = (neural_req == 1) & (has_neural == 0)
        if violation.any():
            count = int(violation.sum())
            raise ValueError(
                f"{count} row(s) have neural_requested=1 but missing neural_score. "
                "This is a hard error; either provide the score or set neural_requested=0."
            )

    # --- decision_route ---
    if ROUTE_COL_DECISION in frame.columns:
        bad_routes = frame.loc[
            ~frame[ROUTE_COL_DECISION].isin(DECISION_ROUTES), ROUTE_COL_DECISION
        ].unique().tolist()
        if bad_routes:
            raise ValueError(
                f"decision_route contains invalid values: {bad_routes}. "
                f"Allowed: {sorted(DECISION_ROUTES)}"
            )

    # --- match_probability ---
    if SCORE_COL_FUSED in frame.columns:
        _validate_score_column(frame, SCORE_COL_FUSED, nullable=False)


def _validate_score_column(
    frame: pd.DataFrame, col: str, *, nullable: bool
) -> None:
    """Validate that a numeric score column contains finite values in [0, 1]."""
    series = pd.to_numeric(frame[col], errors="coerce")
    if not nullable:
        if series.isna().any():
            raise ValueError(f"{col} contains null or non-numeric values")
        if not series.map(math.isfinite).all():
            raise ValueError(f"{col} contains non-finite values (inf/nan)")
        if ((series < 0) | (series > 1)).any():
            raise ValueError(f"{col} contains values outside [0, 1]")
    else:
        non_null = series.dropna()
        if len(non_null) > 0:
            if not non_null.map(math.isfinite).all():
                raise ValueError(
                    f"{col} contains non-finite values (inf/nan) in non-null rows"
                )
            if ((non_null < 0) | (non_null > 1)).any():
                raise ValueError(f"{col} values outside [0, 1] in non-null rows")


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------


def check_provenance(
    provenance: dict[str, Any], *, require_all: bool = False
) -> list[str]:
    """Return list of missing provenance fields.  Raises if require_all=True."""
    missing = [f for f in PROVENANCE_FIELDS if f not in provenance or provenance[f] is None]
    if require_all and missing:
        raise ValueError(f"Provenance is missing required fields: {missing}")
    return missing
