"""Strict Member D ID-based score assembly with route/provenance enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .member_d_contracts import (
    DECISION_ROUTE,
    HAS_NEURAL_SCORE,
    NEURAL_LOGIT,
    NEURAL_PROBABILITY,
    NEURAL_REQUESTED,
    PAIR_KEY,
    ROUTE_DISCARD,
    ROUTE_NEURAL_FUSION,
    ROUTE_TREE_ONLY,
    TREE_SCORE,
    validate_joined_scores,
    validate_pair_keys,
    validate_tree_provenance,
)
from .member_d_manifest import MemberDManifest, write_manifest
from .neural_contracts import sha256_file


def _read_many(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        resolved = str(path.resolve())
        if resolved in seen:
            raise ValueError(f"Duplicate shard path: {path}")
        seen.add(resolved)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(path))
        else:
            frames.append(
                pd.read_csv(
                    path,
                    sep="\t",
                    dtype=object,
                    keep_default_na=False,
                )
            )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def _key_set(
    frame: pd.DataFrame,
) -> set[tuple[str, str, str]]:
    return set(
        map(tuple, frame[PAIR_KEY].astype(str).to_numpy())
    )


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_tree_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _load_json(path)
    features = manifest.get("feature_columns") or []
    validate_tree_provenance(manifest, features)
    return manifest


def _validate_candidate_manifest(
    path: str | Path,
) -> dict[str, Any]:
    manifest = _load_json(path)
    state = manifest.get(
        "completion_state", manifest.get("completion")
    )
    if state != "complete":
        raise ValueError("Candidate artifact manifest is incomplete")

    complete = manifest.get(
        "complete_by_source_record",
        manifest.get("metadata", {}).get(
            "complete_by_source_record"
        ),
    )
    if complete is not True:
        raise ValueError(
            "D competition/fusion candidates must be "
            "complete_by_source_record=true"
        )
    return manifest


_ROUTE_ALIASES = {
    "neural": ROUTE_NEURAL_FUSION,
    "neural_fusion": ROUTE_NEURAL_FUSION,
    "tree": ROUTE_TREE_ONLY,
    "tree_only": ROUTE_TREE_ONLY,
    "discard": ROUTE_DISCARD,
}


def _normalize_routes(route: pd.DataFrame) -> pd.DataFrame:
    """Normalize C/A handoff route columns to D's canonical contract."""
    out = route.copy()

    if DECISION_ROUTE not in out.columns:
        if "route" not in out.columns:
            raise ValueError(
                "Route table requires decision_route or route"
            )
        out[DECISION_ROUTE] = out["route"]

    raw = out[DECISION_ROUTE].astype(str)
    normalized = raw.map(_ROUTE_ALIASES)
    if normalized.isna().any():
        bad = sorted(raw[normalized.isna()].unique())
        raise ValueError(f"Unknown route values: {bad}")
    out[DECISION_ROUTE] = normalized

    inferred = (
        out[DECISION_ROUTE] == ROUTE_NEURAL_FUSION
    ).astype("int8")
    if NEURAL_REQUESTED not in out.columns:
        out[NEURAL_REQUESTED] = inferred
    else:
        requested = pd.to_numeric(
            out[NEURAL_REQUESTED], errors="raise"
        ).astype(int)
        if (~requested.isin([0, 1])).any():
            raise ValueError("neural_requested must be 0/1")
        if (requested != inferred).any():
            raise ValueError(
                "route and neural_requested disagree"
            )
        out[NEURAL_REQUESTED] = requested

    return out


def _route_from_candidates(
    joined: pd.DataFrame,
) -> pd.DataFrame:
    out = joined.copy()
    if DECISION_ROUTE in out.columns:
        out = _normalize_routes(out)
    elif NEURAL_REQUESTED in out.columns:
        requested = pd.to_numeric(
            out[NEURAL_REQUESTED], errors="raise"
        ).astype(int)
        if (~requested.isin([0, 1])).any():
            raise ValueError("neural_requested must be 0/1")
        out[DECISION_ROUTE] = np.where(
            requested == 1,
            ROUTE_NEURAL_FUSION,
            ROUTE_TREE_ONLY,
        )
        out[NEURAL_REQUESTED] = requested
    else:
        raise ValueError(
            "No route table and candidate artifact has neither "
            "decision_route/route nor neural_requested; "
            "D refuses to guess routes"
        )
    return out


def join_scores(
    decision_candidate_shards: Sequence[str | Path],
    tree_score_shards: Sequence[str | Path],
    *,
    output_dir: str | Path,
    candidate_manifest_path: str | Path,
    tree_score_manifest_path: str | Path,
    neural_score_shards: Sequence[str | Path] | None = None,
    neural_score_manifest_path: str | Path | None = None,
    route_table_path: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_manifest = _validate_candidate_manifest(
        candidate_manifest_path
    )
    tree_manifest = _validate_tree_manifest(
        tree_score_manifest_path
    )

    candidate_version = candidate_manifest.get(
        "candidate_version"
    )
    tree_candidate_version = tree_manifest.get(
        "candidate_version"
    )
    if (
        candidate_version
        and tree_candidate_version
        and tree_candidate_version != candidate_version
    ):
        raise ValueError(
            "Tree artifact candidate_version does not match "
            "candidate artifact"
        )

    candidate_schema = candidate_manifest.get(
        "candidate_schema_version",
        candidate_manifest.get("metadata", {}).get(
            "candidate_schema_version"
        ),
    )
    tree_candidate_schema = tree_manifest.get(
        "candidate_schema_version"
    )
    if (
        candidate_schema
        and tree_candidate_schema
        and tree_candidate_schema != candidate_schema
    ):
        raise ValueError(
            "Tree artifact candidate schema does not match "
            "candidate artifact"
        )

    candidates = _read_many(decision_candidate_shards)
    if candidates.empty:
        raise ValueError("Decision candidate artifact is empty")
    validate_pair_keys(candidates)

    tree = _read_many(tree_score_shards)
    if "score" in tree.columns and TREE_SCORE not in tree.columns:
        tree = tree.rename(columns={"score": TREE_SCORE})
    validate_pair_keys(tree)
    if TREE_SCORE not in tree.columns:
        raise ValueError(f"Tree artifact missing {TREE_SCORE}")
    tree[TREE_SCORE] = pd.to_numeric(
        tree[TREE_SCORE], errors="raise"
    )
    if tree[TREE_SCORE].isna().any():
        raise ValueError("Tree scores contain missing values")

    candidate_keys = _key_set(candidates)
    tree_keys = _key_set(tree)
    extra_tree = tree_keys - candidate_keys
    missing_tree = candidate_keys - tree_keys
    if extra_tree:
        raise ValueError(
            f"Tree scores contain unknown pairs: "
            f"{sorted(extra_tree)[:5]}"
        )
    if missing_tree:
        raise ValueError(
            f"Tree scores missing candidate pairs: "
            f"{sorted(missing_tree)[:5]}"
        )

    joined = candidates.merge(
        tree[PAIR_KEY + [TREE_SCORE]],
        on=PAIR_KEY,
        how="left",
        validate="1:1",
    )

    if route_table_path:
        route_path = Path(route_table_path)
        route = (
            pd.read_parquet(route_path)
            if route_path.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(
                route_path,
                sep="\t",
                dtype=object,
                keep_default_na=False,
            )
        )
        validate_pair_keys(route)
        route = _normalize_routes(route)

        route_keys = _key_set(route)
        extra_route = route_keys - candidate_keys
        missing_route = candidate_keys - route_keys
        if extra_route or missing_route:
            raise ValueError(
                "Route-table coverage mismatch: "
                f"extra={len(extra_route)} "
                f"missing={len(missing_route)}"
            )

        joined = joined.drop(
            columns=[
                c
                for c in (NEURAL_REQUESTED, DECISION_ROUTE, "route")
                if c in joined.columns
            ],
            errors="ignore",
        )
        joined = joined.merge(
            route[PAIR_KEY + [NEURAL_REQUESTED, DECISION_ROUTE]],
            on=PAIR_KEY,
            how="left",
            validate="1:1",
        )
    else:
        joined = _route_from_candidates(joined)

    neural_manifest: dict[str, Any] = {}
    if neural_score_shards:
        if not neural_score_manifest_path:
            raise ValueError(
                "Neural score shards require "
                "neural_score_manifest_path"
            )
        neural_manifest = _load_json(
            neural_score_manifest_path
        )
        state = neural_manifest.get(
            "completion_state",
            neural_manifest.get("completion"),
        )
        if state != "complete":
            raise ValueError("Neural score manifest is incomplete")
        if not (
            neural_manifest.get("checkpoint_id")
            or neural_manifest.get("neural_model_version")
        ):
            raise ValueError(
                "Neural score manifest must identify "
                "checkpoint_id/neural_model_version"
            )
        if (
            candidate_version
            and neural_manifest.get("candidate_version")
            and neural_manifest["candidate_version"]
            != candidate_version
        ):
            raise ValueError(
                "Neural artifact candidate_version does not match "
                "candidate artifact"
            )

        neural = _read_many(neural_score_shards)
        validate_pair_keys(neural)
        if (
            "score" in neural.columns
            and NEURAL_PROBABILITY not in neural.columns
            and NEURAL_LOGIT not in neural.columns
        ):
            neural = neural.rename(
                columns={"score": NEURAL_PROBABILITY}
            )
        if (
            NEURAL_LOGIT not in neural.columns
            and NEURAL_PROBABILITY not in neural.columns
        ):
            raise ValueError(
                "Neural scores require neural_logit "
                "or neural_probability"
            )

        neural_keys = _key_set(neural)
        extra = neural_keys - candidate_keys
        if extra:
            raise ValueError(
                f"Neural artifact contains unknown pairs: "
                f"{sorted(extra)[:5]}"
            )

        cols = PAIR_KEY + [
            c
            for c in (NEURAL_LOGIT, NEURAL_PROBABILITY)
            if c in neural.columns
        ]
        joined = joined.merge(
            neural[cols],
            on=PAIR_KEY,
            how="left",
            validate="1:1",
        )
    else:
        joined[NEURAL_LOGIT] = np.nan

    neural_present = pd.Series(False, index=joined.index)
    for col in (NEURAL_LOGIT, NEURAL_PROBABILITY):
        if col in joined.columns:
            neural_present |= pd.to_numeric(
                joined[col], errors="coerce"
            ).notna()
    joined[HAS_NEURAL_SCORE] = neural_present.astype("int8")

    requested = pd.to_numeric(
        joined[NEURAL_REQUESTED], errors="raise"
    ).astype(int)
    if ((requested == 1) & ~neural_present).any():
        missing = joined.loc[
            (requested == 1) & ~neural_present, PAIR_KEY
        ].head(5).to_dict("records")
        raise ValueError(
            "At least one neural_requested pair is missing its "
            f"neural score: {missing}"
        )

    validate_joined_scores(joined)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "joined_scores.parquet"
    joined.to_parquet(out_path, index=False)

    metadata = {
        "tree_feature_contract_validated": True,
        "tree_feature_contract_validation_version": "member_d_v1",
        "tree_training_git_sha": tree_manifest["training_git_sha"],
        "tree_feature_schema_version": tree_manifest[
            "feature_schema_version"
        ],
        "tree_feature_columns_sha256": tree_manifest[
            "feature_columns_sha256"
        ],
        "feature_contract_valid": True,
        "complete_by_source_record": True,
        **dict(provenance or {}),
    }
    manifest = MemberDManifest(
        artifact_kind="joined_scores",
        version=str((provenance or {}).get("version", "v1")),
        completion_state="complete",
        row_count=len(joined),
        candidate_version=candidate_version,
        tree_model_version=tree_manifest["tree_model_version"],
        neural_model_version=(
            neural_manifest.get("neural_model_version")
            or neural_manifest.get("checkpoint_id")
        )
        if neural_manifest
        else None,
        files={out_path.name: sha256_file(out_path)},
        metadata=metadata,
    )
    write_manifest(out_dir / "join_manifest.json", manifest)
    return {
        "output_path": str(out_path),
        "row_count": len(joined),
        "manifest": manifest.to_dict(),
    }
