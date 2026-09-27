"""Validate and adapt Member C's source-complete threshold handoff.

This adapter does not weaken Member D's canonical provenance contract.

Member C's current provenance file contains the complete ordered feature list
but hashes that list with a legacy comma-joined serialization. Member D hashes
the same ordered list as compact JSON. When (and only when) the supplied hash
matches the documented legacy serialization exactly, this module writes a
normalized local manifest containing D's canonical hash while preserving:
- the original hash,
- the original manifest checksum,
- the detected hash scheme.

The source manifest itself is never modified.

The prepared threshold candidate universe is selection-eligible for the
historical threshold split, but it is explicitly NOT marked as the production
test candidate universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .member_d_contracts import (
    DECISION_ROUTE,
    FORBIDDEN_PRODUCTION_COLUMNS,
    PAIR_KEY,
    ROUTE_DISCARD,
    ROUTE_NEURAL_FUSION,
    ROUTE_TREE_ONLY,
    TREE_SCORE,
    feature_columns_sha256,
    validate_feature_allowlist,
    validate_pair_keys,
    validate_tree_provenance,
)
from .neural_contracts import sha256_file

STRUCTURAL_GATE_COLUMNS = frozenset(
    {
        *PAIR_KEY,
        TREE_SCORE,
        "route",
        DECISION_ROUTE,
        "gate_rank",
        "gate_score_gap",
        "neural_requested",
        "has_neural_score",
        "source_group_complete",
        "diagnostic_only",
        "competition_features_trusted",
    }
)

_ROUTE_ALIASES = {
    "neural": ROUTE_NEURAL_FUSION,
    "neural_fusion": ROUTE_NEURAL_FUSION,
    "tree": ROUTE_TREE_ONLY,
    "tree_only": ROUTE_TREE_ONLY,
    "discard": ROUTE_DISCARD,
}


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _write_table(frame: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, sep="\t", index=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[PAIR_KEY].astype(str))


def _normalized_routes(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    normalized = raw.map(_ROUTE_ALIASES)
    if normalized.isna().any():
        bad = sorted(raw[normalized.isna()].unique())
        raise ValueError(f"Unknown route values: {bad}")
    return normalized


def legacy_comma_feature_columns_sha256(columns: list[str]) -> str:
    """Hash scheme used by the supplied C threshold provenance v1."""
    return hashlib.sha256(",".join(map(str, columns)).encode("utf-8")).hexdigest()


def _gate_model_features(
    gate: pd.DataFrame,
    provenance_features: list[str],
) -> list[str]:
    """Return the model feature columns present in the gate.

    Physical parquet column order is not treated as model order. Model order
    comes from provenance.feature_columns. We do, however, require exact set
    equality between the model feature list and non-structural gate columns.
    """
    gate_features = [
        str(c)
        for c in gate.columns
        if str(c) not in STRUCTURAL_GATE_COLUMNS
    ]
    missing = sorted(set(provenance_features) - set(gate_features))
    extra = sorted(set(gate_features) - set(provenance_features))
    if missing or extra:
        raise ValueError(
            "Gate/model feature-set mismatch: "
            f"missing_from_gate={missing[:10]} "
            f"extra_in_gate={extra[:10]}"
        )
    return gate_features


def normalize_tree_provenance(
    provenance_path: str | Path,
    gate_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate C provenance and write a D-canonical local copy.

    Accepted source feature-hash schemes:
    1. D canonical compact-JSON hash.
    2. C legacy comma-joined hash, only when it matches exactly.

    Any other hash is rejected.
    """
    source = _load_json(provenance_path)
    gate = _read_table(gate_path)
    validate_pair_keys(gate)

    forbidden = sorted(set(gate.columns) & FORBIDDEN_PRODUCTION_COLUMNS)
    if forbidden:
        raise ValueError(
            "C gate contains forbidden/truth-derived fields: "
            f"{forbidden}"
        )

    required = {
        "tree_model_version",
        "training_git_sha",
        "feature_columns",
        "feature_columns_sha256",
        "feature_schema_version",
        "candidate_version",
        "feature_contract_valid",
        "complete_by_source_record",
    }
    missing_fields = sorted(required - set(source))
    if missing_fields:
        raise ValueError(
            f"C tree provenance missing fields: {missing_fields}"
        )
    if source.get("feature_contract_valid") is not True:
        raise ValueError("C tree provenance is not feature_contract_valid=true")
    if source.get("complete_by_source_record") is not True:
        raise ValueError(
            "C threshold handoff must be complete_by_source_record=true"
        )

    features = [str(x) for x in source["feature_columns"]]
    if not features:
        raise ValueError("C tree provenance feature_columns is empty")
    if len(features) != len(set(features)):
        raise ValueError("C tree provenance has duplicate feature columns")

    validate_feature_allowlist(features)
    gate_features = _gate_model_features(gate, features)

    source_hash = str(source["feature_columns_sha256"])
    canonical_hash = feature_columns_sha256(features)
    legacy_hash = legacy_comma_feature_columns_sha256(features)

    if source_hash == canonical_hash:
        source_hash_scheme = "member_d_compact_json_v1"
    elif source_hash == legacy_hash:
        source_hash_scheme = "member_c_comma_join_v1"
    else:
        raise ValueError(
            "C feature_columns_sha256 matches neither the Member-D canonical "
            "compact-JSON serialization nor the recognized C comma-joined "
            "serialization. Refusing to normalize provenance."
        )

    normalized = dict(source)
    normalized["feature_columns_sha256"] = canonical_hash
    normalized["source_feature_columns_sha256"] = source_hash
    normalized["source_feature_hash_scheme"] = source_hash_scheme
    normalized["source_manifest_sha256"] = sha256_file(provenance_path)
    normalized["member_d_normalization"] = {
        "version": "member_d_tree_provenance_adapter_v1",
        "changed_semantics": False,
        "changed_feature_order": False,
        "hash_serialization_only": source_hash != canonical_hash,
        "gate_feature_set_exact_match": True,
        "gate_column_order_matches_model_feature_order": (
            gate_features == features
        ),
    }

    # The canonical validator must pass on the normalized local copy.
    validate_tree_provenance(normalized, features)
    _write_json(normalized, output_path)
    return normalized


def validate_gate_route(
    gate_path: str | Path,
    route_table_path: str | Path,
) -> dict[str, Any]:
    gate = _read_table(gate_path)
    route = _read_table(route_table_path)
    validate_pair_keys(gate)
    validate_pair_keys(route)

    forbidden = sorted(set(gate.columns) & FORBIDDEN_PRODUCTION_COLUMNS)
    if forbidden:
        raise ValueError(
            f"C gate contains forbidden/truth-derived fields: {forbidden}"
        )
    if TREE_SCORE not in gate.columns:
        raise ValueError(f"C gate is missing {TREE_SCORE}")
    if TREE_SCORE not in route.columns:
        raise ValueError(f"C route table is missing {TREE_SCORE}")

    gate_route_col = (
        DECISION_ROUTE
        if DECISION_ROUTE in gate.columns
        else "route"
        if "route" in gate.columns
        else None
    )
    route_route_col = (
        DECISION_ROUTE
        if DECISION_ROUTE in route.columns
        else "route"
        if "route" in route.columns
        else None
    )
    if gate_route_col is None:
        raise ValueError("C gate is missing route/decision_route")
    if route_route_col is None:
        raise ValueError("C route table is missing route/decision_route")

    gate_keys = _key_index(gate)
    route_keys = _key_index(route)
    missing_route = gate_keys.difference(route_keys)
    extra_route = route_keys.difference(gate_keys)
    if len(missing_route) or len(extra_route):
        raise ValueError(
            "C gate/route key coverage mismatch: "
            f"missing_route={len(missing_route)} "
            f"extra_route={len(extra_route)}"
        )

    merged = gate[
        PAIR_KEY + [TREE_SCORE, gate_route_col]
    ].merge(
        route[PAIR_KEY + [TREE_SCORE, route_route_col]],
        on=PAIR_KEY,
        how="inner",
        validate="1:1",
        suffixes=("_gate", "_route"),
    )

    gate_tree = pd.to_numeric(
        merged[f"{TREE_SCORE}_gate"], errors="raise"
    ).to_numpy(float)
    route_tree = pd.to_numeric(
        merged[f"{TREE_SCORE}_route"], errors="raise"
    ).to_numpy(float)

    if not np.isfinite(gate_tree).all() or not np.isfinite(route_tree).all():
        raise ValueError("C tree scores contain non-finite values")
    if (
        (gate_tree < 0).any()
        or (gate_tree > 1).any()
        or (route_tree < 0).any()
        or (route_tree > 1).any()
    ):
        raise ValueError("C tree scores must be in [0,1]")

    tree_mismatch = ~np.isclose(
        gate_tree, route_tree, rtol=0.0, atol=1e-12
    )
    if tree_mismatch.any():
        raise ValueError(
            f"C gate/route tree_score mismatch on "
            f"{int(tree_mismatch.sum())} rows"
        )

    gate_routes = _normalized_routes(
        merged[f"{gate_route_col}_gate"]
    )
    route_routes = _normalized_routes(
        merged[f"{route_route_col}_route"]
    )
    route_mismatch = gate_routes != route_routes
    if route_mismatch.any():
        raise ValueError(
            f"C gate/route route mismatch on "
            f"{int(route_mismatch.sum())} rows"
        )

    return {
        "row_count": len(gate),
        "gate_sha256": sha256_file(gate_path),
        "route_table_sha256": sha256_file(route_table_path),
        "duplicate_gate_keys": int(gate.duplicated(PAIR_KEY).sum()),
        "duplicate_route_keys": int(route.duplicated(PAIR_KEY).sum()),
        "missing_route_keys": 0,
        "extra_route_keys": 0,
        "tree_score_mismatch_count": 0,
        "route_mismatch_count": 0,
        "route_counts": {
            str(k): int(v)
            for k, v in gate_routes.value_counts().to_dict().items()
        },
    }


def prepare_source_complete_threshold_handoff(
    gate_path: str | Path,
    route_table_path: str | Path,
    provenance_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare C's handoff for Member D's strict score_join contract."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    validation = validate_gate_route(gate_path, route_table_path)
    normalized_manifest_path = out / "tree_manifest.normalized.json"
    normalized = normalize_tree_provenance(
        provenance_path,
        gate_path,
        normalized_manifest_path,
    )

    gate = _read_table(gate_path)
    route = _read_table(route_table_path)
    features = [str(x) for x in normalized["feature_columns"]]

    # Candidate artifact deliberately excludes tree outputs/routing diagnostics;
    # score_join will attach tree_score + canonical route separately.
    candidate = gate[PAIR_KEY + features].copy()
    candidate["source_group_complete"] = True

    candidate_path = out / "candidates.parquet"
    tree_path = out / "tree_scores.parquet"
    route_path = out / "routes.parquet"

    _write_table(candidate, candidate_path)
    _write_table(route[PAIR_KEY + [TREE_SCORE]].copy(), tree_path)

    route_col = (
        DECISION_ROUTE
        if DECISION_ROUTE in route.columns
        else "route"
    )
    route_out = route[PAIR_KEY + [route_col]].copy()
    if route_col != "route":
        route_out = route_out.rename(columns={route_col: "route"})
    _write_table(route_out, route_path)

    candidate_manifest = {
        "artifact_kind": "member_d_source_complete_threshold_candidates",
        "completion_state": "complete",
        "row_count": len(candidate),
        "candidate_version": normalized["candidate_version"],
        "complete_by_source_record": True,
        "files": {
            candidate_path.name: sha256_file(candidate_path),
        },
        "metadata": {
            "selection_scope": "historical_threshold",
            "historical_bootstrap_candidate_universe": True,
            "production_candidate_universe": False,
            "source_gate_sha256": sha256_file(gate_path),
            "source_route_table_sha256": sha256_file(route_table_path),
            "source_tree_provenance_sha256": sha256_file(provenance_path),
            "tree_model_version": normalized["tree_model_version"],
            "feature_schema_version": normalized[
                "feature_schema_version"
            ],
        },
    }
    candidate_manifest_path = out / "candidate_manifest.json"
    _write_json(candidate_manifest, candidate_manifest_path)

    report = {
        "artifact_kind": "member_d_source_complete_threshold_handoff",
        "version": "source_complete_threshold_handoff_v1",
        "selection_eligible": True,
        "production_ready": False,
        "production_blocker": (
            "This is C's historical bootstrap threshold candidate universe, "
            "not B's new reverse-retrieval production/test candidate universe."
        ),
        "complete_by_source_record": True,
        "feature_contract_valid": True,
        "candidate_version": normalized["candidate_version"],
        "tree_model_version": normalized["tree_model_version"],
        "tree_training_git_sha": normalized["training_git_sha"],
        "feature_schema_version": normalized["feature_schema_version"],
        "feature_columns_sha256": normalized[
            "feature_columns_sha256"
        ],
        "source_feature_columns_sha256": normalized[
            "source_feature_columns_sha256"
        ],
        "source_feature_hash_scheme": normalized[
            "source_feature_hash_scheme"
        ],
        "validation": validation,
        "outputs": {
            "candidate_artifact": str(candidate_path),
            "candidate_manifest": str(candidate_manifest_path),
            "tree_scores": str(tree_path),
            "tree_manifest": str(normalized_manifest_path),
            "route_table": str(route_path),
        },
    }
    report_path = out / "handoff_report.json"
    _write_json(report, report_path)
    report["outputs"]["handoff_report"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--route-table", required=True)
    parser.add_argument("--tree-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result = prepare_source_complete_threshold_handoff(
        args.gate,
        args.route_table,
        args.tree_provenance,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
