"""Audit Member C's threshold-v2 and final-standin handoff for Member D.

The current C handoff contains two different classes of artifacts:

* threshold-v2: source-record-complete historical threshold artifacts that are
  valid for D's frozen threshold selection;
* production_*_v1-final-standin: a technically complete end-to-end rehearsal
  built from the historical ``final_pairs`` universe.  These artifacts are
  explicitly NOT the real B reverse-union-v1 production candidate universe.

This module validates both classes without silently promoting the stand-in to
production.  It checks manifests, per-shard completion sidecars, checksums,
pair-key coverage, tree-score agreement, route counts, feature provenance and
source-record shard integrity.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .member_d_c_handoff import (
    legacy_comma_feature_columns_sha256,
    normalize_tree_provenance,
    validate_gate_route,
)
from .member_d_contracts import (
    FORBIDDEN_PRODUCTION_COLUMNS,
    PAIR_KEY,
    SOURCE_GROUP_KEY,
    TREE_SCORE,
    feature_columns_sha256,
    validate_feature_allowlist,
    validate_pair_keys,
)
from .neural_contracts import sha256_file


_ROUTE_ALIASES = {
    "neural": "neural_fusion",
    "neural_fusion": "neural_fusion",
    "tree": "tree_only",
    "tree_only": "tree_only",
    "discard": "discard",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _manifest_state(manifest: dict[str, Any]) -> str:
    return str(
        manifest.get(
            "completion_state",
            manifest.get("completion", ""),
        )
    )


def _shard_id_from_name(name: str) -> str:
    stem = Path(name).name
    if not stem.startswith("shard-"):
        raise ValueError(f"Unexpected C shard name: {name}")
    value = stem.split("shard-", 1)[1].split(".", 1)[0]
    if not value.isdigit():
        raise ValueError(f"Unexpected C shard id in {name}")
    return value


def _validate_parent_manifest(
    manifest_path: str | Path,
    *,
    expected_standin: bool,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _load_json(path)
    if _manifest_state(manifest) != "complete":
        raise ValueError(f"Incomplete C manifest: {path}")
    if bool(manifest.get("is_production_standin")) != expected_standin:
        raise ValueError(
            f"{path} is_production_standin does not match "
            f"expected={expected_standin}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"C manifest has no shards: {path}")
    if int(manifest.get("shard_count", -1)) != len(shards):
        raise ValueError(f"C manifest shard_count mismatch: {path}")
    row_sum = sum(int(item["row_count"]) for item in shards)
    if row_sum != int(manifest.get("total_rows", -1)):
        raise ValueError(
            f"C manifest row total mismatch: {path}: "
            f"{row_sum} != {manifest.get('total_rows')}"
        )
    ids = [_shard_id_from_name(item["path"]) for item in shards]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate C shard ids in {path}")
    return manifest


def _resolve_shards(
    manifest: dict[str, Any],
    shard_dir: str | Path,
    *,
    expected_kind: str,
) -> list[dict[str, Any]]:
    root = Path(shard_dir)
    result: list[dict[str, Any]] = []
    for item in manifest["shards"]:
        name = Path(item["path"]).name
        shard_id = _shard_id_from_name(name)
        data_path = root / name
        sidecar_path = root / f"{name}.complete.json"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not sidecar_path.exists():
            raise FileNotFoundError(sidecar_path)

        sidecar = _load_json(sidecar_path)
        if sidecar.get("status") != "complete":
            raise ValueError(f"Incomplete sidecar: {sidecar_path}")
        if str(sidecar.get("kind")) != expected_kind:
            raise ValueError(
                f"Unexpected sidecar kind in {sidecar_path}: "
                f"{sidecar.get('kind')!r}"
            )
        if str(sidecar.get("shard_id")) != shard_id:
            raise ValueError(
                f"Sidecar shard id mismatch in {sidecar_path}"
            )
        expected_rows = int(item["row_count"])
        if int(sidecar.get("row_count", -1)) != expected_rows:
            raise ValueError(
                f"Sidecar row count mismatch in {sidecar_path}"
            )
        actual_sha = sha256_file(data_path)
        if actual_sha != str(sidecar.get("sha256", "")):
            raise ValueError(
                f"Checksum mismatch for C shard {data_path}"
            )
        result.append(
            {
                "shard_id": shard_id,
                "path": data_path,
                "sidecar_path": sidecar_path,
                "row_count": expected_rows,
                "sha256": actual_sha,
            }
        )
    return result


def _key_tokens(frame: pd.DataFrame) -> Iterable[str]:
    for source, candidate, s1 in frame[PAIR_KEY].astype(str).itertuples(
        index=False, name=None
    ):
        yield f"{source}\x1f{candidate}\x1f{s1}"


def _group_tokens(frame: pd.DataFrame) -> Iterable[str]:
    for source, candidate in frame[SOURCE_GROUP_KEY].astype(str).itertuples(
        index=False, name=None
    ):
        yield f"{source}\x1f{candidate}"


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(map(tuple, frame[PAIR_KEY].astype(str).to_numpy()))


def _normalize_routes(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip().str.lower()
    normalized = raw.map(_ROUTE_ALIASES)
    if normalized.isna().any():
        bad = sorted(raw[normalized.isna()].unique())
        raise ValueError(f"Unknown C route values: {bad}")
    return normalized


def _feature_provenance(
    candidate_manifest: dict[str, Any],
    tree_manifest: dict[str, Any],
    threshold_provenance_path: str | Path,
) -> dict[str, Any]:
    threshold = _load_json(threshold_provenance_path)
    features = [str(x) for x in threshold.get("feature_columns") or []]
    if not features:
        raise ValueError("C threshold provenance has no feature_columns")
    validate_feature_allowlist(features)

    candidate_features = [
        str(x) for x in candidate_manifest.get("feature_columns") or []
    ]
    if candidate_features != features:
        raise ValueError(
            "C final-standin candidate feature list differs from "
            "threshold tree provenance"
        )
    if list(candidate_manifest.get("pair_key") or []) != list(PAIR_KEY):
        raise ValueError(
            f"C candidate pair_key is not D canonical key {PAIR_KEY}"
        )

    candidate_version = str(candidate_manifest.get("candidate_version", ""))
    if candidate_version != str(threshold.get("candidate_version", "")):
        raise ValueError(
            "C stand-in candidate_version differs from threshold provenance"
        )

    for field in ("tree_model_version", "training_git_sha"):
        if str(tree_manifest.get(field, "")) != str(threshold.get(field, "")):
            raise ValueError(
                f"C stand-in tree {field} differs from threshold provenance"
            )

    source_hash = str(threshold.get("feature_columns_sha256", ""))
    tree_hash = str(tree_manifest.get("feature_columns_sha256", ""))
    if tree_hash != source_hash:
        raise ValueError(
            "C stand-in tree feature hash differs from threshold provenance"
        )

    canonical_hash = feature_columns_sha256(features)
    legacy_hash = legacy_comma_feature_columns_sha256(features)
    if source_hash == canonical_hash:
        scheme = "member_d_compact_json_v1"
    elif source_hash == legacy_hash:
        scheme = "member_c_comma_join_v1"
    else:
        raise ValueError(
            "C feature hash matches neither D canonical nor C legacy scheme"
        )

    return {
        "feature_count": len(features),
        "feature_columns": features,
        "source_feature_columns_sha256": source_hash,
        "canonical_feature_columns_sha256": canonical_hash,
        "source_feature_hash_scheme": scheme,
        "feature_schema_version": threshold.get(
            "feature_schema_version"
        ),
        "candidate_version": candidate_version,
        "tree_model_version": threshold.get("tree_model_version"),
        "training_git_sha": threshold.get("training_git_sha"),
        "feature_contract_valid": (
            threshold.get("feature_contract_valid") is True
        ),
    }


def _validate_sharded_standin(
    *,
    candidate_manifest_path: str | Path,
    candidate_dir: str | Path,
    tree_manifest_path: str | Path,
    tree_dir: str | Path,
    route_manifest_path: str | Path,
    route_dir: str | Path,
    threshold_provenance_path: str | Path,
) -> dict[str, Any]:
    candidate_manifest = _validate_parent_manifest(
        candidate_manifest_path, expected_standin=True
    )
    tree_manifest = _validate_parent_manifest(
        tree_manifest_path, expected_standin=True
    )
    route_manifest = _validate_parent_manifest(
        route_manifest_path, expected_standin=True
    )
    if route_manifest.get("complete_by_source_record") is not True:
        raise ValueError(
            "C route manifest must declare complete_by_source_record=true"
        )

    totals = {
        int(candidate_manifest["total_rows"]),
        int(tree_manifest["total_rows"]),
        int(route_manifest["total_rows"]),
    }
    if len(totals) != 1:
        raise ValueError(
            "C candidate/tree/route total_rows do not agree"
        )

    candidate_shards = _resolve_shards(
        candidate_manifest,
        candidate_dir,
        expected_kind="candidates",
    )
    tree_shards = _resolve_shards(
        tree_manifest,
        tree_dir,
        expected_kind="tree_scores",
    )
    route_shards = _resolve_shards(
        route_manifest,
        route_dir,
        expected_kind="routes",
    )
    by_id = lambda rows: {row["shard_id"]: row for row in rows}
    c_by_id = by_id(candidate_shards)
    t_by_id = by_id(tree_shards)
    r_by_id = by_id(route_shards)
    if set(c_by_id) != set(t_by_id) or set(c_by_id) != set(r_by_id):
        raise ValueError(
            "C candidate/tree/route shard-id coverage differs"
        )

    provenance = _feature_provenance(
        candidate_manifest,
        tree_manifest,
        threshold_provenance_path,
    )
    features = provenance["feature_columns"]

    seen_pairs: set[str] = set()

    # C declares semantic complete_by_source_record coverage, but the
    # physical stand-in shard order is not guaranteed to be source-group
    # contiguous. Reappearance is therefore recorded as an ordering
    # diagnostic rather than treated as an artifact-validity failure.
    completed_source_groups: set[str] = set()
    last_source_group: str | None = None
    source_group_boundary_spans = 0
    source_group_reappearance_count = 0

    raw_route_counts: Counter[str] = Counter()
    normalized_route_counts: Counter[str] = Counter()
    actual_rows = 0

    expected_candidate_columns = set(PAIR_KEY + features)
    expected_tree_columns = set(PAIR_KEY + [TREE_SCORE])

    for shard_id in sorted(c_by_id):
        c_meta = c_by_id[shard_id]
        t_meta = t_by_id[shard_id]
        r_meta = r_by_id[shard_id]

        expected_rows = c_meta["row_count"]
        if (
            t_meta["row_count"] != expected_rows
            or r_meta["row_count"] != expected_rows
        ):
            raise ValueError(
                f"C shard {shard_id} row-count disagreement "
                "across candidate/tree/route"
            )

        candidates = pd.read_parquet(c_meta["path"])
        tree = pd.read_parquet(t_meta["path"])
        routes = pd.read_parquet(r_meta["path"])

        if len(candidates) != expected_rows:
            raise ValueError(
                f"Candidate parquet row count mismatch: {c_meta['path']}"
            )
        if len(tree) != expected_rows:
            raise ValueError(
                f"Tree parquet row count mismatch: {t_meta['path']}"
            )
        if len(routes) != expected_rows:
            raise ValueError(
                f"Route parquet row count mismatch: {r_meta['path']}"
            )
        actual_rows += expected_rows

        validate_pair_keys(candidates)
        validate_pair_keys(tree)
        validate_pair_keys(routes)

        forbidden = sorted(
            set(candidates.columns) & FORBIDDEN_PRODUCTION_COLUMNS
        )
        if forbidden:
            raise ValueError(
                "C stand-in candidates contain forbidden/truth fields: "
                f"{forbidden}"
            )
        if set(candidates.columns) != expected_candidate_columns:
            missing = sorted(
                expected_candidate_columns - set(candidates.columns)
            )
            extra = sorted(
                set(candidates.columns) - expected_candidate_columns
            )
            raise ValueError(
                "C candidate shard schema mismatch: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        if set(tree.columns) != expected_tree_columns:
            raise ValueError(
                f"C tree shard schema mismatch: {t_meta['path']}"
            )

        route_col = (
            "decision_route"
            if "decision_route" in routes.columns
            else "route"
            if "route" in routes.columns
            else None
        )
        required_route = set(PAIR_KEY + [TREE_SCORE])
        if route_col is None or not required_route.issubset(routes.columns):
            raise ValueError(
                f"C route shard is missing required fields: "
                f"{r_meta['path']}"
            )

        c_keys = _key_set(candidates)
        t_keys = _key_set(tree)
        r_keys = _key_set(routes)
        if c_keys != t_keys or c_keys != r_keys:
            raise ValueError(
                f"C pair-key coverage mismatch in shard {shard_id}"
            )

        compare = tree[PAIR_KEY + [TREE_SCORE]].merge(
            routes[PAIR_KEY + [TREE_SCORE]],
            on=PAIR_KEY,
            how="inner",
            validate="1:1",
            suffixes=("_tree", "_route"),
        )
        left = pd.to_numeric(
            compare[f"{TREE_SCORE}_tree"], errors="raise"
        ).to_numpy(float)
        right = pd.to_numeric(
            compare[f"{TREE_SCORE}_route"], errors="raise"
        ).to_numpy(float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(
                f"Non-finite C tree scores in shard {shard_id}"
            )
        if (
            (left < 0).any()
            or (left > 1).any()
            or (right < 0).any()
            or (right > 1).any()
        ):
            raise ValueError(
                f"C tree scores outside [0,1] in shard {shard_id}"
            )
        if not np.isclose(left, right, rtol=0.0, atol=1e-12).all():
            raise ValueError(
                f"C tree/route score mismatch in shard {shard_id}"
            )

        raw_routes = routes[route_col].astype(str).str.strip().str.lower()
        raw_route_counts.update(raw_routes.tolist())
        normalized_route_counts.update(
            _normalize_routes(routes[route_col]).tolist()
        )

        for token in _key_tokens(candidates):
            if token in seen_pairs:
                raise ValueError(
                    "Duplicate canonical pair key across C shards"
                )
            seen_pairs.add(token)

        group_tokens = list(_group_tokens(candidates))
        if group_tokens:
            # Legitimate case:
            #   shard N ends with source-record X
            #   shard N+1 starts with source-record X
            if (
                last_source_group is not None
                and group_tokens[0] == last_source_group
            ):
                source_group_boundary_spans += 1

            for token in group_tokens:
                if token == last_source_group:
                    continue

                if last_source_group is not None:
                    completed_source_groups.add(last_source_group)

                if token in completed_source_groups:
                    source_group_reappearance_count += 1

                last_source_group = token

    expected_total = int(candidate_manifest["total_rows"])
    if actual_rows != expected_total:
        raise ValueError(
            f"C actual total rows {actual_rows} != {expected_total}"
        )

    declared_route_counts = {
        str(k).strip().lower(): int(v)
        for k, v in (
            route_manifest.get("route_distribution") or {}
        ).items()
    }
    if declared_route_counts and dict(raw_route_counts) != declared_route_counts:
        raise ValueError(
            "C actual route distribution differs from route manifest: "
            f"actual={dict(raw_route_counts)} "
            f"declared={declared_route_counts}"
        )

    return {
        "is_production_standin": True,
        "production_ready": False,
        "integration_rehearsal_eligible": True,
        "total_rows": actual_rows,
        "shard_count": len(c_by_id),
        "pair_key": list(PAIR_KEY),
        "duplicate_pair_keys": 0,
        "source_group_split_across_shards": source_group_boundary_spans,
        "source_group_reappearance_count": source_group_reappearance_count,
        "source_group_streaming_compatible": (
            source_group_reappearance_count == 0
        ),
        "declared_complete_by_source_record": True,
        "pair_universe_alignment_valid": True,
        "route_distribution_raw": dict(raw_route_counts),
        "route_distribution_normalized": dict(
            normalized_route_counts
        ),
        "feature_provenance": provenance,
        "candidate_manifest_sha256": sha256_file(
            candidate_manifest_path
        ),
        "tree_manifest_sha256": sha256_file(tree_manifest_path),
        "route_manifest_sha256": sha256_file(route_manifest_path),
        "sidecar_and_shard_checksums_verified": True,
        "warning": candidate_manifest.get("warning"),
    }


def _b_status(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    report = _load_json(p)
    retrieval = report.get("retrieval") or {}
    return {
        "report_path": str(p),
        "retrieval_version": retrieval.get("version"),
        "train_union_complete": bool(
            (retrieval.get("train_union") or {}).get("complete")
        ),
        "test_union_complete": bool(
            (retrieval.get("test_union") or {}).get("complete")
        ),
        "ownership_production_eligible": bool(
            (report.get("ownership") or {}).get("production_eligible")
        ),
    }


def audit_c_handoff(
    *,
    threshold_gate_path: str | Path,
    threshold_route_path: str | Path,
    threshold_provenance_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_dir: str | Path,
    tree_manifest_path: str | Path,
    tree_dir: str | Path,
    route_manifest_path: str | Path,
    route_dir: str | Path,
    output_dir: str | Path,
    b_handoff_report_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    threshold_validation = validate_gate_route(
        threshold_gate_path, threshold_route_path
    )
    normalized_threshold_provenance = normalize_tree_provenance(
        threshold_provenance_path,
        threshold_gate_path,
        out / "threshold_tree_provenance.normalized.json",
    )

    standin = _validate_sharded_standin(
        candidate_manifest_path=candidate_manifest_path,
        candidate_dir=candidate_dir,
        tree_manifest_path=tree_manifest_path,
        tree_dir=tree_dir,
        route_manifest_path=route_manifest_path,
        route_dir=route_dir,
        threshold_provenance_path=threshold_provenance_path,
    )

    b_status = _b_status(b_handoff_report_path)
    reverse_v1_now_available = bool(
        b_status
        and b_status.get("retrieval_version") == "reverse-union-v1"
        and b_status.get("train_union_complete")
        and b_status.get("test_union_complete")
    )

    blockers = [
        "C final artifacts are explicitly marked is_production_standin=true.",
        "C stand-in source is historical final_pairs, not the real "
        "B reverse-union-v1 production candidate universe.",
    ]
    if reverse_v1_now_available:
        blockers.append(
            "B reverse-union-v1 train/test manifests are now complete; "
            "C must retrain/check the tree on reverse-v1 training candidates "
            "and score/route the reverse-v1 test universe."
        )
    blockers.append(
        "A production neural inference is still required for the primary "
        "neural_fusion route; otherwise D must use the separately frozen "
        "tree-only fallback."
    )

    report = {
        "artifact_kind": "member_d_c_handoff_audit",
        "version": "c_threshold_v2_plus_final_standin_v1",
        "completion_state": "complete",
        "threshold": {
            "selection_eligible": True,
            "production_ready": False,
            "validation": threshold_validation,
            "tree_model_version": normalized_threshold_provenance[
                "tree_model_version"
            ],
            "training_git_sha": normalized_threshold_provenance[
                "training_git_sha"
            ],
            "candidate_version": normalized_threshold_provenance[
                "candidate_version"
            ],
            "feature_schema_version": normalized_threshold_provenance[
                "feature_schema_version"
            ],
            "feature_columns_sha256": normalized_threshold_provenance[
                "feature_columns_sha256"
            ],
            "complete_by_source_record": True,
        },
        "final_standin": standin,
        "b_handoff": b_status,
        "member_d_status": {
            "c_threshold_v2_accepted_for_selection": True,
            "c_final_standin_accepted_for_rehearsal": True,
            "c_final_standin_accepted_for_production": False,
            "c_reverse_v1_rebuild_required": reverse_v1_now_available,
            "production_ready": False,
            "production_blockers": blockers,
        },
    }
    report_path = out / "c_handoff_audit.json"
    _write_json(report, report_path)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold-gate", required=True)
    p.add_argument("--threshold-route", required=True)
    p.add_argument("--threshold-provenance", required=True)
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--candidate-dir", required=True)
    p.add_argument("--tree-manifest", required=True)
    p.add_argument("--tree-dir", required=True)
    p.add_argument("--route-manifest", required=True)
    p.add_argument("--route-dir", required=True)
    p.add_argument("--b-handoff-report")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    result = audit_c_handoff(
        threshold_gate_path=args.threshold_gate,
        threshold_route_path=args.threshold_route,
        threshold_provenance_path=args.threshold_provenance,
        candidate_manifest_path=args.candidate_manifest,
        candidate_dir=args.candidate_dir,
        tree_manifest_path=args.tree_manifest,
        tree_dir=args.tree_dir,
        route_manifest_path=args.route_manifest,
        route_dir=args.route_dir,
        b_handoff_report_path=args.b_handoff_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
