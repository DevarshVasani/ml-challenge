"""Validate Member B's frozen reverse-v1 handoff for Member D.

This module deliberately separates three facts:

1. B's starter package is a bootstrap/evaluation package.  It is complete by
   S1, NOT complete by source record, so D must never use it as production
   competition groups.
2. The reverse-v1 train/test manifests certify B's frozen retrieval universe
   metadata.  If the bucket files are locally available, this validator can
   additionally verify every declared bucket checksum.
3. Ownership is enabled only after a full-training truth audit proves every
   S2/S3 record has at most one S1 owner.

Large retrieval bucket files are not required on D's machine merely to record
B provenance; C is expected to consume those buckets and return source-complete
tree/routes to D.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .member_d_contracts import PAIR_KEY, validate_pair_keys


EXPECTED_STARTER_SCHEMA = "member-b-starter-1"
EXPECTED_RETRIEVAL_VERSION = "reverse-union-v1"
EXPECTED_PRODUCTION_CHANNELS = {"lexical", "dense"}
_ENTITY_RE = re.compile(r"^(S[23])-(\d+)$")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tsv_rows(path: str | Path) -> int:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _rows(path: str | Path) -> int:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return len(pd.read_parquet(p))
    if p.suffix.lower() in {".tsv", ".txt"}:
        return _tsv_rows(p)
    raise ValueError(f"Unsupported starter artifact for row count: {p}")


def _validate_starter_file(
    *,
    logical_name: str,
    path: str | Path,
    starter: dict[str, Any],
) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    declared = starter.get("outputs", {}).get(logical_name)
    if not isinstance(declared, dict):
        raise ValueError(
            f"Starter manifest does not declare {logical_name}"
        )
    expected_sha = str(declared.get("sha256", ""))
    actual_sha = _sha256(p)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Starter checksum mismatch for {logical_name}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    expected_rows = int(declared.get("rows", -1))
    actual_rows = _rows(p)
    if actual_rows != expected_rows:
        raise ValueError(
            f"Starter row-count mismatch for {logical_name}: "
            f"expected {expected_rows}, got {actual_rows}"
        )
    return {
        "path": str(p),
        "rows": actual_rows,
        "sha256": actual_sha,
        "verified": True,
    }


def _validate_union_manifest(
    path: str | Path,
    *,
    split: str,
    expected_version: str,
    bucket_dir: str | Path | None = None,
) -> dict[str, Any]:
    p = Path(path)
    m = _load_json(p)
    if m.get("complete") is not True:
        raise ValueError(f"{split} union manifest is not complete")
    config = m.get("config") or {}
    if str(config.get("version")) != expected_version:
        raise ValueError(
            f"{split} union version mismatch: "
            f"{config.get('version')} != {expected_version}"
        )
    buckets = m.get("buckets")
    if not isinstance(buckets, dict) or len(buckets) != 8:
        raise ValueError(
            f"{split} union must declare exactly 8 buckets"
        )

    pair_sum = sum(int(v["pairs"]) for v in buckets.values())
    group_sum = sum(int(v["source_groups"]) for v in buckets.values())
    if pair_sum != int(m.get("pairs", -1)):
        raise ValueError(
            f"{split} bucket pair sum {pair_sum} "
            f"!= manifest total {m.get('pairs')}"
        )
    if group_sum != int(m.get("source_groups", -1)):
        raise ValueError(
            f"{split} bucket source-group sum {group_sum} "
            f"!= manifest total {m.get('source_groups')}"
        )

    bucket_files_verified = False
    if bucket_dir is not None:
        root = Path(bucket_dir)
        for name, meta in sorted(buckets.items()):
            bp = root / name
            if not bp.exists():
                raise FileNotFoundError(bp)
            if bp.stat().st_size != int(meta["bytes"]):
                raise ValueError(
                    f"{split} bucket size mismatch: {bp}"
                )
            actual = _sha256(bp)
            if actual != str(meta["sha256"]):
                raise ValueError(
                    f"{split} bucket checksum mismatch: {bp}"
                )
        bucket_files_verified = True

    return {
        "manifest_path": str(p),
        "manifest_sha256": _sha256(p),
        "complete": True,
        "version": expected_version,
        "bucket_count": len(buckets),
        "pairs": int(m["pairs"]),
        "source_groups": int(m["source_groups"]),
        "bucket_pair_sum": pair_sum,
        "bucket_source_group_sum": group_sum,
        "bucket_files_verified": bucket_files_verified,
        "config_hash": m.get("config_hash"),
    }


def _encode_candidate(entity_id: str) -> tuple[int, str]:
    match = _ENTITY_RE.fullmatch(entity_id)
    if not match:
        raise ValueError(
            f"Malformed matched entity id {entity_id!r}; "
            "expected S2-<digits> or S3-<digits>"
        )
    source, number = match.groups()
    # Source code in high bits keeps S2/S3 numeric IDs distinct.
    key = ((2 if source == "S2" else 3) << 40) | int(number)
    return key, source


def _decode_candidate(key: int) -> str:
    source_code = key >> 40
    number = key & ((1 << 40) - 1)
    source = "S2" if source_code == 2 else "S3"
    return f"{source}-{number}"


def audit_full_training_ownership(
    ground_truth_path: str | Path,
    *,
    first_n_examples: int = 10,
) -> dict[str, Any]:
    """Memory-bounded two-pass ownership audit for the full training truth.

    The first pass stores one compact integer per distinct S2/S3 record and
    records only duplicate keys.  A second pass is needed only when duplicate
    source records exist; it distinguishes harmless repeated edges from true
    multi-owner violations.
    """
    path = Path(ground_truth_path)
    seen: set[int] = set()
    duplicate_keys: set[int] = set()
    source_unique = {"S2": 0, "S3": 0}
    positive_edges = 0
    s1_rows = 0
    s1_rows_with_no_matches = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Full training truth must contain "
                "source1_entity_id and matched_entity_ids"
            )
        for row in reader:
            s1_rows += 1
            s1 = (row.get("source1_entity_id") or "").strip()
            if not s1:
                raise ValueError("Blank source1_entity_id in training truth")
            values = [
                value.strip()
                for value in (row.get("matched_entity_ids") or "").split(",")
                if value.strip()
            ]
            if not values:
                s1_rows_with_no_matches += 1
            for candidate_id in values:
                positive_edges += 1
                key, source = _encode_candidate(candidate_id)
                if key in seen:
                    duplicate_keys.add(key)
                else:
                    seen.add(key)
                    source_unique[source] += 1

    owner_sets: dict[int, set[str]] = {}
    if duplicate_keys:
        owner_sets = {key: set() for key in duplicate_keys}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                owner = (row.get("source1_entity_id") or "").strip()
                for candidate_id in [
                    value.strip()
                    for value in (
                        row.get("matched_entity_ids") or ""
                    ).split(",")
                    if value.strip()
                ]:
                    key, _ = _encode_candidate(candidate_id)
                    if key in owner_sets:
                        owner_sets[key].add(owner)

    violations = [
        (key, sorted(owners))
        for key, owners in owner_sets.items()
        if len(owners) > 1
    ]
    violations.sort(key=lambda item: item[0])

    return {
        "audit_scope": "full_training",
        "ground_truth_path": str(path),
        "ground_truth_sha256": _sha256(path),
        "s1_row_count": s1_rows,
        "s1_rows_with_no_matches": s1_rows_with_no_matches,
        "positive_edge_count": positive_edges,
        "source_record_count": len(seen),
        "candidate_source_counts": source_unique,
        "duplicate_source_record_count": len(duplicate_keys),
        "records_with_one_owner": len(seen) - len(violations),
        "records_with_multiple_owners": len(violations),
        "multi_owner_violation_count": len(violations),
        "example_violations": [
            {
                "candidate_source": (
                    "S2" if (key >> 40) == 2 else "S3"
                ),
                "candidate_entity_id": _decode_candidate(key),
                "owner_s1_ids": owners,
            }
            for key, owners in violations[:first_n_examples]
        ],
        "ownership_enabled": len(violations) == 0,
    }


def validate_b_handoff(
    *,
    starter_manifest_path: str | Path,
    evaluation_truth_path: str | Path,
    threshold_candidate_pairs_path: str | Path,
    threshold_queries_path: str | Path,
    threshold_records_path: str | Path,
    train_labels_path: str | Path,
    train_ground_truth_path: str | Path,
    country_truth_audit_path: str | Path,
    frozen_policy_path: str | Path,
    train_union_manifest_path: str | Path,
    test_union_manifest_path: str | Path,
    output_dir: str | Path,
    train_union_dir: str | Path | None = None,
    test_union_dir: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    starter = _load_json(starter_manifest_path)
    if starter.get("completion") != "complete":
        raise ValueError("Member-B starter manifest is incomplete")
    if starter.get("schema_version") != EXPECTED_STARTER_SCHEMA:
        raise ValueError(
            f"Unexpected starter schema: {starter.get('schema_version')}"
        )
    if list(starter.get("pair_key") or []) != list(PAIR_KEY):
        raise ValueError(
            f"Starter pair key is not D canonical key {PAIR_KEY}"
        )

    verified_starter = {}
    for logical_name, path in {
        "evaluation_truth.tsv": evaluation_truth_path,
        "threshold_candidate_pairs.parquet":
            threshold_candidate_pairs_path,
        "threshold_queries.tsv": threshold_queries_path,
        "threshold_records.parquet": threshold_records_path,
        "train_labels.parquet": train_labels_path,
    }.items():
        verified_starter[logical_name] = _validate_starter_file(
            logical_name=logical_name,
            path=path,
            starter=starter,
        )

    # Pair-key validation on the small starter pair/label files.
    threshold_pairs = pd.read_parquet(threshold_candidate_pairs_path)
    validate_pair_keys(threshold_pairs)
    labels = pd.read_parquet(train_labels_path)
    validate_pair_keys(labels)

    starter_train_complete = bool(
        starter.get("outputs", {})
        .get("train_sample", {})
        .get("complete_by_source_record")
    )
    starter_threshold_complete = bool(
        starter.get("outputs", {})
        .get("threshold_sample", {})
        .get("complete_by_source_record")
    )
    if starter_train_complete or starter_threshold_complete:
        raise ValueError(
            "Unexpected starter contract: bootstrap samples are expected "
            "to be incomplete by source record"
        )

    policy = _load_json(frozen_policy_path)
    retrieval_version = str(policy.get("version", ""))
    if retrieval_version != EXPECTED_RETRIEVAL_VERSION:
        raise ValueError(
            f"Unexpected frozen retrieval version {retrieval_version!r}"
        )
    channels = set(map(str, policy.get("production_channels") or []))
    if channels != EXPECTED_PRODUCTION_CHANNELS:
        raise ValueError(
            f"Unexpected production channels: {sorted(channels)}"
        )

    train_union = _validate_union_manifest(
        train_union_manifest_path,
        split="train",
        expected_version=retrieval_version,
        bucket_dir=train_union_dir,
    )
    test_union = _validate_union_manifest(
        test_union_manifest_path,
        split="test",
        expected_version=retrieval_version,
        bucket_dir=test_union_dir,
    )

    country = _load_json(country_truth_audit_path)
    if int(country.get("country_mismatches", -1)) != 0:
        raise ValueError("B country truth audit has mismatches")
    if int(country.get("source_rows_mapped", -1)) != int(
        train_union["source_groups"]
    ):
        raise ValueError(
            "B country audit source_rows_mapped does not match "
            "train_union source_groups"
        )

    ownership = audit_full_training_ownership(train_ground_truth_path)
    ownership_path = out / "full_training_ownership_audit.json"
    ownership_path.write_text(
        json.dumps(ownership, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "artifact_kind": "member_d_b_handoff_validation",
        "version": "b_handoff_reverse_union_v1",
        "completion_state": "complete",
        "pair_key": list(PAIR_KEY),
        "starter": {
            "schema_version": starter["schema_version"],
            "manifest_sha256": _sha256(starter_manifest_path),
            "verified_files": verified_starter,
            "complete_by_source_record": False,
            "production_competition_eligible": False,
            "role": "bootstrap/evaluation only",
        },
        "retrieval": {
            "frozen_policy_sha256": _sha256(frozen_policy_path),
            "version": retrieval_version,
            "production_channels": sorted(channels),
            "train_union": train_union,
            "test_union": test_union,
        },
        "country_truth_audit": {
            **country,
            "sha256": _sha256(country_truth_audit_path),
            "validated_against_train_union": True,
        },
        "ownership": {
            **ownership,
            "audit_path": str(ownership_path),
            "production_eligible": bool(
                ownership["ownership_enabled"]
            ),
        },
        "member_d_status": {
            "b_retrieval_manifest_received": True,
            "b_train_union_complete": True,
            "b_test_union_complete": True,
            "b_ownership_audit_complete": True,
            "waiting_for_c_production_tree_routes": True,
            "waiting_for_a_production_neural_if_primary_route": True,
        },
    }
    report_path = out / "b_handoff_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--starter-manifest", required=True)
    p.add_argument("--evaluation-truth", required=True)
    p.add_argument("--threshold-candidate-pairs", required=True)
    p.add_argument("--threshold-queries", required=True)
    p.add_argument("--threshold-records", required=True)
    p.add_argument("--train-labels", required=True)
    p.add_argument("--train-ground-truth", required=True)
    p.add_argument("--country-truth-audit", required=True)
    p.add_argument("--frozen-policy", required=True)
    p.add_argument("--train-union-manifest", required=True)
    p.add_argument("--test-union-manifest", required=True)
    p.add_argument("--train-union-dir")
    p.add_argument("--test-union-dir")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    result = validate_b_handoff(
        starter_manifest_path=args.starter_manifest,
        evaluation_truth_path=args.evaluation_truth,
        threshold_candidate_pairs_path=args.threshold_candidate_pairs,
        threshold_queries_path=args.threshold_queries,
        threshold_records_path=args.threshold_records,
        train_labels_path=args.train_labels,
        train_ground_truth_path=args.train_ground_truth,
        country_truth_audit_path=args.country_truth_audit,
        frozen_policy_path=args.frozen_policy,
        train_union_manifest_path=args.train_union_manifest,
        test_union_manifest_path=args.test_union_manifest,
        train_union_dir=args.train_union_dir,
        test_union_dir=args.test_union_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
