import json

import pandas as pd
import pytest

from src.member_d_b_handoff import (
    _validate_union_manifest,
    audit_full_training_ownership,
)


def test_full_training_ownership_audit_enables_unique_owner(tmp_path):
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-10,S3-20\n"
        "S1-2\tS2-11\n",
        encoding="utf-8",
    )
    audit = audit_full_training_ownership(truth)
    assert audit["audit_scope"] == "full_training"
    assert audit["positive_edge_count"] == 3
    assert audit["source_record_count"] == 3
    assert audit["multi_owner_violation_count"] == 0
    assert audit["ownership_enabled"] is True


def test_full_training_ownership_audit_detects_multi_owner(tmp_path):
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-10\n"
        "S1-2\tS2-10\n",
        encoding="utf-8",
    )
    audit = audit_full_training_ownership(truth)
    assert audit["source_record_count"] == 1
    assert audit["multi_owner_violation_count"] == 1
    assert audit["ownership_enabled"] is False
    assert audit["example_violations"][0]["candidate_entity_id"] == "S2-10"


def test_union_manifest_requires_exact_bucket_sums(tmp_path):
    buckets = {
        f"bucket-{i:02d}.parquet": {
            "bytes": 1,
            "pairs": 10,
            "sha256": "0" * 64,
            "source_groups": 2,
        }
        for i in range(8)
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "config": {"version": "reverse-union-v1"},
                "config_hash": "x",
                "pairs": 80,
                "source_groups": 16,
                "buckets": buckets,
            }
        ),
        encoding="utf-8",
    )
    out = _validate_union_manifest(
        manifest,
        split="train",
        expected_version="reverse-union-v1",
    )
    assert out["bucket_count"] == 8
    assert out["pairs"] == 80
    assert out["source_groups"] == 16

    payload = json.loads(manifest.read_text())
    payload["pairs"] = 79
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pair sum"):
        _validate_union_manifest(
            manifest,
            split="train",
            expected_version="reverse-union-v1",
        )
