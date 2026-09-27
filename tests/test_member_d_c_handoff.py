import hashlib
import json

import pandas as pd
import pytest

from src.member_d_c_handoff import (
    legacy_comma_feature_columns_sha256,
    normalize_tree_provenance,
    validate_gate_route,
)
from src.member_d_contracts import feature_columns_sha256


def _write_tsv(frame, path):
    frame.to_csv(path, sep="\t", index=False)


def _base_gate():
    return pd.DataFrame(
        {
            "candidate_source": ["S2", "S3"],
            "candidate_entity_id": ["S2-A", "S3-B"],
            "source1_entity_id": ["S1-X", "S1-X"],
            "name_test_feature": [0.9, 0.2],
            "address_test_feature": [0.8, 0.1],
            "tree_score": [0.95, 0.05],
            "gate_rank": [1, 2],
            "gate_score_gap": [0.90, 0.90],
            "route": ["neural", "discard"],
        }
    )


def _base_route():
    return pd.DataFrame(
        {
            "candidate_source": ["S2", "S3"],
            "candidate_entity_id": ["S2-A", "S3-B"],
            "source1_entity_id": ["S1-X", "S1-X"],
            "tree_score": [0.95, 0.05],
            "gate_rank": [1, 2],
            "gate_score_gap": [0.90, 0.90],
            "route": ["neural", "discard"],
        }
    )


def test_c_legacy_feature_hash_is_normalized_without_changing_features(
    tmp_path,
):
    features = ["name_test_feature", "address_test_feature"]
    source_hash = legacy_comma_feature_columns_sha256(features)
    assert source_hash == hashlib.sha256(
        ",".join(features).encode("utf-8")
    ).hexdigest()

    provenance = {
        "tree_model_version": "v5",
        "training_git_sha": "abc123",
        "feature_columns": features,
        "feature_columns_sha256": source_hash,
        "feature_schema_version": "test-v1",
        "candidate_version": "candidate-v1",
        "feature_contract_valid": True,
        "complete_by_source_record": True,
    }

    gate_path = tmp_path / "gate.tsv"
    prov_path = tmp_path / "prov.json"
    out_path = tmp_path / "normalized.json"
    _write_tsv(_base_gate(), gate_path)
    prov_path.write_text(json.dumps(provenance), encoding="utf-8")

    normalized = normalize_tree_provenance(
        prov_path, gate_path, out_path
    )

    assert normalized["feature_columns"] == features
    assert normalized["source_feature_columns_sha256"] == source_hash
    assert normalized["source_feature_hash_scheme"] == (
        "member_c_comma_join_v1"
    )
    assert normalized["feature_columns_sha256"] == (
        feature_columns_sha256(features)
    )
    assert normalized["member_d_normalization"][
        "hash_serialization_only"
    ] is True


def test_c_provenance_rejects_unknown_feature_hash(tmp_path):
    gate_path = tmp_path / "gate.tsv"
    prov_path = tmp_path / "prov.json"
    out_path = tmp_path / "normalized.json"
    _write_tsv(_base_gate(), gate_path)

    provenance = {
        "tree_model_version": "v5",
        "training_git_sha": "abc123",
        "feature_columns": [
            "name_test_feature",
            "address_test_feature",
        ],
        "feature_columns_sha256": "0" * 64,
        "feature_schema_version": "test-v1",
        "candidate_version": "candidate-v1",
        "feature_contract_valid": True,
        "complete_by_source_record": True,
    }
    prov_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="matches neither"):
        normalize_tree_provenance(
            prov_path, gate_path, out_path
        )


def test_c_gate_route_requires_exact_key_and_value_coverage(tmp_path):
    gate_path = tmp_path / "gate.tsv"
    route_path = tmp_path / "route.tsv"
    _write_tsv(_base_gate(), gate_path)
    _write_tsv(_base_route(), route_path)

    report = validate_gate_route(gate_path, route_path)
    assert report["row_count"] == 2
    assert report["missing_route_keys"] == 0
    assert report["extra_route_keys"] == 0
    assert report["tree_score_mismatch_count"] == 0
    assert report["route_mismatch_count"] == 0
    assert report["route_counts"] == {
        "neural_fusion": 1,
        "discard": 1,
    }


def test_c_gate_route_rejects_route_mismatch(tmp_path):
    gate_path = tmp_path / "gate.tsv"
    route_path = tmp_path / "route.tsv"
    gate = _base_gate()
    route = _base_route()
    route.loc[0, "route"] = "discard"
    _write_tsv(gate, gate_path)
    _write_tsv(route, route_path)

    with pytest.raises(ValueError, match="route mismatch"):
        validate_gate_route(gate_path, route_path)
