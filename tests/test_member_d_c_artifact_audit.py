import json
from pathlib import Path

import pandas as pd
import pytest

from src.member_d_c_artifact_audit import audit_c_handoff
from src.member_d_c_handoff import legacy_comma_feature_columns_sha256
from src.neural_contracts import sha256_file


PAIR = ["candidate_source", "candidate_entity_id", "source1_entity_id"]


def _write_sidecar(path, kind, shard_id, rows):
    payload = {
        "shard_id": shard_id,
        "kind": kind,
        "row_count": rows,
        "sha256": sha256_file(path),
        "status": "complete",
    }
    Path(f"{path}.complete.json").write_text(json.dumps(payload))


def _write_parent(path, directory_name, rows, *, extra=None):
    shards = []
    for shard_id, count in rows:
        shards.append(
            {
                "path": (
                    f"artifacts/gate/{directory_name}/"
                    f"shard-{shard_id}.parquet"
                ),
                "row_count": count,
            }
        )
    payload = {
        "is_production_standin": True,
        "total_rows": sum(count for _, count in rows),
        "shard_count": len(shards),
        "shards": shards,
        "completion": "complete",
        "warning": "PRODUCTION STAND-IN, NOT REAL PRODUCTION DATA.",
    }
    payload.update(extra or {})
    path.write_text(json.dumps(payload))


def _build(tmp_path, *, split_group=False):
    features = ["name_raw_similarity"]
    legacy_hash = legacy_comma_feature_columns_sha256(features)
    candidate_version = "candidate-v1"

    gate = pd.DataFrame(
        [
            {
                "candidate_source": "S2",
                "candidate_entity_id": "S2-10",
                "source1_entity_id": "S1-1",
                "name_raw_similarity": 0.9,
                "tree_score": 0.9,
                "gate_rank": 1,
                "gate_score_gap": 0.8,
                "route": "neural",
            },
            {
                "candidate_source": "S3",
                "candidate_entity_id": "S3-20",
                "source1_entity_id": "S1-2",
                "name_raw_similarity": 0.1,
                "tree_score": 0.1,
                "gate_rank": 1,
                "gate_score_gap": 0.0,
                "route": "discard",
            },
        ]
    )
    route = gate[
        PAIR + ["tree_score", "gate_rank", "gate_score_gap", "route"]
    ].copy()
    gate_path = tmp_path / "threshold_gate_v2.parquet"
    route_path = tmp_path / "threshold_route_table_v2.parquet"
    gate.to_parquet(gate_path, index=False)
    route.to_parquet(route_path, index=False)

    provenance = {
        "tree_model_version": "v5",
        "training_git_sha": "abc123",
        "feature_columns": features,
        "feature_columns_sha256": legacy_hash,
        "feature_schema_version": "c-string-features-v2",
        "candidate_version": candidate_version,
        "feature_contract_valid": True,
        "complete_by_source_record": True,
    }
    provenance_path = tmp_path / "threshold_tree_provenance_v2.json"
    provenance_path.write_text(json.dumps(provenance))

    cdir = tmp_path / "production_candidates_v1-final-standin"
    tdir = tmp_path / "production_tree_scores_v1-final-standin"
    rdir = tmp_path / "production_routes_v1-final-standin"
    cdir.mkdir()
    tdir.mkdir()
    rdir.mkdir()

    if split_group:
        candidate_ids = ["S2-10", "S2-10"]
    else:
        candidate_ids = ["S2-10", "S3-20"]

    for idx in range(2):
        sid = f"{idx:04d}"
        candidate = pd.DataFrame(
            [
                {
                    "candidate_source": candidate_ids[idx].split("-")[0],
                    "candidate_entity_id": candidate_ids[idx],
                    "source1_entity_id": f"S1-{idx + 1}",
                    "name_raw_similarity": 0.9 - idx * 0.8,
                }
            ]
        )
        score = 0.9 - idx * 0.8
        tree = candidate[PAIR].copy()
        tree["tree_score"] = score
        routes = tree.copy()
        routes["gate_rank"] = 1
        routes["gate_score_gap"] = 0.8 if idx == 0 else 0.0
        routes["route"] = "neural" if idx == 0 else "discard"

        cp = cdir / f"shard-{sid}.parquet"
        tp = tdir / f"shard-{sid}.parquet"
        rp = rdir / f"shard-{sid}.parquet"
        candidate.to_parquet(cp, index=False)
        tree.to_parquet(tp, index=False)
        routes.to_parquet(rp, index=False)
        _write_sidecar(cp, "candidates", sid, 1)
        _write_sidecar(tp, "tree_scores", sid, 1)
        _write_sidecar(rp, "routes", sid, 1)

    rows = [("0000", 1), ("0001", 1)]
    cm = tmp_path / "production_candidate_manifest.json"
    tm = tmp_path / "production_tree_manifest.json"
    rm = tmp_path / "production_route_manifest.json"
    _write_parent(
        cm,
        cdir.name,
        rows,
        extra={
            "standin_source_split": "final_pairs",
            "candidate_version": candidate_version,
            "pair_key": PAIR,
            "feature_columns": features,
        },
    )
    _write_parent(
        tm,
        tdir.name,
        rows,
        extra={
            "tree_model_version": "v5",
            "training_git_sha": "abc123",
            "feature_columns_sha256": legacy_hash,
        },
    )
    _write_parent(
        rm,
        rdir.name,
        rows,
        extra={
            "route_distribution": {"neural": 1, "discard": 1},
            "complete_by_source_record": True,
        },
    )

    return {
        "gate": gate_path,
        "threshold_route": route_path,
        "provenance": provenance_path,
        "cm": cm,
        "cdir": cdir,
        "tm": tm,
        "tdir": tdir,
        "rm": rm,
        "rdir": rdir,
    }


def test_c_standin_is_validated_but_never_promoted(tmp_path):
    p = _build(tmp_path)
    result = audit_c_handoff(
        threshold_gate_path=p["gate"],
        threshold_route_path=p["threshold_route"],
        threshold_provenance_path=p["provenance"],
        candidate_manifest_path=p["cm"],
        candidate_dir=p["cdir"],
        tree_manifest_path=p["tm"],
        tree_dir=p["tdir"],
        route_manifest_path=p["rm"],
        route_dir=p["rdir"],
        output_dir=tmp_path / "out",
    )
    assert result["threshold"]["selection_eligible"] is True
    assert result["final_standin"]["integration_rehearsal_eligible"] is True
    assert result["final_standin"]["production_ready"] is False
    assert result["final_standin"]["total_rows"] == 2
    assert result["member_d_status"][
        "c_final_standin_accepted_for_production"
    ] is False


def test_c_standin_allows_contiguous_source_group_across_shard_boundary(
    tmp_path,
):
    p = _build(tmp_path, split_group=True)
    result = audit_c_handoff(
        threshold_gate_path=p["gate"],
        threshold_route_path=p["threshold_route"],
        threshold_provenance_path=p["provenance"],
        candidate_manifest_path=p["cm"],
        candidate_dir=p["cdir"],
        tree_manifest_path=p["tm"],
        tree_dir=p["tdir"],
        route_manifest_path=p["rm"],
        route_dir=p["rdir"],
        output_dir=tmp_path / "out",
    )

    assert (
        result["final_standin"]["source_group_split_across_shards"]
        == 1
    )
    assert (
        result["final_standin"]["source_group_reappearance_count"]
        == 0
    )
