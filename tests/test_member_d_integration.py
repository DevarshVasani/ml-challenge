import json

import pandas as pd
import pytest

from src.decoder import DecoderA, expected_f05_for_set
from src.member_d_contracts import (
    PAIR_KEY,
    feature_columns_sha256,
    validate_feature_allowlist,
    validate_joined_scores,
)
from src.ownership import apply_best_owner
from src.score_join import join_scores
from src.source_group_io import iter_complete_source_groups


def pair(s1, cid, src, **kw):
    row = {
        "source1_entity_id": s1,
        "candidate_entity_id": cid,
        "candidate_source": src,
    }
    row.update(kw)
    return row


def test_source_qualified_pair_key_and_empty_prediction():
    frame = pd.DataFrame(
        [
            pair(
                "S1-1",
                "S2-123",
                "S2",
                tree_score=.8,
                neural_requested=0,
                has_neural_score=0,
                decision_route="tree_only",
            ),
            pair(
                "S1-1",
                "S3-123",
                "S3",
                tree_score=.7,
                neural_requested=0,
                has_neural_score=0,
                decision_route="tree_only",
            ),
        ]
    )
    validate_joined_scores(frame)
    assert not frame.duplicated(PAIR_KEY).any()
    frame["match_probability"] = [.2, .3]
    assert (
        DecoderA()
        .decode(frame, .9, ["S1-1", "S1-2"])["S1-2"]
        == []
    )


def test_neural_requested_missing_fails():
    frame = pd.DataFrame(
        [
            pair(
                "S1-1",
                "S2-1",
                "S2",
                tree_score=.8,
                neural_requested=1,
                has_neural_score=0,
                decision_route="neural_fusion",
            )
        ]
    )
    with pytest.raises(ValueError, match="missing neural"):
        validate_joined_scores(frame)


def test_tree_only_may_carry_precomputed_neural_score():
    frame = pd.DataFrame(
        [
            pair(
                "S1-1",
                "S2-1",
                "S2",
                tree_score=.8,
                neural_probability=.9,
                neural_requested=0,
                has_neural_score=1,
                decision_route="tree_only",
            )
        ]
    )
    validate_joined_scores(frame)


def test_source_group_spans_files_and_reappearance_fails(
    tmp_path,
):
    first = pd.DataFrame(
        [pair("S1-1", "S2-9", "S2", retrieval_score=.9)]
    )
    second = pd.DataFrame(
        [
            pair("S1-2", "S2-9", "S2", retrieval_score=.8),
            pair("S1-1", "S3-1", "S3", retrieval_score=.7),
        ]
    )
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    first.to_parquet(p1)
    second.to_parquet(p2)

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "completion": "complete",
                "complete_by_source_record": True,
                "shard_order": [p1.name, p2.name],
            }
        )
    )
    groups = list(
        iter_complete_source_groups(
            [p1, p2], manifest_paths=[manifest]
        )
    )
    assert len(groups[0]) == 2

    third = pd.DataFrame(
        [pair("S1-3", "S2-9", "S2", retrieval_score=.1)]
    )
    p3 = tmp_path / "c.parquet"
    third.to_parquet(p3)
    manifest.write_text(
        json.dumps(
            {
                "completion": "complete",
                "complete_by_source_record": True,
                "shard_order": [p1.name, p2.name, p3.name],
            }
        )
    )
    with pytest.raises(ValueError, match="reappeared"):
        list(
            iter_complete_source_groups(
                [p1, p2, p3],
                manifest_paths=[manifest],
            )
        )


def test_missing_declared_shard_fails(tmp_path):
    p = tmp_path / "a.parquet"
    pd.DataFrame(
        [pair("S1-1", "S2-1", "S2")]
    ).to_parquet(p)
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            {
                "completion": "complete",
                "complete_by_source_record": True,
                "shard_order": [
                    p.name,
                    "missing.parquet",
                ],
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        list(
            iter_complete_source_groups(
                [p], manifest_paths=[m]
            )
        )


def test_best_owner_tie_and_many_records_per_s1():
    frame = pd.DataFrame(
        [
            pair(
                "S1-B",
                "S2-1",
                "S2",
                tree_score=.8,
                match_probability=.7,
                source_group_complete=True,
            ),
            pair(
                "S1-A",
                "S2-1",
                "S2",
                tree_score=.8,
                match_probability=.7,
                source_group_complete=True,
            ),
            pair(
                "S1-A",
                "S2-2",
                "S2",
                tree_score=.9,
                match_probability=.9,
                source_group_complete=True,
            ),
        ]
    )
    out = apply_best_owner(frame, threshold=.5)
    selected = out[out.ownership_selected]
    assert (
        selected[
            selected.candidate_entity_id == "S2-1"
        ].iloc[0].source1_entity_id
        == "S1-A"
    )
    assert len(
        selected[selected.source1_entity_id == "S1-A"]
    ) == 2


def test_feature_leak_contract():
    for bad in (
        "is_injected_positive",
        "positive_injected_for_training",
        "member_d_partition",
        "truth_count",
    ):
        with pytest.raises(ValueError):
            validate_feature_allowlist(["tree_score", bad])
    validate_feature_allowlist(
        ["tree_score", "competing_s1_count"]
    )


def _tree_manifest(path):
    features = ["name_raw_similarity"]
    path.write_text(
        json.dumps(
            {
                "tree_model_version": "tree_v1",
                "training_git_sha": "abc",
                "feature_columns": features,
                "feature_columns_sha256": feature_columns_sha256(
                    features
                ),
                "feature_schema_version": "pair-features-v1",
                "feature_contract_valid": True,
            }
        )
    )


def test_score_join_rejects_extra_tree_key_and_checks_provenance(
    tmp_path,
):
    candidates = pd.DataFrame(
        [
            pair(
                "S1-1",
                "S2-1",
                "S2",
                retrieval_score=.9,
                competing_s1_count=1,
                neural_requested=0,
                source_group_complete=True,
            )
        ]
    )
    tree = pd.DataFrame(
        [
            pair(
                "S1-1", "S2-1", "S2", tree_score=.8
            ),
            pair(
                "S1-2", "S2-2", "S2", tree_score=.1
            ),
        ]
    )
    candidate_path = tmp_path / "c.parquet"
    tree_path = tmp_path / "t.parquet"
    candidates.to_parquet(candidate_path)
    tree.to_parquet(tree_path)

    candidate_manifest = tmp_path / "cm.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "completion_state": "complete",
                "complete_by_source_record": True,
                "candidate_version": "v1",
            }
        )
    )
    tree_manifest = tmp_path / "tm.json"
    _tree_manifest(tree_manifest)

    with pytest.raises(ValueError, match="unknown pairs"):
        join_scores(
            [candidate_path],
            [tree_path],
            output_dir=tmp_path / "o",
            candidate_manifest_path=candidate_manifest,
            tree_score_manifest_path=tree_manifest,
        )


def test_score_join_accepts_c_route_aliases(tmp_path):
    candidates = pd.DataFrame(
        [
            pair(
                "S1-1",
                "S2-1",
                "S2",
                retrieval_score=.9,
                source_group_complete=True,
            ),
            pair(
                "S1-2",
                "S3-1",
                "S3",
                retrieval_score=.7,
                source_group_complete=True,
            ),
        ]
    )
    tree = pd.DataFrame(
        [
            pair(
                "S1-1", "S2-1", "S2", tree_score=.8
            ),
            pair(
                "S1-2", "S3-1", "S3", tree_score=.2
            ),
        ]
    )
    route = pd.DataFrame(
        [
            pair("S1-1", "S2-1", "S2", route="tree_only"),
            pair("S1-2", "S3-1", "S3", route="discard"),
        ]
    )
    candidate_path = tmp_path / "c.parquet"
    tree_path = tmp_path / "t.parquet"
    route_path = tmp_path / "r.parquet"
    candidates.to_parquet(candidate_path)
    tree.to_parquet(tree_path)
    route.to_parquet(route_path)

    candidate_manifest = tmp_path / "cm.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "completion_state": "complete",
                "complete_by_source_record": True,
                "candidate_version": "v1",
            }
        )
    )
    tree_manifest = tmp_path / "tm.json"
    _tree_manifest(tree_manifest)

    result = join_scores(
        [candidate_path],
        [tree_path],
        output_dir=tmp_path / "o",
        candidate_manifest_path=candidate_manifest,
        tree_score_manifest_path=tree_manifest,
        route_table_path=route_path,
    )
    joined = pd.read_parquet(result["output_path"])
    assert joined["decision_route"].tolist() == [
        "tree_only",
        "discard",
    ]
    assert joined["neural_requested"].tolist() == [0, 0]


def test_expected_f05_uses_total_m_not_ratio_of_expectations():
    value = expected_f05_for_set(
        pd.Series([.9]).to_numpy(),
        pd.Series([.5]).to_numpy(),
    )
    assert 0 < value < 1
