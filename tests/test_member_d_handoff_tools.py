import json

import pandas as pd
import pytest

from src.member_d_gate_audit import audit_gate
from src.member_d_labels import build_keyed_labels
from src.member_d_rehearsal import build_rehearsal_join


def _write_tsv(frame, path):
    frame.to_csv(path, sep="\t", index=False)


def test_gate_audit_scopes_truth_to_threshold_manifest(tmp_path):
    queries = pd.DataFrame(
        {
            "source1_entity_id": ["S1-T"],
            "split": ["threshold"],
            "country": ["US"],
        }
    )
    truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-T", "S1-F"],
            "split": ["threshold", "final"],
            "true_match_count": ["1", "1"],
            "matched_entity_ids": ["S2-G", "S2-FINAL"],
        }
    )
    gate = pd.DataFrame(
        {
            "candidate_source": ["S2"],
            "candidate_entity_id": ["S2-G"],
            "source1_entity_id": ["S1-T"],
            "tree_score": [0.5],
            "route": ["neural"],
        }
    )

    qp = tmp_path / "queries.tsv"
    tp = tmp_path / "truth.tsv"
    gp = tmp_path / "gate.tsv"
    _write_tsv(queries, qp)
    _write_tsv(truth, tp)
    _write_tsv(gate, gp)

    report = audit_gate(gp, qp, tp, cutoffs=[0.001])
    assert report["scope"]["query_count"] == 1
    assert report["gate"]["post_gate_oracle"]["gold_edge_count"] == 1
    assert report["gate"]["post_gate_oracle"]["micro_oracle_recall"] == 1.0


def test_gate_audit_rejects_leaked_field(tmp_path):
    queries = pd.DataFrame(
        {"source1_entity_id": ["S1-T"], "split": ["threshold"]}
    )
    truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-T"],
            "matched_entity_ids": ["S2-G"],
        }
    )
    gate = pd.DataFrame(
        {
            "candidate_source": ["S2"],
            "candidate_entity_id": ["S2-G"],
            "source1_entity_id": ["S1-T"],
            "tree_score": [0.5],
            "route": ["neural"],
            "is_injected_positive": [0],
        }
    )
    qp = tmp_path / "q.tsv"
    tp = tmp_path / "t.tsv"
    gp = tmp_path / "g.tsv"
    _write_tsv(queries, qp)
    _write_tsv(truth, tp)
    _write_tsv(gate, gp)

    with pytest.raises(ValueError, match="forbidden"):
        audit_gate(gp, qp, tp)


def test_keyed_labels_stay_separate(tmp_path):
    queries = pd.DataFrame(
        {"source1_entity_id": ["S1-T"], "split": ["threshold"]}
    )
    truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-T"],
            "matched_entity_ids": ["S2-G"],
        }
    )
    pairs = pd.DataFrame(
        [
            {
                "candidate_source": "S2",
                "candidate_entity_id": "S2-G",
                "source1_entity_id": "S1-T",
            },
            {
                "candidate_source": "S3",
                "candidate_entity_id": "S3-N",
                "source1_entity_id": "S1-T",
            },
        ]
    )
    qp = tmp_path / "q.tsv"
    tp = tmp_path / "t.tsv"
    pp = tmp_path / "pairs.tsv"
    out = tmp_path / "labels.tsv"
    _write_tsv(queries, qp)
    _write_tsv(truth, tp)
    _write_tsv(pairs, pp)

    manifest = build_keyed_labels(pp, qp, tp, out)
    labels = pd.read_csv(out, sep="\t")
    assert labels["label"].tolist() == [1, 0]
    assert set(labels.columns) == {
        "candidate_source",
        "candidate_entity_id",
        "source1_entity_id",
        "label",
    }
    assert manifest["positive_count"] == 1


def test_rehearsal_drops_incomplete_competition_features(tmp_path):
    gate = pd.DataFrame(
        {
            "candidate_source": ["S2", "S3"],
            "candidate_entity_id": ["S2-G", "S3-N"],
            "source1_entity_id": ["S1-T", "S1-T"],
            "tree_score": [0.8, 0.1],
            "route": ["neural", "discard"],
            "competing_s1_count": [2, 3],
            "competing_s1_score_gap": [0.2, 0.3],
            "competing_best_second_gap": [0.1, 0.2],
            "competing_near_tie_count": [1, 0],
            "is_top_competing_s1": [1, 0],
        }
    )
    neural = pd.DataFrame(
        {
            "candidate_source": ["S2", "S3"],
            "candidate_entity_id": ["S2-G", "S3-N"],
            "source1_entity_id": ["S1-T", "S1-T"],
            "neural_logit": [2.0, -1.0],
        }
    )
    gp = tmp_path / "gate.tsv"
    npth = tmp_path / "neural.tsv"
    out = tmp_path / "joined.tsv"
    _write_tsv(gate, gp)
    _write_tsv(neural, npth)

    manifest = build_rehearsal_join(gp, [npth], out)
    joined = pd.read_csv(out, sep="\t")
    for column in (
        "competing_s1_count",
        "competing_s1_score_gap",
        "competing_best_second_gap",
        "competing_near_tie_count",
        "is_top_competing_s1",
    ):
        assert column not in joined.columns
    assert joined["decision_route"].tolist() == [
        "neural_fusion",
        "discard",
    ]
    assert joined["source_group_complete"].eq(False).all()
    assert manifest["selection_eligible"] is False
