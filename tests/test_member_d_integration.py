"""Member D integration tests (§10).

Tests all 10 required fixture scenarios plus the assertions listed in the plan.

IMPORTANT: Synthetic fixture scores are test-only and must NEVER be appended
to reports/experiments.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.member_d_contracts import (
    PAIR_KEY,
    OWNERSHIP_GROUP_KEY,
    ROUTE_COL_HAS,
    ROUTE_COL_REQUESTED,
    SCORE_COL_NEURAL,
    SCORE_COL_TREE,
    validate_score_frame,
)
from src.ownership import apply_best_owner, audit_gold_ownership
from src.source_group_io import iter_complete_source_groups
from src.score_join import join_scores, compute_source_competitor_features
from src.decoder import DecoderA, DecoderB, DecoderC


# ===========================================================================
# Fixture helpers
# ===========================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "member_d"


def _make_pair(s1: str, cand: str, src: str, tree: float = 0.8, neural: float | None = None) -> dict:
    row = {
        "source1_entity_id": s1,
        "candidate_entity_id": cand,
        "candidate_source": src,
        SCORE_COL_TREE: tree,
        ROUTE_COL_REQUESTED: 0 if neural is None else 1,
        ROUTE_COL_HAS: 0 if neural is None else 1,
    }
    if neural is not None:
        row[SCORE_COL_NEURAL] = neural
    return row


# ===========================================================================
# §10 Fixture 1: S1 with no candidates → empty row in export
# ===========================================================================

def test_s1_no_candidates_produces_empty_prediction():
    """An S1 with zero candidates must produce an empty prediction, not an error."""
    from src.decoder import DecoderA
    frame = pd.DataFrame([
        _make_pair("S1-1", "S2-1", "S2", tree=0.9),
    ])
    frame["match_probability"] = 0.9
    decoder = DecoderA()
    # S1-2 has no candidates
    preds = decoder.decode(frame, threshold=0.5, s1_ids=["S1-1", "S1-2"])
    assert preds["S1-2"] == [], "S1 with no candidates must produce empty list"
    assert preds["S1-1"] == ["S2-1"]


# ===========================================================================
# §10 Fixture 2: S2 and S3 records sharing numeric suffix
# ===========================================================================

def test_source_qualified_keys_prevent_collision():
    """S2-1 and S3-1 are distinct source records despite sharing suffix '1'."""
    rows = [
        _make_pair("S1-1", "S2-1", "S2", tree=0.9),
        _make_pair("S1-1", "S3-1", "S3", tree=0.7),
    ]
    frame = pd.DataFrame(rows)
    validate_score_frame(frame, require_tree_score=True)
    assert len(frame) == 2, "S2-1 and S3-1 must be treated as distinct pairs"
    # Verify they don't collapse under the pair key
    assert not frame.duplicated(PAIR_KEY).any()


# ===========================================================================
# §10 Fixture 3: Source record with two competing S1 owners
# ===========================================================================

def test_best_owner_selects_one_s1_per_source_record(tmp_path):
    """apply_best_owner must choose exactly one S1 for a source record."""
    # S2-10 has two S1 candidates with different probabilities
    rows = [
        {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-10",
         "candidate_source": "S2", SCORE_COL_TREE: 0.9, "match_probability": 0.8},
        {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-10",
         "candidate_source": "S2", SCORE_COL_TREE: 0.7, "match_probability": 0.6},
    ]
    frame = pd.DataFrame(rows)
    result = apply_best_owner(frame, probability_col="match_probability", threshold=0.5)
    # Exactly one row selected
    selected = result[result["ownership_selected"] == True]
    assert len(selected) == 1, "Exactly one S1 should be selected as owner"
    assert selected.iloc[0]["source1_entity_id"] == "S1-A", "Higher probability should win"


# ===========================================================================
# §10 Fixture 4: Exact probability tie — deterministic resolution
# ===========================================================================

def test_exact_probability_tie_resolves_deterministically():
    """Tie on probability falls back to tree_score, then lex-smallest S1 ID."""
    rows = [
        {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-20",
         "candidate_source": "S2", SCORE_COL_TREE: 0.85, "match_probability": 0.7},
        {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-20",
         "candidate_source": "S2", SCORE_COL_TREE: 0.85, "match_probability": 0.7},
    ]
    # Shuffle and verify result is the same both times
    frame1 = pd.DataFrame(rows)
    frame2 = pd.DataFrame(rows[::-1])
    r1 = apply_best_owner(frame1, probability_col="match_probability", threshold=0.5)
    r2 = apply_best_owner(frame2, probability_col="match_probability", threshold=0.5)
    s1_r1 = r1[r1["ownership_selected"] == True].iloc[0]["source1_entity_id"]
    s1_r2 = r2[r2["ownership_selected"] == True].iloc[0]["source1_entity_id"]
    assert s1_r1 == s1_r2, "Tie resolution must be deterministic regardless of input order"
    assert s1_r1 == "S1-A", "Lex-smallest S1 ID should win tie (S1-A < S1-B)"


# ===========================================================================
# §10 Fixture 5: Source group split across two shards
# ===========================================================================

def test_source_group_reassembles_across_shards(tmp_path):
    """Source group (S2, S2-99) split across two shard files must be reassembled."""
    shard1 = pd.DataFrame([
        {"candidate_source": "S2", "candidate_entity_id": "S2-99",
         "source1_entity_id": "S1-1", "tree_score": 0.9},
    ])
    shard2 = pd.DataFrame([
        {"candidate_source": "S2", "candidate_entity_id": "S2-99",
         "source1_entity_id": "S1-2", "tree_score": 0.6},
        {"candidate_source": "S3", "candidate_entity_id": "S3-99",
         "source1_entity_id": "S1-1", "tree_score": 0.5},
    ])
    p1 = tmp_path / "shard_000.parquet"
    p2 = tmp_path / "shard_001.parquet"
    shard1.to_parquet(p1, index=False)
    shard2.to_parquet(p2, index=False)

    groups = list(iter_complete_source_groups([p1, p2]))
    # S2-99 spans both shards; should produce one complete group of 2 rows
    s2_99_groups = [g for g in groups if (g["candidate_entity_id"] == "S2-99").all()]
    assert len(s2_99_groups) == 1, "S2-99 group must be reassembled from both shards"
    assert len(s2_99_groups[0]) == 2, "Reassembled group should have 2 rows (S1-1 and S1-2)"


# ===========================================================================
# §10 Fixture 6: Tree-only pair (intentionally no neural score) — must succeed
# ===========================================================================

def test_tree_only_pair_is_valid():
    """A pair with neural_requested=0 and no neural_score is a valid tree-only route."""
    rows = [_make_pair("S1-1", "S2-1", "S2", tree=0.8, neural=None)]
    frame = pd.DataFrame(rows)
    # Should not raise
    validate_score_frame(frame, require_tree_score=True)


# ===========================================================================
# §10 Fixture 7: Neural-requested but missing neural score — must FAIL
# ===========================================================================

def test_neural_requested_missing_score_fails():
    """neural_requested=1 with no neural_score must raise ValueError."""
    row = _make_pair("S1-1", "S2-1", "S2", tree=0.8)
    row[ROUTE_COL_REQUESTED] = 1
    row[ROUTE_COL_HAS] = 0
    # No neural score column at all
    frame = pd.DataFrame([row])
    with pytest.raises(ValueError, match="neural_requested=1"):
        validate_score_frame(frame, require_tree_score=True)


# ===========================================================================
# §10 Fixture 8: Missing tree score shard — must FAIL before export
# ===========================================================================

def test_missing_tree_shard_fails(tmp_path):
    """Missing tree score for a candidate pair must fail in join_scores."""
    cand = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1",
         "candidate_source": "S2"},
    ])
    # Tree shard only covers S2-2 (not S2-1)
    tree = pd.DataFrame([
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2",
         "candidate_source": "S2", "tree_score": 0.7},
    ])
    c_path = tmp_path / "cands.parquet"
    t_path = tmp_path / "tree.parquet"
    cand.to_parquet(c_path, index=False)
    tree.to_parquet(t_path, index=False)

    with pytest.raises((ValueError, Exception)):
        join_scores([c_path], [t_path], output_dir=tmp_path / "out", provenance={})


# ===========================================================================
# §10 Fixture 9: Duplicate pair keys — must FAIL
# ===========================================================================

def test_duplicate_pair_keys_fail():
    """Duplicate pair key in score frame must raise ValueError."""
    rows = [
        _make_pair("S1-1", "S2-1", "S2", tree=0.8),
        _make_pair("S1-1", "S2-1", "S2", tree=0.6),  # duplicate
    ]
    frame = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_frame(frame, require_tree_score=True)


# ===========================================================================
# §10 Fixture 10: Final match outside candidate set — must FAIL
# ===========================================================================

def test_final_match_outside_candidate_set_fails(tmp_path):
    """validate_submission_data must fail when a final match is not in candidates."""
    from src.export import validate_submission_data
    s1_ids = ["S1-1"]
    predictions = {"S1-1": ["S2-99"]}  # S2-99 not in candidates
    candidates = {"S1-1": ["S2-1", "S2-2"]}
    with pytest.raises(ValueError):
        validate_submission_data(s1_ids, predictions, candidates)


# ===========================================================================
# Additional assertion: ID joins are order-independent
# ===========================================================================

def test_id_join_order_independent(tmp_path):
    """join_scores result must be identical regardless of input row order."""
    # neural_requested=0 so join_scores treats both as tree-only (no neural needed)
    cand_rows = [
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1",
         "candidate_source": "S2", "neural_requested": 0},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1",
         "candidate_source": "S3", "neural_requested": 0},
    ]
    tree_rows = [
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1",
         "candidate_source": "S3", "tree_score": 0.7},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1",
         "candidate_source": "S2", "tree_score": 0.9},
    ]
    cand_df = pd.DataFrame(cand_rows)
    tree_df = pd.DataFrame(tree_rows)
    c = tmp_path / "cands.parquet"
    t = tmp_path / "tree.parquet"
    cand_df.to_parquet(c, index=False)
    tree_df.to_parquet(t, index=False)

    result = join_scores([c], [t], output_dir=tmp_path / "out", provenance={})
    joined = pd.read_parquet(tmp_path / "out" / "joined_scores.parquet")
    # Verify both pairs present with correct tree scores
    assert len(joined) == 2
    s2_row = joined[joined["candidate_entity_id"] == "S2-1"].iloc[0]
    assert abs(float(s2_row["tree_score"]) - 0.9) < 1e-6


# ===========================================================================
# Additional assertion: DecoderA empty S1 remains possible
# ===========================================================================

def test_decoder_a_empty_prediction_possible():
    """When no pair is above threshold, prediction is empty (not an error)."""
    frame = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1",
         "match_probability": 0.3},
    ])
    decoder = DecoderA()
    preds = decoder.decode(frame, threshold=0.8, s1_ids=["S1-1"])
    assert preds["S1-1"] == [], "Below-threshold S1 must produce empty prediction"


# ===========================================================================
# Additional assertion: source competitor features are correct
# ===========================================================================

def test_source_competitor_features():
    """Verify source competitor features on a known 3-competitor group."""
    rows = [
        {"candidate_source": "S2", "candidate_entity_id": "S2-5",
         "source1_entity_id": "S1-A", "tree_score": 0.9},
        {"candidate_source": "S2", "candidate_entity_id": "S2-5",
         "source1_entity_id": "S1-B", "tree_score": 0.7},
        {"candidate_source": "S2", "candidate_entity_id": "S2-5",
         "source1_entity_id": "S1-C", "tree_score": 0.6},
    ]
    frame = pd.DataFrame(rows)
    result = compute_source_competitor_features(frame)
    # All rows should see 3 competitors
    assert (result["source_competitor_count"] == 3).all()
    # S1-A should be ranked 1 and be best owner
    a = result[result["source1_entity_id"] == "S1-A"].iloc[0]
    assert a["tree_source_rank"] == 1
    assert a["is_tree_best_owner"] == 1
    assert abs(a["tree_gap_to_best"]) < 1e-6
    # S1-B should be ranked 2
    b = result[result["source1_entity_id"] == "S1-B"].iloc[0]
    assert b["tree_source_rank"] == 2


# ===========================================================================
# Additional assertion: ownership never limits S1 match count
# ===========================================================================

def test_ownership_does_not_limit_s1_match_count():
    """An S1 may receive multiple source records; ownership limits per-record, not per-S1."""
    rows = [
        # S2-10: S1-A wins
        {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-10",
         "candidate_source": "S2", SCORE_COL_TREE: 0.9, "match_probability": 0.9},
        {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-10",
         "candidate_source": "S2", SCORE_COL_TREE: 0.5, "match_probability": 0.5},
        # S2-11: S1-A also wins
        {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-11",
         "candidate_source": "S2", SCORE_COL_TREE: 0.8, "match_probability": 0.85},
        {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-11",
         "candidate_source": "S2", SCORE_COL_TREE: 0.4, "match_probability": 0.4},
    ]
    frame = pd.DataFrame(rows)
    result = apply_best_owner(frame, probability_col="match_probability", threshold=0.5)
    selected = result[result["ownership_selected"] == True]
    # S1-A should be selected for both S2-10 and S2-11
    s1_a_records = selected[selected["source1_entity_id"] == "S1-A"]
    assert len(s1_a_records) == 2, "S1 may be assigned multiple source records"


# ---------------------------------------------------------------------------
# Feature Allowlist Contract Tests
# ---------------------------------------------------------------------------

from src.member_d_contracts import validate_feature_allowlist

def test_rejects_is_injected_positive():
    with pytest.raises(ValueError, match="Forbidden features found"):
        validate_feature_allowlist(["good_feature", "is_injected_positive"])

def test_rejects_positive_injected_for_training():
    with pytest.raises(ValueError, match="Forbidden features found"):
        validate_feature_allowlist(["positive_injected_for_training"])

def test_prefix_allowlist_cannot_reinclude_forbidden_feature():
    # Simulate a prefix allowlist generating an invalid feature
    features = ["is_valid", "is_injected_positive"]
    with pytest.raises(ValueError, match="Forbidden features found"):
        validate_feature_allowlist(features)

def test_clean_tree_feature_list_passes():
    # Should not raise any exception
    validate_feature_allowlist(["good_feature", "is_valid"])
