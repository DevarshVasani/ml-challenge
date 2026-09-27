from src.neural_diagnostics import RetrievalDiagnostics
from src.retrieval_experiments import norm, suffix_key
from src.retrieval_scan import keys_for
from src.merge_retrieval_delta import union_pairs
import pytest


def test_oracle_scores_only_retrieved_positives():
    d = RetrievalDiagnostics({"partial": ["S2-a", "S3-b"], "miss": ["S2-z"], "single": []})
    d.add_query("partial", [{"candidate_entity_id": "S2-a", "candidate_source": "S2"},
                            {"candidate_entity_id": "S2-no", "candidate_source": "S2"},
                            {"candidate_entity_id": "S2-no", "candidate_source": "S2"}])
    d.add_query("miss", [])
    d.add_query("single", [{"candidate_entity_id": "S2-no", "candidate_source": "S2"}])
    assert d.oracle_scores == [5 / 6, 0, 1]


def test_folded_suffix_and_blank_seed_keys():
    assert norm("Café, LLC") == "cafe llc"
    assert suffix_key("Café, LLC") == ""
    assert suffix_key("Central Café, LLC") == "central cafe"
    assert keys_for("", "", "seed_name") == []
    assert keys_for("", "", "seed_address") == []


def test_union_retains_baseline_and_checks_overlap_and_coverage():
    baseline={"q1":{"S2-a"},"q2":set()}
    assert union_pairs(baseline,[("q1","S3-b")])=={"q1":{"S2-a","S3-b"},"q2":set()}
    assert baseline["q1"]=={"S2-a"}
    with pytest.raises(ValueError):union_pairs(baseline,[("q1","S2-a")])
    with pytest.raises(ValueError):union_pairs(baseline,[("missing","S2-c")])
