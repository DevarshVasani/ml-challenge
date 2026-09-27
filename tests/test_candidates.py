import pandas as pd

from src.candidates import candidate_diagnostics, retrieve_channel, union_candidates


def test_empty_retrieval_field_does_not_get_arbitrary_neighbors():
    s1 = pd.DataFrame({"entity_id": ["S1-1"], "business_name": [""]})
    source = pd.DataFrame({"entity_id": ["S2-1", "S2-2"], "business_name": ["Alpha", "Beta"]})
    assert retrieve_channel(s1, source, "business_name", 2).empty


def test_union_keeps_channel_scores_and_deduplicates():
    name = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2", "score": 0.8, "rank": 1}])
    address = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2", "score": 0.6, "rank": 2}])
    result = union_candidates({"S2_name": name, "S2_address": address})
    assert len(result) == 1
    row = result.iloc[0]
    assert row["name_tfidf_score"] == 0.8
    assert row["address_tfidf_score"] == 0.6
    assert row["retrieved_by_name"] == 1 and row["retrieved_by_address"] == 1


def test_candidate_recall_and_oracle_score_on_tiny_fixture():
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["France", "India"]})
    s2 = pd.DataFrame({"entity_id": ["S2-1"]})
    s3 = pd.DataFrame({"entity_id": ["S3-1"]})
    candidates = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"]})
    report = candidate_diagnostics(candidates, s1, s2, s3, {"S1-1": ["S2-1"], "S1-2": ["S3-1"]})
    assert report["candidate_recall_overall"] == 0.5
    assert report["oracle_macro_f05"] == 0.5
    assert report["candidate_recall_by_country"]["France"]["recall"] == 1.0


def test_retrieval_does_not_emit_zero_similarity_candidates():
    s1 = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["zzzzzz"]})
    source = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "business_name": ["alpha", "beta"],
    })
    assert retrieve_channel(s1, source, "business_name", 2).empty


def test_union_keeps_same_candidate_id_from_different_sources_separate():
    s2 = pd.DataFrame([{
        "source1_entity_id": "S1-1",
        "candidate_entity_id": "shared-1",
        "candidate_source": "S2",
        "score": 0.9,
        "rank": 1,
    }])
    s3 = pd.DataFrame([{
        "source1_entity_id": "S1-1",
        "candidate_entity_id": "shared-1",
        "candidate_source": "S3",
        "score": 0.8,
        "rank": 1,
    }])
    result = union_candidates({"S2_name": s2, "S3_name": s3})
    assert len(result) == 2
    assert set(result["candidate_source"]) == {"S2", "S3"}


def test_union_preserves_min_positive_rank_and_exact_overlap_provenance():
    base = {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2"}
    name = pd.DataFrame([{**base, "score": 0.7, "rank": 3}, {**base, "score": 0.8, "rank": 2}])
    exact = pd.DataFrame([{**base, "score": 1.0, "rank": 1, "retrieval_provenance": "exact", "frequent_key_expansion": 1}])
    result = union_candidates({"S2_name": name, "S2_exact_name": exact})
    row = result.iloc[0]
    assert row.name_rank == 2 and row.address_rank == 0
    assert row.retrieval_provenance == "exact,tfidf"
    assert row.retrieved_by_exact == 1 and row.frequent_key_expansion == 1
