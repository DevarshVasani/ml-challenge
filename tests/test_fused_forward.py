import pandas as pd

from src.candidate_index import DiskBackedTfidfIndex, IndexConfig
from src.fused_forward import FusedForwardIndex


def test_fused_search_uses_all_shards_and_preserves_string_ids(tmp_path):
    source = pd.DataFrame({"entity_id": ["S2-01", "S2-02", "S2-03", "S2-04"],
                           "business_name": ["cedar bakery", "harbor clinic", "lotus traders", "pine books"]})
    DiskBackedTfidfIndex.build(source, tmp_path / "index",
                              IndexConfig(field="business_name", top_k=2, sparse_threads=1),
                              candidate_source="S2", source_chunk_rows=2)
    query = pd.DataFrame({"entity_id": ["S1-001", "S1-002", "S1-empty"],
                          "business_name": ["cedar bakery", "pine books", ""]})
    fused = FusedForwardIndex(tmp_path / "index")
    try:
        result = fused.query(query, top_k=2, batch_size=2)
    finally:
        fused.close()
    first = result[result.source1_entity_id == "S1-001"].iloc[0]
    second = result[result.source1_entity_id == "S1-002"].iloc[0]
    assert (first["candidate_entity_id"], first["rank"]) == ("S2-01", 1)
    assert (second["candidate_entity_id"], second["rank"]) == ("S2-04", 1)
    assert "S1-empty" not in set(result.source1_entity_id)
