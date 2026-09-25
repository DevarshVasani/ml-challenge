import pandas as pd

from src.pipeline import run_pipeline


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_small_synthetic_pipeline_produces_valid_outputs(tmp_path):
    data = tmp_path / "dataset"
    train = data / "train"; test = data / "test"
    train.mkdir(parents=True); test.mkdir(parents=True)
    _write(train / "train_source1.tsv", [
        {"entity_id": "S1-1", "business_name": "Alpha One", "business_address": "1 Main", "country": "France"},
        {"entity_id": "S1-2", "business_name": "Beta Two", "business_address": "2 Main", "country": "India"},
        {"entity_id": "S1-3", "business_name": "Gamma Three", "business_address": "3 Main", "country": "US"},
        {"entity_id": "S1-4", "business_name": "Delta Four", "business_address": "4 Main", "country": "France"},
        {"entity_id": "S1-5", "business_name": "Epsilon Five", "business_address": "5 Main", "country": "India"},
        {"entity_id": "S1-6", "business_name": "Zeta Six", "business_address": "6 Main", "country": "US"},
    ])
    _write(train / "train_source2.tsv", [{"entity_id": f"S2-{i}", "business_name": name, "business_address": f"{i} Main", "country": country} for i, (name, country) in enumerate([("Alpha One", "France"), ("Beta Two", "India"), ("Gamma Three", "US"), ("Delta Four", "France"), ("Epsilon Five", "India"), ("Zeta Six", "US")], 1)])
    _write(train / "train_source3.tsv", [{"entity_id": "S3-1", "business_name": "Unrelated", "business_address": "90 Road", "country": "France"}])
    _write(train / "train_ground_truth.tsv", [{"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i}"} for i in range(1, 7)])
    _write(test / "test_source1.tsv", [{"entity_id": "S1-101", "business_name": "Alpha One", "business_address": "1 Main", "country": "France"}, {"entity_id": "S1-102", "business_name": "No Match", "business_address": "999 Road", "country": "India"}, {"entity_id": "S1-103", "business_name": "Beta Two", "business_address": "2 Main", "country": "US"}])
    _write(test / "test_source2.tsv", [{"entity_id": "S2-101", "business_name": "Alpha One", "business_address": "1 Main", "country": "France"}, {"entity_id": "S2-102", "business_name": "Beta Two", "business_address": "2 Main", "country": "US"}])
    _write(test / "test_source3.tsv", [{"entity_id": "S3-101", "business_name": "Other", "business_address": "3 Road", "country": "France"}])
    result = run_pipeline(data, tmp_path / "artifacts", tmp_path / "output", n_splits=3, top_k_name=2, top_k_address=2, batch_size=2)
    assert (tmp_path / "output/matching_results.tsv").exists()
    assert (tmp_path / "output/candidate_pairs.tsv").exists()
    assert result["matching_results"].endswith("matching_results.tsv")
    assert len(pd.read_csv(tmp_path / "output/matching_results.tsv", sep="\t", dtype=str)) == 3
