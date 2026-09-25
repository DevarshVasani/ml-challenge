"""Small offline contract tests; no competition files or model downloads."""

from pathlib import Path

import pandas as pd
import pytest

from src.candidate_io import GroupedParquetWriter, iter_complete_groups
from src.candidate_index import IndexConfig, SourceTfidfIndex
from src.export import export_streaming_shards
from src.neural_adapters import AdapterConfig, FakePairAdapter
from src.neural_contracts import validate_pair_frame, validate_prediction_frame
from src.neural_data import read_text_tsv, select_query_ids, select_training_pairs, stable_unmatched_fold
from src.neural_models import NeuralTrainingConfig, evaluate_scores, load_training_state, predict_pairs, save_checkpoint, train_neural
from src.predict_neural import execute as execute_prediction
from src.source_store import SourceStore, make_pair_group
from src.train_neural import load_config, plan


def _pair(sid="S1-1", cid="S2-1", source="S2", label=0):
    return {"source1_entity_id": sid, "candidate_entity_id": cid, "candidate_source": source, "label": label,
            "source1_fold": 1, "candidate_fold": 1, "name_a": "Café NA", "address_a": "001", "country_a": "France",
            "name_b": "Cafe", "address_b": "001", "country_b": "France"}


def test_literal_text_and_injected_positive_are_safe(tmp_path):
    path = tmp_path / "raw.tsv"
    path.write_text("entity_id\tbusiness_name\tcountry\nS1-001\tNA\tFrance\n", encoding="utf-8")
    frame = read_text_tsv(path)
    assert frame.iloc[0].business_name == "NA"
    candidates = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "candidate_source": "S2", "retrieval_score": 0.9}])
    selected, report = select_training_pairs(candidates, ["S1-1"], {"S1-1": ["S2-1", "S2-2"]}, seed=7)
    assert set(selected.loc[selected.label == 1, "candidate_entity_id"]) == {"S2-1", "S2-2"}
    assert report["injected_positives"] == 1


def test_index_reuse_and_identity_invalidation(tmp_path):
    source = pd.DataFrame({"entity_id": ["S2-01", "S2-02"], "business_name": ["Café Bleu", "Other"]})
    config = IndexConfig("business_name", top_k=2)
    index = SourceTfidfIndex(source, config, source_identity={"version": 1})
    query = pd.DataFrame({"entity_id": ["S1-a"], "business_name": ["cafe bleu"]})
    assert len(index.query(query, candidate_source="S2")) == 1
    index.save(tmp_path / "index")
    loaded = SourceTfidfIndex.load(tmp_path / "index", expected_config=config, expected_source_identity={"version": 1})
    assert loaded.query(query, candidate_source="S2").equals(index.query(query, candidate_source="S2"))
    with pytest.raises(ValueError):
        SourceTfidfIndex.load(tmp_path / "index", expected_config=config, expected_source_identity={"version": 2})


def test_group_carry_and_writer_never_splits_s1(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frame = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2"}, {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "candidate_source": "S2"}, {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-3", "candidate_source": "S2"}])
    frame.iloc[:1].to_parquet(first, index=False); frame.iloc[1:].to_parquet(second, index=False)
    groups = list(iter_complete_groups([first, second]))
    assert [len(group) for group in groups] == [2, 1]
    writer = GroupedParquetWriter(tmp_path / "out", target_rows=2)
    for group in groups: writer.write_group(group)
    writer.close()
    assert len(writer.files) == 2


def test_pilot_configs_target_cuda_without_dry_run_execution():
    for name, adapter_type in (
        ("mdeberta_pilot.json", "mdeberta"),
        ("byt5_pilot.json", "byt5"),
    ):
        config = load_config(Path("configs/neural") / name)
        planned = plan(config)
        assert planned["adapter_type"] == adapter_type
        assert planned["device"] == "cuda"
        assert planned["model_initialization"] == "deferred"
        assert planned["dataset_iteration"] == "deferred"


def test_fake_adapter_training_prediction_and_export(tmp_path):
    torch = pytest.importorskip("torch")
    rows = pd.DataFrame([_pair("S1-1", "S2-1", label=1), _pair("S1-1", "S2-2", label=0), _pair("S1-2", "S2-3", label=0)])
    adapter = FakePairAdapter(AdapterConfig("fake", checkpoint="offline-fake"), seed=3)
    result = train_neural(adapter, rows, NeuralTrainingConfig(adapter_type="fake", microbatch=2, gradient_accumulation=2, epochs=1, max_steps=2, device="cpu"))
    assert result["steps"] == 1
    checkpoint_config = NeuralTrainingConfig(adapter_type="fake", microbatch=2, gradient_accumulation=2, epochs=1, max_steps=2, device="cpu")
    save_checkpoint(adapter, tmp_path / "checkpoint", checkpoint_config, step=result["steps"])
    state = load_training_state(tmp_path / "checkpoint", expected_config=checkpoint_config)
    assert state["step"] == 1
    FakePairAdapter.load_pretrained(str(tmp_path / "checkpoint" / "adapter"))
    scores = predict_pairs(adapter, rows, batch_size=2)
    validate_prediction_frame(scores)
    metrics = evaluate_scores(scores, {"S1-1": ["S2-1"], "S1-2": []}, ["S1-1", "S1-2"], 0.5)
    assert metrics["num_entities"] == 2
    candidate = tmp_path / "candidate.parquet"; scored = tmp_path / "scores.parquet"
    rows.drop(columns="label").to_parquet(candidate, index=False); scores.to_parquet(scored, index=False)
    match, candidate_out = export_streaming_shards(["S1-1", "S1-2"], [str(scored)], [str(candidate)], str(tmp_path / "export"), threshold=0.5)
    assert Path(match).exists() and Path(candidate_out).exists()


def test_sqlite_store_join_preserves_accents_and_leading_zero(tmp_path):
    source = tmp_path / "s2.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-001\tÉlite\t00123\tFrance\n", encoding="utf-8")
    store = SourceStore.build({"S2": source}, tmp_path / "sources.sqlite")
    candidates = pd.DataFrame([{"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2", "retrieval_score": 1.0}])
    pair = make_pair_group(candidates, {"S1-001": {"business_name": "Élite", "business_address": "00123", "country": "France"}}, store, folds={"S1-001": 0, "S2-001": 0}, ground_truth={"S1-001": ["S2-001"]})
    assert pair.iloc[0].name_b == "Élite" and pair.iloc[0].address_b == "00123" and int(pair.iloc[0].label) == 1
def test_contracts_reject_fractional_labels_and_infinite_scores():
    pairs = pd.DataFrame([_pair(label=0.5)])
    with pytest.raises(ValueError, match="integer"):
        validate_pair_frame(pairs, labeled=True)
    scores = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2", "score": float("inf")}])
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_frame(scores)


def test_component_selection_and_heldout_negative_filter_are_deterministic():
    s1 = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "country": ["France", "France", "India"]})
    gt = {"S1-a": ["shared"], "S1-b": ["shared"], "S1-c": []}
    folds = {"S1-a": 1, "S1-b": 1, "S1-c": 1, "shared": 1, "S2-held": 0}
    selected, report = select_query_ids(s1, gt, fold_map=folds, fold=[1, 2], requested=1, seed=9)
    assert {"S1-a", "S1-b"}.issubset(selected) or selected == ["S1-c"]
    for component in report["component_groups"]:
        assert not ({"S1-a", "S1-b"} & set(component)) or {"S1-a", "S1-b"}.issubset(component)
    candidates = pd.DataFrame([
        {"source1_entity_id": "S1-a", "candidate_entity_id": "shared", "candidate_source": "S2", "retrieval_score": 1.0},
        {"source1_entity_id": "S1-a", "candidate_entity_id": "S2-held", "candidate_source": "S2", "retrieval_score": 0.9},
        {"source1_entity_id": "S1-a", "candidate_entity_id": "unmatched", "candidate_source": "S2", "retrieval_score": 0.8},
    ])
    first, summary = select_training_pairs(candidates, ["S1-a"], gt, seed=4, fold_map=folds, held_out_folds={0}, fold_seed=11)
    second, _ = select_training_pairs(candidates, ["S1-a"], gt, seed=4, fold_map=folds, held_out_folds={0}, fold_seed=11)
    assert first.equals(second)
    assert "S2-held" not in set(first["candidate_entity_id"])
    assert summary["excluded_heldout_negatives"] >= 1
    assert stable_unmatched_fold("unmatched", seed=11) == stable_unmatched_fold("unmatched", seed=11)


def test_streaming_export_includes_zero_candidate_queries_and_rejects_failed_scoring(tmp_path):
    candidate = tmp_path / "candidate.parquet"; score = tmp_path / "score.parquet"
    pd.DataFrame([_pair()]).drop(columns="label").to_parquet(candidate, index=False)
    pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2", "score": 0.9}]).to_parquet(score, index=False)
    matching, candidates = export_streaming_shards(["S1-1", "S1-zero"], [str(score)], [str(candidate)], str(tmp_path / "out"), threshold=0.5)
    assert Path(matching).read_text(encoding="utf-8").splitlines()[-1] == "S1-zero\t"
    assert Path(candidates).read_text(encoding="utf-8").splitlines()[-1] == "S1-zero\t"
    empty_score = tmp_path / "empty-score.parquet"
    pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score"]).to_parquet(empty_score, index=False)
    with pytest.raises(ValueError, match="missing scoring coverage"):
        export_streaming_shards(["S1-1"], [str(empty_score)], [str(candidate)], str(tmp_path / "bad"), threshold=0.5)


def test_prediction_shard_resume_and_input_invalidation(tmp_path):
    pytest.importorskip("torch")
    pairs = pd.DataFrame([_pair("S1-1", "S2-1", label=1)]).drop(columns="label")
    pair_path = tmp_path / "pairs.parquet"; pairs.to_parquet(pair_path, index=False)
    checkpoint = tmp_path / "checkpoint"
    adapter = FakePairAdapter(AdapterConfig("fake", checkpoint="offline-fake"), seed=2)
    save_checkpoint(adapter, checkpoint, NeuralTrainingConfig(adapter_type="fake"))
    query_manifest = tmp_path / "queries.json"
    query_manifest.write_text('{"query_ids":{"final":["S1-1"]}}', encoding="utf-8")
    config = {"adapter_type": "fake", "smoke": True, "pairs": [str(pair_path)], "checkpoint": str(checkpoint), "query_manifest": str(query_manifest), "query_subset": "final", "device": "cpu", "output_dir": str(tmp_path / "predictions")}
    first = execute_prediction(config); second = execute_prediction(config)
    assert first["rows"] == 1 and second["resumed_shards"] == 1
    changed = pd.concat([pairs, pd.DataFrame([_pair("S1-1", "S2-2")]).drop(columns="label")], ignore_index=True)
    changed.to_parquet(pair_path, index=False)
    with pytest.raises(ValueError, match="identity changed"):
        execute_prediction(config)
