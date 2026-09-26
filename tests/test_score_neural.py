"""Production neural scorer contracts, on the fake adapter; no downloads."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.neural_adapters import AdapterConfig, FakePairAdapter
from src.neural_card import build_card
from src.neural_contracts import file_identity
from src.neural_models import NeuralTrainingConfig, save_checkpoint
from src.score_neural import OUTPUT_COLUMNS, benchmark, execute, token_budget_batches
from src.source_store import SourceStore

KEY = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
QUIET = dict(log=lambda *_: None)


def _pair(sid, cid, source="S2", name_b="Cafe", **extra):
    return {"source1_entity_id": sid, "candidate_entity_id": cid, "candidate_source": source, "source1_fold": 1, "candidate_fold": 1,
            "name_a": "Café", "address_a": "001 Main", "country_a": "US", "name_b": name_b, "address_b": "001 Main St", "country_b": "US", **extra}


def _checkpoint(path, seed=2):
    if path.exists():  # re-saving would change the hash: torch.save embeds a random serialization id
        return str(path)
    save_checkpoint(FakePairAdapter(AdapterConfig("fake", checkpoint="offline-fake"), seed=seed), path, NeuralTrainingConfig(adapter_type="fake"))
    return str(path)


def _config(tmp_path, pairs, **extra):
    return {"adapter_type": "fake", "smoke": True, "checkpoint": _checkpoint(tmp_path / "ckpt"), "pairs": pairs, "device": "cpu",
            "max_tokens": 64, "output_dir": str(tmp_path / "scores"), **extra}


def _read(directory):
    return pd.concat([pd.read_parquet(p) for p in sorted(directory.glob("*.neural.parquet"))], ignore_index=True)


def test_token_budget_batches_respect_budget_and_row_cap():
    lengths = np.array([50, 40, 10, 10, 10, 10, 3])
    batches = token_budget_batches(lengths, max_tokens=100, max_rows=3)
    assert batches == [(0, 2), (2, 5), (5, 7)]
    assert all((b - a) * lengths[a] <= 100 or b - a == 1 for a, b in batches)
    assert token_budget_batches(np.array([500]), max_tokens=100, max_rows=8) == [(0, 1)]


def test_scores_are_keyed_logits_and_resume(tmp_path):
    pairs = pd.DataFrame([_pair("S1-1", f"S2-{i:03d}", name_b="x" * i) for i in range(20)])
    pairs.to_parquet(tmp_path / "pairs-a.parquet", index=False)
    config = _config(tmp_path, [str(tmp_path / "pairs-a.parquet")])
    first = execute(config, **QUIET)
    assert first["completion"] == "complete" and first["rows"] == 20 and first["model_pairs"] == 20
    scores = _read(tmp_path / "scores")
    assert list(scores.columns) == OUTPUT_COLUMNS
    assert scores["candidate_entity_id"].tolist() == pairs["candidate_entity_id"].tolist()  # IDs keep leading zeros, input order kept
    assert scores["neural_scored"].all() and np.isfinite(scores["neural_logit"]).all()
    assert scores["checkpoint_id"].nunique() == 1 and scores["checkpoint_id"].iloc[0].startswith("ckpt-")
    # The logit is logit[1]-logit[0] of the same model the pilot scorer uses.
    adapter = FakePairAdapter.load_pretrained(str(tmp_path / "ckpt" / "adapter"))
    with torch.no_grad():
        raw = adapter.logits(adapter.collate(pairs.to_dict(orient="records")))
    assert np.allclose(scores["neural_logit"], (raw[:, 1] - raw[:, 0]).numpy(), atol=1e-6)
    second = execute(config, **QUIET)
    assert second["shards_resumed"] == 1 and second["shards_scored"] == 0
    manifest = json.loads((tmp_path / "scores" / "neural_manifest.json").read_text())
    assert manifest["completion"] == "complete" and manifest["shards"]["pairs-a.neural.parquet"]["rows"] == 20


def test_route_shards_join_text_and_flag_missing_records(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame({"entity_id": ["S1-001", "S1-002"], "business_name": ["Café", "Two"], "business_address": ["1 Main", "2 Main"], "country": ["US", "India"]}).to_csv(data / "s1.tsv", sep="\t", index=False)
    pd.DataFrame({"entity_id": ["S2-001", "S2-002"], "business_name": ["Cafe", "Too"], "business_address": ["1 Main St", "2 Main"], "country": ["US", "India"]}).to_csv(data / "s2.tsv", sep="\t", index=False)
    pd.DataFrame({"entity_id": ["S3-001"], "business_name": ["Cafe Inc"], "business_address": ["1 Main"], "country": ["US"]}).to_csv(data / "s3.tsv", sep="\t", index=False)
    store = tmp_path / "store.sqlite"
    SourceStore.build({s: str(data / f"{s.lower()}.tsv") for s in ("S1", "S2", "S3")}, store)
    route = pd.DataFrame({"s1_id": ["S1-001", "S1-001", "S1-002", "S1-002", "S1-001"],
                          "source_record_id": ["S2-001", "S3-001", "S2-002", "S2-999", "S2-002"],
                          "source": ["S2", "S3", "S2", "S2", "S2"],
                          "route": ["neural", "neural", "neural", "neural", "tree_only"], "tree_score": [0.9, 0.8, 0.7, 0.5, 0.01]})
    route.to_parquet(tmp_path / "route-0001.parquet", index=False)
    config = _config(tmp_path, [str(tmp_path / "route-0001.parquet")], source_store=str(store),
                     key_columns={"source1_entity_id": "s1_id", "candidate_entity_id": "source_record_id", "candidate_source": "source"})
    result = execute(config, **QUIET)
    scores = _read(tmp_path / "scores").set_index("candidate_entity_id")
    assert len(scores) == 4 and "S2-002" in scores.index and scores.loc["S2-002", "source1_entity_id"] == "S1-002"  # tree_only row dropped
    assert scores.loc["S2-999", "neural_status"] == "missing_record" and not scores.loc["S2-999", "neural_scored"] and np.isnan(scores.loc["S2-999", "neural_logit"])
    assert scores.drop(index="S2-999")["neural_scored"].all() and result["missing_record"] == 1
    # Text came through the same record->text mapping as training pairs.
    adapter = FakePairAdapter.load_pretrained(str(tmp_path / "ckpt" / "adapter"))
    with torch.no_grad():
        raw = adapter.logits(adapter.collate([_pair("S1-001", "S2-001", name_b="Cafe", address_a="1 Main", address_b="1 Main St")]))
    assert np.isclose(scores.loc["S2-001", "neural_logit"], float(raw[0, 1] - raw[0, 0]), atol=1e-6)


def test_reuse_only_when_key_text_and_checkpoint_match(tmp_path):
    pairs = pd.DataFrame([_pair("S1-1", f"S2-{i}") for i in range(6)])
    pairs.to_parquet(tmp_path / "p.parquet", index=False)
    execute(_config(tmp_path, [str(tmp_path / "p.parquet")]), **QUIET)
    first = _read(tmp_path / "scores")
    changed = pairs.copy(); changed.loc[2, "name_b"] = "Different"
    changed = pd.concat([changed, pd.DataFrame([_pair("S1-1", "S2-new")])], ignore_index=True)
    changed.to_parquet(tmp_path / "q.parquet", index=False)
    config = _config(tmp_path, [str(tmp_path / "q.parquet")], output_dir=str(tmp_path / "scores-v2"), reuse_from=[str(tmp_path / "scores")])
    result = execute(config, **QUIET)
    second = _read(tmp_path / "scores-v2").set_index("candidate_entity_id")
    assert result["cached"] == 5 and result["model_pairs"] == 2
    assert second.loc["S2-2", "neural_status"] == "scored" and second.loc["S2-new", "neural_status"] == "scored"
    same = first.set_index("candidate_entity_id").drop(index="S2-2")
    assert (second.loc[same.index, "neural_status"] == "cached").all()
    assert np.array_equal(second.loc[same.index, "neural_logit"].to_numpy(), same["neural_logit"].to_numpy())
    # Another checkpoint never reuses these scores.
    other = _config(tmp_path, [str(tmp_path / "q.parquet")], output_dir=str(tmp_path / "scores-v3"), reuse_from=[str(tmp_path / "scores")])
    other["checkpoint"] = _checkpoint(tmp_path / "ckpt-other", seed=9)
    assert execute(other, **QUIET)["cached"] == 0


def test_changed_input_refuses_unless_rescored_and_checkpoint_change_refuses(tmp_path):
    path = tmp_path / "p.parquet"
    pd.DataFrame([_pair("S1-1", "S2-1"), _pair("S1-1", "S2-2")]).to_parquet(path, index=False)
    config = _config(tmp_path, [str(path)])
    execute(config, **QUIET)
    pd.DataFrame([_pair("S1-1", "S2-1"), _pair("S1-1", "S2-2"), _pair("S1-1", "S3-9", source="S3")]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="changed since it was scored"):
        execute(config, **QUIET)
    result = execute(config, rescore_changed=True, **QUIET)
    assert result["cached"] == 2 and result["model_pairs"] == 1 and len(_read(tmp_path / "scores")) == 3
    config["checkpoint"] = _checkpoint(tmp_path / "ckpt-other", seed=9)
    with pytest.raises(ValueError, match="different checkpoint"):
        execute(config, **QUIET)


def test_unpublished_shards_stay_pending_and_numeric_ids_are_rejected(tmp_path):
    shards = tmp_path / "routes"; shards.mkdir()
    pd.DataFrame([_pair("S1-1", "S2-1")]).to_parquet(shards / "a.parquet", index=False)
    pd.DataFrame([_pair("S1-2", "S2-2")]).to_parquet(shards / "b.parquet", index=False)
    (shards / "a.parquet.complete.json").write_text("{}")
    config = _config(tmp_path, str(shards / "*.parquet"), require_complete_sidecar=True)
    assert execute(config, **QUIET)["completion"] == "partial"
    (shards / "b.parquet.complete.json").write_text("{}")
    result = execute(config, **QUIET)
    assert result["completion"] == "complete" and result["shards_resumed"] == 1 and result["shards_scored"] == 1
    numeric = pd.DataFrame([_pair("S1-1", "S2-1")]).assign(candidate_entity_id=[7])
    numeric.to_parquet(tmp_path / "n.parquet", index=False)
    with pytest.raises(ValueError, match="leading zeros"):
        execute(_config(tmp_path, [str(tmp_path / "n.parquet")], output_dir=str(tmp_path / "n")), **QUIET)


def test_benchmark_reports_stages_and_projection(tmp_path):
    for shard in range(3):
        pd.DataFrame([_pair(f"S1-{shard}", f"S2-{i}", name_b="n" * (i % 7)) for i in range(40)]).to_parquet(tmp_path / f"p{shard}.parquet", index=False)
    config = _config(tmp_path, str(tmp_path / "p*.parquet"))
    report = benchmark(config, pairs=50, max_tokens_sweep=[32, 128], project_pairs=1_000_000, **QUIET)
    assert report["sample_pairs"] == 50 and len(report["sweep"]) == 2
    assert set(report["stage_seconds"]) == {"read", "join", "hash", "tokenize", "write", "forward"}
    assert report["pairs_per_second_overlapped"] >= report["pairs_per_second_serial"] > 0
    assert report["projection"]["hours_serial"] >= report["projection"]["hours_overlapped"] > 0
    assert list((tmp_path / "scores" / "benchmark").glob("benchmark-*.json"))


def test_checkpoint_card_verifies_training_inputs_and_lists_exposure(tmp_path):
    data = tmp_path / "neural-data"; (data / "train_pairs").mkdir(parents=True)
    train = pd.DataFrame([_pair("S1-1", "S2-1", label=1), _pair("S1-1", "S2-2", label=0), _pair("S1-2", "S3-1", source="S3", label=0)])
    train.to_parquet(data / "train_pairs" / "pairs-0.parquet", index=False)
    (data / "train_pairs_manifest.json").write_text(json.dumps({"shard_order": ["train_pairs/pairs-0.parquet"]}))
    (data / "query_manifest.json").write_text(json.dumps({"query_ids": {"train": ["S1-1", "S1-2"], "threshold": ["S1-5"], "final": ["S1-2"]},
                                                          "query_metadata": {"S1-1": {"fold": 1}, "S1-2": {"fold": 2}, "S1-5": {"fold": 0}}}))
    identities = {"pair_manifest": file_identity(data / "train_pairs_manifest.json", hash_content=True),
                  "query_manifest": file_identity(data / "query_manifest.json", hash_content=True)}
    save_checkpoint(FakePairAdapter(AdapterConfig("fake", checkpoint="offline-fake")), tmp_path / "ckpt", NeuralTrainingConfig(adapter_type="fake"), manifest_identities=identities)
    train_config = tmp_path / "train.json"
    train_config.write_text(json.dumps({"training": {"adapter_type": "fake"}, "pair_manifest": str(data / "train_pairs_manifest.json"), "query_manifest": str(data / "query_manifest.json")}))
    card = build_card(str(tmp_path / "ckpt"), str(train_config), str(tmp_path / "card"))
    assert card["training"]["inputs_verified"] and card["checkpoint_id"].startswith("ckpt-")
    exposure = pd.read_parquet(tmp_path / "card" / "training_exposure.parquet")
    assert set(exposure["entity_id"]) == {"S1-1", "S1-2", "S2-1", "S2-2", "S3-1"}
    assert exposure.set_index("entity_id").loc["S2-1", "seen_as_positive"] and not exposure.set_index("entity_id").loc["S2-2", "seen_as_positive"]
    assert card["exposure"]["heldout_queries_seen_in_training"] == {"threshold": 0, "final": 1}
    (data / "query_manifest.json").write_text("{}")
    assert not build_card(str(tmp_path / "ckpt"), str(train_config), str(tmp_path / "card2"))["training"]["inputs_verified"]
