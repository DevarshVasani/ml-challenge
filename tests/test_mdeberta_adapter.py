"""Offline mDeBERTa adapter tests on a tiny random DeBERTa-v2 model.

A throwaway sentencepiece vocabulary and a 2-layer model stand in for
microsoft/mdeberta-v3-base, so nothing is downloaded.
"""

import math

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
spm = pytest.importorskip("sentencepiece")
pytest.importorskip("google.protobuf")

from src.neural_adapters import AdapterConfig, get_adapter_class
from src.neural_adapters.mdeberta import MDebertaPairAdapter
from src.neural_models import NeuralTrainingConfig, predict_pairs, train_neural

CORPUS = [
    "Café Bleu | 12 Rue de Rivoli | France", "Cafe Bleu | 12 rue Rivoli | France",
    "Sharma Traders | 45 MG Road Bengaluru | India", "Sharma Trading Co | 54 MG Road | India",
    "Joe's Pizza | 100 Main Street Springfield | United States", "Joes Pizza | 100 Main St | United States",
    "Bäckerei Müller | Hauptstraße 7 | Germany", "Müller Bakery | Hauptstr. 7 | Germany",
]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-mdeberta")
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(CORPUS * 20), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / "spm"), vocab_size=80, character_coverage=1.0,
                                   pad_id=0, unk_id=3, bos_id=1, eos_id=2, pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[CLS]", eos_piece="[SEP]",
                                   user_defined_symbols=["[MASK]"], minloglevel=2)
    checkpoint = root / "checkpoint"
    tokenizer = transformers.DebertaV2TokenizerFast(vocab_file=str(root / "spm.model"))
    tokenizer.save_pretrained(checkpoint)
    config = transformers.DebertaV2Config(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                          max_position_embeddings=128, type_vocab_size=0, relative_attention=True, position_buckets=16,
                                          pos_att_type=["p2c", "c2p"], pad_token_id=tokenizer.pad_token_id, num_labels=2)
    torch.manual_seed(0)
    transformers.DebertaV2ForSequenceClassification(config).save_pretrained(checkpoint)
    return str(checkpoint)


def _row(sid="S1-1", cid="S2-1", label=0, **text):
    row = {"source1_entity_id": sid, "candidate_entity_id": cid, "candidate_source": "S2", "source1_fold": 1, "candidate_fold": 1, "label": label,
           "name_a": "Café Bleu", "address_a": "12 Rue de Rivoli", "country_a": "France", "name_b": "Cafe Bleu", "address_b": "12 rue Rivoli", "country_b": "France"}
    row.update(text)
    return row


def _adapter(checkpoint, max_length=64):
    return MDebertaPairAdapter(AdapterConfig("mdeberta", checkpoint=checkpoint, max_length=max_length))


def test_registry_and_dry_run_guard():
    assert get_adapter_class("mdeberta") is MDebertaPairAdapter
    with pytest.raises(RuntimeError, match="--execute"):
        MDebertaPairAdapter.from_config(AdapterConfig("mdeberta"))


def test_serialization_order_missing_values_and_literal_text(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    row = _row(address_a=None, country_a=float("nan"), name_b="NA", address_b="  007   Main\tSt ", country_b=pd.NA)
    assert adapter.segments(row) == ("Café Bleu |  | ", "NA | 007 Main St | ")
    assert adapter.serialize_pair(row) == "Café Bleu |  |  [SEP] NA | 007 Main St | "
    # IDs, folds, labels, and retrieval scores never reach model text.
    assert "S1-1" not in adapter.serialize_pair({**row, "retrieval_score": 0.93}) and "0.93" not in adapter.serialize_pair({**row, "retrieval_score": 0.93})


def test_collate_pads_truncates_and_counts(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    rows = [_row(), _row(address_b=" ".join(["Hauptstraße 7 Bengaluru Springfield"] * 20))]
    budget = adapter.pair_lengths(rows[:1])[0] + 2
    adapter.config.max_length = budget
    batch = adapter.collate(rows)
    assert batch["input_ids"].shape[0] == 2 and batch["input_ids"].shape[1] == budget
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["truncated"] == 1
    assert batch["untruncated_lengths"][1] > budget > batch["untruncated_lengths"][0]
    assert int(batch["attention_mask"][0].sum()) == budget - 2
    assert adapter.pair_lengths(rows) == batch["untruncated_lengths"]
    # longest_first keeps part of the short S1 side rather than dropping it wholesale
    first_sep = batch["input_ids"][1].tolist().index(adapter.tokenizer.sep_token_id)
    assert first_sep > 1


def test_logits_shape_and_probabilities(tiny_checkpoint):
    adapter = _adapter(tiny_checkpoint)
    adapter.model.eval()
    logits = adapter.logits(adapter.collate([_row(), _row(name_b="Joes Pizza")]))
    assert tuple(logits.shape) == (2, 2) and logits.dtype.is_floating_point


def test_save_reload_round_trip_matches_predictions(tiny_checkpoint, tmp_path):
    adapter = _adapter(tiny_checkpoint, max_length=48)
    pairs = pd.DataFrame([_row(), _row(cid="S2-2", name_b="Sharma Traders", address_b="45 MG Road", country_b="India")]).drop(columns="label")
    before = predict_pairs(adapter, pairs, batch_size=2)
    adapter.save_pretrained(str(tmp_path / "adapter"))
    reloaded = MDebertaPairAdapter.load_pretrained(str(tmp_path / "adapter"))
    after = predict_pairs(reloaded, pairs, batch_size=2)
    assert reloaded.config.max_length == 48
    assert reloaded.provenance == adapter.provenance
    assert (before["score"] - after["score"]).abs().max() < 1e-6


def test_shared_trainer_overfits_tiny_set_and_checkpoints(tiny_checkpoint, tmp_path):
    adapter = _adapter(tiny_checkpoint)
    rows = pd.DataFrame([
        _row("S1-1", "S2-1", 1), _row("S1-1", "S2-2", 0, name_b="Joe's Pizza", address_b="100 Main Street", country_b="United States"),
        _row("S1-2", "S2-3", 1, name_a="Sharma Traders", address_a="45 MG Road", country_a="India", name_b="Sharma Trading Co", address_b="45 MG Road", country_b="India"),
        _row("S1-2", "S2-4", 0, name_a="Sharma Traders", address_a="45 MG Road", country_a="India", name_b="Bäckerei Müller", address_b="Hauptstraße 7", country_b="Germany"),
    ])
    config = NeuralTrainingConfig(adapter_type="mdeberta", microbatch=4, gradient_accumulation=1, learning_rate=3e-3, epochs=40,
                                  warmup_fraction=0.0, log_every=1, device="cpu", run_dir=str(tmp_path / "run"), gradient_checkpointing=True)
    result = train_neural(adapter, rows, config, save_final_checkpoint=True)
    losses = [entry["loss"] for entry in result["log_history"]]
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < 0.5 * losses[0]
    reloaded = MDebertaPairAdapter.load_pretrained(str(tmp_path / "run" / "adapter"))
    scores = predict_pairs(reloaded, rows.drop(columns="label"), batch_size=4)
    assert (scores["score"].to_numpy() > 0.5).tolist() == [True, False, True, False]
