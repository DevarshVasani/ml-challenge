    assert adapter.tokenizer is None


def test_collate_dynamic_padding_truncation_and_eos():
    torch = pytest.importorskip("torch")
    adapter = ByT5EncoderPairAdapter(AdapterConfig("byt5", checkpoint="local", max_length=36))
    adapter.tokenizer = _ByteTokenizer()
    # Model is only required because the adapter deliberately guards runtime initialization.
    adapter.model = torch.nn.Linear(1, 1)
    adapter.device = torch.device("cpu")

    batch = adapter.collate([_row(name_a="A"), _row(name_a="é" * 40)])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] <= 36
    assert batch["truncated"] >= 1
    long_length = int(batch["attention_mask"][1].sum())
    assert int(batch["input_ids"][1, long_length - 1]) == adapter.tokenizer.eos_token_id
    assert (batch["attention_mask"].sum(dim=1) <= 36).all()


def test_masked_mean_pool_excludes_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [1000.0, 1000.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = _masked_mean_pool(hidden, mask)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


def test_wrapper_padding_invariance_and_two_logits():
    torch = pytest.importorskip("torch")

    class ToyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(d_model=4, dropout_rate=0.0)
            self.embedding = torch.nn.Embedding(512, 4)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    torch.manual_seed(7)
    model = _build_pair_classifier(ToyEncoder()).eval()
    short_ids = torch.tensor([[4, 5, 1]])
    short_mask = torch.tensor([[1, 1, 1]])
    padded_ids = torch.tensor([[4, 5, 1, 0, 0]])
    padded_mask = torch.tensor([[1, 1, 1, 0, 0]])

    with torch.no_grad():
        first = model(short_ids, short_mask)
        second = model(padded_ids, padded_mask)
    assert tuple(first.shape) == (1, 2)
    torch.testing.assert_close(first, second, rtol=0, atol=1e-6)


def test_local_tiny_t5_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import ByT5Tokenizer, T5Config, T5EncoderModel

    checkpoint = tmp_path / "tiny-byt5"
    checkpoint.mkdir()
    config = T5Config(
        vocab_size=384,
        d_model=32,
        d_ff=64,
        d_kv=8,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    config.tokenizer_class = "ByT5Tokenizer"
    T5EncoderModel(config).save_pretrained(checkpoint)
    ByT5Tokenizer().save_pretrained(checkpoint)

    torch.manual_seed(123)
    adapter = ByT5EncoderPairAdapter.from_config(
        AdapterConfig("byt5", checkpoint=str(checkpoint), max_length=64, device="cpu"),
        execute=True,
    )
    adapter.to_device("cpu")
    adapter.model.eval()
    batch = adapter.collate([_row(), _row(name_b="Completely different")])
    with torch.no_grad():
        before = adapter.logits(batch).clone()

    saved = tmp_path / "saved"
    adapter.save_pretrained(str(saved))
    reloaded = ByT5EncoderPairAdapter.load_pretrained(str(saved), map_location="cpu")
    reloaded.model.eval()
    batch2 = reloaded.collate([_row(), _row(name_b="Completely different")])
    with torch.no_grad():
        after = reloaded.logits(batch2)

    torch.testing.assert_close(before, after, rtol=0, atol=1e-6)
    assert (saved / "adapter_metadata.json").is_file()
    assert (saved / "encoder").is_dir()
    assert (saved / "tokenizer").is_dir()
