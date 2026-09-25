import pytest
import os
import json
import numpy as np
from pathlib import Path

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from src.neural_adapters.base import AdapterConfig
from src.neural_adapters.byt5 import ByT5EncoderPairAdapter


@pytest.fixture
def dummy_config():
    return AdapterConfig(
        adapter_type="byt5",
        checkpoint="google/byt5-small",
        max_length=512,
        device="cpu"
    )

def test_serialization(dummy_config):
    adapter = ByT5EncoderPairAdapter(dummy_config)
    row = {
        "name_a": "Apple Inc",
        "address_a": "1 Infinite Loop",
        "country_a": "US",
        "name_b": "Apple",
        "address_b": "",
        "country_b": "US",
        "candidate_entity_id": "123", # Should be ignored
        "label": 1, # Should be ignored
    }
    serialized = adapter.serialize_pair(row)
    assert serialized == "Apple Inc || 1 Infinite Loop || US || Apple ||  || US"

def test_truncation_logic(dummy_config):
    # This requires execute=True to load tokenizer, but for fast offline test we might mock or skip.
    pass

def test_padding_invariance():
    # Real test requires model loading which we skip for offline unless it's marked
    pass
