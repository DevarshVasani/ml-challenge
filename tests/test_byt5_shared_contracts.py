from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from src.neural_contracts import seed_everything
from src.neural_models import evaluate_scores


def test_seed_everything_repeats_python_numpy_and_torch_sequences():
    seed_everything(1234)
    python_first = [random.random() for _ in range(4)]
    numpy_first = np.random.random(4)
    try:
        import torch
        torch_first = torch.rand(4)
    except ImportError:
        torch = None
        torch_first = None

    seed_everything(1234)
    assert python_first == [random.random() for _ in range(4)]
    np.testing.assert_array_equal(numpy_first, np.random.random(4))
    if torch is not None:
        torch.testing.assert_close(torch_first, torch.rand(4), rtol=0, atol=0)


def test_evaluate_scores_reports_singleton_and_non_singleton_groups():
    scores = pd.DataFrame([
        {"source1_entity_id": "singleton", "candidate_entity_id": "c0", "candidate_source": "S2", "score": 0.10},
        {"source1_entity_id": "matched", "candidate_entity_id": "c1", "candidate_source": "S2", "score": 0.90},
    ])
    ground_truth = {"singleton": [], "matched": ["c1"]}
    metadata = {
        "singleton": {"singleton": True, "match_count": 0, "country": "US"},
        "matched": {"singleton": False, "match_count": 1, "country": "India"},
    }

    result = evaluate_scores(
        scores,
        ground_truth,
        ["singleton", "matched"],
        threshold=0.5,
        query_metadata=metadata,
    )

    assert set(result["by_singleton"]) == {"False", "True"}
    assert result["by_singleton"]["True"]["num_entities"] == 1
    assert result["by_singleton"]["False"]["num_entities"] == 1
    assert "US" in result["by_country"]
    assert "India" in result["by_country"]
