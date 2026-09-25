"""Encoder-only ByT5 adapter for binary business-entity pair classification.

Transformers and Torch are imported only inside runtime paths so importing the
package, ``--help``, and ``--dry-run`` remain cheap and network-free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import (
    AdapterConfig,
    BasePairAdapter,
    PAIR_SERIALIZATION_SEPARATOR,
    serialize_pair_fields,
)


_ADAPTER_ARCHITECTURE = "byt5-encoder-pair-v1"


def _validate_config(config: AdapterConfig) -> None:
    if not config.checkpoint:
        raise ValueError("ByT5 requires a pretrained checkpoint")
    if int(config.max_length) < 2:
        raise ValueError("ByT5 max_length must be at least 2 so content and EOS can be represented")


def _masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool sequence states while excluding all padding positions."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * mask).sum(dim=1) / denominator


def _build_pair_classifier(encoder: Any) -> Any:
    """Build the small trainable classification wrapper around a T5 encoder."""
    import torch.nn as nn

    class _ByT5PairClassifier(nn.Module):
        def __init__(self, encoder_model: Any):
            super().__init__()
            self.encoder = encoder_model
            dropout_rate = float(getattr(encoder_model.config, "dropout_rate", 0.1))
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(int(encoder_model.config.d_model), 2)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _masked_mean_pool(output.last_hidden_state, attention_mask)
            return self.classifier(self.dropout(pooled))

    return _ByT5PairClassifier(encoder)


def _resolved_revision(encoder: Any, tokenizer: Any) -> str | None:
    """Best-effort resolved HF commit hash for reproducibility metadata."""
    model_hash = getattr(getattr(encoder, "config", None), "_commit_hash", None)
    if model_hash:
        return str(model_hash)
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    token_hash = init_kwargs.get("_commit_hash")
    return str(token_hash) if token_hash else None


class ByT5EncoderPairAdapter(BasePairAdapter):
    """``google/byt5-small`` encoder + masked mean pooling + two-logit head."""

    def __init__(self, config: AdapterConfig):
        _validate_config(config)
        super().__init__(config)
        self.device: Any = "cpu"
        self.model: Any = None
        self.tokenizer: Any = None

    @classmethod
    def from_config(cls, config: AdapterConfig, *, execute: bool = False) -> "ByT5EncoderPairAdapter":
        if not execute:
            raise RuntimeError("adapter construction is runtime work; use --execute on a compute machine")
        _validate_config(config)

        from transformers import AutoTokenizer, T5EncoderModel

        instance = cls(config)
        instance.tokenizer = AutoTokenizer.from_pretrained(
            config.checkpoint,
            revision=config.revision,
            use_fast=False,
        )
        encoder = T5EncoderModel.from_pretrained(
            config.checkpoint,
            revision=config.revision,
        )