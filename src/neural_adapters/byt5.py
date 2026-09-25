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
        instance.model = _build_pair_classifier(encoder)
        return instance

    def _require_initialized(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ByT5 adapter is not initialized; use from_config(..., execute=True) or load_pretrained()")

    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        return serialize_pair_fields(row, missing_text=self.config.missing_text)

    def collate(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self._require_initialized()
        if not rows:
            raise ValueError("cannot collate an empty ByT5 batch")

        texts = [self.serialize_pair(row) for row in rows]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_id_lists = [list(ids) for ids in encoded["input_ids"]]
        max_length = int(self.config.max_length)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        original_lengths = [len(ids) for ids in input_id_lists]
        truncated = sum(length > max_length for length in original_lengths)

        clipped: list[list[int]] = []
        for ids in input_id_lists:
            if len(ids) <= max_length:
                clipped.append(ids)
                continue
            shortened = ids[:max_length]
            if eos_token_id is not None:
                shortened[-1] = int(eos_token_id)
            clipped.append(shortened)

        padded = self.tokenizer.pad(
            {"input_ids": clipped},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": padded["input_ids"].to(self.device, non_blocking=True),
            "attention_mask": padded["attention_mask"].to(self.device, non_blocking=True),
            "truncated": int(truncated),
        }

    def logits(self, batch: Mapping[str, Any]) -> Any:
        self._require_initialized()
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def parameters(self):
        self._require_initialized()
        return self.model.parameters()

    def to_device(self, device: str) -> "ByT5EncoderPairAdapter":
        import torch

        self._require_initialized()
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def supports_gradient_checkpointing(self) -> bool:
        return True

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        super().set_gradient_checkpointing(enabled)
        self._require_initialized()
        if enabled:
            self.model.encoder.gradient_checkpointing_enable()
        else:
            self.model.encoder.gradient_checkpointing_disable()

    def save_pretrained(self, output_dir: str) -> None:
        import torch
        import transformers

        self._require_initialized()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        config_payload = self.config_dict()
        (output / "adapter_config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "architecture": _ADAPTER_ARCHITECTURE,
            "original_checkpoint": self.config.checkpoint,
            "requested_revision": self.config.revision,
            "resolved_revision": _resolved_revision(self.model.encoder, self.tokenizer),
            "max_length": int(self.config.max_length),
            "d_model": int(self.model.encoder.config.d_model),
            "output_size": 2,
            "pooling": "masked_mean",
            "serialization_separator": PAIR_SERIALIZATION_SEPARATOR,
            "encoder_class": type(self.model.encoder).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        (output / "adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        self.tokenizer.save_pretrained(str(output / "tokenizer"))
        self.model.encoder.save_pretrained(str(output / "encoder"))

        classifier_state = {
            key: value.detach().cpu()
            for key, value in self.model.classifier.state_dict().items()
        }
        temporary = output / ".classifier.pt.tmp"
        torch.save(classifier_state, temporary)
        os.replace(temporary, output / "classifier.pt")

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu") -> "ByT5EncoderPairAdapter":
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        output = Path(output_dir)
        required = [
            output / "adapter_config.json",
            output / "classifier.pt",
            output / "tokenizer",
            output / "encoder",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete ByT5 adapter checkpoint; missing: {missing}")

        config = AdapterConfig(**json.loads((output / "adapter_config.json").read_text(encoding="utf-8")))
        instance = cls(config)
        instance.tokenizer = AutoTokenizer.from_pretrained(
            str(output / "tokenizer"),
            use_fast=False,
            local_files_only=True,
        )
        encoder = T5EncoderModel.from_pretrained(
            str(output / "encoder"),
            local_files_only=True,
        )
        instance.model = _build_pair_classifier(encoder)

        # Load on CPU first for predictable peak memory, then move the completed model.
        classifier_state = torch.load(output / "classifier.pt", map_location="cpu", weights_only=True)
        instance.model.classifier.load_state_dict(classifier_state)
        instance.to_device(map_location)
        return instance
