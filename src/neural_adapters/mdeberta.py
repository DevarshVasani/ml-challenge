"""Member B: microsoft/mdeberta-v3-base two-logit cross-encoder.

Torch and Transformers are imported only when an adapter is constructed, so
help/dry-run commands keep working on machines without the GPU extra.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterConfig, BasePairAdapter

DEFAULT_CHECKPOINT = "microsoft/mdeberta-v3-base"
FIELD_SEPARATOR = " | "
SIDE_A = ("name_a", "address_a", "country_a")
SIDE_B = ("name_b", "address_b", "country_b")
MODEL_INPUTS = ("input_ids", "attention_mask", "token_type_ids")
ADAPTER_FILE = "adapter_config.json"
SERIALIZATION_VERSION = "mdeberta-pair-1"
PRECISION_ENV = "MDEBERTA_INFERENCE_PRECISION"


def _require_runtime():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("mDeBERTa adapter requires torch, transformers, sentencepiece and protobuf; install requirements-gpu.txt on the compute machine") from exc
    return torch, transformers


def select_inference_precision(capability: tuple[int, int] | None, override: str | None = None) -> str:
    """bf16 on Ampere+ (A10G, L4, A100), fp16 on Turing/Volta tensor cores (T4), else fp32.

    torch.cuda.is_bf16_supported() also counts software emulation, which is
    true but slow on a T4, so the choice is made from compute capability.
    ``override`` (from MDEBERTA_INFERENCE_PRECISION) forces a GPU choice.
    """
    if capability is None:
        return "fp32"
    if override and override != "auto":
        if override not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"{PRECISION_ENV} must be auto, bf16, fp16 or fp32, got {override!r}")
        return override
    return "bf16" if capability[0] >= 8 else "fp16" if capability[0] >= 7 else "fp32"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class MDebertaPairAdapter(BasePairAdapter):
    """Target: microsoft/mdeberta-v3-base, two-logit sequence-pair classifier.

    The S1 record is segment A and the candidate is segment B, each serialized
    as ``name | address | country``. Truncation is ``longest_first`` so neither
    record loses its tail while the other still has spare room.
    """

    def __init__(self, config: AdapterConfig, *, weights_dir: str | None = None, provenance: Mapping[str, Any] | None = None):
        super().__init__(config)
        torch, transformers = _require_runtime()
        self._torch = torch
        source = weights_dir or config.checkpoint or DEFAULT_CHECKPOINT
        revision = None if weights_dir else config.revision
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision, use_fast=True, local_files_only=weights_dir is not None)
        if not self.tokenizer.is_fast:
            raise RuntimeError("mDeBERTa adapter needs the fast tokenizer; install sentencepiece and protobuf so it can be converted")
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            source, revision=revision, num_labels=2, id2label={0: "no_match", 1: "match"}, label2id={"no_match": 0, "match": 1},
            local_files_only=weights_dir is not None,
        )
        self.provenance = dict(provenance) if provenance else {
            "base_checkpoint": source,
            "base_revision_requested": revision,
            "base_revision_resolved": getattr(self.model.config, "_commit_hash", None),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
        self.device = torch.device("cpu")
        self.inference_precision = "fp32"
        self.to_device(config.device)

    # -- serialization -------------------------------------------------

    def _text(self, value: Any) -> str:
        if _is_missing(value):
            return self.config.missing_text
        text = " ".join(str(value).split())
        return text if text else self.config.missing_text

    def segments(self, row: Mapping[str, Any]) -> tuple[str, str]:
        """Return (S1 text, candidate text); the single source of model input text."""
        side = lambda keys: FIELD_SEPARATOR.join(self._text(row.get(key)) for key in keys)
        return side(SIDE_A), side(SIDE_B)

    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        first, second = self.segments(row)
        return f"{first} {self.tokenizer.sep_token} {second}"

    # -- tokenization --------------------------------------------------

    def pair_lengths(self, rows: Sequence[Mapping[str, Any]]) -> list[int]:
        """Untruncated subword lengths including special tokens."""
        if not rows:
            return []
        firsts, seconds = zip(*(self.segments(row) for row in rows))
        encoded = self.tokenizer(list(firsts), list(seconds), truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def collate(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        firsts, seconds = zip(*(self.segments(row) for row in rows))
        lengths = [len(ids) for ids in self.tokenizer(list(firsts), list(seconds), truncation=False, verbose=False)["input_ids"]]
        encoded = self.tokenizer(
            list(firsts), list(seconds), truncation="longest_first", max_length=self.config.max_length,
            padding="longest", return_tensors="pt",
        )
        batch: dict[str, Any] = {key: encoded[key].to(self.device) for key in MODEL_INPUTS if key in encoded}
        batch["truncated"] = sum(length > self.config.max_length for length in lengths)
        batch["untruncated_lengths"] = lengths
        return batch

    # -- model ---------------------------------------------------------

    def logits(self, batch: Mapping[str, Any]):
        torch = self._torch
        inputs = {key: batch[key] for key in MODEL_INPUTS if key in batch}
        # The shared trainer supplies autocast while training; shared inference
        # runs without it, so eval on a GPU autocasts here (bf16, or fp16 on T4).
        if not self.model.training and self.inference_precision != "fp32":
            dtype = torch.bfloat16 if self.inference_precision == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                return self.model(**inputs).logits.float()
        return self.model(**inputs).logits

    def parameters(self):
        return self.model.parameters()

    def to_device(self, device: str) -> "MDebertaPairAdapter":
        self.device = self._torch.device(device)
        self.config.device = str(device)
        self.model.to(self.device)
        capability = self._torch.cuda.get_device_capability(self.device) if self.device.type == "cuda" else None
        self.inference_precision = select_inference_precision(capability, os.environ.get(PRECISION_ENV))
        return self

    def supports_gradient_checkpointing(self) -> bool:
        return True

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif self.model.is_gradient_checkpointing:
            self.model.gradient_checkpointing_disable()

    # -- persistence ---------------------------------------------------

    def save_pretrained(self, output_dir: str) -> None:
        from ..neural_contracts import atomic_write_json

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        atomic_write_json(path / ADAPTER_FILE, {
            "adapter": self.config_dict(),
            "serialization": {"version": SERIALIZATION_VERSION, "separator": FIELD_SEPARATOR, "side_a": list(SIDE_A), "side_b": list(SIDE_B), "truncation": "longest_first"},
            "provenance": self.provenance,
        })

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu") -> "MDebertaPairAdapter":
        path = Path(output_dir)
        saved = json.loads((path / ADAPTER_FILE).read_text(encoding="utf-8"))
        version = saved.get("serialization", {}).get("version")
        if version != SERIALIZATION_VERSION:
            raise ValueError(f"checkpoint serialization {version!r} does not match adapter {SERIALIZATION_VERSION!r}")
        known = {field.name for field in fields(AdapterConfig)}
        config = AdapterConfig(**{key: value for key, value in saved["adapter"].items() if key in known})
        config.device = map_location
        return cls(config, weights_dir=str(path), provenance=saved.get("provenance"))
