"""ByT5 encoder adapter for pair classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterConfig, BasePairAdapter, serialize_pair_fields


class _ByT5PairClassifier:
    """Wrapper to encapsulate encoder, dropout, and classifier to avoid nn.Module import at module level."""
    pass


class ByT5EncoderPairAdapter(BasePairAdapter):
    """ByT5 encoder-only adapter."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.device = "cpu"
        # We defer building the torch models until from_config(execute=True) or load_pretrained()
        self.model = None
        self.tokenizer = None

    @classmethod
    def from_config(cls, config: AdapterConfig, *, execute: bool = False) -> "BasePairAdapter":
        if not execute:
            raise RuntimeError("adapter construction is runtime work; use --execute on a compute machine")
        
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, T5EncoderModel
        
        class _Wrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(encoder.config.dropout_rate)
                self.classifier = nn.Linear(encoder.config.d_model, 2)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                
                pooled = (hidden * mask).sum(dim=1)
                pooled = pooled / mask.sum(dim=1).clamp_min(1.0)
                
                return self.classifier(self.dropout(pooled))

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
        instance.model = _Wrapper(encoder)
        return instance

    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        return serialize_pair_fields(row, missing_text=self.config.missing_text)

    def collate(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        import torch
        texts = [self.serialize_pair(row) for row in rows]
        
        # Tokenize without padding or truncation first to count truncation exactly
        batch_tokens = self.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        original_lengths = [len(ids) for ids in batch_tokens["input_ids"]]
        
        truncated_count = sum(1 for length in original_lengths if length > self.config.max_length)
        
        # Now truncate manually to keep EOS if needed, or simply let the tokenizer do it
        # The exact instruction: "truncate each token-id list to max_length; preserve final EOS"
        truncated_ids = []
        for ids in batch_tokens["input_ids"]:
            if len(ids) > self.config.max_length:
                # keep up to max_length - 1, then append EOS
                eos = ids[-1] if ids else 1 # default eos token id is usually 1
                truncated = ids[:self.config.max_length - 1] + [eos]
                truncated_ids.append(truncated)
            else:
                truncated_ids.append(ids)
                
        # Pad dynamically to the longest in the batch
        padded = self.tokenizer.pad(
            {"input_ids": truncated_ids}, 
            padding=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": padded["input_ids"].to(self.device),
            "attention_mask": padded["attention_mask"].to(self.device),
            "truncated": truncated_count
        }

    def logits(self, batch: Any) -> Any:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def parameters(self):
        return self.model.parameters()

    def to_device(self, device: str) -> "BasePairAdapter":
        import torch
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def supports_gradient_checkpointing(self) -> bool:
        return True

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        super().set_gradient_checkpointing(enabled)
        if enabled:
            self.model.encoder.gradient_checkpointing_enable()
        else:
            self.model.encoder.gradient_checkpointing_disable()

    def save_pretrained(self, output_dir: str) -> None:
        import torch
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        (out / "adapter_config.json").write_text(json.dumps(self.config_dict()), encoding="utf-8")
        
        metadata = {
            "architecture": "byt5-encoder-pair-v1",
            "original_checkpoint": self.config.checkpoint,
            "revision": self.config.revision,
            "max_length": self.config.max_length,
            "d_model": self.model.encoder.config.d_model,
            "output_size": 2,
            "pooling": "masked_mean",
            "serialization_separator": " || "
        }
        (out / "adapter_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        
        self.tokenizer.save_pretrained(str(out / "tokenizer"))
        self.model.encoder.save_pretrained(str(out / "encoder"))
        torch.save(self.model.classifier.state_dict(), out / "classifier.pt")

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu") -> "BasePairAdapter":
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, T5EncoderModel
        
        out = Path(output_dir)
        config_data = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
        config = AdapterConfig(**config_data)
        
        instance = cls(config)
        instance.tokenizer = AutoTokenizer.from_pretrained(str(out / "tokenizer"), use_fast=False)
        
        encoder = T5EncoderModel.from_pretrained(str(out / "encoder"))
        
        class _Wrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(encoder.config.dropout_rate)
                self.classifier = nn.Linear(encoder.config.d_model, 2)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                
                pooled = (hidden * mask).sum(dim=1)
                pooled = pooled / mask.sum(dim=1).clamp_min(1.0)
                
                return self.classifier(self.dropout(pooled))
                
        instance.model = _Wrapper(encoder)
        classifier_state = torch.load(out / "classifier.pt", map_location=map_location, weights_only=True)
        instance.model.classifier.load_state_dict(classifier_state)
        
        instance.to_device(map_location)
        return instance
