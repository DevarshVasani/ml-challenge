"""Member B skeleton; intentionally does not import or download Transformers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import AdapterConfig, BasePairAdapter


class MDebertaPairAdapter(BasePairAdapter):
    """Target: microsoft/mdeberta-v3-base, two-logit sequence-pair classifier."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        raise NotImplementedError("Member B must implement tokenizer/model loading for microsoft/mdeberta-v3-base; keep it behind explicit --execute")

    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        raise NotImplementedError("Member B: implement shared six-field pair serialization")

    def collate(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        raise NotImplementedError("Member B: implement dynamic subword padding and truncation counters")

    def logits(self, batch: Any) -> Any:
        raise NotImplementedError("Member B: return [batch, 2] logits with class 1 as match")

    def parameters(self):
        raise NotImplementedError("Member B: expose the pretrained classifier trainable parameters")

    def to_device(self, device: str):
        raise NotImplementedError("Member B: move model and tokenizer-owned tensors to device")

    def save_pretrained(self, output_dir: str) -> None:
        raise NotImplementedError("Member B: save model, tokenizer revision, and adapter configuration")

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu"):
        raise NotImplementedError("Member B: implement checkpoint reload")

