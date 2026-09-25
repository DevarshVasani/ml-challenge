"""Small deterministic adapter used only by offline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterConfig, BasePairAdapter

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on minimal CPU installs
    torch = None
    nn = object  # type: ignore


class FakePairAdapter(BasePairAdapter):
    def __init__(self, config: AdapterConfig | None = None, *, seed: int = 0):
        if torch is None:
            raise RuntimeError("FakePairAdapter requires an existing PyTorch installation; do not install it for dry-run")
        super().__init__(config or AdapterConfig("fake", checkpoint="offline-fake", max_length=64))
        torch.manual_seed(seed)
        self.model = nn.Linear(8, 2)
        self.device = torch.device(self.config.device)

    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        return " || ".join(str(row.get(field, "") or "") for field in ("name_a", "address_a", "country_a", "name_b", "address_b", "country_b"))

    def _features(self, row: Mapping[str, Any]) -> list[float]:
        text = self.serialize_pair(row)
        fields = [str(row.get(field, "") or "") for field in ("name_a", "address_a", "country_a", "name_b", "address_b", "country_b")]
        return [float(len(value)) / 100.0 for value in fields] + [float(text.count(" || ")), float(sum(ch.isdigit() for ch in text)) / 20.0]

    def collate(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = torch.tensor([self._features(row) for row in rows], dtype=torch.float32, device=self.device)
        # The fake adapter has no tokenizer; these fields mirror the real batch contract.
        return {"features": values, "attention_mask": torch.ones((len(rows), 1), dtype=torch.long, device=self.device), "truncated": 0}

    def logits(self, batch: Mapping[str, Any]):
        return self.model(batch["features"])

    def parameters(self):
        return self.model.parameters()

    def to_device(self, device: str) -> "FakePairAdapter":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def save_pretrained(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "fake_model.pt")
        (path / "adapter_config.json").write_text(json.dumps(self.config_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu") -> "FakePairAdapter":
        path = Path(output_dir)
        config = AdapterConfig(**json.loads((path / "adapter_config.json").read_text(encoding="utf-8")))
        adapter = cls(config)
        adapter.model.load_state_dict(torch.load(path / "fake_model.pt", map_location=map_location, weights_only=True))
        return adapter

