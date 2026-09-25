"""Versioned configuration for the HistGradientBoosting baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


FEATURE_SCHEMA_VERSION = "pair-features-v1"
CANDIDATE_SCHEMA_VERSION = "candidate-artifact-v1"


@dataclass(frozen=True)
class BaselineModelConfig:
    model_name: str = "hist_gradient_boosting"
    seed: int = 42
    max_iter: int = 150
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    l2_regularization: float = 0.5
    positive_weight: str | float = "none"
    negatives_per_s1: int = 10
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    candidate_schema_version: str = CANDIDATE_SCHEMA_VERSION
    feature_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_columns"] = list(self.feature_columns)
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BaselineModelConfig":
        data = dict(values)
        if "feature_columns" in data:
            data["feature_columns"] = tuple(data["feature_columns"])
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

