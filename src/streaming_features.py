"""Partition-at-a-time feature materialization."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .candidate_io import iter_candidate_partitions
from .context_features import add_candidate_context_features


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def write_feature_parts(
    candidate_path: str | Path,
    output_dir: str | Path,
    columns: Sequence[str] | None = None,
    batch_size: int = 100_000,
    score_margin: float = 0.05,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    configuration: dict | None = None,
) -> dict:
    """Write one atomically-created Parquet feature part per input partition."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    configuration = dict(configuration or {})
    configuration.update({"score_margin": score_margin, "columns": list(columns) if columns else None, "batch_size": batch_size})
    digest = _config_hash(configuration)
    parts: list[dict] = []
    for index, partition in enumerate(iter_candidate_partitions(candidate_path, columns=columns, batch_size=batch_size)):
        name = f"part-{index:05d}.parquet"
        target = destination / name
        sidecar = destination / f"{name}.json"
        if target.exists() and sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if metadata.get("configuration_hash") == digest:
                parts.append(metadata)
                continue
        result = add_candidate_context_features(partition, score_margin=score_margin)
        if feature_builder is not None:
            result = feature_builder(result)
            if not isinstance(result, pd.DataFrame):
                raise TypeError("feature_builder must return a pandas DataFrame")
        temp = destination / f".{name}.{uuid.uuid4().hex}.tmp"
        result.to_parquet(temp, index=False, engine="pyarrow")
        os.replace(temp, target)
        metadata = {"part": name, "row_count": int(len(result)), "configuration_hash": digest, "configuration": configuration}
        side_temp = destination / f".{name}.json.{uuid.uuid4().hex}.tmp"
        side_temp.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        os.replace(side_temp, sidecar)
        parts.append(metadata)
    manifest = {"configuration_hash": digest, "candidate_path": str(candidate_path), "parts": parts}
    manifest_temp = destination / f".manifest.{uuid.uuid4().hex}.tmp"
    manifest_temp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(manifest_temp, destination / "feature_parts_manifest.json")
    return manifest


write_streaming_features = write_feature_parts

