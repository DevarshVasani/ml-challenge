"""Measure ByT5 byte-token lengths and where a 512-token budget cuts fields.

This is a diagnostic only. It reuses the exact shared pair serializer and never
changes candidate sets, labels, or query splits.
"""

from __future__ import annotations

import argparse
import json
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.neural_adapters.base import PAIR_SERIALIZATION_SEPARATOR, PAIR_TEXT_FIELDS, serialize_pair_fields


def _iter_batches(frame: pd.DataFrame, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    columns = list(PAIR_TEXT_FIELDS)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"pair shard is missing text fields: {missing}")
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size][columns].to_dict(orient="records")


def _utf8_cut_location(row: dict[str, Any], content_budget: int) -> tuple[str, bool, bool]:
    """Return exact serialized-field cut location for ByT5's raw UTF-8 content.

    ``content_budget`` excludes the final EOS token. Separators are accounted for
    explicitly. Address flags mean at least part of that address is unavailable
    after truncation, including when truncation occurs before the field begins.
    """
    separator_bytes = len(PAIR_SERIALIZATION_SEPARATOR.encode("utf-8"))
    cursor = 0
    boundaries: dict[str, tuple[int, int]] = {}

    for index, field in enumerate(PAIR_TEXT_FIELDS):
        value = str(row.get(field, "") or "")
        field_bytes = len(value.encode("utf-8"))
        start, end = cursor, cursor + field_bytes
        boundaries[field] = (start, end)
        if content_budget < end:
            location = field
            break
        cursor = end
        if index < len(PAIR_TEXT_FIELDS) - 1:
            separator_end = cursor + separator_bytes
            if content_budget < separator_end:
                location = f"separator_after_{field}"
                break
            cursor = separator_end
    else:
        location = "after_serialized_text"

    address_a_end = boundaries.get("address_a", (0, 0))[1]
    address_b_end = boundaries.get("address_b", (0, 0))[1]
    address_a_affected = content_budget < address_a_end
    address_b_affected = content_budget < address_b_end
    return location, address_a_affected, address_b_affected


def _manifest_shards(manifest_path: Path, pair_dir: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest.get("shard_order", []))
    if not names:
        raise ValueError(f"pair manifest has no shard_order: {manifest_path}")
    shards = [pair_dir / name for name in names]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pair manifest references missing shards: {missing[:5]}")
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ByT5 truncation on existing pair shards.")
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--checkpoint", default="google/byt5-small")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.max_length < 2:
        raise ValueError("max-length must be at least 2")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    from transformers import AutoTokenizer