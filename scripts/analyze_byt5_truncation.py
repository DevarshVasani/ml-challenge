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

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        use_fast=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("ByT5 tokenizer unexpectedly has no EOS token")

    manifest_path = Path(args.pair_manifest)
    pair_dir = Path(args.pair_dir)
    shards = _manifest_shards(manifest_path, pair_dir)

    lengths = array("I")
    total_pairs = 0
    truncated_pairs = 0
    cut_locations: Counter[str] = Counter()
    address_a_affected = 0
    address_b_affected = 0
    country_totals: Counter[str] = Counter()
    country_truncated: Counter[str] = Counter()

    for shard in tqdm(shards, desc="ByT5 truncation shards"):
        frame = pd.read_parquet(shard)
        total_pairs += len(frame)
        if "country_a" in frame.columns:
            for value, count in frame["country_a"].fillna("").astype(str).value_counts(dropna=False).items():
                country_totals[str(value)] += int(count)

        for rows in _iter_batches(frame, args.batch_size):
            texts = [serialize_pair_fields(row) for row in rows]
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]

            for row, ids in zip(rows, encoded):
                token_length = len(ids)
                lengths.append(token_length)
                if token_length <= args.max_length:
                    continue

                truncated_pairs += 1
                country_truncated[str(row.get("country_a", "") or "")] += 1
                # ByT5 maps raw UTF-8 content bytes directly; one position is kept for EOS.
                location, a_affected, b_affected = _utf8_cut_location(row, args.max_length - 1)
                cut_locations[location] += 1
                address_a_affected += int(a_affected)
                address_b_affected += int(b_affected)

    if total_pairs != len(lengths):
        raise RuntimeError(f"internal row-count mismatch: read {total_pairs}, measured {len(lengths)}")

    length_values = np.frombuffer(lengths, dtype=np.uint32)
    if len(length_values):
        percentiles = np.percentile(length_values, [50, 90, 95, 99])
        length_stats = {
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p99": float(percentiles[3]),
            "max": int(length_values.max()),
        }
    else:
        length_stats = {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}

    by_country = {}
    for country in sorted(country_totals):
        total = int(country_totals[country])
        cut = int(country_truncated[country])
        by_country[country] = {
            "pairs": total,
            "truncated_pairs": cut,
            "truncation_rate": cut / total if total else 0.0,
        }

    report = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "max_length": args.max_length,
        "pair_manifest": str(manifest_path),
        "pair_dir": str(pair_dir),
        "total_pairs": total_pairs,
        "truncated_pairs": truncated_pairs,
        "truncation_rate": truncated_pairs / total_pairs if total_pairs else 0.0,
        "token_length_stats": length_stats,
        "cut_locations": dict(sorted(cut_locations.items())),
        "address_a_affected_pairs": address_a_affected,
        "address_b_affected_pairs": address_b_affected,
        "address_a_affected_rate": address_a_affected / total_pairs if total_pairs else 0.0,
        "address_b_affected_rate": address_b_affected / total_pairs if total_pairs else 0.0,
        "by_country": by_country,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
