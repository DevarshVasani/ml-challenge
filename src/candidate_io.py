"""Bounded-memory readers for candidate TSV and Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
from .artifact_contract import validate_candidate_partition


def _check_projection(columns: Sequence[str] | None, available: Iterable[str]) -> list[str] | None:
    available_set = set(available)
    if columns is None and "source1_entity_id" not in available_set:
        raise ValueError("Candidate input must contain source1_entity_id")
    if columns is None:
        return None
    requested = list(dict.fromkeys(columns))
    missing = [column for column in requested if column not in available_set]
    if missing:
        raise ValueError(f"Requested candidate columns are missing: {missing}")
    if "source1_entity_id" not in requested:
        raise ValueError("source1_entity_id is required to keep S1 rows together")
    return requested


def _yield_grouped(chunks: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    carry: pd.DataFrame | None = None
    completed: set[str] = set()
    for chunk in chunks:
        if chunk.empty:
            continue
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        keys = chunk["source1_entity_id"].astype(str).to_numpy()
        boundaries = [0] + [index for index in range(1, len(keys)) if keys[index] != keys[index - 1]] + [len(keys)]
        # Keep the final group as carry because its rows may continue in the
        # next scanner batch. Every preceding group is complete now.
        for start, end in zip(boundaries[:-2], boundaries[1:-1]):
            key = str(keys[start])
            if key in completed:
                raise ValueError(f"Candidate input splits non-contiguous rows for {key}")
            completed.add(key)
            yield chunk.iloc[start:end].reset_index(drop=True)
        last_start = boundaries[-2]
        carry = chunk.iloc[last_start:].copy()
    if carry is not None and not carry.empty:
        key = str(carry["source1_entity_id"].iloc[0])
        if key in completed:
            raise ValueError(f"Candidate input splits non-contiguous rows for {key}")
        yield carry.reset_index(drop=True)


def _iter_tsv(path: Path, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, sep="\t", nrows=0)
    selected = _check_projection(columns, header.columns)
    chunks = pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False, usecols=selected, chunksize=batch_size)
    yield from _yield_grouped(chunks)


def _iter_parquet(path: Path, columns: Sequence[str] | None) -> Iterator[pd.DataFrame]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parquet support requires pyarrow") from exc
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    selected = _check_projection(columns, dataset.schema.names)
    for fragment in sorted(dataset.get_fragments(), key=lambda item: str(item.path)):
        scanner = fragment.scanner(columns=selected, batch_size=65536)
        yield from _yield_grouped(batch.to_pandas() for batch in scanner.to_batches())


def iter_candidate_partitions(
    path: str | Path,
    columns: Sequence[str] | None = None,
    batch_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Iterate complete-S1 candidate partitions without full-dataset concat."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Candidate input does not exist: {source}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if source.is_file() and source.suffix.casefold() in {".tsv", ".txt"}:
        yield from _iter_tsv(source, columns, batch_size)
    elif source.is_file() and source.suffix.casefold() in {".parquet", ".pq"}:
        yield from _iter_parquet(source, columns)
    elif source.is_dir():
        if not any(source.rglob("*.parquet")):
            raise ValueError(f"Candidate directory contains no Parquet files: {source}")
        yield from _iter_parquet(source, columns)
    else:
        raise ValueError(f"Unsupported candidate input; expected TSV or Parquet: {source}")


def validate_candidate_partition_file(path: str | Path) -> None:
    for frame in iter_candidate_partitions(path):
        validate_candidate_partition(frame)


def load_candidate_manifest(path: str | Path) -> dict:
    from .artifact_contract import load_manifest
    return load_manifest(path)
