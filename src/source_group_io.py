"""Strict complete source-record group streaming for Member D."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from .neural_contracts import sha256_file

DEFAULT_GROUP_COLUMNS = ("candidate_source", "candidate_entity_id")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=object, keep_default_na=False)
    raise ValueError(f"Unsupported shard format: {path}")


def _manifest_declared_files(manifest: dict) -> dict[str, str | None]:
    if "generated_file_checksums" in manifest:
        return {str(k): str(v) for k, v in manifest["generated_file_checksums"].items()}
    if "files" in manifest:
        return {str(k): str(v) for k, v in manifest["files"].items()}
    return {str(x): None for x in manifest.get("shard_order", [])}


def validate_source_complete_manifests(
    shard_paths: Sequence[str | Path], manifest_paths: Sequence[str | Path]
) -> None:
    present = {Path(p).name: Path(p) for p in shard_paths}
    declared: dict[str, str | None] = {}
    for raw in manifest_paths:
        path = Path(raw)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        state = manifest.get("completion_state", manifest.get("completion"))
        if state != "complete":
            raise ValueError(f"Incomplete shard manifest: {path}")
        if manifest.get("complete_by_source_record") is not True:
            raise ValueError(f"Manifest does not guarantee complete source-record groups: {path}")
        for name, checksum in _manifest_declared_files(manifest).items():
            base = Path(name).name
            if base in declared and declared[base] != checksum:
                raise ValueError(f"Conflicting duplicate shard declaration: {base}")
            declared[base] = checksum
    missing = sorted(set(declared) - set(present))
    extra = sorted(set(present) - set(declared))
    if missing:
        raise FileNotFoundError(f"Declared source-complete shards missing: {missing}")
    if extra:
        raise ValueError(f"Undeclared shards supplied to source-complete stream: {extra}")
    for name, expected in declared.items():
        if expected and sha256_file(present[name]) != expected:
            raise ValueError(f"Checksum mismatch for source-complete shard: {present[name]}")


def iter_complete_source_groups(
    paths: Sequence[str | Path],
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    manifest_paths: Sequence[str | Path] | None = None,
) -> Iterator[pd.DataFrame]:
    shard_paths = [Path(p) for p in paths]
    if not shard_paths:
        return
    for p in shard_paths:
        if not p.exists():
            raise FileNotFoundError(p)
    if manifest_paths is not None:
        validate_source_complete_manifests(shard_paths, manifest_paths)

    schema: list[str] | None = None
    carry: list[pd.DataFrame] = []
    carry_key: tuple[str, ...] | None = None
    completed: set[tuple[str, ...]] = set()

    def flush() -> pd.DataFrame | None:
        nonlocal carry, carry_key
        if not carry:
            return None
        assert carry_key is not None
        if carry_key in completed:
            raise ValueError(f"Source group reappeared after completion: {carry_key}")
        completed.add(carry_key)
        result = pd.concat(carry, ignore_index=True)
        carry = []
        carry_key = None
        return result

    for path in shard_paths:
        frame = _read(path)
        if schema is None:
            schema = list(frame.columns)
        elif list(frame.columns) != schema:
            raise ValueError(f"Shard schema mismatch: {path}")
        missing = [c for c in group_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Shard missing source-group columns {missing}: {path}")
        if frame.empty:
            continue
        keys = list(map(tuple, frame[list(group_columns)].astype(str).to_numpy()))
        start = 0
        for i in range(1, len(frame) + 1):
            boundary = i == len(frame) or keys[i] != keys[i - 1]
            if not boundary:
                continue
            key = keys[start]
            piece = frame.iloc[start:i].copy()
            if carry_key is None:
                carry_key = key
            if key != carry_key:
                result = flush()
                if result is not None:
                    yield result
                carry_key = key
            carry.append(piece)
            start = i
    result = flush()
    if result is not None:
        yield result
