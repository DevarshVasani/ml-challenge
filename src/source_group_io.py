"""Complete source-record group streaming (Member D, §4.2).

Generalises the S1-centric iter_complete_groups in candidate_io.py to work
with any group key -- in practice (candidate_source, candidate_entity_id).

A source-record group is allowed to span multiple shards as long as:
  - it is contiguous within each shard, and
  - it appears in at most one continuous run across all shards.

Once a group is yielded it is "complete"; if rows for that group appear again
in a later shard the function raises ValueError.

Do NOT use this with the Member B bootstrap sample because it is S2/S3-incomplete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd


def _check_group_key(columns: Sequence[str], group_columns: Sequence[str]) -> None:
    missing = [c for c in group_columns if c not in columns]
    if missing:
        raise ValueError(f"Shard is missing group key columns: {missing}")


def _yield_groups_from_chunks(
    chunks: Iterable[pd.DataFrame],
    group_columns: Sequence[str],
) -> Iterator[pd.DataFrame]:
    """Yield complete groups from a stream of DataFrame chunks."""
    carry: pd.DataFrame | None = None
    completed: set[tuple] = set()

    for chunk in chunks:
        if chunk.empty:
            continue

        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None

        _check_group_key(list(chunk.columns), group_columns)

        # Build a tuple key per row
        keys = list(
            map(tuple, chunk[list(group_columns)].astype(str).to_numpy())
        )

        # Find group boundaries
        boundaries = [0]
        for i in range(1, len(keys)):
            if keys[i] != keys[i - 1]:
                boundaries.append(i)
        boundaries.append(len(keys))

        # Yield all complete groups except the last (may be incomplete)
        for start, end in zip(boundaries[:-2], boundaries[1:-1]):
            group_key = keys[start]
            if group_key in completed:
                raise ValueError(
                    f"Source-record group {group_key} reappeared after completion. "
                    "Shards must not interleave groups."
                )
            completed.add(group_key)
            yield chunk.iloc[start:end].reset_index(drop=True)

        last_start = boundaries[-2]
        carry = chunk.iloc[last_start:].copy()

    # Yield the last carry if non-empty
    if carry is not None and not carry.empty:
        group_key = tuple(carry[list(group_columns)].astype(str).iloc[0])
        if group_key in completed:
            raise ValueError(
                f"Source-record group {group_key} reappeared after completion."
            )
        yield carry.reset_index(drop=True)


def _iter_shard(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from a single shard file (parquet or TSV)."""
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        yield pd.read_parquet(path)
    elif suffix in (".tsv", ".csv"):
        sep = "\t" if suffix == ".tsv" else ","
        yield from pd.read_csv(
            path,
            sep=sep,
            dtype=object,
            keep_default_na=False,
            chunksize=batch_size,
        )
    else:
        raise ValueError(f"Unsupported shard file format: {path}")


def _validate_shard_manifests(
    shard_paths: list[Path],
    shard_manifests: Sequence[str | Path] | None,
) -> None:
    """Verify all declared shards are present and complete."""
    if shard_manifests is None:
        return
    import json

    for manifest_path in shard_manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("completion") != "complete":
            raise ValueError(
                f"Shard manifest {manifest_path} is not complete "
                f"(status: {manifest.get('completion', 'unknown')})"
            )
        declared = manifest.get("shard_order", [])
        declared_set = set(declared)
        present = {str(p.name) for p in shard_paths}
        missing = declared_set - present
        if missing:
            raise FileNotFoundError(
                f"Declared shards not found: {sorted(missing)}"
            )


def iter_complete_source_groups(
    paths: Sequence[str | Path],
    *,
    group_columns: Sequence[str] = ("candidate_source", "candidate_entity_id"),
    batch_size: int = 100_000,
    shard_manifests: Sequence[str | Path] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream complete (candidate_source, candidate_entity_id) groups across shards.

    Parameters
    ----------
    paths : sequence of file paths or a single directory
        If a directory is given, all .parquet and .tsv files are sorted and
        processed in order.
    group_columns : sequence of str
        Columns defining the group key. Default: (candidate_source, candidate_entity_id).
    batch_size : int
        Chunk size when reading TSV files.
    shard_manifests : sequence of manifest JSON paths (optional)
        When provided, validates that every declared shard is present and marked
        complete before streaming begins.

    Yields
    ------
    pd.DataFrame
        One complete group per yield, in deterministic order.

    Raises
    ------
    ValueError
        If a group reappears after completion, or manifest validation fails.
    FileNotFoundError
        If declared shards are missing.
    """
    shard_paths: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.casefold() in (".parquet", ".tsv", ".csv") and f.is_file():
                    shard_paths.append(f)
        elif p.is_file():
            shard_paths.append(p)
        else:
            raise FileNotFoundError(f"Shard path not found: {p}")

    if not shard_paths:
        return

    _validate_shard_manifests(shard_paths, shard_manifests)

    # Validate schema consistency across shards
    first_schema: list[str] | None = None
    for shard_path in shard_paths:
        if shard_path.suffix.casefold() == ".parquet":
            df_head = pd.read_parquet(shard_path).head(0)
        else:
            df_head = pd.read_csv(shard_path, sep="\t", nrows=0, dtype=object)
        schema = list(df_head.columns)
        if first_schema is None:
            first_schema = schema
        elif schema != first_schema:
            raise ValueError(
                f"Schema mismatch between shards:\n"
                f"  first shard : {first_schema}\n"
                f"  {shard_path.name}: {schema}"
            )

    def _all_chunks() -> Iterator[pd.DataFrame]:
        for shard_path in shard_paths:
            yield from _iter_shard(shard_path, batch_size)

    yield from _yield_groups_from_chunks(_all_chunks(), group_columns)
