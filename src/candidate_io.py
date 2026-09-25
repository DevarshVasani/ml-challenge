"""Incremental long-format candidate shard reading and writing.

The public boundary is a complete-S1-group iterator.  A writer may create a
large shard from many groups, but never cuts one group at a shard boundary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from .neural_contracts import PAIR_COLUMNS, atomic_write_json, sha256_file, validate_pair_frame

GROUP_KEY = "source1_entity_id"


def _read_fragment(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def iter_complete_groups(paths: Sequence[str | os.PathLike[str]], *, reject_repeats: bool = True) -> Iterator[pd.DataFrame]:
    """Yield complete groups while carrying the last group across files/batches.

    A group that reappears after it has been yielded is malformed.  This is
    intentionally strict: returning a partial group would silently change
    candidate recall and pair counts.
    """
    carry: list[pd.DataFrame] = []
    carry_id: str | None = None
    yielded: set[str] = set()
    columns: list[str] | None = None
    for raw_path in paths:
        frame = _read_fragment(Path(raw_path))
        if GROUP_KEY not in frame.columns:
            raise ValueError(f"candidate fragment lacks {GROUP_KEY}: {raw_path}")
        if columns is None:
            columns = list(frame.columns)
        elif list(frame.columns) != columns:
            raise ValueError(f"candidate fragment schema mismatch: {raw_path}")
        if frame.empty:
            continue
        for _, group in frame.groupby(GROUP_KEY, sort=False, dropna=False):
            group_id = str(group.iloc[0][GROUP_KEY])
            if carry_id is None:
                carry_id = group_id
            if group_id != carry_id:
                result = pd.concat(carry, ignore_index=True)
                if reject_repeats and carry_id in yielded:
                    raise ValueError(f"S1 group repeated after completion: {carry_id}")
                yielded.add(carry_id)
                yield result
                carry = []
                carry_id = group_id
            carry.append(group.copy())
    if carry:
        assert carry_id is not None
        if reject_repeats and carry_id in yielded:
            raise ValueError(f"S1 group repeated after completion: {carry_id}")
        yield pd.concat(carry, ignore_index=True)


class GroupedParquetWriter:
    """Write complete S1 groups into atomic parquet shards near a row target."""

    def __init__(self, output_dir: str | os.PathLike[str], target_rows: int = 75_000, prefix: str = "pairs", *, allow_existing: bool = False):
        if target_rows <= 0:
            raise ValueError("target_rows must be positive")
        self.output_dir = Path(output_dir)
        self.target_rows = int(target_rows)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.output_dir.glob(f"{prefix}-*.parquet*"))
        if existing and not allow_existing:
            raise FileExistsError(
                f"refusing to overwrite existing {prefix} shards in {self.output_dir}; "
                "choose a new output directory or use a manifest-aware resume command"
            )
        self._groups: list[pd.DataFrame] = []
        self._rows = 0
        self._shard = 0
        self.files: list[str] = []
        self.row_counts: dict[str, int] = {}
        self.large_groups: list[dict[str, int | str]] = []
        self._seen_groups: set[str] = set()

    def write_group(self, group: pd.DataFrame) -> None:
        if group.empty:
            return
        if set(PAIR_COLUMNS).issubset(group.columns):
            validate_pair_frame(group, labeled="label" in group.columns)
        else:
            required = {"source1_entity_id", "candidate_entity_id", "candidate_source"}
            if not required.issubset(group.columns):
                raise ValueError(f"candidate group is missing required columns: {sorted(required - set(group.columns))}")
            if group.duplicated(sorted(required)).any():
                raise ValueError("duplicate candidate pair key detected")
        ids = group[GROUP_KEY].astype(str).unique()
        if len(ids) != 1:
            raise ValueError("write_group requires exactly one S1 group")
        if ids[0] in self._seen_groups:
            raise ValueError(f"S1 group was already written: {ids[0]}")
        self._seen_groups.add(ids[0])
        count = len(group)
        if count > self.target_rows:
            self.large_groups.append({"source1_entity_id": ids[0], "rows": count})
        if self._groups and self._rows + count > self.target_rows:
            self.flush()
        self._groups.append(group.copy())
        self._rows += count

    def flush(self) -> None:
        if not self._groups:
            return
        table = pd.concat(self._groups, ignore_index=True)
        final = self.output_dir / f"{self.prefix}-{self._shard:06d}.parquet"
        temporary = final.with_suffix(final.suffix + ".tmp")
        try:
            table.to_parquet(temporary, index=False)
            os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        digest = sha256_file(final)
        self.files.append(str(final))
        self.row_counts[final.name] = len(table)
        atomic_write_json(f"{final}.complete.json", {"rows": len(table), "sha256": digest, "groups": int(table[GROUP_KEY].nunique())})
        self._shard += 1
        self._groups = []
        self._rows = 0

    def close(self) -> list[str]:
        self.flush()
        return list(self.files)

    def __enter__(self) -> "GroupedParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()


def write_grouped_parquet(groups: Iterable[pd.DataFrame], output_dir: str | os.PathLike[str], target_rows: int = 75_000) -> GroupedParquetWriter:
    writer = GroupedParquetWriter(output_dir, target_rows)
    for group in groups:
        writer.write_group(group)
    writer.close()
    return writer
