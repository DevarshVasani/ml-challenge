"""Incremental candidate shard IO with generic complete-group streaming."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from .artifact_contract import validate_candidate_partition
from .neural_contracts import PAIR_COLUMNS, atomic_write_json, sha256_file, validate_pair_frame

GROUP_KEY = "source1_entity_id"


def _check_projection(columns: Sequence[str] | None, available: Iterable[str]) -> list[str] | None:
    available_set = set(available)
    if columns is None and GROUP_KEY not in available_set: raise ValueError(f"Candidate input must contain {GROUP_KEY}")
    if columns is None: return None
    requested = list(dict.fromkeys(columns)); missing = [c for c in requested if c not in available_set]
    if missing: raise ValueError(f"Requested candidate columns are missing: {missing}")
    if GROUP_KEY not in requested: raise ValueError(f"{GROUP_KEY} is required to keep S1 rows together")
    return requested


def _yield_s1_grouped(chunks: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    carry = None; completed: set[str] = set()
    for chunk in chunks:
        if chunk.empty: continue
        if carry is not None: chunk = pd.concat([carry, chunk], ignore_index=True); carry = None
        keys = chunk[GROUP_KEY].astype(str).to_numpy(); boundaries = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i-1]] + [len(keys)]
        for start, end in zip(boundaries[:-2], boundaries[1:-1]):
            key = str(keys[start])
            if key in completed: raise ValueError(f"Candidate input splits non-contiguous rows for {key}")
            completed.add(key); yield chunk.iloc[start:end].reset_index(drop=True)
        carry = chunk.iloc[boundaries[-2]:].copy()
    if carry is not None and not carry.empty:
        key = str(carry[GROUP_KEY].iloc[0])
        if key in completed: raise ValueError(f"Candidate input splits non-contiguous rows for {key}")
        yield carry.reset_index(drop=True)


def _iter_tsv(path: Path, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, sep="\t", nrows=0); selected = _check_projection(columns, header.columns)
    yield from _yield_s1_grouped(pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False, usecols=selected, chunksize=batch_size))


def _iter_parquet(path: Path, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    try: import pyarrow.dataset as ds
    except ImportError as exc: raise RuntimeError("Parquet support requires pyarrow") from exc
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive"); selected = _check_projection(columns, dataset.schema.names)
    def batches():
        for fragment in sorted(dataset.get_fragments(), key=lambda x: str(x.path)):
            for batch in fragment.scanner(columns=selected, batch_size=batch_size).to_batches(): yield batch.to_pandas()
    yield from _yield_s1_grouped(batches())


def iter_candidate_partitions(path: str | Path, columns: Sequence[str] | None = None, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
    source = Path(path)
    if not source.exists(): raise FileNotFoundError(source)
    if source.is_file() and source.suffix.lower() in {".tsv", ".txt"}: yield from _iter_tsv(source, columns, batch_size)
    elif source.is_file() and source.suffix.lower() in {".parquet", ".pq"}: yield from _iter_parquet(source, columns, batch_size)
    elif source.is_dir(): yield from _iter_parquet(source, columns, batch_size)
    else: raise ValueError(f"Unsupported candidate input: {source}")


def _read_fragment(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False)


def iter_complete_groups(
    paths: Sequence[str | os.PathLike[str]],
    *,
    group_columns: Sequence[str] = (GROUP_KEY,),
    reject_repeats: bool = True,
) -> Iterator[pd.DataFrame]:
    """Yield complete contiguous groups for one or more group columns across files."""
    group_columns = tuple(group_columns)
    carry: list[pd.DataFrame] = []; carry_key: tuple[str, ...] | None = None; yielded: set[tuple[str, ...]] = set(); columns = None
    for raw in paths:
        frame = _read_fragment(Path(raw))
        missing = [c for c in group_columns if c not in frame.columns]
        if missing: raise ValueError(f"Candidate fragment lacks group columns {missing}: {raw}")
        if columns is None: columns = list(frame.columns)
        elif list(frame.columns) != columns: raise ValueError(f"Candidate fragment schema mismatch: {raw}")
        if frame.empty: continue
        keys = list(map(tuple, frame[list(group_columns)].astype(str).to_numpy())); start = 0
        for i in range(1, len(frame) + 1):
            if i != len(frame) and keys[i] == keys[i-1]: continue
            key = keys[start]; piece = frame.iloc[start:i].copy()
            if carry_key is None: carry_key = key
            if key != carry_key:
                if reject_repeats and carry_key in yielded: raise ValueError(f"Group repeated after completion: {carry_key}")
                yielded.add(carry_key); yield pd.concat(carry, ignore_index=True); carry = []; carry_key = key
            carry.append(piece); start = i
    if carry:
        assert carry_key is not None
        if reject_repeats and carry_key in yielded: raise ValueError(f"Group repeated after completion: {carry_key}")
        yield pd.concat(carry, ignore_index=True)


def iter_complete_source_groups(paths: Sequence[str | os.PathLike[str]], *, reject_repeats: bool = True) -> Iterator[pd.DataFrame]:
    yield from iter_complete_groups(paths, group_columns=("candidate_source", "candidate_entity_id"), reject_repeats=reject_repeats)


def validate_candidate_partition_file(path: str | Path) -> None:
    for frame in iter_candidate_partitions(path): validate_candidate_partition(frame)


def load_candidate_manifest(path: str | Path) -> dict:
    from .artifact_contract import load_manifest
    return load_manifest(path)


class GroupedParquetWriter:
    def __init__(self, output_dir: str | os.PathLike[str], target_rows: int = 75_000, prefix: str = "pairs", *, allow_existing: bool = False):
        self.output_dir = Path(output_dir); self.target_rows = int(target_rows); self.prefix = prefix; self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.target_rows <= 0: raise ValueError("target_rows must be positive")
        if list(self.output_dir.glob(f"{prefix}-*.parquet*")) and not allow_existing: raise FileExistsError(f"Existing {prefix} shards in {self.output_dir}")
        self._groups=[]; self._rows=0; self._shard=0; self.files=[]; self.row_counts={}; self.large_groups=[]; self._seen_groups=set()
    def write_group(self, group: pd.DataFrame) -> None:
        if group.empty: return
        if set(PAIR_COLUMNS).issubset(group.columns): validate_pair_frame(group, labeled="label" in group.columns)
        ids = group[GROUP_KEY].astype(str).unique()
        if len(ids) != 1: raise ValueError("write_group requires exactly one S1 group")
        if ids[0] in self._seen_groups: raise ValueError(f"S1 group already written: {ids[0]}")
        self._seen_groups.add(ids[0]); count=len(group)
        if self._groups and self._rows + count > self.target_rows: self.flush()
        self._groups.append(group.copy()); self._rows += count
    def flush(self) -> None:
        if not self._groups: return
        table=pd.concat(self._groups, ignore_index=True); final=self.output_dir/f"{self.prefix}-{self._shard:06d}.parquet"; tmp=final.with_suffix(final.suffix+".tmp")
        table.to_parquet(tmp,index=False); os.replace(tmp,final); digest=sha256_file(final); self.files.append(str(final)); self.row_counts[final.name]=len(table)
        atomic_write_json(f"{final}.complete.json", {"rows":len(table),"sha256":digest,"groups":int(table[GROUP_KEY].nunique())}); self._shard+=1; self._groups=[]; self._rows=0
    def close(self): self.flush(); return list(self.files)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None: self.close()


def write_grouped_parquet(groups: Iterable[pd.DataFrame], output_dir: str | os.PathLike[str], target_rows: int = 75_000) -> GroupedParquetWriter:
    writer=GroupedParquetWriter(output_dir,target_rows)
    for group in groups: writer.write_group(group)
    writer.close(); return writer
