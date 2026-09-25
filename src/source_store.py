"""SQLite-backed raw source store for bounded candidate-to-record joins."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .neural_data import _column


class SourceStore:
    def __init__(self, path: str | Path):
        self.path = str(path)

    @classmethod
    def build(cls, paths: Mapping[str, str | Path], path: str | Path, *, batch_size: int = 10_000) -> "SourceStore":
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing source store: {target}")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        Path(temporary_name).unlink()
        connection = sqlite3.connect(temporary_name)
        succeeded = False
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE IF NOT EXISTS records (source TEXT NOT NULL, entity_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(source, entity_id))")
            connection.execute("CREATE INDEX IF NOT EXISTS records_entity ON records(entity_id)")
            for source, raw_path in paths.items():
                with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    batch: list[tuple[str, str, str]] = []
                    for row in reader:
                        entity_id = str(row.get("entity_id", "")).strip()
                        if not entity_id:
                            raise ValueError(f"{raw_path} contains an empty entity_id")
                        batch.append((str(source), entity_id, json.dumps({key: ("" if value is None else str(value)) for key, value in row.items()}, ensure_ascii=False)))
                        if len(batch) >= batch_size:
                            try:
                                connection.executemany("INSERT INTO records VALUES (?, ?, ?)", batch); connection.commit()
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(f"duplicate source/entity key while building source store: {source}") from exc
                            batch.clear()
                    if batch:
                        try:
                            connection.executemany("INSERT INTO records VALUES (?, ?, ?)", batch); connection.commit()
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"duplicate source/entity key while building source store: {source}") from exc
            succeeded = True
        finally:
            connection.close()
            if succeeded:
                os.replace(temporary_name, target)
            else:
                Path(temporary_name).unlink(missing_ok=True)
        return cls(target)

    def lookup(self, source: str, entity_ids: Iterable[str]) -> dict[str, dict[str, str]]:
        ids = list(dict.fromkeys(map(str, entity_ids)))
        if not ids:
            return {}
        connection = sqlite3.connect(self.path)
        try:
            result: dict[str, dict[str, str]] = {}
            for start in range(0, len(ids), 900):
                chunk = ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(f"SELECT entity_id, payload FROM records WHERE source = ? AND entity_id IN ({placeholders})", [source, *chunk]).fetchall()
                result.update({str(entity_id): json.loads(payload) for entity_id, payload in rows})
            return result
        finally:
            connection.close()

    def resolve_source(self, entity_id: str) -> str:
        """Resolve an endpoint to one declared source, rejecting ambiguity."""
        connection = sqlite3.connect(self.path)
        try:
            rows = connection.execute("SELECT source FROM records WHERE entity_id = ? ORDER BY source", (str(entity_id),)).fetchall()
        finally:
            connection.close()
        sources = [str(row[0]) for row in rows]
        if len(sources) != 1:
            raise ValueError(f"candidate endpoint {entity_id!r} resolves to {len(sources)} sources: {sources}")
        return sources[0]


def make_pair_group(candidate_group, source1_records: Mapping[str, Mapping[str, Any]], store: SourceStore, *, folds: Mapping[str, int] | None = None, ground_truth: Mapping[str, Iterable[str]] | None = None):
    """Resolve one complete candidate group; no global pair dataframe is built."""
    if candidate_group.empty:
        return candidate_group.copy()
    sid = str(candidate_group.iloc[0]["source1_entity_id"])
    if any(str(value) != sid for value in candidate_group["source1_entity_id"]):
        raise ValueError("candidate_group must contain one S1")
    if sid not in source1_records:
        raise ValueError(f"missing Source 1 endpoint: {sid}")
    output = []
    for source, source_rows in candidate_group.groupby("candidate_source", sort=False):
        ids = source_rows["candidate_entity_id"].astype(str).tolist()
        records = store.lookup(str(source), ids)
        missing = set(ids) - set(records)
        if missing:
            raise ValueError(f"candidate endpoints missing from source store: {sorted(missing)[:5]}")
        left = source1_records[sid]
        for row in source_rows.to_dict(orient="records"):
            candidate_id = str(row["candidate_entity_id"]); right = records[candidate_id]
            values = {
                "source1_entity_id": sid, "candidate_entity_id": candidate_id, "candidate_source": str(source),
                "name_a": str(left.get("business_name", left.get("name", "")) or ""), "address_a": str(left.get("business_address", left.get("address", "")) or ""), "country_a": str(left.get("country", "") or ""),
                "name_b": str(right.get("business_name", right.get("name", "")) or ""), "address_b": str(right.get("business_address", right.get("address", "")) or ""), "country_b": str(right.get("country", "") or ""),
            }
            if folds is not None:
                if sid not in folds or candidate_id not in folds: raise ValueError(f"missing fold endpoint for {sid}/{candidate_id}")
                values["source1_fold"], values["candidate_fold"] = int(folds[sid]), int(folds[candidate_id])
            if ground_truth is not None:
                values["label"] = int(candidate_id in set(map(str, ground_truth.get(sid, []))))
            for key, value in row.items():
                if key not in values and key not in {"source1_entity_id", "candidate_entity_id", "candidate_source"}:
                    values[key] = value
            output.append(values)
    import pandas as pd
    return pd.DataFrame(output)
