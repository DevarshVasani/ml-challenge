"""SQLite-backed raw source store for bounded candidate-to-record joins."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping


class SourceStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> "SourceStore":
        self._connection = sqlite3.connect(self.path)
        return self

    def __exit__(self, *_: object) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _connect(self) -> sqlite3.Connection:
        return self._connection or sqlite3.connect(self.path)

    @classmethod
    def build(
        cls, paths: Mapping[str, str | Path], path: str | Path, *, batch_size: int = 10_000,
        input_identities: Mapping[str, Any] | None = None, resume: bool = True,
    ) -> "SourceStore":
        """Build with row checkpoints committed atomically with inserted rows."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return cls(target)
        working = Path(f"{target}.inprogress")
        requested = json.dumps({"paths": {str(k): str(v) for k, v in paths.items()}, "input_identities": dict(input_identities or {})}, sort_keys=True)
        if working.exists() and not resume:
            raise FileExistsError(f"in-progress source store exists: {working}")
        connection = sqlite3.connect(working)
        completed = False
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE IF NOT EXISTS records (source TEXT NOT NULL, entity_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(source, entity_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS build_progress (source TEXT PRIMARY KEY, rows_done INTEGER NOT NULL, complete INTEGER NOT NULL DEFAULT 0)")
            previous = connection.execute("SELECT value FROM build_meta WHERE key='identity'").fetchone()
            if previous is None:
                connection.execute("INSERT INTO build_meta VALUES ('identity', ?)", (requested,))
                connection.commit()
            elif str(previous[0]) != requested:
                raise ValueError(f"in-progress source store is incompatible: {working}")
            for source, raw_path in paths.items():
                progress = connection.execute("SELECT rows_done, complete FROM build_progress WHERE source=?", (str(source),)).fetchone()
                rows_done = int(progress[0]) if progress else 0
                if progress and int(progress[1]) == 1:
                    continue
                with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    batch: list[tuple[str, str, str]] = []
                    seen = 0
                    for row in reader:
                        if seen < rows_done:
                            seen += 1
                            continue
                        seen += 1
                        entity_id = str(row.get("entity_id", "")).strip()
                        if not entity_id:
                            raise ValueError(f"{raw_path} contains an empty entity_id")
                        batch.append((str(source), entity_id, json.dumps({key: ("" if value is None else str(value)) for key, value in row.items()}, ensure_ascii=False)))
                        if len(batch) >= batch_size:
                            try:
                                connection.execute("BEGIN")
                                connection.executemany("INSERT INTO records VALUES (?, ?, ?)", batch)
                                rows_done += len(batch)
                                connection.execute("INSERT INTO build_progress VALUES (?, ?, 0) ON CONFLICT(source) DO UPDATE SET rows_done=excluded.rows_done", (str(source), rows_done))
                                connection.commit()
                            except sqlite3.IntegrityError as exc:
                                connection.rollback()
                                raise ValueError(f"duplicate source/entity key while building source store: {source}") from exc
                            batch.clear()
                    if batch:
                        try:
                            connection.execute("BEGIN")
                            connection.executemany("INSERT INTO records VALUES (?, ?, ?)", batch)
                            rows_done += len(batch)
                            connection.execute("INSERT INTO build_progress VALUES (?, ?, 0) ON CONFLICT(source) DO UPDATE SET rows_done=excluded.rows_done", (str(source), rows_done))
                            connection.commit()
                        except sqlite3.IntegrityError as exc:
                            connection.rollback()
                            raise ValueError(f"duplicate source/entity key while building source store: {source}") from exc
                    connection.execute("INSERT INTO build_progress VALUES (?, ?, 1) ON CONFLICT(source) DO UPDATE SET rows_done=excluded.rows_done, complete=1", (str(source), rows_done))
                    connection.commit()
            connection.execute("CREATE INDEX IF NOT EXISTS records_entity ON records(entity_id)")
            connection.commit()
            completed = True
        finally:
            connection.close()
        if completed:
            os.replace(working, target)
        return cls(target)

    def lookup(self, source: str, entity_ids: Iterable[str]) -> dict[str, dict[str, str]]:
        ids = list(dict.fromkeys(map(str, entity_ids)))
        if not ids:
            return {}
        connection = self._connect()
        try:
            result: dict[str, dict[str, str]] = {}
            for start in range(0, len(ids), 900):
                chunk = ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(f"SELECT entity_id, payload FROM records WHERE source = ? AND entity_id IN ({placeholders})", [source, *chunk]).fetchall()
                result.update({str(entity_id): json.loads(payload) for entity_id, payload in rows})
            return result
        finally:
            if connection is not self._connection:
                connection.close()
        return result

    def lookup_many(self, requests: Mapping[str, Iterable[str]]) -> dict[tuple[str, str], dict[str, str]]:
        """Fetch a pair batch with one persistent connection and bounded IN lists."""
        result: dict[tuple[str, str], dict[str, str]] = {}
        owned = self._connection is None
        if owned:
            self._connection = sqlite3.connect(self.path)
        try:
            for source, ids in requests.items():
                for entity_id, payload in self.lookup(source, ids).items():
                    result[(str(source), entity_id)] = payload
        finally:
            if owned and self._connection is not None:
                self._connection.close()
                self._connection = None
        return result

    def resolve_source(self, entity_id: str) -> str:
        """Resolve an endpoint to one declared source, rejecting ambiguity."""
        connection = self._connect()
        try:
            rows = connection.execute("SELECT source FROM records WHERE entity_id = ? ORDER BY source", (str(entity_id),)).fetchall()
        finally:
            if connection is not self._connection:
                connection.close()
        sources = [str(row[0]) for row in rows]
        if len(sources) != 1:
            raise ValueError(f"candidate endpoint {entity_id!r} resolves to {len(sources)} sources: {sources}")
        return sources[0]

    def resolve_sources(self, entity_ids: Iterable[str]) -> dict[str, str]:
        ids = list(dict.fromkeys(map(str, entity_ids)))
        if not ids:
            return {}
        connection = self._connect()
        result: dict[str, list[str]] = {value: [] for value in ids}
        try:
            for start in range(0, len(ids), 900):
                chunk = ids[start:start + 900]
                marks = ",".join("?" for _ in chunk)
                for entity_id, source in connection.execute(f"SELECT entity_id, source FROM records WHERE entity_id IN ({marks}) ORDER BY entity_id, source", chunk):
                    result[str(entity_id)].append(str(source))
        finally:
            if connection is not self._connection:
                connection.close()
        invalid = {key: values for key, values in result.items() if len(values) != 1}
        if invalid:
            key = sorted(invalid)[0]
            raise ValueError(f"candidate endpoint {key!r} resolves to {len(invalid[key])} sources: {invalid[key]}")
        return {key: values[0] for key, values in result.items()}


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
    positives = set(map(str, ground_truth.get(sid, []))) if ground_truth is not None else None
    requests = {
        str(source): rows["candidate_entity_id"].astype(str).tolist()
        for source, rows in candidate_group.groupby("candidate_source", sort=False)
    }
    fetched = store.lookup_many(requests)
    for source, source_rows in candidate_group.groupby("candidate_source", sort=False):
        ids = source_rows["candidate_entity_id"].astype(str).tolist()
        records = {entity_id: fetched[(str(source), entity_id)] for entity_id in ids if (str(source), entity_id) in fetched}
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
                values["label"] = int(candidate_id in positives)
            for key, value in row.items():
                if key not in values and key not in {"source1_entity_id", "candidate_entity_id", "candidate_source"}:
                    values[key] = value
            output.append(values)
    import pandas as pd
    return pd.DataFrame(output)
