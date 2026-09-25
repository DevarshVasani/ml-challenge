"""Explicit candidate index/query separation for future scalable retrieval."""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    from .candidates import sp_matmul_topn

from .neural_contracts import atomic_write_json, config_hash, sha256_file
from .text_utils import normalize_text
from .candidate_io import GroupedParquetWriter


def _index_semantics(config: Any) -> dict[str, Any]:
    """Fields that require rebuilding vocabulary/source vectors."""
    value = asdict(config)
    return {key: value[key] for key in ("field", "analyzer", "ngram_min", "ngram_max", "min_df", "max_vocabulary")}


class ExactMatchIndex:
    """Bounded-by-query exact expansion index; empty keys are never indexed."""

    def __init__(self, source: pd.DataFrame, field: str, *, candidate_source: str, source_identity: Mapping[str, Any] | None = None):
        if field not in source.columns:
            raise ValueError(f"source is missing exact field {field}")
        self.field = field
        self.candidate_source = candidate_source
        self.source_identity = dict(source_identity or {})
        self.mapping: dict[str, list[str]] = {}
        for entity_id, value in zip(source["entity_id"].astype(str), source[field].map(normalize_text)):
            if value:
                self.mapping.setdefault(value, []).append(entity_id)
        self.frequent_keys = sorted(((key, len(ids)) for key, ids in self.mapping.items() if len(ids) > 100), key=lambda x: (-x[1], x[0]))

    def query(self, s1: pd.DataFrame, field: str | None = None) -> pd.DataFrame:
        field = field or self.field
        if field not in s1:
            raise ValueError(f"query is missing exact field {field}")
        rows: list[dict[str, Any]] = []
        for source1_id, value in zip(s1["entity_id"].astype(str), s1[field].map(normalize_text)):
            if not value:
                continue
            for rank, candidate_id in enumerate(self.mapping.get(value, []), start=1):
                rows.append({"source1_entity_id": source1_id, "candidate_entity_id": candidate_id, "candidate_source": self.candidate_source, "score": 1.0, "rank": rank, "retrieval_provenance": "exact", "frequent_key_expansion": int(len(self.mapping.get(value, [])) > 100)})
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank", "retrieval_provenance", "frequent_key_expansion"])

    def save(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "exact_index.pkl"
        temporary = path / ".exact_index.pkl.tmp"
        with temporary.open("wb") as handle:
            pickle.dump({"field": self.field, "candidate_source": self.candidate_source, "source_identity": self.source_identity, "mapping": self.mapping, "frequent_keys": self.frequent_keys}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        atomic_write_json(path / "exact_metadata.json", {
            "index_schema": "candidate-exact-index-1",
            "field": self.field,
            "candidate_source": self.candidate_source,
            "source_identity": self.source_identity,
            "nonempty_keys": len(self.mapping),
            "frequent_keys": [{"normalized_key": key, "records": count} for key, count in self.frequent_keys[:100]],
        })

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> "ExactMatchIndex":
        with (Path(directory) / "exact_index.pkl").open("rb") as handle:
            value = pickle.load(handle)
        obj = cls.__new__(cls)
        obj.field = value["field"]
        obj.candidate_source = value["candidate_source"]
        obj.source_identity = value.get("source_identity", {})
        obj.mapping = value["mapping"]
        obj.frequent_keys = value["frequent_keys"]
        return obj


class SqliteExactMatchIndex:
    """Disk-backed exact postings index for machines where a pickle is too large."""

    schema = "candidate-exact-sqlite-1"

    def __init__(self, path: str | os.PathLike[str], *, field: str, candidate_source: str):
        self.path = str(path)
        self.field = field
        self.candidate_source = candidate_source
        self._connection: sqlite3.Connection | None = None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @classmethod
    def build(cls, source: pd.DataFrame, path: str | os.PathLike[str], field: str, *, candidate_source: str, source_identity: Mapping[str, Any] | None = None) -> "SqliteExactMatchIndex":
        if field not in source.columns:
            raise ValueError(f"source is missing exact field {field}")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        connection = sqlite3.connect(target)
        try:
            connection.execute("CREATE TABLE postings (normalized_key TEXT NOT NULL, entity_id TEXT NOT NULL)")
            rows = ((normalize_text(value), str(entity_id)) for entity_id, value in zip(source["entity_id"], source[field]))
            connection.executemany("INSERT INTO postings VALUES (?, ?)", ((key, entity) for key, entity in rows if key))
            connection.execute("CREATE INDEX postings_key ON postings(normalized_key)")
            connection.commit()
        finally:
            connection.close()
        atomic_write_json(target.with_suffix(".json"), {
            "index_schema": cls.schema,
            "field": field,
            "candidate_source": candidate_source,
            "source_rows": len(source),
            "source_identity": dict(source_identity or {}),
        })
        return cls(target, field=field, candidate_source=candidate_source)

    @classmethod
    def build_from_tsv(
        cls, source_path: str | os.PathLike[str], path: str | os.PathLike[str], field: str,
        *, candidate_source: str, source_identity: Mapping[str, Any], chunk_rows: int = 25_000, resume: bool = True,
    ) -> "SqliteExactMatchIndex":
        """Build exact postings without materializing the source table in RAM."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not resume:
            raise FileExistsError(target)
        connection = sqlite3.connect(target)
        requested = json.dumps({"field": field, "candidate_source": candidate_source, "source_identity": dict(source_identity)}, sort_keys=True)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS postings (normalized_key TEXT NOT NULL, entity_id TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            prior = connection.execute("SELECT value FROM build_meta WHERE key='identity'").fetchone()
            if prior is None:
                if connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]:
                    raise ValueError(f"unrecognized partial exact index: {target}")
                connection.execute("INSERT INTO build_meta VALUES ('identity', ?)", (requested,))
                connection.execute("INSERT INTO build_meta VALUES ('rows_seen', '0')")
                connection.commit()
            elif str(prior[0]) != requested:
                raise ValueError(f"in-progress exact index is incompatible: {target}")
            rows_seen = int(connection.execute("SELECT value FROM build_meta WHERE key='rows_seen'").fetchone()[0])
            offset = 0
            for chunk in pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, usecols=["entity_id", field], chunksize=chunk_rows):
                end = offset + len(chunk)
                if end <= rows_seen:
                    offset = end
                    continue
                if offset < rows_seen:
                    chunk = chunk.iloc[rows_seen - offset:].copy()
                values = ((normalize_text(value), str(entity_id)) for entity_id, value in zip(chunk["entity_id"], chunk[field]))
                connection.execute("BEGIN")
                connection.executemany("INSERT INTO postings VALUES (?, ?)", ((key, entity_id) for key, entity_id in values if key))
                rows_seen += len(chunk)
                connection.execute("UPDATE build_meta SET value=? WHERE key='rows_seen'", (str(rows_seen),))
                connection.commit()
                offset = end
            connection.execute("CREATE INDEX IF NOT EXISTS postings_key ON postings(normalized_key)")
            connection.commit()
        except BaseException:
            connection.close()
            raise
        finally:
            try:
                connection.close()
            except Exception:
                pass
        atomic_write_json(target.with_suffix(".json"), {
            "index_schema": cls.schema, "field": field, "candidate_source": candidate_source,
            "source_rows": rows_seen, "source_identity": dict(source_identity),
        })
        return cls(target, field=field, candidate_source=candidate_source)

    def query(self, s1: pd.DataFrame, field: str | None = None) -> pd.DataFrame:
        field = field or self.field
        if field not in s1:
            raise ValueError(f"query is missing exact field {field}")
        requests = [(str(entity_id), normalize_text(value)) for entity_id, value in zip(s1["entity_id"], s1[field])]
        keys = sorted({key for _, key in requests if key})
        if self._connection is None:
            self._connection = sqlite3.connect(self.path)
        connection = self._connection
        try:
            postings: dict[str, list[str]] = {}
            for start in range(0, len(keys), 900):
                chunk = keys[start:start + 900]
                if not chunk:
                    continue
                marks = ",".join("?" for _ in chunk)
                for key, entity_id in connection.execute(f"SELECT normalized_key, entity_id FROM postings WHERE normalized_key IN ({marks}) ORDER BY normalized_key, entity_id", chunk):
                    postings.setdefault(str(key), []).append(str(entity_id))
        finally:
            pass
        rows = []
        for source_id, key in requests:
            for rank, entity_id in enumerate(postings.get(key, []), 1):
                rows.append({"source1_entity_id": source_id, "candidate_entity_id": entity_id, "candidate_source": self.candidate_source, "score": 1.0, "rank": rank, "retrieval_provenance": "exact", "frequent_key_expansion": int(len(postings.get(key, [])) > 100)})
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank", "retrieval_provenance", "frequent_key_expansion"])


class DiskBackedTfidfIndex:
    """Chunked source matrix index with a globally fitted TF-IDF model.

    The vectorizer is fitted once using the complete source corpus.  Only one
    transformed source shard is materialized at a time during build and query
    keeps only a bounded top-k accumulator per query.
    """

    schema = "candidate-tfidf-disk-1"

    def __init__(self, directory: str | os.PathLike[str], *, expected_config: IndexConfig | None = None):
        path = Path(directory)
        metadata = json.loads((path / "disk_metadata.json").read_text(encoding="utf-8"))
        expected_hash = (config_hash(_index_semantics(expected_config)) if metadata.get("index_semantics_hash") else config_hash(asdict(expected_config))) if expected_config is not None else None
        actual_hash = metadata.get("index_semantics_hash", metadata.get("configuration_hash"))
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError("disk-backed index configuration changed; rebuild required")
        self.path = path
        self.config = IndexConfig(**metadata["config"])
        self.field = self.config.field
        self.candidate_source = metadata["candidate_source"]
        with (path / "vectorizer.pkl").open("rb") as handle:
            self.vectorizer = pickle.load(handle)
        self.metadata = metadata
        self.shard_specs = list(metadata.get("shards", []))
        # Compatibility with the short-lived v1 prototype.  It remains
        # readable, but new builds use mmap-able NPY arrays.
        if not self.shard_specs:
            offset = 0
            for matrix_file, ids_file in zip(sorted(path.glob("matrix-*.npz")), sorted(path.glob("ids-*.json"))):
                with np.load(matrix_file) as archive:
                    shape = tuple(int(v) for v in archive["shape"])
                self.shard_specs.append({"legacy_matrix": matrix_file.name, "ids": ids_file.name, "row_start": offset, "rows": shape[0], "features": shape[1]})
                offset += shape[0]
        if sum(int(spec["rows"]) for spec in self.shard_specs) != int(metadata["source_rows"]):
            raise ValueError("disk-backed index row/id coverage is inconsistent")

    @classmethod
    def build(
        cls, source: pd.DataFrame, directory: str | os.PathLike[str], config: IndexConfig,
        *, candidate_source: str, source_identity: Mapping[str, Any] | None = None,
        source_chunk_rows: int = 25_000,
    ) -> "DiskBackedTfidfIndex":
        if config.field not in source.columns:
            raise ValueError(f"source is missing retrieval field {config.field}")
        if source_chunk_rows <= 0:
            raise ValueError("source_chunk_rows must be positive")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty index directory: {path}")
        texts = source[config.field].map(normalize_text).tolist()
        if not any(texts):
            raise ValueError(f"source index field {config.field} has no non-empty values")
        kwargs: dict[str, Any] = {"analyzer": config.analyzer, "ngram_range": (config.ngram_min, config.ngram_max), "min_df": config.min_df}
        if config.max_vocabulary is not None:
            kwargs["max_features"] = config.max_vocabulary
        vectorizer = TfidfVectorizer(**kwargs)
        vectorizer.fit(texts)
        with (path / "vectorizer.pkl.tmp").open("wb") as handle:
            pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path / "vectorizer.pkl.tmp", path / "vectorizer.pkl")
        shard_specs = []
        for shard_no, start in enumerate(range(0, len(source), source_chunk_rows)):
            end = min(start + source_chunk_rows, len(source))
            matrix = vectorizer.transform(texts[start:end]).astype(np.float32).tocsr()
            spec = cls._write_shard(path, shard_no, start, matrix, source["entity_id"].astype(str).iloc[start:end].tolist())
            shard_specs.append(spec)
        metadata = {
            "index_schema": cls.schema, "config": asdict(config),
            "configuration_hash": config_hash(asdict(config)), "index_semantics_hash": config_hash(_index_semantics(config)), "candidate_source": candidate_source,
            "source_identity": dict(source_identity or {}), "source_rows": len(source),
            "vocabulary_size": len(vectorizer.vocabulary_), "source_chunk_rows": source_chunk_rows,
            "storage": "mmap-csr-transpose-npy", "shards": shard_specs,
        }
        atomic_write_json(path / "disk_metadata.json", metadata)
        return cls(path)

    @classmethod
    def build_from_tsv(
        cls, source_path: str | os.PathLike[str], directory: str | os.PathLike[str], config: IndexConfig,
        *, candidate_source: str, source_identity: Mapping[str, Any], source_chunk_rows: int = 25_000,
        resume: bool = True,
    ) -> "DiskBackedTfidfIndex":
        """Build a global-IDF sharded index with disk-spilled corpus counts.

        The first pass stores exact term/document frequencies in SQLite.  The
        second pass transforms bounded source chunks with one global
        vocabulary/IDF and writes mmap-able, query-ready sparse arrays.
        """
        if source_chunk_rows <= 0:
            raise ValueError("source_chunk_rows must be positive")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "build_state.json"
        requested = {
            "schema": "candidate-tfidf-disk-build-1", "config_hash": config_hash(_index_semantics(config)),
            "source_identity": dict(source_identity), "candidate_source": candidate_source,
            "source_chunk_rows": int(source_chunk_rows),
        }
        if (path / "disk_metadata.json").exists():
            existing = cls(path, expected_config=config)
            if existing.metadata.get("source_identity") != dict(source_identity) or existing.candidate_source != candidate_source:
                raise ValueError(f"completed disk index is incompatible: {path}")
            return existing
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for key, value in requested.items():
                if state.get(key) != value:
                    raise ValueError(f"in-progress disk index is incompatible ({key})")
            if not resume:
                raise FileExistsError(f"in-progress index exists: {path}")
        elif any(path.iterdir()):
            raise FileExistsError(f"unrecognized files in index directory: {path}")
        else:
            state = {**requested, "stage": "counting", "counted_rows": 0, "matrix_rows": 0, "shards": []}
            atomic_write_json(state_path, state)

        counts_path = path / "vocabulary_counts.sqlite"
        connection = sqlite3.connect(counts_path)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS terms (term TEXT PRIMARY KEY, tf INTEGER NOT NULL, df INTEGER NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS entities (entity_id TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE IF NOT EXISTS build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db_identity = connection.execute("SELECT value FROM build_meta WHERE key='identity'").fetchone()
            identity_value = json.dumps(requested, sort_keys=True)
            if db_identity is None:
                connection.execute("INSERT INTO build_meta VALUES ('identity', ?)", (identity_value,))
                connection.execute("INSERT INTO build_meta VALUES ('counted_rows', ?)", (str(int(state.get("counted_rows", 0))),))
                connection.commit()
            elif str(db_identity[0]) != identity_value:
                raise ValueError(f"vocabulary spill database is incompatible: {counts_path}")
            if state["stage"] == "counting":
                template = TfidfVectorizer(analyzer=config.analyzer, ngram_range=(config.ngram_min, config.ngram_max), min_df=config.min_df)
                analyzer = template.build_analyzer()
                counted = int(connection.execute("SELECT value FROM build_meta WHERE key='counted_rows'").fetchone()[0])
                state["counted_rows"] = counted
                offset = 0
                reader = pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, usecols=["entity_id", config.field], chunksize=source_chunk_rows)
                from collections import Counter
                for chunk in reader:
                    end = offset + len(chunk)
                    if end <= counted:
                        offset = end
                        continue
                    if offset < counted:
                        chunk = chunk.iloc[counted - offset:].copy()
                        offset = counted
                    # Keep feature counters bounded independently of the much
                    # larger source read chunk, and checkpoint those rows in
                    # the same transaction as their frequency updates.
                    for block_start in range(0, len(chunk), 256):
                        block = chunk.iloc[block_start:block_start + 256]
                        term_tf: Counter[str] = Counter()
                        term_df: Counter[str] = Counter()
                        for raw in block[config.field]:
                            document = Counter(analyzer(normalize_text(raw)))
                            term_tf.update(document)
                            term_df.update(document.keys())
                        try:
                            connection.execute("BEGIN")
                            connection.executemany("INSERT INTO entities(entity_id) VALUES (?)", ((str(value),) for value in block["entity_id"]))
                            connection.executemany(
                                "INSERT INTO terms(term, tf, df) VALUES (?, ?, ?) ON CONFLICT(term) DO UPDATE SET tf=tf+excluded.tf, df=df+excluded.df",
                                ((term, int(tf), int(term_df[term])) for term, tf in term_tf.items()),
                            )
                            connection.execute("UPDATE build_meta SET value=? WHERE key='counted_rows'", (str(counted + len(block)),))
                            connection.commit()
                        except sqlite3.IntegrityError as exc:
                            connection.rollback()
                            raise ValueError("source contains duplicate entity_id values") from exc
                        counted += len(block)
                        state["counted_rows"] = counted
                        atomic_write_json(state_path, state)
                    offset = end
                if counted == 0:
                    raise ValueError("source contains no rows")
                minimum = int(config.min_df)
                available = int(connection.execute("SELECT COUNT(*) FROM terms WHERE df >= ?", (minimum,)).fetchone()[0])
                if available == 0:
                    raise ValueError(f"source index field {config.field} has no terms after min_df")
                limit = available if config.max_vocabulary is None else min(available, int(config.max_vocabulary))
                # Explicit term ordering makes max-frequency ties deterministic.
                selected = connection.execute("SELECT term, df FROM terms WHERE df >= ? ORDER BY tf DESC, term ASC LIMIT ?", (minimum, limit)).fetchall()
                selected.sort(key=lambda row: str(row[0]))
                vocabulary = {str(term): index for index, (term, _) in enumerate(selected)}
                vectorizer = TfidfVectorizer(
                    analyzer=config.analyzer, ngram_range=(config.ngram_min, config.ngram_max),
                    min_df=config.min_df, vocabulary=vocabulary,
                )
                # Fit initializes sklearn's transformer; replace its IDF with
                # exact full-corpus document frequencies from the spill DB.
                vectorizer.fit([" "])
                n_docs = counted
                vectorizer.idf_ = np.asarray([np.log((1.0 + n_docs) / (1.0 + int(df))) + 1.0 for _, df in selected], dtype=np.float64)
                temporary = path / ".vectorizer.pkl.tmp"
                with temporary.open("wb") as handle:
                    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, path / "vectorizer.pkl")
                state.update({"stage": "transforming", "vocabulary_size": len(vocabulary)})
                atomic_write_json(state_path, state)
            else:
                with (path / "vectorizer.pkl").open("rb") as handle:
                    vectorizer = pickle.load(handle)

            transformed = int(state.get("matrix_rows", 0))
            offset = 0
            shard_no = len(state.get("shards", []))
            reader = pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, usecols=["entity_id", config.field], chunksize=source_chunk_rows)
            for chunk in reader:
                end = offset + len(chunk)
                if end <= transformed:
                    offset = end
                    continue
                if offset < transformed:
                    chunk = chunk.iloc[transformed - offset:].copy()
                    offset = transformed
                texts = chunk[config.field].map(normalize_text).tolist()
                matrix = vectorizer.transform(texts).astype(np.float32).tocsr()
                spec = cls._write_shard(path, shard_no, transformed, matrix, chunk["entity_id"].astype(str).tolist())
                state["shards"].append(spec)
                transformed += len(chunk)
                shard_no += 1
                offset = end
                state["matrix_rows"] = transformed
                atomic_write_json(state_path, state)
            if transformed != int(state["counted_rows"]):
                raise ValueError("source changed between vocabulary and matrix passes")
        finally:
            connection.close()

        metadata = {
            "index_schema": cls.schema, "config": asdict(config), "configuration_hash": config_hash(asdict(config)), "index_semantics_hash": config_hash(_index_semantics(config)),
            "candidate_source": candidate_source, "source_identity": dict(source_identity),
            "source_rows": int(state["counted_rows"]), "vocabulary_size": int(state["vocabulary_size"]),
            "source_chunk_rows": source_chunk_rows, "storage": "mmap-csr-transpose-npy",
            "shards": state["shards"], "feature_tie_policy": "term_frequency_desc_then_term_asc",
            "ranking_tie_policy": "score_desc_then_candidate_id_asc", "score_dtype": "float32",
        }
        atomic_write_json(path / "disk_metadata.json", metadata)
        state["stage"] = "complete"
        atomic_write_json(state_path, state)
        return cls(path)

    @staticmethod
    def _write_shard(path: Path, shard_no: int, row_start: int, matrix: sparse.csr_matrix, ids: Sequence[str]) -> dict[str, Any]:
        """Persist the multiplication-friendly transpose as mmap-able arrays."""
        transposed = matrix.T.tocsr()
        prefix = f"matrix-{shard_no:06d}"
        files = {
            "data": f"{prefix}-data.npy", "indices": f"{prefix}-indices.npy",
            "indptr": f"{prefix}-indptr.npy", "ids": f"ids-{shard_no:06d}.json",
        }
        def save_array(name: str, value: np.ndarray) -> None:
            target = path / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, target)
        save_array(files["data"], transposed.data.astype(np.float32, copy=False))
        save_array(files["indices"], transposed.indices.astype(np.int32, copy=False))
        save_array(files["indptr"], transposed.indptr.astype(np.int32, copy=False))
        atomic_write_json(path / files["ids"], list(map(str, ids)))
        return {
            **files, "row_start": int(row_start), "rows": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "resident_bytes": int(transposed.data.nbytes + transposed.indices.nbytes + transposed.indptr.nbytes),
        }

    def _open_shard(self, spec: Mapping[str, Any]) -> tuple[sparse.csr_matrix, list[str]]:
        ids = list(map(str, json.loads((self.path / str(spec["ids"])).read_text(encoding="utf-8"))))
        if "legacy_matrix" in spec:
            matrix = sparse.load_npz(self.path / str(spec["legacy_matrix"])).tocsr().T.tocsr()
        else:
            # Copy-on-write mappings satisfy native sparse kernels that require
            # writable buffers without reading/copying the whole shard.
            data = np.load(self.path / str(spec["data"]), mmap_mode="c", allow_pickle=False).view(np.ndarray)
            indices = np.load(self.path / str(spec["indices"]), mmap_mode="c", allow_pickle=False).view(np.ndarray)
            indptr = np.load(self.path / str(spec["indptr"]), mmap_mode="c", allow_pickle=False).view(np.ndarray)
            matrix = sparse.csr_matrix((data, indices, indptr), shape=(int(spec["features"]), int(spec["rows"])), copy=False)
        if len(ids) != int(spec["rows"]):
            raise ValueError(f"disk-backed index ID coverage is inconsistent in {spec['ids']}")
        return matrix, ids

    def query(self, s1: pd.DataFrame, *, candidate_source: str | None = None, top_k: int | None = None, batch_size: int | None = None) -> pd.DataFrame:
        if self.field not in s1:
            raise ValueError(f"query is missing retrieval field {self.field}")
        source_rows = int(self.metadata["source_rows"])
        k = min(int(top_k or self.config.top_k), source_rows)
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"]
        if s1.empty or k <= 0:
            return pd.DataFrame(columns=columns)
        ids = s1["entity_id"].astype(str).tolist()
        texts = s1[self.field].map(normalize_text).tolist()
        rows: list[dict[str, Any]] = []
        step = max(1, int(batch_size or self.config.batch_size))
        for start in range(0, len(s1), step):
            batch_indices = [i for i in range(start, min(start + step, len(s1))) if texts[i]]
            if not batch_indices:
                continue
            query_matrix = self.vectorizer.transform([texts[i] for i in batch_indices]).astype(np.float32).tocsr()
            accum: list[dict[str, float]] = [dict() for _ in batch_indices]
            for spec in self.shard_specs:
                query_ready, shard_ids = self._open_shard(spec)
                similarities = sp_matmul_topn(
                    query_matrix, query_ready, top_n=min(k, len(shard_ids)), threshold=0.0,
                    sort=True, n_threads=max(1, int(self.config.sparse_threads)),
                )
                for local in range(similarities.shape[0]):
                    begin, end = similarities.indptr[local], similarities.indptr[local + 1]
                    for index, score in zip(similarities.indices[begin:end], similarities.data[begin:end]):
                        entity_id = shard_ids[int(index)]
                        accum[local][entity_id] = max(accum[local].get(entity_id, 0.0), float(score))
                del similarities, query_ready, shard_ids
            for local, query_index in enumerate(batch_indices):
                chosen = sorted(accum[local].items(), key=lambda item: (-item[1], item[0]))[:k]
                rows.extend({"source1_entity_id": ids[query_index], "candidate_entity_id": entity_id,
                             "candidate_source": candidate_source or self.candidate_source, "score": score, "rank": rank}
                            for rank, (entity_id, score) in enumerate(chosen, 1))
        return pd.DataFrame(rows, columns=columns)


def validate_channel_coverage(manifest: Mapping[str, Any], query_ids: Sequence[str], channels: Sequence[str]) -> None:
    """Reject missing batches/channels, including batches whose result is empty."""
    expected = {str(value) for value in query_ids}
    covered = {str(value) for value in manifest.get("query_ids", [])}
    if covered != expected:
        raise ValueError(f"candidate coverage mismatch: missing={sorted(expected - covered)[:5]} extra={sorted(covered - expected)[:5]}")
    completed = manifest.get("completed", {})
    batch_size = int(manifest.get("query_batch_size", 0))
    expected_batches = (
        [f"{start:012d}-{min(start + batch_size, len(query_ids)):012d}"
         for start in range(0, len(query_ids), batch_size)]
        if batch_size > 0 else list(completed)
    )
    missing = [
        (batch, channel)
        for batch in expected_batches
        for channel in [*channels, "merged"]
        if completed.get(batch, {}).get(channel) is not True
    ]
    if missing:
        raise ValueError(f"incomplete candidate channel coverage: {missing[:5]}")


def _atomic_parquet(frame: pd.DataFrame, path: Path, root: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": path.relative_to(root).as_posix(), "rows": int(len(frame)), "sha256": sha256_file(path)}


def _artifact_valid(output: Path, artifact: Mapping[str, Any] | None) -> bool:
    if not artifact:
        return False
    path = output / str(artifact.get("path", ""))
    return path.is_file() and int(artifact.get("bytes", path.stat().st_size)) == path.stat().st_size and str(artifact.get("sha256", "")) == sha256_file(path)


def _release_loader(loader: Any) -> None:
    """Release a lazy channel index before the next channel is opened."""
    index = getattr(loader, "_index", None)
    if index is not None and hasattr(index, "close"):
        index.close()
    if hasattr(loader, "close"):
        loader.close()
    elif hasattr(loader, "_index"):
        loader._index = None
    gc.collect()


def run_resumable_channel_query(
    s1: pd.DataFrame,
    channel_specs: Sequence[tuple[str, Any]],
    output_dir: str | os.PathLike[str],
    *,
    query_batch_size: int = 128,
    shard_target_rows: int = 75_000,
    resume: bool = True,
    lineage: Mapping[str, Any] | None = None,
    diagnostics: Any | None = None,
) -> dict[str, Any]:
    """Run one channel at a time and checkpoint every batch/channel atomically.

    ``channel_specs`` contains ``(name, loader)`` pairs. A loader is called once
    per channel with ``loader(batch)`` and returns a candidate DataFrame.
    """
    started = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    query_ids = s1["entity_id"].astype(str).tolist()
    channels = [name for name, _ in channel_specs]
    manifest_path = output / "channel_manifest.json"
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    requested_lineage = dict(lineage or {})
    if manifest_path.exists() and resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("query_ids") != query_ids or manifest.get("channels") != channels
                or int(manifest.get("query_batch_size", 0)) != int(query_batch_size)
                or manifest.get("lineage", {}) != requested_lineage):
            raise ValueError("existing channel manifest is incompatible with requested queries/channels")
    elif manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate run: {output}")
    else:
        manifest = {"schema_version": "candidate-channel-manifest-2", "query_ids": query_ids, "channels": channels, "query_batch_size": int(query_batch_size), "lineage": requested_lineage, "completed": {}, "artifacts": {}, "completion": "in_progress"}
        atomic_write_json(manifest_path, manifest)
    final_dir = output / "final"
    batch_ranges = [(start, s1.iloc[start:start + query_batch_size].copy()) for start in range(0, len(s1), max(1, int(query_batch_size)))]
    batch_ids = [f"{start:012d}-{min(start + query_batch_size, len(s1)):012d}" for start, _ in batch_ranges]
    # Channel-major execution bounds index residency to one channel.
    for name, loader in channel_specs:
        try:
            for start, batch in batch_ranges:
                batch_id = f"{start:012d}-{min(start + query_batch_size, len(s1)):012d}"
                status = manifest["completed"].setdefault(batch_id, {})
                artifacts = manifest["artifacts"].setdefault(batch_id, {})
                if status.get(name) is True and _artifact_valid(output, artifacts.get(name)) and resume:
                    continue
                frame = loader(batch)
                path = output / "channels" / name / f"{batch_id}.parquet"
                artifact = _atomic_parquet(frame, path, output)
                artifact["bytes"] = path.stat().st_size
                artifacts[name] = artifact
                status[name] = True
                completed_units = sum(int(value is True) for batch_status in manifest["completed"].values() for key, value in batch_status.items() if key != "merged")
                manifest["progress"] = {"completed_channel_batches": completed_units, "total_channel_batches": len(batch_ranges) * len(channels), "elapsed_seconds": time.monotonic() - started}
                atomic_write_json(manifest_path, manifest)
        finally:
            _release_loader(loader)
    from .candidates import union_candidates
    for start, batch in batch_ranges:
        batch_id = f"{start:012d}-{min(start + query_batch_size, len(s1)):012d}"
        status = manifest["completed"].setdefault(batch_id, {})
        artifacts = manifest["artifacts"].setdefault(batch_id, {})
        if status.get("merged") is True and _artifact_valid(output, artifacts.get("merged")) and resume:
            continue
        missing = [name for name in channels if status.get(name) is not True or not _artifact_valid(output, artifacts.get(name))]
        if missing:
            raise ValueError(f"cannot merge batch {batch_id}; incomplete channels: {missing}")
        channel_frames = {
            name: pd.read_parquet(output / str(artifacts[name]["path"]))
            for name in channels
        }
        union = union_candidates(channel_frames)
        if not union.empty:
            order = {value: index for index, value in enumerate(batch["entity_id"].astype(str))}
            union["_query_order"] = union["source1_entity_id"].astype(str).map(order)
            union = union.sort_values(["_query_order", "source1_entity_id", "candidate_source", "candidate_entity_id"], kind="stable").drop(columns="_query_order").reset_index(drop=True)
        path = final_dir / f"candidates-{batch_id}.parquet"
        artifact = _atomic_parquet(union, path, output)
        artifact["bytes"] = path.stat().st_size
        artifacts["merged"] = artifact
        status["merged"] = True
        atomic_write_json(manifest_path, manifest)
    validate_channel_coverage(manifest, query_ids, channels)
    merged_artifacts = [manifest["artifacts"][batch]["merged"] for batch in batch_ids]
    writer_files = [str(output / str(item["path"])) for item in merged_artifacts]
    row_counts = {str(item["path"]): int(item["rows"]) for item in merged_artifacts}
    if diagnostics is not None:
        for batch, batch_frame in zip(batch_ranges, writer_files):
            frame = pd.read_parquet(batch_frame)
            groups = {str(key): value for key, value in frame.groupby("source1_entity_id", sort=False)} if not frame.empty else {}
            for source_id in batch[1]["entity_id"].astype(str):
                diagnostics.add_query(source_id, groups.get(source_id, []))
        manifest["diagnostics"] = diagnostics.report()
    manifest["completion"] = "complete"
    manifest["files"] = [str(item["path"]) for item in merged_artifacts]
    manifest["row_counts"] = row_counts
    elapsed = time.monotonic() - started
    total_pairs = sum(row_counts.values())
    peak_rss = None
    try:
        import psutil
        peak_rss = int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    manifest["stage_metrics"] = {
        "elapsed_seconds": elapsed, "query_rows": len(query_ids), "pairs": total_pairs,
        "pairs_per_second": total_pairs / elapsed if elapsed else 0.0,
        "candidate_shard_bytes": sum(int(item.get("bytes", 0)) for item in merged_artifacts),
        "process_rss_bytes_at_completion": peak_rss, "completed_batches": len(batch_ids),
        "pending_batches": 0, "estimated_remaining_seconds": 0.0,
    }
    atomic_write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "files": writer_files, "row_counts": row_counts, "query_ids": query_ids}


def stream_query_candidates(
    s1: pd.DataFrame,
    indexes: Mapping[tuple[str, str], SourceTfidfIndex],
    exact_indexes: Mapping[tuple[str, str], ExactMatchIndex] | None,
    output_dir: str | os.PathLike[str],
    *,
    query_batch_size: int = 2048,
    shard_target_rows: int = 75_000,
    strict_country: bool = False,
    preserve_missing_country: bool = True,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Query reusable indexes and write complete groups immediately.

    ``top_k`` remains per source/field channel.  It is not a total candidate
    cap; exact expansion is retained in full and reported by the writer.
    """
    exact_indexes = exact_indexes or {}
    writer = GroupedParquetWriter(output_dir, target_rows=shard_target_rows)
    all_ids = s1["entity_id"].astype(str).tolist()
    for start in range(0, len(s1), max(1, int(query_batch_size))):
        batch = s1.iloc[start : start + query_batch_size].copy()
        channels: dict[str, pd.DataFrame] = {}
        for key, index in indexes.items():
            source, field = key
            channels[f"{source}_{field}"] = index.query(batch, candidate_source=source)
        for key, index in exact_indexes.items():
            source, field = key
            channels[f"{source}_exact_{field}"] = index.query(batch, field=field)
        # Import lazily to keep this module's cache/index path lightweight.
        from .candidates import union_candidates
        union = union_candidates(channels)
        if strict_country and source_frames is not None and not union.empty:
            country_col = "country" if "country" in batch else None
            if country_col:
                left = batch[["entity_id", country_col]].rename(columns={"entity_id": "source1_entity_id", country_col: "_country_a"})
                candidate_parts = []
                for source, frame in source_frames.items():
                    if "country" in frame:
                        candidate_parts.append(frame[["entity_id", "country"]].assign(candidate_source=source).rename(columns={"entity_id": "candidate_entity_id", "country": "_country_b"}))
                if candidate_parts:
                    countries = pd.concat(candidate_parts, ignore_index=True)
                    union = union.merge(left, on="source1_entity_id", how="left").merge(countries, on=["candidate_source", "candidate_entity_id"], how="left")
                    left_country = union["_country_a"].fillna("").astype(str).str.strip()
                    right_country = union["_country_b"].fillna("").astype(str).str.strip()
                    keep = left_country.eq(right_country)
                    if preserve_missing_country:
                        keep |= left_country.eq("") | right_country.eq("")
                    union = union[keep].drop(columns=["_country_a", "_country_b"])
        groups = {str(key): value for key, value in union.groupby("source1_entity_id", sort=False)}
        for source1_id in batch["entity_id"].astype(str):
            group = groups.get(source1_id)
            if group is not None and not group.empty:
                writer.write_group(group)
    writer.close()
    return {"files": writer.files, "row_counts": writer.row_counts, "large_groups": writer.large_groups, "queries": len(all_ids), "query_ids": all_ids, "pairs": sum(writer.row_counts.values()), "strict_country": strict_country, "preserve_missing_country": preserve_missing_country}


@dataclass
class IndexConfig:
    field: str
    top_k: int = 50
    analyzer: str = "char"
    ngram_min: int = 2
    ngram_max: int = 4
    min_df: int = 1
    max_vocabulary: int | None = None
    memory_budget_mb: int | None = None
    batch_size: int = 2048
    sparse_threads: int = 1


class SourceTfidfIndex:
    """A fitted source index reusable across arbitrary S1 query batches."""

    def __init__(self, source: pd.DataFrame, config: IndexConfig, *, source_identity: Mapping[str, Any] | None = None):
        if config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if config.max_vocabulary is not None and config.max_vocabulary <= 0:
            raise ValueError("max_vocabulary must be positive")
        if config.field not in source.columns:
            raise ValueError(f"source is missing retrieval field {config.field}")
        self.config = config
        self.source_identity = dict(source_identity or {})
        self.source_ids = source["entity_id"].astype(str).tolist()
        texts = source[config.field].map(normalize_text).tolist()
        if not any(texts):
            raise ValueError(f"source index field {config.field} has no non-empty values")
        kwargs: dict[str, Any] = {"analyzer": config.analyzer, "ngram_range": (config.ngram_min, config.ngram_max), "min_df": config.min_df}
        if config.max_vocabulary is not None:
            kwargs["max_features"] = config.max_vocabulary
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.vectorizer.fit(texts)
        self.matrix = self.vectorizer.transform(texts).astype(np.float32).tocsr()
        # Convert once per loaded channel, never once per query batch.
        self.query_matrix = self.matrix.T.tocsr()
        self.metadata = {"index_schema": "candidate-index-1", "config": asdict(config), "configuration_hash": config_hash(asdict(config)), "index_semantics_hash": config_hash(_index_semantics(config)), "source_identity": self.source_identity, "source_rows": len(source), "vocabulary_size": len(self.vectorizer.vocabulary_), "estimated_matrix_mb": float((self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes) / 1024**2)}
        if config.memory_budget_mb is not None and self.metadata["estimated_matrix_mb"] > config.memory_budget_mb:
            raise MemoryError(f"source TF-IDF index estimate {self.metadata['estimated_matrix_mb']:.1f} MB exceeds declared budget {config.memory_budget_mb} MB; reduce vocabulary or build on a larger machine")

    def query(self, s1: pd.DataFrame, *, candidate_source: str, top_k: int | None = None, batch_size: int | None = None) -> pd.DataFrame:
        k = min(int(top_k or self.config.top_k), len(self.source_ids))
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"]
        if s1.empty or k <= 0:
            return pd.DataFrame(columns=columns)
        field = self.config.field
        if field not in s1:
            raise ValueError(f"query is missing retrieval field {field}")
        ids = s1["entity_id"].astype(str).tolist()
        texts = s1[field].map(normalize_text).tolist()
        query_indices = [idx for idx, value in enumerate(texts) if value]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(query_indices), int(batch_size or self.config.batch_size)):
            batch_indices = query_indices[start : start + int(batch_size or self.config.batch_size)]
            matrix = self.vectorizer.transform([texts[idx] for idx in batch_indices]).astype(np.float32).tocsr()
            similarities = sp_matmul_topn(matrix, self.query_matrix, top_n=k, threshold=0.0, sort=True, n_threads=max(1, int(self.config.sparse_threads)))
            for local, query_index in enumerate(batch_indices):
                begin, end = similarities.indptr[local], similarities.indptr[local + 1]
                for rank, (source_index, score) in enumerate(zip(similarities.indices[begin:end], similarities.data[begin:end]), start=1):
                    rows.append({"source1_entity_id": ids[query_index], "candidate_entity_id": self.source_ids[int(source_index)], "candidate_source": candidate_source, "score": float(score), "rank": rank})
        return pd.DataFrame(rows, columns=columns)

    def save(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        target = path / "index.pkl"
        if target.exists() or (path / "metadata.json").exists():
            raise FileExistsError(f"refusing to overwrite existing candidate index: {path}")
        temporary = path / ".index.pkl.tmp"
        with temporary.open("wb") as handle:
            pickle.dump({"config": asdict(self.config), "source_identity": self.source_identity, "source_ids": self.source_ids, "vectorizer": self.vectorizer, "matrix": self.matrix}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        atomic_write_json(path / "metadata.json", self.metadata)

    @classmethod
    def load(cls, directory: str | os.PathLike[str], *, expected_config: IndexConfig | None = None, expected_source_identity: Mapping[str, Any] | None = None) -> "SourceTfidfIndex":
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        expected_hash = (config_hash(_index_semantics(expected_config)) if metadata.get("index_semantics_hash") else config_hash(asdict(expected_config))) if expected_config is not None else None
        actual_hash = metadata.get("index_semantics_hash", metadata.get("configuration_hash"))
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError("cached source index configuration changed; rebuild required")
        if expected_source_identity is not None and dict(expected_source_identity) != metadata.get("source_identity"):
            raise ValueError("cached source index input identity changed; rebuild required")
        with (path / "index.pkl").open("rb") as handle:
            value = pickle.load(handle)
        obj = cls.__new__(cls)
        obj.config = expected_config or IndexConfig(**value["config"]); obj.source_identity = value["source_identity"]; obj.source_ids = value["source_ids"]; obj.vectorizer = value["vectorizer"]; obj.matrix = value["matrix"]; obj.query_matrix = obj.matrix.T.tocsr(); obj.metadata = metadata
        return obj
