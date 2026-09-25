"""Explicit candidate index/query separation for future scalable retrieval."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    from .candidates import sp_matmul_topn

from .neural_contracts import atomic_write_json, config_hash, file_identity
from .text_utils import normalize_text
from .candidate_io import GroupedParquetWriter


class ExactMatchIndex:
    """Bounded-by-query exact expansion index; empty keys are never indexed."""

    def __init__(self, source: pd.DataFrame, field: str, *, candidate_source: str):
        if field not in source.columns:
            raise ValueError(f"source is missing exact field {field}")
        self.field = field
        self.candidate_source = candidate_source
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
                rows.append({"source1_entity_id": source1_id, "candidate_entity_id": candidate_id, "candidate_source": self.candidate_source, "score": 1.0, "rank": rank, "retrieval_provenance": "exact"})
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank", "retrieval_provenance"])

    def save(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "exact_index.pkl"
        temporary = path / ".exact_index.pkl.tmp"
        with temporary.open("wb") as handle:
            pickle.dump({"field": self.field, "candidate_source": self.candidate_source, "mapping": self.mapping, "frequent_keys": self.frequent_keys}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        atomic_write_json(path / "exact_metadata.json", {
            "index_schema": "candidate-exact-index-1",
            "field": self.field,
            "candidate_source": self.candidate_source,
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
        obj.mapping = value["mapping"]
        obj.frequent_keys = value["frequent_keys"]
        return obj


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
        for source1_id in batch["entity_id"].astype(str):
            group = union[union["source1_entity_id"].astype(str) == source1_id]
            writer.write_group(group) if not group.empty else None
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
        self.metadata = {"index_schema": "candidate-index-1", "config": asdict(config), "configuration_hash": config_hash(asdict(config)), "source_identity": self.source_identity, "source_rows": len(source), "vocabulary_size": len(self.vectorizer.vocabulary_), "estimated_matrix_mb": float((self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes) / 1024**2)}
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
            similarities = sp_matmul_topn(matrix, self.matrix.T.tocsr(), top_n=k, threshold=0.0, sort=True, n_threads=1)
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
        if expected_config is not None and config_hash(asdict(expected_config)) != metadata.get("configuration_hash"):
            raise ValueError("cached source index configuration changed; rebuild required")
        if expected_source_identity is not None and dict(expected_source_identity) != metadata.get("source_identity"):
            raise ValueError("cached source index input identity changed; rebuild required")
        with (path / "index.pkl").open("rb") as handle:
            value = pickle.load(handle)
        obj = cls.__new__(cls)
        obj.config = IndexConfig(**value["config"]); obj.source_identity = value["source_identity"]; obj.source_ids = value["source_ids"]; obj.vectorizer = value["vectorizer"]; obj.matrix = value["matrix"]; obj.metadata = metadata
        return obj
