"""One-resident-channel sparse search over a historical disk-backed index."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

from .candidate_index import DiskBackedTfidfIndex
from .text_utils import normalize_text


class FusedForwardIndex:
    """Combine the source shards once, then reuse the matrix for query batches.

    Scores use the original fitted vectorizer and source vectors. At tied
    top-k boundaries, a fused search can select different IDs than the
    historical shard-by-shard search; the rank budget needs validation.
    """

    def __init__(self, location: str | Path, *, max_resident_gb: float = 12):
        self.disk = DiskBackedTfidfIndex(location)
        estimate = sum(int(s.get("resident_bytes", 0)) for s in self.disk.shard_specs)
        if estimate > max_resident_gb * (1024 ** 3):
            raise MemoryError("fused source matrix exceeds configured resident budget")
        matrices = []
        self.source_ids = []
        for spec in self.disk.shard_specs:
            matrix, ids = self.disk._open_shard(spec)
            matrices.append(matrix)
            self.source_ids.extend(ids)
        self.matrix = sparse.hstack(matrices, format="csr")
        if self.matrix.shape[1] != len(self.source_ids):
            raise ValueError("fused source matrix and ID map differ")
        del matrices
        gc.collect()

    def close(self) -> None:
        self.matrix = None
        self.source_ids = []
        self.disk = None
        gc.collect()

    def query(self, s1: pd.DataFrame, *, candidate_source: str | None = None,
              top_k: int = 128, batch_size: int = 128) -> pd.DataFrame:
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"]
        if top_k <= 0 or batch_size <= 0:
            raise ValueError("top_k and batch_size must be positive")
        field = self.disk.field
        if field not in s1 or "entity_id" not in s1:
            raise ValueError(f"query is missing entity_id or {field}")
        if s1.empty:
            return pd.DataFrame(columns=columns)
        names = s1["entity_id"].astype(str).tolist()
        values = s1[field].map(normalize_text).tolist()
        source = candidate_source or self.disk.candidate_source
        rows = []
        for start in range(0, len(s1), batch_size):
            selected = [i for i in range(start, min(start + batch_size, len(s1))) if values[i]]
            if not selected:
                continue
            query_matrix = self.disk.vectorizer.transform([values[i] for i in selected]).astype(np.float32).tocsr()
            results = sp_matmul_topn(query_matrix, self.matrix,
                                     top_n=min(top_k, len(self.source_ids)),
                                     threshold=0.0, sort=True,
                                     n_threads=max(1, int(self.disk.config.sparse_threads)))
            for local, query_index in enumerate(selected):
                begin, end = results.indptr[local:local + 2]
                for rank, (column, value) in enumerate(zip(results.indices[begin:end], results.data[begin:end]), 1):
                    rows.append({"source1_entity_id": names[query_index],
                                 "candidate_entity_id": self.source_ids[int(column)],
                                 "candidate_source": source, "score": float(value), "rank": rank})
        return pd.DataFrame(rows, columns=columns)


class GpuShardedForwardIndex:
    """Run the historical per-source-shard top-k on CUDA, then merge exact IDs."""

    def __init__(self, location: str | Path, *, max_resident_gb: float = 12):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GpuShardedForwardIndex")
        self.torch = torch
        self.disk = DiskBackedTfidfIndex(location)
        estimated = sum(int(s.get("resident_bytes", 0)) for s in self.disk.shard_specs)
        if estimated > max_resident_gb * (1024 ** 3):
            raise MemoryError("GPU source matrix exceeds configured resident budget")
        self.shards = []
        self.source_ids = []
        offset = 0
        for spec in self.disk.shard_specs:
            matrix, ids = self.disk._open_shard(spec)
            tensor = torch.sparse_csr_tensor(
                torch.as_tensor(np.asarray(matrix.indptr, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(matrix.indices, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(matrix.data, dtype=np.float32), device="cuda"),
                size=matrix.shape, device="cuda",
            )
            shard_ids = list(map(str, ids))
            self.shards.append((tensor, shard_ids, offset))
            self.source_ids.extend(shard_ids)
            offset += len(ids)
        self.source_count = offset
        torch.cuda.synchronize()

    def close(self) -> None:
        self.shards = []
        self.disk = None
        self.torch.cuda.empty_cache()
        gc.collect()

    def query(self, s1: pd.DataFrame, *, candidate_source: str | None = None,
              top_k: int = 100, batch_size: int = 128) -> pd.DataFrame:
        torch = self.torch
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"]
        if top_k <= 0 or batch_size <= 0:
            raise ValueError("top_k and batch_size must be positive")
        field = self.disk.field
        if field not in s1 or "entity_id" not in s1:
            raise ValueError(f"query is missing entity_id or {field}")
        ids = s1["entity_id"].astype(str).tolist()
        values = s1[field].map(normalize_text).tolist()
        rows = []
        for start in range(0, len(s1), batch_size):
            selected = [i for i in range(start, min(start + batch_size, len(s1))) if values[i]]
            if not selected:
                continue
            query = self.disk.vectorizer.transform([values[i] for i in selected]).astype(np.float32).tocsr()
            qgpu = torch.sparse_csr_tensor(
                torch.as_tensor(np.asarray(query.indptr, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(query.indices, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(query.data, dtype=np.float32), device="cuda"),
                size=query.shape, device="cuda",
            )
            all_indices, all_scores = [], []
            for source, _, offset in self.shards:
                product = torch.sparse.mm(qgpu, source)
                dense = product.to_dense()
                scores, indexes = torch.topk(dense, k=min(top_k, dense.shape[1]), dim=1)
                all_indices.append(indexes + offset)
                all_scores.append(scores)
                del product, dense, scores, indexes
            torch.cuda.synchronize()
            indexes = torch.cat(all_indices, dim=1).cpu().numpy()
            scores = torch.cat(all_scores, dim=1).cpu().numpy()
            for row, query_index in enumerate(selected):
                found = {}
                for col, value in zip(indexes[row], scores[row]):
                    score = float(value)
                    if score > 0:
                        source_id = self.source_ids[int(col)]
                        if score > found.get(source_id, -1.0):
                            found[source_id] = score
                chosen = sorted(found.items(), key=lambda item: (-item[1], item[0]))[:top_k]
                rows.extend({"source1_entity_id": ids[query_index], "candidate_entity_id": source_id,
                             "candidate_source": candidate_source or self.disk.candidate_source,
                             "score": score, "rank": rank}
                            for rank, (source_id, score) in enumerate(chosen, 1))
        return pd.DataFrame(rows, columns=columns)


class GpuGroupedForwardIndex:
    """Use GPU top-k over bounded groups of source shards, then merge groups.

    ``min_idf`` optionally excludes common character n-grams for candidate
    generation. Scores then describe the filtered vectors and should be treated
    as retrieval ranks, not as the historical full-vector cosine feature.
    """

    def __init__(self, location: str | Path, *, group_shards: int = 32,
                 local_k: int = 256, max_resident_gb: float = 12,
                 min_idf: float | None = None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GpuGroupedForwardIndex")
        if group_shards <= 0 or local_k <= 0:
            raise ValueError("group_shards and local_k must be positive")
        self.torch = torch
        self.disk = DiskBackedTfidfIndex(location)
        self.feature_mask = (self.disk.vectorizer.idf_ >= min_idf) if min_idf is not None else None
        estimate = sum(int(s.get("resident_bytes", 0)) for s in self.disk.shard_specs)
        if estimate > max_resident_gb * (1024 ** 3):
            raise MemoryError("GPU source matrix exceeds configured resident budget")
        self.local_k = local_k
        self.groups = []
        self.source_ids = []
        offset = 0
        for start in range(0, len(self.disk.shard_specs), group_shards):
            matrices, group_ids = [], []
            for spec in self.disk.shard_specs[start:start + group_shards]:
                matrix, ids = self.disk._open_shard(spec)
                if self.feature_mask is not None:
                    matrix = matrix[self.feature_mask, :].tocsr()
                matrices.append(matrix)
                group_ids.extend(map(str, ids))
            fused = sparse.hstack(matrices, format="csr")
            tensor = torch.sparse_csr_tensor(
                torch.as_tensor(np.asarray(fused.indptr, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(fused.indices, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(fused.data, dtype=np.float32), device="cuda"),
                size=fused.shape, device="cuda",
            )
            self.groups.append((tensor, offset))
            self.source_ids.extend(group_ids)
            offset += len(group_ids)
            del matrices, fused, group_ids
            gc.collect()
        torch.cuda.synchronize()

    def close(self) -> None:
        self.groups = []
        self.source_ids = []
        self.disk = None
        self.torch.cuda.empty_cache()
        gc.collect()

    def query(self, s1: pd.DataFrame, *, candidate_source: str | None = None,
              top_k: int = 128, batch_size: int = 128) -> pd.DataFrame:
        torch = self.torch
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"]
        if top_k <= 0 or batch_size <= 0:
            raise ValueError("top_k and batch_size must be positive")
        field = self.disk.field
        if field not in s1 or "entity_id" not in s1:
            raise ValueError(f"query is missing entity_id or {field}")
        ids = s1["entity_id"].astype(str).tolist()
        values = s1[field].map(normalize_text).tolist()
        source = candidate_source or self.disk.candidate_source
        rows = []
        for start in range(0, len(s1), batch_size):
            selected = [i for i in range(start, min(start + batch_size, len(s1))) if values[i]]
            if not selected:
                continue
            query = self.disk.vectorizer.transform([values[i] for i in selected]).astype(np.float32).tocsr()
            if self.feature_mask is not None:
                query = query[:, self.feature_mask].tocsr()
            qgpu = torch.sparse_csr_tensor(
                torch.as_tensor(np.asarray(query.indptr, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(query.indices, dtype=np.int32), device="cuda"),
                torch.as_tensor(np.asarray(query.data, dtype=np.float32), device="cuda"),
                size=query.shape, device="cuda",
            )
            group_indexes, group_scores = [], []
            for matrix, offset in self.groups:
                product = torch.sparse.mm(qgpu, matrix)
                dense = product.to_dense()
                scores, indexes = torch.topk(dense, k=min(self.local_k, dense.shape[1]), dim=1)
                group_indexes.append(indexes + offset)
                group_scores.append(scores)
                del product, dense, scores, indexes
            torch.cuda.synchronize()
            candidate_indexes = torch.cat(group_indexes, dim=1)
            candidate_scores = torch.cat(group_scores, dim=1)
            buffer_k = min(max(top_k, 256), candidate_scores.shape[1])
            scores, positions = torch.topk(candidate_scores, k=buffer_k, dim=1)
            indexes = torch.gather(candidate_indexes, 1, positions).cpu().numpy()
            scores = scores.cpu().numpy()
            for row, query_index in enumerate(selected):
                keep = scores[row] > 0
                candidates = indexes[row][keep]
                candidate_scores = scores[row][keep]
                names = [self.source_ids[int(column)] for column in candidates]
                order = np.lexsort((np.asarray(names, dtype=object), -candidate_scores))[:top_k]
                rows.extend({"source1_entity_id": ids[query_index], "candidate_entity_id": names[i],
                             "candidate_source": source, "score": float(candidate_scores[i]),
                             "rank": rank}
                            for rank, i in enumerate(order, 1))
        return pd.DataFrame(rows, columns=columns)
