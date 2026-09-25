"""Scalable TF-IDF candidate generation and diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # Keep tiny CPU/offline checks usable without optional native wheels.
    def sp_matmul_topn(A, B, top_n, threshold=0.0, sort=True, n_threads=1):
        product = (A @ B).tocsr()
        rows = []
        for row in range(product.shape[0]):
            start, end = product.indptr[row], product.indptr[row + 1]
            indices = product.indices[start:end]
            values = product.data[start:end]
            keep = np.flatnonzero(values > threshold)
            order = keep[np.argsort(values[keep])[::-1][:top_n]] if len(keep) else keep
            rows.append((indices[order], values[order]))
        from scipy.sparse import csr_matrix
        data = np.concatenate([value for _, value in rows]) if rows else np.array([], dtype=float)
        indices = np.concatenate([index for index, _ in rows]) if rows else np.array([], dtype=int)
        indptr = np.zeros(product.shape[0] + 1, dtype=int)
        for row, (index, _) in enumerate(rows):
            indptr[row + 1] = indptr[row] + len(index)
        return csr_matrix((data, indices, indptr), shape=product.shape)

from .evaluate import evaluate_predictions
from .text_utils import normalize_text


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_source(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows).copy()
    if "entity_id" not in frame.columns:
        raise ValueError(f"{path} must contain entity_id")
    frame["entity_id"] = frame["entity_id"].astype(str).str.strip()
    if frame["entity_id"].eq("").any() or frame["entity_id"].duplicated().any():
        raise ValueError(f"{path} contains empty or duplicate entity_id values")
    return frame


def _text_column(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[field].map(normalize_text)


def _vectorizer(texts: Iterable[str]) -> TfidfVectorizer | None:
    corpus = list(texts)
    if not any(corpus):
        return None
    for kwargs in (
        {"analyzer": "char", "ngram_range": (2, 4), "min_df": 2},
        {"analyzer": "char", "ngram_range": (1, 4), "min_df": 1},
        {"analyzer": "word", "ngram_range": (1, 2), "min_df": 1},
    ):
        vectorizer = TfidfVectorizer(**kwargs)
        try:
            vectorizer.fit(corpus)
            if vectorizer.vocabulary_:
                return vectorizer
        except ValueError:
            continue
    return None


def retrieve_channel(
    s1: pd.DataFrame,
    source: pd.DataFrame,
    field: str,
    top_k: int,
    batch_size: int = 2048,
    candidate_source: str = "S2",
) -> pd.DataFrame:
    """Retrieve one channel with sparse exact top-K cosine similarity.

    sklearn.neighbors.NearestNeighbors falls back to brute-force search on
    sparse TF-IDF input. Sparse top-N multiplication instead follows only
    overlapping TF-IDF features and keeps at most top_k matches per query.
    """
    columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"]
    if top_k <= 0 or source.empty or s1.empty:
        return pd.DataFrame(columns=columns)

    source_text = _text_column(source, field)
    query_text = _text_column(s1, field)

    # Query-only features can never contribute to a source match, so fitting
    # only on the indexed source keeps the vocabulary and IDF work smaller.
    vectorizer = _vectorizer(source_text.tolist())
    if vectorizer is None:
        return pd.DataFrame(columns=columns)

    source_matrix = vectorizer.transform(source_text.tolist()).astype(np.float32).tocsr()
    source_matrix_t = source_matrix.T.tocsr()
    query_indices = [i for i, value in enumerate(query_text.tolist()) if value]
    if not query_indices:
        return pd.DataFrame(columns=columns)

    n_neighbors = min(int(top_k), len(source))
    batch_size = max(1, int(batch_size))
    n_threads = max(1, min(8, os.cpu_count() or 1))
    source_ids = source["entity_id"].astype(str).tolist()
    s1_ids = s1["entity_id"].astype(str).tolist()
    rows: list[dict[str, Any]] = []

    for start in range(0, len(query_indices), batch_size):
        batch_indices = query_indices[start : start + batch_size]
        query_matrix = (
            vectorizer.transform(query_text.iloc[batch_indices].tolist())
            .astype(np.float32)
            .tocsr()
        )

        # TfidfVectorizer L2-normalizes rows by default, so sparse dot product
        # equals cosine similarity. The sparse kernel retains only top-K
        # nonzero results instead of scanning every source row for every query.
        similarities = sp_matmul_topn(
            query_matrix,
            source_matrix_t,
            top_n=n_neighbors,
            threshold=0.0,
            sort=True,
            n_threads=n_threads,
        )

        for local, s1_index in enumerate(batch_indices):
            row_start = similarities.indptr[local]
            row_end = similarities.indptr[local + 1]
            candidate_indices = similarities.indices[row_start:row_end]
            candidate_scores = similarities.data[row_start:row_end]

            for rank, (source_index, score) in enumerate(
                zip(candidate_indices, candidate_scores), start=1
            ):
                rows.append({
                    "source1_entity_id": s1_ids[s1_index],
                    "candidate_entity_id": source_ids[int(source_index)],
                    "candidate_source": candidate_source,
                    "score": float(score),
                    "rank": rank,
                })

    return pd.DataFrame(rows, columns=columns)

def retrieve_candidates(s1_df: pd.DataFrame, source_df: pd.DataFrame, column: str, n_candidates: int = 50, batch_size: int = 2048) -> pd.DataFrame:
    """Compatibility wrapper for the original prototype API."""
    field = "__retrieval_text__"
    left = s1_df.copy(); right = source_df.copy()
    original = column.removeprefix("norm_")
    left[field] = left[column] if column in left.columns else _text_column(left, original)
    right[field] = right[column] if column in right.columns else _text_column(right, original)
    result = retrieve_channel(left, right, field, n_candidates, batch_size, "S2")
    return result[["source1_entity_id", "candidate_entity_id", "score"]]


def union_candidates(channels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Union channels while retaining scores, flags, and ranks."""
    parts: list[pd.DataFrame] = []
    for channel_name, frame in channels.items():
        if frame.empty:
            continue
        item = frame.copy()
        item["retrieved_by_name"] = int("name" in channel_name)
        item["retrieved_by_address"] = int("address" in channel_name)
        item["retrieval_provenance"] = item.get("retrieval_provenance", "tfidf")
        is_exact = item["retrieval_provenance"].astype(str).eq("exact") | ("exact" in channel_name)
        item["retrieved_by_exact"] = is_exact.astype(int)
        item["frequent_key_expansion"] = pd.to_numeric(item.get("frequent_key_expansion", 0), errors="coerce").fillna(0).astype(int) if "frequent_key_expansion" in item else 0
        item["name_tfidf_score"] = item["score"].where(("name" in channel_name) & ~is_exact, 0.0)
        item["address_tfidf_score"] = item["score"].where(("address" in channel_name) & ~is_exact, 0.0)
        item["name_rank"] = item["rank"].where(("name" in channel_name) & ~is_exact, 0)
        item["address_rank"] = item["rank"].where(("address" in channel_name) & ~is_exact, 0)
        parts.append(item)
    output_columns = [
        "source1_entity_id", "candidate_entity_id", "candidate_source",
        "name_tfidf_score", "address_tfidf_score", "name_rank", "address_rank",
        "retrieved_by_name", "retrieved_by_address", "retrieval_score",
        "retrieved_by_exact", "retrieval_provenance", "frequent_key_expansion",
    ]
    if not parts:
        return pd.DataFrame(columns=output_columns)
    combined = pd.concat(parts, ignore_index=True)
    keys = ["source1_entity_id", "candidate_source", "candidate_entity_id"]
    grouped = combined.groupby(keys, sort=False, dropna=False)
    result = grouped.agg(
        name_tfidf_score=("name_tfidf_score", "max"),
        address_tfidf_score=("address_tfidf_score", "max"),
        retrieved_by_name=("retrieved_by_name", "max"),
        retrieved_by_address=("retrieved_by_address", "max"),
        retrieved_by_exact=("retrieved_by_exact", "max"),
        frequent_key_expansion=("frequent_key_expansion", "max"),
        retrieval_provenance=("retrieval_provenance", lambda values: ",".join(sorted(set(map(str, values))))),
    ).reset_index()
    for column in ("name_tfidf_score", "address_tfidf_score"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    for column in ("name_rank", "address_rank"):
        # Zero is the missing-rank sentinel; min must ignore it.
        ranked = combined[column].where(pd.to_numeric(combined[column], errors="coerce").fillna(0) > 0)
        result[column] = ranked.groupby([combined[key] for key in keys], sort=False).min().fillna(0).astype(int).to_numpy()
    result["retrieval_score"] = result[["name_tfidf_score", "address_tfidf_score"]].max(axis=1)
    result.loc[result["retrieved_by_exact"].astype(bool), "retrieval_score"] = result.loc[result["retrieved_by_exact"].astype(bool), "retrieval_score"].clip(lower=1.0)
    return result[output_columns]


def retrieve_exact(s1: pd.DataFrame, source: pd.DataFrame, field: str, label: str) -> pd.DataFrame:
    norm_field = f"norm_{field}"
    
    s1_copy = s1.copy()
    source_copy = source.copy()
    
    if norm_field not in s1_copy.columns:
        s1_copy[norm_field] = s1_copy[field].apply(normalize_text)
    if norm_field not in source_copy.columns:
        source_copy[norm_field] = source_copy[field].apply(normalize_text)
        
    s1_valid = s1_copy[s1_copy[norm_field].astype(str).str.strip() != ""]
    source_valid = source_copy[source_copy[norm_field].astype(str).str.strip() != ""]
    
    merged = pd.merge(s1_valid[["entity_id", norm_field]], source_valid[["entity_id", norm_field]], on=norm_field, suffixes=("_s1", "_s2"))
    
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "source1_entity_id": str(row["entity_id_s1"]),
            "candidate_entity_id": str(row["entity_id_s2"]),
            "candidate_source": label,
            "score": 1.0,
            "rank": 1,
        })
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "score", "rank"])


def generate_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, top_k_name: int = 50, top_k_address: int = 50, batch_size: int = 2048) -> pd.DataFrame:
    channels: dict[str, pd.DataFrame] = {}
    for label, source in (("S2", s2), ("S3", s3)):
        channels[f"{label}_exact_name"] = retrieve_exact(s1, source, "business_name", label)
        channels[f"{label}_exact_address"] = retrieve_exact(s1, source, "business_address", label)
        channels[f"{label}_name"] = retrieve_channel(s1, source, "business_name", top_k_name, batch_size, label)
        channels[f"{label}_address"] = retrieve_channel(s1, source, "business_address", top_k_address, batch_size, label)
    return union_candidates(channels)


def _parse_matches(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return list(dict.fromkeys(token.strip() for token in str(value).split(",") if token.strip()))


def load_ground_truth(path: str | Path) -> dict[str, list[str]]:
    frame = read_tsv(path)
    return {str(row.source1_entity_id): _parse_matches(row.matched_entity_ids) for row in frame.itertuples(index=False)}


def evaluate_recall(candidates_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> tuple[float, int]:
    """Compatibility helper returning (recall, number of true pairs)."""
    ground_truth = {
        str(row.source1_entity_id): _parse_matches(row.matched_entity_ids)
        for row in ground_truth_df.itertuples(index=False)
    }
    candidate_sets = candidates_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict() if not candidates_df.empty else {}
    total = sum(len(values) for values in ground_truth.values())
    found = sum(candidate in candidate_sets.get(source1, set()) for source1, values in ground_truth.items() for candidate in values)
    return (found / total if total else 0.0), total


def candidate_diagnostics(candidates: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, ground_truth: Mapping[str, Iterable[str]] | None = None, runtime_seconds: float | None = None) -> dict[str, Any]:
    counts = candidates.groupby("source1_entity_id").size() if not candidates.empty else pd.Series(dtype=int)
    all_s1 = s1["entity_id"].astype(str).tolist()
    count_values = np.array([int(counts.get(value, 0)) for value in all_s1], dtype=float)
    report: dict[str, Any] = {
        "num_s1": len(s1), "num_s2": len(s2), "num_s3": len(s3),
        "total_candidate_pairs": int(len(candidates)),
        "average_candidates_per_s1": float(count_values.mean()) if len(count_values) else 0.0,
        "median_candidates_per_s1": float(np.median(count_values)) if len(count_values) else 0.0,
        "p95_candidates_per_s1": float(np.percentile(count_values, 95)) if len(count_values) else 0.0,
        "max_candidates_per_s1": int(count_values.max()) if len(count_values) else 0,
        "reduction_ratio": float(1.0 - len(candidates) / max(1, len(s1) * (len(s2) + len(s3)))),
    }
    if runtime_seconds is not None:
        report["runtime_seconds"] = float(runtime_seconds)
    if ground_truth is None:
        report["candidate_recall_overall"] = None
        return report
    candidate_sets = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict() if not candidates.empty else {}
    gt_pairs = [(s1_id, str(candidate)) for s1_id, ids in ground_truth.items() for candidate in ids]
    hits = [(s1_id, candidate) for s1_id, candidate in gt_pairs if candidate in candidate_sets.get(s1_id, set())]
    report["candidate_recall_overall"] = len(hits) / len(gt_pairs) if gt_pairs else 1.0
    source_sets = {str(row.entity_id): "S2" for row in s2.itertuples(index=False)}
    source_sets.update({str(row.entity_id): "S3" for row in s3.itertuples(index=False)})
    for source in ("S2", "S3"):
        total = sum(source_sets.get(candidate) == source for _, candidate in gt_pairs)
        found = sum(source_sets.get(candidate) == source for _, candidate in hits)
        report[f"candidate_recall_{source.lower()}"] = found / total if total else 1.0
    countries = s1.set_index("entity_id").get("country", pd.Series(dtype=str)).to_dict()
    by_country: dict[str, Any] = {}
    for country in sorted({str(value) for value in countries.values()}):
        pairs = [(sid, cid) for sid, cid in gt_pairs if str(countries.get(sid, "")) == country]
        found = [(sid, cid) for sid, cid in pairs if cid in candidate_sets.get(sid, set())]
        by_country[country] = {"retrieved": len(found), "total": len(pairs), "recall": len(found) / len(pairs) if pairs else 1.0}
    report["candidate_recall_by_country"] = by_country
    by_match_count: dict[str, Any] = {}
    for label, predicate in {
        "singleton": lambda count: count == 0,
        "one": lambda count: count == 1,
        "two_or_three": lambda count: 2 <= count <= 3,
        "four_plus": lambda count: count >= 4,
    }.items():
        subset_ids = {sid for sid, values in ground_truth.items() if predicate(len(list(values)))}
        pairs = [(sid, cid) for sid, cid in gt_pairs if sid in subset_ids]
        found = [(sid, cid) for sid, cid in pairs if cid in candidate_sets.get(sid, set())]
        oracle_subset = {sid: [cid for cid in ground_truth[sid] if cid in candidate_sets.get(sid, set())] for sid in subset_ids}
        oracle_score = evaluate_predictions({sid: ground_truth[sid] for sid in subset_ids}, oracle_subset)["macro_f05"] if subset_ids else 1.0
        by_match_count[label] = {"queries": len(subset_ids), "retrieved": len(found), "total": len(pairs), "recall": len(found) / len(pairs) if pairs else 1.0, "oracle_macro_f05": float(oracle_score)}
    report["candidate_recall_by_match_count"] = by_match_count
    oracle = {sid: [cid for cid in ids if cid in candidate_sets.get(sid, set())] for sid, ids in ground_truth.items()}
    oracle_metrics = evaluate_predictions(dict(ground_truth), oracle)
    report["oracle_macro_f05"] = float(oracle_metrics["macro_f05"])
    report["oracle_metrics"] = {key: value for key, value in oracle_metrics.items() if key != "per_entity_scores"}
    return report


def prune_candidates(candidates: pd.DataFrame, total_budget: int | None, *, exact_overflow: str = "allow") -> pd.DataFrame:
    """Cheap deterministic final pruning; disabled when ``total_budget`` is None.

    Exact matches are never silently discarded. If they alone exceed the
    budget, ``exact_overflow='allow'`` emits the full exact set and reports an
    over-budget group through downstream diagnostics.
    """
    if total_budget is None:
        return candidates.copy()
    if total_budget <= 0:
        raise ValueError("total_budget must be positive or null")
    if exact_overflow not in {"allow", "error"}:
        raise ValueError("exact_overflow must be 'allow' or 'error'")
    parts = []
    for _, group in candidates.groupby("source1_entity_id", sort=False):
        exact_mask = group.get("retrieved_by_exact", pd.Series(0, index=group.index)).astype(int).eq(1)
        exact = group[exact_mask]
        if len(exact) > total_budget and exact_overflow == "error":
            raise ValueError(f"exact-match expansion exceeds total budget for {group.iloc[0]['source1_entity_id']}")
        remaining = max(0, total_budget - len(exact))
        other = group[~exact_mask].copy()
        other["_score"] = pd.to_numeric(other.get("retrieval_score", 0.0), errors="coerce").fillna(0.0)
        other = other.sort_values(["_score", "candidate_source", "candidate_entity_id"], ascending=[False, True, True]).head(remaining).drop(columns="_score")
        parts.append(pd.concat([exact, other], ignore_index=True))
    return pd.concat(parts, ignore_index=True) if parts else candidates.iloc[:0].copy()


def save_candidates(candidates: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "name_tfidf_score", "address_tfidf_score", "name_rank", "address_rank", "retrieved_by_name", "retrieved_by_address", "retrieved_by_exact", "retrieval_provenance", "retrieval_score"]
    candidates.reindex(columns=[column for column in columns if column in candidates.columns]).to_csv(output, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-format entity-resolution candidates.")
    parser.add_argument("--s1", required=True); parser.add_argument("--s2", required=True); parser.add_argument("--s3", required=True); parser.add_argument("--gt", default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--top-k-name", type=int, default=50); parser.add_argument("--top-k-address", type=int, default=50); parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", required=True); parser.add_argument("--report", required=True); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); del args.seed
    started = time.perf_counter()
    s1, s2, s3 = load_source(args.s1, args.subset), load_source(args.s2), load_source(args.s3)
    candidates = generate_candidates(s1, s2, s3, args.top_k_name, args.top_k_address, args.batch_size)
    save_candidates(candidates, args.output)
    gt = load_ground_truth(args.gt) if args.gt else None
    if gt is not None:
        gt = {str(s1_id): values for s1_id, values in gt.items() if str(s1_id) in set(s1["entity_id"].astype(str))}
    report = candidate_diagnostics(candidates, s1, s2, s3, gt, time.perf_counter() - started)
    report_path = Path(args.report); report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(candidates)} candidate pairs to {args.output}")


if __name__ == "__main__":
    main()
