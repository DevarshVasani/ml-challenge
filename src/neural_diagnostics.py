"""Streaming diagnostics for candidate retrieval and pair preparation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .evaluate import score_single_s1


def _bin(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4+"


@dataclass
class RetrievalDiagnostics:
    """Accumulate per-query metrics without retaining candidate pairs."""

    ground_truth: Mapping[str, Iterable[str]]
    query_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    counts: list[int] = field(default_factory=list)
    zero_candidate_queries: int = 0
    total_pairs: int = 0
    exact_only: int = 0
    tfidf_only: int = 0
    overlap: int = 0
    frequent_key_expansions: int = 0
    recall_hits: Counter[str] = field(default_factory=Counter)
    recall_total: Counter[str] = field(default_factory=Counter)
    oracle_scores: list[float] = field(default_factory=list)
    by_country: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    by_match_bin: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add_query(self, query_id: str, candidates: Iterable[Mapping[str, Any]]) -> None:
        true = {str(value) for value in self.ground_truth.get(str(query_id), []) if str(value)}
        rows = candidates.to_dict("records") if hasattr(candidates, "to_dict") else list(candidates)
        ids = {str(row.get("candidate_entity_id", "")) for row in rows if row.get("candidate_entity_id")}
        count = len(ids)
        self.counts.append(count)
        self.total_pairs += len(rows)
        self.zero_candidate_queries += int(count == 0)
        for row in rows:
            exact = bool(row.get("retrieved_by_exact")) or str(row.get("retrieval_provenance", "")) == "exact"
            tfidf = (float(row.get("name_tfidf_score", 0) or 0) > 0 or float(row.get("address_tfidf_score", 0) or 0) > 0 or "tfidf" in str(row.get("retrieval_provenance", "")))
            if exact and tfidf:
                self.overlap += 1
            elif exact:
                self.exact_only += 1
            elif tfidf:
                self.tfidf_only += 1
            if int(row.get("frequent_key_expansion", 0)):
                self.frequent_key_expansions += 1
        hits = len(true & ids)
        self.recall_hits["overall"] += hits
        self.recall_total["overall"] += len(true)
        for source in ("S2", "S3"):
            source_true = {value for value in true if value.upper().startswith(source)}
            source_ids = {str(row.get("candidate_entity_id", "")) for row in rows if str(row.get("candidate_source", "")).upper() == source}
            self.recall_hits[f"source:{source}"] += len(source_true & source_ids)
            self.recall_total[f"source:{source}"] += len(source_true)
        metadata = self.query_metadata.get(str(query_id), {})
        country = str(metadata.get("country", ""))
        match_bin = str(metadata.get("match_bin", _bin(len(true))))
        self.recall_hits[f"country:{country}"] += hits
        self.recall_total[f"country:{country}"] += len(true)
        self.recall_hits[f"match_count:{match_bin}"] += hits
        self.recall_total[f"match_count:{match_bin}"] += len(true)
        self.by_country[country].append(count)
        self.by_match_bin[match_bin].append(count)
        self.oracle_scores.append(score_single_s1(true, ids))

    def report(self) -> dict[str, Any]:
        ordered = sorted(self.counts)
        percentile = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))] if ordered else 0
        recall = {
            key: (self.recall_hits[key] / self.recall_total[key] if self.recall_total[key] else 1.0)
            for key in self.recall_total
        }
        return {
            "queries": len(self.counts),
            "total_pairs": self.total_pairs,
            "zero_candidate_queries": self.zero_candidate_queries,
            "candidate_count": {"mean": (sum(self.counts) / len(self.counts) if self.counts else 0.0), "p50": percentile(0.50), "p95": percentile(0.95), "max": max(self.counts, default=0)},
            "recall": recall,
            "oracle_macro_f05": (sum(self.oracle_scores) / len(self.oracle_scores) if self.oracle_scores else 1.0),
            "exact_only": self.exact_only,
            "tfidf_only": self.tfidf_only,
            "overlap": self.overlap,
            "frequent_key_expansions": self.frequent_key_expansions,
            "by_country_counts": {key: values for key, values in self.by_country.items()},
            "by_match_bin_counts": {key: values for key, values in self.by_match_bin.items()},
        }
