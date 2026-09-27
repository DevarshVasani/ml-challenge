# Candidate artifact contract

Candidate generation produces one long-format row per unique
`(source1_entity_id, candidate_entity_id)` pair. The required columns are:

`source1_entity_id`, `candidate_entity_id`, `candidate_source`,
`name_tfidf_score`, `address_tfidf_score`, `name_rank`, `address_rank`,
`retrieved_by_name`, `retrieved_by_address`, and `retrieval_score`.

`candidate_source` is `S2` or `S3`; IDs must use the matching `S2-` or `S3-`
prefix, and Source 1 IDs use `S1-`. Scores are finite, ranks are non-negative
integers, and retrieval flags are 0/1. Candidate generation is retrieval-only:
ground truth must never be used to add or select candidate rows.

## Partitioning

Recommended layout is `candidates/split=<split>/country=<country>/bucket=<id>/`.
The bucket is:

```text
stable_hash(source1_entity_id, seed) % number_of_buckets
```

All rows for one Source 1 entity must be in one bucket. Keep
`candidate_source` as a regular column so S2 and S3 rows can be compared
together when context features are computed.

Each artifact is accompanied by `candidate_manifest.json` containing:

```text
schema_version, split, git_commit, seed, top_k_name, top_k_address,
minimum_score_settings, number_of_buckets, partition_columns, row_count,
created_at
```

The implementation in `src.artifact_contract.py` validates both candidate
partitions and manifests. Unmatched entities may use the deterministic
SHA-256 fold fallback in `src.fold_lookup.py`; explicit positive-component
assignments always take priority.

