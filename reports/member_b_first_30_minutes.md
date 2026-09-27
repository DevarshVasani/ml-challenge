# Member B: first 30-minute handoff

## Ready for A, C, and D

The immutable starter package is [`artifacts/member-b-starter-v1/`](../artifacts/member-b-starter-v1/). Its [`manifest.json`](../artifacts/member-b-starter-v1/manifest.json) lists every file, row count, and SHA-256 checksum. Regenerate it with `python scripts/build_member_b_starter.py` from the repository root. The script verifies the source shard checksums and atomically publishes each output.

| File | Contents | Consumer |
| --- | --- | --- |
| `train_queries.tsv`, `threshold_queries.tsv`, `final_queries.tsv` | Frozen S1 IDs, split, country: 50,000 / 2,500 / 2,500 rows | All |
| `train_candidate_pairs.parquet` | 1,691 keyed training pairs, retrieval scores/ranks, candidate version | C |
| `train_labels.parquet` | Keyed training labels and positive-injection flag; join only for fitting/evaluation | C |
| `train_records.parquet` | 1,819 referenced original-text records | A/C |
| `threshold_candidate_pairs.parquet` | 25,428 keyed pairs from 64 threshold queries; no labels | A/C/D |
| `threshold_records.parquet` | 25,290 referenced original-text records | A/C |
| `evaluation_truth.tsv` | All 5,000 threshold/final query gold IDs and full true-match counts, including zero-match queries | D only |
| `train_sample_queries.tsv`, `threshold_sample_queries.tsv` | Explicit sampled query IDs | All |

Pair-key mapping: `(candidate_source, candidate_entity_id, source1_entity_id)` corresponds to the shared `(source, source_record_id, s1_id)`. IDs remain strings. The bootstrap samples preserve complete **S1 query** groups from the historical shards. They do **not** preserve complete competing S1 groups for each S2/S3 record, so source-level ranks, margins, and ownership must wait for B's repartitioned candidate shards. The existing full historical pair manifests are at `../neural-data/{train,threshold,final}_pairs_manifest.json`; their shards have 672,948 / 993,174 / 993,198 rows. These are bootstrap inputs, not the new reverse-retrieval candidate version.

The record text is attached for the samples. For more records, use the read-only source TSVs under `../student_resource/dataset/train/`; the historical source-store SQLite file is absent. No teammate needs to build indexes to consume this package. Keep labels, true counts, split/fold IDs, and `positive_injected_for_training` outside the feature allowlist.

## Full-denominator retrieval baseline

Existing metrics from [`artifacts/retrieval-experiments/20260926-cpu-01/`](../artifacts/retrieval-experiments/20260926-cpu-01/) evaluate every frozen query, including zero-candidate queries. The current baseline retrieves roughly 397 candidates per S1 query. The CPU recovery-channel union is small and versioned; it is not reverse retrieval.

| Split | Queries | Gold positive edges | Missed edges | Queries with partial/complete misses | Complete misses | Raw oracle F0.5 | Recovery union oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Threshold | 2,500 | 8,639 | 252 | 203 | 12 | 0.987343 | 0.987643 |
| Final comparison | 2,500 | 8,605 | 237 | 187 | 7 | 0.990332 | 0.990514 |

The final-comparison set has already been inspected at aggregate level and must not drive repeated tuning. Threshold has 132 gold singletons; final has 147. The baseline has 12 threshold and 7 final non-singleton queries with zero retrieved positives. Of the affected queries, 191 threshold and 180 final are partial misses. The recovery union adds 688 threshold pairs and 756 final pairs. Its frozen policy hash is `6ede64dcaf28c170015942cd925bd80367fd5489a5d7baf33abd7f5b15a2732d`; score baseline-plus-delta as a deduplicated union if testing it.

## Source and machine inventory

| Source | Train rows | Test rows |
| --- | ---: | ---: |
| S1 | 2,206,821 | 1,732,544 |
| S2 | 5,034,616 | 4,887,273 |
| S3 | 5,285,603 | 5,082,316 |

The current environment exposes 4 CPUs, a 13.6 GiB cgroup memory limit, and about 327 GiB free disk. `nvidia-smi` is unavailable. Full candidate indexes, the source store, and the fold map referenced by historical manifests are absent; only the completed pair shards and the small CPU retrieval experiment are here. Do not assume the plan's 64 GiB CPU or retrieval GPU on this machine.

## B's next measured handoff

Pilot reverse lexical retrieval by indexing the full S1 background and querying S2/S3 records at top 20, retaining the historical channels as recovery. Measure build/search throughput, memory, and full-gold hit recovery before widening top-k. Provide a fallback when country is missing or inconsistent. A matched-endpoint pilot is only a recall diagnostic; complete validation must include unmatched/background S2/S3 records retrieving evaluation S1s. Repartition each published candidate shard by `(candidate_source, candidate_entity_id)` before C computes source-level competition. Publish only closed, checksum-validated shards and include all query IDs, including those with zero candidates. France has no labeled oracle; include it in test processing without presenting synthetic probes as accuracy.

The prior bounded index pilot built 50,000 S2-name rows in 21.0 seconds with 248.2 MiB child RSS. Its linear four-channel full-build projection was 2.41 hours before safety margin; that is a historical measurement, not a benchmark of the new S1 reverse index. This environment needs a fresh bounded pilot before committing to production timing.
