# Retrieval handoff — 2026-09-27 00:43 UTC

## Production decision

Use the frozen `reverse-union-v1` policy in `artifacts/reverse-v1/frozen_policy.json`: S2/S3 source records query the complete same-country S1 background with reverse lexical top 50 and Granite exact dense top 20. The selected policy averages **35.31 unique candidates per source record** on the 103,321-record unlabeled training sample (p95 69, p99 70). The completed test union has **9,969,589 closed source groups and 357,134,359 distinct pairs** (35.82/group); all eight bucket checksums and Parquet row counts were verified. Use `artifacts/reverse-v1/test_union/manifest.json` for test candidate features now. Its candidate rows contain no labels.

The cheaper lexical-20 + dense-20 variant averages 27.52 candidates/group (p95 40) but lowers full threshold reverse oracle from **0.978558** to **0.976977** and positive recall from **0.950689** to **0.948374**. Lexical 50 gains recall with a still bounded candidate set and is already generated for full test. No new embedding production run is needed.

The historical S1-forward baseline is a validation diagnostic, not a test production channel. Its union with reverse retrieval scores **0.992927 threshold** and **0.995945 frozen final** oracle. Reverse production alone scores **0.978558 threshold** and **0.981742 frozen final**. The requested **0.998 final target is not met** by a deployable policy. Do not report the historical-forward union as a completed test artifact.

## Fresh 50-query threshold replay

The first 50 SHA-256 ordered frozen threshold S1 queries (`artifacts/handoff-retrieval-v1/threshold_query_ids_50.txt`) have 165 distinct positive S2/S3 source endpoints. The production lexical and dense implementations reran against all 2,206,821 training S1 references. They closed all 165 source groups and produced **5,659 distinct pair keys** in `artifacts/handoff-retrieval-v1/subset_union.parquet`. Fresh oracle and recall match the earlier full-background rank artifacts exactly. This is a **positive-endpoint diagnostic**; its per-S1 candidate counts are not a complete validation export. Arbitrary-record candidate volume is measured on the 103,321-record unlabeled sample above.

| Nested S1 queries | Gold edges | Reverse oracle macro F0.5 | Positive-pair recall | Zero-positive retrieved queries |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 16 | 0.994444 | 0.937500 | 0 |
| 10 | 27 | 0.897222 | 0.925926 | 1 |
| 20 | 60 | 0.948611 | 0.966667 | 1 |
| 50 | 165 | 0.974092 | 0.969697 | 1 |

These nested subsets are small and volatile; the 10-query drop comes from one newly included query with zero recovered positives. Full threshold and final metrics above are the reliable comparisons. `artifacts/handoff-retrieval-v1/{fresh_subset_report.json,threshold_subset_report.json}` contain exact counts and provenance. The latter also reports top-5/10/20/50 **candidate-rank** sensitivity across all 2,500 threshold queries, in case “5/10/20/50 queries” meant candidate budgets.

## Experiments closed

- Widening all four historical forward TF-IDF channels to rank 500 found 51 additional threshold positives beyond the baseline/reverse union, raising threshold oracle to 0.995906. It took **751.7 seconds for only 203 diagnostic S1 queries**, so it is infeasible for 1.73 million test S1 records. It also requires a missing full test forward index.
- A full-S1 phonetic name/address index improved the baseline/reverse threshold oracle to 0.995087 at top 50 (33 new positives). On a 1,000-record non-Latin representative sample it processed **10.9 records/second**, projecting roughly 19 hours for the non-Latin share alone. It is not a production channel. Its index and pilot remain in `artifacts/phonetic-v2/` for research.
- A broader reverse address rescue found 51 positives at top 50 and reached 0.995362 threshold oracle, but needed many token-pair probes per source record. The cheaper three-probe form found only 29 of 134 residual positives in its candidate pool. It was not deployed.
- The diagnostic union of historical forward rank 500 and phonetic top 50 reached **0.997018 threshold oracle**. It did not reach 0.998 and both additions failed production throughput requirements. No further frozen-final tuning or evaluation was performed.
- The incomplete experimental forward FTS build was stopped and its partial index removed. The production queues were left running.

## Production state and next handoff action

Dense test and train candidate manifests are complete. The test lexical and merged union manifests are complete. Lexical train retrieval remains active in the detached queue, with approximately **6.49M / 10.32M (62.9%)** source groups closed at this snapshot; `merge` is waiting for lexical train. Queue states are `artifacts/reverse-v1/{dense,lexical,merge}_queue_state.json`; the queue logs are under `artifacts/`. The merge queue will publish the train union after lexical train completes. Re-running a stopped queue with `python -m scripts.production_queue lexical` or `python -m scripts.production_queue merge` resumes compatible shards; check for an existing process first.

For downstream members: consume the complete test union now; wait for `artifacts/reverse-v1/train_union/manifest.json` to say `complete: true` before train feature extraction. Preserve `(source, source_record_id, s1_id)` as the pair key and keep truth/labels outside candidate features. The frozen query list, integrity audit, and fresh subset output are under `artifacts/handoff-retrieval-v1/`.

Reproduce the subset verification from the repository root:

```bash
python -m scripts.handoff_retrieval_subset
python -m scripts.prepare_handoff_subset
python -m src.reverse_retrieval retrieve --split train --root artifacts/reverse-v1 --out artifacts/handoff-retrieval-v1/lexical_candidates --k 50 --batch-size 1000 --source2 artifacts/handoff-retrieval-v1/S2_source_endpoints.tsv --source3 artifacts/handoff-retrieval-v1/S3_source_endpoints.tsv
python -m src.dense_retrieval retrieve --split train --out artifacts/dense-v1 --candidate-out artifacts/handoff-retrieval-v1/dense_candidates --sample artifacts/handoff-retrieval-v1/source_endpoints_50_queries.parquet --k 20 --batch-size 1000
python -m scripts.finalize_handoff_subset
```

Focused tests: `python -m pytest -q tests/test_reverse_retrieval.py tests/test_retrieval_experiments.py` — **4 passed**. The test union integrity audit checked eight SHA-256 hashes and 357,134,359 Parquet rows.
