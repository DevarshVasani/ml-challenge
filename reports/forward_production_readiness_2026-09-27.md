# Forward plus reverse production readiness (2026-09-27)

## What the measured scores mean

The reported 0.992927 threshold and 0.995945 frozen-final oracle scores include the **historical train-only S1-forward TF-IDF/exact candidate shards**. The complete test reverse union contains 357,134,359 pairs and 9,969,589 closed source groups, but no equivalent historical forward test index or complete forward candidate run exists. These validation scores cannot be assigned to a test production artifact.

The available full-background forward FTS pilot is a distinct policy. On the frozen 2,500-query threshold set, its top-50 union with reverse lexical-50 and dense-20 scored 0.983572 oracle and 0.963653 positive recall. It gained 0.005014 oracle over reverse alone (0.978558), while leaving 184 true edges that the historical forward channel found and the FTS/reverse union missed. The lost edges include cross-script names and addresses with changed tokens; expanding token probes without a bounded throughput and threshold test would not establish the historical score.

## Time and compute

The forward FTS pilot processed 2,500 threshold queries in 109.6 seconds including startup (22.8 queries/second overall on one process). A simple linear estimate is 21.1 hours for all 1,732,544 test S1 queries on one process, or 2.6 hours at perfect eight-way scaling; this is an estimate, not a measured production rate. The historical rank-500 experiment took 751.7 seconds for 203 queries (3.70 seconds/query). At that measured rate, a full test pass would take roughly 74 days on one worker, before index construction, contention, and merge. The two policies have different accuracy and cost.

An exploratory fused sparse matrix for the historical S2 name channel took 6.2 seconds to query 32 real S1 records after 22.7 seconds of assembly, versus 19.1 seconds for the existing shard loop under load. Direct top-100 fusion changed six of 32 candidate sets at score ties. Top-128 fusion contained all historical top-100 pairs in those 32 queries, but that sample is too small to establish validation equivalence. `scripts/benchmark_fused_forward.py` is the full frozen-threshold gate; it queries all four historical channels plus exact postings and compares the reverse union oracle without using frozen-final misses.

## Production gate

`scripts/run_forward_partitions.py` now validates both source-index SHA-256 checksums before starting workers. It terminates sibling workers if any partition fails. Before publishing `run_report.json`, it audits every partition for one consistent policy, contiguous query ranges, expected total query coverage, Parquet row counts, and SHA-256 checksums. `src/merge_forward_reverse.py` repeats that audit and verifies reverse bucket checksums before unioning pair keys. The forward test output currently contains only an incomplete first partition shard, and the audit rejects it.

For a policy that aims to preserve the historical oracle, the next accuracy gate is a **new forward retrieval implementation** tested on the frozen threshold IDs against the full S2/S3 train backgrounds. It must recover the historical forward wins while meeting an explicit elapsed-time budget on the current 8-CPU/31-GiB machine. Only after that gate should the full test forward run be launched and merged with the completed reverse test artifact. The frozen final subset has already been used for the reported aggregate; do not tune rules on final misses.

Focused verification: `python -m pytest -q tests/test_forward_artifact.py tests/test_reverse_retrieval.py tests/test_retrieval_experiments.py` (6 passed). The whole test suite currently stops during collection because `src.predict_neural` imports a missing `manifest_shard_paths` symbol from `src.neural_contracts`; this is outside the retrieval changes.

## GPU threshold experiment (2026-09-27)

The fused GPU implementation was run over all 2,500 frozen threshold queries and all four historical train-background channels. With IDF cutoffs name 8/address 6, group size 64, local top 512 and output top 128, it completed in 68.2 seconds and the positive-only fused-forward plus reverse diagnostic was 0.993336. It missed 26 historical-forward positive edges. Lowering cutoffs to name 6/address 4 reduced those misses to 9, but raised elapsed time to 112.4 seconds and only raised the diagnostic to 0.993354. These are threshold-only positive-edge diagnostics, not deployed candidate scores or final-set measurements. Reproducible outputs/checkpoints are under `artifacts/gpu-forward-v5/` and `artifacts/gpu-forward-v6/`; the benchmark is `scripts/benchmark_fused_forward.py`.

The lower-cutoff measured rate projects to about 21.6 hours for 1,732,544 queries if run serially at the same batch throughput. Even the faster setting projects to about 13.1 hours. Those projections omit building a full test-background index (which is absent), writing and merging candidate rows, and integrity checks. The IDF-pruned score also changes historical TF-IDF cosine semantics. Therefore this run is an experiment, not a handoff-ready production artifact, and it does not demonstrate the 0.995945 frozen-final oracle. Do not present it as meeting the requested under-three-hour/0.995 gate.
