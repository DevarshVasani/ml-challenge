# Amazon ML Challenge - Business Entity Resolution

This repository resolves Source 1 businesses against matching records in Sources 2 and 3. It contains:

- a classical TF-IDF and pair-model pipeline;
- a scalable, resumable data-preparation path shared by the neural model owners;
- leakage-safe connected-component folds;
- candidate, pair, prediction, and export contracts with coverage checks;
- offline synthetic tests that do not require model downloads or competition-scale preprocessing.

The scalable preparation path is the recommended route for the shared neural dataset. It uses global TF-IDF statistics, memory-mapped source shards, SQLite exact-match and source stores, one loaded retrieval channel at a time, and durable batch manifests.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate er-challenge
```

GPU-only dependencies are listed separately in `requirements-gpu.txt`. A GPU is not used for index building, candidate retrieval, source-store construction, or pair preparation.

Expected training files:

```text
student_resource/dataset/train/
  train_source1.tsv
  train_source2.tsv
  train_source3.tsv
  train_ground_truth.tsv
```

Test sources belong under `student_resource/dataset/test/`.

## Recommended scalable preparation workflow

The index-build, source-store, candidate-query, and neural-preparation commands below require `--execute`. Running those commands without `--execute`, or with `--dry-run`, only validates configuration and reports cheap file metadata.

### 1. Inspect the machine and existing artifacts

These commands do not scan the datasets, load index pickles, initialize CUDA, download models, or launch jobs:

```bash
python -m src.environment_status \
  --artifact-dir artifacts \
  --config configs/neural/data_pilot.json

python -m src.preprocess_status \
  --config configs/neural/index_pilot.json

python -m src.build_candidate_index \
  --config configs/neural/index_pilot.json \
  --dry-run

python -m src.build_source_store \
  --config configs/neural/source_store_pilot.json \
  --dry-run
```

Use `preprocess_status --verify` only when a full source and artifact checksum pass is intended:

```bash
python -m src.preprocess_status \
  --config configs/neural/index_pilot.json \
  --verify
```

### 2. Create and verify leakage-safe folds

```bash
python -m src.split \
  --data-dir student_resource/dataset/train \
  --out-folds artifacts/folds.tsv \
  --out-summary reports/split_summary.json \
  --verify
```

Known-positive connected components are kept in one fold. Missing folds for known-positive endpoints are errors. Hash fallback is used only for unmatched retrieved endpoints, using the same versioned algorithm throughout the repository.

### 3. Build or resume source indexes and the raw-text store

```bash
python -m src.build_candidate_index \
  --config configs/neural/index_pilot.json \
  --execute

python -m src.build_source_store \
  --config configs/neural/source_store_pilot.json \
  --execute
```

The default index backend:

- streams source TSV columns instead of loading the full source table;
- spills exact corpus term and document frequencies to SQLite;
- fits one global vocabulary and IDF per source/field channel;
- stores query-ready sparse matrices as uncompressed, memory-mapped arrays;
- stores full nonempty exact-match postings in SQLite, including duplicate normalized strings;
- checkpoints source batches and resumes only when the source/configuration identity matches.

`top_k=100` applies independently to each TF-IDF source/field channel. It is not a total per-query candidate cap. Exact matches are not truncated.

### 4. Freeze the query selection

This is an expensive dataset-processing command, not a metadata-only status command:

```bash
python -m src.prepare_neural_data \
  --config configs/neural/data_pilot.json \
  --execute \
  --queries-only
```

It writes `artifacts/neural-data/query_manifest.json`, freezing the train, threshold-tuning, and final-comparison query IDs. A compatible existing manifest is consumed unchanged rather than resampling queries.

The pilot requests 50,000 training queries and 5,000 held-out queries, split component-safely into approximately 2,500 threshold and 2,500 final queries. Connected components can make actual counts differ slightly from the requested targets.

### 5. Query every retrieval channel

```bash
python -m src.query_candidate_index \
  --config configs/neural/candidate_query_pilot.json \
  --dry-run

python -m src.query_candidate_index \
  --config configs/neural/candidate_query_pilot.json \
  --execute
```

Retrieval is channel-major and defaults to one resident index worker. Every stable query batch records completion for every exact and TF-IDF channel, including channels that returned zero rows. After all channels for a batch are durable and checksummed, they are unioned into an immutable candidate batch.

Re-running with `"resume": true` skips only compatible, checksum-valid work. A corrupt or truncated channel file is regenerated without discarding unrelated batches.

### 6. Create the shared raw-text pair shards

```bash
python -m src.prepare_neural_data \
  --config configs/neural/data_pilot.json \
  --execute
```

Pair preparation:

- requires exact coverage of the frozen query manifest;
- never interprets an unprocessed validation query as having zero candidates;
- injects missed positives only for training and reports them separately from retrieval recall;
- excludes held-out endpoints from training negatives;
- joins only the current candidate batch through the SQLite source store;
- writes atomic, checksummed, resumable train/threshold/final pair batches;
- publishes an immutable bootstrap manifest for the leading completed training batches;
- preserves the pair-column and manifest interface used by both neural models.

The primary outputs are:

```text
artifacts/neural-data/
  query_manifest.json
  pair_progress.json
  bootstrap_pairs_manifest.json
  train_pairs_manifest.json
  threshold_pairs_manifest.json
  final_pairs_manifest.json
  train_pairs/
  threshold_pairs/
  final_pairs/
```

## Artifact and resume model

Important preparation artifacts are organized as follows:

```text
artifacts/
  candidate-index/
    manifest.json
    build_progress.json
    S2_business_name/
      disk_metadata.json
      build_state.json
      vectorizer.pkl
      vocabulary_counts.sqlite
      matrix-*-{data,indices,indptr}.npy
      ids-*.json
      exact_sqlite.db
      exact_sqlite.json
    ...
  source-store.sqlite
  source-store.sqlite.manifest.json
  neural-candidates/
    channel_manifest.json
    manifest.json
    channels/
    final/
  neural-data/
    ...
```

Recovery rules:

| Observed state | Behavior |
| --- | --- |
| Completed component with compatible identities and valid checksums | Reuse it. |
| Compatible `build_state.json` | Resume the disk-backed index from committed rows/shards. |
| Compatible `source-store.sqlite.inprogress` | Resume source insertion from the committed row checkpoint. |
| Completed zero-result retrieval batch | Preserve it as valid coverage. |
| Missing or corrupt channel batch | Regenerate only that batch/channel. |
| Changed source, index semantics, frozen queries, or candidate policy | Refuse reuse and require a new versioned output. |
| Valid components but missing root index manifest | Verify first, then run `preprocess_status --recover-manifest`. |
| Legacy pickle without strong exact-index provenance | Report it as unverified; validate on sufficient system RAM or rebuild into a new versioned location. |
| Missing candidate coverage for any frozen query | Refuse pair preparation. |
| Published bootstrap shard | Never overwrite it; later manifests may reference the same checksum. |

Do not delete a nonempty artifact root merely to bypass compatibility checks. Status inspection preserves existing files by default and never unpickles a legacy index.

## Resource guidance

Conservative defaults are:

```json
{
  "index_workers": 1,
  "sparse_threads": 2,
  "query_batch_size": 128,
  "source_read_chunk_rows": 25000,
  "pair_join_target_rows": 25000,
  "shard_target_rows": 75000,
  "bootstrap_train_queries": 1000,
  "resume": true,
  "strict_country": false
}
```

For approximately 16 GB of system RAM, retain these defaults and keep roughly 25-35% of usable host or cgroup memory free for Python, the operating system, and transient sparse operations. For 32-64 GB, increase source or query chunks only after measuring peak RSS and disk throughput; keep one index channel resident unless a measured experiment justifies otherwise.

Disk requirements depend on vocabulary density, source sizes, exact-posting multiplicity, and candidate counts. Reserving 150-250 GB free is only a planning allowance, not a guarantee. Use `environment_status`, component sizes, and a small real benchmark to decide whether the machine is sufficient. More GPU VRAM does not help these stages.

Recommended benchmark ladder for the user-run environment:

1. Inspect status and actual host/cgroup RAM and free disk.
2. Validate or recover the existing S2 business-name component.
3. Run 500-1,000 frozen bootstrap queries against the full source background.
4. Record wall time, peak RSS, index/shard bytes, candidate counts, recall, and pairs/second.
5. Increase to 5,000 queries and project the remaining pilot from measured throughput.
6. Run the full frozen pilot only if measured memory and time are acceptable.

Real-data performance and hardware sufficiency have not been measured in this repository.

## Neural training and evaluation

After shared pair manifests exist, follow [the neural GPU runbook](docs/neural_gpu_runbook.md) and [the adapter contract](docs/neural_adapter_contract.md).

Example training commands:

```bash
python -m src.train_neural --config configs/neural/mdeberta_pilot.json --dry-run
python -m src.train_neural --config configs/neural/mdeberta_pilot.json --execute

python -m src.train_neural --config configs/neural/byt5_pilot.json --dry-run
python -m src.train_neural --config configs/neural/byt5_pilot.json --execute
```

Tune a decision threshold only on the threshold subset. Freeze it before evaluating the final-comparison subset.

The mDeBERTa and ByT5 production adapters remain model-owner work. The shared contracts, fake-adapter smoke workflow, training loop, resumable prediction, evaluation, and streaming export infrastructure are present.

## Classical pipeline

The original in-memory/classical workflow remains available for smaller runs and experiments:

```bash
python -m src.pipeline \
  --data-dir student_resource/dataset \
  --work-dir artifacts \
  --output-dir output \
  --n-splits 3 \
  --top-k-name 50 \
  --top-k-address 50 \
  --seed 42
```

Its stages are also callable separately through `src.candidates`, `src.features`, `src.train_pair_model`, `src.predict_pair_model`, and `src.export`. Do not confuse this path with the disk-backed shared neural preparation workflow above.

Official export files are:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

## Evaluation metric

The challenge metric is macro-averaged F0.5 over all Source 1 entities.

For a non-singleton query:

```text
F0.5 = 5 * TP / (5 * TP + 4 * FP + FN)
```

For a singleton with no true matches, an empty prediction scores 1.0 and any predicted match scores 0.0. Zero-candidate queries remain in the macro denominator.

Run the metric self-check with:

```bash
python -m src.evaluate --check
```

## Testing

The test suite uses small synthetic fixtures and does not run competition-scale preprocessing or download models:

```bash
python -m pytest -q
```

The current implementation is covered for global-IDF sharding equivalence, duplicate exact postings, deterministic folds and pair sampling, one-channel residency, zero-result coverage, corruption retry, frozen-query pair preparation, immutable bootstrap manifests, streaming metrics, and resume/input invalidation.

## Current project status

Implemented:

- disk-backed global-IDF candidate indexes and SQLite exact postings;
- source-store and index build checkpoints;
- channel-sequential candidate retrieval with checksummed batch resume;
- frozen component-safe query manifests;
- coverage-aware train/threshold/final pair preparation;
- immutable bootstrap pair manifests;
- incremental retrieval diagnostics and environment/status commands;
- shared neural contracts, fake-adapter testing, prediction, evaluation, and export infrastructure.

Not yet produced or measured:

- real competition source indexes and source store;
- real blocking recall, throughput, peak RSS, and disk usage;
- the first real bootstrap pair shard;
- the final immutable shared dataset manifest;
- trained production mDeBERTa and ByT5 checkpoints.

Record unavailable experiment values as `N/A` in `reports/experiments.csv`; do not replace them with fabricated zeros.
