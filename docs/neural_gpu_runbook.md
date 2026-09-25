# Neural GPU runbook

The repository contains infrastructure only. No competition-data preparation,
checkpoint download, GPU run, recall measurement, or performance validation
was performed during implementation. Install a PyTorch build that matches the
compute machine's CUDA driver, then install `requirements.txt` and the
explicitly unverified packages in `requirements-gpu.txt`.

All expensive commands require `--execute`. `--help` and `--dry-run` parse
metadata only: they do not iterate datasets, initialize models, inspect CUDA,
or access the network.

## Ordered future workflow

1. Create and verify the connected-component fold map with `src.split`. The
   neural data config expects `artifacts/folds.tsv`; all known-positive
   endpoints must be present. Hash fallback is used only for genuinely
   unmatched retrieval endpoints.

2. Build the reusable source indexes and disk-backed raw source store:

   ```bash
   python -m src.build_candidate_index --config configs/neural/index_pilot.json --dry-run
   python -m src.build_candidate_index --config configs/neural/index_pilot.json --execute
   python -m src.build_source_store --config configs/neural/source_store_pilot.json --dry-run
   python -m src.build_source_store --config configs/neural/source_store_pilot.json --execute
   ```

3. Freeze the train/threshold/final query selection, then query the cached
   indexes. The first command deliberately writes only the query manifest so
   candidate retrieval can use that exact selection.

   ```bash
   python -m src.prepare_neural_data --config configs/neural/data_pilot.json --execute --queries-only
   python -m src.query_candidate_index --config configs/neural/candidate_query_pilot.json --dry-run
   python -m src.query_candidate_index --config configs/neural/candidate_query_pilot.json --execute
   python -m src.prepare_neural_data --config configs/neural/data_pilot.json --execute
   ```

   The full preparation command joins raw text through SQLite, injects missed
   positives only into training, samples negatives, and writes grouped raw pair
   shards shared by both models. It never pretokenizes the full dataset.

4. After the owner implements an adapter, train through the shared loop:

   ```bash
   python -m src.train_neural --config configs/neural/mdeberta_pilot.json --dry-run
   python -m src.train_neural --config configs/neural/mdeberta_pilot.json --execute
   python -m src.train_neural --config configs/neural/byt5_pilot.json --dry-run
   python -m src.train_neural --config configs/neural/byt5_pilot.json --execute
   ```

5. Score threshold and final pair manifests separately with
   `src.predict_neural`. Example configs are
   `predict_mdeberta_threshold_pilot.json`, `predict_mdeberta_pilot.json`, and
   `predict_byt5_pilot.json`. Prediction shards are atomic and resumable only
   when checkpoint, configuration, query manifest, candidate manifest, and
   input content identities match.

6. Tune only on `threshold`, freeze the threshold, and evaluate `final` once:

   ```bash
   python -m src.evaluate_neural --config configs/neural/evaluate_pilot.json --dry-run
   python -m src.evaluate_neural --config configs/neural/evaluate_pilot.json --execute
   ```

7. For a test query manifest and selected checkpoint, stream the official
   files without accumulating all pairs:

   ```bash
   python -m src.export --config configs/neural/export_pilot.json --dry-run
   python -m src.export --config configs/neural/export_pilot.json --execute
   ```

8. Candidate K and optional total-budget pruning can be compared without
   refitting TF-IDF:

   ```bash
   python -m src.experiment_c_k_tuning --config configs/neural/k_tuning_pilot.json --dry-run
   python -m src.experiment_c_k_tuning --config configs/neural/k_tuning_pilot.json --execute
   ```

   K is per field/source channel, not a total cap. Exact candidates are kept
   separately. Total-budget pruning is disabled in production unless selected
   explicitly after measured tuning; exact overflow defaults to exceeding the
   budget rather than silently dropping exact matches.

## Resource and artifact assumptions

Query batching bounds query-side sparse multiplication and output buffering;
it does not bound the fitted source vocabulary or full source TF-IDF matrix.
`max_vocabulary`, query batch size, and a declared matrix budget are
configurable. The builder records the measured CSR matrix size and refuses a
matrix over the declared budget, but peak vectorizer construction memory can
be higher and must be measured on the compute machine. Independently fitted
source shards are not supported because their cosine scores would not be
comparable; current indexes use one common vocabulary/IDF per source/field.

Grouped writers target 75,000 rows, never split an S1 group, and report an
oversized single group. Query manifests retain every selected S1, including
zero-candidate entities. Pair, checkpoint, prediction, and index manifests
record configuration/input identities, shard order, counts, checksums, and
completion state. Use a new output directory after a failed non-resumable data
preparation run.

Inference planning uses:

```text
estimated seconds = final candidate-pair count / measured pairs per second
```

Pass a measured value in prediction configuration; no default throughput is
invented. Historical planning context only: a prior audit reported 1,732,544
test S1 queries, approximately 34.7 million pairs at 20 candidates/query.

## Current status and limitations

- Infrastructure implemented: offline contracts and fake-adapter workflow are complete.
- Data preparation not run: real indexes, stores, candidates, and pair shards await compute.
- Member B implementation pending: the mDeBERTa adapter intentionally raises an actionable error.
- Member C implementation pending: the ByT5 encoder adapter intentionally raises an actionable error.
- GPU validation not run: fit, throughput, memory, accuracy, and real checkpoint compatibility remain unmeasured.
- Exact resume currently requires deterministic single-process loading with `num_workers=0`.
- Source indexes are not sharded. If sharding is added, all shards must share one fitted vocabulary/IDF and global top-K merging.
