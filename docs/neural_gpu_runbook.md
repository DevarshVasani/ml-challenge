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
   python -m src.environment_status --artifact-dir artifacts --config configs/neural/data_pilot.json
   python -m src.preprocess_status --config configs/neural/index_pilot.json
   python -m src.build_candidate_index --config configs/neural/index_pilot.json --dry-run
   python -m src.build_candidate_index --config configs/neural/index_pilot.json --execute
   python -m src.build_source_store --config configs/neural/source_store_pilot.json --dry-run
   python -m src.build_source_store --config configs/neural/source_store_pilot.json --execute
   ```

   An interrupted disk-backed build resumes from `build_state.json`; source-store
   rows resume from `artifacts/source-store.sqlite.inprogress`. To perform full
   checksum/source validation, or reconstruct a missing root manifest only after
   validation, use:

   ```bash
   python -m src.preprocess_status --config configs/neural/index_pilot.json --verify
   python -m src.preprocess_status --config configs/neural/index_pilot.json --recover-manifest
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

Query batching bounds query-side sparse multiplication and output buffering.
The default disk-backed builder makes one counting pass with exact term/document
frequencies spilled to SQLite, creates one global vocabulary/IDF, and transforms
bounded source chunks into uncompressed, memory-mapped sparse arrays. Querying
opens one source shard and one channel at a time and merges shard-local top-K to
global top-K. Independently fitted shards are never used because their cosine
scores would not be comparable. `max_vocabulary` remains 500,000 and K remains
per source/field; resource pressure never lowers either automatically.
New disk-backed artifacts record deterministic ties (term frequency then term
text for vocabulary selection; score then candidate ID for retrieval) and
float32 scores. This can order exact score ties differently from a legacy
native-kernel index; score comparisons use a `1e-6` synthetic-test tolerance.

Candidate and pair files are atomic stable-query-batch shards. Their manifests
record every channel completion, including zero-row results, plus content
checksums. Resume verifies files before skipping them. Pair preparation accepts
only exact frozen-query coverage and publishes an immutable bootstrap manifest
after enough complete leading training batches exist.

For 16 GB system RAM, keep `index_workers=1`, `sparse_threads=2`, 128 query rows,
and 25,000 source rows per transform chunk. Keep about 30% of usable host/cgroup
memory free and use the disk-backed backend. For 32–64 GB system RAM, first
measure peak RSS and I/O; then increase source/query chunks cautiously, while
still keeping one index channel resident. GPU VRAM is irrelevant to these CPU,
SQLite, and sparse-index stages. Reserve 150–250 GB free disk only as a planning
allowance; the environment/status output and measured shard sizes decide the
actual requirement.

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
- Member B adapter implemented and tested offline on a tiny random DeBERTa-v2 model; see `docs/member_b_lightning_runbook.md` for the GPU steps.
- Member C implementation pending: the ByT5 encoder adapter intentionally raises an actionable error.
- GPU validation not run: fit, throughput, memory, accuracy, and real checkpoint compatibility remain unmeasured.
- Exact resume currently requires deterministic single-process loading with `num_workers=0`.
- Real source/index execution, blocking recall, throughput, peak RSS, the first real pair shard, and the final shared manifest remain outstanding user-run work.

## Recovery decisions

| Existing artifact | Action |
| --- | --- |
| Metadata and checksum-valid completed index | Reuse it; query-time K/batch/thread changes do not rebuild vocabulary or source vectors. |
| `build_state.json` or source-store `.inprogress` with matching input/config identity | Resume from the committed source/batch checkpoint. |
| Metadata missing or source/config identity changed | Preserve it and build into a new versioned directory. |
| Valid components but missing root manifest | Run status with `--verify`, then explicitly use `--recover-manifest`. |
| Legacy TF-IDF + exact pickle with weak exact provenance | Report `complete_unverified`; validate eagerly only on a machine where the pickle fits, or rebuild the exact component as SQLite in a new versioned location. |
| `channel_manifest.json` with compatible query IDs/channels | Resume; completed channel batches, including empty ones, are retained. |
| Channel file exists but its completion flag/checksum is absent or invalid | Treat only that batch/channel as incomplete and regenerate it. |
| Candidate coverage is missing for any selected query | Refuse pair preparation; do not treat it as zero candidates. |
| Corrupt/truncated Parquet | Leave unrelated batches intact; resume regenerates the named invalid batch. |
| Legacy TF-IDF pickle cannot fit RAM for explicit validation/conversion | Leave it untouched and rebuild disk-backed from source, or validate/convert on a larger-system-RAM host; never lower K silently. |
| Immutable bootstrap shard already published | Never overwrite it; later manifests may reference it by checksum. |

The implementation does not promise that a legacy TF-IDF index can be converted within
the constrained budget: vectorizer fitting can temporarily require more memory than the
final CSR estimate. Measure peak RSS before attempting that one-time migration.
