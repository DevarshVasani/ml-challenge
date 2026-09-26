# Member A — neural scoring runbook (final 24 hours)

A owns one thing: fast, keyed, raw neural scores for the pairs C routes to the
matcher. The working checkpoint is preserved as-is (`artifacts/neural-mdeberta`);
ByT5 training is stopped. `predict_neural` remains the pilot tool; production
scoring is `src.score_neural`.

## 1. Checkpoint card for D (first 30 minutes)

```bash
python -m src.neural_card \
  --checkpoint artifacts/neural-mdeberta \
  --train-config configs/neural/mdeberta_t4_pilot.json \
  --output artifacts/neural-card \
  --check-pairs artifacts/neural-data/threshold_pairs_manifest.json artifacts/neural-data/final_pairs_manifest.json
```

Publish `artifacts/neural-card/` (small) to D:

- `checkpoint_card.json`: `checkpoint_id`, input format (segments, separator,
  `max_length`, truncation), score definition, training step/config, and
  `inputs_verified` (today's train pair/query manifests hash-match what the
  checkpoint recorded; if false, say so and do not claim the provenance).
- `training_exposure.parquet`: every entity the matcher saw in training
  (`train_query`/`train_candidate`, fold, `seen_as_positive`). A component is
  clean for D's neural fusion/calibration only if none of its entities appears
  here. Training queries came from folds 1–2; threshold/final are fold 0.

## 2. Throughput budget for D (by hour 2)

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_threshold.json \
  --benchmark 10000 --max-tokens 8192,16384,32768 --project-pairs <C's routed pair count>
```

Report `pairs_per_second_overlapped`, the best `max_tokens`, peak GPU memory
and `projection.hours_*`. The report is also saved under
`<output_dir>/benchmark/`. Set `max_tokens` in the scoring configs to the best
value. On an RTX 3050 laptop with untrained weights (same architecture), the
GPU forward pass took 97% of the time: ~650 pairs/s. CPU prep ran at ~19k
pairs/s. Token lengths were p50 47, p99 96, max 135, so nothing was truncated at
256. Measure the real GPU; do not reuse these numbers.

## 3. Threshold validation logits

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_threshold.json --dry-run
python -m src.score_neural --config configs/neural/score_mdeberta_threshold.json --execute
```

Labels in the pair shards are never read: only key and text columns are loaded.
Do not score `final` for tuning. Its existing predictions only reproduce the
reported baseline.

## 4. Production: C's `route=neural` shards

Edit `configs/neural/score_mdeberta_routes.json` (`pairs` glob, `key_columns`
mapping to C's column names, `output_dir` per candidate/route version), then:

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_routes.json --execute
```

- Key-only shards get their text from `source_store` via `record_text`, the same
  mapping used to build the training pairs. Shards that already carry the six
  text fields use them directly.
- With `require_complete_sidecar`, only shards whose `<shard>.complete.json`
  exists are scored. The rest are listed as `pending_inputs`. Rerunning is
  idempotent and picks up newly published shards, so loop it while C publishes:
  `while true; do python -m src.score_neural --config … --execute; sleep 120; done`.
- If an input shard changes after scoring, the run stops. `--rescore-changed`
  rescores it and reuses unchanged pairs. A different checkpoint or input format
  refuses to write into an existing output dir. Use a new `output_dir` per version.
- `reuse_from` lists earlier score dirs. A pair is reused only if its key, text,
  checkpoint and input format all match (a 64-bit hash over all of them).
- OOM splits the batch and retries. Rows that come out non-finite in fp16 are
  rescored in fp32.

## Output contract (A → D)

One `<input-stem>.neural.parquet` per input shard, plus a `.complete.json`
sidecar (input sha256, rows, status counts, output sha256), and
`neural_manifest.json` (`completion`: `complete`/`partial`, checkpoint and
scoring identity, `pending_inputs`).

| column | meaning |
|---|---|
| `source1_entity_id`, `candidate_entity_id`, `candidate_source` | pair key, strings |
| `neural_logit` | `logit[match] - logit[no_match]`, float32, uncalibrated; `sigmoid` = model probability. NaN iff not scored |
| `neural_scored` | true iff `neural_status` ∈ {`scored`, `cached`} |
| `neural_status` | `scored`, `cached`, `missing_record` (ID absent from source store), `nonfinite` |
| `checkpoint_id` | `ckpt-<hash of adapter weights/tokenizer/config>`; one per output dir |
| `cache_key` | hash of key + model text + checkpoint/input format |
| `truncated`, `token_length` | input diagnostics (`-1` when not tokenized in this run) |

Every selected (`route=neural`) input row gets exactly one output row. Missing
scores are explicit, never zero.

## Freeze rules

- Default to the existing checkpoint. A short continuation on new hard
  negatives is optional, only if production inference still fits the budget.
  Freeze by hour 10.
- A new checkpoint means a new `output_dir`. Never mix checkpoint IDs in one
  submission.
