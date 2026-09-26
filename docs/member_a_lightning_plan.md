# Member A — Lightning machine plan

Run steps in order on the T4 Studio. Commands and the output contract are in
`docs/member_a_runbook.md`; this is the sequence, the checks, and what to send
whom. Run everything long inside `tmux` so a closed tab does not kill it.

## Progress so far

| Step | State | Result |
|---|---|---|
| 0. Setup | done | torch/transformers match the training provenance |
| 1. Checkpoint card | done | `ckpt-e315e181e5896f9c`, `inputs_verified=true`, 0 held-out queries seen in training |
| 2. T4 benchmark | done | 1,028 pairs/s fp16 at `max_tokens=16384`, peak 3.9 GB, 0 OOM (table in step 2) |
| 3. Threshold logits | done | 993,174 pairs in `artifacts/neural-scores/threshold`; run the pilot comparison in step 3 if not done |
| 3b. B starter rehearsal | **do after 3** | laptop check passed; confirm on the Studio |
| 3c. Test source store | **start now** (CPU) | needed before any test route shard can be scored |
| 3d. C's threshold gate v1 | done | 963,843 routed pairs, all `cached` from step 3, 0 missing, 0 non-finite, 34 s; `artifacts/neural-scores/gate-threshold-v1` → D |
| 4–5. C's test route shards | **blocked on C** | still need C's routed pair count on B's new candidates; do not `--execute` production before the budget check in step 2 |

While waiting for C: B's machine has no GPU and a 13.6 GiB memory limit,
so B's new reverse retrieval, and therefore C's routed pair count, may arrive
later than the plan's timeline. Use that time for 3, 3b and 3c; nothing else
on A's side is blocked.

## 0. Setup (≈10 min) — before hour 0:30

```bash
cd /teamspace/studios/this_studio/ml-challenge
git fetch origin && git checkout team-a-neural && git pull
tmux new -s a
nvidia-smi                                   # Tesla T4, nothing else running
ls artifacts/neural-mdeberta/adapter artifacts/neural-mdeberta/checkpoint.json
ls artifacts/neural-data/{train,threshold,final}_pairs_manifest.json artifacts/neural-data/query_manifest.json
python -m pytest -q tests/test_score_neural.py tests/test_mdeberta_adapter.py
```

Check the scoring environment matches training (tokenization must not drift):

```bash
python -c "import json,torch,transformers; p=json.load(open('artifacts/neural-mdeberta/adapter/adapter_config.json'))['provenance']; print('trained with', p['transformers_version'], p['torch_version'], '| now', transformers.__version__, torch.__version__)"
```

If `transformers` differs, `pip install transformers==<trained version>`
before scoring anything.

Loading the checkpoint prints `The tokenizer you are loading ... with an
incorrect regex pattern ... set fix_mistral_regex=True`. **Do not set that
flag or change the tokenizer.** Training and scoring load the same
`tokenizer.json` under the same versions, so they tokenize identically;
changing it now would create exactly the drift this step guards against. A
0% truncation rate is not evidence about tokenization; the step-3 comparison
with the pilot predictions is. Mention the warning to the team only as
"known, left unchanged on purpose". Do not touch `artifacts/neural-mdeberta/`; copy it
aside first if you will run the optional continuation (step 6).

## 1. Checkpoint card → D (≈5 min) — by hour 0:30

```bash
python -m src.neural_card --checkpoint artifacts/neural-mdeberta \
  --train-config configs/neural/mdeberta_t4_pilot.json --output artifacts/neural-card \
  --check-pairs artifacts/neural-data/threshold_pairs_manifest.json artifacts/neural-data/final_pairs_manifest.json
```

Check before sending:
- `inputs_verified: true`. If false, the train manifests changed since
  training; tell D the exposure list is from today's files, not proven.
- `heldout_queries_seen_in_training` is 0 for threshold and final.
- `heldout_candidates_seen_in_training` is expected to be non-zero
  (background records retrieved for many queries); D decides whether that
  disqualifies a component.

Send D `artifacts/neural-card/` (JSON + one small parquet) and the
`checkpoint_id`.

## 2. Throughput benchmark → D (≈5 min) — by hour 2

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_threshold.json \
  --benchmark 10000 --max-tokens 8192,16384,32768 --project-pairs 34700000 | tee artifacts/neural-benchmark.log
```

- Put `best_max_tokens` into `max_tokens` of both `score_mdeberta_*.json`
  before the first `--execute` (changing it later is allowed; it does not
  affect resume).
- If any sweep entry shows `oom_splits > 0`, do not use that budget.
- Send D: `pairs_per_second_overlapped`, `peak_gpu_reserved_gb` of the chosen
  budget, `precision` (expect fp16), and the projection. Re-run the projection
  with C's actual routed pair count once known:
  `hours = routed_pairs / pairs_per_second_overlapped / 3600`.
- If projected hours for C's routed pairs exceed ≈ 8, tell C and D now: the
  gate must get tighter or D plans a tree-only route for part of the data.

### Measured on the T4 (checkpoint `ckpt-e315e181e5896f9c`, fp16)

| `max_tokens` | pairs/s | OOM splits | peak reserved |
|---|---|---|---|
| 8192 | 922 | 0 | 3.90 GB |
| **16384** | **1,028** | 0 | 3.90 GB |
| 32768 | 991 | 0 | 4.28 GB |

Use `max_tokens=16384` (already the config default). 32K is slower and uses
more memory. Throughput is flat across budgets and peak memory is 3.9 of
16 GB: the GPU is saturated, so larger batches will not help.

Projection for 34.7M pairs: 9.38 h overlapped, 10.96 h serial. The serial
figure is about 17% higher, meaning CPU prep on the Studio is slower than on
the laptop but still hidden behind the GPU by the shard prefetch.

### The time budget in real numbers

The 34.7M figure is the **old** plan's assumption (1,732,544 test S1 queries
× 20 candidates). The new plan routes about the top S1 candidates per S2/S3
record, so the pair count scales with test S2 + S3 records:

| Test file | Records |
|---|---|
| S2 | 4,887,273 |
| S3 | 5,082,316 |
| **S2 + S3** | **9,969,589** |

Upper bounds at 1,028 pairs/s (fewer when a record has fewer candidates, more
if C widens near-ties):

| Gate | Pairs (max) | Hours |
|---|---|---|
| top-1 per record | 10.0M | 2.7 |
| top-2 per record | 19.9M | 5.4 |
| **top-3 per record** | **29.9M** | **8.1** |
| old 34.7M assumption | 34.7M | 9.4 |

A plain top-3 gate sits right at the 8-hour limit. Tell C and D now: to get
margin, some routes must skip the neural model, either confident tree-only
rejects (especially 3rd-ranked candidates with a large score gap) or top-2
plus top-3 only where scores are close. The real budget is
`routed_pairs / 1028 / 3600` hours once C reports the routed count.

### If the budget is still too tight

- A second GPU halves the time: run a second scorer over half of C's shards
  (split the `pairs` glob) with **its own `output_dir`**. Two scorers must
  never share one output dir. D reads both manifests; both carry the same
  `checkpoint_id`.
- Do not lower `max_length` (nothing is truncated; bucketing already skips
  padding) and do not raise `max_tokens` (the GPU is saturated).

## 3. Threshold logits → D (≈ 16–20 min at 1,028 pairs/s) — do now

The 8-hour gate applies only to production scoring of C's routed pairs
(step 5). Threshold scoring is 993k pairs, does not depend on C, and D needs
these logits to start calibration and fusion. Do not hold it back.

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_threshold.json --execute | tee artifacts/score-threshold.log
```

Progress: `ls artifacts/neural-scores/threshold/*.neural.parquet | wc -l` (21
shards). Rerun the same command after any interruption; finished shards are
kept.

When done, confirm it reproduces the pilot scorer (if the pilot predictions
exist on this machine):

```bash
python - <<'EOF'
import glob, numpy as np, pandas as pd
key = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
new = pd.concat(map(pd.read_parquet, sorted(glob.glob("artifacts/neural-scores/threshold/*.neural.parquet"))))
old = pd.concat(map(pd.read_parquet, sorted(glob.glob("artifacts/predictions-mdeberta-threshold/scores-*.parquet"))))
m = new.merge(old, on=key, validate="1:1")
p = 1 / (1 + np.exp(-m["neural_logit"].astype("float64")))
d = (p - m["score"]).abs()
print(f"rows new={len(new)} old={len(old)} joined={len(m)} unscored={(~new.neural_scored).sum()}")
print(f"|dp| max={d.max():.4g} p99={d.quantile(.99):.4g}; flips at 0.5: {((p > .5) != (m.score > .5)).sum()}")
EOF
```

Expect joined = new = old, unscored = 0, and tiny differences (fp16 with
different batch composition). A large difference means a format mismatch:
stop and investigate before sending anything.

Send D `artifacts/neural-scores/threshold/` (keys + logits; no labels) and
the checkpoint ID. D fits calibration/fusion on it only within the clean
components defined by the card.

## 3b. B's starter package: key-only pipeline rehearsal (≈1 min) — right after step 3

B's starter (`member-b-starter-v1`) is in B's handoff format: key-only pair
shards plus separate `*_records.parquet` text. Its query sets are identical to
the checkpoint's query manifest (sha `789fc4ff5016…`), and on the laptop its
25,428 threshold-sample pairs joined through `threshold_records.parquet`
reproduced the training text exactly (0 mismatches in all six fields).

Put the folder at `artifacts/member-b-starter-v1/`, then:

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_b_starter.json --execute | tee artifacts/score-b-starter.log
```

Expect `cached: 25428`, `model_pairs: 0`, `missing_record: 0`: every pair's
key, text and checkpoint hash-matches the threshold scores from step 3, which
proves the key-only + records path feeds the model byte-identical input on
this machine. Anything scored or missing here means a text/format mismatch;
stop and compare before production. Do not open `evaluation_truth.tsv` or
`train_labels.parquet`; they are D's and C's.

## 3c. Test-set source store (CPU, run in a second tmux window now)

B confirmed no SQLite source store exists. Production route shards will be
key-only test pairs, so build one from the test TSVs (≈11.7M records; CPU
only, resumable, runs alongside GPU scoring):

```bash
python -m src.build_source_store --config configs/neural/source_store_test.json --dry-run
python -m src.build_source_store --config configs/neural/source_store_test.json --execute | tee artifacts/source-store-test.log
```

`score_mdeberta_routes.json` already points at `artifacts/source-store-test.sqlite`.
If B instead publishes test `*_records.parquet` files, replace `source_store`
with `"records": [...]` (the records path holds everything in RAM; fine for
samples, check free memory before loading all 11.7M test records).

## 3d. C's threshold gate v1: first C → A → D pass (≈1 min if step 3 is done)

C sent `threshold_gate_v1.parquet` (62 MB, sha256 `f99ad4e3…`): one row per
historical threshold pair (993,174; keys identical to the threshold pair
shards, no duplicates), key-only (no text), 67 columns, with `route` ∈
{`neural`: 963,843, `discard`: 29,331}. There is no `tree_only` route yet.

What the gate does: plain **top-3 S1 candidates per S2/S3 record**
(`gate_rank ≤ 3` → `neural`, ranks 4+ → `discard`). On this sample it keeps
97% because the sample is S1-grouped: each S2/S3 record has only about 1.2
competing S1s here. So this file says **nothing about production volume**;
in production top-3 is still the ~29.9M-pair, ~8.1 h case from step 2.

Text for these key-only train-split pairs comes from the historical threshold
shards themselves, so it is exactly the model's training-format text (checked
on the laptop: 963,843 routed pairs, 0 missing, 0 text mismatches):

**Put C's file in `artifacts/c-threshold-gate-v1/`, not `artifacts/tree-routes/`.**
`tree-routes/test-v1/` is the production glob of `score_mdeberta_routes.json`,
whose text source is the **test** store. A threshold file there gets scored
as if it were a test shard and every pair comes out `missing_record`. (That
happened on the Studio: the benchmark reported `model_pairs: 0`,
`forward: 0.0` and a meaningless 71,882 pairs/s. The scorer now stops with
"have no record in the text source … wrong split" instead.)

```bash
mkdir -p artifacts/c-threshold-gate-v1
mv artifacts/tree-routes/v1/threshold_gate_v1.parquet artifacts/c-threshold-gate-v1/ 2>/dev/null || true
sha256sum artifacts/c-threshold-gate-v1/threshold_gate_v1.parquet   # f99ad4e3530c78df…
python -m src.records_from_pairs --pair-manifest artifacts/neural-data/threshold_pairs_manifest.json \
  --pair-subdir threshold_pairs --output artifacts/neural-data/threshold_records_from_pairs.parquet
python -m src.score_neural --config configs/neural/score_mdeberta_gate_threshold.json --execute | tee artifacts/score-gate-threshold.log
```

Optional CPU/GPU check of the key-only path first (must show
`model_pairs: 10000`, `status_counts: {scored: 10000}`, `bottleneck: gpu_forward`;
on the laptop the records join cost 0.22 s per 10k pairs):

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_gate_threshold.json --benchmark 10000 --project-pairs 963843
```

Studio result of that benchmark: `model_pairs: 10000`, `status_counts:
{scored: 10000}`, 972 pairs/s forward (1,028 earlier; normal run-to-run
spread), `bottleneck: gpu_forward`. CPU per 10k pairs: join 0.26 s, hash
0.19 s, tokenize 1.12 s (about 3× slower than the laptop CPU), write 0.03 s,
vs 10.3 s GPU, so the prefetch hides it. Token lengths p50 47, max 132, no
truncation, padding efficiency 97%. Scoring all 963,843 routed pairs
uncached would take about 17 min.

Expect `rows: 963843`, `cached: 963843`, `model_pairs: 0`, `missing_record: 0`
(all reused from step 3). If step 3 has not finished, this scores the routed
pairs on the GPU instead (~16 min); still correct, just not free. Send D
`artifacts/neural-scores/gate-threshold-v1/`. It holds `route=neural` rows
only; D takes `discard` rows (and later `tree_only`) from C's file.

### Send back to C and D (A does not act on these)

- **To C and D, label leak risk:** the file carries `is_injected_positive`
  (all 0 here). On training data it marks injected positives, so it is truth
  derived and must be outside the tree's feature allowlist.
- **To C and D:** the `competing_s1_*` features in this sample come from
  incomplete groups (B's starter warning: S1-grouped, not complete per S2/S3
  record). Do not trust them until B's repartitioned shards exist.
- **To C:** production route shards need a `<shard>.complete.json` sidecar
  when closed; A's route config only scores shards that have one.
- **To C:** a gate run on B's new reverse-retrieval candidates is what gives
  the routed pair count A needs for the go/no-go.
- **To D, budget lever (label-free, from this file):** most neural-routed
  pairs have a near-zero tree score. If D validates a tree-only reject below a
  threshold, top-3 production would shrink roughly as below. Whether it is
  safe needs D's recall check with labels. These fractions come from the old
  ~398-candidates-per-query retrieval; production top-3 candidates are
  stronger, so real savings will be smaller:

| tree-only reject if `tree_score <` | neural keeps | top-3 production | hours at 1,028/s |
|---|---|---|---|
| 0.0001 | 69.9% | ~20.9M | ~5.6 |
| 0.001 | 33.0% | ~9.9M | ~2.7 |
| 0.01 | 13.0% | ~3.9M | ~1.1 |
| 0.05 | 6.5% | ~1.9M | ~0.5 |

## 4. First real handoff: new candidate sample from C — by hour 6

When C publishes the first route shards for the new candidate sample:

1. C's column names are known from gate v1 (`source1_entity_id`,
   `candidate_entity_id`, `candidate_source`, `route`), which are already the
   config defaults. Set the `pairs` glob and a versioned `output_dir` (e.g.
   `artifacts/neural-scores/routes-v1`) in
   `configs/neural/score_mdeberta_routes.json`.
2. Point the text source at the right split: `artifacts/source-store-test.sqlite`
   (step 3c) for test routes; for a validation (train-split) sample use B's
   `*_records.parquet` via `records`. A wrong-split store marks every pair
   `missing_record`. If C's shards already carry the six text columns, no text
   source is needed.
3. Dry run, then execute, then check the manifest:

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_routes.json --dry-run
python -m src.score_neural --config configs/neural/score_mdeberta_routes.json --execute | tee -a artifacts/score-routes.log
python -c "import json; m=json.load(open('artifacts/neural-scores/routes-v1/neural_manifest.json')); print(m['completion'], len(m['shards']), 'shards', len(m['pending_inputs']), 'pending')"
```

Any `missing_record` or `nonfinite` counts in the log go to B (missing IDs)
or stay flagged for D (non-finite); never patch them by hand.

4. Re-benchmark on C's route shards before trusting the production
   projection. The step-2 benchmark used threshold shards that already carry
   text; key-only route shards add a source-store lookup per pair that was not
   measured:

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_routes.json \
  --benchmark 10000 --project-pairs <C's routed pair count>
```

   If `bottleneck` becomes `cpu_prepare`, tell D: the projection is then the
   serial figure, not the overlapped one.

## 5. Production scoring — hours 6–20

Once the candidate/route version is frozen, loop the scorer while C publishes:

```bash
while true; do python -m src.score_neural --config configs/neural/score_mdeberta_routes.json --execute >> artifacts/score-routes.log 2>&1; sleep 120; done
```

- Stop the loop when the manifest says `complete` and C confirms all shards
  are published.
- If C republishes a shard (e.g. B added a retrieval channel for those
  groups), the run stops with "changed since it was scored". Confirm with C,
  then add `--rescore-changed`; unchanged pairs are reused, not rescored.
- A new candidate or route version gets a new `output_dir`, with the previous
  one in `reuse_from`.
- Every ~2 hours, send D: shards done / total, rows scored, current pairs/s
  from the log, and the projected finish time.

## 6. Optional: continuation on new hard negatives — only if step 2 shows slack

Only if production scoring is projected to finish with ≥ 4 hours to spare.
Train into a new run dir, score threshold into a new output dir, and give D
both threshold score sets to compare on clean components. D decides; if it
is not clearly better, keep the existing checkpoint. **Freeze the checkpoint
by hour 10**; after that no new checkpoint enters production.

## 7. Hour 20 — stop experiments; hand over

- Final `neural_manifest.json` with `completion: complete`, one
  `checkpoint_id`, and zero unexplained `missing_record` rows.
- Share the output dir path and checksums (they are in the manifest and
  sidecars); do not copy indexes or checkpoints around.
- Tell D the exact output dir to use and that no other checkpoint ID exists in
  it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Error "… have no record in the text source … wrong split" | the text source does not hold these IDs (test store for train/threshold IDs, or B's 64-query sample records for full threshold) | use the matching config: threshold/gate files use `score_mdeberta_gate_threshold.json`; test routes use `score_mdeberta_routes.json` |
| Benchmark "none of the sampled pairs reached the model" | every sampled row was `missing_record` or cached | same as above; the benchmark has no reuse cache, so it is always the text source |
| Error "input shard changed since it was scored" | the file was replaced after scoring | confirm with C, then `--rescore-changed` |
| Error "was scored with different settings (<fields>)" | the output dir was made with another checkpoint, input format, precision, route, key mapping or text source; the error lists the fields and both values | use a new `output_dir` |
| Same error but both checkpoint IDs are equal (Studio, `artifacts/neural-scores/threshold`) | code before `cdf1f0d` recorded "no text source" differently; not a real change | fixed in the scorer: `git pull` and rerun the same command; finished shards resume |
| `peak_gpu_reserved_gb` far above `peak_gpu_allocated_gb` (Studio: 12.6 vs 2.7 GB) | PyTorch's cache holds on to memory across variable batch shapes; the real need is the allocated figure | harmless for one scorer (OOM splits batches); never share the T4 with a second GPU job, or if unavoidable start the scorer with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `ready_inputs: 0` in `--dry-run` for routes | C's shards lack `.complete.json` | ask C to publish sidecars; never hand-create them for unfinished shards |

A shard is refused, with nothing written, when more than
`max_missing_fraction` (default 0.01) of its routed pairs have no text. Raise
it in the config only for a known, reported gap, never to get past a
wrong-split store.

## Data and repo rules

- B's package, pair shards, stores, checkpoints and score outputs never go
  into Git (`member-b-starter-v1/` holds D's evaluation truth). Share paths
  and checksums instead.
- A never opens `evaluation_truth.tsv` or `train_labels.parquet`, and never
  tunes anything on `final`.
- `src/neural_error_analysis.py` (used by the older Lightning runbook's
  error-analysis step) is not on this branch yet; push it separately if needed.

## Status message template (to the team channel)

```
A status @ hour <h>
checkpoint: <checkpoint_id> (inputs_verified=<true/false>)
throughput: <pairs/s> on T4 <precision>, max_tokens=<n>, peak <GB> GB
threshold logits: <done/in progress> -> artifacts/neural-scores/threshold (<rows> rows, 0 unscored)
b-starter rehearsal: <cached>/25428 cached, <missing> missing (expect 25428 / 0)
C gate v1 (threshold): <cached>/963843 routed pairs scored -> artifacts/neural-scores/gate-threshold-v1
test source store: <building / done, N records>
routes: <version> <done>/<total> shards, <rows> rows, missing_record=<n>, nonfinite=<n>, ETA <time>
blockers: <none | what I need from B/C/D>
```

Current message (fill in the hour and the threshold ETA):

```
A status @ hour <h>
checkpoint: ckpt-e315e181e5896f9c (inputs_verified=true, 0 held-out queries seen in training)
throughput: 972-1,028 pairs/s on T4 fp16, max_tokens=16384, 0 OOM; GPU memory actually used 2.7 GB (cache may reserve up to ~12.6 GB)
threshold logits: running now -> artifacts/neural-scores/threshold, ETA ~20 min
budget: test S2+S3 = 9.97M records. Top-3/record <= 29.9M pairs = ~8.1h (at the 8h limit);
        top-2 <= 19.9M = ~5.4h. Need C's routed count; tree-only rejects on rank 3 would give margin.
        (the 34.7M / 9.4h figure is the old S1x20 assumption, not this plan)
tokenizer: "incorrect regex pattern" warning is known; left unchanged on purpose (matches training)
C gate v1: plain top-3 per S2/S3 record; 963,843 of 993,174 threshold pairs routed, keys match;
        97% kept only because this sample has ~1.2 S1s per record, so it does not size production.
        Scores for its neural rows -> artifacts/neural-scores/gate-threshold-v1 (<done/ETA>)
flags: is_injected_positive is in C's feature file (truth-derived, keep out of features);
       competing_s1_* on this sample come from incomplete groups
lever for D: tree_score < 0.001 as tree-only reject would keep ~33% of neural work (needs D's recall check)
blockers: C's routed count on B's new candidates; test source store building (step 3c)
```
