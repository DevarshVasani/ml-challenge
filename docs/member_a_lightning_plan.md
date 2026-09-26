# Member A — Lightning machine plan

Run steps in order on the T4 Studio. Commands and the output contract are in
`docs/member_a_runbook.md`; this is the sequence, the checks, and what to send
whom. Run everything long inside `tmux` so a closed tab does not kill it.

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
before scoring anything. Do not touch `artifacts/neural-mdeberta/`; copy it
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

## 3. Threshold logits → D (≈ 993k / measured pairs/s) — start by hour 2

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

## 4. First real handoff: new candidate sample from C — by hour 6

When C publishes the first route shards for the new candidate sample:

1. Agree C's column names; set `key_columns`, the `pairs` glob and a versioned
   `output_dir` (e.g. `artifacts/neural-scores/routes-v1`) in
   `configs/neural/score_mdeberta_routes.json`.
2. Get the source store for those records from B. The config expects
   `artifacts/source-store.sqlite`; for test-set routes it must be a store built
   from the **test** S1/S2/S3 files — the train store will mark every pair
   `missing_record`. If C's shards already carry the six text columns the
   store is not needed.
3. Dry run, then execute, then check the manifest:

```bash
python -m src.score_neural --config configs/neural/score_mdeberta_routes.json --dry-run
python -m src.score_neural --config configs/neural/score_mdeberta_routes.json --execute | tee -a artifacts/score-routes.log
python -c "import json; m=json.load(open('artifacts/neural-scores/routes-v1/neural_manifest.json')); print(m['completion'], len(m['shards']), 'shards', len(m['pending_inputs']), 'pending')"
```

Any `missing_record` or `nonfinite` counts in the log go to B (missing IDs)
or stay flagged for D (non-finite); never patch them by hand.

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

## Status message template (to the team channel)

```
A status @ hour <h>
checkpoint: <checkpoint_id> (inputs_verified=<true/false>)
throughput: <pairs/s> on T4 <precision>, max_tokens=<n>, peak <GB> GB
threshold logits: <done/in progress> -> artifacts/neural-scores/threshold (<rows> rows, 0 unscored)
routes: <version> <done>/<total> shards, <rows> rows, missing_record=<n>, nonfinite=<n>, ETA <time>
blockers: <none | what I need from B/C/D>
```
