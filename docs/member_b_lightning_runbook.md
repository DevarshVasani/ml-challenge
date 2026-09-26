# Member B — mDeBERTa on a Lightning AI Studio

This runbook targets a **T4 (16GB)**. The T4 has no native BF16, so its
configs train with fp16 mixed precision plus a gradient scaler
(`configs/neural/mdeberta_t4_*.json`), and the adapter runs fp16 inference on
it automatically. On an L4/A10G use the non-`t4` configs (BF16) instead.
Everything else, including the predict/evaluate configs, is shared.

fp16 has a narrower range than BF16 and DeBERTa-v3 can overflow in it. The
trainer stops on a non-finite loss and prediction refuses non-finite scores,
so a problem fails loudly instead of producing bad numbers; the fallbacks are
in steps 5 and 7.

Do steps 1–3 on the free CPU machine and switch the Studio to the T4 for step
4 onward. Run every long command under `tmux` (or `nohup ... > log 2>&1 &`)
so a closed browser tab does not kill it. All outputs land under
`artifacts/`, which lives on the Studio's persistent disk.

## Overview

| Step | Phase | Machine | Output |
|---|---|---|---|
| 1–3 | setup, data check, model download | CPU | verified data, cached checkpoint |
| 4 | 2 — sanity checks | T4 | `artifacts/mdeberta-sanity*/sanity_report.json` |
| 5 | 3 — short pilot (200 steps) | T4 | VRAM, pairs/s, truncation, fp16 health |
| 6 | 4 — full pilot (≈10.5k steps) | T4 | `artifacts/neural-mdeberta/` |
| 7 | 5/6 — score threshold + final | T4 | `artifacts/predictions-mdeberta-{threshold,final}/` |
| 8 | 5/6 — tune on threshold, evaluate final once | CPU or T4 | `artifacts/neural-evaluation-mdeberta.json` |
| 9 | 7 — error analysis on threshold | CPU or T4 | `artifacts/mdeberta-errors-threshold/` |
| 10 | handoff | — | MDEBERTA BASELINE COMPLETE block |

## Data version

Member A's canonical pair data (all built from the same `query_manifest.json`,
sha256 `789fc4ff5016…`):

| Split | Queries | Pairs | Shards | Used for |
|---|---|---|---|---|
| train | 50,000 | 672,948 | 391 | training (≈10.5k optimizer steps at effective batch 64) |
| bootstrap | 896 | 12,021 | 7 (subset of train shards) | optional quick checks |
| threshold | 2,500 | 993,174 | 21 | threshold tuning and error analysis |
| final | 2,500 (1,462 US, 1,038 India, 147 singletons) | 993,198 | 20 | one final evaluation |

Candidate recall on the threshold split is 97.1% (8,387 of 8,639 true match
pairs retrieved), so about 3% of true matches are out of reach of any matcher.

Shards are named `pairs-<first>-<last>.parquet` and manifests list them
relative to `artifacts/neural-data/` (e.g. `train_pairs/pairs-…parquet`); the
shared loaders resolve that form. `artifacts/neural-candidates/` is **not**
needed: the pair manifests already record its hash, and the evaluator computes
the candidate-oracle ceiling from the pair shards.

## 1. Code and environment (CPU machine)

The T4 configs, the adapter precision change and `src/neural_error_analysis.py`
must be pushed before cloning.

```bash
cd /teamspace/studios/this_studio
git clone <repo-url> ml-challenge && cd ml-challenge
pip install -r requirements.txt -r requirements-gpu.txt pytest
python -m pytest tests/test_mdeberta_adapter.py tests/test_neural_infrastructure.py -q
```

The Studio image already has a CUDA PyTorch; keep it if it satisfies
`torch>=2.2,<3` (`python -c "import torch; print(torch.__version__)"`).

## 2. Data (CPU machine)

Upload `lightning-upload.zip` (211MB) into the `ml-challenge/` folder, then:

```bash
unzip -q lightning-upload.zip && rm lightning-upload.zip
python - <<'EOF'
import json
from pathlib import Path
from src.neural_contracts import manifest_shard_paths, sha256_file
root = Path("artifacts/neural-data")
for split in ("train", "threshold", "final"):
    manifest_path = root / f"{split}_pairs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = manifest_shard_paths(manifest_path, manifest["shard_order"], f"{split}_pairs")
    bad = [p for p in paths if not Path(p).is_file() or sha256_file(p) != manifest["generated_file_checksums"][str(Path(p).relative_to(root))]]
    print(split, len(paths), "shards,", sum(manifest["row_counts"].values()), "pairs, bad:", len(bad))
assert Path("student_resource/dataset/train/train_ground_truth.tsv").is_file()
print("query manifest sha256:", sha256_file(root / "query_manifest.json")[:12])
EOF
```

Expect `bad: 0` for every split and a query-manifest hash starting `789fc4ff5016`.

## 3. Pre-download the checkpoint (CPU machine)

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base'); M.from_pretrained('microsoft/mdeberta-v3-base', num_labels=2)"
```

## 4. Sanity checks — Phase 2 (switch to T4)

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"   # True Tesla T4 (7, 5)
python scripts/mdeberta_sanity.py --synthetic --device cuda --output-dir artifacts/mdeberta-sanity-synthetic
python scripts/mdeberta_sanity.py --pairs artifacts/neural-data/train_pairs --device cuda --output-dir artifacts/mdeberta-sanity
python scripts/mdeberta_sanity.py --pairs artifacts/neural-data/final_pairs --device cuda --output-dir artifacts/mdeberta-sanity-final
```

Exit code 0 means every check passed; each report should say
`"precision": "fp16"` and `"inference_precision": "fp16"`. In each
`sanity_report.json`, `length_profile.truncation_rate_at["256"]` is the
truncation rate at the pilot budget and `inference` gives pairs/s and VRAM per
batch size. The final-split run sets the inference batch size and throughput
for step 7 (only its text lengths and speed are used, never its labels for
decisions).

If a sanity run fails `loss_decreases` or stops with a non-finite loss, rerun
it with `--precision fp32` to confirm fp16 is the cause before step 5.

## 5. Short GPU pilot — Phase 3 (200 optimizer steps ≈ 12.8k pairs)

```bash
python -m src.train_neural --config configs/neural/mdeberta_t4_short_pilot.json --execute | tee artifacts/mdeberta-short.log
```

Record `peak_gpu_memory_bytes`, `pairs_per_second`, `truncation_rate`, and
`log_history` (loss should trend down). These are the clean numbers for the
handoff. Full-pilot time ≈ 672,948 / `pairs_per_second`. If peak VRAM is well
under 16GB keep the settings; only if it runs out of memory set
`"gradient_checkpointing": true`.

If it stops with `non-finite neural loss`, fp16 overflowed: delete
`artifacts/neural-mdeberta-short/`, set `"precision": "fp32"` in both T4
configs, and rerun. fp32 fits on a T4 at microbatch 8 but is several times
slower, so re-estimate the full-pilot time from the new `pairs_per_second`.

## 6. Full pilot — Phase 4

```bash
python -m src.train_neural --config configs/neural/mdeberta_t4_pilot.json --execute | tee artifacts/mdeberta-pilot.log
```

The trainer prints only when it finishes. Watch progress from a second
terminal; `step` advances every 250 optimizer steps, out of ≈10.5k:

```bash
watch -n 120 'grep -E "\"step\"" artifacts/neural-mdeberta/checkpoint.json; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

If the Studio stops, resume with the identical config plus a `resume` key
(changing any training setting makes resume refuse):

```bash
python - <<'EOF'
import json; c = json.load(open("configs/neural/mdeberta_t4_pilot.json"))
c["resume"] = "artifacts/neural-mdeberta"
json.dump(c, open("artifacts/mdeberta_pilot_resume.json", "w"), indent=2)
EOF
python -m src.train_neural --config artifacts/mdeberta_pilot_resume.json --execute | tee -a artifacts/mdeberta-pilot.log
```

After a resume, the printed throughput and VRAM cover only the resumed part;
use the step-5 numbers in the handoff.

## 7. Score threshold and final pairs — Phase 5/6

About 993k pairs per split. Before the first `--execute`, put the best batch
size from the final-split sanity report into `batch_size`, and its pairs/s into
`measured_pairs_per_second`, in both predict configs; `--dry-run` then prints
the estimated seconds. Do not edit a predict config after its first execute.

```bash
python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --dry-run
python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --execute | tee artifacts/predict-threshold.log
python -m src.predict_neural --config configs/neural/predict_mdeberta_pilot.json --execute | tee artifacts/predict-final.log
```

Progress: `ls artifacts/predictions-mdeberta-threshold/scores-*.parquet | wc -l`
(21 shards for threshold, 20 for final). Both resume per shard: rerun the same
command after an interruption. The printed `pairs_per_second_this_run` and
`truncated_pairs_this_run / rows` of an uninterrupted run give inference
throughput and truncation rate for the handoff.

If prediction stops with a "finite" score error, fp16 inference overflowed.
Delete that prediction output directory (so fp16 and fp32 shards never mix)
and rerun with fp32 inference forced, for both threshold and final:

```bash
MDEBERTA_INFERENCE_PRECISION=fp32 python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --execute
```

## 8. Tune threshold on `threshold`, evaluate `final` once

```bash
python -m src.evaluate_neural --config configs/neural/evaluate_pilot.json --execute | tee artifacts/evaluate.log
```

This picks the threshold only from the threshold subset, freezes it, then
scores final once. The grid runs 0.05–0.95 in steps of 0.05 plus 0.97, 0.98
and 0.99: threshold/final pairs are about 99% negatives while training pairs
are about 24% positive, so the best F0.5 threshold may sit high. Share this
grid with Member C so both models are tuned the same way. If the selected
threshold is 0.99 (the grid edge), note it in the handoff rather than
re-tuning after seeing final.

`artifacts/neural-evaluation-mdeberta.json`:

- `threshold_tuning.threshold` — the frozen threshold
- `final_comparison.macro_f05` — headline metric
- `final_comparison.singleton_accuracy`, `final_comparison.non_singleton_macro_f05`
- `final_comparison.by_country.US.macro_f05`, `final_comparison.by_country.India.macro_f05`
- `final_comparison.candidate_oracle_ceiling` — best possible macro-F0.5 given retrieval

Do not rerun with a different grid after looking at the final numbers.

## 9. Error analysis — Phase 7 (threshold split)

```bash
python -m src.neural_error_analysis --subset threshold \
    --predictions artifacts/predictions-mdeberta-threshold \
    --pairs artifacts/neural-data/threshold_pairs \
    --evaluation artifacts/neural-evaluation-mdeberta.json \
    --output-dir artifacts/mdeberta-errors-threshold
```

It uses the frozen threshold and writes `summary.json` plus the top 50 rows of
each bucket as TSV: `false_positives` (highest-scoring wrong matches),
`false_negatives` (true matches scored below threshold), `singleton_false_positives`,
and `retrieval_misses` (true matches never retrieved). Each error row carries
flags: `same_name`, `number_conflict` (addresses with different numbers),
`same_address`, `missing_name`, `missing_address`, `non_latin_name`. Read
about 20 rows per bucket and decide where the next gain is:

| Dominant pattern | Next experiment belongs in |
|---|---|
| many `retrieval_misses` vs matcher errors | retrieval (Member A) |
| false positives with `same_name` + `number_conflict` | pair serialization / training signal on numbers |
| errors concentrated in truncated long addresses | sequence length (raise `max_length`) |
| scores well separated but threshold at the grid edge | thresholding / calibration |
| broad errors across buckets | model training (epochs, learning rate) |

## 10. Handoff

| Field | Source |
|---|---|
| Data version | query manifest sha256 `789fc4ff5016…` + `sha256sum artifacts/neural-data/*_pairs_manifest.json` |
| Checkpoint | `artifacts/neural-mdeberta/` (`checkpoint.json` step, `adapter/adapter_config.json` provenance) |
| Training config | `configs/neural/mdeberta_t4_pilot.json` + git commit |
| Threshold selected | step 8 `threshold_tuning.threshold` |
| Final macro-F0.5, singleton, non-singleton, US, India | step 8 keys above |
| Peak VRAM | step 5 `peak_gpu_memory_bytes` (training), sanity report `inference.*.peak_vram_gb` |
| Inference pairs/sec | step 7 `pairs_per_second_this_run` |
| Truncation rate | step 5 `truncation_rate`; step 7 `truncated_pairs_this_run / rows` |
| Failure modes / next experiment | step 9 |

Download `artifacts/neural-evaluation-mdeberta.json`,
`artifacts/mdeberta-errors-threshold/`, the sanity reports and all `.log`
files. Keep `artifacts/neural-mdeberta/` (checkpoint, about 3.3GB with
optimizer state) on the Studio or copy it to shared storage.
