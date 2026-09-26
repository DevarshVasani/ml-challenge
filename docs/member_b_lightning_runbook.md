# Member B — mDeBERTa on a Lightning AI Studio

Pick a **24GB BF16-capable GPU: L4 or A10G**. T4 has no BF16 and the shared
trainer refuses `precision: bf16` on it; fp16 is a known overflow risk for
mDeBERTa-v3. Do setup on the free CPU machine and switch the Studio to the GPU
only for steps 4 onward.

Run every long command under `tmux` (or `nohup ... > log 2>&1 &`) so a closed
browser tab does not kill it. All outputs land under `artifacts/`, which lives
on the Studio's persistent disk.

## 1. Environment (CPU machine)

```bash
git clone <repo-url> ml-challenge && cd ml-challenge
pip install -r requirements.txt -r requirements-gpu.txt pytest
python -m pytest tests/test_mdeberta_adapter.py tests/test_neural_infrastructure.py -q
```

The Studio image already has a CUDA PyTorch; keep it if it satisfies
`torch>=2.2,<3` (`python -c "import torch; print(torch.__version__)"`).

## 2. Data from Member A (CPU machine)

Upload Member A's canonical data version unchanged into the same paths the
configs expect, plus the raw ground truth used by the evaluator:

```text
artifacts/neural-data/query_manifest.json
artifacts/neural-data/{train,threshold,final}_pairs_manifest.json
artifacts/neural-data/{train,threshold,final}_pairs/*.parquet
artifacts/neural-candidates/manifest.json      # + candidate-recall report
student_resource/dataset/train/train_ground_truth.tsv
```

Record the data version (e.g. `sha256sum artifacts/neural-data/*manifest.json`).

## 3. Pre-download the checkpoint (CPU machine)

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base'); M.from_pretrained('microsoft/mdeberta-v3-base', num_labels=2)"
```

## 4. Sanity checks — Phase 2 (switch to GPU)

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"   # both True
python scripts/mdeberta_sanity.py --synthetic --device cuda --output-dir artifacts/mdeberta-sanity-synthetic
python scripts/mdeberta_sanity.py --pairs artifacts/neural-data/train_pairs --device cuda --output-dir artifacts/mdeberta-sanity
```

Exit code 0 means every check passed. Read `sanity_report.json`:
`length_profile.truncation_rate_at["256"]` is the truncation rate at the
pilot budget; `inference` gives pairs/s and VRAM per batch size.

## 5. Short GPU pilot — Phase 3 (200 optimizer steps ≈ 12.8k pairs)

```bash
python -m src.train_neural --config configs/neural/mdeberta_short_pilot.json --execute | tee artifacts/mdeberta-short.log
```

Record `peak_gpu_memory_bytes`, `pairs_per_second`, `truncation_rate`, and
`log_history` (loss should trend down). If peak VRAM is well under 24GB, keep
the settings; only if it runs out of memory set `"gradient_checkpointing": true`.

## 6. Full pilot — Phase 4

```bash
python -m src.train_neural --config configs/neural/mdeberta_pilot.json --execute | tee artifacts/mdeberta-pilot.log
```

A checkpoint is written every 500 steps to `artifacts/neural-mdeberta/`. If
the Studio stops, resume with the identical config plus a `resume` key:

```bash
python - <<'EOF'
import json; c = json.load(open("configs/neural/mdeberta_pilot.json"))
c["resume"] = "artifacts/neural-mdeberta"
json.dump(c, open("artifacts/mdeberta_pilot_resume.json", "w"), indent=2)
EOF
python -m src.train_neural --config artifacts/mdeberta_pilot_resume.json --execute | tee -a artifacts/mdeberta-pilot.log
```

## 7. Score threshold and final pairs — Phase 5/6

Set `measured_pairs_per_second` in the predict configs from step 4 to get a
time estimate from `--dry-run`, and raise `batch_size` to the best measured
value.

```bash
python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --dry-run
python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --execute
python -m src.predict_neural --config configs/neural/predict_mdeberta_pilot.json --execute
```

Both are resumable: rerun the same command after an interruption.

## 8. Tune threshold on `threshold`, evaluate `final` once

```bash
python -m src.evaluate_neural --config configs/neural/evaluate_pilot.json --execute
```

This picks the threshold only from the threshold subset, freezes it, then
scores final once. `artifacts/neural-evaluation-mdeberta.json` holds macro-F0.5,
`candidate_oracle_ceiling`, `by_country` (US, India, …) and `by_match_count`
(singleton vs non-singleton). Do not rerun with a different threshold grid
after looking at the final numbers.

## 9. Bring back

Download `artifacts/neural-evaluation-mdeberta.json`, the sanity report, both
training logs, and keep `artifacts/neural-mdeberta/` (checkpoint, about 3.3GB
with optimizer state) on the Studio or copy it to shared storage.
