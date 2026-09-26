# Member B — mDeBERTa on a Lightning AI Studio

Pick a **24GB BF16-capable GPU: L4 or A10G**. T4 has no BF16 and the shared
trainer refuses `precision: bf16` on it; fp16 is a known overflow risk for
mDeBERTa-v3. Do steps 1–3 on the free CPU machine and switch the Studio to
the GPU for step 4 onward.

Run every long command under `tmux` (or `nohup ... > log 2>&1 &`) so a closed
browser tab does not kill it. All outputs land under `artifacts/`, which lives
on the Studio's persistent disk.

## Data version

Member A's canonical pair data (all built from the same `query_manifest.json`,
sha256 `789fc4ff5016…`):

| Split | Queries | Pairs | Shards | Used for |
|---|---|---|---|---|
| train | 50,000 | 672,948 | 391 | training (≈10.5k optimizer steps at effective batch 64) |
| bootstrap | 896 | 12,021 | 7 (subset of train shards) | optional quick checks |
| threshold | 2,500 | 993,174 | 21 | threshold tuning only |
| final | 2,500 (1,462 US, 1,038 India, 147 singletons) | 993,198 | 20 | one final evaluation |

Shards are named `pairs-<first>-<last>.parquet` and manifests list them
relative to `artifacts/neural-data/` (e.g. `train_pairs/pairs-…parquet`); the
shared loaders resolve that form. `artifacts/neural-candidates/` is **not**
needed: the pair manifests already record its hash, and the evaluator computes
the candidate-oracle ceiling from the pair shards.

## 1. Code and environment (CPU machine)

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

## 4. Sanity checks — Phase 2 (switch to GPU)

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"   # both True
python scripts/mdeberta_sanity.py --synthetic --device cuda --output-dir artifacts/mdeberta-sanity-synthetic
python scripts/mdeberta_sanity.py --pairs artifacts/neural-data/train_pairs --device cuda --output-dir artifacts/mdeberta-sanity
python scripts/mdeberta_sanity.py --pairs artifacts/neural-data/final_pairs --device cuda --output-dir artifacts/mdeberta-sanity-final
```

Exit code 0 means every check passed. In each `sanity_report.json`,
`length_profile.truncation_rate_at["256"]` is the truncation rate at the pilot
budget and `inference` gives pairs/s and VRAM per batch size. The final-split
run is the one that matters for inference throughput, since threshold/final
carry about 400 candidates per query.

## 5. Short GPU pilot — Phase 3 (200 optimizer steps ≈ 12.8k pairs)

```bash
python -m src.train_neural --config configs/neural/mdeberta_short_pilot.json --execute | tee artifacts/mdeberta-short.log
```

Record `peak_gpu_memory_bytes`, `pairs_per_second`, `truncation_rate`, and
`log_history` (loss should trend down). Full-pilot time ≈ 672,948 /
`pairs_per_second`. If peak VRAM is well under 24GB keep the settings; only if
it runs out of memory set `"gradient_checkpointing": true`.

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

About 993k pairs per split. Put the best batch size from the final-split
sanity report into `batch_size`, and its pairs/s into
`measured_pairs_per_second`, in both predict configs; `--dry-run` then prints
the estimated seconds.

```bash
python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --dry-run
python -m src.predict_neural --config configs/neural/predict_mdeberta_threshold_pilot.json --execute
python -m src.predict_neural --config configs/neural/predict_mdeberta_pilot.json --execute
```

Both are resumable per shard: rerun the same command after an interruption.
Do not edit a predict config between an interrupted run and its resume.

## 8. Tune threshold on `threshold`, evaluate `final` once

```bash
python -m src.evaluate_neural --config configs/neural/evaluate_pilot.json --execute
```

This picks the threshold only from the threshold subset, freezes it, then
scores final once. `artifacts/neural-evaluation-mdeberta.json` holds
macro-F0.5, `candidate_oracle_ceiling`, `by_country` (`US`, `India`) and
`by_match_count` (key `"0"` is singletons; non-singleton is the entity-weighted
mean of the other keys). Do not rerun with a different threshold grid after
looking at the final numbers.

## 9. Bring back

Download `artifacts/neural-evaluation-mdeberta.json`, the sanity reports, and
both training logs. Keep `artifacts/neural-mdeberta/` (checkpoint, about 3.3GB
with optimizer state) on the Studio or copy it to shared storage.
