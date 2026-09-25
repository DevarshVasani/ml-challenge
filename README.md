# Amazon ML Challenge — Business Entity Resolution

Robust, production-grade infrastructure for Business Entity Resolution across multi-source commercial datasets.

---

## 🛠️ Setup & Environment

Use Python 3.10+ and install the project dependencies:

```bash
# Option 1: Standard pip / venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Option 2: Conda / Mamba
conda env create -f environment.yml
conda activate er-challenge
```

The expected data layout is `student_resource/dataset/train/` containing
`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and
`train_ground_truth.tsv`, plus the corresponding test source files under
`student_resource/dataset/test/`.

---

## 🚀 Full Pipeline Run

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

Intermediate folds, candidates, features, model artifacts, OOF predictions,
and test scores are cached in `artifacts/`. Add `--force` to recompute them.

---

## 🧩 Pipeline Stages & Modules

### 1. Validation Splitting (`src/split.py`)
Creates fixed 3-fold cross-validation splits balancing match counts and countries (`US`, `India`).
- **Group Preservation**: Disjoint Set Union (DSU) groups each $S_1$ entity and all its transitively matched records ($S_2/S_3$) into connected components. If multiple $S_1$ entities share a match, they stay in the same fold.
- **Strict Leakage Prevention**: Held-out validation records are strictly isolated and prevented from entering training candidate pairs or negative sampling sets.

```bash
python -m src.split \
  --data-dir student_resource/dataset/train \
  --out-folds artifacts/folds.tsv \
  --out-summary reports/split_summary.json \
  --verify
```

### 2. Candidate Generation / Blocking (`src/candidates.py`)
```bash
python -m src.candidates \
  --s1 student_resource/dataset/train/train_source1.tsv \
  --s2 student_resource/dataset/train/train_source2.tsv \
  --s3 student_resource/dataset/train/train_source3.tsv \
  --gt student_resource/dataset/train/train_ground_truth.tsv \
  --output artifacts/train_candidates.tsv \
  --report artifacts/train_candidates.json
```

### 3. Feature Engineering (`src/features.py`)
```bash
python -m src.features \
  --s1 student_resource/dataset/train/train_source1.tsv \
  --s2 student_resource/dataset/train/train_source2.tsv \
  --s3 student_resource/dataset/train/train_source3.tsv \
  --candidates artifacts/train_candidates.tsv \
  --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
  --folds artifacts/folds.tsv \
  --output artifacts/train_features.parquet
```

### 4. Model Training & Evaluation (`src/train_pair_model.py`, `src/evaluate.py`)
```bash
python -m src.train_pair_model \
  --features artifacts/train_features.parquet \
  --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
  --output-dir artifacts/model
```

### 5. Prediction & Submission Export (`src/predict_pair_model.py`, `src/export.py`)
```bash
python -m src.predict_pair_model \
  --features artifacts/test_features.parquet \
  --model-dir artifacts/model \
  --output-dir artifacts/test_scores \
  --s1-ids student_resource/dataset/test/test_source1.tsv
```

Official output files written to `output/`:
- `output/matching_results.tsv` (`source1_entity_id\tmatched_entity_ids`)
- `output/candidate_pairs.tsv` (`source1_entity_id\tcandidate_entity_ids`)

---

## 📊 Evaluation Metric (`src/evaluate.py`)

Submissions are evaluated on a **macro-averaged $F_{0.5}$ score** across all Source 1 ($S_1$) entities.

### Metric Formula:
For non-singletons (entities with $\ge 1$ true matches):
$$F_{0.5} = \frac{5 \cdot \text{TP}}{5 \cdot \text{TP} + 4 \cdot \text{FP} + \text{FN}}$$
Where:
- $\text{TP}$: Correct predicted matches ($|\text{True} \cap \text{Pred}|$)
- $\text{FP}$: Incorrect predicted matches ($|\text{Pred} \setminus \text{True}|$)
- $\text{FN}$: Missed true matches ($|\text{True} \setminus \text{Pred}|$)

For singletons (entities with no true matches):
- Score $= 1.0$ if the prediction is empty ($\emptyset$).
- Score $= 0.0$ if any match is predicted (false merge).

### Tiny Example Verifications:
| Case | Ground Truth | Prediction | TP | FP | FN | Score |
|---|---|---|---|---|---|---|
| **Perfect Match** | `[S2-1, S3-2]` | `[S2-1, S3-2]` | 2 | 0 | 0 | **1.0000** |
| **Extra Match** | `[S2-1]` | `[S2-1, S2-2]` | 1 | 1 | 0 | **5/9 ≈ 0.5556** |
| **Missed Match** | `[S2-1, S3-2]` | `[S2-1]` | 1 | 0 | 1 | **5/6 ≈ 0.8333** |
| **Empty Prediction (Non-Singleton)** | `[S2-1]` | `[]` | 0 | 0 | 1 | **0.0000** |
| **Singleton (Empty Prediction)** | `[]` | `[]` | - | - | - | **1.0000** |
| **Singleton (False Merge)** | `[]` | `[S2-1]` | - | - | - | **0.0000** |

Run metric self-check:
```bash
python3 src/evaluate.py --check
```

## Neural baseline infrastructure

Shared, label-safe pair contracts, reusable candidate indexes, grouped Parquet
shards, SQLite source joins, fake-adapter smoke training, resumable scoring,
threshold evaluation, and streaming export are provided for the future GPU
baselines. Start with the [GPU runbook](docs/neural_gpu_runbook.md) and the
[adapter contract](docs/neural_adapter_contract.md). All new expensive
commands require `--execute`; `--dry-run` and `--help` do not load models or
iterate the competition dataset.

---

## 🧪 Testing

Run test suite:
```bash
python -m pytest -q
# or
python3 -m unittest discover tests
```
# Experiment log

`reports/experiments.csv` is a scaffold for measured runs. Values unavailable
because a run has not been performed must be recorded as `N/A`; they must not
be replaced with fabricated zeros.
