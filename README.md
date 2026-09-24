# Entity-resolution matching pipeline

## Setup

Use Python 3.10+ and install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The expected layout is `student_resource/dataset/train/` containing
`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and
`train_ground_truth.tsv`, plus the corresponding three test source files under
`student_resource/dataset/test/`.

## Full run

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

## Individual stages

```bash
python -m src.split --data-dir student_resource/dataset/train --out-folds artifacts/folds.tsv --verify
python -m src.candidates --s1 student_resource/dataset/train/train_source1.tsv --s2 student_resource/dataset/train/train_source2.tsv --s3 student_resource/dataset/train/train_source3.tsv --gt student_resource/dataset/train/train_ground_truth.tsv --output artifacts/train_candidates.tsv --report artifacts/train_candidates.json
python -m src.features --s1 student_resource/dataset/train/train_source1.tsv --s2 student_resource/dataset/train/train_source2.tsv --s3 student_resource/dataset/train/train_source3.tsv --candidates artifacts/train_candidates.tsv --ground-truth student_resource/dataset/train/train_ground_truth.tsv --folds artifacts/folds.tsv --output artifacts/train_features.parquet
python -m src.train_pair_model --features artifacts/train_features.parquet --ground-truth student_resource/dataset/train/train_ground_truth.tsv --output-dir artifacts/model
python -m src.predict_pair_model --features artifacts/test_features.parquet --model-dir artifacts/model --output-dir artifacts/test_scores --s1-ids student_resource/dataset/test/test_source1.tsv
```

The internal candidate table is long format and contains the exact pairs and
retrieval diagnostics scored by the model. The official `candidate_pairs.tsv`
is grouped to one row per test S1 entity for submission.

Folds keep known positive connected components together. During OOF training,
validation rows are selected by `source1_fold`; training rows require both
`source1_fold != validation_fold` and `candidate_fold != validation_fold`.
Cross-fold negatives are retained, while cross-fold positives fail validation.

## Artifacts and reproducibility

The model directory contains `final_model.joblib`, `feature_columns.json`,
`training_config.json`, OOF predictions in Parquet and TSV, the selected
threshold, and `metrics.json`. Test inference writes pair scores and
`predictions.json`; the pipeline writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv` and validates both files.

Run `python -m pytest -q` for tests. Candidate retrieval is deterministic for
fixed input and CLI settings. No external identity data, geocoding,
translation, business lookup, or identity-resolution API is used.

## Known limitations

TF-IDF retrieval is lexical and may miss entities with no shared character or
word evidence. The baseline pair model is intentionally small and does not
perform external enrichment. Full-size datasets can require substantial RAM,
particularly when reading source tables and validating IDs.
