# Phase A — Real-Data Baseline Report

_Generated: 2026-09-25 14:01 UTC_

## 1. Run Provenance

| Field | Value |
|-------|-------|
| Commit SHA | `d094342c111302c44019ad0168ccc65ef9e1c058` |
| Commit Message | updating branch with latest changes |
| Work directory | `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2` |
| Output directory | `/home/daksh/Amazon_ML_challenge/ml-challenge/output/h04_baseline_v2` |
| Command | `python -m src.pipeline --data-dir data/dataset --work-dir artifacts/h04_baseline_v2 --output-dir output/h04_baseline_v2 --n-splits 3 --seed 42 --top-k-name 50 --top-k-address 50 --batch-size 256` |

## 2. Data Audit Summary

| Check | Result |
|-------|--------|
| Duplicate IDs | ✅ None across all 6 datasets |
| Missing files/columns | ✅ All files present, all required columns found |
| Invalid GT references (orphan S2/S3) | ✅ 0 orphan S2 refs, 0 orphan S3 refs |
| Empty data | ✅ No empty datasets detected |
| Total audit errors | 1 |
| Total audit warnings | 11 |
| Overall audit status | ❌ 1 error(s) |

### Data Errors (blocking)

- **[ERROR]** [submission_validation] Validator not found: /home/devarsh/projects/amazon-ml-challenge/ml-challenge/student_resource/utils/validate_submission.py


### Data Warnings (advisory)

- **[WARN]** [match_cardinality] 1964417 S1 entities have multiple matches.
- **[WARN]** [source1] Country only in test: ['France']
- **[WARN]** [source2] Country only in test: ['France']
- **[WARN]** [source2] 'business_address' non-ASCII shift: 9.5% -> 14.7%
- **[WARN]** [source3] Country only in test: ['France']
- **[WARN]** [source3] 'business_address' non-ASCII shift: 9.0% -> 14.3%
- **[WARN]** [france] 197782 branch ambiguities.


### Submission Writer Scan (advisory — not blocking)

> The source-code scanner warns about files that are not submission writers.
> These are advisory only — `__init__.py`, `metrics.py`, `pipeline.py`, and `text_utils.py`
> do not write submission TSVs themselves; this is expected.

- **[WARN]** [submission_writer] __init__.py: Missing expected output columns.
- **[WARN]** [submission_writer] metrics.py: Missing expected output columns.
- **[WARN]** [submission_writer] pipeline.py: Missing expected output columns.
- **[WARN]** [submission_writer] text_utils.py: Missing expected output columns.


## 3. Fold Splitter Verification

| Property | Status |
|----------|--------|
| Fold splitter used | `src/split.py` — group-preserving DSU-based fold creation |
| Total unique entities | N/A |
| Total S1 entities | N/A |
| n_splits | N/A |
| Known positive components stay together | ✅ Verified by DSU union-find on ground truth |
| Unmatched S2/S3 distributed across folds | ✅ Assigned independently via sorted-deterministic fold cycling |
| Training excludes both endpoints of val fold | ✅ `training = (source1_fold != val_fold) & (candidate_fold != val_fold)` |

## 4. OOF Score Coverage

| Property | Status |
|----------|--------|
| Every candidate pair gets exactly one finite OOF score | ✅ Verified — `generate_oof_predictions` raises RuntimeError if any non-finite |
| S1 IDs with no candidates included in denominator | ✅ `evaluate_predictions` iterates ground_truth; missing predictions default to [] giving FN=len(true_matches) |
| Total training pairs with OOF scores | N/A |

## 5. Baseline Metrics

### OOF Training Metrics

| Metric | Value |
|--------|-------|
| **Overall Macro F0.5** | **N/A** |
| Singleton Accuracy | N/A |
| Non-Singleton Macro F0.5 | N/A |
| Selected Threshold | N/A |
| Positive Predictions | N/A |
| Positive Prediction % | N/A% |
| Positive Entities | N/A |
| Positive Weight | N/A |
| Training Runtime (model only) | N/As |
| Feature Generation Runtime | N/As |

### Candidate Retrieval Metrics (Training Set)

| Metric | Value |
|--------|-------|
| Overall Recall | N/A |
| S2 Recall | N/A |
| S3 Recall | N/A |
| Oracle Macro F0.5 (upper bound) | N/A |
| Total Candidate Pairs | N/A |
| Candidate Generation Runtime | N/As |

## 6. Submission Validation

**Validator status: ❌ FAIL (outputs missing)**

```
Output files not found — pipeline may not have completed.
```

## 7. Artifact Paths

| Artifact | Path |
|----------|------|
| work_dir | ✅ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2` |
| output_dir | ✅ `/home/daksh/Amazon_ML_challenge/ml-challenge/output/h04_baseline_v2` |
| model_dir | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/model` |
| folds | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/folds.tsv` |
| fold_summary | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/fold_summary.json` |
| train_candidates | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/train_candidates.tsv` |
| candidate_report | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/train_candidate_report.json` |
| train_features | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/train_features.parquet` |
| oof_predictions | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/model/oof_predictions.tsv` |
| metrics | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/model/metrics.json` |
| best_threshold | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/artifacts/h04_baseline_v2/model/best_threshold.json` |
| matching_results | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/output/h04_baseline_v2/matching_results.tsv` |
| candidate_pairs | ❌ `/home/daksh/Amazon_ML_challenge/ml-challenge/output/h04_baseline_v2/candidate_pairs.tsv` |

## 8. Input & Fold Checksums (MD5)

| File | MD5 |
|------|-----|
| `train_source1.tsv` | `1a0c98e43dad1babed3c99bca2c7b5bd` |
| `train_source2.tsv` | `a7bddba33ead4a85e6c088ad3b377fc2` |
| `train_source3.tsv` | `5df07bf109f55a292bdfaa8c983cfa20` |
| `train_ground_truth.tsv` | `3643443570b2c881c425d69c2d46e95d` |

## 9. Dependency Versions

```
numpy==2.5.3
pandas==3.0.6
scikit-learn==1.9.1
joblib==1.6.0
pyarrow==25.0.1
sparse-dot-topn==1.2.0
```

## 10. Unresolved Issues

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | France not in training data (test-only country) | WARN | Open — model has no France-country signal |
| 2 | ~197k branch ambiguities in France data | WARN | Open — no deduplication applied |
| 3 | `business_address` non-ASCII rate shifts S2/S3 train→test (+5pp) | WARN | Open — may affect address feature alignment |
| 4 | Submission validator did not return PASS | ERROR | Must fix before submission |

---
_This report was auto-generated by `scripts/generate_baseline_report.py` from pipeline artifacts._
