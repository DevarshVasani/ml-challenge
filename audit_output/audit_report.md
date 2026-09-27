# Data Quality & Robustness Audit

## Executive Summary

- **Errors**: 1
- **Warnings**: 11
- **Info**: 13
- **Datasets audited**: 6

### Critical Issues

- **[ERROR]** [submission_validation] Validator not found: /home/devarsh/projects/amazon-ml-challenge/ml-challenge/student_resource/utils/validate_submission.py

### Warnings

- **[WARN]** [match_cardinality] 1964417 S1 entities have multiple matches.
- **[WARN]** [source1] Country only in test: ['France']
- **[WARN]** [source2] Country only in test: ['France']
- **[WARN]** [source2] 'business_address' non-ASCII shift: 9.5% -> 14.7%
- **[WARN]** [source3] Country only in test: ['France']
- **[WARN]** [source3] 'business_address' non-ASCII shift: 9.0% -> 14.3%
- **[WARN]** [france] 197782 branch ambiguities.
- **[WARN]** [submission_writer] __init__.py: Missing expected output columns.
- **[WARN]** [submission_writer] metrics.py: Missing expected output columns.
- **[WARN]** [submission_writer] pipeline.py: Missing expected output columns.
- **[WARN]** [submission_writer] text_utils.py: Missing expected output columns.

## Dataset Inventory

### test_source1

- Rows: 1732544
- Columns: 4 - `['entity_id', 'business_name', 'business_address', 'country']`
- Unique IDs: 1732544
- Duplicate IDs: 0
- Countries:
  - 'India': 809986
  - 'US': 663106
  - 'France': 259452

### test_source2

- Rows: 4887273
- Columns: 4 - `['entity_id', 'business_name', 'business_address', 'country']`
- Unique IDs: 4887273
- Duplicate IDs: 0
- Countries:
  - 'India': 2312565
  - 'US': 1871330
  - 'France': 703378

### test_source3

- Rows: 5082316
- Columns: 4 - `['entity_id', 'business_name', 'business_address', 'country']`
- Unique IDs: 5082316
- Duplicate IDs: 0
- Countries:
  - 'India': 2405000
  - 'US': 1945701
  - 'France': 731615

### train_source1

- Rows: 2206821
- Columns: 4 - `['entity_id', 'business_name', 'business_address', 'country']`
- Unique IDs: 2206821
- Duplicate IDs: 0
- Countries:
  - 'US': 1323633
  - 'India': 883188

### train_source2

- Rows: 5034616
- Columns: 4 - `['entity_id', 'business_name', 'business_address', 'country']`
- Unique IDs: 5034616
- Duplicate IDs: 0
- Countries:
  - 'US': 3016817
  - 'India': 2017799

### train_source3

- Rows: 5285603
- Columns: 4 - `['entity_id', 'business_name', 'business_address', 'country']`
- Unique IDs: 5285603
- Duplicate IDs: 0
- Countries:
  - 'US': 3170056
  - 'India': 2115547

## Missingness

| Dataset | Column | Total | Missing | Missing % | Sentinels |
|---------|--------|-------|---------|-----------|-----------|
| test_source1 | entity_id | 1732544 | 0 | 0.0% |  |
| test_source1 | business_name | 1732544 | 0 | 0.0% |  |
| test_source1 | business_address | 1732544 | 0 | 0.0% |  |
| test_source1 | country | 1732544 | 0 | 0.0% |  |
| test_source2 | entity_id | 4887273 | 0 | 0.0% |  |
| test_source2 | business_name | 4887273 | 0 | 0.0% | {"NA": 46} |
| test_source2 | business_address | 4887273 | 129408 | 2.65% |  |
| test_source2 | country | 4887273 | 0 | 0.0% |  |
| test_source3 | entity_id | 5082316 | 0 | 0.0% |  |
| test_source3 | business_name | 5082316 | 0 | 0.0% | {"NA": 59} |
| test_source3 | business_address | 5082316 | 136098 | 2.68% |  |
| test_source3 | country | 5082316 | 0 | 0.0% |  |
| train_source1 | entity_id | 2206821 | 0 | 0.0% |  |
| train_source1 | business_name | 2206821 | 0 | 0.0% |  |
| train_source1 | business_address | 2206821 | 0 | 0.0% |  |
| train_source1 | country | 2206821 | 0 | 0.0% |  |
| train_source2 | entity_id | 5034616 | 0 | 0.0% |  |
| train_source2 | business_name | 5034616 | 0 | 0.0% | {"NA": 2} |
| train_source2 | business_address | 5034616 | 168967 | 3.36% |  |
| train_source2 | country | 5034616 | 0 | 0.0% |  |
| train_source3 | entity_id | 5285603 | 0 | 0.0% |  |
| train_source3 | business_name | 5285603 | 0 | 0.0% | {"NA": 13} |
| train_source3 | business_address | 5285603 | 175916 | 3.33% |  |
| train_source3 | country | 5285603 | 0 | 0.0% |  |

## Duplicate IDs

No duplicate IDs detected.


## Ground-Truth Integrity

- **total_gt_rows**: 2206821
- **unique_s1_in_gt**: 2206821
- **total_s1_in_train**: 2206821
- **s1_missing_from_gt**: 0
- **s1_not_in_train**: 0
- **orphan_s2_refs**: 0
- **orphan_s3_refs**: 0
- **invalid_prefix_refs**: 0
- **blank_s1_ids**: 0
- **duplicate_gt_rows**: 0


## Source 1 Match Cardinality

- 0 matches: 123247
- 1 match: 119157
- >1 matches: 1964417

| Match Count | S1 Entities |
|-------------|-------------|
| 0 | 123247 |
| 1 | 119157 |
| 2 | 375212 |
| 3 | 530841 |
| 4 | 484115 |
| 5 | 321957 |
| 6 | 164868 |
| 7 | 63968 |
| 8 | 18680 |
| 9 | 4205 |
| 10 | 534 |
| 11 | 37 |

## Train vs Test Differences

| Source | Field | Metric | Train | Test | Diff |
|--------|-------|--------|-------|------|------|
| source1 | entity_id | missing_pct | 0.0 | 0.0 | 0.0 |
| source1 | business_name | missing_pct | 0.0 | 0.0 | 0.0 |
| source1 | business_address | missing_pct | 0.0 | 0.0 | 0.0 |
| source1 | country | missing_pct | 0.0 | 0.0 | 0.0 |
| source1 | business_name | len_median | 24.0 | 24.0 | 0.0 |
| source1 | business_name | len_p90 | 34.0 | 34.0 | 0.0 |
| source1 | business_name | len_p95 | 37.0 | 36.0 | -1.0 |
| source1 | business_name | len_max | 87 | 92 | 5 |
| source1 | business_address | len_median | 41.0 | 50.0 | 9.0 |
| source1 | business_address | len_p90 | 90.0 | 93.0 | 3.0 |
| source1 | business_address | len_p95 | 103.0 | 105.0 | 2.0 |
| source1 | business_address | len_max | 256 | 240 | -16 |
| source1 | country | country_France | 0 | 259452 | 259452 |
| source1 | country | country_India | 883188 | 809986 | -73202 |
| source1 | country | country_US | 1323633 | 663106 | -660527 |
| source1 | business_name | non_ascii_pct | 0.0 | 2.35 | 2.35 |
| source1 | business_address | non_ascii_pct | 0.03 | 4.26 | 4.23 |
| source2 | entity_id | missing_pct | 0.0 | 0.0 | 0.0 |
| source2 | business_name | missing_pct | 0.0 | 0.0 | 0.0 |
| source2 | business_address | missing_pct | 3.36 | 2.65 | -0.71 |
| source2 | country | missing_pct | 0.0 | 0.0 | 0.0 |
| source2 | business_name | len_median | 25 | 25 | 0 |
| source2 | business_name | len_p90 | 37 | 38.0 | 1.0 |
| source2 | business_name | len_p95 | 40 | 42.0 | 2.0 |
| source2 | business_name | len_max | 104 | 90 | -14 |
| source2 | business_address | len_median | 37 | 43 | 6 |
| source2 | business_address | len_p90 | 84.0 | 87.0 | 3.0 |
| source2 | business_address | len_p95 | 97.0 | 99.0 | 2.0 |
| source2 | business_address | len_max | 230 | 238 | 8 |
| source2 | country | country_France | 0 | 703378 | 703378 |
| source2 | country | country_India | 2017799 | 2312565 | 294766 |
| source2 | country | country_US | 3016817 | 1871330 | -1145487 |
| source2 | business_name | non_ascii_pct | 15.19 | 18.99 | 3.8 |
| source2 | business_address | non_ascii_pct | 9.5 | 14.75 | 5.24 |
| source3 | entity_id | missing_pct | 0.0 | 0.0 | 0.0 |
| source3 | business_name | missing_pct | 0.0 | 0.0 | 0.0 |
| source3 | business_address | missing_pct | 3.33 | 2.68 | -0.65 |
| source3 | country | missing_pct | 0.0 | 0.0 | 0.0 |
| source3 | business_name | len_median | 25.0 | 25 | 0.0 |
| source3 | business_name | len_p90 | 37.0 | 38 | 1.0 |
| source3 | business_name | len_p95 | 42.0 | 42.0 | 0.0 |
| source3 | business_name | len_max | 97 | 99 | 2 |
| source3 | business_address | len_median | 42 | 44 | 2 |
| source3 | business_address | len_p90 | 78.0 | 82 | 4.0 |
| source3 | business_address | len_p95 | 92.0 | 95.0 | 3.0 |
| source3 | business_address | len_max | 234 | 235 | 1 |
| source3 | country | country_France | 0 | 731615 | 731615 |
| source3 | country | country_India | 2115547 | 2405000 | 289453 |
| source3 | country | country_US | 3170056 | 1945701 | -1224355 |
| source3 | business_name | non_ascii_pct | 11.48 | 14.51 | 3.03 |
| source3 | business_address | non_ascii_pct | 9.02 | 14.35 | 5.33 |

## France-Specific Risks

### country_label_variants

- **France**: 1694445

### dataset_distribution

- **test_source1**: 259452
- **test_source2**: 703378
- **test_source3**: 731615

### diacritics

- **accent_in_names**: 388586
- **accent_in_addresses**: 335837

### postal_codes

- **total**: 8262
- **unique**: 703
- **leading_zero**: 3396
- **samples**: 10 items

### address_abbreviations

- **Boulevard/Bd**: 69800
- **Rue/R.**: 743152
- **Avenue/Av.**: 183944
- **Saint/St**: 181055
- **Sainte/Ste**: 31504
- **CEDEX**: 181

### branch_ambiguity

- **count**: 197782
- **samples**: 10 items

## Difficult Labeled Nonmatches

Top 100 pairs:

| S1 ID | Other ID | Name Sim | Addr Sim | Reasons |
|-------|----------|----------|----------|---------|
| S1-101322744 | S2-720262455 | 1.0 | 0.125 | similar_name |
| S1-101709014 | S3-678119393 | 1.0 | 0.125 | similar_name |
| S1-101056835 | S2-869258427 | 1.0 | 0.111 | similar_name |
| S1-101123477 | S2-291860287 | 1.0 | 0.111 | similar_name |
| S1-101123477 | S3-680732410 | 1.0 | 0.111 | similar_name |
| S1-101645806 | S2-25283388 | 1.0 | 0.091 | similar_name |
| S1-100057571 | S2-435473605 | 1.0 | 0.083 | similar_name |
| S1-10031717 | S3-425838121 | 1.0 | 0.083 | similar_name |
| S1-100057571 | S3-620824264 | 1.0 | 0.067 | similar_name |
| S1-100176728 | S2-436436446 | 1.0 | 0.043 | similar_name |
| S1-100018231 | S2-748515887 | 1.0 | 0.0 | similar_name |
| S1-100174601 | S3-161920382 | 1.0 | 0.0 | similar_name |
| S1-100174601 | S2-845783437 | 1.0 | 0.0 | similar_name |
| S1-100192399 | S2-96364540 | 1.0 | 0.0 | similar_name |
| S1-100196139 | S3-958314666 | 1.0 | 0.0 | similar_name |
| S1-100542041 | S3-251224720 | 1.0 | 0.0 | similar_name |
| S1-100542041 | S2-705562118 | 1.0 | 0.0 | similar_name |
| S1-100589899 | S2-750481128 | 1.0 | 0.0 | similar_name |
| S1-100619468 | S3-979991785 | 1.0 | 0.0 | similar_name |
| S1-100619468 | S2-124733904 | 1.0 | 0.0 | similar_name |
| S1-100669420 | S3-670007738 | 1.0 | 0.0 | similar_name |
| S1-100707047 | S2-315510371 | 1.0 | 0.0 | similar_name |
| S1-100707047 | S3-933640849 | 1.0 | 0.0 | similar_name |
| S1-100758515 | S2-148649228 | 1.0 | 0.0 | similar_name |
| S1-10080885 | S2-220323466 | 1.0 | 0.0 | similar_name |
| S1-10080885 | S2-669325806 | 1.0 | 0.0 | similar_name |
| S1-100836636 | S3-252872740 | 1.0 | 0.0 | similar_name |
| S1-100914897 | S3-37235820 | 1.0 | 0.0 | similar_name |
| S1-100946908 | S3-639561144 | 1.0 | 0.0 | similar_name |
| S1-100946908 | S2-359182640 | 1.0 | 0.0 | similar_name |

... and 70 more in error_review.tsv

## Submission Format Validation

- **Status**: validator_not_found

## Confirmed Findings

- **[WARN]** [match_cardinality] 1964417 S1 entities have multiple matches.
- **[WARN]** [source1] Country only in test: ['France']
- **[WARN]** [source2] Country only in test: ['France']
- **[WARN]** [source2] 'business_address' non-ASCII shift: 9.5% -> 14.7%
- **[WARN]** [source3] Country only in test: ['France']
- **[WARN]** [source3] 'business_address' non-ASCII shift: 9.0% -> 14.3%
- **[WARN]** [france] 197782 branch ambiguities.
- **[WARN]** [submission_writer] __init__.py: Missing expected output columns.
- **[WARN]** [submission_writer] metrics.py: Missing expected output columns.
- **[WARN]** [submission_writer] pipeline.py: Missing expected output columns.
- **[WARN]** [submission_writer] text_utils.py: Missing expected output columns.
- **[ERROR]** [submission_validation] Validator not found: /home/devarsh/projects/amazon-ml-challenge/ml-challenge/student_resource/utils/validate_submission.py

## Unresolved Risks

- GT semantics: Does absence from GT mean 'no match' or 'unknown'?
- Country normalization: Raw labels not collapsed.
- France (zero-shot): Only in test, no training data.
- Branch ambiguity: Same name, different addresses.
- Unicode NFC/NFD: May cause comparison failures.
