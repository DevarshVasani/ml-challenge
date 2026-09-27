# Member D Implementation Plan 2

## Main branch update

- Previous processed `main`: `8e508be4c6c5571eabd61af31135e841c3b863a2`
- Current analyzed `main`: `6914c5e71dcb58f7e6cde758be68d522dddd7832`
- Change set: 2 commits ahead; one new source file, `src/evaluate_fake_patterns.py`
- Upstream owner: Member C
- Existing `member_D_implementation_plan_1.md` remains the base implementation plan. This file contains only Member D work added or changed by this update.

## What changed

C added targeted evaluation for difficult fake-record patterns:

- branch/name-token additions or removals
- legal-form disagreement
- house-number shifts
- postcode-match plus house-number-mismatch contradictions

The script also measures a guard rail: legitimate positive pairs that resemble these fake patterns should not be over-rejected.

The commit reports, for C's v4 validation setup:

- fake-pattern rejection/catch rate: 98.15%
- legitimate-variation keep rate: 95.93%

These numbers are useful diagnostics, but they must **not** be treated as Member D selection-set results. `src/evaluate_fake_patterns.py` creates its own `GroupShuffleSplit` over C's training-pair data and uses a fixed tree threshold of `0.5`.

## Member D impact

No pair-key, candidate schema, ownership, neural-score, export, or artifact-manifest contract changed in this update.

Therefore, do **not** redesign the D pipeline from Plan 1.

The new work for D is to absorb C's fake-pattern diagnostics into the canonical D evaluation framework, while keeping D's frozen `fusion_fit`, `calibration`, and `selection` partitions authoritative.

## New task 1 — Add fake-pattern slices to `src/evaluate_pipeline.py`

When implementing the unified D evaluator from Plan 1, add optional diagnostic slices equivalent to C's new evaluator.

Define reusable masks for:

```text
branch_word_added
legal_form_swapped
house_number_shifted
postcode_masks_house
fake_pattern_union
legit_variation_guardrail
```

Use the existing feature columns:

```text
name_raw_similarity
name_tokens_extra_in_b
name_tokens_missing_from_b
legal_form_agreement
house_number_match
house_number_a_present
house_number_b_present
postcode_match_house_mismatch
```

Do not duplicate these formulas in multiple D modules. Put slice construction in one helper, for example:

```text
src/evaluation_slices.py
```

with an API such as:

```python
build_fake_pattern_masks(frame)
evaluate_fake_pattern_slices(frame, label_col, prediction_col, score_col=None)
```

## New task 2 — Evaluate these slices on D's frozen partitions

Do not reuse the random `GroupShuffleSplit` inside `src/evaluate_fake_patterns.py` for D model selection.

D evaluation must use the partitions from:

```text
configs/member_d/split.tsv
```

Report fake-pattern diagnostics separately for:

```text
fusion_fit
calibration
selection
```

The `selection` slice is the one relevant when comparing final D decision policies.

Do not tune fusion, calibration, thresholds, ownership rules, or decoder behavior directly on C's validation split.

## New task 3 — Add guard-rail metrics to D reports

Extend D evaluation output with:

```text
fake_pattern_negative_count
fake_pattern_catch_rate
fake_pattern_false_accept_count

legit_variation_positive_count
legit_variation_keep_rate
legit_variation_false_reject_count
```

Also report per-pattern values for:

```text
branch_word_added
legal_form_swapped
house_number_shifted
postcode_masks_house
```

For each pattern, record at minimum:

```text
negative_count
negative_catch_rate
positive_count
positive_keep_rate
mean_match_probability
```

When a slice has zero rows, emit:

```text
N/A
```

rather than `0.0`.

## New task 4 — Compare D routes on hard-fake slices

Once D's route-aware scoring framework exists, report these diagnostics by decision route:

```text
neural_fusion
tree_only
```

This is important because C's new script only evaluates the tree model.

D should eventually answer:

```text
Does neural fusion rescue legitimate variations rejected by the tree?
Does neural fusion accidentally accept fake-pattern negatives?
Does the tree-only fallback remain safe on contradiction-heavy examples?
```

Add the following report dimensions when data is available:

```text
pattern × route
pattern × source
pattern × country
```

Do not block the base evaluator if one of these metadata dimensions is unavailable.

## New task 5 — Preserve fake-pattern fields in the D feature allowlist review

Plan 1 already requires an explicit fusion feature allowlist.

When that allowlist is finalized, explicitly review these C features as candidate D fusion inputs:

```text
postcode_match_house_mismatch
legal_form_agreement
house_number_match
name_tokens_extra_in_b
name_tokens_missing_from_b
```

Do not automatically add them solely because C's targeted evaluator shows strong tree performance.

Their inclusion must still be decided using `fusion_fit` and validated through D's calibration/selection process.

## New task 6 — Add regression tests

Extend:

```text
tests/test_member_d_integration.py
```

or add:

```text
tests/test_evaluation_slices.py
```

Test at least:

1. a negative with an added branch token is classified into `branch_word_added`;
2. a legal-form mismatch is classified correctly;
3. a house-number mismatch with both numbers present enters `house_number_shifted`;
4. postcode match + house mismatch enters `postcode_masks_house`;
5. a legitimate positive can belong to the fake-pattern mask and contributes to the guard-rail keep rate;
6. zero-row slices return `N/A`, not a misleading zero;
7. slice evaluation does not alter the canonical macro-F0.5 calculation;
8. D partition membership is supplied externally rather than recreated with `GroupShuffleSplit`.

## What not to change

This update does **not** justify changes to:

```text
PAIR_KEY
SOURCE_GROUP_KEY
score joining
ownership semantics
calibration architecture
decoder architecture
candidate export semantics
tree-only fallback semantics
artifact manifest schema
```

Continue those items exactly as specified in `member_D_implementation_plan_1.md`.

Do not make `src/evaluate_fake_patterns.py` the canonical D evaluator. Treat it as C's diagnostic implementation and reuse its pattern definitions in D's evaluation framework.

## Immediate implementation order

1. Continue Plan 1's core contract/key work first.
2. Add `src/evaluation_slices.py`.
3. Add fake-pattern slice metrics to `src/evaluate_pipeline.py`.
4. Add partition-aware reporting using D's frozen split.
5. Add route-aware hard-fake diagnostics.
6. Add regression tests.
7. Add the slice metrics to D's comparison report schema.
8. Run the complete test suite.

## Definition of done for this update

This update is absorbed when:

- D can reproduce C's four fake-pattern definitions;
- D evaluates them on D's own frozen partitions rather than C's random validation split;
- fake rejection and legitimate-variation retention are both reported;
- zero-size slices are represented as `N/A`;
- diagnostics can later be broken down by `neural_fusion` and `tree_only`;
- canonical macro F0.5 remains unchanged;
- no D artifact contract was unnecessarily modified;
- regression tests pass.

## Estimated incremental Member D effort

This update adds approximately **1–2 focused hours** on top of Plan 1.

The work is diagnostic/evaluation integration only. It does not block the core D contract, joining, ownership, fusion, calibration, decoder, or export implementation.
