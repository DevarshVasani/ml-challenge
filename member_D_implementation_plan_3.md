# Member D Implementation Plan 3

## Main update
- Previous processed `main`: `6914c5e71dcb58f7e6cde758be68d522dddd7832`
- Current analyzed `main`: `7c1faeffd11f214867a919ef094315b991506700`
- Delta: 2 commits; `.gitignore` and `src/train_pair_model.py`
- Plans 1 and 2 remain the base. This plan contains only new/changed D work.

## What changed
Member C fixed a feature leak: `is_injected_positive` could be selected through a prefix allowlist even though it is training-only provenance. It is now explicitly in `NON_FEATURE_COLUMNS` and removed from `FEATURE_ALLOWLIST_PREFIXES`. C reports retrained v5 F0.5 remains 0.9596. `.gitignore` now ignores `oracle_recall_report.tsv*`.

## D impact
Treat tree models trained before this fix as provenance-tainted unless their actual feature list proves the leaked field was absent. No pair-key, ownership, decoder, neural-score, calibration architecture, or export contract changed.

## 1. Strengthen D's forbidden-feature contract
In `src/member_d_contracts.py`, explicitly forbid:
```text
positive_injected_for_training
is_injected_positive
label
fold
source1_fold
candidate_fold
member_d_partition
```
Continue forbidding truth counts and truth-derived provenance from Plan 1.

Add:
```python
validate_feature_allowlist(feature_columns)
```
It must reject forbidden fields regardless of prefix/pattern allowlists. The explicit forbidden set must have final precedence.

## 2. Gate C tree artifacts by real model provenance
Require tree artifacts consumed by D to identify:
```text
tree_model_version
training_git_sha
feature_columns_sha256
feature_schema_version
```
Before accepting tree scores for fusion/calibration/selection:
1. load the actual tree feature list;
2. assert `is_injected_positive` is absent;
3. assert `positive_injected_for_training` is absent;
4. run D's complete forbidden-feature validator;
5. record the validation in D's joined-artifact manifest.

Suggested fields:
```text
tree_feature_contract_validated: true
tree_feature_contract_validation_version: member_d_v1
tree_feature_columns_sha256: ...
tree_training_git_sha: ...
```
Do not infer cleanliness from a filename such as `v5`.

## 3. Invalidate contaminated local D artifacts
If any D comparison/fusion/calibration artifact was produced from a tree model whose feature list contained `is_injected_positive`, mark it:
```text
invalid_for_selection = true
invalid_reason = "tree training feature leakage: is_injected_positive"
```
Do not use it as valid model/decoder selection evidence.

## 4. Add regression tests
Add:
```text
test_rejects_is_injected_positive
test_rejects_positive_injected_for_training
test_prefix_allowlist_cannot_reinclude_forbidden_feature
test_clean_tree_feature_list_passes
```
Include the exact regression class: a field is present in an exclusion set but also matches a broad allowlist prefix. D must still reject it.

## 5. Extend comparison-report provenance
Add:
```text
tree_training_git_sha
tree_feature_schema_version
tree_feature_columns_sha256
feature_contract_valid
```
Only configurations with `feature_contract_valid == true` are eligible for final model/decoder selection.

Do not treat the unchanged 0.9596 metric as evidence that an old artifact is methodologically valid.

## 6. Keep generated oracle reports out of Git
The new `.gitignore` rule confirms `oracle_recall_report.tsv*` is a generated artifact. Do not force-add it.

Commit only D code, tests, schemas, and safe config templates. Keep generated evaluation/truth-derived reports in D's versioned artifact/report location.

## Immediate order
1. Rebase/pull `team-d-integration` onto `7c1faeff...`.
2. Add `is_injected_positive` to D's forbidden set.
3. Implement `validate_feature_allowlist()`.
4. Add tree feature-list/provenance validation at ingestion.
5. Add leakage regression tests.
6. Invalidate any affected pre-fix local D artifacts.
7. Add feature-contract provenance to comparison reports.
8. Continue Plans 1 and 2 unchanged.

## Done when
- both injected-positive provenance fields are explicitly forbidden;
- prefix selection cannot override D's forbidden set;
- every production tree artifact is validated from its actual feature list;
- training SHA/schema/feature hash are recorded;
- tainted artifacts cannot participate in D selection;
- regression tests prevent this leak class;
- no unnecessary key/ownership/decoder/neural/export changes are made.

## Incremental effort
**30–60 minutes.** This is contract/provenance hardening, not a D pipeline redesign.
