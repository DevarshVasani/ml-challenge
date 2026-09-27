# Member D — Dependency-Free Implementation Plan

## Scope

This plan contains only work that can be completed **right now** from the current repository state, without waiting for:

- Member A's real neural score artifacts
- Member B's new reverse-retrieval/source-complete production shards
- Member C's unpublished production tree score files
- GPU checkpoints or model weights
- external retrieval indexes
- any additional datasets beyond what is already committed

Current base branch state analyzed:

- `main` at commit `8e508be4c6c5571eabd61af31135e841c3b863a2`
- C's latest string/competition feature work is already merged
- `team-a-neural` exists but is diverged/unmerged
- no `team-d-integration` branch currently exists
- no D fusion/calibration/ownership/decoder implementation exists yet

Your objective in this phase is to make Member D's framework complete enough that future A/B/C artifacts can be plugged in by path + manifest, without redesigning the pipeline.

---

# 1. Create the Member D integration branch

Create a clean branch from latest `main`.

```bash
git checkout main
git pull
git checkout -b team-d-integration
```

Base SHA:

```text
8e508be4c6c5571eabd61af31135e841c3b863a2
```

Do not branch from stale branches such as:

```text
feature/phase-d-streaming
feature/integrate-members-1-2-3
feature/member3-pair-model
```

Their useful work is already merged into `main`.

---

# 2. Freeze the canonical Member D artifact contract

Create:

```text
src/member_d_contracts.py
```

Canonical unique pair key:

```python
PAIR_KEY = [
    "candidate_source",
    "candidate_entity_id",
    "source1_entity_id",
]
```

Canonical source-record competition key:

```python
SOURCE_GROUP_KEY = [
    "candidate_source",
    "candidate_entity_id",
]
```

Canonical decision routes:

```python
VALID_DECISION_ROUTES = {
    "neural_fusion",
    "tree_only",
    "discard",
}
```

Canonical score/decision fields:

```text
tree_score
neural_logit
neural_probability
neural_requested
has_neural_score
decision_route
match_probability
chosen_owner
selected
```

Implement:

```python
validate_pair_keys(frame)
validate_tree_scores(frame)
validate_neural_scores(frame)
validate_joined_scores(frame)
validate_decisions(frame)
```

Validation requirements:

- all IDs must be strings
- no empty IDs
- pair key must be unique
- `candidate_source` must be `S2` or `S3`
- `S2` rows must have candidate IDs beginning with `S2-`
- `S3` rows must have candidate IDs beginning with `S3-`
- duplicate pair keys must fail
- tree scores must be finite
- neural scores must be finite when present
- `neural_requested == 1` and neural score missing must fail
- `tree_only` and neural score missing is valid
- production score artifacts must reject label/truth-derived columns
- no truth counts, folds, split labels, positive-injection flags, or gold-derived provenance may enter fusion features

Do not join any artifacts by row order.

---

# 3. Fix tree score output to preserve the full pair key

Modify:

```text
src/predict_pair_model.py
```

Current output drops:

```text
candidate_source
```

Change required identifier fields to:

```python
required = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
}
```

Tree score output must preserve:

```text
source1_entity_id
candidate_entity_id
candidate_source
score
```

Keep the generic `score` column at the tree-model boundary to minimize changes to C's current interface.

Rename to:

```text
tree_score
```

inside Member D's join layer.

Update duplicate detection to use the full 3-part key.

Add a test proving these are distinct:

```text
S2 / S2-123 / S1-1
S3 / S3-123 / S1-1
```

Never deduplicate based only on numeric suffixes.

---

# 4. Freeze Member D's clean fitting partitions

Create:

```text
src/member_d_split.py
```

Use committed files:

```text
member-b-starter-v1/threshold_queries.tsv
member-b-starter-v1/evaluation_truth.tsv
```

Use **threshold only**.

Do not use `final` for repeated model selection.

Build connected components across:

```text
S1 query ID
gold S2/S3 matched IDs
```

If two S1 records share a gold S2/S3 record, they must stay in the same Member D partition.

Create deterministic partitions:

```text
fusion_fit      40%
calibration     30%
selection       30%
```

Use seed:

```text
42
```

Balance approximately by:

```text
country
singleton vs non-singleton
gold edge count
```

Write:

```text
configs/member_d/split.tsv
configs/member_d/split_manifest.json
configs/member_d/base_model_exclusion_ids.tsv
```

`split.tsv` schema:

```text
source1_entity_id
member_d_partition
```

Allowed values:

```text
fusion_fit
calibration
selection
```

Manifest must include:

```text
seed
git_commit
threshold_queries_sha256
evaluation_truth_sha256
component_count
partition_counts
country_counts
singleton_counts
gold_edge_counts
```

`base_model_exclusion_ids.tsv` must include every S1/S2/S3 entity in all Member D held-out components.

This file will later be given to A and C to prove their base-model training data does not overlap Member D fitting/calibration/selection components.

---

# 5. Build one canonical Member D evaluator

Create:

```text
src/evaluate_pipeline.py
```

Target CLI:

```bash
python -m src.evaluate_pipeline \
  --config configs/member_d/evaluate.json
```

Do not reimplement macro F0.5.

Reuse:

```text
src.evaluate.evaluate_predictions
src.evaluate.compute_confusion_components
```

Supported inputs should include any subset of:

```text
query manifest / query TSV
ground truth TSV
raw candidate shards
post-gate candidate shards
pair-score shards
final selected pairs
country metadata
decision-route metadata
```

Always evaluate against the full query manifest, including queries with zero candidates.

For raw/post-gate candidate stages, compute:

```text
query_count
candidate_pair_count
zero_candidate_query_count
gold_edge_count
retrieved_gold_edge_count
missed_gold_edge_count
candidate_pair_recall
oracle_macro_f05
partial_miss_query_count
complete_miss_non_singleton_count
mean_candidates_per_s1
p95_candidates_per_s1
max_candidates_per_s1
```

For final/scored predictions, compute:

```text
macro_f05
singleton_accuracy
non_singleton_macro_f05
tp
fp
fn
pair_precision
pair_recall
predicted_edge_count
predicted_nonempty_s1_count
```

Breakdowns when metadata exists:

```text
by_country
by_source
by_route
```

`by_source` must distinguish:

```text
S2
S3
```

`by_route` must distinguish:

```text
neural_fusion
tree_only
```

Write:

```text
reports/member_d/<version>/evaluation.json
reports/member_d/<version>/evaluation_summary.tsv
```

Do not write synthetic fixture metrics to `reports/experiments.csv`.

---

# 6. Generalize complete-group streaming

Current implementation:

```text
src/candidate_io.py
```

is S1-grouped.

Member D requires complete source-record groups for competition and ownership.

Generalize:

```python
iter_complete_groups(
    paths,
    group_columns=("source1_entity_id",),
)
```

Support compound keys.

Required source grouping:

```python
group_columns=(
    "candidate_source",
    "candidate_entity_id",
)
```

Add helper:

```python
iter_complete_source_groups(...)
```

Required behavior:

- carry the last group across file boundaries
- carry the last group across batch boundaries
- reject a group that reappears after it has already been yielded
- validate identical schemas across shards
- reject malformed fragments
- support compound group keys
- preserve deterministic group order
- never return partial groups

Keep backward compatibility with current S1 grouping.

Do not rewrite C's competition-feature formulas.

Reuse:

```python
src.context_features.add_competition_features()
```

only after a complete source group has been assembled.

---

# 7. Build strict score joining

Create:

```text
src/score_join.py
```

API:

```python
join_pair_scores(
    candidates,
    tree_scores,
    neural_scores=None,
    route_table=None,
)
```

All joins must use:

```text
candidate_source
candidate_entity_id
source1_entity_id
```

Output schema should include:

```text
candidate_source
candidate_entity_id
source1_entity_id
tree_score
neural_logit
neural_requested
has_neural_score
decision_route
```

and selected feature columns needed by D.

Hard failure cases:

```text
duplicate candidate key
duplicate tree score key
duplicate neural score key
tree score exists for unknown pair
neural score exists for unknown pair
tree score missing for decision pair
neural score missing when neural_requested == 1
missing declared shard
duplicate shard coverage
schema version mismatch
model/candidate version mismatch
```

Valid case:

```text
decision_route == tree_only
has_neural_score == 0
```

Never use:

```python
fillna(0)
```

to represent missing neural inference.

A missing neural value and a neural score of zero are not equivalent.

---

# 8. Implement source-level competition orchestration

Do not reimplement competition-feature calculations.

Use latest merged:

```python
src.context_features.add_competition_features()
```

Member D must guarantee complete source groups before calling it.

Create orchestration in either:

```text
src/source_competition.py
```

or inside `score_join.py`.

For each complete:

```text
(candidate_source, candidate_entity_id)
```

group:

1. validate all pair keys
2. validate complete group provenance
3. compute C's competition features
4. attach tree/neural scores
5. preserve group identity
6. write versioned output

Expected useful competition features already available include:

```text
competing_s1_count
best_competing_s1_score
competing_best_second_gap
competing_near_tie_count
competing_s1_rank
competing_s1_score_gap
is_top_competing_s1
```

Do not compute these from Member B's bootstrap sample and report them as source-complete production values.

The current bootstrap explicitly states:

```text
complete_by_s1 = true
complete_by_source_record = false
```

---

# 9. Implement ownership audit and selector

Create:

```text
src/ownership.py
```

Implement:

```python
audit_gold_ownership(ground_truth)
```

Composite source-record key:

```text
(candidate_source, candidate_entity_id)
```

Report:

```text
source_record_count
records_with_one_owner
records_with_multiple_owners
multi_owner_violation_count
example_violations
```

You can run this now on committed threshold/final evaluation truth for code validation.

Label that result:

```text
evaluation truth ownership audit
```

Do not label it as full-training ownership verification until the full training truth becomes available.

Implement:

```python
apply_best_owner(
    frame,
    probability_col="match_probability",
    threshold=...
)
```

Per complete source-record group:

1. discard rows below threshold
2. if no rows survive, source record is unmatched
3. if one survives, select it
4. if multiple survive, choose highest `match_probability`
5. tie-break by higher `tree_score`
6. remaining tie-break by lexicographically smallest `source1_entity_id`

Important:

```text
each S2/S3 record <= 1 S1 owner
```

but:

```text
one S1 may own unlimited S2/S3 records
```

Never impose one-match-per-S1.

---

# 10. Implement fusion framework

Create:

```text
src/fusion.py
```

Default model:

```python
sklearn.linear_model.LogisticRegression
```

Do not require LightGBM for D's first production path.

Use explicit feature allowlist.

Possible initial features:

```text
tree_score
neural_logit
competing_s1_count
competing_s1_score_gap
competing_best_second_gap
competing_near_tie_count
is_top_competing_s1
postcode_match_house_mismatch
strong_name_contradictory_address
address_carries_despite_name_conflict
candidate_source indicator
decision_route indicator
```

Do not automatically select arbitrary numeric columns.

Exclude:

```text
labels
ground-truth counts
fold IDs
split IDs
positive_injected_for_training
truth-derived provenance
member_d_partition
```

Support two separate routes:

```text
neural_fusion
tree_only
```

Never fit a neural-only fusion model and apply it to tree-only rows.

Persist:

```text
model artifact
feature list
feature order
route
training partition
row count
positive count
git SHA
input hashes
fusion version
```

Implementation and tests can be completed now using synthetic data.

Production fitting waits for real keyed score artifacts.

---

# 11. Implement calibration framework

Create:

```text
src/calibration.py
```

Default calibrator:

```text
Platt / logistic calibration
```

Provide API:

```python
fit_calibrator(scores, labels)
apply_calibrator(scores)
```

Calibrate routes separately:

```text
neural_fusion
tree_only
```

Persist:

```text
route
calibration version
training partition
row count
positive count
input hashes
git SHA
```

Optional isotonic calibration can be added later only if sample size supports it.

Do not implement test-prior correction.

Do not infer France accuracy from unlabeled score distributions.

---

# 12. Implement decoder framework

Create:

```text
src/decoder.py
```

Mandatory decoders:

```python
decode_threshold(...)
decode_threshold_with_ownership(...)
```

Decoder 1:

```text
calibrated threshold
```

Decoder 2:

```text
calibrated threshold
+
best-owner selection
```

Both must allow:

```text
empty S1 prediction
```

Add:

```python
compare_decoders(...)
```

Selection rule:

```text
highest macro F0.5 on Member D selection split
```

Tie-break toward simpler decoder:

```text
threshold
before
threshold + ownership
before
expected-F0.5
```

Do not implement expected-F0.5 until all mandatory infrastructure is complete.

It is optional.

---

# 13. Implement decision-to-S1 regrouping

Ownership operates by:

```text
(candidate_source, candidate_entity_id)
```

Final contest output operates by:

```text
source1_entity_id
```

Create:

```text
src/final_regroup.py
```

Pipeline:

```text
complete source groups
→ competition
→ score join
→ fusion
→ calibration
→ ownership
→ retained edge rows
→ regroup by S1
→ final set decoder
```

Never finalize one S1 while source-record shards that could still assign more records to it remain incomplete.

Output:

```text
source1_entity_id
candidate_source
candidate_entity_id
match_probability
decision_route
selected
```

before final export.

---

# 14. Make final export route-aware

Current:

```text
src/export.py::export_streaming_shards()
```

assumes one score stream over one candidate stream.

Member D must support:

```text
neural_fusion
tree_only
discard
```

Create either:

```text
src/final_assembly.py
```

or a new route-aware export function in `src/export.py`.

Inputs:

```text
full query manifest
final decision pair shards
final selected match shards
source registries
completion manifests
```

`candidate_pairs.tsv` must represent:

```text
all pairs that reached a real final matching decision
```

including:

```text
tree_only pairs
```

Do not generate the candidate file from neural scores only.

Final output files:

```text
matching_results.tsv
candidate_pairs.tsv
```

Exact schemas remain:

```text
matching_results.tsv
source1_entity_id    matched_entity_ids
```

```text
candidate_pairs.tsv
source1_entity_id    candidate_entity_ids
```

Validate before atomic publish:

```text
every S1 exactly once
exact S1 order
valid S2/S3 IDs
no duplicate pair
no duplicate IDs within an S1 row
all final matches are contained in final candidates
all required shards complete
no unknown S1
exact TSV headers
correct tab formatting
ownership uniqueness if ownership enabled
```

Retain current temp-file plus atomic-rename behavior.

---

# 15. Build a complete Member D integration fixture

Create:

```text
tests/fixtures/member_d/
tests/test_member_d_integration.py
```

Fixture must include:

```text
S1 with zero candidates
S2 and S3 IDs sharing the same numeric suffix
one source record competing across two S1s
exact probability tie
source group split across two files
tree-only pair with missing neural score
neural-required pair with missing neural score
missing tree score
duplicate pair key
extra score key
missing declared shard
match outside candidate set
one S1 owning multiple valid source records
```

Required assertions:

```text
3-part pair keys remain unique
joins are order-independent
source groups reassemble across files
reappearing source groups fail
ties resolve deterministically
empty queries remain empty output rows
tree-only without neural is valid
neural-required without neural fails
missing tree score fails
duplicate score keys fail
unknown score pairs fail
final match outside candidate set fails
ownership gives <= 1 owner per S2/S3 record
one S1 may own multiple records
```

Use synthetic scores.

Never report synthetic fixture metrics as model quality.

---

# 16. Add Member D artifact manifests

Create:

```text
src/member_d_manifest.py
```

Reuse checksum helpers from:

```text
src/neural_contracts.py
```

where possible.

Every D artifact manifest should record:

```text
schema_version
artifact_kind
git_commit
input_versions
input_hashes
candidate_version
tree_model_version
neural_model_version
fusion_version
calibration_version
decoder_version
split
shard_id
row_count
completion_state
sha256
created_at
```

Recommended layout:

```text
configs/member_d/
artifacts/member_d/
reports/member_d/
output/member_d/
```

Suggested structure:

```text
configs/member_d/
  split.tsv
  split_manifest.json
  base_model_exclusion_ids.tsv
  evaluate.json
  fusion.json
  calibration.json
  decoder.json
  runtime_budget.json

artifacts/member_d/<version>/
  joined_scores/
  source_group_scores/
  fusion/
  calibrated_scores/
  selected_edges/
  manifests/

reports/member_d/<version>/
  evaluation.json
  evaluation_summary.tsv
  ownership_audit.json
  comparison.json
  comparison.tsv
  runtime_budget.json

output/member_d/<version>/
  matching_results.tsv
  candidate_pairs.tsv
  export_manifest.json
```

---

# 17. Build the cheap-path fallback framework

Create a fully independent tree-only path.

Architecture:

```text
candidate pairs
→ tree score
→ tree-only calibration
→ threshold
→ optional ownership
→ S1 regroup
→ export
→ validation
```

Route:

```text
tree_only
```

Suggested version:

```text
fallback_tree_only_v1
```

Suggested artifact path:

```text
artifacts/member_d/fallback_tree_only_v1/
output/member_d/fallback_tree_only_v1/
```

This fallback must have its own:

```text
manifest
metrics
calibration artifact
threshold config
decoder config
final outputs
```

Do not silently switch from neural fusion to fallback.

Fallback must be independently versioned and evaluated.

---

# 18. Add runtime-budget infrastructure

Create:

```text
src/runtime_budget.py
configs/member_d/runtime_budget.json
```

Inputs:

```text
stage
row_count
pairs_per_second
peak_ram_gb
disk_input_gb
disk_output_gb
worker_count
gpu_count
```

Outputs:

```text
projected_stage_runtime
projected_total_runtime
critical_path
peak_concurrent_ram
projected_disk_usage
latest_safe_start_time
```

Resource limits must be configurable.

Do not hardcode 64 GB.

Member B's current environment handoff reports:

```text
4 CPUs
13.6 GiB cgroup memory limit
~327 GiB free disk
```

so production limits must come from config.

Add a simple guard:

```text
refuse to start high-memory/high-I/O concurrent stages
when configured aggregate memory/workers exceed budget
```

---

# 19. Comparison-report framework

Create:

```text
reports/member_d/
```

Comparison rows should eventually support:

```text
retrieval raw oracle
post-gate oracle
tree-only calibrated
neural fusion calibrated
threshold + ownership
expected-F0.5 if tested
fallback tree-only
final frozen configuration
```

Columns:

```text
candidate_version
tree_version
neural_version
fusion_version
calibration_version
decoder_version
macro_f05
singleton_accuracy
pair_precision
pair_recall
raw_oracle
post_gate_oracle
candidate_pair_count
US_macro_f05
India_macro_f05
S2_pair_precision
S2_pair_recall
S3_pair_precision
S3_pair_recall
runtime_seconds
peak_ram_gb
artifact_path
```

Unavailable values must be:

```text
N/A
```

Never fabricate zero values for unavailable measurements.

---

# 20. Dependency-free implementation order

Implement in this exact order:

```text
1. Create team-d-integration branch

2. Add src/member_d_contracts.py

3. Fix src/predict_pair_model.py
   to preserve candidate_source

4. Add Member D split logic
   src/member_d_split.py

5. Add unified evaluator
   src/evaluate_pipeline.py

6. Generalize src/candidate_io.py
   for compound complete-group keys

7. Add strict score join
   src/score_join.py

8. Add ownership audit/selector
   src/ownership.py

9. Add tests/test_member_d_integration.py

10. Add fusion framework
    src/fusion.py

11. Add calibration framework
    src/calibration.py

12. Add decoder framework
    src/decoder.py

13. Add source-to-S1 regrouping
    src/final_regroup.py

14. Add route-aware final assembly/export
    src/final_assembly.py

15. Add artifact manifests
    src/member_d_manifest.py

16. Wire the tree-only fallback

17. Add runtime budget utility

18. Run the full pytest suite

19. Fix regressions

20. Commit the dependency-free Member D framework
```

---

# 21. Do not work on these items yet

Do not spend time on:

```text
reverse retrieval
retrieval indexes
mDeBERTa training
ByT5 training
new string-similarity features
new tree feature engineering
new LightGBM experiments
dense retrieval
France prior correction
production threshold tuning
production fusion fitting
production decoder selection
expected-F0.5 decoder
final submission tuning
```

These either belong to A/B/C or require future production artifacts.

Do not rewrite:

```python
src.context_features.add_competition_features()
```

The latest C merge already implements it.

Member D's responsibility is to guarantee that its input is source-complete.

---

# 22. Expected dependency-free milestone

Before needing anything else from the team, you should be able to complete:

```text
team-d-integration branch

3-part pair contract

tree output key fix

Member D clean split

base-model exclusion manifest

unified evaluation CLI

complete source-group iterator

strict ID-based score joining

source-group competition orchestration

ownership audit

best-owner selector

fusion API

route-specific calibration API

threshold decoder

threshold + ownership decoder

S1 regrouping

route-aware final assembly

submission validation

tree-only fallback framework

strong synthetic integration fixture

artifact manifests

runtime-budget utility
```

At that point, future inputs become plug-in artifacts.

You should only need:

```text
path
manifest
checksum
version
```

from A/B/C.

---

# 23. Definition of done for this phase

This dependency-free phase is complete only when:

- all pair joins use the 3-part key
- tree prediction preserves `candidate_source`
- D split is deterministic and component-safe
- `final` is not used for repeated tuning
- unified evaluator includes zero-candidate queries
- source groups can span multiple shards safely
- incomplete/repeated groups fail loudly
- C's competition features run only on complete source groups
- tree-only route works without neural scores
- neural-required missing score fails
- ownership selector is deterministic
- no S1 is artificially limited to one match
- calibration supports tree-only and neural-fusion separately
- empty predictions remain possible
- candidate export includes tree-only decision pairs
- final matches must be a subset of final candidates
- every test S1 can be emitted exactly once
- fixture tests cover missing shards, duplicates, ties, empty queries, and route behavior
- every D artifact has version/provenance/checksum metadata
- fallback is explicit and separately versioned
- full test suite passes

---

# 24. Realistic implementation effort

Approximate engineering effort:

```text
Contracts + tree-key fix          1–1.5 h
Member D split                    1 h
Unified evaluator                 1.5–2 h
Complete-group infrastructure     1 h
Score joining                     1–1.5 h
Ownership                         1 h
Fusion + calibration framework    1.5–2 h
Decoder                           1 h
Route-aware export                1–1.5 h
Integration fixture/tests         1.5–2 h
Manifests/fallback/runtime        1–1.5 h
```

Total:

```text
~11–15 focused engineering hours
```

A strong first checkpoint is:

```text
~5–6 hours
```

with these completed:

```text
contracts
tree-key fix
clean D split
unified evaluator
complete source grouping
strict score joins
ownership audit/selector
integration tests
```

After that, A/B/C artifacts can be incorporated without redesigning Member D's pipeline.
