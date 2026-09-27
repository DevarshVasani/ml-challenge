# Member D Implementation Plan

## 0. Current repository state to build on

Base implementation on `main` at commit `3a867e5258b309c6d69712c5d144fd7ab4a8d875`. Existing branches `Daksh`, `member2-candidates`, `feature/member3-pair-model`, `feature/integrate-members-1-2-3`, and `feature/phase-d-streaming` are all behind `main` with no unique commits; do not re-integrate them.

Reuse these existing pieces:

- `src/evaluate.py`: canonical per-S1 macro F0.5 implementation, singleton handling, TP/FP/FN helpers.
- `src/neural_models.py::evaluate_scores`: full-query denominator, candidate oracle, country/match-count breakdown patterns.
- `src/split.py`: positive-component-preserving DSU logic. Existing 3-fold CV is for base-model evaluation; it is not the fusion/calibration/selection split.
- `src/artifact_contract.py`: candidate schema `candidate-artifact-v1`.
- `src/neural_contracts.py`: pair/prediction schema validation and artifact manifests.
- `src/candidate_io.py`: complete-S1 grouped streaming with repeat detection and atomic shard writing.
- `src/export.py`: exact TSV headers, every-S1-once validation, valid-ID checks, candidate/match subset check, streaming export.
- `src/predict_pair_model.py`: tree score production.
- `src/predict_neural.py`: resumable neural score shards with checksums/manifests.
- `src/context_features.py`: S1-centric context only; it does not provide source-record competitor features.
- `tests/test_phase_d_integration.py`: current Phase-D smoke coverage.
- `reports/experiments.csv`: experiment log scaffold.
- `member-b-starter-v1/`: frozen query/truth handoff and baseline oracle report.

Important current limitations:

- The committed baseline report contains no verified real-data model score. Do not invent one.
- Member B's bootstrap candidate sample is complete by S1 but explicitly incomplete by S2/S3 source record. It is valid for join/evaluation smoke tests, not for source-level margins or ownership.
- Current candidate/streaming grouping is S1-centric. Ownership requires complete `(candidate_source, candidate_entity_id)` groups.
- Current tree score output drops `candidate_source`; D must restore a 3-part pair key before fusion.
- Current streaming export assumes one complete score artifact over the candidate artifact. It does not model selective neural plus tree-only routes.
- No fusion model, route-aware calibration, ownership audit/selector, expected-F0.5 decoder, or unified raw/post-gate evaluation command exists.
- `lightgbm` is not a core dependency. Default fusion must use scikit-learn logistic regression unless LightGBM is deliberately added after data-size validation.

Canonical pair key for all new D code:

`(candidate_source, candidate_entity_id, source1_entity_id)`

Never join scores by row order.

---

## 1. Freeze contracts before fitting anything

### 1.1 Add `src/member_d_contracts.py`

Define constants and validators for:

- `PAIR_KEY = ["candidate_source", "candidate_entity_id", "source1_entity_id"]`
- score columns:
  - tree: `tree_score`
  - neural: `neural_score`
  - calibrated/fused: `match_probability`
- route columns:
  - `neural_requested`
  - `has_neural_score`
  - `decision_route` in `{"neural_fusion", "tree_only"}`
- ownership group key:
  - `["candidate_source", "candidate_entity_id"]`
- required provenance:
  - candidate version
  - tree model/version
  - neural checkpoint/version
  - fusion version
  - calibration version
  - decoder version
  - query split hash

Validation rules:

- pair key unique
- IDs non-empty strings
- `candidate_source` must agree with candidate ID prefix
- finite scores in expected range
- a missing neural score is valid only when `neural_requested == 0`
- `neural_requested == 1` with no neural score is a hard error
- tree score is required for every final decision pair
- no labels/truth columns may enter production score artifacts

### 1.2 Update `src/predict_pair_model.py`

Preserve `candidate_source` in tree score outputs.

Output columns must become:

`source1_entity_id, candidate_entity_id, candidate_source, score`

Then normalize to `tree_score` in D's join layer. If old inputs lack `candidate_source`, derive it only from validated `S2-`/`S3-` prefixes and fail on any ambiguous/invalid ID.

### 1.3 Keep neural score contract unchanged

`src/predict_neural.py` already emits the required 3-part key plus `score`. D renames it to `neural_score` only after validation.

---

## 2. Create clean D-only fitting partitions

Do not reuse base-model 3-fold OOF as fusion fitting data.

### 2.1 Add `src/member_d_split.py`

Use the frozen threshold query set from `member-b-starter-v1/threshold_queries.tsv` and its rows from `evaluation_truth.tsv`.

Build connected components over:

- S1 query ID
- every gold S2/S3 matched ID

This prevents a source record linked in truth from crossing D partitions.

Create three deterministic, component-preserving partitions:

- `fusion_fit`
- `calibration`
- `selection`

Recommended default proportions: 40/30/30 by S1 count, balanced greedily by country, singleton/non-singleton, and total gold edges. Make proportions configurable.

Do not use `final_queries.tsv` for model/threshold/decoder choice. Final remains comparison-only because its aggregate result has already been inspected.

Outputs:

- `configs/member_d/selection_split.tsv` with `source1_entity_id, member_d_partition`
- `configs/member_d/selection_split_manifest.json` with seed, component hash, source file hashes, counts by country/singleton, and partition proportions

### 2.2 Publish exclusion IDs for A and C

Also write:

- `configs/member_d/base_model_exclusion_ids.tsv`

It must contain every S1/S2/S3 entity present in `fusion_fit`, `calibration`, or `selection` components.

A and C must not fit their base models on these entities.

Add an assertion helper that rejects a supplied base-model training manifest if it contains any excluded entity.

---

## 3. Implement one unified evaluation command

### 3.1 Add `src/evaluate_pipeline.py`

Target command:

`python -m src.evaluate_pipeline --config configs/member_d/evaluate_threshold.json`

Inputs in config:

- query manifest or query TSV
- ground truth TSV
- S1 metadata with country
- raw candidate shards
- optional post-gate candidate shards
- optional scored/final pair shards
- optional final prediction TSV
- source registry paths

Always evaluate over the full query manifest, including zero-candidate queries.

### 3.2 Metrics to produce

For raw candidates and post-gate candidates:

- candidate pair count
- mean/p95/max candidates per S1
- zero-candidate query count
- gold edge count
- retrieved gold edge count
- missed gold edge count
- raw/post-gate pair recall
- raw/post-gate oracle macro F0.5
- non-singleton queries with zero retrieved positives
- queries with partial positive misses

For scored/final predictions:

- macro F0.5
- singleton accuracy
- non-singleton macro F0.5
- global TP/FP/FN edge counts
- pair precision
- pair recall
- predicted edge count
- predicted-nonempty S1 count

Breakdowns:

- by S1 country: all query-level metrics
- by candidate source S2/S3: edge counts, pair precision/recall, oracle recall, predicted count
- by route `neural_fusion` / `tree_only` when route data exists

Use `src.evaluate.evaluate_predictions` as the only macro-F0.5 implementation.

Write:

- machine-readable JSON
- one compact TSV/CSV row for `reports/experiments.csv`
- concise stdout summary

Do not report synthetic fixture scores as model quality.

---

## 4. Audit and implement the ownership constraint

### 4.1 Add `src/ownership.py`

First implement:

`audit_gold_ownership(ground_truth) -> report`

For every gold S2/S3 record, collect owning S1 IDs using the composite source-record key.

Report:

- total unique S2/S3 gold records
- records with zero/one/multiple S1 owners
- violation count
- first N violating examples

Ownership may be enabled only when the violation count is zero, or when an explicit contest rule says how multi-owner truth should be handled.

Never constrain an S1 to one match.

### 4.2 Add complete source-group streaming

Current `candidate_io.py` only guarantees complete S1 groups.

Implement either:

- a generalized `iter_complete_groups(..., group_columns=...)`, or
- a dedicated `src/source_group_io.py`

Required source group:

`(candidate_source, candidate_entity_id)`

It must:

- carry a group across shard boundaries
- reject a group that reappears after completion
- validate identical schemas across fragments
- fail on missing/incomplete shard manifests
- allow deterministic source-group ordering

Do not compute ownership or competitor margins from Member B's bootstrap sample because it is not source-complete.

### 4.3 Best-owner selector

Implement:

`apply_best_owner(frame, probability_col, threshold)`

For each complete source-record group:

1. Keep only candidate S1 owners with probability >= threshold.
2. If none remain, output unmatched.
3. Otherwise choose exactly one S1 owner.
4. Tie-break deterministically by:
   - higher probability
   - then higher tree score
   - then lexicographically smaller S1 ID

Output one row per retained source record with selected S1 or unmatched.

Ownership is many-to-one from source records to S1s; an S1 may receive any number of S2/S3 records.

---

## 5. Build strict ID-based score assembly

### 5.1 Add `src/score_join.py`

Inputs:

- decision candidate shards
- tree score shards
- neural score shards
- route/gate manifest

Join on the 3-part pair key only.

Output columns include:

- pair key
- `tree_score`
- `neural_score` nullable
- `neural_requested`
- `has_neural_score`
- `decision_route`
- selected contradiction features from C
- source-level competition features
- provenance/version fields

Hard failures:

- duplicate pair key in any input
- extra score key not in decision candidates
- missing tree score
- missing neural score when requested
- duplicate shard coverage
- missing shard declared by manifest
- schema/version mismatch

A deliberately tree-only pair must carry `has_neural_score=0`; never fill missing neural with 0.

### 5.2 Compute complete source competitor features

On complete source-record groups, add at minimum:

- `source_competitor_count`
- `tree_source_rank`
- `tree_gap_to_best`
- `tree_best_second_gap`
- `tree_near_tie_count`
- `is_tree_best_owner`

Use all candidate S1 competitors for that source record, not only neural-scored survivors.

Only add neural competition features if the whole compared set is explicitly known; otherwise do not present partial neural ranks as complete-source features.

---

## 6. Fit fusion and calibrate routes

### 6.1 Add `src/fusion.py`

Use only `fusion_fit` rows.

Default model: regularized `sklearn.linear_model.LogisticRegression`.

Explicit feature allowlist. Start with:

- tree score or clipped tree logit
- neural logit when available
- contradiction features from C
- complete source-level tree competitor features
- source indicator S2/S3
- route indicator

Do not auto-discover numeric columns.

Do not include:

- label-derived counts
- fold IDs
- truth-derived provenance
- `positive_injected_for_training`
- query partition labels

Fit two routes when tree-only rows exist:

1. neural-fusion route
2. tree-only route

Never apply a neural-trained fusion model to rows with no neural score.

Persist:

- model(s)
- exact feature list/order
- clipping/logit transform constants
- training row counts
- class counts
- source hashes
- git commit
- model version

Optional shallow LightGBM is allowed only if:

- enough `fusion_fit` rows exist,
- dependency is explicitly added/pinned,
- it improves held-out selection after calibration,
- production cost remains acceptable.

Do not make LightGBM required for the first complete path.

### 6.2 Add `src/calibration.py`

Use only the `calibration` partition.

Calibrate each route separately.

Preferred order:

1. Platt/logistic calibration
2. isotonic only if calibration sample size is clearly sufficient and validated

Persist route-specific calibrators and diagnostics.

Output a single `match_probability` column plus `decision_route`.

Test-prior correction stays disabled. Add no unlabeled-France prior fitting.

---

## 7. Implement and compare final decoders

### 7.1 Add `src/decoder.py`

Implement these decoders behind one interface.

#### A. Calibrated threshold

Per pair:

`match_probability >= threshold`

Threshold selected only on `selection`.

Empty S1 predictions must remain possible.

#### B. Threshold + best-owner

Apply threshold, then source-record best-owner selection, then regroup surviving edges by S1.

Only enable if the gold ownership audit passed.

#### C. Optional expected-F0.5 set decoder

Run only after A/B work.

For an S1 with predicted set size `k`, true-set size `M`, and true positives `TP`:

`F0.5 = 5 * TP / (4 * k + M)`

Requirements:

- explicitly evaluate the empty-set case
- under independent Bernoulli candidate probabilities, empty utility is `P(M=0)`
- compute expectation over candidate truth states or an equivalent Poisson-binomial DP
- do not use ratio-of-expectations as "exact expected F0.5"
- document that candidate independence and missing unretrieved positives make this approximate
- keep runtime bounded; if groups are too large, restrict this experiment to the post-gate candidate set or skip it

### 7.2 Selection rule

On `selection`, compare:

- calibrated threshold
- threshold + best-owner
- expected-F0.5 only if implemented

Select highest macro F0.5. On exact metric ties, choose the simpler decoder in the order above.

Freeze threshold/ownership/decoder config after this comparison.

---

## 8. Regroup safely between ownership and S1 decoding

Ownership needs complete source-record groups. Final set decoding needs complete S1 groups.

Pipeline order:

1. Candidate generation/gate complete.
2. Join tree/neural scores by ID.
3. Repartition/join into complete source-record groups.
4. Compute source competition.
5. Fusion + route calibration.
6. Apply source ownership if selected.
7. Write retained edge shards.
8. Repartition retained edges by S1.
9. Run S1 decoder only after all contributing source shards are complete.
10. Export.

Do not finalize an S1 set while source shards that may assign additional records to that S1 are still pending.

---

## 9. Make export route-aware

### 9.1 Define the final decision-pair artifact

Create a versioned artifact representing every pair that reached an actual final matching decision route.

It must include:

- neural-fusion pairs
- validated tree-only pairs

It must exclude pairs removed only by upstream blocking/gating if the contest definition says those pairs did not reach the matching scorer.

Use this decision-pair artifact, not "neural scores only", as the source of `candidate_pairs.tsv`.

### 9.2 Extend `src/export.py`

Keep current exact schemas:

- `matching_results.tsv`: `source1_entity_id, matched_entity_ids`
- `candidate_pairs.tsv`: `source1_entity_id, candidate_entity_ids`

Add a route-aware streaming export path that consumes:

- full test S1 manifest
- final decision-pair shards
- final selected match shards
- S2/S3 registries
- shard completion manifests

Validate before atomic rename:

- every test S1 exactly once and in test order
- valid S2/S3 IDs
- no duplicate pair
- no duplicate ID within an S1 output row
- all declared shards present and checksum-complete
- every final match is in the final candidate set
- no unknown S1
- exact headers and tab formatting
- ownership uniqueness if ownership is enabled

Run the official contest validator when it exists in the supplied resource tree.

---

## 10. Add the required tiny integration fixture

Create text fixtures under `tests/fixtures/member_d/` and tests in `tests/test_member_d_integration.py`.

Fixture must include:

- an S1 with no candidates
- S2 and S3 records sharing the same numeric suffix to prove source-qualified keys
- one source record with two competing S1 owners
- an exact probability tie
- a source group split across two shards
- a tree-only pair with intentionally missing neural score
- a neural-requested pair with missing neural score that must fail
- a missing tree score shard that must fail
- duplicate pair keys that must fail
- a final match outside candidate set that must fail

Assertions:

- ID joins are order-independent
- source groups reassemble correctly
- ties resolve deterministically
- empty queries remain empty rows in exports
- missing required shards fail before export
- tree-only route remains valid
- neural-requested/missing score fails
- final matches are a subset of candidate pairs
- source ownership chooses at most one S1 and never limits S1 match count

Synthetic fixture scores are test-only. Never append their metrics to `reports/experiments.csv`.

---

## 11. Produce a cheap-path fallback first

Implement and validate this path before optional expected-F decoding:

`candidates -> tree score -> tree-only calibration -> threshold -> optional ownership -> export`

Version it independently, e.g.:

- `artifacts/member_d/fallback_tree_only_v1/`
- `output/member_d/fallback_tree_only_v1/`

Evaluate its full behavior with the same unified evaluation command.

If neural scoring is incomplete or exceeds budget, this becomes the recoverable submission path.

Do not silently switch to fallback outputs. Each route must have its own manifest, metrics, decoder config, and output directory.

---

## 12. Runtime and resource budget

Add `src/runtime_budget.py` or a small equivalent command.

Inputs per B/C/A stage:

- measured rows/pairs per second
- input row count
- projected runtime
- peak RSS
- disk read/write volume
- artifact size
- CPU worker count
- GPU count if relevant

Output:

- stage ETA
- critical-path ETA
- aggregate disk requirement
- peak concurrent memory estimate
- latest safe start time before the final 4-hour reserve

Do not hardcode 64 GiB. Member B's current handoff reports a 13.6 GiB cgroup on one machine, so resource limits must be configured from the active production host.

Add one simple concurrency guard for high-memory/high-I/O CPU stages: refuse to launch when configured concurrent peak memory or worker count exceeds the production budget.

Cut optional experiments if the projected complete path violates the recovery reserve.

---

## 13. Reporting and artifact layout

Recommended layout:

```
configs/member_d/
  selection_split.tsv
  selection_split_manifest.json
  base_model_exclusion_ids.tsv
  evaluate_threshold.json
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

reports/member_d/
  baseline.json
  comparison.json
  comparison.tsv
  ownership_audit.json
  runtime_budget.json

output/member_d/<version>/
  matching_results.tsv
  candidate_pairs.tsv
  export_manifest.json
```

Comparison table rows:

- existing/reproduced baseline when keyed predictions are actually available
- raw retrieval oracle
- post-gate oracle
- tree-only calibrated
- neural fusion calibrated
- + ownership
- + expected-F0.5 if tested
- fallback tree-only

Columns:

- candidate version
- model versions
- decoder
- macro F0.5
- singleton accuracy
- pair precision/recall
- raw/post-gate oracle
- candidate counts
- US/India metrics
- S2/S3 edge metrics
- runtime
- peak RAM
- artifact path

Unavailable values stay `N/A`; never fabricate zeros.

---

## 14. Implementation order

1. Freeze contracts and 3-part pair key.
2. Add D-only component split and A/C exclusion list.
3. Build unified evaluator.
4. Add gold ownership audit.
5. Generalize complete grouping for source-record groups.
6. Make tree scores source-qualified.
7. Implement strict score joins and route flags.
8. Add source competitor features.
9. Implement tree-only fallback calibration + threshold + export.
10. Add fusion and separate tree-only/neural route calibration.
11. Add best-owner path and compare on selection.
12. Optionally add expected-F0.5 decoder.
13. Extend route-aware export and validator.
14. Run tiny fixture tests.
15. Run frozen threshold comparison.
16. Freeze fusion/calibration/decoder.
17. Start bulk production scoring.
18. Stop experiments by hour 20; finish shards, regroup, export, validate.
19. Reserve the final four hours only for assembly, validation, submission handoff, and recovery.

---

## 15. Acceptance checks

Member D is done only when all are true:

- one command reproduces all requested evaluation metrics from keyed artifacts
- evaluation denominator includes every query, including empty-query cases
- D-only fusion/calibration/selection partitions are component-clean and hashed
- A/C training can be checked against the exclusion IDs
- ownership truth audit result is recorded before ownership is enabled
- no source-level feature or ownership decision uses an incomplete source group
- all score joins use the 3-part ID key
- missing neural scores are route-explicit
- tree-only route is calibrated/validated separately
- simplest winning decoder is frozen on selection only
- empty predictions remain possible
- expected-F decoder, if present, handles the exact `5*TP/(4*k+M)` denominator and empty case
- test-prior correction remains off unless separately justified
- final candidate export contains all actual decision routes, including tree-only
- every S1 appears once
- all IDs are valid
- no duplicate pairs
- no missing/incomplete shards
- final matches are a subset of final candidates
- official validator passes when available
- comparison report and runtime budget are versioned
- a complete tree-only fallback exists and is independently evaluated
- final outputs are recoverable from manifests/checksums without rerunning completed scoring
