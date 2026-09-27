# Four-teammate execution plan: final 24 hours

Use this alongside `final_architecture_24h.md`. This document assigns ownership and defines handoffs. Give everyone the general instructions and their member section. Each section is suitable to paste into Codex with the repository available.

## Assign the roles this way

| Member | Existing assets | Owns | Main compute |
|---|---|---|---|
| A | Working mDeBERTa and checkpoint | Neural scoring and inference throughput | Matcher GPU |
| B | Existing preprocessing code/data | Shared records, retrieval, candidate shards | 64GB CPU; retrieval GPU if available |
| C | No useful artifacts | Fast features, first LightGBM, candidate filtering | CPU |
| D | No useful artifacts | Evaluation, fusion, calibration, ownership, final exports and integration | CPU |

A does not need to rebuild preprocessing. B does not need to train the matcher. C and D start immediately from existing small artifacts while B improves retrieval.

Final flow: **B produces candidates → C scores/filters → A scores selected pairs → D combines scores and produces matches.** C's validated tree-only decisions also go directly to D.

The 0.99 target is aspirational. Improving the retrieval ceiling and closing the matcher gap are separate requirements. Keep the best completed system even if the target is not reached.

## General instructions for everyone

### First 30 minutes: exchange the minimum useful package

- B publishes the current train/threshold/final query manifests, a small labeled training pair shard, a threshold sample, their referenced record text, and the full truth counts for those evaluation queries. Existing pair shards can bootstrap work before reverse retrieval is ready.
- A publishes keyed threshold predictions from the current checkpoint, or starts producing them immediately. Include exactly which query components were used for neural training. Existing final predictions can reproduce the reported baseline but must not become tuning data.
- D freezes split assignments and creates the shared contract below. Identify clean examples for fusion fitting and separate calibration/selection. Prefer unused components; otherwise divide threshold by components. Do not use final for repeated tuning.
- C runs feature extraction and a small tree fit on B's existing sample. Do not wait for the new index.

Only B needs the large retrieval indexes. C needs pair shards and access to their record text. A needs selected pairs and text. D needs keyed scores, selected feature columns, manifests and evaluation truth; D does not need GPU weights or indexes.

### Shared repository and file ownership

Use one shared Git repository with four branches: `team-a-neural`, `team-b-retrieval`, `team-c-tree`, `team-d-integration`. D integrates tested changes into the agreed release branch at the hour-2, hour-6 and hour-10 checkpoints. Do not put datasets, indexes or checkpoints in Git.

Use one shared artifact location, or transfer completed immutable shards to an agreed location. Publish paths plus checksums; do not copy all indexes onto every machine. Each producer writes temporary files and publishes a shard only after closing and validating it.

Each member owns their modules and config. Coordinate changes to shared loaders/schema before merging. Implement small adapters around existing interfaces instead of rewriting the repository.

### Shared contracts — agree once, then freeze

These are logical field names; D may map existing column names once. Preserve original IDs as strings, including leading zeros. The unique pair key is `(source, source_record_id, s1_id)`. Never join by row position.

| Artifact | Producer → consumer | Required content |
|---|---|---|
| Query manifest | B → all | S1 ID, split, country; evaluation truth/counts stored separately from model inputs |
| Records | B → A/C | Source, ID, original name/address/country; versioned normalized fields if provided |
| Candidate shards | B → C | Pair key, retrieval channel scores/ranks, candidate version |
| Tree scores and routes | C → A/D | Pair key, tree score, useful feature columns, source-level rank/gap, route: neural/tree-only/discard |
| Neural scores | A → D | Pair key, raw logit, neural-scored flag, checkpoint ID |
| Final scores/decisions | D | Pair key, calibrated score, decision route, chosen owner, selected flag |

Keep labels in separate keyed training/evaluation tables or enforce an explicit feature allowlist. Never feed gold counts, positive-injection flags, split/fold IDs or truth-derived provenance to a model.

Every shard manifest records schema version, input versions, split, shard ID, row count and completion state. Use one deterministic partition of `(source, source_record_id)` so all competing S1s for a messy record are available together. If existing shards use another layout, B repartitions before competition features. D performs a separate regrouping by S1 for final set selection.

Record groups must be complete before ranks, margins or ownership are finalized. Appending a retrieval channel after scoring invalidates affected groups and downstream scores. Do not mix candidate or model versions within a submission.

## Member A — mDeBERTa owner

**Your goal:** provide reliable, fast neural scores for the selected pairs. Preserve the working checkpoint.

### Start immediately

1. Record checkpoint path, revision, training component IDs, input formatting, max length and score interpretation.
2. Score B's threshold sample with the existing checkpoint. Write predictions keyed by the shared pair key, not a standalone score array.
3. Benchmark 10,000 representative pairs including text loading, tokenization, inference and output writes. Give D measured pairs/second, peak GPU memory and projected full-run time.
4. Make a resumable scorer that consumes C's `route=neural` shards. Use mixed precision where supported, dynamic padding and length-bucketed batches. Keep the trained input format unchanged unless validation supports a change.

### Once C's first gate is ready

- Score the complete selected validation pairs, including new retrieval candidates. Reuse cached scores only when pair text and model/input versions match.
- Default to the existing checkpoint. A short continuation on new training hard negatives is optional only if production inference still fits; freeze the checkpoint by hour 10.
- Stop new ByT5 training. Existing ByT5 predictions are only an optional comparison on the same pairs; adding it to production requires a useful held-out gain and an affordable runtime.
- Return raw logits plus metadata. D owns calibration/fusion, avoiding competing probability transformations.

### Done means

One tested streaming scorer, the selected checkpoint, aligned validation logits, a measured production budget and resumable full-test neural score shards. Missing scores have an explicit status; they are never silently filled with zero.

## Member B — preprocessing/retrieval owner

**Your goal:** deliver complete, high-recall candidate groups fast enough to finish the submission.

### Start immediately

1. Send the starter package to C and D before rebuilding anything. Expose shared read-only record lookup to A/C, or attach the required text to small bootstrap shards.
2. Confirm source sizes, available disk and usable indexes. Reuse valid artifacts. Preserve original text and IDs.
3. Measure baseline oracle using full gold denominators. Report missing positives, partial misses and zero-positive queries; do not focus exclusively on the seven complete misses.
4. Implement/pilot reverse retrieval: each S2/S3 record searches the full S1 background. Keep successful existing retrieval channels as recovery channels. Country blocking needs a fallback for missing/inconsistent country fields.

### Retrieval experiment budget

- Start with top-20 lexical candidates per source record; adapt existing code rather than inventing a new search engine.
- If a retrieval GPU is available, pilot the small multilingual dense channel in the architecture document. Benchmark encoding, indexing and search; drop it by hour 4 if its recovery/runtime does not justify deployment.
- Compare unions and larger top-k on difficult records. Aim for roughly 0.998–0.999 raw oracle, without claiming that this is guaranteed.
- Querying only known positive endpoints is acceptable for a quick recall diagnostic. Full matching validation must also include unmatched/background records that retrieve evaluation S1s, with complete competing S1 groups. Label the diagnostic honestly.
- France has no labeled oracle. Include France in production processing; synthetic robustness tests do not establish its accuracy.

### Handoffs and production

- Publish small complete candidate groups as soon as possible; C must not wait for the entire dataset.
- Freeze a candidate configuration early enough to start production by hour 6 if feasible. Later changes require explicit affected-shard rebuilds, not mixed versions.
- Stream record lookup and shards; cache per-record normalization; avoid full-table pandas joins or duplicated text in every large feature artifact.
- Maintain query manifests including S1s with zero candidates. D still needs to emit their output rows.

### Done means

Versioned candidate shards for training/evaluation/test, record access, query manifests, raw oracle reports, complete source groups and a measured full-run estimate. B owns data/index issues; do not send A or C off to rebuild the indexes.

## Member C — features and first LightGBM

**Your goal:** distinguish fake look-alikes and reduce neural workload without removing important true matches.

### Start immediately with existing data

1. Use B's bootstrap pair/text sample. Reuse the repository's streaming feature and pair-model code; do not wait for reverse retrieval.
2. Implement an explicit feature allowlist and cache record-level normalization. Use compiled string operations for high-volume comparisons; benchmark 100,000 representative pairs.
3. Train a baseline LightGBM on training components only. Exclude D's fusion/calibration components as well as final.

### Add the useful signals

- Name: raw/core-name similarity, directional extra/missing tokens, rare-token disagreement, initials and length ratio.
- Address: street agreement; house number, unit number and postcode treated separately; missing flags. A matching postcode must not hide a different house number.
- Legal form/branch terms: preserve them as features rather than deleting them. Learn their effect; do not hard-reject every mismatch.
- Joint evidence: strong name with contradictory address, and vice versa.
- Retrieval competition: ranks/gaps among candidate S1s for the same S2/S3 record.

Fit the first tree using pair features and retrieval-derived context. Compute tree-score ranks/gaps AFTER it scores complete groups; send those to D for fusion. Do not create circular features where a model requires its own final predictions as inputs.

### Build the neural gate

- Start with top-three S1 candidates PER S2/S3 record; widen near ties as evidence requires.
- Send proposed gate outputs to D to measure post-filter oracle against B's raw oracle. The provisional loss budget is at most 0.0005 macro F0.5; this is a target, not permission to hide a failure.
- A tree-only accept/reject route is allowed only after D validates it. Begin with neural routing if confidence is unproven.
- New retrieval candidates change the negative distribution: retrain/check the tree on representative training candidates and rerun the gate report.
- Keep tree scores for all routes so D can build a complete tree-only fallback.

### Done means

Fast versioned features, a saved tree model, complete grouped tree predictions, route manifests for A, feature columns for D, and a gate with measured recall loss/runtime. If CPU features dominate runtime, stage expensive features behind a measured cheap gate.

## Member D — fusion, evaluation and integration

**Your goal:** make the independent modules a valid, measured submission. You own the metric, split policy and final decision logic.

### Start immediately; no GPU artifacts required

1. Freeze schemas, splits and branch integration. Reproduce the reported baselines using existing keyed predictions when available.
2. Implement one evaluation command for raw/post-gate oracle, macro F0.5, singleton accuracy, pair precision/recall and counts, with country/source breakdowns.
3. Verify the gold ownership constraint across training truth. If supported, allow each S2/S3 record at most one S1 owner, with an unmatched option. Never limit S1 to one match.
4. Build ID-based score joins and contest export validation now. Exercise them on a tiny fixture covering empty queries, duplicate IDs across sources, ties and missing score shards. Synthetic fixture scores are for integration tests only, not reported model quality.
5. Define clean fusion-fit and calibration/selection components. Neither A nor C may train its base model on them. Do not call held-out stacking three-fold OOF.

### As C/A deliver scores

- Fit a small fusion model using tree score, neural logit, contradiction features and complete source-level competitor margins. Use regularized logistic fusion if the clean fitting set is small; shallow LightGBM only if data supports it.
- Missing neural scores need an explicit route flag. Calibrate/validate the tree-only route separately when necessary; do not extrapolate a neural-only fusion model to it.
- Compare calibrated thresholding, thresholding plus best-owner selection, and optionally expected-F0.5 set selection. Keep the simplest method that wins on the held-out selection set. Empty predictions must remain possible.
- Any expected-F decoder must handle `F0.5 = 5*TP/(4*k+M)` and the empty-set case. Independence-based probability calculations are approximations, not guaranteed global optimization.
- Leave test-prior correction off unless the earlier document's assumptions and validation checks are satisfied. Do not estimate France accuracy from unlabeled score histograms.

### Production responsibilities

- Maintain an aggregate runtime budget from B/C/A measurements. Prevent concurrent CPU jobs from exhausting the shared 64GB RAM or disk throughput.
- Assemble scores incrementally but finalize ownership only after a source group is complete, and finalize an S1 set only after all contributing source shards are accounted for.
- Produce a complete cheap-path fallback early, ideally by hour 12. If the budget cannot support this, cut optional experiments immediately.
- Export the required candidate file consistently with the actual final matching routes, including tree-only routes. Final matches must be a subset. Follow the contest's exact schema and definition of the final scored candidate set.
- Validate every S1 appears once, valid source IDs, no duplicate pairs, no missing shards and correct formatting. Reserve four hours for final assembly/submission/recovery.

### Done means

One comparison report, frozen fusion/calibration/decoder, integrated release commit, complete validated output files and a recoverable fallback. D coordinates the submission; uploading is performed by the teammate authorized to use the contest account.

## Checkpoints and dependency rules

| Deadline from start | Required shared result |
|---|---|
| 30 minutes | B's starter package available; A's checkpoint/training provenance shared; D's contracts agreed |
| 2 hours | C's first tree; A's throughput; B's retrieval pilot; D's baseline and full-run budget |
| 6 hours | New candidate sample has passed B → C → A → D; raw and post-gate oracle measured; production started where frozen |
| 10 hours | Freeze retrieval, feature/model versions and decoder; bulk scoring underway |
| 20 hours | Stop experiments; finish missing shards and assemble outputs |
| 24 hours | Validated submission completed with recovery buffer already used as needed |

If B's new retrieval is late, C/A/D continue with existing shards and swap in the new version through the same interfaces. If C's new tree is late, A preserves the baseline neural path while D evaluates a simpler gate. If neural throughput cannot finish, D uses a validated selective-neural/tree-only fallback. If fusion does not improve clean validation, retain the best calibrated simpler scorer.

Do not let fallback become a silent change in the evaluated pipeline: measure each fallback's full behavior and keep its own versioned outputs. No teammate needs to repeat another teammate's full EDA, copy every large artifact, or wait for all preprocessing to finish before starting useful work.
