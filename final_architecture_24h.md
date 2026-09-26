# Final architecture and 24-hour execution plan

## Decision

Keep the trained mDeBERTa. Build a source-centric hybrid pipeline around it:

**S2/S3 record → candidate S1 records → LightGBM filter → selective mDeBERTa → calibrated fusion → one S1 owner or none → per-S1 match set.**

Retain successful existing retrieval channels as recovery channels. Stop new ByT5 training. Do not start three neural folds or replace the working matcher with an unidentified IBM model. The other team's summary motivates experiments; it does not establish which changes will reproduce their score.

The target is 0.99 test macro F0.5. This plan cannot guarantee it. At the reported threshold oracle of 0.987343 it is impossible; at the final oracle of 0.990332 almost all remaining candidate matches would have to be classified perfectly. Aim for roughly 0.998–0.999 retrieval oracle, with minimal loss through filtering, while reducing matcher errors. These are engineering targets, not predicted results.

## 1. Establish a common, trustworthy benchmark — first 2 hours

- ByT5's reported score uses threshold; mDeBERTa's uses final. Compare checkpoints on identical query IDs, candidate IDs and gold denominators before selecting or blending them.
- Record raw retrieval oracle, post-filter oracle, macro F0.5, non-singleton F0.5, singleton accuracy, pair precision/recall, zero-positive retrieval rate and candidate counts. Break down by country and source.
- Verify each S2/S3 ID has at most one gold S1 owner across the complete training truth before enforcing ownership. An S1 may have many S2/S3 matches. Use composite source/record IDs if IDs are not globally unique.
- Inspect the largest query-level F0.5 losses, including false positives and partial retrieval failures. Count error types: wrong house/unit number, extra branch/location term, legal-form change, transliteration, weak address, wrong competing S1, missing positive. Do not assume the other team's fake-generation pattern is identical to yours.
- Confirm the actual checkpoints, training steps, token truncation and score orientation. A pilot configuration is not evidence of how long a submitted checkpoint trained.
- Keep final out of feature/threshold iteration. Since its aggregate results have already been inspected, describe it as a limited holdout, not an untouched test.

Deliver one baseline report and a fixed evaluation command. Avoid a new experiment framework.

## 2. Retrieval: make competing S1 records visible

### Production direction

Index the deduplicated S1 reference records. Query that index with every S2/S3 record. Retrieve within country where training truth supports country consistency; provide a fallback for missing or inconsistent country values.

Use the existing lexical retrieval implementation if it can support this direction efficiently. Add a complementary dense channel only after a short recall/throughput pilot. A practical small model to pilot is `ibm-granite/granite-embedding-97m-multilingual-r2` (Apache-2.0, 384-dimensional vectors). Pin the model revision. This is a proposed retrieval encoder, not an identification of the other team's pair classifier.

Encode original name + address + country, using the documented pooling/normalization. Start with a short sequence length justified by observed text lengths; measure truncation. Build country-specific FAISS indexes over S1 and query in batches. Avoid a full S2/S3 × S1 similarity matrix. Use approximate search only after measuring its loss relative to exact search on a sample.

Pilot top-20 per channel and top-50 only for ambiguous records. Union with successful existing lexical candidates; deduplicate pairs. Numbers are starting points, not unconditional caps. Preserve raw strings and add normalized/core-name views instead of destructively removing legal forms or digits.

If pretrained dense retrieval adds little, stop it by hour 4. A short train-only contrastive fine-tune is allowed only if the pilot shows a clear recoverable gap and fits the schedule; never hold up the working lexical route waiting for it.

### Evaluate the right ceiling

For a cheap positive-recovery diagnostic, query held-out gold-matched S2/S3 endpoints against the FULL S1 background. This measures retrieval hits and oracle coverage; it does not measure false positives or complete matching F0.5. Full matching evaluation must include unmatched/background S2/S3 records that can retrieve evaluation S1s.

Measure oracle after every pruning stage, including the LightGBM top-k gate. A raw 0.999 oracle is useless if top-3 pruning reduces it to 0.985. Set a provisional post-filter loss budget of 0.0005 macro F0.5 and widen the gate if it fails and runtime permits.

Train-only transliteration dictionaries may be an additional view if error inspection supports them. Learn reusable token mappings across multiple distinct businesses; do not use held-out identity mappings or outside lookups. France has no labeled oracle in this dataset: synthetic perturbation recovery is a robustness diagnostic, not France accuracy.

## 3. LightGBM: distinguish close look-alikes

Reuse `src/features.py`, the streaming/source-store path, and `src/train_pair_model.py`; avoid materializing all raw tables and all pairs in pandas. Cache per-record normalization. Replace Python `difflib.SequenceMatcher` in high-volume paths with compiled string operations such as RapidFuzz, retraining models when feature definitions change.

Use an explicit feature allowlist. Exclude labels, gold match counts, fold IDs, truth-derived provenance and `positive_injected_for_training`. Numeric-column autodiscovery is unsafe for these shards.

Prioritize these feature groups rather than trying to reproduce exactly 56 features:

| Group | Useful signals |
|---|---|
| Name | Raw/core-name similarities; directional missing/extra tokens; rare-token disagreement; length ratio; initials; script |
| Legal form and modifiers | Separate legal-form agreement; extra branch/location tokens; agreement of the remaining name |
| Address | Token/character similarity; street agreement; house, unit and postcode agreement separately; missing-field flags |
| Numbers | House-number mismatch and distance; unit mismatch; number-role confidence; do not let a matching postcode hide a different house number |
| Joint evidence | Strong name with contradictory address; weak name with strong address; name/commonness interactions |
| Competition | Rank, best/second-best gap, near-tie count and competing S1 scores for the SAME S2/S3 record |

The current context-feature path groups by S1. Add the opposite grouping: all candidate S1 owners of a source record. Compute features over complete source groups, not partial shards or only the sampled S1 queries.

Train on actual retrieved hard negatives plus training positives. Positive injection is acceptable for training only, never retrieval evaluation. If negatives are sampled or class weights used, calibrate on the natural retained-candidate distribution afterward.

Train a first LightGBM to rank/filter. If rich features on the full candidate union are too slow, use a cheap feature subset for the first gate and compute richer features on survivors. Treat this as another measured recall gate.

## 4. Selective neural scoring and fusion

Use mDeBERTa as the main pair cross-encoder. Start with the best three S1 candidates PER S2/S3 record from LightGBM. Expand to five or more for near ties when the measured oracle requires it. Highly confident tree-only decisions may bypass neural scoring only after their precision/recall has been validated.

Use mixed precision where supported, dynamic padding, length-bucketed batches, batched tokenization, and streaming writes. Benchmark real text and storage. Do not change sequence length or quantize without checking the resulting validation loss.

Train a small fusion model using LightGBM score, mDeBERTa score, contradiction features, source-level competition and a flag indicating whether neural scoring ran. A shallow LightGBM is suitable with enough clean held-out examples; use regularized logistic fusion if data is scarce. Missing neural scores are missing, not zero probabilities.

Fusion training predictions must come from examples unseen by the base models. Prefer unused held-out training components. If none exist, split the threshold set by entity components into fusion-fit and threshold/calibration subsets, accepting the smaller sample. Never train fusion on in-sample neural training predictions and call the result OOF. Cheap tree OOF is useful; three new neural folds are outside this deadline.

Only add ByT5 if cached/aligned scores show a meaningful held-out improvement and its full-test inference cost fits. Do not average the models just because both exist. A short mDeBERTa continuation on newly retrieved hard negatives is optional, with a hard checkpoint deadline at hour 10 and an unchanged holdout.

## 5. Ownership, calibration and final sets

Compare three ablations: calibrated threshold alone; threshold plus source ownership; ownership plus expected-F0.5 set selection.

If gold supports unique ownership, each S2/S3 may select one S1 or remain unmatched. Do not force an assignment. Do not restrict an S1 to one match. Simple best-owner selection is a practical heuristic, not a proof of globally optimal constrained macro F0.5.

Calibrate probabilities on held-out, naturally distributed pipeline outputs. Optimize the actual query-level macro metric, including empty predictions. For selected-set size k, true-set size M and true positives TP:

`F0.5 = 5 * TP / (4 * k + M)`

An empty predicted set has expected utility `P(M = 0)`. A probability-based decoder can compare candidate prefixes and the empty set using a Poisson-binomial calculation under conditional independence. That independence assumption and missing candidates make this an approximation to the real task. Do not substitute a ratio of expectations and label it exact expected F0.5. Keep a tuned threshold decoder if the more complex decoder does not improve held-out macro F0.5.

Test prior correction is OFF by default. A changed score histogram alone does not prove more negatives. Enable country-specific correction only after a label-shift stress test supports calibrated estimates and stable class-conditional behavior. France's new domain makes those assumptions particularly uncertain; do not aggressively infer its fake rate from scores alone.

## 6. Runtime and memory are acceptance criteria

Use the actual local row counts. Using the previously audited test sizes, approximately 9.97M S2/S3 records × top-20 means 199.4M cheap pairs; top-three neural scoring still means 29.9M neural pairs. Finishing those in 10 hours requires about 831 pairs/second aggregate, excluding upstream work. Reducing to 10M neural pairs requires about 278/second. GPU memory capacity alone does not establish either throughput.

In the first two hours benchmark 100k record embeddings, 100k representative feature pairs and 10k neural pairs. Include reads, tokenization and writes. Estimate full encode/index/search/feature/scoring times separately and allow overlap only where resources genuinely permit it. Reserve four hours for final assembly, validation and recovery.

For 64GB RAM / 400GB disk: stream batches, use numeric ID maps, store embeddings once, store numeric pair features without duplicated text, and checkpoint completed shards. A 1.73M × 384 float32 S1 embedding array is about 2.66GB before index overhead. This does not include all other arrays or worker copies. Limit workers to measured memory and disk throughput. Monitor peak RSS and free disk; avoid swapping and duplicate full intermediate exports. Do not delete baseline artifacts to make room without preserving a recoverable copy.

Assume two GPUs only if actually available: one handles retrieval encoding/search early, the other handles matcher experiments and then inference. On one GPU, serialize these jobs and cut optional retriever fine-tuning and ByT5 first. If a measured full run cannot fit, reduce neural work through validated tree gates before spending more time training.

## 7. Four parallel workstreams and deadlines

| Time | Retrieval owner | Feature/tree owner | Neural/fusion owner | Evaluation/integration owner |
|---|---|---|---|---|
| 0–2h | Full-background recall and throughput pilot | Error taxonomy, feature allowlist, fast feature benchmark | Align checkpoint predictions; inference benchmark | Freeze IDs/splits/contracts; ownership truth audit; end-to-end budget |
| 2–6h | Select retrieval union; start full-test preprocessing as soon as fixed | Train LightGBM and test hard-negative features | Calibrate existing mDeBERTa; fit clean fusion | Compare stage-by-stage oracle/F0.5; build streaming export |
| 6–10h | Finalize retrieval and production shards | Freeze gate with measured pruning loss | Freeze neural/fusion checkpoints by hour 10 | Select decoder; validate a complete small end-to-end run |
| 10–20h | Finish retrieval and unblock downstream batches | Stream feature/gate scoring | Bulk inference; no architecture experiments | Assemble outputs incrementally; validate coverage; prepare fallback |
| 20–24h | Repair failed shards only | Repair failed shards only | Finish inference only | Final validation, submission and recovery buffer |

Start a complete cheap-path fallback early enough to finish, ideally by hour 12. This means actual full-test tree predictions and valid exports, not a sample. If projected throughput cannot support that schedule, cut experiments immediately. Never wait until hour 20 to discover that inference takes another day.

## 8. Required handoff and stopping rules

Use one versioned pair schema: source name, source record ID, S1 ID, retrieval channels/scores, feature version, model version, scores and decision route. Keep complete source groups together for ranking/ownership. Join by IDs, never row order.

Deliver:

1. One reproducible comparison table: baseline; retrieval union; post-filter oracle; tree-only; +mDeBERTa; +ownership; +decoder. Include timings and country metrics.
2. Frozen preprocessing, feature list, model checkpoints, calibration and selection configuration.
3. Resumable full-test predictions and exact output manifests.
4. Both required contest files. Candidate-pair export must reflect the final pair set reaching the ML matching decision, including validated tree-only routes, and exclude candidates discarded solely by blocking. Final matches must be a subset. Follow the supplied contest schema exactly.
5. Validator results: every S1 once; valid source IDs; no duplicate pairs; required formatting; output subset relation; ownership if enabled; no missing shards.

Do not spend the deadline reproducing an exact feature count, training a 300M model from scratch, adding an unproven third matcher, or tuning on France synthetic probes as though they were labels. Retain any extra stage only when its measured gain and production cost justify it. If 0.99 remains unsupported, report the gap honestly and submit the best completed validated system.

## Primary references

- IBM small multilingual encoder and license: https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2
- IBM embedding implementations: https://github.com/ibm-granite/granite-embedding-models
- FAISS GPU behavior: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
- Expected F-measure decision rules: https://proceedings.mlr.press/v28/dembczynski13.pdf
- Assumptions and calibration for label-shift correction: https://proceedings.mlr.press/v119/alexandari20a.html

These sources support individual methods. None demonstrates a 0.99 score on this competition. Repository observations refer to main commit `9e0664fdc146b38c19dd14503a6e86a2c1142dd8`; adapt to newer local changes.
