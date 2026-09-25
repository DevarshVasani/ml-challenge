# Implementation Plan — ByT5 GPU Baseline

## 1. Mission

Implement the repository's **ByT5 byte-level neural matcher baseline** as a fair challenger to mDeBERTa using the exact same:

- training pair shards,
- candidate sets,
- query manifest,
- threshold-tuning split,
- untouched final-validation split,
- shared trainer,
- shared prediction writer,
- shared threshold/evaluation logic.

The baseline checkpoint is:

`google/byt5-small`

The model must be used as an **encoder-only pair classifier**:

```text
S1 record + candidate record
        |
        v
shared six-field pair serialization
        |
        v
ByT5 byte tokenizer
        |
        v
T5 encoder only
        |
        v
attention-mask-aware mean pooling
        |
        v
dropout
        |
        v
linear classifier -> 2 logits
        |
        v
softmax(logits)[:, 1] = P(match)
```

Do **not** instantiate or use a decoder for training/inference, do not generate IDs/text, and do not create a second training/evaluation framework.

---

## 2. Repository snapshot analyzed

Repository:

`DevarshVasani/ml-challenge`

Snapshot analyzed:

- branch: `main`
- current commit observed: `b5284141725763ed18ff1ba38eddd64bf8593df2`
- commit message: `READY`
- commit time: 2026-09-25
- no open PR was visible during the analysis.

This plan is intentionally tied to that repository state. If `main` moves before implementation, the agent should first diff the changed neural files listed below and adapt the plan rather than blindly overwriting newer work.

---

## 3. Current state of the neural pipeline

### 3.1 Infrastructure that already exists and should be reused

The repository already has the correct shared architecture.

#### Adapter contract

`src/neural_adapters/base.py`

Defines `BasePairAdapter` with:

- `serialize_pair(row)`
- `collate(rows)`
- `logits(batch)`
- `parameters()`
- `to_device(device)`
- `save_pretrained(output_dir)`
- `load_pretrained(output_dir)`
- optional gradient checkpointing hooks

It also already registers:

- `fake`
- `mdeberta`
- `byt5`

The ByT5 adapter type is therefore already integrated into adapter selection.

#### Shared pair schema

`src/neural_contracts.py`

The model text fields are already fixed as:

1. `name_a`
2. `address_a`
3. `country_a`
4. `name_b`
5. `address_b`
6. `country_b`

IDs, folds, labels, retrieval scores, and candidate provenance must not be serialized into model text.

#### Shared trainer

`src/train_neural.py` + `src/neural_models.py`

Already supports:

- raw pair shards,
- deterministic shard/row ordering,
- microbatching,
- gradient accumulation,
- BF16 / FP16 / FP32,
- AdamW,
- warmup + linear decay,
- gradient clipping,
- optional adapter gradient checkpointing,
- exact single-worker resume,
- checkpoint manifests,
- loss reporting,
- training pairs/sec,
- collate/forward timing,
- truncation counting,
- CUDA peak allocated memory.

This is sufficient for the ByT5 baseline. Do not create a separate ByT5 trainer.

#### Shared predictor

`src/predict_neural.py`

Already supports:

- adapter checkpoint loading,
- threshold/final pair-shard scoring,
- atomic Parquet score shards,
- resumable prediction,
- checkpoint/input identity validation,
- overall inference pairs/sec,
- collate/forward timing,
- truncation counting.

Do not create a separate ByT5 inference engine.

#### Shared evaluator

`src/evaluate_neural.py` + `src/neural_models.py`

Already supports:

- threshold-set-only threshold tuning,
- frozen-threshold evaluation on final validation,
- macro-F0.5,
- candidate oracle ceiling,
- by-country metrics,
- by-match-count metrics.

This should remain the source of truth for ByT5 evaluation.

#### Existing ByT5 configuration

`configs/neural/byt5_pilot.json`

Already matches the requested starting point:

- checkpoint: `google/byt5-small`
- max length: 512
- microbatch: 4
- gradient accumulation: 16
- effective batch size: 64
- learning rate: `5e-5`
- epochs: 1
- BF16
- CUDA
- run directory: `artifacts/neural-byt5`
- `num_workers=0`

Do not broaden the first baseline into a hyperparameter search.

---

## 4. Important gaps in the current repository

### 4.1 ByT5 adapter is only a skeleton

`src/neural_adapters/byt5.py`

Every model-specific method currently raises `NotImplementedError`.

This is the main implementation task.

### 4.2 mDeBERTa is also still a skeleton on analyzed `main`

`src/neural_adapters/mdeberta.py`

The mDeBERTa adapter is also unimplemented in the analyzed snapshot.

Implication:

The ByT5 code can be implemented and validated independently, but the final A/B comparison cannot be completed until Member B's checkpoint, predictions, threshold, evaluation metrics, and runtime measurements exist.

Do not fabricate comparison values. Build a comparison artifact that can consume Member B's outputs when available.

### 4.3 No ByT5 threshold prediction config exists

There is:

`configs/neural/predict_byt5_pilot.json`

but it scores the `final` subset.

There is an mDeBERTa threshold config:

`configs/neural/predict_mdeberta_threshold_pilot.json`

but no equivalent ByT5 file was found.

Create:

`configs/neural/predict_byt5_threshold_pilot.json`

### 4.4 Evaluation config is currently mDeBERTa-specific

`configs/neural/evaluate_pilot.json`

points to:

- `predictions-mdeberta-threshold`
- `predictions-mdeberta-final`
- `neural-evaluation-mdeberta.json`

Create a separate ByT5 config rather than mutating the mDeBERTa file:

`configs/neural/evaluate_byt5_pilot.json`

### 4.5 Singleton/non-singleton aggregate reporting is incomplete

The query manifest built in `src/prepare_neural_data.py` already stores:

- `country`
- `match_count`
- `singleton`

But `evaluate_scores()` currently groups only by:

- country,
- match count.

Add a generic `by_singleton` grouping so both model owners can report:

- singleton,
- non-singleton

from the same shared evaluator.

### 4.6 Reproducibility needs one shared fix

There are two related issues.

First, `seed_everything()` currently seeds Python and NumPy but not Torch.

Second, the adapter is constructed in `src/train_neural.py` **before** `train_neural()` calls `seed_everything()`.

That means a newly initialized classification head can differ between otherwise identical runs.

Fix this in the shared pipeline instead of adding ByT5-only seeding:

1. extend `seed_everything(seed)` to seed Torch when Torch is installed;
2. call `seed_everything(run_cfg.seed)` before first adapter construction in `train_neural.execute()`.

Do not force global deterministic CUDA algorithms unless explicitly needed; they can severely change performance or fail on unsupported kernels. Reproducibility should instead record the environment and acknowledge that GPU kernels may not be bitwise deterministic across hardware/software versions.

### 4.7 Real GPU/data execution has not been performed in the repository snapshot

The repository's GPU runbook explicitly says:

- real neural data preparation has not been run,
- real checkpoints have not been downloaded/run,
- GPU fit/throughput/memory/accuracy are unmeasured.

Therefore the AI agent must distinguish:

**code implementation** from **compute execution**.

If the required Member A artifacts are absent on the compute machine, stop and report the missing artifacts. Do not silently create alternative candidates or validation splits.

---

# 5. Required implementation changes

## 5.1 Modify `src/neural_adapters/base.py`

Add one small shared serialization utility so ByT5 and mDeBERTa cannot drift in pair formatting.

Recommended behavior:

```python
PAIR_TEXT_FIELDS = (
    "name_a",
    "address_a",
    "country_a",
    "name_b",
    "address_b",
    "country_b",
)

def serialize_pair_fields(row, missing_text=""):
    values = [
        str(row.get(field, missing_text) or missing_text)
        for field in PAIR_TEXT_FIELDS
    ]
    return " || ".join(values)
```

Rationale:

- it follows the exact existing field order;
- the fake adapter already uses `" || "` serialization;
- it avoids adding verbose field labels that consume ByT5's byte budget;
- both neural baselines can use exactly the same textual representation;
- it prevents accidental inclusion of IDs/retrieval features.

Keep `serialize_pair()` as part of the adapter interface. ByT5 should simply call this shared helper.

If Member B has already merged a different shared serializer by implementation time, use the newer shared serializer instead. Do not create a ByT5-only format.

---

## 5.2 Implement `src/neural_adapters/byt5.py`

### 5.2.1 Preserve lazy dependency loading

The module should remain importable on machines without Transformers.

Do not add top-level imports such as:

```python
from transformers import ...
```

Instead, import Torch/Transformers inside runtime-only construction helpers or `__init__`.

This preserves the current guarantee that `--help` and `--dry-run` do not initialize/download a model.

### 5.2.2 Encoder class

Prefer:

```python
from transformers import AutoTokenizer, T5EncoderModel
```

Load:

```python
tokenizer = AutoTokenizer.from_pretrained(
    config.checkpoint,
    revision=config.revision,
    use_fast=False,
)

encoder = T5EncoderModel.from_pretrained(
    config.checkpoint,
    revision=config.revision,
)
```

Why:

- Hugging Face exposes `T5EncoderModel` as the bare T5 encoder;
- `google/byt5-small` is a T5-family checkpoint with a ByT5 tokenizer;
- this avoids retaining a decoder in the trainable runtime model.

Do not use `AutoModelForSeq2SeqLM` for the classifier runtime unless a compatibility problem with the pinned Transformers version is demonstrated.

The current `google/byt5-small` config has approximately:

- `d_model = 1472`
- 12 encoder layers
- dropout rate 0.1
- byte vocabulary size 384.

Always read dimensions from `encoder.config`; do not hard-code `1472`.

### 5.2.3 Wrapper module

Create a small internal `torch.nn.Module`, for example:

```python
class _ByT5PairClassifier(nn.Module):
    def __init__(self, encoder):
        ...
```

It should own:

- `encoder`
- `nn.Dropout(encoder.config.dropout_rate)`
- `nn.Linear(encoder.config.d_model, 2)`

Store the wrapper as:

```python
self.model
```

This matters because the existing shared trainer/predictor call:

- `adapter.model.train()`
- `adapter.model.eval()`

when `model` exists.

### 5.2.4 Forward path

The encoder forward pass should receive only:

- `input_ids`
- `attention_mask`

No decoder inputs.

Then:

```python
hidden = outputs.last_hidden_state
mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

pooled = (hidden * mask).sum(dim=1)
pooled = pooled / mask.sum(dim=1).clamp_min(1.0)

logits = classifier(dropout(pooled))
```

Required invariant:

`logits.shape == [batch_size, 2]`

Class index 1 means match.

### 5.2.5 Pair serialization

`serialize_pair(row)` must call the shared six-field serializer.

Do not include:

- source entity IDs,
- candidate IDs,
- candidate source,
- folds,
- label,
- retrieval score,
- rank,
- provenance.

### 5.2.6 Tokenization and dynamic padding

`collate(rows)` must:

1. serialize each row;
2. tokenize with the pretrained ByT5 tokenizer;
3. enforce `config.max_length`;
4. dynamically pad only to the longest item in the current microbatch;
5. return tensors on `self.device`;
6. return an integer `truncated` count.

The trainer does not move arbitrary adapter batch tensors to GPU, so `collate()` must place `input_ids` and `attention_mask` on the adapter device.

### 5.2.7 One-pass truncation accounting

Avoid tokenizing every text twice during the measured baseline.

Recommended approach:

1. tokenize the batch without padding and without truncation;
2. record each original token length;
3. truncate each token-id list to `max_length`;
4. preserve the final EOS token when truncation happens;
5. call `tokenizer.pad(..., padding=True, return_tensors="pt")`;
6. set `truncated = count(original_length > max_length)`.

This gives exact per-example truncation accounting without a second full tokenizer pass.

Do not assume Unicode characters are one byte. Let the ByT5 tokenizer define actual token length.

### 5.2.8 Batch return contract

Return something similar to:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "truncated": int(...),
}
```

Optional diagnostic metadata is acceptable, but keep it off GPU when it is not needed by `logits()`.

### 5.2.9 Parameters and device movement

Implement:

```python
def parameters(self):
    return self.model.parameters()
```

and:

```python
def to_device(self, device):
    self.device = torch.device(device)
    self.model.to(self.device)
    return self
```

Do not attempt to move tokenizer state.

### 5.2.10 Gradient checkpointing

Implement:

```python
def supports_gradient_checkpointing(self) -> bool:
    return True
```

and delegate enable/disable to the encoder:

```python
self.model.encoder.gradient_checkpointing_enable()
self.model.encoder.gradient_checkpointing_disable()
```

Initial baseline:

`gradient_checkpointing = false`

Only enable it when the short GPU pilot shows it is needed for memory.

### 5.2.11 Checkpoint format

Save a self-contained adapter checkpoint under the shared:

`<run_dir>/adapter/`

Recommended layout:

```text
adapter/
  adapter_config.json
  adapter_metadata.json
  encoder/
    config.json
    model.safetensors or pytorch_model.bin
    ...
  tokenizer/
    tokenizer_config.json
    special_tokens_map.json
    ...
  classifier.pt
```

`adapter_metadata.json` should record at least:

- architecture version, e.g. `byt5-encoder-pair-v1`
- original checkpoint name
- requested revision
- resolved Hugging Face revision/commit if available
- tokenizer class
- encoder class
- `d_model`
- classifier output size = 2
- pooling = `masked_mean`
- serialization version/string separator
- max length.

`save_pretrained()` should save:

- encoder with `encoder.save_pretrained(...)`,
- tokenizer with `tokenizer.save_pretrained(...)`,
- classifier state dict,
- adapter config,
- adapter metadata.

### 5.2.12 Reload without network access

`load_pretrained(output_dir, map_location=...)` must load only the saved local files.

It must not require downloading `google/byt5-small` again.

Recommended behavior:

- read `adapter_config.json`,
- load tokenizer from `output_dir/tokenizer`,
- load `T5EncoderModel` from `output_dir/encoder`,
- recreate the classifier using the saved encoder config,
- load `classifier.pt`,
- move to `map_location`,
- return adapter.

A checkpoint round trip should work with Hugging Face offline mode enabled.

---

# 6. Shared reproducibility patch

## 6.1 Modify `src/neural_contracts.py`

Extend:

```python
seed_everything(seed)
```

to seed Torch lazily:

```python
try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except ImportError:
    pass
```

Keep Torch imported only inside the function.

## 6.2 Modify `src/train_neural.py`

Construct `NeuralTrainingConfig` before a fresh adapter is created.

Then call:

```python
seed_everything(run_cfg.seed)
```

before:

```python
adapter_cls.from_config(...)
```

The trainer can still seed again at the beginning of training.

This ensures deterministic initialization of the newly added two-class head for a fixed seed.

## 6.3 Reproducibility metadata

For each real ByT5 run, preserve:

- repository commit SHA,
- full training config,
- pair manifest SHA256,
- query manifest SHA256,
- candidate manifest identity where relevant,
- Torch version,
- Transformers version,
- CUDA runtime version,
- GPU model,
- resolved ByT5 checkpoint revision,
- training seed.

The checkpoint system already records most data identities. Add the model revision/environment record rather than building a second checkpoint system.

---

# 7. Tests to add

Create:

`tests/test_byt5_adapter.py`

The default repository test suite must still run without automatically downloading a 1.2 GB model.

Use `pytest.importorskip("torch")` / `pytest.importorskip("transformers")` for optional Transformer-specific tests.

Where possible, construct a very small local T5 encoder checkpoint in a temporary directory rather than downloading `google/byt5-small`.

Also add a separately marked real-checkpoint/GPU integration test or executable sanity script that is not part of the fast default suite.

## 7.1 Serialization test

Verify:

- fields occur in the required order;
- missing values become empty strings;
- IDs/labels/folds are absent;
- the serializer is deterministic.

## 7.2 Forward shape test

For a small local encoder fixture:

- collate 2 rows;
- call `logits`;
- assert floating dtype;
- assert shape `[2, 2]`.

## 7.3 Dynamic-padding test

Use two differently sized serialized records and assert:

- batch width equals the longest sequence in that batch, bounded by max length;
- shorter item has zero attention-mask values on padded positions.

## 7.4 Padding invariance / pooling-mask test

This is critical.

In `eval()` mode:

1. score row A by itself;
2. score row A in a batch with a much longer row B.

The logits for A should be equal within a small floating-point tolerance.

This proves padding tokens are not entering masked mean pooling.

## 7.5 Truncation counter test

Use a deliberately long UTF-8 string containing multi-byte characters.

Assert:

- the row is marked truncated;
- output length is bounded by `max_length`;
- the attention mask is valid;
- EOS handling remains valid.

Do not test truncation using Python character count.

## 7.6 Save/reload round-trip test

In eval mode:

1. score a fixed batch;
2. save adapter;
3. reload from the local saved directory;
4. score the same batch;
5. assert logits/probabilities match within tolerance.

This test must include the classifier head, not just the encoder.

## 7.7 Gradient checkpointing smoke test

If the tiny local T5 encoder supports it:

- enable,
- verify encoder reports checkpointing enabled;
- disable,
- verify disabled.

## 7.8 Shared seeding regression test

Create two fresh adapters/classification heads with the same seed initialization path and verify classifier weights match.

The test should fail under the current pre-patch behavior and pass after seeding is moved before adapter construction.

---

# 8. Tiny real-model correctness phase

This happens on a compute machine after the code tests pass.

Use the actual:

`google/byt5-small`

Do not yet run the full pilot.

## 8.1 Required checks

Confirm:

- checkpoint downloads/loads,
- tokenizer is the ByT5 tokenizer,
- model object contains only the encoder path used for forward classification,
- no decoder forward is called,
- one CUDA forward works at max length 512,
- BF16 autocast works on the selected GPU,
- gradients reach encoder and classifier,
- padding invariance passes on the real model,
- save/reload passes,
- reload works from local checkpoint files,
- truncation count is nonnegative/sensible.

## 8.2 Tiny overfit

Select a very small balanced sample from the **existing training pair shards**.

Do not create new candidate pairs.

Suggested size:

- 8–32 pairs,
- include both positive and negative examples.

For the tiny overfit only, it is acceptable to use:

- gradient accumulation 1,
- multiple passes / more steps,
- a temporary run directory.

Success criteria should be behavioral, for example:

- clear loss decrease from initial to final;
- positive examples receive higher scores than negatives;
- preferably near-perfect training separation on the tiny sample.

This is an integration test, not the reported baseline.

---

# 9. Add ByT5 threshold prediction config

Create:

`configs/neural/predict_byt5_threshold_pilot.json`

Mirror the mDeBERTa threshold config but with ByT5 checkpoint/output settings.

Expected structure:

```json
{
  "adapter_type": "byt5",
  "checkpoint": "artifacts/neural-byt5",
  "pair_manifest": "artifacts/neural-data/threshold_pairs_manifest.json",
  "pair_subdir": "threshold_pairs",
  "candidate_manifest": "artifacts/neural-candidates/manifest.json",
  "query_manifest": "artifacts/neural-data/query_manifest.json",
  "query_subset": "threshold",
  "batch_size": 16,
  "device": "cuda",
  "output_dir": "artifacts/predictions-byt5-threshold"
}
```

Do not point threshold scoring at final pairs.

---

# 10. Add ByT5 evaluation config

Create:

`configs/neural/evaluate_byt5_pilot.json`

Use:

- ByT5 threshold scores,
- ByT5 final scores,
- the same threshold/final candidate pair directories,
- the same ground truth,
- the same query manifest,
- a ByT5-specific output file.

Example targets:

```text
scores:
  artifacts/predictions-byt5-threshold
  artifacts/predictions-byt5-final

prediction manifests:
  artifacts/predictions-byt5-threshold/prediction_manifest.json
  artifacts/predictions-byt5-final/prediction_manifest.json

output:
  artifacts/neural-evaluation-byt5.json
```

## Threshold grid

Do not reuse the mDeBERTa threshold and do not assume 0.5.

Use the existing shared `tune_threshold()`.

For a serious baseline, use a sufficiently dense probability grid, preferably 0.01 increments over a practical range such as `[0.01, 0.99]`.

For fairness, Member B should use the same threshold-search resolution if their threshold has not already been frozen.

If mDeBERTa has already been evaluated under a different agreed threshold grid, do not silently change the comparison protocol; document the difference or rerun both under the same grid.

---

# 11. Add singleton/non-singleton reporting to shared evaluator

Modify `evaluate_scores()` in:

`src/neural_models.py`

The manifest already provides `singleton`.

Extend the metadata grouping to include it.

Conceptually:

```python
(
    ("country", "by_country"),
    ("match_count", "by_match_count"),
    ("singleton", "by_singleton"),
)
```

Then the final report can directly map:

- `singleton=True` -> singleton result
- `singleton=False` -> non-singleton result

Keep existing by-match-count output.

Add/extend tests in:

`tests/test_neural_infrastructure.py`

to validate the new grouping.

---

# 12. Truncation analysis

The shared trainer only needs a total truncated-pair count. The ByT5 experiment needs more detail.

Create a diagnostic script, for example:

`scripts/analyze_byt5_truncation.py`

This is not a new training/evaluation framework. It is a model-specific diagnostic.

## 12.1 Inputs

Accept:

- pair manifest or pair directory,
- tokenizer checkpoint/revision,
- max length,
- optional score directory,
- optional decision threshold,
- output JSON/CSV path.

## 12.2 Pre-training truncation report

Measure for each split of interest:

- total pairs,
- truncated pairs,
- truncation rate,
- token-length p50,
- p90,
- p95,
- p99,
- max,
- first field affected by truncation,
- whether `address_a` is partially/fully cut,
- whether `address_b` is partially/fully cut,
- truncation rate by country.

Use the same shared serialization helper as the adapter.

Do not infer ByT5 length from subword length.

## 12.3 Field-cut analysis

Because the serializer has six ordered fields, identify the point where `max_length` falls relative to cumulative serialized field boundaries.

Report at minimum:

- truncation ends inside `name_a`
- inside `address_a`
- inside `country_a`
- inside `name_b`
- inside `address_b`
- inside `country_b`.

This directly answers whether the 512-byte budget frequently removes candidate addresses.

## 12.4 False-negative correlation

After threshold-set predictions exist:

- freeze/use the current ByT5 threshold candidate for the threshold set;
- join threshold false negatives to the truncation diagnostics;
- report FN rate for truncated vs non-truncated rows/queries.

Do not inspect final validation for threshold tuning.

After final evaluation is complete and frozen, the same analysis may be run on final strictly as post-hoc error analysis.

---

# 13. Pre-compute artifact guardrail

Before any GPU training, verify the exact Member A inputs exist.

Expected artifacts include:

```text
artifacts/neural-data/query_manifest.json
artifacts/neural-data/train_pairs_manifest.json
artifacts/neural-data/train_pairs/
artifacts/neural-data/threshold_pairs_manifest.json
artifacts/neural-data/threshold_pairs/
artifacts/neural-data/final_pairs_manifest.json
artifacts/neural-data/final_pairs/
artifacts/neural-candidates/manifest.json
```

Also locate the candidate-recall/oracle report produced by the shared candidate pipeline.

Verify manifest checksums/identities where available.

If these are missing:

**STOP.**

Do not:

- generate a new validation split,
- generate a new candidate set,
- mine new negatives,
- substitute a local random subset as the reported baseline.

The tiny-overfit sanity check may sample existing training rows, but the reported pilot must use the shared manifests unchanged.

---

# 14. Short GPU pilot

Before full one-epoch training, run a bounded ByT5 GPU pilot using the same training shards.

Create a temporary config such as:

`configs/neural/byt5_short_pilot.json`

with a distinct run directory and a small `max_steps` value.

Suggested first pilot:

- checkpoint: `google/byt5-small`
- max length: 512
- BF16
- microbatch: 4
- gradient accumulation: 16
- effective batch: 64
- LR: `5e-5`
- gradient checkpointing: false
- `max_steps`: approximately 25–100
- `num_workers`: 0.

The exact max steps are diagnostic, not a scored experiment.

## 14.1 Measure

Capture:

- peak allocated VRAM,
- training pairs/sec,
- collate seconds,
- forward/loss seconds,
- truncation rate,
- loss trajectory,
- whether any OOM occurs.

The shared trainer already returns most of this.

Preserve command output, for example with shell `tee`, in the pilot artifact directory.

## 14.2 40 GB GPU test

Use a 40 GB-class GPU first if available.

Goal:

- establish the default microbatch-4 baseline without checkpointing;
- measure actual headroom.

## 14.3 24 GB viability test

Run the same short configuration on a 24 GB GPU if available.

Adjustment order if it OOMs:

1. enable gradient checkpointing;
2. if still necessary, reduce microbatch from 4 to 2 and increase accumulation from 16 to 32;
3. keep effective batch size 64;
4. keep max length 512 for the baseline.

Do not reduce sequence length merely to force a 24 GB fit, because that changes the truncation experiment.

Record every deviation.

---

# 15. Full one-epoch ByT5 pilot

After all acceptance gates pass, run the existing:

`configs/neural/byt5_pilot.json`

with only justified memory-setting changes from the short pilot.

Command:

```bash
python -m src.train_neural \
  --config configs/neural/byt5_pilot.json \
  --execute
```

Recommended operational form:

```bash
python -m src.train_neural \
  --config configs/neural/byt5_pilot.json \
  --execute \
  | tee artifacts/neural-byt5/train_result.json
```

If `tee` produces non-pure JSON because logging is later added, write a separate run log instead of treating it as machine-readable JSON.

## Freeze after training

Once the one-epoch baseline completes:

- do not continue training based on final-validation behavior;
- treat the resulting checkpoint as the baseline checkpoint;
- save exact run config/environment;
- record peak GPU memory and training throughput;
- verify checkpoint reload once more before scoring.

---

# 16. Threshold-set scoring

Run:

```bash
python -m src.predict_neural \
  --config configs/neural/predict_byt5_threshold_pilot.json \
  --execute
```

Expected output:

`artifacts/predictions-byt5-threshold/`

Verify:

- manifest completion is `complete`,
- every threshold pair has exactly one score,
- probabilities are finite and in `[0,1]`,
- no final queries are present,
- measured inference pairs/sec is captured,
- truncation rate is captured.

This is the only prediction set used to select the ByT5 threshold.

---

# 17. Threshold selection

Run the shared evaluator only after threshold predictions are complete.

The selected threshold must maximize the shared macro-F0.5 objective over the threshold subset and the configured threshold grid.

Persist:

- chosen threshold,
- threshold-set macro-F0.5,
- threshold search grid,
- query count,
- checkpoint identity,
- score manifest identity.

Once selected:

**freeze the threshold.**

Do not adjust it after looking at final validation.

The thresholding rule remains the existing independent pair rule:

```text
score >= threshold -> include candidate
```

Because every candidate is independently thresholded, this naturally permits:

- zero selected candidates,
- one selected candidate,
- multiple selected candidates.

Do not convert this into top-1 classification.

---

# 18. Final-validation scoring

Only after the threshold is frozen, run:

```bash
python -m src.predict_neural \
  --config configs/neural/predict_byt5_pilot.json \
  --execute
```

Expected output:

`artifacts/predictions-byt5-final/`

Do not use final-validation labels to:

- alter max length,
- change the threshold,
- choose another checkpoint,
- change LR,
- change epoch count.

If a pipeline bug is discovered, fix the bug and rerun the complete affected protocol for both threshold and final; document the rerun.

---

# 19. Final evaluation

Run:

```bash
python -m src.evaluate_neural \
  --config configs/neural/evaluate_byt5_pilot.json \
  --execute
```

Required ByT5 outputs:

- tuned threshold,
- final macro-F0.5,
- singleton result,
- non-singleton result,
- US result,
- India result,
- candidate oracle ceiling,
- candidate-count context,
- threshold frozen before final = true.

Interpret country labels exactly as emitted by the shared query manifest. Do not silently remap/normalize country groups only for ByT5.

---

# 20. Inference speed and projected full-test runtime

Use the measured ByT5 inference pairs/sec from real prediction runs.

Prefer the larger threshold/final scoring run over a tiny synthetic benchmark.

Calculate:

```text
projected full-test seconds =
    actual full-test candidate pair count
    / measured ByT5 inference pairs/sec
```

Report also in minutes/hours.

Use the actual test candidate-pair manifest when it exists.

If it does not yet exist, report the projection as pending rather than inventing a pair count.

Do not use mDeBERTa throughput for the ByT5 projection.

---

# 21. Error analysis

Perform error analysis after the baseline metrics are frozen.

Focus on ByT5's hypothesis: byte-level robustness to noisy text.

## 21.1 Positive cases where ByT5 may help

Inspect true positives / mDeBERTa misses involving:

- misspellings,
- punctuation differences,
- inconsistent spacing,
- accents/diacritics,
- transliteration,
- non-Latin text,
- mixed-language strings,
- unusual business names,
- compact/noisy addresses,
- numeric formatting variants.

## 21.2 Failure cases

Inspect false negatives/false positives involving:

- truncated long inputs,
- shared addresses,
- near-identical names with different numbers,
- missing names,
- missing addresses,
- singleton false positives,
- house/unit/phone-like numeric collisions,
- repeated generic business words.

## 21.3 Diagnostic output

Produce a small machine-readable table, for example:

`artifacts/neural-byt5/error_analysis.csv`

Useful columns:

- query ID
- candidate ID
- label
- ByT5 score
- threshold
- predicted class
- mDeBERTa score if available
- truncated flag
- serialized token length
- country
- match count
- name/address text fields
- heuristic category flags
- manual note/category.

Do not include this diagnostic file in training input.

---

# 22. Direct comparison with mDeBERTa

Create a final comparison artifact once Member B's outputs exist.

Recommended file:

`reports/neural_baseline_comparison.md`

Use exactly one shared table:

| Metric | mDeBERTa | ByT5 |
|---|---:|---:|
| Data version | | |
| Checkpoint | microsoft/mdeberta-v3-base | google/byt5-small |
| Max model length | 256 subword tokens | 512 byte tokens |
| Macro-F0.5 | | |
| Singleton metric | | |
| Non-singleton metric | | |
| US metric | | |
| India metric | | |
| Threshold | | |
| Candidate oracle ceiling | same candidate context | same candidate context |
| Peak VRAM | | |
| Train pairs/sec | | |
| Inference pairs/sec | | |
| Truncation rate | | |
| Projected full-test runtime | | |

Before calculating deltas, verify that both rows reference the same:

- query-manifest identity,
- train-pair manifest identity,
- threshold-pair manifest identity,
- final-pair manifest identity,
- candidate manifest/candidate version.

If any of those differ, mark the comparison invalid rather than reporting a misleading A/B result.

---

# 23. Quality-versus-cost decision analysis

Do not judge ByT5 by macro-F0.5 alone.

Report:

```text
quality delta
vs
peak VRAM delta
vs
training throughput ratio
vs
inference throughput ratio
vs
projected full-test runtime delta
```

Also report where the quality delta comes from.

Possible evidence-based outcomes include:

### Strong quality improvement at acceptable cost

Keep ByT5 as a serious final-model candidate.

### Small quality improvement with very large latency/memory cost

Document the marginal score gain and cost. Decide at project level whether the test-time budget supports it.

### Similar or lower quality

Keep the result as negative evidence. Do not keep tuning only because the model is more expensive.

### Improvement concentrated in character-noise subsets

Use the result to guide:

- preprocessing,
- augmentation,
- hard-example selection,
- or later model combination,

instead of automatically scaling ByT5.

---

# 24. Files the AI agent should change

## Required code changes

- `src/neural_adapters/byt5.py`
- `src/neural_adapters/base.py` — shared serializer helper
- `src/neural_contracts.py` — Torch seeding
- `src/train_neural.py` — seed before fresh adapter construction
- `src/neural_models.py` — singleton/non-singleton grouping

## Required tests

- `tests/test_byt5_adapter.py` — new
- `tests/test_neural_infrastructure.py` — extend evaluation/seeding coverage as appropriate

## Required configs

- `configs/neural/predict_byt5_threshold_pilot.json` — new
- `configs/neural/evaluate_byt5_pilot.json` — new

## Recommended diagnostics

- `scripts/analyze_byt5_truncation.py` — new
- optional `scripts/compare_neural_baselines.py` if a tiny report generator is preferable to manual Markdown

## Do not modify for this task unless a genuine shared bug requires it

- candidate generation algorithms,
- query split construction,
- negative mining,
- candidate indexes,
- source indexes,
- ground truth,
- mDeBERTa model logic owned by Member B.

---

# 25. Agent execution sequence

The AI coding agent should execute the task in this order.

## Stage A — Repository safety

1. Confirm current branch and commit.
2. Inspect changes since `b528414...` in all neural files.
3. Do not overwrite newer Member B/shared changes.
4. Run existing CPU tests before modification.
5. Record baseline test status.

## Stage B — Shared correctness fixes

1. Add shared serializer.
2. Add Torch seeding.
3. Seed before adapter creation.
4. Add singleton grouping.
5. Add/adjust tests.
6. Run fast test suite.

## Stage C — Implement ByT5 adapter

1. Add lazy Transformers imports.
2. Load `AutoTokenizer`.
3. Load `T5EncoderModel`.
4. Build wrapper classifier.
5. Implement serialization.
6. Implement one-pass tokenization/truncation/dynamic padding.
7. Implement masked mean pooling.
8. Implement device movement.
9. Implement parameter exposure.
10. Implement gradient checkpointing hooks.
11. Implement local save/reload.
12. Add adapter tests.
13. Run all non-network tests.

## Stage D — Add configs/diagnostics

1. Add threshold prediction config.
2. Add ByT5 evaluation config.
3. Add truncation analysis script.
4. Dry-run all configs.
5. Ensure dry-run does not load/download models.

## Stage E — Real-model sanity

1. Verify required shared pair artifacts exist.
2. Load `google/byt5-small`.
3. Run real forward/backward sanity.
4. Run padding invariance test.
5. Run save/reload test.
6. Run tiny overfit.

## Stage F — Truncation pre-analysis

1. Scan train/threshold/final pair lengths.
2. Save truncation report.
3. Record address-cut statistics.
4. Do not change 512 yet.

## Stage G — Short GPU pilot

1. Run on ~40 GB GPU if available.
2. Capture VRAM, pps, loss, truncation.
3. Test 24 GB viability.
4. Enable checkpointing/reduce microbatch only if required.
5. Lock full-baseline config.

## Stage H — Full training

1. Run one epoch on exact shared train shards.
2. Save checkpoint.
3. Verify checkpoint reload.
4. Save training metrics/environment.

## Stage I — Threshold set

1. Score exact shared threshold shards.
2. Validate prediction completeness.
3. Tune ByT5-specific threshold.
4. Freeze threshold.

## Stage J — Final validation

1. Score exact shared final shards.
2. Evaluate once using frozen threshold.
3. Save final metrics.
4. Perform post-hoc error/truncation analysis.

## Stage K — Comparison/handoff

1. Load Member B's metrics when available.
2. Verify all shared manifest identities match.
3. Build comparison table.
4. Estimate full-test runtime.
5. Write final handoff.

---

# 26. Acceptance criteria

The implementation is complete only when all of the following are true.

## Adapter

- [ ] `google/byt5-small` loads through explicit `--execute`.
- [ ] only encoder hidden states are used for classification.
- [ ] no decoder inference/generation is used.
- [ ] six shared raw fields are serialized in the agreed format.
- [ ] dynamic padding works.
- [ ] masked mean pooling excludes padding.
- [ ] output shape is `[batch, 2]`.
- [ ] class 1 is match.
- [ ] encoder + classifier parameters are trainable.
- [ ] gradient checkpointing can be toggled.
- [ ] save/reload is self-contained.

## Correctness

- [ ] forward sanity passes.
- [ ] padding invariance passes.
- [ ] tiny overfit passes.
- [ ] save/reload predictions match.
- [ ] truncation counting is verified with multi-byte text.
- [ ] shared tests pass.

## Reproducibility

- [ ] Torch seed is set before classifier construction.
- [ ] git commit is recorded.
- [ ] data/query manifest identities are recorded.
- [ ] ByT5 resolved revision is recorded.
- [ ] Torch/Transformers/CUDA/GPU environment is recorded.
- [ ] full training config is preserved.

## Experiment

- [ ] train pair shards are identical to mDeBERTa's.
- [ ] threshold pair shards are identical.
- [ ] final pair shards are identical.
- [ ] candidate set/version is identical.
- [ ] one-epoch baseline completes.
- [ ] checkpoint exists.
- [ ] threshold predictions exist.
- [ ] ByT5 threshold is tuned independently.
- [ ] threshold is frozen before final evaluation.
- [ ] final predictions exist.

## Reporting

- [ ] final macro-F0.5.
- [ ] singleton metric.
- [ ] non-singleton metric.
- [ ] US metric.
- [ ] India metric.
- [ ] candidate oracle context.
- [ ] peak VRAM.
- [ ] training pairs/sec.
- [ ] inference pairs/sec.
- [ ] truncation rate.
- [ ] projected full-test runtime or explicit pending status if test pair count is unavailable.
- [ ] ByT5-specific error analysis.
- [ ] direct mDeBERTa comparison after Member B artifacts exist.

---

# 27. Final handoff template

```text
BYT5 BASELINE COMPLETE

Repository commit:
Data version / manifest identities:
Checkpoint:
Resolved pretrained revision:
Training config:
GPU / Torch / Transformers / CUDA:

Threshold selected:
Threshold-set macro-F0.5:

Final macro-F0.5:
Singleton result:
Non-singleton result:
US result:
India result:
Candidate oracle ceiling:

Peak VRAM:
Train pairs/sec:
Inference pairs/sec:
Truncation rate:
Address-truncation rate:
Projected full-test runtime:

mDeBERTa macro-F0.5:
mDeBERTa inference pairs/sec:
Quality delta:
Inference speed ratio:
Peak-memory delta:

Where ByT5 helped:
Where ByT5 struggled:

Recommendation for next experiment:
```

Do not write `BYT5 BASELINE COMPLETE` until the real training, threshold tuning, untouched final evaluation, and required measurements have actually completed.

---

# 28. First implementation PR target

The first code PR should be deliberately smaller than the full GPU experiment.

It should contain:

1. functional ByT5 encoder adapter;
2. shared deterministic serialization helper;
3. reproducibility seeding fix;
4. singleton evaluation grouping;
5. ByT5 threshold/evaluation configs;
6. offline-capable adapter tests;
7. truncation diagnostic script;
8. documentation/update to the Member C handoff if useful.

It should **not** contain claimed accuracy/throughput results unless those runs were actually performed and their artifacts/logs are available.

After that PR is merged and tests pass, run the GPU experiment as a separate reproducible execution step.
