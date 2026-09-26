# Member B handoff

`MDebertaPairAdapter` (`src/neural_adapters/mdeberta.py`) wraps
`microsoft/mdeberta-v3-base` as a two-logit sequence-pair classifier at 256
subword tokens. Segment A is the S1 record and segment B the candidate, each
serialized as `name | address | country`; missing values become
`missing_text` (empty by default) and whitespace is collapsed. Truncation is
`longest_first`, and every batch reports how many pairs exceeded the budget.
Inference on a BF16-capable GPU autocasts to BF16 inside the adapter; training
precision comes from the shared trainer. Gradient checkpointing is supported
but off by default. Checkpoints record the resolved Hub revision and a
serialization version that reload checks.

Offline tests: `tests/test_mdeberta_adapter.py`. GPU sanity checks:
`scripts/mdeberta_sanity.py`. Step-by-step GPU run:
`docs/member_b_lightning_runbook.md`.

Do not add a second training loop or threshold policy.
