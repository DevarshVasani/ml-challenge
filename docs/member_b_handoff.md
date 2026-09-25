# Member B handoff

Implement `MDebertaPairAdapter` in `src/neural_adapters/mdeberta.py`. The
required methods are `serialize_pair`, `collate`, `logits`, `parameters`,
`to_device`, `save_pretrained`, and `load_pretrained`; optionally expose
gradient-checkpointing support. Use `microsoft/mdeberta-v3-base`, a two-logit
sequence-pair classifier, and an initial maximum length of 256 subword tokens.

Do not add a second training loop or threshold policy. Run the shared trainer
and evaluator only on a compute machine after pair shards and query manifests
exist. Add model-specific truncation and checkpoint round-trip tests.

