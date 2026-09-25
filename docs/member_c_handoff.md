# Member C handoff

Implement `ByT5EncoderPairAdapter` in `src/neural_adapters/byt5.py`. The
required methods are `serialize_pair`, `collate`, `logits`, `parameters`,
`to_device`, `save_pretrained`, and `load_pretrained`; optionally expose
gradient-checkpointing support. Use `google/byt5-small`'s encoder only, mean
pool hidden states with the attention mask, apply dropout, and return two
classifier logits. The initial maximum is 512 byte tokens.

ByT5 has no CLS token. Do not build a decoder or generate IDs. Do not add a
second training loop or threshold policy. Add model-specific truncation and
checkpoint round-trip tests, then use the shared commands in the GPU runbook.

