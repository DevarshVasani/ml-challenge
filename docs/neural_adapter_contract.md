# Neural adapter contract

The shared trainer consumes raw, untokenized pair shards. An adapter is the
only model-specific code. It must implement `serialize_pair(row)`,
`collate(rows)`, `logits(batch)`, `parameters()`, `to_device(device)`,
`save_pretrained(output_dir)`, and `load_pretrained(output_dir)` as defined in
`src/neural_adapters/base.py`.

`logits` must return floating point `[batch_size, 2]`; column 1 is the match
class. `collate` dynamically pads a batch and returns a `truncated` counter.
The shared field order is `name_a, address_a, country_a, name_b, address_b,
country_b`. Missing text is the empty string. IDs, folds, labels, retrieval
scores, and provenance are never serialized into model text.

Member B implements `microsoft/mdeberta-v3-base` as a two-logit sequence-pair
classifier, initially at 256 subword tokens. Member C implements an encoder
over `google/byt5-small` at 512 byte tokens, attention-mask-aware mean pooling,
dropout, and a two-logit linear head. ByT5 has no BERT CLS token; padding must
be excluded from pooling. This is encoder-only adaptation, not a full
encoder-decoder generation task, and byte lengths are not comparable to
subword lengths.

Neither skeleton imports Transformers or downloads weights. Their current
`NotImplementedError` is intentional and actionable. The fake adapter is only
for explicit `smoke=true` offline checks and must never be labeled a production
prediction adapter.

