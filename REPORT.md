# Member B retrieval report

**2026-09-27 handoff update:** See [reports/retrieval_handoff_2026-09-27.md](reports/retrieval_handoff_2026-09-27.md) for the selected production policy, verified complete test union, fresh 5/10/20/50-query replay, rejected rescue experiments, and current train queue state. The snapshot below is earlier and its production progress counts are superseded by the handoff.

Snapshot: 2026-09-26 18:58 UTC. Production continues under the detached queues below; a running stage is **not** a completed candidate export.

## Inventory and policy

| Data | Train | Test |
| --- | ---: | ---: |
| S1 references | 2,206,821 | 1,732,544 |
| S2 source records | 5,034,616 | 4,887,273 |
| S3 source records | 5,285,603 | 5,082,316 |

The frozen S1 train/threshold/final query IDs are in `artifacts/member-b-starter-v1/{train,threshold,final}_queries.tsv` (50,000/2,500/2,500). The existing split metadata is in `../neural-data/query_manifest.json` and `reports/split_summary.json`. Historical pair shards live in `../neural-data/{train,threshold,final}_pairs_manifest.json`; their full query manifests include zero-candidate queries. The retrieved train S2/S3 forward index from `../artifacts.tar` is now at `artifacts/candidate-index/`: all 3,336 manifest checksums and both raw source SHA-256 fingerprints match. It has no corresponding full test forward index, so its old validation pairs are used only as a diagnostic baseline, never counted as deployed test retrieval.

The deployable policy is `artifacts/reverse-v1/frozen_policy.json`: every S2/S3 record queries the **full same-country S1 reference index**, with full-index fallback for absent or unknown country. `artifacts/reverse-v1/country_truth_audit.json` found zero country mismatches across 7,621,121 training positive edges after excluding frozen threshold/final S1 queries. It unions lexical top 50 (`reverse-fts-country-postfilter-v5`) and Granite exact dense top 20. The Granite revision is `835ad14087e140460703cf0fae09f97d469d65c2`, max length 128, CLS pooling, normalized 384-dimensional vectors. Its [model card](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) lists Apache-2.0. No truth edge enters candidate output. Internal pair identity is the exact string triple `(source, source_record_id, s1_id)`.

The machine has 8 CPUs, 31 GiB RAM (about 26 GiB initially available), a 23 GiB NVIDIA L4, and 369 GB disk (270 GB free at this snapshot). Torch CUDA works; FAISS is CPU-only here. Source text and country remain in the original TSVs. Cached S1 embeddings use float16 memory maps and string ID maps, with source/model/serializer fingerprints and SHA-256 checksums.

## Retrieval quality

All oracles use every frozen query, deduplicated positives, `5r/(4r+M)` for nonempty gold, and 1 for empty gold. Positive-only diagnostics query **all held-out positive source endpoints against full S1 country backgrounds**. They establish recall, but their candidate counts are deliberately omitted; arbitrary-record counts come from the separate unlabeled sample.

| Threshold policy | Macro oracle F0.5 | Positive recall | Zero-positive non-singletons | New baseline positives |
| --- | ---: | ---: | ---: | ---: |
| Historical forward baseline | 0.987343 | 0.970830 | 12 | 0 |
| Reverse lexical top 50 alone | 0.936656 | 0.854381 | 64 | — |
| Dense top 20 alone | 0.939054 | 0.865725 | 65 | — |
| Dense top 50 alone | 0.948880 | 0.887834 | 54 | — |
| Lexical 50 + dense 20, deployable | 0.978558 | 0.950689 | 22 | — |
| Historical baseline + deployed reverse channels, diagnostic | 0.992927 | 0.984489 | 8 | 118 |

The reverse-only threshold non-singleton oracle is 0.977363. Its 426 remaining missed positive edges are 320 India and 106 US; S2 recall is 0.944258 and S3 recall 0.956667. The full baseline miss table is `artifacts/reverse-v1/threshold_all_missing_positives.parquet` (all 252 edges, with raw text, scripts, country, baseline count, and suspected category). Its leading patterns are 93 uncertain, 65 complementary descriptions, 46 missing fields, and 52 Latin/Devanagari name pairs. No final miss identities were inspected to choose rules.

The **one frozen final evaluation** is `artifacts/dense-v1/frozen_final_aggregate.json`: reverse-only oracle **0.981742**, positive recall **0.950494**, 14 zero-positive non-singletons; historical baseline union oracle **0.995945**, positive recall **0.986519**, 2 zero-positive non-singletons. This is still below the 0.998 target. France has no label oracle.

## Measured throughput and size

- Granite encoded 100,000 real S1 records in 45.9 seconds at batch 256; no sample record exceeded 128 tokens. The full train/test S1 caches finished in 692/633 seconds. On the stratified 103,321-record train sample (source, country, script, length), exact dense search wrote 2,066,420 pairs in 68.0 steady seconds, 86.5 seconds including startup. Peak RSS was 3.54 GiB and peak CUDA allocation 2.97 GiB.
- Seven lexical workers processed the same 103,321 records at top 50 in 192.1 seconds and wrote 1,855,136 pairs. The bounded sample union closed **103,321 of 103,321 source groups**, with 3,647,765 unique pairs: mean **35.31**, p95 **69**, p99/max **70**. Its 8 hash buckets were merged in 3.5 seconds and occupied 40.1 MB. Source/country mean counts: S2 India 35.48, S2 US 33.86, S3 India 37.45, S3 US 35.13.
- IVF-Flat on 883,188 India references at 1,024 lists lost 13 of 139 exact top-20 true-positive hits at 16 probes and 5 at 64 probes on 200 positive endpoints. Approximate neighbor recall at 20 was 0.777/0.896. Exact GPU search was selected; this small sample cannot establish a 0.0002 oracle tolerance for IVF.
- The France-only **unlabeled** test sample processed 14,268 records and wrote 285,360 dense pairs in 23.6 steady seconds under concurrent production load (606 records/s). No France accuracy claim is made.

The current 20.29M-record train+test source volume projects about 406M dense pairs and, from the representative union mean, about 716M union pairs. At observed warm production rates around 1.1k dense and 0.9–1.2k lexical records/s, each channel's train+test pass is about 5–6 hours before a roughly 30% contingency; jobs share CPU/GPU, so these are elapsed stage estimates, not summed throughput. Current disk headroom exceeds 60 GB by a wide margin. The union projection is based on sample overlap and may move with France's distribution.

## Production status and restart

As of this snapshot, exact dense test retrieval has closed 1,004,000/9,969,589 source groups and written 20,080,000 pairs; 144,325 of those processed source records are France S2 records. Seven lexical test workers have closed 339,000 source groups and written 5,833,783 pairs. **Neither test channel is complete yet.** The dense queue will run full train after test; the lexical queue will likewise run full train; the merge queue will publish test/train hash-partitioned unions only after both channel manifests are complete. Queue states are `artifacts/reverse-v1/{dense,lexical,merge}_queue_state.json`; logs are `artifacts/{dense-v1,reverse-v1}-*-production.log` and `artifacts/reverse-v1-*-merge.log`.

The queues and all shards are restartable: source/model/policy fingerprints must match, completed files are checksum-checked, writes are temporary then atomic, and a source group is written only after both fields/channels in its shard finish. The final union groups by `(source, source_record_id, s1_id)` within deterministic source hash buckets and retains separate channel scores and ranks. Labels and evaluation truth stay outside candidate rows. An incomplete manifest is a partial diagnostic, not a production artifact.

From the repository root, the main reproducible commands are:

```bash
python -m src.reverse_retrieval build --split train --root artifacts/reverse-v1
python -m src.reverse_retrieval build --split test --root artifacts/reverse-v1
python -m src.dense_retrieval encode-s1 --split train --out artifacts/dense-v1 --batch-size 256
python -m src.dense_retrieval encode-s1 --split test --out artifacts/dense-v1 --batch-size 256
python scripts/sample_sources.py --split train --output artifacts/dense-v1/train_representative_sample.parquet
python scripts/partition_sources.py --split test --output artifacts/reverse-v1/test_partitions --buckets 8
python scripts/partition_sources.py --split train --output artifacts/reverse-v1/train_partitions --buckets 8
python -m scripts.production_queue lexical
python -m scripts.production_queue dense
python -m scripts.production_queue merge
```

Run `python -m pytest -q tests/test_reverse_retrieval.py tests/test_retrieval_experiments.py` for the focused pair-key, direction, oracle and lexical resume checks (4 passed). `artifacts/dense-v1/resume_verification.json` additionally confirms that a dense 1,000-record restart produced the same 40,000 pair keys across two shards as an uninterrupted sample. The production queues are already running; rerunning a queue uses compatible completed manifests. Remaining work is completion and checksum audit of full test/train union manifests, then downstream feature extraction. The restored train forward index can support separate train-only comparisons; no full test forward index was present, so the historical forward union is not a production policy.
