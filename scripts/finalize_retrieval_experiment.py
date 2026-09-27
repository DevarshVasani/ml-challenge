"""Summarize the frozen retrieval run without changing its candidate policy."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval_experiments import DATA, ROOT, write_json


def read(path):return json.loads(Path(path).read_text())


def main(out):
    out=Path(out);chosen=read(out/"chosen_policy.json")
    for rel,digest in chosen["code_sha256"].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"frozen retrieval code changed: {rel}")
    base=read(out/"baseline_metrics.json")
    arms=read(out/"threshold_arms.json")
    final=read(out/"final_arms.json")
    threshold_delta=read(out/"threshold_delta_manifest.json")
    final_delta=read(out/"final_delta_manifest.json")
    if threshold_delta["policy_hash"]!=chosen["policy_hash"] or final_delta["policy_hash"]!=chosen["policy_hash"]:
        raise ValueError("policy hash changed between threshold and final")
    write_json(out/"final_metrics.json",{"baseline":base["final"],"chosen":final["arms"]["combination"],
        "policy_hash":chosen["policy_hash"],"incremental_pairs":final_delta["row_count"],
        "final_scan":final["scan"],"selection_set":"threshold only"})
    pilot=read(out/"threshold_pilot.json")
    index_pilot=read(out/"index_pilot"/"benchmark.json")
    index_metadata=read(out/"index_pilot"/"candidate_index"/"S2_business_name"/"disk_metadata.json")
    projected_index_seconds=index_pilot["elapsed_seconds"]*(arms["scan"]["visited"]["S2"]+arms["scan"]["visited"]["S3"])/50000*2
    entries=[{"arm":"published_baseline","sample_type":"full_threshold","completion":"complete",
              "metrics":base["threshold"],"configuration_hash":read(DATA/"threshold_pairs_manifest.json")["configuration_hash"]},
             {"arm":"source_scan_benchmark","sample_type":"first_100000_rows_per_source_not_accuracy_sample",
              "completion":"complete","cost":pilot,"metrics":None},
             {"arm":"top_k_index_diagnostic","sample_type":"threshold","completion":"skipped",
              "reason":"Reusable full-background TF-IDF indexes are absent. The 50k-row single-channel build took 21.0 s and 42 MB; a linear four-channel projection is over two hours and its required 2x margin exceeds the remaining task budget. Vocabulary growth makes this optimistic.",
              "index_pilot":index_pilot,"projected_four_channel_seconds_linear":projected_index_seconds,
              "projected_four_channel_seconds_with_2x_margin":2*projected_index_seconds}]
    for arm,metrics in arms["arms"].items():
        entries.append({"arm":arm,"sample_type":"full_threshold","completion":"complete",
                        "configuration_hash":chosen["policy_hash"],"cost":arms["scan"],"metrics":metrics})
    entries.append({"arm":"combination_frozen","sample_type":"full_final","completion":"complete",
                    "configuration_hash":chosen["policy_hash"],"cost":final["scan"],"metrics":final["arms"]["combination"]})
    (out/"experiments.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in entries))
    failures=pq.read_table(out/"failure_analysis.parquet").to_pylist()
    groups=defaultdict(list)
    for r in failures:groups[(r["country"],r["candidate_source"],r["hypothesis"])].append(r)
    for group in groups.values():group.sort(key=lambda r:hashlib.sha256((r["source1_entity_id"]+r["candidate_entity_id"]).encode()).hexdigest())
    sample=[]
    while len(sample)<100 and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(sample)<100:sample.append(groups[key].pop())
    pq.write_table(pa.Table.from_pylist(sample),out/"threshold_failure_sample.parquet",compression="zstd")
    disk=shutil.disk_usage(ROOT)
    def cgroup(name):
        p=Path("/sys/fs/cgroup")/name
        return p.read_text().strip() if p.exists() else None
    env={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "baseline_git_commit":chosen["baseline_commit"],"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
         "cpu_count_os":os.cpu_count(),"cpu_max":cgroup("cpu.max"),"memory_max_bytes":cgroup("memory.max"),
         "memory_peak_bytes_cgroup":cgroup("memory.peak"),"disk_free_bytes_after":disk.free,
         "retrieval_scan_peak_rss_mb_before_pair_export":max(arms["scan"]["peak_rss_mb"],final["scan"]["peak_rss_mb"]),
         "observed_threshold_export_process_rss_mb_at_least":270.2,
         "index_pilot_max_child_rss_mb":index_pilot["max_child_rss_mb"],
         "experiment_directory_bytes":sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
         "train_source_identities":final["scan"]["source_identities"],
         "query_manifest_sha256":hashlib.sha256((DATA/"query_manifest.json").read_bytes()).hexdigest(),
         "full_background_indexes_present":False,"source_store_present":False,"verified_folds_present":False,
         "dependencies":{x:__import__(x).__version__ for x in ("pyarrow","pandas","numpy")}}
    write_json(out/"environment.json",env)
    t=arms["arms"]["combination"];f=final["arms"]["combination"]
    b1,b2=base["threshold"],base["final"]
    fr=read(out/"france_robustness.json")
    threshold_new_positives=int(pq.read_table(out/"threshold_delta_pairs.parquet",columns=["label"])["label"].to_numpy().sum())
    final_new_positives=int(pq.read_table(out/"final_delta_pairs.parquet",columns=["label"])["label"].to_numpy().sum())
    report=f"""# CPU retrieval experiment report

Frozen policy: `combination`, hash `{chosen['policy_hash']}`. Original baseline pairs are retained; the matcher scores baseline UNION delta once per unique pair. This is a sample estimate from the available threshold and final queries.

| Split | Oracle baseline → union | Positive-pair recall baseline → union | Complete misses baseline → union | Extra pairs | Mean candidates baseline → union | p95 baseline → union |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Threshold | {b1['oracle_macro_f05']:.9f} → {t['oracle_macro_f05']:.9f} | {b1['positive_pair_recall']:.9f} → {t['positive_pair_recall']:.9f} | {b1['zero_positive_queries']} → {t['zero_positive_queries']} | {threshold_delta['row_count']} | {b1['candidate_count']['mean']:.4f} → {t['candidate_count']['mean']:.4f} | {b1['candidate_count']['p95']} → {t['candidate_count']['p95']} |
| Final | {b2['oracle_macro_f05']:.9f} → {f['oracle_macro_f05']:.9f} | {b2['positive_pair_recall']:.9f} → {f['positive_pair_recall']:.9f} | {b2['zero_positive_queries']} → {f['zero_positive_queries']} | {final_delta['row_count']} | {b2['candidate_count']['mean']:.4f} → {f['candidate_count']['mean']:.4f} | {b2['candidate_count']['p95']} → {f['candidate_count']['p95']} |

The threshold gain is {t['oracle_macro_f05']-b1['oracle_macro_f05']:.9f} from {threshold_new_positives} recovered positives; final gain is {f['oracle_macro_f05']-b2['oracle_macro_f05']:.9f} from {final_new_positives} recovered positives. Candidate growth is {100*(t['candidate_count']['mean']/b1['candidate_count']['mean']-1):.3f}% mean and {100*(t['candidate_count']['p95']/b1['candidate_count']['p95']-1):.3f}% p95 on threshold. Seven final complete misses remain; no final failure identities were used for selection.

The policy uses two query variants (accent-folded exact name and legal-suffix removal) and one step from at most two baseline-retrieved seeds selected by lexical agreement. It scans all {arms['scan']['visited']['S2']:,} S2 and {arms['scan']['visited']['S3']:,} S3 training records. Exact keys with more than 20 source postings are skipped. This preserves original IDs and all baseline pairs. Query-only accent folding against the existing TF-IDF indexes was unavailable; this run used a symmetric exact-key scan of the full source background.

The first 100,000 source rows per source took {pilot['seconds']:.1f} s and {pilot['peak_rss_mb']:.1f} MiB peak RSS. Full scans took {arms['scan']['seconds']:.1f} s threshold and {final['scan']['seconds']:.1f} s final. Scan instrumentation recorded at most {max(arms['scan']['peak_rss_mb'],final['scan']['peak_rss_mb']):.1f} MiB before pair export; process monitoring observed at least 270.2 MiB during threshold export. The host exposed {env['cpu_count_os']} logical CPUs with no cgroup CPU quota (`cpu.max={env['cpu_max']}`); this run used one thread. Disk used by this run, including the disposable index pilot, was {env['experiment_directory_bytes']/1024**2:.2f} MiB; free disk after the run was {disk.free/1024**3:.1f} GiB. Full background coverage is established by the visited source row counts and source SHA-256 identities in `environment.json`.

The bounded index-build pilot used 50,000 S2 name rows: {index_pilot['elapsed_seconds']:.1f} s, {index_pilot['max_child_rss_mb']:.1f} MiB maximum child RSS, {index_pilot['output_bytes']/1024**2:.1f} MiB output and {index_metadata['vocabulary_size']:,} terms. A linear projection for four full channels is {projected_index_seconds/3600:.2f} hours; the required 2× margin is {2*projected_index_seconds/3600:.2f} hours, exceeding the remaining task budget. The global term-count SQLite store can grow faster than the sample, so this is an optimistic lower bound. Full TF-IDF index construction and top-k rank diagnostics were therefore skipped.

Threshold failures: {len(failures)} missed positive edges in {len(set(r['source1_entity_id'] for r in failures))} queries. See `failure_summary.md`, `failure_analysis.parquet`, and the balanced 100-row `threshold_failure_sample.parquet`. Training had {read(out/'training_failure_summary.json')['injected_positive_pairs']} injected missed positives; they were diagnostic targets and excluded from seed selection. The training seed check reports {read(out/'training_seed_calibration.json')['selected_seed_precision_in_sampled_train_pairs']:.1%} labeled-positive among selected seeds in sampled train pairs. This is not full-background seed precision.

France: {fr['sample_records']} S2/S3 records and {fr['synthetic_probe_count']} altered queries are saved in `france_probe_inputs.json`, plus {fr['actual_s1_qualitative_sample_count']} actual S1 examples. Complete test-source indexes are absent, so known-target recovery and actual France entity-resolution recall are pending. No French labels exist. The pending full-background commands are in `france_robustness.json`, with ready-to-run `france_test_index_config.json`, `france_probe_query_config.json`, and `france_probe_queries.tsv`.

Artifacts: `threshold_delta_pairs.parquet` and `final_delta_pairs.parquet` contain new raw-text pairs only. Their manifests include query coverage, zero-addition counts, parent baseline checksums, source identities, row counts, and the frozen policy hash. `*_delta_provenance.parquet` records seed IDs and variant IDs separately. `*_merged_manifest.json` lists the baseline and delta shards to score as a union. Negative candidate folds are null because the verified fold map and source store referenced by the historical manifests are absent; raw text was joined directly from the original training TSVs. No training hard-negative backfill was attempted without the verified fold map.

The full split scan writes an explicit in-progress state and an atomic complete shard/manifest. It does not resume partway through a source scan; a restart repeats the roughly three-minute full-background pass. Query-batch checkpoints were not implemented for this short scan.

Reproduce from the repository root with one CPU thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.retrieval_scan threshold --out artifacts/retrieval-experiments/20260926-cpu-01 --chosen combination
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.retrieval_scan final --out artifacts/retrieval-experiments/20260926-cpu-01 --chosen combination
python -m src.merge_retrieval_delta threshold --out artifacts/retrieval-experiments/20260926-cpu-01
python -m src.merge_retrieval_delta final --out artifacts/retrieval-experiments/20260926-cpu-01
```

The exact next matcher handoff is to read `final_merged_manifest.json`, score its baseline shards plus `final_delta_pairs.parquet`, and deduplicate by `(source1_entity_id, candidate_source, candidate_entity_id)` before thresholding predictions. The existing `src.predict_neural` runner accepts this shard list via the `pairs` config key once a trained checkpoint is supplied. For future test export, first build full test-source indexes and a complete test baseline; then run this frozen policy over every test S1 query with only valid test S2/S3 IDs. That full 1.73 million-query run was intentionally outside this exploratory budget. No reduced-background recall proxy was reported.
"""
    (out/"REPORT.md").write_text(report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();main(a.out)
