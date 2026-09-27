"""Externally deduplicate complete reverse and forward test candidates by pair key."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .retrieval_experiments import sha, write_json
from .forward_fts import audit_output


def literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def merge(reverse: Path, forward: Path, output: Path, memory_gb: int = 12,
          expected_source_groups: int = 9_969_589, expected_queries: int = 1_732_544,
          forward_k: int = 50) -> dict:
    reverse_meta = json.loads((reverse / "manifest.json").read_text())
    forward_report = json.loads((forward / "run_report.json").read_text())
    if not reverse_meta["complete"] or not forward_report["complete"] or forward_report["split"] != "test":
        raise ValueError("input channels are incomplete or not test")
    if int(reverse_meta["source_groups"]) != expected_source_groups or int(forward_report["queries"]) != expected_queries:
        raise ValueError("input query/source coverage differs from full test population")
    audited = audit_output(forward, "test", expected_queries)
    for key in ("queries", "pairs", "partitions", "k_per_field", "partition_manifest_sha256"):
        if audited[key] != forward_report.get(key):
            raise ValueError(f"forward run report differs from verified shards: {key}")
    for name, bucket in reverse_meta["buckets"].items():
        path = reverse / name
        if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != bucket["pairs"] or sha(path) != bucket["sha256"]:
            raise ValueError(f"corrupt reverse union bucket {name}")
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= forward_k <= int(forward_report["k_per_field"]):
        raise ValueError("forward merge rank budget exceeds generated candidates")
    config = {"version": "forward-reverse-union-v1", "reverse_manifest_sha256": sha(reverse / "manifest.json"),
              "forward_report_sha256": sha(forward / "run_report.json"), "buckets": 8,
              "forward_k": forward_k}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    marker = output / "manifest.json"
    prior = json.loads(marker.read_text()) if marker.exists() else None
    if prior and prior["config_hash"] != config_hash:
        raise ValueError("incompatible forward/reverse union output")
    completed = dict(prior.get("buckets", {})) if prior else {}
    started = time.time()
    db = duckdb.connect()
    db.execute(f"SET memory_limit='{int(memory_gb)}GB'")
    db.execute("SET threads=4")
    temp_dir = output / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory={literal(temp_dir)}")
    forward_glob = forward / "bucket-*" / "*.parquet"
    for bucket in range(8):
        name = f"bucket-{bucket:02d}.parquet"
        target = output / name
        if name in completed:
            if not target.exists() or sha(target) != completed[name]["sha256"]:
                raise ValueError(f"corrupt completed union bucket {name}")
            continue
        reverse_file = reverse / name
        expected_groups = reverse_meta["buckets"][name]["source_groups"]
        tmp = target.with_suffix(".parquet.tmp")
        sql = f"""
        COPY (
          SELECT source, source_record_id, s1_id,
                 max(dense_score) AS dense_score, min(dense_rank) AS dense_rank,
                 max(reverse_name_score) AS name_score, min(reverse_name_rank) AS name_rank,
                 max(reverse_address_score) AS address_score, min(reverse_address_rank) AS address_rank,
                 max(forward_name_score) AS forward_name_score, min(forward_name_rank) AS forward_name_rank,
                 max(forward_address_score) AS forward_address_score, min(forward_address_rank) AS forward_address_rank,
                 min(source_bucket) AS source_bucket
          FROM (
            SELECT source,source_record_id,s1_id,dense_score,dense_rank,
                   name_score AS reverse_name_score,name_rank AS reverse_name_rank,
                   address_score AS reverse_address_score,address_rank AS reverse_address_rank,
                   NULL::DOUBLE AS forward_name_score,NULL::BIGINT AS forward_name_rank,
                   NULL::DOUBLE AS forward_address_score,NULL::BIGINT AS forward_address_rank,
                   source_bucket
            FROM read_parquet({literal(reverse_file)})
            UNION ALL
            SELECT source,source_record_id,s1_id,NULL::DOUBLE,NULL::BIGINT,
                   NULL::DOUBLE,NULL::BIGINT,NULL::DOUBLE,NULL::BIGINT,
                   name_score,name_rank,address_score,address_rank,source_bucket
            FROM read_parquet({literal(forward_glob)})
            WHERE source_bucket={bucket} AND (name_rank<={forward_k} OR address_rank<={forward_k})
          ) GROUP BY source,source_record_id,s1_id
        ) TO {literal(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        db.execute(sql)
        os.replace(tmp, target)
        count = pq.ParquetFile(target).metadata.num_rows
        completed[name] = {"sha256": sha(target), "pairs": count, "source_groups": expected_groups,
                           "bytes": target.stat().st_size}
        write_json(marker, {"config": config, "config_hash": config_hash, "buckets": completed,
                            "complete": False})
        print(json.dumps({"bucket": bucket, "pairs": count, "seconds": time.time() - started}), flush=True)
    db.close()
    result = {"config": config, "config_hash": config_hash, "buckets": completed, "complete": True,
              "pairs": sum(v["pairs"] for v in completed.values()),
              "source_groups": sum(v["source_groups"] for v in completed.values()),
              "seconds_this_run": time.time() - started}
    write_json(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", type=Path, default=Path("artifacts/reverse-v1/test_union"))
    parser.add_argument("--forward", type=Path, default=Path("artifacts/forward-fts-v2/test_candidates"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/forward-reverse-v1/test_union"))
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--forward-k", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(merge(args.reverse, args.forward, args.output, args.memory_gb,
                           forward_k=args.forward_k), indent=2))


if __name__ == "__main__":
    main()
