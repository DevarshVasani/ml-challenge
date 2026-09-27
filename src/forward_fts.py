"""Resumable full-background S1-to-S2/S3 lexical retrieval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import time
from itertools import islice
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .reverse_retrieval import identity, source_path, terms_for, tokens
from .retrieval_experiments import sha, write_json

VERSION = "forward-fts-v2"


@lru_cache(maxsize=100_000)
def token_set(value: str) -> frozenset[str]:
    return frozenset(tokens(value))


def index_path(root: Path, split: str, source: str) -> Path:
    return root / f"{split}_{source}.sqlite"


def build(root: Path, split: str, source: str, batch_size: int = 20_000) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    original = source_path(split, source)
    path = index_path(root, split, source)
    meta_path = path.with_suffix(".json")
    config = {"version": VERSION, "source": source, "input": identity(original)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("config") != config or not meta.get("complete"):
            raise ValueError("incompatible forward index")
        return meta
    started = time.time()
    db = sqlite3.connect(path)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("CREATE TABLE IF NOT EXISTS build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        prior = db.execute("SELECT value FROM build_meta WHERE key='config'").fetchone()
        requested = json.dumps(config, sort_keys=True)
        if prior and prior[0] != requested:
            raise ValueError("incompatible partial forward index")
        if not prior:
            if db.execute("SELECT count(*) FROM sqlite_master WHERE name IN ('records','search')").fetchone()[0]:
                raise ValueError("unrecognized partial forward index")
            db.execute("CREATE TABLE records (rid INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE, country TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE search USING fts5(name,address,country, tokenize='unicode61 remove_diacritics 2')")
            db.execute("INSERT INTO build_meta VALUES ('config', ?)", (requested,))
            db.execute("INSERT INTO build_meta VALUES ('rows', '0')")
            db.commit()
        completed = int(db.execute("SELECT value FROM build_meta WHERE key='rows'").fetchone()[0])
        count = 0
        with original.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            while batch := list(__import__("itertools").islice(reader, batch_size)):
                end = count + len(batch)
                if end <= completed:
                    count = end
                    continue
                if count < completed:
                    batch = batch[completed - count:]
                    count = completed
                records = [(count + i + 1, row["entity_id"], row["country"]) for i, row in enumerate(batch)]
                search = [(count + i + 1, row["business_name"], row["business_address"], row["country"]) for i, row in enumerate(batch)]
                db.executemany("INSERT INTO records VALUES (?,?,?)", records)
                db.executemany("INSERT INTO search(rowid,name,address,country) VALUES (?,?,?,?)", search)
                count += len(batch)
                db.execute("UPDATE build_meta SET value=? WHERE key='rows'", (str(count),))
                db.commit()
        if count < completed:
            raise ValueError("source is shorter than partial index")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("PRAGMA journal_mode=DELETE")
    finally:
        db.close()
    result = {"config": config, "complete": True, "rows": count, "bytes": path.stat().st_size,
              "sha256": sha(path), "seconds_this_run": time.time() - started}
    write_json(meta_path, result)
    return result


class ForwardIndex:
    def __init__(self, root: Path, split: str):
        self.connections = {}
        for source in ("S2", "S3"):
            path = index_path(root, split, source)
            meta = json.loads(path.with_suffix(".json").read_text())
            expected = {"version": VERSION, "source": source, "input": identity(source_path(split, source))}
            if not meta.get("complete") or meta.get("config") != expected:
                raise ValueError(f"incompatible {source} forward index")
            self.connections[source] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def close(self):
        for db in self.connections.values():
            db.close()

    def query(self, name: str, address: str, country: str, k: int = 50) -> list[dict]:
        found = {}
        for source, db in self.connections.items():
            for field, terms in terms_for(name, address):
                if not terms:
                    continue
                original = token_set(name if field == "name" else address)
                probes = [" AND ".join(f'{field}:"{term}"' for term in terms[:2])]
                probes += [f'{field}:"{term}"' for term in terms[:2]]
                for query in dict.fromkeys(probes):
                    sql = "SELECT r.entity_id,search.name,search.address FROM search JOIN records r ON r.rid=search.rowid WHERE search MATCH ?"
                    if country:
                        sql += " AND r.country=?"
                    sql += " LIMIT 501"
                    rows = db.execute(sql, (query, country) if country else (query,)).fetchall()
                    if len(rows) > 500:
                        continue
                    scored = []
                    for cid, candidate_name, candidate_address in rows:
                        candidate = token_set(candidate_name if field == "name" else candidate_address)
                        similarity = len(original & candidate) / len(original | candidate) if original and candidate else 0.0
                        scored.append((str(cid), similarity))
                    for rank, (cid, value) in enumerate(sorted(scored, key=lambda x: (-x[1], x[0]))[:k], 1):
                        item = found.setdefault((source, cid), {"source": source, "source_record_id": cid,
                                                              "name_rank": None, "address_rank": None,
                                                              "name_score": None, "address_score": None})
                        old_rank = item[f"{field}_rank"]
                        if old_rank is None or rank < old_rank:
                            item[f"{field}_rank"] = rank
                            item[f"{field}_score"] = value
                    if scored:
                        break
        return sorted(found.values(), key=lambda x: (-max(x["name_score"] or 0, x["address_score"] or 0), x["source"], x["source_record_id"]))


def retrieve(root: Path, split: str, output: Path, k: int, batch_size: int,
             partition: int = 0, partitions: int = 1, limit: int = 0) -> dict:
    if not 0 <= partition < partitions or batch_size <= 0 or k <= 0:
        raise ValueError("invalid forward retrieval partition or budget")
    output.mkdir(parents=True, exist_ok=True)
    index_meta = {source: json.loads(index_path(root, split, source).with_suffix(".json").read_text())
                  for source in ("S2", "S3")}
    config = {"version": VERSION, "split": split, "k_per_field": k,
              "batch_size": batch_size, "partition": partition, "partitions": partitions,
              "s1_source": identity(source_path(split, "S1")),
              "indexes": {s: {"config": m["config"], "sha256": m["sha256"]} for s, m in index_meta.items()}}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["config_hash"] != config_hash:
        raise ValueError("incompatible forward output")
    completed = dict(previous.get("shards", {})) if previous else {}
    index = ForwardIndex(root, split)
    started = time.time()
    processed = 0
    written = 0
    selected = []
    def write_batch(batch):
        nonlocal processed, written
        start = processed
        name = f"S1-{start:09d}-{start + len(batch):09d}.parquet"
        target = output / name
        if name in completed:
            if not target.exists() or sha(target) != completed[name]["sha256"]:
                raise ValueError(f"corrupt completed forward shard {name}")
            processed += len(batch)
            return
        rows = []
        for query in batch:
            q = query["entity_id"]
            for candidate in index.query(query["business_name"], query["business_address"], query["country"], k):
                sid = candidate["source_record_id"]
                source = candidate["source"]
                rows.append({"s1_id": q, "source": source, "source_record_id": sid,
                             "name_rank": candidate["name_rank"], "name_score": candidate["name_score"],
                             "address_rank": candidate["address_rank"], "address_score": candidate["address_score"],
                             "source_bucket": int.from_bytes(hashlib.blake2b(f"{source}:{sid}".encode(), digest_size=2).digest(), "big") % 8,
                             "policy_version": VERSION})
        tmp = target.with_suffix(".parquet.tmp")
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
        else:
            pq.write_table(pa.table({"s1_id": pa.array([], pa.string()),
                                     "source": pa.array([], pa.string()),
                                     "source_record_id": pa.array([], pa.string())}), tmp)
        os.replace(tmp, target)
        digest = sha(target)
        completed[name] = {"sha256": digest, "queries": len(batch), "pairs": len(rows),
                           "start": start, "end": start + len(batch)}
        processed += len(batch)
        written += len(rows)
        write_json(manifest_path, {"config": config, "config_hash": config_hash, "shards": completed,
                                   "complete": False, "queries_this_run": processed,
                                   "seconds_this_run": time.time() - started})
        print(json.dumps({"partition": partition, "queries": processed,
                          "pairs_this_run": written, "queries_per_second": processed / (time.time() - started)}), flush=True)
    try:
        with source_path(split, "S1").open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if int.from_bytes(hashlib.blake2b(row["entity_id"].encode(), digest_size=2).digest(), "big") % partitions != partition:
                    continue
                selected.append(row)
                if len(selected) == batch_size:
                    write_batch(selected)
                    selected = []
                    if limit and processed >= limit:
                        break
            else:
                if selected:
                    write_batch(selected)
                selected = []
        complete = not limit
        write_json(manifest_path, {"config": config, "config_hash": config_hash, "shards": completed,
                                   "complete": complete, "queries_this_run": processed,
                                   "seconds_this_run": time.time() - started})
    finally:
        index.close()
    return {"queries": processed, "pairs_this_run": written, "complete": complete,
            "seconds": time.time() - started}


def audit_output(output: Path, split: str, expected_queries: int | None = None) -> dict:
    """Verify every partition and shard before publishing a complete forward run."""
    partitions = sorted(output.glob("bucket-[0-9][0-9]/manifest.json"))
    if not partitions:
        raise ValueError("forward output has no partition manifests")
    total_queries = total_pairs = 0
    hashes = {}
    seen = set()
    partition_count = len(partitions)
    common_config = None
    for manifest_path in partitions:
        folder = manifest_path.parent
        partition = int(folder.name.removeprefix("bucket-"))
        if partition in seen or partition >= partition_count:
            raise ValueError("forward partition numbering is incomplete")
        seen.add(partition)
        manifest = json.loads(manifest_path.read_text())
        cfg = manifest["config"]
        if not manifest.get("complete") or cfg["split"] != split or cfg["partition"] != partition or cfg["partitions"] != partition_count:
            raise ValueError(f"incomplete or incompatible forward partition {partition}")
        shared = {key: value for key, value in cfg.items() if key != "partition"}
        if common_config is None:
            common_config = shared
        elif shared != common_config:
            raise ValueError("forward partitions use different retrieval policies or source indexes")
        position = 0
        for name, shard in sorted(manifest["shards"].items(), key=lambda item: item[1]["start"]):
            path = folder / name
            if shard["start"] != position or shard["end"] - shard["start"] != shard["queries"]:
                raise ValueError(f"forward query range gap in {path}")
            if not path.is_file() or sha(path) != shard["sha256"] or pq.ParquetFile(path).metadata.num_rows != shard["pairs"]:
                raise ValueError(f"corrupt forward shard {path}")
            position = shard["end"]
            total_pairs += shard["pairs"]
        if position != manifest["queries_this_run"]:
            raise ValueError(f"forward partition query count differs in {folder}")
        total_queries += position
        hashes[folder.name] = sha(manifest_path)
    if seen != set(range(partition_count)) or (expected_queries is not None and total_queries != expected_queries):
        raise ValueError("forward output does not cover the expected query count or partitions")
    return {"split": split, "queries": total_queries, "pairs": total_pairs,
            "partitions": partition_count, "k_per_field": common_config["k_per_field"],
            "partition_manifest_sha256": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["build", "retrieve"])
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--source", choices=["S2", "S3"])
    parser.add_argument("--root", type=Path, default=Path("artifacts/forward-fts-v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.operation == "build":
        if args.source is None:
            parser.error("build requires --source")
        result = build(args.root, args.split, args.source)
    else:
        if args.output is None:
            parser.error("retrieve requires --output")
        result = retrieve(args.root, args.split, args.output, args.k, args.batch_size,
                          args.partition, args.partitions, args.limit)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
