"""Resumable character-trigram source-name index over full S2/S3 backgrounds."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from .forward_fts import index_path as base_path
from .retrieval_experiments import sha, write_json

VERSION = "forward-name-trigram-v1"


def index_path(root: Path, split: str, source: str) -> Path:
    return root / f"{split}_{source}_trigram.sqlite"


def build(root: Path, split: str, source: str, batch_size: int = 20_000) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    base = base_path(Path("artifacts/forward-fts-v2"), split, source)
    base_meta = json.loads(base.with_suffix(".json").read_text())
    if not base_meta.get("complete"):
        raise ValueError("base full-background forward index incomplete")
    target = index_path(root, split, source)
    marker = target.with_suffix(".json")
    config = {"version": VERSION, "split": split, "source": source,
              "base_sha256": base_meta["sha256"], "rows": base_meta["rows"]}
    if marker.exists():
        meta = json.loads(marker.read_text())
        if not meta.get("complete") or meta.get("config") != config:
            raise ValueError("incompatible trigram index")
        return meta
    started = time.time()
    db = sqlite3.connect(target, uri=True)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("ATTACH DATABASE ? AS base", (f"file:{base.resolve()}?mode=ro",))
        db.execute("CREATE TABLE IF NOT EXISTS build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        old = db.execute("SELECT value FROM build_meta WHERE key='config'").fetchone()
        requested = json.dumps(config, sort_keys=True)
        if old and old[0] != requested:
            raise ValueError("incompatible partial trigram index")
        if not old:
            db.execute("CREATE VIRTUAL TABLE trigram USING fts5(name, tokenize='trigram')")
            db.execute("INSERT INTO build_meta VALUES ('config', ?)", (requested,))
            db.execute("INSERT INTO build_meta VALUES ('rows', '0')")
            db.commit()
        completed = int(db.execute("SELECT value FROM build_meta WHERE key='rows'").fetchone()[0])
        while completed < base_meta["rows"]:
            end = min(base_meta["rows"], completed + batch_size)
            db.execute("INSERT INTO trigram(rowid,name) SELECT rowid,name FROM base.search WHERE rowid>? AND rowid<=?", (completed, end))
            db.execute("UPDATE build_meta SET value=? WHERE key='rows'", (str(end),))
            db.commit()
            completed = end
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("PRAGMA journal_mode=DELETE")
    finally:
        db.close()
    meta = {"config": config, "complete": True, "rows": completed, "bytes": target.stat().st_size,
            "sha256": sha(target), "seconds_this_run": time.time() - started}
    write_json(marker, meta)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=["build"])
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--source", choices=["S2", "S3"], required=True)
    p.add_argument("--root", type=Path, default=Path("artifacts/forward-trigram-v1"))
    a = p.parse_args()
    print(json.dumps(build(a.root, a.split, a.source)))


if __name__ == "__main__":
    main()
