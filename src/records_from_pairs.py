"""Rebuild a records file (B's format) from text-bearing pair shards.

C's gate/route shards are key-only. For pairs drawn from the historical pair
shards, the exact model text already exists in those shards; this writes it
as ``source, record_id, business_name, business_address, country`` so the
scorer's ``records`` option can join it. The round trip through
``record_text`` reproduces the pair-shard text exactly. A record that appears
with two different texts is an error, not a silent pick.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .neural_contracts import atomic_write_json, file_identity, manifest_shard_paths, sha256_file

COLUMNS = ["source", "record_id", "business_name", "business_address", "country"]


def records_from_pairs(paths: list[str]) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id", "candidate_source",
                                               "name_a", "address_a", "country_a", "name_b", "address_b", "country_b"])
        left = frame[["source1_entity_id", "name_a", "address_a", "country_a"]].drop_duplicates()
        left.columns = COLUMNS[1:]; left.insert(0, "source", "S1")
        right = frame[["candidate_source", "candidate_entity_id", "name_b", "address_b", "country_b"]].drop_duplicates()
        right.columns = COLUMNS
        parts += [left, right]
    records = pd.concat(parts, ignore_index=True).astype(str).drop_duplicates()
    conflicts = records[records.duplicated(["source", "record_id"], keep=False)]
    if len(conflicts):
        raise ValueError(f"records with conflicting text across pair shards: {conflicts.head(4).to_dict(orient='records')}")
    return records.sort_values(["source", "record_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a records parquet from text-bearing pair shards.")
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--pair-subdir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.pair_manifest).read_text(encoding="utf-8"))
    paths = manifest_shard_paths(args.pair_manifest, manifest.get("shard_order", []), args.pair_subdir)
    records = records_from_pairs(paths)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    records.to_parquet(temporary, index=False); temporary.replace(output)
    summary = {"output": str(output), "rows": len(records), "by_source": records["source"].value_counts().to_dict(),
               "sha256": sha256_file(output), "pair_manifest": file_identity(args.pair_manifest, hash_content=True)}
    atomic_write_json(Path(f"{output}.complete.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
