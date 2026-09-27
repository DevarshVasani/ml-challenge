"""Publish a small, immutable Member B handoff from completed neural pair shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


KEY = ["candidate_source", "candidate_entity_id", "source1_entity_id"]
PAIR_COLUMNS = KEY + [
    "name_tfidf_score", "address_tfidf_score", "name_rank", "address_rank",
    "retrieved_by_name", "retrieved_by_address", "retrieval_score",
    "retrieved_by_exact", "retrieval_provenance", "frequent_key_expansion",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-data", type=Path, default=Path("../neural-data"))
    parser.add_argument("--truth", type=Path, default=Path("../student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/member-b-starter-v1"))
    parser.add_argument("--train-queries", type=int, default=128)
    parser.add_argument("--threshold-queries", type=int, default=64)
    args = parser.parse_args()
    root = args.neural_data.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    query_manifest_path = root / "query_manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    split_ids = query_manifest["query_ids"]
    if any(set(split_ids[a]) & set(split_ids[b]) for a, b in (("train", "threshold"), ("train", "final"), ("threshold", "final"))):
        raise ValueError("Query splits overlap")

    outputs: dict[str, dict] = {}
    for split, ids in split_ids.items():
        rows = [{"source1_entity_id": sid, "split": split,
                 "country": query_manifest["query_metadata"][sid]["country"]} for sid in ids]
        path = out / f"{split}_queries.tsv"
        write_tsv(pd.DataFrame(rows), path)
        outputs[path.name] = {"rows": len(rows), "sha256": sha256(path)}

    evaluation_ids = set(split_ids["threshold"]) | set(split_ids["final"])
    truth_rows = []
    with args.truth.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid = row["source1_entity_id"]
            if sid not in evaluation_ids:
                continue
            matches = [value for value in row["matched_entity_ids"].split(",") if value]
            truth_rows.append({"source1_entity_id": sid,
                               "split": "threshold" if sid in split_ids["threshold"] else "final",
                               "true_match_count": len(matches),
                               "matched_entity_ids": ",".join(matches)})
    if len(truth_rows) != len(evaluation_ids):
        raise ValueError(f"Evaluation truth coverage {len(truth_rows)} != {len(evaluation_ids)}")
    truth = pd.DataFrame(truth_rows).sort_values(["split", "source1_entity_id"])
    truth_path = out / "evaluation_truth.tsv"
    write_tsv(truth, truth_path)
    outputs[truth_path.name] = {"rows": len(truth), "sha256": sha256(truth_path)}

    for split, limit in (("train", args.train_queries), ("threshold", args.threshold_queries)):
        manifest_path = root / f"{split}_pairs_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["completion"] != "complete":
            raise ValueError(f"{split} pair manifest incomplete")
        frames = []
        available_ids = []
        source_shards = []
        for shard in manifest["shard_order"]:
            shard_path = root / shard
            checksum = sha256(shard_path)
            if checksum != manifest["generated_file_checksums"][shard]:
                raise ValueError(f"Checksum mismatch: {shard_path}")
            part = pd.read_parquet(shard_path)
            frames.append(part)
            available_ids.extend(part["source1_entity_id"].astype(str).unique())
            source_shards.append({"path": str(shard_path), "sha256": checksum})
            if len(set(available_ids)) >= limit:
                break
        selected_ids = list(dict.fromkeys(available_ids))[:limit]
        frame = pd.concat(frames, ignore_index=True)
        frame = frame[frame["source1_entity_id"].isin(selected_ids)].copy()
        if frame.duplicated(KEY).any():
            raise ValueError(f"Duplicate {split} pair keys")
        if not set(selected_ids) <= set(split_ids[split]):
            raise ValueError(f"{split} shard has wrong split IDs")
        if split == "train":
            labels = frame[KEY + ["label", "positive_injected_for_training"]].copy()
            label_path = out / "train_labels.parquet"
            write_parquet(labels, label_path)
            outputs[label_path.name] = {"rows": len(labels), "sha256": sha256(label_path)}
        records_a = frame[["source1_entity_id", "name_a", "address_a", "country_a"]].rename(
            columns={"source1_entity_id": "record_id", "name_a": "business_name",
                     "address_a": "business_address", "country_a": "country"})
        records_a.insert(0, "source", "S1")
        records_b = frame[["candidate_source", "candidate_entity_id", "name_b", "address_b", "country_b"]].rename(
            columns={"candidate_source": "source", "candidate_entity_id": "record_id",
                     "name_b": "business_name", "address_b": "business_address", "country_b": "country"})
        records = pd.concat([records_a, records_b], ignore_index=True).drop_duplicates(["source", "record_id"])
        records_path = out / f"{split}_records.parquet"
        write_parquet(records, records_path)
        outputs[records_path.name] = {"rows": len(records), "sha256": sha256(records_path)}
        pairs = frame[PAIR_COLUMNS].copy()
        pairs.insert(len(KEY), "candidate_version", "historical-neural-pairs-1")
        pairs_path = out / f"{split}_candidate_pairs.parquet"
        write_parquet(pairs, pairs_path)
        outputs[pairs_path.name] = {"rows": len(pairs), "sha256": sha256(pairs_path)}
        sample_queries_path = out / f"{split}_sample_queries.tsv"
        write_tsv(pd.DataFrame({"source1_entity_id": selected_ids}), sample_queries_path)
        outputs[sample_queries_path.name] = {"rows": len(selected_ids), "sha256": sha256(sample_queries_path)}
        outputs[f"{split}_sample"] = {"query_count": len(selected_ids), "source_shards": source_shards,
                                       "complete_by_s1": True, "complete_by_source_record": False}

    package = {"schema_version": "member-b-starter-1", "completion": "complete",
               "query_manifest": str(query_manifest_path), "query_manifest_sha256": sha256(query_manifest_path),
               "pair_key": KEY, "outputs": outputs,
               "warning": "Bootstrap pairs are S1-grouped and do not provide complete source-record competition groups. Evaluation truth and training labels are separate from candidate features."}
    temporary = out / "manifest.json.tmp"
    temporary.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    temporary.replace(out / "manifest.json")
    print(json.dumps({"out": str(out), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
