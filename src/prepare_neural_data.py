"""Future-run preparation of query manifests and identical raw pair shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .candidate_io import GroupedParquetWriter, iter_complete_groups
from .neural_contracts import ArtifactManifest, atomic_write_json, config_hash, file_identity, git_commit, sha256_file
from .neural_data import parse_ground_truth, read_text_tsv, select_query_ids, select_training_pairs, stable_unmatched_fold
from .source_store import SourceStore, make_pair_group


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def plan(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    return {"command": "prepare_neural_data", "configuration_hash": config_hash(config), "execute_required": True, "dataset_iteration": "deferred", "inputs": {key: file_identity(value) for key, value in data.items() if isinstance(value, str)}, "requested_queries": config.get("queries", {}), "stages": ["queries", "pairs"]}


def _split_heldout(groups: list[list[str]], target: int, seed: int) -> tuple[list[str], list[str]]:
    import hashlib
    def rank(group: list[str]) -> str:
        return hashlib.sha256(f"{seed}:{','.join(sorted(group))}".encode()).hexdigest()
    threshold: list[str] = []
    final: list[str] = []
    for group in sorted((list(map(str, group)) for group in groups), key=rank):
        if abs((len(threshold) + len(group)) - target) <= abs(len(threshold) - target):
            threshold.extend(group)
        else:
            final.extend(group)
    return threshold, final


def _query_payload(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    data = config["data"]
    s1 = read_text_tsv(data["s1"])
    gt = parse_ground_truth(data["ground_truth"])
    folds_frame = read_text_tsv(data["folds"])
    if folds_frame["entity_id"].duplicated().any():
        raise ValueError("fold map contains duplicate entity_id values")
    fold_map = {str(row.entity_id): int(row.fold) for row in folds_frame.itertuples(index=False)}
    queries = config.get("queries", {})
    train_ids, train_report = select_query_ids(s1, gt, fold_map=fold_map, fold=queries.get("train_folds", [1, 2]), requested=int(queries.get("train", 50_000)), seed=int(config.get("seed", 42)))
    heldout_ids, heldout_report = select_query_ids(s1, gt, fold_map=fold_map, fold=int(queries.get("heldout_fold", 0)), requested=int(queries.get("heldout", 5_000)), seed=int(config.get("seed", 42)) + 1)
    threshold_target = int(queries.get("threshold", len(heldout_ids) // 2))
    threshold_ids, final_ids = _split_heldout(heldout_report.get("component_groups", []), threshold_target, int(config.get("seed", 42)) + 2)
    selected = set(train_ids) | set(threshold_ids) | set(final_ids)
    metadata = {}
    s1_index = s1.set_index(s1["entity_id"].astype(str), drop=False)
    for sid in selected:
        row = s1_index.loc[sid]
        matches = list(map(str, gt.get(sid, [])))
        metadata[sid] = {"country": str(row.get("country", "")), "match_count": len(matches), "singleton": len(matches) == 0, "fold": int(fold_map[sid])}
    identities = {key: file_identity(data[key], hash_content=True) for key in ("s1", "ground_truth", "folds")}
    payload = {"schema_version": "neural-query-manifest-1", "configuration_hash": config_hash(config), "input_identities": identities, "fold_map_identity": identities["folds"], "seed": int(config.get("seed", 42)), "query_ids": {"train": train_ids, "threshold": threshold_ids, "final": final_ids}, "query_metadata": metadata, "reports": {"train": train_report, "heldout": heldout_report, "threshold_requested": threshold_target, "threshold_actual": len(threshold_ids), "final_actual": len(final_ids), "component_safe_split": True}, "source_pool_policy": config.get("source_pool_policy", "full verified retrieval background; labels only select training pairs")}
    return payload, {"train": train_ids, "threshold": threshold_ids, "final": final_ids}, gt, fold_map


def _candidate_paths(config: dict[str, Any]) -> list[str]:
    if config.get("candidate_shards"):
        paths = [str(value) for value in config["candidate_shards"]]
    elif config.get("candidate_dir"):
        paths = [str(path) for path in sorted(Path(config["candidate_dir"]).glob("*.parquet"))]
    else:
        paths = []
    if not paths:
        raise FileNotFoundError("no candidate shards configured; run query_candidate_index first")
    return paths


def _write_pair_manifests(output: Path, writers: dict[str, GroupedParquetWriter], config: dict[str, Any], inputs: dict[str, Any], query_ids: dict[str, list[str]], fold_identity: Any) -> None:
    for subset, writer in writers.items():
        manifest = ArtifactManifest(kind=f"neural-pairs-{subset}", config=config, input_identities=inputs, fold_map_identity=fold_identity, query_ids=query_ids[subset], shard_order=[Path(path).name for path in writer.files], row_counts=writer.row_counts, completion="complete", git_commit=git_commit(), seed=config.get("seed"))
        manifest.generated_file_checksums = {Path(path).name: sha256_file(path) for path in writer.files}
        atomic_write_json(output / f"{subset}_pairs_manifest.json", manifest.to_dict())


def execute(config: dict[str, Any], *, queries_only: bool = False) -> dict[str, Any]:
    payload, query_ids, gt, fold_map = _query_payload(config)
    output = Path(config.get("output_dir", "artifacts/neural-data"))
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "query_manifest.json"
    if query_path.exists():
        if json.loads(query_path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"existing query manifest does not match configuration/inputs: {query_path}")
    else:
        atomic_write_json(query_path, payload)
    if queries_only:
        return {"query_manifest": str(query_path), "train": len(query_ids["train"]), "threshold": len(query_ids["threshold"]), "final": len(query_ids["final"]), "pairs_prepared": False}

    candidate_paths = _candidate_paths(config)
    store_path = config.get("source_store")
    if not store_path or not Path(store_path).exists():
        raise FileNotFoundError("source_store is missing; run build_source_store first")
    store = SourceStore(store_path)
    target_rows = int(config.get("shard_target_rows", 75_000))
    writers = {subset: GroupedParquetWriter(output / f"{subset}_pairs", target_rows=target_rows, prefix="pairs") for subset in query_ids}
    subset_by_id = {sid: subset for subset, ids in query_ids.items() for sid in ids}
    seen: set[str] = set()
    reports = {"train": {"queries": 0, "rows": 0, "injected_positives": 0, "excluded_heldout_negatives": 0}, "threshold": {"queries": 0, "rows": 0}, "final": {"queries": 0, "rows": 0}}

    def write_one(sid: str, subset: str, candidates: pd.DataFrame) -> None:
        if subset == "train":
            selected, report = select_training_pairs(candidates, [sid], gt, seed=int(config.get("seed", 42)), fold_map=fold_map, held_out_folds={int(config.get("queries", {}).get("heldout_fold", 0))}, fold_seed=int(config.get("fold_seed", 42)), candidate_source_resolver=store.resolve_source)
            for key in reports["train"]:
                reports["train"][key] += int(report.get(key, 0))
        else:
            selected = candidates.copy()
            reports[subset]["queries"] += 1
            reports[subset]["rows"] += len(selected)
        if selected.empty:
            return
        local_folds = dict(fold_map)
        positives = set(map(str, gt.get(sid, [])))
        for candidate_id in selected["candidate_entity_id"].astype(str):
            if candidate_id not in local_folds:
                if candidate_id in positives:
                    raise ValueError(f"known-positive endpoint missing from verified fold map: {candidate_id}")
                local_folds[candidate_id] = stable_unmatched_fold(candidate_id, seed=int(config.get("fold_seed", 42)))
        left = store.lookup("S1", [sid])
        pair_group = make_pair_group(selected, left, store, folds=local_folds, ground_truth=gt)
        writers[subset].write_group(pair_group)

    for group in iter_complete_groups(candidate_paths):
        sid = str(group.iloc[0]["source1_entity_id"])
        subset = subset_by_id.get(sid)
        if subset is None:
            continue
        if sid in seen:
            raise ValueError(f"candidate group repeated for selected query: {sid}")
        seen.add(sid)
        write_one(sid, subset, group)
    empty_columns = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    for sid in query_ids["train"]:
        if sid not in seen:
            write_one(sid, "train", pd.DataFrame(columns=empty_columns))
    for writer in writers.values():
        writer.close()
    inputs = {"source_store": file_identity(store_path, hash_content=True), "query_manifest": file_identity(query_path, hash_content=True), **{f"candidate_{index:06d}": file_identity(path, hash_content=True) for index, path in enumerate(candidate_paths)}}
    _write_pair_manifests(output, writers, config, inputs, query_ids, payload["fold_map_identity"])
    return {"query_manifest": str(query_path), "pairs_prepared": True, "pair_manifests": {subset: str(output / f"{subset}_pairs_manifest.json") for subset in writers}, "reports": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic neural query manifests and raw pair shards.")
    parser.add_argument("--config", required=True, help="JSON configuration")
    parser.add_argument("--dry-run", action="store_true", help="validate configuration and print metadata only")
    parser.add_argument("--execute", action="store_true", help="run data preparation on a compute machine")
    parser.add_argument("--queries-only", action="store_true", help="write only the selection manifest before candidate retrieval")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if args.queries_only and not args.execute:
        parser.error("--queries-only requires --execute")
    print(json.dumps(plan(config) if not args.execute else execute(config, queries_only=args.queries_only), indent=2))


if __name__ == "__main__":
    main()
