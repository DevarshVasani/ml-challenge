"""Future-run preparation of query manifests and identical raw pair shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .candidate_index import validate_channel_coverage
from .neural_contracts import ArtifactManifest, atomic_write_json, config_hash, file_identity, git_commit, sha256_file
from .neural_data import _component_ids, parse_ground_truth, read_text_tsv, select_query_ids, select_training_pairs, stable_unmatched_fold
from .source_store import SourceStore, make_pair_group


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    return value


def plan(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    return {"command": "prepare_neural_data", "configuration_hash": config_hash(config), "execute_required": True, "dataset_iteration": "deferred", "inputs": {key: file_identity(value) for key, value in data.items() if isinstance(value, str)}, "requested_queries": config.get("queries", {}), "stages": ["queries", "pairs"]}


def _selection_hash(config: dict[str, Any]) -> str:
    return config_hash({key: config.get(key) for key in ("data", "queries", "seed", "fold_seed", "source_pool_policy")})


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
    prepared_gt = {str(key): set(map(str, values)) for key, values in gt.items()}
    components = _component_ids(s1["entity_id"].astype(str), prepared_gt, fold_map)
    train_ids, train_report = select_query_ids(s1, gt, fold_map=fold_map, fold=queries.get("train_folds", [1, 2]), requested=int(queries.get("train", 50_000)), seed=int(config.get("seed", 42)), prepared_ground_truth=prepared_gt, prepared_components=components)
    heldout_ids, heldout_report = select_query_ids(s1, gt, fold_map=fold_map, fold=int(queries.get("heldout_fold", 0)), requested=int(queries.get("heldout", 5_000)), seed=int(config.get("seed", 42)) + 1, prepared_ground_truth=prepared_gt, prepared_components=components)
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
    payload = {"schema_version": "neural-query-manifest-1", "configuration_hash": _selection_hash(config), "input_identities": identities, "fold_map_identity": identities["folds"], "fold_policy": {"algorithm": "sha256-prefix64-modulo", "seed": int(config.get("fold_seed", 42)), "n_folds": 3}, "seed": int(config.get("seed", 42)), "query_ids": {"train": train_ids, "threshold": threshold_ids, "final": final_ids}, "query_metadata": metadata, "reports": {"train": train_report, "heldout": heldout_report, "threshold_requested": threshold_target, "threshold_actual": len(threshold_ids), "final_actual": len(final_ids), "component_safe_split": True}, "source_pool_policy": config.get("source_pool_policy", "full verified retrieval background; labels only select training pairs")}
    return payload, {"train": train_ids, "threshold": threshold_ids, "final": final_ids}, gt, fold_map


def _candidate_run(config: dict[str, Any], expected_ids: list[str]) -> tuple[list[tuple[str, list[str], str]], dict[str, Any]]:
    root = Path(config.get("candidate_dir", ""))
    manifest_path = Path(config.get("candidate_manifest", root / "manifest.json"))
    channel_path = root / "channel_manifest.json"
    if not manifest_path.exists() or not channel_path.exists():
        raise FileNotFoundError("candidate completion manifests are required; run query_candidate_index first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = json.loads(channel_path.read_text(encoding="utf-8"))
    if manifest.get("completion") != "complete" or channels.get("completion") != "complete":
        raise ValueError("candidate retrieval is incomplete")
    if list(map(str, manifest.get("query_ids", []))) != expected_ids:
        raise ValueError("candidate manifest query coverage/order does not match the frozen query manifest")
    validate_channel_coverage(channels, expected_ids, channels.get("channels", []))
    batches = []
    for batch_id in sorted(channels.get("completed", {})):
        artifact = channels.get("artifacts", {}).get(batch_id, {}).get("merged")
        if not artifact:
            raise ValueError(f"candidate batch has no durable artifact: {batch_id}")
        path = root / str(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"candidate batch is missing or corrupt: {path}")
        start, end = (int(value) for value in batch_id.split("-"))
        batches.append((batch_id, expected_ids[start:end], str(path)))
    covered = [sid for _, values, _ in batches for sid in values]
    if covered != expected_ids:
        raise ValueError("candidate batch coverage is incomplete")
    return batches, manifest


def _write_pair_manifests(output: Path, progress: dict[str, Any], config: dict[str, Any], inputs: dict[str, Any], query_ids: dict[str, list[str]], fold_identity: Any) -> None:
    for subset in query_ids:
        artifacts = [value for batch in sorted(progress["batches"]) for key, value in progress["batches"][batch].get("artifacts", {}).items() if key == subset and int(value.get("rows", 0)) > 0]
        manifest = ArtifactManifest(kind=f"neural-pairs-{subset}", config=config, input_identities=inputs, fold_map_identity=fold_identity, query_ids=query_ids[subset], shard_order=[str(value["path"]) for value in artifacts], row_counts={str(value["path"]): int(value["rows"]) for value in artifacts}, completion="complete", git_commit=git_commit(), seed=config.get("seed"))
        manifest.generated_file_checksums = {str(value["path"]): str(value["sha256"]) for value in artifacts}
        atomic_write_json(output / f"{subset}_pairs_manifest.json", manifest.to_dict())


def _atomic_pair_file(frame: pd.DataFrame, path: Path, output: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": path.relative_to(output).as_posix(), "rows": len(frame), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _publish_bootstrap(output: Path, progress: dict[str, Any], config: dict[str, Any], train_ids: list[str], inputs: dict[str, Any], fold_identity: Any) -> None:
    target = int(config.get("bootstrap_train_queries", 0))
    path = output / "bootstrap_pairs_manifest.json"
    if target <= 0 or path.exists():
        return
    train_set = set(train_ids)
    chosen_batches = []
    chosen_ids: list[str] = []
    completed_ids = 0
    for batch_id in sorted(progress["batches"]):
        value = progress["batches"][batch_id]
        ids = list(map(str, value.get("query_ids", [])))
        if value.get("complete") is not True or not ids or not set(ids).issubset(train_set):
            break
        completed_ids += len(ids)
        if len(chosen_ids) + len(ids) <= target or not chosen_ids:
            chosen_batches.append(batch_id)
            chosen_ids.extend(ids)
        if completed_ids >= target:
            break
    if completed_ids < target or not chosen_ids:
        return
    artifacts = [progress["batches"][batch].get("artifacts", {}).get("train") for batch in chosen_batches]
    artifacts = [value for value in artifacts if value and int(value.get("rows", 0)) > 0]
    manifest = ArtifactManifest(kind="neural-pairs-bootstrap", config=config, input_identities=inputs, fold_map_identity=fold_identity, query_ids=chosen_ids, shard_order=[str(value["path"]) for value in artifacts], row_counts={str(value["path"]): int(value["rows"]) for value in artifacts}, completion="complete", git_commit=git_commit(), seed=config.get("seed"))
    value = manifest.to_dict()
    value["generated_file_checksums"] = {str(item["path"]): str(item["sha256"]) for item in artifacts}
    value["parent_query_manifest_sha256"] = inputs["query_manifest"]["sha256"]
    value["immutable"] = True
    atomic_write_json(path, value)


def execute(config: dict[str, Any], *, queries_only: bool = False) -> dict[str, Any]:
    output = Path(config.get("output_dir", "artifacts/neural-data"))
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "query_manifest.json"
    if query_path.exists():
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        data = config["data"]
        identities = {key: file_identity(data[key], hash_content=True) for key in ("s1", "ground_truth", "folds")}
        if payload.get("configuration_hash") != _selection_hash(config) or payload.get("input_identities") != identities:
            raise ValueError(f"existing query manifest does not match selection configuration/inputs: {query_path}")
        query_ids = {key: list(map(str, payload.get("query_ids", {}).get(key, []))) for key in ("train", "threshold", "final")}
        gt = parse_ground_truth(data["ground_truth"])
        folds_frame = read_text_tsv(data["folds"])
        fold_map = {str(row.entity_id): int(row.fold) for row in folds_frame.itertuples(index=False)}
    else:
        payload, query_ids, gt, fold_map = _query_payload(config)
        atomic_write_json(query_path, payload)
    if queries_only:
        return {"query_manifest": str(query_path), "train": len(query_ids["train"]), "threshold": len(query_ids["threshold"]), "final": len(query_ids["final"]), "pairs_prepared": False}

    all_query_ids = [sid for subset in ("train", "threshold", "final") for sid in query_ids[subset]]
    candidate_batches, _ = _candidate_run(config, all_query_ids)
    store_path = config.get("source_store")
    if not store_path or not Path(store_path).exists():
        raise FileNotFoundError("source_store is missing; run build_source_store first")
    inputs = {
        "source_store": file_identity(store_path, hash_content=True),
        "query_manifest": file_identity(query_path, hash_content=True),
        "candidate_manifest": file_identity(Path(config.get("candidate_manifest", Path(config["candidate_dir"]) / "manifest.json")), hash_content=True),
    }
    lineage = {"configuration_hash": config_hash(config), "inputs": inputs}
    progress_path = output / "pair_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("lineage") != lineage:
            raise ValueError("existing pair preparation is incompatible with requested inputs/configuration")
    else:
        progress = {"schema_version": "neural-pair-progress-1", "lineage": lineage, "completion": "in_progress", "batches": {}}
        atomic_write_json(progress_path, progress)
    prepared_gt = {str(key): set(map(str, values)) for key, values in gt.items()}
    subset_by_id = {sid: subset for subset, ids in query_ids.items() for sid in ids}
    reports = {"train": {"queries": 0, "rows": 0, "injected_positives": 0, "excluded_heldout_negatives": 0}, "threshold": {"queries": 0, "rows": 0}, "final": {"queries": 0, "rows": 0}}
    empty_columns = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    with SourceStore(store_path) as store:
        for batch_id, batch_query_ids, candidate_path in candidate_batches:
            existing = progress["batches"].get(batch_id, {})
            if existing.get("complete") is True:
                valid = all((output / value["path"]).is_file() and sha256_file(output / value["path"]) == value["sha256"] for value in existing.get("artifacts", {}).values())
                if valid:
                    continue
            candidates = pd.read_parquet(candidate_path)
            groups = {str(key): value for key, value in candidates.groupby("source1_entity_id", sort=False)} if not candidates.empty else {}
            frames: dict[str, list[pd.DataFrame]] = {key: [] for key in query_ids}
            batch_reports = {"train": {"queries": 0, "rows": 0, "injected_positives": 0, "excluded_heldout_negatives": 0}, "threshold": {"queries": 0, "rows": 0}, "final": {"queries": 0, "rows": 0}}
            for sid in batch_query_ids:
                subset = subset_by_id[sid]
                group = groups.get(sid, pd.DataFrame(columns=empty_columns))
                positives = prepared_gt.get(sid, set())
                if subset == "train":
                    resolved = store.resolve_sources(positives)
                    selected, report = select_training_pairs(group, [sid], gt, seed=int(config.get("seed", 42)), fold_map=fold_map, held_out_folds={int(config.get("queries", {}).get("heldout_fold", 0))}, fold_seed=int(config.get("fold_seed", 42)), candidate_source_resolver=lambda value, resolved=resolved: resolved[str(value)], prepared_ground_truth=prepared_gt)
                    for key in batch_reports["train"]:
                        batch_reports["train"][key] += int(report.get(key, 0))
                else:
                    selected = group.copy()
                    batch_reports[subset]["queries"] += 1
                    batch_reports[subset]["rows"] += len(selected)
                if selected.empty:
                    continue
                local_folds = {sid: fold_map[sid]}
                for candidate_id in selected["candidate_entity_id"].astype(str):
                    if candidate_id in fold_map:
                        local_folds[candidate_id] = fold_map[candidate_id]
                    elif candidate_id in positives:
                        raise ValueError(f"known-positive endpoint missing from verified fold map: {candidate_id}")
                    else:
                        local_folds[candidate_id] = stable_unmatched_fold(candidate_id, seed=int(config.get("fold_seed", 42)))
                pair_group = make_pair_group(selected, store.lookup("S1", [sid]), store, folds=local_folds, ground_truth=prepared_gt)
                frames[subset].append(pair_group)
            artifacts = {}
            for subset, values in frames.items():
                if not values:
                    continue
                frame = pd.concat(values, ignore_index=True)
                artifacts[subset] = _atomic_pair_file(frame, output / f"{subset}_pairs" / f"pairs-{batch_id}.parquet", output)
            progress["batches"][batch_id] = {"complete": True, "query_ids": batch_query_ids, "artifacts": artifacts, "reports": batch_reports}
            atomic_write_json(progress_path, progress)
            _publish_bootstrap(output, progress, config, query_ids["train"], inputs, payload["fold_map_identity"])
    for value in progress["batches"].values():
        for subset, subset_report in value.get("reports", {}).items():
            for key, count in subset_report.items():
                reports[subset][key] += int(count)
    progress["completion"] = "complete"
    atomic_write_json(progress_path, progress)
    _write_pair_manifests(output, progress, config, inputs, query_ids, payload["fold_map_identity"])
    return {"query_manifest": str(query_path), "pairs_prepared": True, "pair_manifests": {subset: str(output / f"{subset}_pairs_manifest.json") for subset in query_ids}, "reports": reports}


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
