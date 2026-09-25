"""Resumable, bounded pair scoring for a completed adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .neural_adapters import get_adapter_class
from .neural_contracts import ArtifactManifest, atomic_write_json, atomic_write_text, config_hash, directory_identity, file_identity, git_commit, sha256_file, validate_pair_frame, validate_prediction_frame
from .neural_models import predict_pairs


def estimate_inference_seconds(pair_count: int, measured_pairs_per_second: float | None) -> float | None:
    if measured_pairs_per_second is None:
        return None
    if measured_pairs_per_second <= 0:
        raise ValueError("measured_pairs_per_second must be positive")
    return int(pair_count) / float(measured_pairs_per_second)


def _inputs(config: dict[str, Any], *, hash_content: bool) -> tuple[list[str], dict[str, Any], list[str]]:
    manifest_path = config.get("pair_manifest")
    query_ids: list[str] = []
    identities: dict[str, Any] = {}
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        base = Path(manifest_path).parent / str(config.get("pair_subdir", "final_pairs"))
        paths = [str(base / value) for value in manifest.get("shard_order", [])]
        query_ids = list(map(str, manifest.get("query_ids", [])))
        identities["pair_manifest"] = file_identity(manifest_path, hash_content=hash_content)
    else:
        raw = config.get("pairs")
        if isinstance(raw, list):
            paths = list(map(str, raw))
        elif raw and Path(raw).is_dir():
            paths = [str(path) for path in sorted(Path(raw).glob("*.parquet"))]
        elif raw:
            paths = [str(raw)]
        else:
            paths = []
        identities.update({f"pair_{index:06d}": file_identity(path, hash_content=hash_content) for index, path in enumerate(paths)})
    query_manifest = config.get("query_manifest")
    if query_manifest:
        value = json.loads(Path(query_manifest).read_text(encoding="utf-8"))
        subset = str(config.get("query_subset", "final"))
        query_ids = list(map(str, value.get("query_ids", {}).get(subset, [])))
        identities["query_manifest"] = file_identity(query_manifest, hash_content=hash_content)
    candidate_manifest = config.get("candidate_manifest")
    if candidate_manifest:
        identities["candidate_manifest"] = file_identity(candidate_manifest, hash_content=hash_content)
    return paths, identities, query_ids


def plan(config: dict[str, Any]) -> dict[str, Any]:
    pairs = None
    manifest_path = config.get("pair_manifest")
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        pairs = sum(int(value) for value in manifest.get("row_counts", {}).values())
    measured = config.get("measured_pairs_per_second")
    return {"command": "predict_neural", "execute_required": True, "model_initialization": "deferred", "dataset_iteration": "deferred", "network": "adapter-controlled only during --execute", "candidate_pairs_from_manifest": pairs, "measured_pairs_per_second": measured, "estimated_inference_seconds": estimate_inference_seconds(pairs, float(measured)) if pairs is not None and measured is not None else None}


def execute(config: dict[str, Any]) -> dict[str, Any]:
    adapter_type = str(config.get("adapter_type", ""))
    smoke = bool(config.get("smoke", False))
    if adapter_type.lower() in {"fake", "smoke"} and not smoke:
        raise RuntimeError("fake adapter requires explicit smoke=true and cannot produce production predictions")
    paths, identities, query_ids = _inputs(config, hash_content=True)
    if not paths or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("configured prediction pair shards are missing")
    checkpoint = Path(config["checkpoint"])
    adapter_path = checkpoint / "adapter" if (checkpoint / "adapter").is_dir() else checkpoint
    checkpoint_identity = directory_identity(checkpoint, hash_content=True)
    identities["checkpoint"] = checkpoint_identity
    expected = ArtifactManifest(kind="neural-predictions", config=config, input_identities=identities, query_ids=query_ids, shard_order=[Path(path).name for path in paths], completion="incomplete", git_commit=git_commit())
    output = Path(config["output_dir"])
    manifest_path = output / "prediction_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.exists():
            raise FileExistsError(f"refusing to use non-empty prediction directory without its manifest: {output}")
        actual = ArtifactManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if not actual.compatible_with(expected):
            raise ValueError("prediction resume identity changed (configuration, checkpoint, candidates, or query manifest)")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, expected.to_dict())
    adapter = get_adapter_class(adapter_type).load_pretrained(str(adapter_path), map_location=config.get("device", "cpu"))
    row_counts: dict[str, int] = {}
    checksums: dict[str, str] = {}
    resumed = 0
    scored_rows = 0
    elapsed = 0.0
    collate_seconds = 0.0
    forward_seconds = 0.0
    truncated_pairs = 0
    for index, pair_path in enumerate(paths):
        pair_identity = file_identity(pair_path, hash_content=True)
        final = output / f"scores-{index:06d}.parquet"
        sidecar = Path(f"{final}.complete.json")
        sidecar_expected = {"schema_version": "neural-prediction-shard-1", "configuration_hash": config_hash(config), "input": pair_identity, "checkpoint_identity_hash": checkpoint_identity.get("identity_hash")}
        if final.exists() and sidecar.exists():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in sidecar_expected.items()):
                raise ValueError(f"prediction shard identity changed; refusing unsafe resume: {final}")
            if saved.get("sha256") != sha256_file(final):
                raise ValueError(f"prediction shard checksum mismatch: {final}")
            row_counts[final.name] = int(saved["rows"]); checksums[final.name] = saved["sha256"]; resumed += 1
            continue
        pairs = pd.read_parquet(pair_path) if Path(pair_path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(pair_path, sep="\t", dtype=str, keep_default_na=False)
        validate_pair_frame(pairs, labeled=False)
        scores, timing = predict_pairs(adapter, pairs, batch_size=int(config.get("batch_size", 32)), device=str(config.get("device", "cpu")), return_metrics=True)
        elapsed += float(timing["seconds"]); collate_seconds += float(timing["collate_seconds"]); forward_seconds += float(timing["forward_seconds"]); truncated_pairs += int(timing["truncated_pairs"])
        expected_keys = set(map(tuple, pairs[["source1_entity_id", "candidate_entity_id", "candidate_source"]].astype(str).itertuples(index=False, name=None)))
        validate_prediction_frame(scores, expected_keys)
        temporary = final.with_suffix(".parquet.tmp")
        try:
            scores.to_parquet(temporary, index=False); os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True); raise
        digest = sha256_file(final)
        row_counts[final.name] = len(scores); checksums[final.name] = digest
        scored_rows += len(scores)
        atomic_write_json(sidecar, {**sidecar_expected, "rows": len(scores), "sha256": digest})
    expected.completion = "complete"; expected.row_counts = row_counts; expected.generated_file_checksums = checksums
    atomic_write_json(manifest_path, expected.to_dict())
    if smoke:
        atomic_write_text(output / "SMOKE_ONLY", "Offline fake-adapter scores; not production predictions.\n")
    rows = sum(row_counts.values())
    return {"output_dir": str(output), "rows": rows, "shards": len(paths), "resumed_shards": resumed, "seconds_this_run": elapsed, "pairs_per_second_this_run": scored_rows / elapsed if elapsed else None, "collate_seconds_this_run": collate_seconds, "forward_seconds_this_run": forward_seconds, "truncated_pairs_this_run": truncated_pairs, "production_eligible": not smoke}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score final candidate pair shards with a neural adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(plan(config) if not args.execute else execute(config), indent=2))


if __name__ == "__main__":
    main()
