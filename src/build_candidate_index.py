"""Explicit future-run source TF-IDF index builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_index import DiskBackedTfidfIndex, ExactMatchIndex, IndexConfig, SourceTfidfIndex, SqliteExactMatchIndex
from .candidates import load_source
from .neural_contracts import ArtifactManifest, atomic_write_json, config_hash, file_identity, git_commit, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable source candidate indexes; expensive and execute-only.")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.dry_run and args.execute: parser.error("--dry-run and --execute are mutually exclusive")
    if not args.execute:
        print(json.dumps({"command": "build_candidate_index", "execute_required": True, "configuration_hash": config_hash(config), "inputs": {key: file_identity(value.get("path", "")) for key, value in config.get("sources", {}).items()}, "memory_estimate": "computed only during explicit index build"}, indent=2)); return
    output = Path(config.get("output_dir", "artifacts/candidate-index"))
    output.mkdir(parents=True, exist_ok=True)
    built = {}
    input_identities = {}
    backend = str(config.get("index_backend", "disk_backed"))
    for source_name, source_config in config["sources"].items():
        source_identity = file_identity(source_config["path"], hash_content=True)
        input_identities[source_name] = source_identity
        frame = None
        for field in source_config.get("fields", ["business_name", "business_address"]):
            index_config = IndexConfig(field=field, top_k=int(config.get("top_k", 100)), max_vocabulary=config.get("max_vocabulary"), memory_budget_mb=config.get("memory_budget_mb"), batch_size=int(config.get("query_batch_size", 128)), sparse_threads=int(config.get("sparse_threads", 2)))
            location = output / f"{source_name}_{field}"
            if backend in {"disk", "disk_backed"}:
                index = DiskBackedTfidfIndex.build_from_tsv(
                    source_config["path"], location, index_config, candidate_source=source_name,
                    source_identity=source_identity, source_chunk_rows=int(config.get("source_read_chunk_rows", 25_000)),
                    resume=bool(config.get("resume", True)),
                )
                matrix_mb = sum(int(spec.get("resident_bytes", 0)) for spec in index.shard_specs) / 1024**2
            else:
                if (location / "metadata.json").exists() and (location / "index.pkl").exists():
                    index = SourceTfidfIndex.load(location, expected_config=index_config, expected_source_identity=source_identity)
                else:
                    if location.exists() and any(location.iterdir()):
                        raise ValueError(f"incomplete legacy index preserved at {location}; validate or rebuild into a new directory")
                    if frame is None:
                        frame = load_source(source_config["path"])
                    index = SourceTfidfIndex(frame, index_config, source_identity=source_identity)
                    index.save(location)
                matrix_mb = index.metadata["estimated_matrix_mb"]
            if backend in {"disk", "disk_backed"}:
                exact_path = location / "exact_sqlite.db"
                exact_metadata = exact_path.with_suffix(".json")
                if exact_path.exists() and exact_metadata.exists():
                    value = json.loads(exact_metadata.read_text(encoding="utf-8"))
                    if value.get("source_identity") != source_identity or value.get("field") != field or value.get("candidate_source") != source_name:
                        raise ValueError(f"incompatible exact index exists: {exact_path}")
                else:
                    SqliteExactMatchIndex.build_from_tsv(source_config["path"], exact_path, field, candidate_source=source_name, source_identity=source_identity, chunk_rows=int(config.get("source_read_chunk_rows", 25_000)), resume=bool(config.get("resume", True)))
                exact_kind = "sqlite"
                frequent = None
            else:
                if (location / "exact_index.pkl").exists() and (location / "exact_metadata.json").exists():
                    exact_metadata = json.loads((location / "exact_metadata.json").read_text(encoding="utf-8"))
                    if exact_metadata.get("source_identity") != source_identity:
                        raise ValueError(f"legacy exact index has missing/incompatible source identity: {location}; preserve it and rebuild exact postings in a new versioned location")
                    exact = ExactMatchIndex.load(location)
                else:
                    if frame is None:
                        frame = load_source(source_config["path"])
                    exact = ExactMatchIndex(frame, field, candidate_source=source_name, source_identity=source_identity)
                    exact.save(location)
                exact_kind = "pickle"
                frequent = len(exact.frequent_keys)
            built[f"{source_name}:{field}"] = {"path": str(location), "backend": backend, "matrix_mb": matrix_mb, "exact_backend": exact_kind, "frequent_exact_keys": frequent}
            atomic_write_json(output / "build_progress.json", {"completion": "in_progress", "configuration_hash": config_hash(config), "built": built, "input_identities": input_identities})
    manifest = ArtifactManifest(kind="candidate-index", config=config, input_identities=input_identities, completion="complete", git_commit=git_commit(), seed=config.get("seed"))
    manifest.shard_order = sorted(built)
    for path in output.rglob("*"):
        if path.is_file() and path.name not in {"manifest.json", "build_progress.json"}:
            manifest.generated_file_checksums[path.relative_to(output).as_posix()] = sha256_file(path)
    atomic_write_json(output / "manifest.json", manifest.to_dict())
    atomic_write_json(output / "build_progress.json", {"completion": "complete", "configuration_hash": config_hash(config), "built": built, "input_identities": input_identities})
    print(json.dumps({"indexes": built, "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
