"""Explicit future-run source TF-IDF index builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_index import ExactMatchIndex, IndexConfig, SourceTfidfIndex
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
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty index directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    built = {}
    input_identities = {}
    for source_name, source_config in config["sources"].items():
        frame = load_source(source_config["path"])
        source_identity = file_identity(source_config["path"], hash_content=True)
        input_identities[source_name] = source_identity
        for field in source_config.get("fields", ["business_name", "business_address"]):
            index_config = IndexConfig(field=field, top_k=int(config.get("top_k", 100)), max_vocabulary=config.get("max_vocabulary"), memory_budget_mb=config.get("memory_budget_mb"), batch_size=int(config.get("query_batch_size", 2048)))
            index = SourceTfidfIndex(frame, index_config, source_identity=source_identity)
            exact = ExactMatchIndex(frame, field, candidate_source=source_name)
            location = output / f"{source_name}_{field}"
            index.save(location)
            exact.save(location)
            built[f"{source_name}:{field}"] = {"path": str(location), "matrix_mb": index.metadata["estimated_matrix_mb"], "frequent_exact_keys": len(exact.frequent_keys)}
    manifest = ArtifactManifest(kind="candidate-index", config=config, input_identities=input_identities, completion="complete", git_commit=git_commit(), seed=config.get("seed"))
    manifest.shard_order = sorted(built)
    for path in output.rglob("*"):
        if path.is_file():
            manifest.generated_file_checksums[path.relative_to(output).as_posix()] = sha256_file(path)
    atomic_write_json(output / "manifest.json", manifest.to_dict())
    print(json.dumps({"indexes": built, "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
