"""Explicit future-run builder for the bounded SQLite raw-text join store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .neural_contracts import ArtifactManifest, atomic_write_json, config_hash, file_identity, git_commit, sha256_file
from .source_store import SourceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the disk-backed source store; expensive and execute-only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths = {str(key): str(value) for key, value in config.get("sources", {}).items()}
    if not args.execute:
        print(json.dumps({"command": "build_source_store", "execute_required": True, "configuration_hash": config_hash(config), "inputs": {key: file_identity(value) for key, value in paths.items()}, "dataset_iteration": "deferred"}, indent=2))
        return
    if not paths:
        raise ValueError("sources must map source labels (S1/S2/S3) to TSV paths")
    output = Path(config["output"])
    identities = {key: file_identity(value, hash_content=True) for key, value in paths.items()}
    SourceStore.build(paths, output, batch_size=int(config.get("batch_size", 10_000)))
    manifest = ArtifactManifest(kind="source-store", config=config, input_identities=identities, completion="complete", git_commit=git_commit())
    manifest.row_counts = {"sqlite_bytes": output.stat().st_size}
    manifest.generated_file_checksums[output.name] = sha256_file(output)
    atomic_write_json(f"{output}.manifest.json", manifest.to_dict())
    print(json.dumps({"output": str(output), "manifest": f"{output}.manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
