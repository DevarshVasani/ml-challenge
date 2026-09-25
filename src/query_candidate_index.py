"""Query reusable TF-IDF/exact indexes into bounded complete-group shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_index import ExactMatchIndex, IndexConfig, SourceTfidfIndex, stream_query_candidates
from .neural_contracts import ArtifactManifest, atomic_write_json, config_hash, file_identity, git_commit, sha256_file
from .neural_data import read_text_tsv


def _selected_ids(config: dict) -> list[str] | None:
    manifest_path = config.get("query_manifest")
    if not manifest_path:
        return None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    subset = str(config.get("query_subset", "test"))
    query_sets = manifest.get("query_ids", {})
    values = ([sid for name in ("train", "threshold", "final") for sid in query_sets.get(name, [])] if subset == "all" else query_sets.get(subset))
    if values is None:
        raise ValueError(f"query manifest has no subset {subset!r}")
    return list(map(str, values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query cached candidate indexes; expensive and execute-only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps({"command": "query_candidate_index", "execute_required": True, "configuration_hash": config_hash(config), "query_input": file_identity(config.get("s1", "")), "index_root": file_identity(Path(config.get("index_root", "")) / "manifest.json"), "dataset_iteration": "deferred"}, indent=2))
        return
    query = read_text_tsv(config["s1"])
    selected = _selected_ids(config)
    if selected is not None:
        order = {sid: index for index, sid in enumerate(selected)}
        query = query[query["entity_id"].astype(str).isin(order)].copy()
        query["_order"] = query["entity_id"].astype(str).map(order)
        query = query.sort_values("_order").drop(columns="_order")
        missing = set(selected) - set(query["entity_id"].astype(str))
        if missing:
            raise ValueError(f"selected queries missing from S1 input: {sorted(missing)[:5]}")
    root = Path(config["index_root"])
    indexes = {}
    exact = {}
    for key, raw in config["indexes"].items():
        source, field = key.split(":", 1)
        location = root / raw if not Path(raw).is_absolute() else Path(raw)
        indexes[(source, field)] = SourceTfidfIndex.load(location)
        exact[(source, field)] = ExactMatchIndex.load(location)
    output = Path(config["output_dir"])
    result = stream_query_candidates(query, indexes, exact, output, query_batch_size=int(config.get("query_batch_size", 2048)), shard_target_rows=int(config.get("shard_target_rows", 75_000)), strict_country=bool(config.get("strict_country", False)), preserve_missing_country=bool(config.get("preserve_missing_country", True)))
    inputs = {"s1": file_identity(config["s1"], hash_content=True), "index_manifest": file_identity(root / "manifest.json", hash_content=True)}
    if config.get("query_manifest"):
        inputs["query_manifest"] = file_identity(config["query_manifest"], hash_content=True)
    manifest = ArtifactManifest(kind="candidate-shards", config=config, input_identities=inputs, query_ids=result.pop("query_ids"), shard_order=[Path(path).name for path in result["files"]], row_counts=result["row_counts"], completion="complete", git_commit=git_commit())
    manifest.generated_file_checksums = {Path(path).name: sha256_file(path) for path in result["files"]}
    atomic_write_json(output / "manifest.json", manifest.to_dict())
    print(json.dumps({**result, "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
