"""Query reusable TF-IDF/exact indexes into bounded complete-group shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_index import DiskBackedTfidfIndex, ExactMatchIndex, SourceTfidfIndex, run_resumable_channel_query
from .neural_contracts import ArtifactManifest, atomic_write_json, config_hash, file_identity, git_commit, sha256_file
from .neural_data import parse_ground_truth, read_text_tsv
from .neural_diagnostics import RetrievalDiagnostics


class _TfidfChannel:
    def __init__(self, location: Path, source: str, *, top_k: int | None = None, sparse_threads: int = 2):
        self.location = location
        self.source = source
        self.top_k = top_k
        self.sparse_threads = sparse_threads
        self._index = None

    def __call__(self, batch):
        if self._index is None:
            self._index = DiskBackedTfidfIndex(self.location) if (self.location / "disk_metadata.json").exists() else SourceTfidfIndex.load(self.location)
            self._index.config.sparse_threads = self.sparse_threads
        return self._index.query(batch, candidate_source=self.source, top_k=self.top_k)

    def close(self) -> None:
        if self._index is not None and hasattr(self._index, "close"):
            self._index.close()
        self._index = None


class _ExactChannel:
    def __init__(self, location: Path, source: str, field: str):
        self.location = location
        self.source = source
        self.field = field
        self._index = None

    def __call__(self, batch):
        if self._index is None:
            exact_path = self.location / "exact_sqlite.db"
            if exact_path.exists():
                from .candidate_index import SqliteExactMatchIndex
                self._index = SqliteExactMatchIndex(exact_path, field=self.field, candidate_source=self.source)
            else:
                self._index = ExactMatchIndex.load(self.location)
        return self._index.query(batch, field=self.field)

    def close(self) -> None:
        if self._index is not None and hasattr(self._index, "close"):
            self._index.close()
        self._index = None


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
    if int(config.get("index_workers", 1)) != 1:
        raise ValueError("only one resident index worker is supported; set index_workers=1")
    if bool(config.get("strict_country", False)):
        raise ValueError("strict_country requires an explicit disk-backed country lookup and is not silently applied; keep it false to preserve the configured candidate policy")
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
    channel_specs = []
    for key, raw in config["indexes"].items():
        source, field = key.split(":", 1)
        location = root / raw if not Path(raw).is_absolute() else Path(raw)
        channel_specs.append((f"{source}_{field}", _TfidfChannel(location, source, top_k=config.get("top_k"), sparse_threads=int(config.get("sparse_threads", 2)))))
        channel_specs.append((f"{source}_exact_{field}", _ExactChannel(location, source, field)))
    output = Path(config["output_dir"])
    inputs = {"s1": file_identity(config["s1"], hash_content=True), "index_manifest": file_identity(root / "manifest.json", hash_content=True)}
    if config.get("query_manifest"):
        inputs["query_manifest"] = file_identity(config["query_manifest"], hash_content=True)
    diagnostics = None
    if config.get("ground_truth"):
        ground_truth = parse_ground_truth(config["ground_truth"])
        metadata = {}
        if config.get("query_manifest"):
            metadata = json.loads(Path(config["query_manifest"]).read_text(encoding="utf-8")).get("query_metadata", {})
        diagnostics = RetrievalDiagnostics(ground_truth, metadata)
        inputs["ground_truth"] = file_identity(config["ground_truth"], hash_content=True)
    lineage = {"configuration_hash": config_hash(config), "inputs": inputs}
    result = run_resumable_channel_query(query, channel_specs, output, query_batch_size=int(config.get("query_batch_size", 128)), shard_target_rows=int(config.get("shard_target_rows", 75_000)), resume=bool(config.get("resume", True)), lineage=lineage, diagnostics=diagnostics)
    manifest = ArtifactManifest(kind="candidate-shards", config=config, input_identities=inputs, query_ids=result.pop("query_ids"), shard_order=[Path(path).name for path in result["files"]], row_counts=result["row_counts"], completion="complete", git_commit=git_commit())
    manifest.shard_order = [Path(path).relative_to(output).as_posix() for path in result["files"]]
    manifest.generated_file_checksums = {Path(path).relative_to(output).as_posix(): sha256_file(path) for path in result["files"]}
    atomic_write_json(output / "manifest.json", manifest.to_dict())
    print(json.dumps({**result, "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
