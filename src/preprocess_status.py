"""Bounded inspection/recovery of preprocessing artifacts; never loads pickles."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .neural_contracts import atomic_write_json, config_hash, file_identity, sha256_file


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _same_file(stored: dict[str, Any], current: dict[str, Any], verify: bool) -> bool:
    keys = ["size", "mtime_ns"] + (["sha256"] if verify else [])
    return all(stored.get(key) == current.get(key) for key in keys if key in stored)


def inspect(root: str | Path, config: dict[str, Any] | None = None, *, verify: bool = False) -> dict[str, Any]:
    path = Path(root)
    result: dict[str, Any] = {"root": str(path), "components": {}}
    expected: dict[str, tuple[str, str, dict[str, Any]]] = {}
    if config:
        for source, value in config.get("sources", {}).items():
            identity = file_identity(value["path"], hash_content=verify)
            for field in value.get("fields", ["business_name", "business_address"]):
                expected[f"{source}_{field}"] = (str(source), str(field), identity)
    names = set(expected)
    if path.exists():
        names.update(item.name for item in path.iterdir() if item.is_dir())
    root_manifest = _json(path / "manifest.json") or {}
    recorded_checksums = root_manifest.get("generated_file_checksums", {})
    for name in sorted(names):
        component = path / name
        if not component.exists():
            result["components"][name] = {"state": "missing", "reason": "directory absent"}
            continue
        disk_meta_path = component / "disk_metadata.json"
        legacy_meta_path = component / "metadata.json"
        tf_path = disk_meta_path if disk_meta_path.exists() else legacy_meta_path
        tf_meta = _json(tf_path) if tf_path.exists() else None
        exact_sqlite_meta_path = component / "exact_sqlite.json"
        exact_legacy_meta_path = component / "exact_metadata.json"
        exact_path = exact_sqlite_meta_path if exact_sqlite_meta_path.exists() else exact_legacy_meta_path
        exact_meta = _json(exact_path) if exact_path.exists() else None
        state_meta = _json(component / "build_state.json")
        state = "complete_unverified"
        reasons: list[str] = []
        if state_meta and state_meta.get("stage") != "complete":
            state, reasons = "in_progress", [f"build stage {state_meta.get('stage')}"]
        elif tf_meta is None and exact_meta is None:
            state, reasons = "missing", ["no recognized component metadata"]
        elif tf_meta is None or exact_meta is None:
            state, reasons = "in_progress", ["TF-IDF or exact component is absent"]
        else:
            spec = expected.get(name)
            if spec:
                source, field, identity = spec
                for label, metadata in (("TF-IDF", tf_meta), ("exact", exact_meta)):
                    if metadata.get("field", metadata.get("config", {}).get("field")) != field or metadata.get("candidate_source", source) != source:
                        reasons.append(f"{label} source/field mismatch")
                    stored_identity = metadata.get("source_identity")
                    if not stored_identity:
                        reasons.append(f"{label} has no strong source identity")
                    elif not _same_file(stored_identity, identity, verify):
                        reasons.append(f"{label} source identity mismatch")
            required_files = []
            if disk_meta_path.exists() and tf_meta:
                required_files.append(component / "vectorizer.pkl")
                for shard in tf_meta.get("shards", []):
                    required_files.extend(component / shard[key] for key in ("data", "indices", "indptr", "ids") if key in shard)
            else:
                required_files.append(component / "index.pkl")
            required_files.append(component / ("exact_sqlite.db" if exact_sqlite_meta_path.exists() else "exact_index.pkl"))
            missing = [str(item) for item in required_files if not item.is_file()]
            if missing:
                state, reasons = "in_progress", [f"missing payload files: {missing[:3]}"]
            elif any("mismatch" in value for value in reasons):
                state = "incompatible"
            elif verify:
                bad = []
                for item in required_files + [tf_path, exact_path]:
                    relative = item.relative_to(path).as_posix()
                    if recorded_checksums.get(relative) and sha256_file(item) != recorded_checksums[relative]:
                        bad.append(relative)
                if bad:
                    state, reasons = "incompatible", [f"checksum mismatch: {bad[:3]}"]
                elif not disk_meta_path.exists() or not exact_sqlite_meta_path.exists():
                    state = "complete_unverified"
                    reasons.append("legacy pickle structural validation is an explicit eager migration step and may require more system RAM")
                elif not reasons:
                    try:
                        for shard in tf_meta.get("shards", []):
                            arrays = {key: np.load(component / shard[key], mmap_mode="r", allow_pickle=False) for key in ("data", "indices", "indptr")}
                            if len(arrays["data"]) != len(arrays["indices"]) or len(arrays["indptr"]) != int(shard["features"]) + 1:
                                raise ValueError(f"invalid sparse array dimensions in {shard.get('data')}")
                        connection = sqlite3.connect(component / "exact_sqlite.db")
                        try:
                            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                                raise ValueError("SQLite quick_check failed")
                        finally:
                            connection.close()
                        state = "complete_verified"
                    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
                        state, reasons = "incompatible", [f"bounded structural validation failed: {exc}"]
            elif not reasons:
                reasons.append("payload checksums not requested")
        result["components"][name] = {
            "state": state, "reasons": reasons, "tfidf_metadata": str(tf_path) if tf_path.exists() else None,
            "exact_metadata": str(exact_path) if exact_path.exists() else None,
        }
    manifest_path = path / "manifest.json"
    result["manifest"] = {"exists": manifest_path.exists(), "path": str(manifest_path), "completion": root_manifest.get("completion")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--config", help="index build config used for compatibility checks")
    parser.add_argument("--verify", action="store_true", help="explicitly hash payloads and source inputs")
    parser.add_argument("--recover-manifest", action="store_true", help="write a minimal root manifest only after full verification")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    root = args.root or (config or {}).get("output_dir")
    if not root:
        parser.error("--root or --config with output_dir is required")
    report = inspect(root, config, verify=args.verify or args.recover_manifest)
    if args.recover_manifest:
        bad = {key: value["state"] for key, value in report["components"].items() if value["state"] != "complete_verified"}
        if bad:
            raise ValueError(f"cannot recover root manifest; components are not verified: {bad}")
        value = {"kind": "candidate-index", "schema_version": "recovered-candidate-index-1", "completion": "complete", "configuration_hash": config_hash(config or {}), "recovered": True, "components": sorted(report["components"])}
        atomic_write_json(Path(root) / "manifest.json", value)
        report["manifest"] = {"exists": True, "path": str(Path(root) / "manifest.json"), "completion": "complete", "recovered": True}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
