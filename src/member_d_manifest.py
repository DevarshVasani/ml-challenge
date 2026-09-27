"""Versioned Member D artifact manifests with checksums and compatibility checks."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .neural_contracts import atomic_write_json, git_commit, sha256_file

D_SCHEMA_VERSION = "member-d-artifact-v1"


@dataclass
class MemberDManifest:
    artifact_kind: str
    version: str
    completion_state: str = "incomplete"
    schema_version: str = D_SCHEMA_VERSION
    git_commit: str | None = None
    split: str | None = None
    shard_id: str | None = None
    row_count: int | None = None
    input_versions: dict[str, Any] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)
    candidate_version: str | None = None
    tree_model_version: str | None = None
    neural_model_version: str | None = None
    fusion_version: str | None = None
    calibration_version: str | None = None
    decoder_version: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["git_commit"]:
            data["git_commit"] = git_commit()
        return data


def build_file_checksums(paths: Iterable[str | Path]) -> dict[str, str]:
    return {Path(p).name: sha256_file(p) for p in paths}


def write_manifest(path: str | Path, manifest: MemberDManifest) -> None:
    if manifest.completion_state not in {"incomplete", "complete"}:
        raise ValueError("completion_state must be incomplete or complete")
    atomic_write_json(path, manifest.to_dict())


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_complete_manifest(
    manifest_path: str | Path,
    *,
    expected_kind: str | None = None,
    required_metadata: Mapping[str, Any] | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    if manifest.get("schema_version") != D_SCHEMA_VERSION:
        raise ValueError(f"Unexpected D manifest schema: {manifest.get('schema_version')}")
    if manifest.get("completion_state") != "complete":
        raise ValueError(f"Artifact manifest is not complete: {path}")
    if expected_kind and manifest.get("artifact_kind") != expected_kind:
        raise ValueError(f"Expected artifact_kind={expected_kind}, got {manifest.get('artifact_kind')}")
    for key, value in (required_metadata or {}).items():
        actual = manifest.get(key, manifest.get("metadata", {}).get(key))
        if actual != value:
            raise ValueError(f"Manifest mismatch for {key}: expected {value!r}, got {actual!r}")
    if verify_files:
        base = path.parent
        for rel, expected_sha in manifest.get("files", {}).items():
            file_path = Path(rel)
            if not file_path.is_absolute():
                file_path = base / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"Manifest-declared artifact missing: {file_path}")
            if sha256_file(file_path) != expected_sha:
                raise ValueError(f"Checksum mismatch: {file_path}")
    return manifest
