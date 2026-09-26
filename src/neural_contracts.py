"""Versioned, label-safe contracts used by the neural baselines.

This module deliberately contains no dataset discovery or model imports.  It is
safe to import from help/dry-run commands on a machine without Torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "neural-contracts-1"
PAIR_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "candidate_source",
    "source1_fold", "candidate_fold", "name_a", "address_a", "country_a",
    "name_b", "address_b", "country_b",
]
PREDICTION_COLUMNS = ["source1_entity_id", "candidate_entity_id", "candidate_source", "score"]
TEXT_FIELDS = ["name_a", "address_a", "country_a", "name_b", "address_b", "country_b"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_identity(path: str | os.PathLike[str], *, hash_content: bool = False) -> dict[str, Any]:
    """Return cheap identity by default; content hashing is explicit and never automatic."""
    p = Path(path)
    result: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.exists():
        stat = p.stat()
        result.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if hash_content:
            digest = hashlib.sha256()
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result["sha256"] = digest.hexdigest()
    return result


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_identity(path: str | os.PathLike[str], *, hash_content: bool = False) -> dict[str, Any]:
    """Return a stable identity for a checkpoint/artifact directory."""
    root = Path(path)
    result: dict[str, Any] = {"path": str(root), "exists": root.is_dir(), "files": []}
    if not root.is_dir():
        return result
    files = []
    for item in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        identity = file_identity(item, hash_content=hash_content)
        identity["path"] = item.relative_to(root).as_posix()
        files.append(identity)
    result["files"] = files
    result["identity_hash"] = config_hash(files)
    return result


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def manifest_shard_paths(manifest_path: str | os.PathLike[str], shard_order: Iterable[str], pair_subdir: str) -> list[str]:
    """Resolve pair-manifest shard entries to file paths.

    prepare_neural_data records shards relative to the manifest directory
    (``train_pairs/pairs-….parquet``); older manifests hold bare names that
    live under ``pair_subdir``. Prefer the first, fall back to the second.
    """
    root = Path(manifest_path).parent
    return [str(root / name if (root / name).is_file() else root / pair_subdir / name) for name in map(str, shard_order)]


def validate_pair_frame(frame: Any, *, labeled: bool = False, allow_extra: bool = True) -> None:
    missing = [column for column in PAIR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"pair shard is missing required columns: {missing}")
    if labeled and "label" not in frame.columns:
        raise ValueError("training/validation pair shards require integer label")
    if "label" in frame.columns:
        if frame["label"].isna().any():
            raise ValueError("labels must not be missing")
        numeric = frame["label"].map(lambda value: float(value))
        if not numeric.map(math.isfinite).all() or not numeric.isin([0.0, 1.0]).all():
            raise ValueError("labels must be integer values in {0, 1}")
    key = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate pair key detected; labels must not be silently changed")
    for column in ("source1_entity_id", "candidate_entity_id", "candidate_source"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains missing or empty values")
    if not allow_extra:
        expected = set(PAIR_COLUMNS) | ({"label"} if labeled else set())
        unexpected = set(frame.columns) - expected
        if unexpected:
            raise ValueError(f"pair shard has unexpected columns: {sorted(unexpected)}")


def validate_prediction_frame(frame: Any, expected_keys: set[tuple[str, str, str]] | None = None) -> None:
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction artifact is missing required columns: {missing}")
    if frame.duplicated(PREDICTION_COLUMNS[:3]).any():
        raise ValueError("prediction artifact contains duplicate pair keys")
    scores = frame["score"]
    try:
        numeric = scores.map(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction scores must be finite numbers") from exc
    if scores.isna().any() or not numeric.map(math.isfinite).all():
        raise ValueError("prediction scores must be finite numbers")
    if ((numeric < 0) | (numeric > 1)).any():
        raise ValueError("prediction scores must be probabilities in [0, 1]")
    for column in PREDICTION_COLUMNS[:3]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains missing or empty values")
    if expected_keys is not None:
        actual = set(map(tuple, frame[PREDICTION_COLUMNS[:3]].astype(str).itertuples(index=False, name=None)))
        if actual != expected_keys:
            raise ValueError(f"prediction coverage mismatch: expected {len(expected_keys)}, got {len(actual)}")


@dataclass
class ArtifactManifest:
    kind: str
    config: dict[str, Any]
    input_identities: dict[str, Any] = field(default_factory=dict)
    fold_map_identity: Any = None
    query_ids: list[str] = field(default_factory=list)
    shard_order: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    completion: str = "incomplete"
    generated_file_checksums: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    git_commit: str | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["configuration_hash"] = config_hash(self.config)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactManifest":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value[key] for key in allowed if key in value})

    def compatible_with(self, other: "ArtifactManifest") -> bool:
        return (
            self.schema_version == other.schema_version
            and config_hash(self.config) == config_hash(other.config)
            and self.input_identities == other.input_identities
            and self.fold_map_identity == other.fold_map_identity
            and self.query_ids == other.query_ids
            and self.shard_order == other.shard_order
        )


def completion_path(path: str | os.PathLike[str]) -> Path:
    return Path(f"{path}.complete.json")


def mark_complete(path: str | os.PathLike[str], manifest: ArtifactManifest) -> None:
    if manifest.completion != "complete":
        raise ValueError("only complete manifests may receive a completion sidecar")
    atomic_write_json(completion_path(path), manifest.to_dict())


def require_compatible_completion(path: str | os.PathLike[str], expected: ArtifactManifest) -> ArtifactManifest:
    sidecar = completion_path(path)
    if not Path(path).exists() or not sidecar.exists():
        raise FileNotFoundError(f"incomplete or missing artifact: {path}")
    actual = ArtifactManifest.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    if actual.completion != "complete" or not actual.compatible_with(expected):
        raise ValueError(f"artifact identity/configuration does not match for resume: {path}")
    return actual


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
