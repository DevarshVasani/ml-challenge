"""Route-aware final contest assembly and strict validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd

from .member_d_contracts import (
    PAIR_KEY,
    ROUTE_DISCARD,
    SOURCE_GROUP_KEY,
    validate_pair_keys,
)
from .member_d_manifest import (
    MemberDManifest,
    validate_complete_manifest,
    write_manifest,
)
from .neural_contracts import sha256_file


def _read_many(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(path))
        else:
            frames.append(
                pd.read_csv(
                    path,
                    sep="\t",
                    dtype=object,
                    keep_default_na=False,
                )
            )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def _ids(path: str | Path) -> list[str]:
    frame = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False
    )
    col = (
        "source1_entity_id"
        if "source1_entity_id" in frame.columns
        else "entity_id"
    )
    ids = frame[col].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test S1 IDs")
    return ids


def _registry(path: str | Path) -> set[str]:
    return set(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["entity_id"],
        )["entity_id"].astype(str)
    )


def _atomic_tsv(
    path: Path,
    header: tuple[str, str],
    rows: list[tuple[str, str]],
) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(
            fd, "w", encoding="utf-8", newline=""
        ) as handle:
            handle.write("\t".join(header) + "\n")
            for first, second in rows:
                handle.write(f"{first}\t{second}\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _validate_input_manifest_coverage(
    manifests: Sequence[str | Path],
    input_paths: Sequence[str | Path],
) -> None:
    declared: set[str] = set()
    for manifest_path in manifests:
        manifest = validate_complete_manifest(manifest_path)
        declared.update(
            Path(name).name
            for name in manifest.get("files", {})
        )
    missing = sorted(
        {
            Path(path).name
            for path in input_paths
            if Path(path).name not in declared
        }
    )
    if missing:
        raise ValueError(
            "Final assembly input shards are not declared by a "
            f"complete checksum manifest: {missing}"
        )


def assemble_final_outputs(
    test_s1_manifest: str | Path,
    decision_pair_shards: Sequence[str | Path],
    selected_match_shards: Sequence[str | Path],
    *,
    s2_registry_path: str | Path,
    s3_registry_path: str | Path,
    completion_manifests: Sequence[str | Path],
    output_dir: str | Path,
    ownership_enabled: bool,
    version: str = "v1",
    require_completion_manifests: bool = True,
) -> dict:
    if require_completion_manifests:
        if not completion_manifests:
            raise ValueError(
                "Production final assembly requires completion manifests"
            )
        _validate_input_manifest_coverage(
            completion_manifests,
            list(decision_pair_shards)
            + list(selected_match_shards),
        )
    else:
        for manifest in completion_manifests:
            validate_complete_manifest(manifest)

    s1_ids = _ids(test_s1_manifest)
    s1_set = set(s1_ids)
    s2_ids = _registry(s2_registry_path)
    s3_ids = _registry(s3_registry_path)

    decisions = _read_many(decision_pair_shards)
    selected = _read_many(selected_match_shards)

    validate_pair_keys(decisions)
    if "decision_route" in decisions.columns:
        decisions = decisions[
            decisions["decision_route"].astype(str)
            != ROUTE_DISCARD
        ].copy()

    if decisions.duplicated(PAIR_KEY).any():
        raise ValueError("Duplicate decision pair")

    unknown_s1 = set(
        decisions["source1_entity_id"].astype(str)
    ) - s1_set
    if unknown_s1:
        raise ValueError(
            f"Decision artifact contains unknown S1: "
            f"{sorted(unknown_s1)[:5]}"
        )

    for source, registry in (("S2", s2_ids), ("S3", s3_ids)):
        ids = set(
            decisions.loc[
                decisions["candidate_source"].astype(str) == source,
                "candidate_entity_id",
            ].astype(str)
        )
        unknown = ids - registry
        if unknown:
            raise ValueError(
                f"Decision artifact contains unknown {source} IDs: "
                f"{sorted(unknown)[:5]}"
            )

    if selected.empty:
        selected = pd.DataFrame(columns=PAIR_KEY)
    else:
        validate_pair_keys(selected)
        unknown_selected_s1 = set(
            selected["source1_entity_id"].astype(str)
        ) - s1_set
        if unknown_selected_s1:
            raise ValueError(
                "Selected artifact contains unknown S1: "
                f"{sorted(unknown_selected_s1)[:5]}"
            )

        decision_keys = set(
            map(
                tuple,
                decisions[PAIR_KEY].astype(str).to_numpy(),
            )
        )
        selected_keys = set(
            map(
                tuple,
                selected[PAIR_KEY].astype(str).to_numpy(),
            )
        )
        missing = selected_keys - decision_keys
        if missing:
            raise ValueError(
                "Final matches are not subset of final decision "
                f"pairs: {sorted(missing)[:5]}"
            )

        if ownership_enabled:
            counts = selected.groupby(
                SOURCE_GROUP_KEY
            )["source1_entity_id"].nunique()
            bad = counts[counts > 1]
            if len(bad):
                raise ValueError(
                    "Ownership uniqueness violated for "
                    f"{len(bad)} source records"
                )

    candidate_map = {s1: [] for s1 in s1_ids}
    match_map = {s1: [] for s1 in s1_ids}

    for row in decisions.sort_values(
        ["source1_entity_id", "candidate_source", "candidate_entity_id"]
    ).itertuples(index=False):
        candidate_map[str(row.source1_entity_id)].append(
            str(row.candidate_entity_id)
        )

    for row in selected.sort_values(
        ["source1_entity_id", "candidate_source", "candidate_entity_id"]
    ).itertuples(index=False):
        match_map[str(row.source1_entity_id)].append(
            str(row.candidate_entity_id)
        )

    for mapping, name in (
        (candidate_map, "candidate"),
        (match_map, "match"),
    ):
        for s1, values in mapping.items():
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Duplicate {name} IDs for {s1}"
                )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = out / "candidate_pairs.tsv"
    matching_path = out / "matching_results.tsv"
    manifest_path = out / "export_manifest.json"

    existing = [
        path
        for path in (candidate_path, matching_path, manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing final export files: "
            f"{[str(path) for path in existing]}"
        )

    _atomic_tsv(
        candidate_path,
        ("source1_entity_id", "candidate_entity_ids"),
        [
            (s1, ",".join(candidate_map[s1]))
            for s1 in s1_ids
        ],
    )
    _atomic_tsv(
        matching_path,
        ("source1_entity_id", "matched_entity_ids"),
        [
            (s1, ",".join(match_map[s1]))
            for s1 in s1_ids
        ],
    )

    manifest = MemberDManifest(
        artifact_kind="contest_export",
        version=version,
        completion_state="complete",
        row_count=len(s1_ids),
        files={
            candidate_path.name: sha256_file(candidate_path),
            matching_path.name: sha256_file(matching_path),
        },
        metadata={
            "ownership_enabled": ownership_enabled,
            "candidate_pair_count": len(decisions),
            "selected_pair_count": len(selected),
        },
    )
    write_manifest(manifest_path, manifest)

    return {
        "candidate_pairs": str(candidate_path),
        "matching_results": str(matching_path),
        "manifest": str(manifest_path),
    }
