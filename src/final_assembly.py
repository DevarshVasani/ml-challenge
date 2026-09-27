"""Route-aware final contest assembly and strict validation."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd

from .member_d_contracts import PAIR_KEY, ROUTE_DISCARD, SOURCE_GROUP_KEY, validate_pair_keys
from .member_d_manifest import MemberDManifest, validate_complete_manifest, write_manifest
from .neural_contracts import sha256_file


def _read_many(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for raw in paths:
        p = Path(raw)
        if not p.exists(): raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ids(path: str | Path) -> list[str]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False); col = "source1_entity_id" if "source1_entity_id" in df.columns else "entity_id"; ids = df[col].tolist()
    if len(ids) != len(set(ids)): raise ValueError("Duplicate test S1 IDs")
    return ids


def _registry(path: str | Path) -> set[str]:
    return set(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id"])["entity_id"])


def _atomic_tsv(path: Path, header: tuple[str, str], rows: list[tuple[str, str]]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(header) + "\n")
            for a, b in rows: f.write(f"{a}\t{b}\n")
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise


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
) -> dict:
    for manifest in completion_manifests: validate_complete_manifest(manifest)
    s1_ids = _ids(test_s1_manifest); s1_set = set(s1_ids); valid_ids = _registry(s2_registry_path) | _registry(s3_registry_path)
    decisions = _read_many(decision_pair_shards); selected = _read_many(selected_match_shards)
    validate_pair_keys(decisions)
    if "decision_route" in decisions.columns: decisions = decisions[decisions.decision_route.astype(str) != ROUTE_DISCARD].copy()
    if decisions.duplicated(PAIR_KEY).any(): raise ValueError("Duplicate decision pair")
    if set(decisions.source1_entity_id.astype(str)) - s1_set: raise ValueError("Decision artifact contains unknown S1")
    if set(decisions.candidate_entity_id.astype(str)) - valid_ids: raise ValueError("Decision artifact contains unknown S2/S3 IDs")

    if selected.empty: selected = pd.DataFrame(columns=PAIR_KEY)
    else:
        validate_pair_keys(selected)
        if set(selected.source1_entity_id.astype(str)) - s1_set: raise ValueError("Selected artifact contains unknown S1")
        decision_keys = set(map(tuple, decisions[PAIR_KEY].astype(str).to_numpy())); selected_keys = set(map(tuple, selected[PAIR_KEY].astype(str).to_numpy()))
        missing = selected_keys - decision_keys
        if missing: raise ValueError(f"Final matches are not subset of final decision pairs: {sorted(missing)[:5]}")
        if ownership_enabled:
            counts = selected.groupby(SOURCE_GROUP_KEY)["source1_entity_id"].nunique(); bad = counts[counts > 1]
            if len(bad): raise ValueError(f"Ownership uniqueness violated for {len(bad)} source records")

    cand_map = {s: [] for s in s1_ids}; match_map = {s: [] for s in s1_ids}
    for r in decisions.sort_values(["source1_entity_id", "candidate_source", "candidate_entity_id"]).itertuples(index=False): cand_map[str(r.source1_entity_id)].append(str(r.candidate_entity_id))
    for r in selected.sort_values(["source1_entity_id", "candidate_source", "candidate_entity_id"]).itertuples(index=False): match_map[str(r.source1_entity_id)].append(str(r.candidate_entity_id))
    for mapping, name in ((cand_map, "candidate"), (match_map, "match")):
        for s, vals in mapping.items():
            if len(vals) != len(set(vals)): raise ValueError(f"Duplicate {name} IDs for {s}")

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    cand_path = out / "candidate_pairs.tsv"; match_path = out / "matching_results.tsv"
    _atomic_tsv(cand_path, ("source1_entity_id", "candidate_entity_ids"), [(s, ",".join(cand_map[s])) for s in s1_ids])
    _atomic_tsv(match_path, ("source1_entity_id", "matched_entity_ids"), [(s, ",".join(match_map[s])) for s in s1_ids])
    manifest = MemberDManifest(
        artifact_kind="contest_export", version=version, completion_state="complete", row_count=len(s1_ids),
        files={cand_path.name: sha256_file(cand_path), match_path.name: sha256_file(match_path)},
        metadata={"ownership_enabled": ownership_enabled, "candidate_pair_count": len(decisions), "selected_pair_count": len(selected)},
    )
    write_manifest(out / "export_manifest.json", manifest)
    return {"candidate_pairs": str(cand_path), "matching_results": str(match_path), "manifest": str(out / "export_manifest.json")}
