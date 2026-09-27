"""Compute C-owned competition features only on source-complete groups."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .context_features import add_competition_features
from .member_d_contracts import SOURCE_GROUP_KEY, validate_pair_keys
from .member_d_manifest import MemberDManifest, write_manifest
from .source_group_io import iter_complete_source_groups
from .neural_contracts import sha256_file


def build_source_competition_artifact(
    candidate_shards: Sequence[str | Path],
    source_complete_manifests: Sequence[str | Path],
    output_path: str | Path,
    *,
    score_margin: float = 0.05,
    candidate_version: str | None = None,
    candidate_schema_version: str | None = None,
    version: str = "v1",
) -> dict[str, Any]:
    """Run C's add_competition_features() on each complete source group.

    This function intentionally does not recreate competition formulas in D.
    The upstream candidate rows must contain C's required retrieval_score fields.
    """
    groups: list[pd.DataFrame] = []
    for group in iter_complete_source_groups(candidate_shards, manifest_paths=source_complete_manifests):
        validate_pair_keys(group)
        if group[SOURCE_GROUP_KEY].drop_duplicates().shape[0] != 1:
            raise ValueError("Complete-source iterator returned a mixed source group")
        enriched = add_competition_features(group, score_margin=score_margin)
        enriched["source_group_complete"] = True
        groups.append(enriched)
    result = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    if not result.empty:
        validate_pair_keys(result)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    manifest_path = output.parent / "source_competition_manifest.json"
    manifest = MemberDManifest(
        artifact_kind="source_competition_candidates",
        version=version,
        completion_state="complete",
        row_count=len(result),
        candidate_version=candidate_version,
        files={output.name: sha256_file(output)},
        metadata={"complete_by_source_record": True, "score_margin": score_margin, "candidate_schema_version": candidate_schema_version},
    )
    payload = manifest.to_dict(); payload["complete_by_source_record"] = True; payload["candidate_schema_version"] = candidate_schema_version
    # Score-join accepts both top-level and metadata location for compatibility.
    from .neural_contracts import atomic_write_json
    atomic_write_json(manifest_path, payload)
    return {"path": str(output), "row_count": len(result), "sha256": sha256_file(output), "manifest": str(manifest_path)}
