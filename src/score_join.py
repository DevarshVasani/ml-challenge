"""Strict ID-based score assembly (Member D, §5).

Joins candidate decision frame with tree scores and optional neural scores
on the 3-part pair key only.  Computes source competitor features on complete
source-record groups.

Never joins by row order.  Never fills a missing neural_score with 0.
Hard failures are raised before writing any output.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .member_d_contracts import (
    DECISION_ROUTES,
    OWNERSHIP_GROUP_KEY,
    PAIR_KEY,
    ROUTE_COL_DECISION,
    ROUTE_COL_HAS,
    ROUTE_COL_REQUESTED,
    ROUTE_NEURAL_FUSION,
    ROUTE_TREE_ONLY,
    SCORE_COL_FUSED,
    SCORE_COL_NEURAL,
    SCORE_COL_TREE,
    validate_score_frame,
)


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


def _load_shards(paths: Sequence[str | Path], label: str) -> pd.DataFrame:
    """Load and concatenate a list of parquet/TSV shard files."""
    frames: list[pd.DataFrame] = []
    seen_files: set[str] = set()
    for p in paths:
        p = Path(p)
        name = str(p)
        if name in seen_files:
            raise ValueError(f"Duplicate shard path in {label}: {p}")
        seen_files.add(name)
        if not p.exists():
            raise FileNotFoundError(f"Declared shard not found ({label}): {p}")
        if p.suffix.casefold() == ".parquet":
            frames.append(pd.read_parquet(p))
        else:
            frames.append(pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Source competitor features
# ---------------------------------------------------------------------------


def compute_source_competitor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add source-level tree competition features on complete source-record groups.

    Requires: PAIR_KEY + tree_score columns in frame.

    Added columns
    -------------
    source_competitor_count   int    # of S1 competitors for this source record
    tree_source_rank          int    rank of this S1's tree_score (1 = best)
    tree_gap_to_best          float  best_score - this score
    tree_best_second_gap      float  rank-1 score - rank-2 score
    tree_near_tie_count       int    # competitors within 0.02 of best
    is_tree_best_owner        int8   1 if this S1 has the highest tree score

    Neural competition features are intentionally omitted unless the entire
    compared set has known neural scores (checked by caller).
    """
    required = set(PAIR_KEY) | {SCORE_COL_TREE}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"compute_source_competitor_features: missing columns {sorted(missing)}")

    out = frame.copy()
    tree = pd.to_numeric(out[SCORE_COL_TREE], errors="coerce").fillna(0.0)

    comp_count = pd.Series(0, index=out.index, dtype="int32")
    src_rank = pd.Series(0, index=out.index, dtype="int32")
    gap_to_best = pd.Series(0.0, index=out.index, dtype="float32")
    best_second_gap = pd.Series(0.0, index=out.index, dtype="float32")
    near_tie = pd.Series(0, index=out.index, dtype="int16")
    is_best = pd.Series(0, index=out.index, dtype="int8")

    NEAR_TIE_MARGIN = 0.02

    for key, grp in out.groupby(OWNERSHIP_GROUP_KEY, sort=False):
        idx = grp.index
        t = tree.loc[idx].to_numpy(dtype="float64")
        n = len(t)
        sorted_t = np.sort(t)[::-1]

        best = float(sorted_t[0])
        second = float(sorted_t[1]) if n > 1 else best
        ranks = (
            pd.Series(t).rank(method="min", ascending=False).astype("int32").to_numpy()
        )
        near = int(np.sum(t >= best - NEAR_TIE_MARGIN))

        comp_count.loc[idx] = n
        src_rank.loc[idx] = ranks
        gap_to_best.loc[idx] = (best - t).astype("float32")
        best_second_gap.loc[idx] = float(best - second)
        near_tie.loc[idx] = near
        is_best.loc[idx] = (t == best).astype("int8")

    out["source_competitor_count"] = comp_count
    out["tree_source_rank"] = src_rank
    out["tree_gap_to_best"] = gap_to_best
    out["tree_best_second_gap"] = best_second_gap
    out["tree_near_tie_count"] = near_tie
    out["is_tree_best_owner"] = is_best

    return out


# ---------------------------------------------------------------------------
# Core join
# ---------------------------------------------------------------------------


def join_scores(
    decision_candidate_shards: Sequence[str | Path],
    tree_score_shards: Sequence[str | Path],
    neural_score_shards: Sequence[str | Path] | None = None,
    route_gate_manifest: str | Path | None = None,
    *,
    output_dir: str | Path,
    provenance: dict[str, Any] | None = None,
    add_competitor_features: bool = True,
) -> dict[str, Any]:
    """Join all score sources on the 3-part pair key and write output shards.

    Hard failures (before writing any output)
    -----------------------------------------
    - Duplicate pair key in any input frame
    - Extra score key not in decision candidates
    - Missing tree_score for any row
    - Missing neural_score when neural_requested == 1
    - Schema/version mismatch

    Tree-only rows carry has_neural_score=0.
    Missing neural_score is NEVER filled with 0.

    Parameters
    ----------
    decision_candidate_shards : paths to candidate shards (the decision set)
    tree_score_shards : paths to tree score shards
    neural_score_shards : paths to neural score shards (optional)
    route_gate_manifest : optional path to a route/gate JSON manifest
    output_dir : where to write joined_scores.parquet
    provenance : dict of version/hash fields
    add_competitor_features : whether to compute source competitor features

    Returns
    -------
    dict with keys: output_path, row_count, route_counts, missing_neural_count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load decision candidates ----
    cands = _load_shards(decision_candidate_shards, "decision_candidates")
    if cands.empty:
        raise ValueError("Decision candidate shards are empty")
    for col in PAIR_KEY:
        if col not in cands.columns:
            raise ValueError(f"Decision candidates missing pair key column: {col}")
    if cands.duplicated(PAIR_KEY).any():
        raise ValueError("Duplicate pair key detected in decision candidate shards")

    # ---- load tree scores ----
    tree_df = _load_shards(tree_score_shards, "tree_scores")
    if tree_df.empty:
        raise ValueError("Tree score shards are empty")

    # Normalise tree score column name
    if "score" in tree_df.columns and SCORE_COL_TREE not in tree_df.columns:
        tree_df = tree_df.rename(columns={"score": SCORE_COL_TREE})

    for col in PAIR_KEY:
        if col not in tree_df.columns:
            raise ValueError(f"Tree score shards missing pair key column: {col}")
    if tree_df.duplicated(PAIR_KEY).any():
        raise ValueError("Duplicate pair key detected in tree score shards")

    # Check no extra tree score keys
    cand_keys = set(map(tuple, cands[PAIR_KEY].astype(str).to_numpy()))
    tree_keys = set(map(tuple, tree_df[PAIR_KEY].astype(str).to_numpy()))
    extra_tree = tree_keys - cand_keys
    if extra_tree:
        ex = sorted(extra_tree)[:5]
        raise ValueError(
            f"{len(extra_tree)} tree score keys not in decision candidates. "
            f"First examples: {ex}"
        )

    # ---- join tree scores ----
    joined = cands.merge(
        tree_df[PAIR_KEY + [SCORE_COL_TREE]],
        on=PAIR_KEY,
        how="left",
        validate="1:1",
    )

    # Every decision pair must have a tree score
    missing_tree = joined[SCORE_COL_TREE].isna().sum()
    if missing_tree > 0:
        raise ValueError(
            f"{missing_tree} decision pairs have no tree_score. "
            "Tree score is required for every final decision pair."
        )

    # ---- load route manifest ----
    route_info: dict[str, Any] = {}
    if route_gate_manifest:
        route_info = json.loads(Path(route_gate_manifest).read_text(encoding="utf-8"))

    # ---- add route columns if not already present ----
    if ROUTE_COL_REQUESTED not in joined.columns:
        # Default: neural is requested for all pairs (conservative)
        joined[ROUTE_COL_REQUESTED] = 1

    # ---- load and join neural scores ----
    if neural_score_shards:
        neural_df = _load_shards(neural_score_shards, "neural_scores")
        if not neural_df.empty:
            if "score" in neural_df.columns and SCORE_COL_NEURAL not in neural_df.columns:
                neural_df = neural_df.rename(columns={"score": SCORE_COL_NEURAL})
            for col in PAIR_KEY:
                if col not in neural_df.columns:
                    raise ValueError(f"Neural score shards missing pair key column: {col}")
            if neural_df.duplicated(PAIR_KEY).any():
                raise ValueError("Duplicate pair key in neural score shards")
            extra_neural = (
                set(map(tuple, neural_df[PAIR_KEY].astype(str).to_numpy())) - cand_keys
            )
            if extra_neural:
                ex = sorted(extra_neural)[:5]
                raise ValueError(
                    f"{len(extra_neural)} neural score keys not in decision candidates: {ex}"
                )
            joined = joined.merge(
                neural_df[PAIR_KEY + [SCORE_COL_NEURAL]],
                on=PAIR_KEY,
                how="left",
            )
    else:
        joined[SCORE_COL_NEURAL] = float("nan")

    # ---- has_neural_score flag ----
    joined[ROUTE_COL_HAS] = joined[SCORE_COL_NEURAL].notna().astype("int8")

    # ---- hard check: neural_requested==1 but no neural score ----
    neural_req = pd.to_numeric(joined[ROUTE_COL_REQUESTED], errors="coerce")
    violation = (neural_req == 1) & (joined[ROUTE_COL_HAS] == 0)
    if violation.any():
        count = int(violation.sum())
        raise ValueError(
            f"{count} pairs have neural_requested=1 but no neural_score. "
            "Either provide the neural score or set neural_requested=0 for tree-only pairs."
        )

    # ---- decision_route ----
    if ROUTE_COL_DECISION not in joined.columns:
        joined[ROUTE_COL_DECISION] = joined[ROUTE_COL_HAS].map(
            {1: ROUTE_NEURAL_FUSION, 0: ROUTE_TREE_ONLY}
        )

    # ---- source competitor features ----
    if add_competitor_features:
        joined = compute_source_competitor_features(joined)

    # ---- provenance ----
    if provenance:
        for field, value in provenance.items():
            if field not in joined.columns:
                joined[field] = value

    # ---- validate with contract ----
    validate_score_frame(joined, require_tree_score=True, check_routes=True)

    # ---- write output ----
    out_path = output_dir / "joined_scores.parquet"
    joined.to_parquet(out_path, index=False)

    route_counts: dict[str, int] = {}
    for route in joined[ROUTE_COL_DECISION].unique():
        route_counts[str(route)] = int((joined[ROUTE_COL_DECISION] == route).sum())

    summary: dict[str, Any] = {
        "output_path": str(out_path),
        "row_count": len(joined),
        "route_counts": route_counts,
        "missing_neural_count": int(joined[SCORE_COL_NEURAL].isna().sum()),
    }
    # Write summary manifest
    (output_dir / "join_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[score_join] {len(joined)} rows -> {out_path}")
    for route, cnt in route_counts.items():
        print(f"  {route}: {cnt}")
    return summary
