"""Build a diagnostic threshold joined-score artifact before source-complete C handoff.

This is deliberately *not* the production score join.  It exists so D can
exercise fusion/calibration with A's completed threshold logits while C's
competition features are known to come from incomplete source-record groups.

Unsafe source-group-derived features are removed and the output manifest is
marked selection-ineligible / diagnostic-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .member_d_contracts import (
    DECISION_ROUTE,
    FORBIDDEN_PRODUCTION_COLUMNS,
    HAS_NEURAL_SCORE,
    NEURAL_LOGIT,
    NEURAL_REQUESTED,
    PAIR_KEY,
    TREE_SCORE,
    validate_joined_scores,
    validate_pair_keys,
)
from .neural_contracts import sha256_file

UNSAFE_INCOMPLETE_GROUP_FEATURES = {
    "competing_s1_count",
    "competing_s1_score_gap",
    "competing_best_second_gap",
    "competing_near_tie_count",
    "is_top_competing_s1",
}

ROUTE_MAP = {
    "neural": "neural_fusion",
    "neural_fusion": "neural_fusion",
    "tree": "tree_only",
    "tree_only": "tree_only",
    "discard": "discard",
}


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)


def _write(frame: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, sep="\t", index=False)


def discover_neural_shards(score_dir: str | Path) -> list[Path]:
    shards = sorted(Path(score_dir).glob("*.neural.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No *.neural.parquet shards under {score_dir}"
        )
    return shards


def build_rehearsal_join(
    gate_path: str | Path,
    neural_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, object]:
    gate = _read(gate_path)
    validate_pair_keys(gate)

    forbidden = sorted(set(gate.columns) & FORBIDDEN_PRODUCTION_COLUMNS)
    if forbidden:
        raise ValueError(
            f"Gate contains forbidden fields and cannot be rehearsed: {forbidden}"
        )
    if TREE_SCORE not in gate.columns:
        raise ValueError("Gate is missing tree_score")

    route_col = "decision_route" if "decision_route" in gate.columns else "route"
    if route_col not in gate.columns:
        raise ValueError("Gate is missing route/decision_route")

    raw_route = gate[route_col].astype(str).str.lower()
    unknown_routes = sorted(set(raw_route) - set(ROUTE_MAP))
    if unknown_routes:
        raise ValueError(f"Unknown gate routes: {unknown_routes}")

    joined = gate.drop(
        columns=[
            c
            for c in (
                "route",
                "decision_route",
                "neural_logit",
                "neural_probability",
                "neural_requested",
                "has_neural_score",
                *UNSAFE_INCOMPLETE_GROUP_FEATURES,
            )
            if c in gate.columns
        ],
        errors="ignore",
    ).copy()
    joined[DECISION_ROUTE] = raw_route.map(ROUTE_MAP)
    joined[NEURAL_REQUESTED] = (
        joined[DECISION_ROUTE] == "neural_fusion"
    ).astype("int8")

    neural_frames = []
    neural_input_hashes = {}
    for path_like in neural_paths:
        path = Path(path_like)
        frame = _read(path)
        validate_pair_keys(frame)
        if NEURAL_LOGIT not in frame.columns:
            raise ValueError(f"{path} is missing neural_logit")
        neural_frames.append(frame[PAIR_KEY + [NEURAL_LOGIT]].copy())
        neural_input_hashes[str(path)] = sha256_file(path)

    neural = pd.concat(neural_frames, ignore_index=True)
    validate_pair_keys(neural)

    gate_keys = pd.MultiIndex.from_frame(joined[PAIR_KEY].astype(str))
    neural_keys = pd.MultiIndex.from_frame(neural[PAIR_KEY].astype(str))
    missing = gate_keys.difference(neural_keys)
    extra = neural_keys.difference(gate_keys)
    if len(missing) or len(extra):
        raise ValueError(
            "Full threshold neural coverage must exactly equal gate keys: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    joined = joined.merge(
        neural,
        on=PAIR_KEY,
        how="left",
        validate="1:1",
    )
    present = pd.to_numeric(joined[NEURAL_LOGIT], errors="coerce").notna()
    joined[HAS_NEURAL_SCORE] = present.astype("int8")

    requested = joined[NEURAL_REQUESTED].astype(int) == 1
    if (requested & ~present).any():
        raise ValueError("At least one neural route row has no neural logit")

    joined["source_group_complete"] = False
    joined["diagnostic_only"] = True
    joined["competition_features_trusted"] = False

    validate_joined_scores(joined)
    _write(joined, output_path)

    manifest = {
        "artifact_kind": "member_d_threshold_rehearsal_join",
        "diagnostic_only": True,
        "selection_eligible": False,
        "selection_ineligible_reason": (
            "C threshold competition features are not source-record complete"
        ),
        "source_groups_complete": False,
        "competition_features_trusted": False,
        "dropped_features": sorted(UNSAFE_INCOMPLETE_GROUP_FEATURES),
        "row_count": len(joined),
        "route_counts": {
            str(k): int(v)
            for k, v in joined[DECISION_ROUTE].value_counts().to_dict().items()
        },
        "gate_sha256": sha256_file(gate_path),
        "neural_input_sha256": neural_input_hashes,
        "output_sha256": sha256_file(output_path),
    }
    manifest_path = Path(f"{output_path}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--neural-score-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_rehearsal_join(
        args.gate,
        discover_neural_shards(args.neural_score_dir),
        args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
