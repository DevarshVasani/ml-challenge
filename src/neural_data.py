"""Deterministic, endpoint-safe preparation helpers for neural pair shards."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .neural_contracts import PAIR_COLUMNS, TEXT_FIELDS
from .split import build_connected_components
from .text_utils import normalize_text


def read_text_tsv(path: str | Path, *, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """Read identifiers/text as literal strings; ``NA`` remains the string NA."""
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, usecols=usecols)
    return frame.fillna("")


def parse_ground_truth(frame_or_path: pd.DataFrame | str | Path) -> dict[str, list[str]]:
    frame = frame_or_path if isinstance(frame_or_path, pd.DataFrame) else read_text_tsv(frame_or_path)
    required = {"source1_entity_id", "matched_entity_ids"}
    if not required.issubset(frame.columns):
        raise ValueError(f"ground truth requires {sorted(required)}")
    result: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        raw = getattr(row, "matched_entity_ids")
        values = list(dict.fromkeys(x.strip() for x in str(raw).split(",") if x.strip()))
        source1 = str(getattr(row, "source1_entity_id")).strip()
        if source1:
            result[source1] = values
    return result


def _column(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype=str)


def pair_text_rows(s1: pd.DataFrame, candidate: pd.DataFrame, source: str, *, folds: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Join one S1/source pair batch without serializing IDs, labels, or folds as text."""
    if "entity_id" not in s1 or "entity_id" not in candidate:
        raise ValueError("source frames require entity_id")
    def values(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return pd.DataFrame({
            f"name_{suffix}": _column(frame, ["business_name", "name"]).tolist(),
            f"address_{suffix}": _column(frame, ["business_address", "address"]).tolist(),
            f"country_{suffix}": _column(frame, ["country"]).tolist(),
        }, index=frame.index)
    left = s1.reset_index(drop=True)
    right = candidate.reset_index(drop=True)
    result = pd.concat([values(left, "a"), values(right, "b")], axis=1)
    result.insert(0, "candidate_source", source)
    result.insert(0, "candidate_entity_id", right["entity_id"].astype(str).to_numpy())
    result.insert(0, "source1_entity_id", left["entity_id"].astype(str).to_numpy())
    if folds is not None:
        result["source1_fold"] = result["source1_entity_id"].map(folds)
        result["candidate_fold"] = result["candidate_entity_id"].map(folds)
        if result[["source1_fold", "candidate_fold"]].isna().any().any():
            raise ValueError("pair endpoint missing from verified fold map")
        result["source1_fold"] = result["source1_fold"].astype(int)
        result["candidate_fold"] = result["candidate_fold"].astype(int)
    return result


def coarse_match_bin(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4+"


def _stable_rank(value: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16)


def stable_unmatched_fold(entity_id: str, *, seed: int = 42, n_folds: int = 3) -> int:
    """Deterministic fallback only for endpoints that are not supervised positives."""
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    return _stable_rank(str(entity_id), seed) % n_folds


def _component_ids(s1_ids: Iterable[str], gt: Mapping[str, Iterable[str]], fold_map: Mapping[str, int] | None = None) -> dict[str, set[str]]:
    ids = set(map(str, s1_ids))
    other = {str(candidate) for values in gt.values() for candidate in values}
    components = build_connected_components(ids, {str(k): list(map(str, v)) for k, v in gt.items()}, other)
    if fold_map:
        missing = ids - set(fold_map)
        if missing:
            raise ValueError(f"S1 IDs missing from verified fold map: {sorted(missing)[:5]}")
    return components


def select_query_ids(
    s1: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]],
    *,
    fold_map: Mapping[str, int],
    fold: int | Sequence[int],
    requested: int,
    seed: int = 42,
    prepared_ground_truth: Mapping[str, set[str]] | None = None,
    prepared_components: Mapping[str, set[str]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Select whole connected components with deterministic country/status strata.

    ``requested`` is a target, not permission to silently take first-N rows.
    Component boundaries may make the returned count differ from the target.
    """
    if requested < 0:
        raise ValueError("requested query count must be non-negative")
    ids = s1["entity_id"].astype(str).tolist()
    meta = s1.set_index("entity_id", drop=False)
    requested_folds = {int(fold)} if isinstance(fold, int) else {int(value) for value in fold}
    missing_s1 = set(ids) - set(fold_map)
    if missing_s1:
        raise ValueError(f"S1 IDs missing from verified fold map: {sorted(missing_s1)[:5]}")
    candidates = [sid for sid in ids if int(fold_map[sid]) in requested_folds]
    gt = prepared_ground_truth if prepared_ground_truth is not None else {str(k): set(map(str, v)) for k, v in ground_truth.items()}
    for sid in candidates:
        for candidate_id in gt.get(sid, set()):
            if candidate_id not in fold_map:
                raise ValueError(f"known-positive endpoint missing from verified fold map: {candidate_id}")
            if int(fold_map[candidate_id]) != int(fold_map[sid]):
                raise ValueError(f"cross-fold known positive: {sid} -> {candidate_id}")
    components = prepared_components if prepared_components is not None else _component_ids(candidates, gt, fold_map)
    component_rows: list[dict[str, Any]] = []
    candidate_set = set(candidates)
    for root, members in components.items():
        sids = sorted(set(members) & candidate_set)
        if not sids:
            continue
        countries = [str(meta.loc[sid].get("country", "")) if sid in meta.index else "" for sid in sids]
        country = min(set(countries), key=lambda value: (-countries.count(value), value)) if countries else ""
        match_count = sum(len(gt.get(sid, set())) for sid in sids)
        component_rows.append({"root": str(root), "s1_ids": sids, "size": len(sids), "country": country, "match_bin": coarse_match_bin(match_count), "rank": _stable_rank(str(root), seed)})
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        strata[(row["country"], row["match_bin"])].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: (row["rank"], row["root"]))
    # Allocate each stratum approximately its natural S1 proportion. Components
    # are indivisible, so the final count may be above or below the target.
    selected_rows: list[dict[str, Any]] = []
    selected_roots: set[str] = set()
    total = sum(row["size"] for row in component_rows)
    target = min(requested, total)
    quotas: dict[tuple[str, str], float] = {
        key: (target * sum(row["size"] for row in rows) / total if total else 0.0)
        for key, rows in strata.items()
    }
    allocations = {key: int(value) for key, value in quotas.items()}
    remainder = target - sum(allocations.values())
    for key in sorted(strata, key=lambda item: (-(quotas[item] - allocations[item]), item))[:remainder]:
        allocations[key] += 1
    for key in sorted(strata):
        count = 0
        for row in strata[key]:
            if count >= allocations[key]:
                break
            selected_rows.append(row); selected_roots.add(row["root"]); count += row["size"]
    selected_count = sum(row["size"] for row in selected_rows)
    if selected_count < target:
        remaining = sorted((row for row in component_rows if row["root"] not in selected_roots), key=lambda row: (row["rank"], row["root"]))
        for row in remaining:
            selected_rows.append(row); selected_roots.add(row["root"]); selected_count += row["size"]
            if selected_count >= target:
                break
    selected = [sid for row in selected_rows for sid in row["s1_ids"]]
    selected = sorted(selected, key=lambda sid: (_stable_rank(sid, seed), sid))
    return selected, {
        "requested": requested,
        "actual": len(selected),
        "folds": sorted(requested_folds),
        "seed": seed,
        "component_count": len(component_rows),
        "component_groups": [row["s1_ids"] for row in selected_rows],
        "strata_population": {f"{country}|{match_bin}": sum(len(r["s1_ids"]) for r in rows) for (country, match_bin), rows in strata.items()},
        "strata_selected": {f"{country}|{match_bin}": sum(len(r["s1_ids"]) for r in selected_rows if (r["country"], r["match_bin"]) == (country, match_bin)) for (country, match_bin) in strata},
        "component_safe": True,
    }


def select_training_pairs(
    candidates: pd.DataFrame,
    s1_ids: Iterable[str],
    ground_truth: Mapping[str, Iterable[str]],
    *,
    seed: int = 42,
    max_hard_negatives: int = 8,
    other_negatives: int = 2,
    fold_map: Mapping[str, int] | None = None,
    held_out_folds: set[int] | None = None,
    fold_seed: int = 42,
    n_folds: int = 3,
    candidate_source_resolver=None,
    prepared_ground_truth: Mapping[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep every positive and choose stable, positive-safe negatives per query."""
    s1_ids = list(map(str, s1_ids))
    required = {"source1_entity_id", "candidate_entity_id", "candidate_source"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates require {sorted(required)}")
    # Callers processing many queries can provide this once; the compatibility
    # path keeps the small-call API unchanged.
    gt = prepared_ground_truth if prepared_ground_truth is not None else {
        str(k): set(map(str, v)) for k, v in ground_truth.items()
    }
    selected: list[pd.DataFrame] = []
    injected = 0
    excluded_heldout_negatives = 0
    for sid in s1_ids:
        true = gt.get(sid, set())
        group = candidates[candidates["source1_entity_id"].astype(str) == sid].copy()
        if fold_map:
            if sid not in fold_map:
                raise ValueError(f"training S1 endpoint missing verified fold mapping: {sid}")
            if held_out_folds and int(fold_map[sid]) in held_out_folds:
                raise ValueError(f"held-out S1 endpoint selected for training: {sid}")
            missing_positives = {value for value in true if value not in fold_map}
            if missing_positives:
                raise ValueError(f"known-positive endpoint missing from verified fold map: {sorted(missing_positives)[:5]}")
            if held_out_folds and any(int(fold_map[value]) in held_out_folds for value in true):
                raise ValueError(f"held-out known-positive endpoint leaked into training query {sid}")
            if not group.empty:
                endpoint_folds = group["candidate_entity_id"].astype(str).map(lambda value: int(fold_map[value]) if value in fold_map else stable_unmatched_fold(value, seed=fold_seed, n_folds=n_folds))
                negative_mask = ~group["candidate_entity_id"].astype(str).isin(true)
                drop_mask = negative_mask & endpoint_folds.isin(held_out_folds or set())
                excluded_heldout_negatives += int(drop_mask.sum())
                group = group[~drop_mask].copy()
        present = set(group["candidate_entity_id"].astype(str))
        for candidate_id in sorted(true - present):
            if candidate_source_resolver is not None:
                source = str(candidate_source_resolver(candidate_id))
            elif candidate_id.startswith("S2"):
                source = "S2"
            elif candidate_id.startswith("S3"):
                source = "S3"
            else:
                raise ValueError(f"cannot determine source for injected positive {candidate_id!r}; provide candidate_source_resolver")
            group = pd.concat([group, pd.DataFrame([{ "source1_entity_id": sid, "candidate_entity_id": candidate_id, "candidate_source": source, "positive_injected_for_training": 1, "retrieval_provenance": "training_injected_positive" }])], ignore_index=True)
            injected += 1
        if group.empty:
            continue
        group["label"] = group["candidate_entity_id"].astype(str).isin(true).astype(int)
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0].copy()
        if "retrieval_score" in negatives:
            negatives["_score"] = pd.to_numeric(negatives["retrieval_score"], errors="coerce").fillna(-1.0)
        else:
            negatives["_score"] = 0.0
        negatives = negatives.sort_values(["_score", "candidate_entity_id"], ascending=[False, True])
        hard = negatives.head(max_hard_negatives)
        rest = negatives.iloc[len(hard):]
        if len(rest) > other_negatives:
            order = sorted(rest.index.tolist(), key=lambda idx: _stable_rank(str(rest.loc[idx, "candidate_entity_id"]), seed + _stable_rank(sid, seed)))
            rest = rest.loc[order[:other_negatives]]
        chosen = pd.concat([positives, hard, rest], ignore_index=True).drop(columns=["_score"], errors="ignore")
        selected.append(chosen)
    output = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=list(candidates.columns) + ["label"])
    if not output.empty:
        if "positive_injected_for_training" not in output:
            output["positive_injected_for_training"] = 0
        output["positive_injected_for_training"] = output["positive_injected_for_training"].fillna(0).astype(int)
    return output, {"queries": len(s1_ids), "rows": len(output), "injected_positives": injected, "excluded_heldout_negatives": excluded_heldout_negatives}


def prepare_pair_shards(
    candidate_shards: Sequence[str | Path],
    source1_records: Mapping[str, Mapping[str, Any]],
    source_store,
    output_dir: str | Path,
    *,
    folds: Mapping[str, int],
    ground_truth: Mapping[str, Iterable[str]] | None = None,
    target_rows: int = 75_000,
) -> dict[str, Any]:
    """Boundedly join complete candidate groups into model-ready pair shards."""
    from .candidate_io import GroupedParquetWriter, iter_complete_groups
    from .source_store import make_pair_group
    writer = GroupedParquetWriter(output_dir, target_rows=target_rows, prefix="pairs")
    queries = 0
    for candidate_group in iter_complete_groups(candidate_shards):
        pair_group = make_pair_group(candidate_group, source1_records, source_store, folds=folds, ground_truth=ground_truth)
        if not pair_group.empty:
            writer.write_group(pair_group)
        queries += 1
    writer.close()
    return {"queries": queries, "files": writer.files, "row_counts": writer.row_counts, "large_groups": writer.large_groups}
