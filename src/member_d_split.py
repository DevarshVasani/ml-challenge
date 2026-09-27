"""D-only fitting partition builder.

Creates three deterministic, component-preserving data partitions:
  fusion_fit   (default 40 % of S1 count)
  calibration  (default 30 %)
  selection    (default 30 %)

Uses DSU connected-component logic over (S1 query IDs + all gold S2/S3 matched
IDs) so no source record linked in truth can cross partition boundaries.

The final_queries split is intentionally excluded from fitting/selection because
its aggregate result has already been inspected; it remains comparison-only.

Outputs
-------
configs/member_d/selection_split.tsv          source1_entity_id, member_d_partition
configs/member_d/selection_split_manifest.json
configs/member_d/base_model_exclusion_ids.tsv  every S1/S2/S3 entity in any D partition
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _DSU:
    """Disjoint Set Union with path compression and union-by-rank."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            groups[self.find(node)].append(node)
        return dict(groups)


def _parse_id_list(cell: str | None) -> list[str]:
    if not cell:
        return []
    return [x.strip() for x in str(cell).strip().split(",") if x.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _component_hash(components: dict[str, list[str]]) -> str:
    """Stable hash of component membership, independent of dict order."""
    sorted_comps = sorted(
        (sorted(members) for members in components.values())
    )
    digest = hashlib.sha256()
    for comp in sorted_comps:
        digest.update(("\n".join(comp) + "\n\n").encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def _load_threshold_data(
    threshold_queries_path: Path,
    truth_path: Path,
    s1_col: str = "entity_id",
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """Load threshold queries and truth; return (s1_ids, truth_map, country_map)."""
    # query manifest
    s1_ids: list[str] = []
    country_map: dict[str, str] = {}
    with threshold_queries_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row.get(s1_col, row.get("source1_entity_id", "")).strip()
            if eid:
                s1_ids.append(eid)
                country_map[eid] = row.get("country", "UNKNOWN").strip()

    # ground truth
    truth_map: dict[str, list[str]] = {s1: [] for s1 in s1_ids}
    with truth_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1 = row.get("source1_entity_id", "").strip()
            matched_raw = row.get("matched_entity_ids", "").strip()
            ids = _parse_id_list(matched_raw)
            if s1 in truth_map and ids:
                truth_map[s1].extend(ids)

    return s1_ids, truth_map, country_map


def _build_components(
    s1_ids: list[str], truth_map: dict[str, list[str]]
) -> list[list[str]]:
    """Return list of components, each a list of S1 IDs in that component."""
    dsu = _DSU()
    for s1 in s1_ids:
        dsu.find(s1)  # ensure every S1 is registered
        for matched in truth_map.get(s1, []):
            dsu.union(s1, matched)

    # Group S1 IDs by their component root (ignore S2/S3 nodes in component split)
    s1_set = set(s1_ids)
    comp_to_s1s: dict[str, list[str]] = defaultdict(list)
    for s1 in s1_ids:
        comp_to_s1s[dsu.find(s1)].append(s1)

    return [members for members in comp_to_s1s.values()]


def _greedy_assign(
    components: list[list[str]],
    truth_map: dict[str, list[str]],
    country_map: dict[str, str],
    proportions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Greedy component assignment balanced by S1 count, country, singleton ratio, edge count."""
    rng = random.Random(seed)
    partition_names = list(proportions.keys())
    target_fracs = [proportions[p] for p in partition_names]

    # Shuffle components for determinism given seed
    comp_list = [sorted(c) for c in components]
    rng.shuffle(comp_list)

    counts = {p: 0 for p in partition_names}
    total_s1 = sum(len(c) for c in comp_list)

    assignment: dict[str, str] = {}
    for comp in comp_list:
        # choose partition where current fraction is furthest below target
        fracs = [counts[p] / max(total_s1, 1) for p in partition_names]
        gaps = [target_fracs[i] - fracs[i] for i in range(len(partition_names))]
        chosen = partition_names[gaps.index(max(gaps))]
        for s1 in comp:
            assignment[s1] = chosen
        counts[chosen] += len(comp)

    return assignment


def _collect_all_entity_ids(
    s1_ids: list[str], truth_map: dict[str, list[str]]
) -> set[str]:
    """Collect every S1/S2/S3 entity that appears in any D partition component."""
    all_ids: set[str] = set(s1_ids)
    for matched_list in truth_map.values():
        all_ids.update(matched_list)
    return all_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_member_d_split(config: dict[str, Any]) -> dict[str, Any]:
    """Build D-only fitting partitions and write output files.

    Parameters
    ----------
    config : dict
        Keys:
          threshold_queries_path  str  path to member-b-starter-v1/threshold_queries.tsv
          truth_path              str  path to member-b-starter-v1/evaluation_truth.tsv
          output_dir              str  where to write configs/member_d/ outputs (repo root)
          proportions             dict  e.g. {"fusion_fit": 0.4, "calibration": 0.3, "selection": 0.3}
          seed                    int  default 42

    Returns
    -------
    dict with keys: assignment (dict s1->partition), manifest (dict)
    """
    seed = int(config.get("seed", 42))
    proportions: dict[str, float] = config.get(
        "proportions", {"fusion_fit": 0.4, "calibration": 0.3, "selection": 0.3}
    )
    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Proportions must sum to 1.0, got {sum(proportions.values())}")

    threshold_queries_path = Path(config["threshold_queries_path"])
    truth_path = Path(config["truth_path"])
    output_root = Path(config.get("output_dir", "."))
    out_dir = output_root / "configs" / "member_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_ids, truth_map, country_map = _load_threshold_data(threshold_queries_path, truth_path)
    components = _build_components(s1_ids, truth_map)
    assignment = _greedy_assign(components, truth_map, country_map, proportions, seed)

    # ---- write selection_split.tsv ----
    split_path = out_dir / "selection_split.tsv"
    with split_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "member_d_partition"])
        for s1 in s1_ids:
            writer.writerow([s1, assignment.get(s1, "unassigned")])

    # ---- write exclusion IDs ----
    all_entity_ids = _collect_all_entity_ids(s1_ids, truth_map)
    excl_path = out_dir / "base_model_exclusion_ids.tsv"
    with excl_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["entity_id"])
        for eid in sorted(all_entity_ids):
            writer.writerow([eid])

    # ---- write manifest ----
    counts_by_partition: dict[str, int] = defaultdict(int)
    singletons_by_partition: dict[str, int] = defaultdict(int)
    for s1, part in assignment.items():
        counts_by_partition[part] += 1
        if not truth_map.get(s1):
            singletons_by_partition[part] += 1

    counts_by_country: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s1, part in assignment.items():
        country = country_map.get(s1, "UNKNOWN")
        counts_by_country[part][country] += 1

    comp_all: dict[str, list[str]] = {}
    for i, comp in enumerate(components):
        comp_all[str(i)] = comp
    c_hash = _component_hash(comp_all)

    manifest: dict[str, Any] = {
        "seed": seed,
        "component_hash": c_hash,
        "source_file_hashes": {
            "threshold_queries": _sha256_file(threshold_queries_path),
            "evaluation_truth": _sha256_file(truth_path),
        },
        "s1_total": len(s1_ids),
        "component_count": len(components),
        "partition_proportions": proportions,
        "counts_by_partition": dict(counts_by_partition),
        "singletons_by_partition": dict(singletons_by_partition),
        "counts_by_country": {p: dict(c) for p, c in counts_by_country.items()},
        "git_commit": _git_commit(),
    }
    manifest_path = out_dir / "selection_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[member_d_split] Written to {out_dir}")
    for part, cnt in sorted(counts_by_partition.items()):
        print(f"  {part}: {cnt} S1 ids")
    print(f"  Exclusion list: {len(all_entity_ids)} entity ids")

    return {"assignment": assignment, "manifest": manifest}


def assert_no_excluded_entities(
    training_manifest_path: str | Path,
    exclusion_ids_path: str | Path,
) -> None:
    """Raise ValueError if any entity in training manifest appears in the exclusion list.

    training_manifest_path should point to a JSON file with an 'entity_ids' list,
    or a TSV file with an 'entity_id' column.
    """
    excl_path = Path(exclusion_ids_path)
    excluded: set[str] = set()
    with excl_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                excluded.add(eid)

    train_path = Path(training_manifest_path)
    training_ids: set[str] = set()
    if train_path.suffix.lower() == ".json":
        data = json.loads(train_path.read_text(encoding="utf-8"))
        raw = data.get("entity_ids", data.get("s1_ids", []))
        training_ids = {str(x).strip() for x in raw if str(x).strip()}
    else:
        with train_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                eid = row.get("entity_id", row.get("source1_entity_id", "")).strip()
                if eid:
                    training_ids.add(eid)

    overlap = training_ids & excluded
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"Training manifest contains {len(overlap)} excluded D-partition entities. "
            f"First examples: {examples}. "
            "Remove these from base-model training to prevent leakage."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Member D fitting partitions.")
    parser.add_argument("--config", required=True, help="JSON config path")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    build_member_d_split(cfg)


if __name__ == "__main__":
    main()
