"""Validation split generator for Entity Resolution.

Principles:
1. Group preservation:
   - Keep each S1 entity and all its known matching records (S2/S3) in the same fold.
   - If multiple S1 entities share a matched record, union them into the same connected component.
2. Leakage prevention:
   - Held-out records (S1, S2, S3) in a validation fold must never enter training pairs, even as negatives.
3. Balance:
   - Create 3 fixed validation folds balanced across countries (e.g., India, US), match counts, and singletons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class DisjointSetUnion:
    """Disjoint Set Union (Union-Find) with path compression and union by rank."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                root1, root2 = root2, root1
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def get_components(self) -> Dict[str, Set[str]]:
        components: Dict[str, Set[str]] = defaultdict(set)
        for item in list(self.parent.keys()):
            root = self.find(item)
            components[root].add(item)
        return components


def parse_id_list(cell: Optional[str]) -> List[str]:
    if not cell:
        return []
    cell_str = str(cell).strip()
    if not cell_str or cell_str == "nan":
        return []
    return [x.strip() for x in cell_str.split(",") if x.strip()]


def load_s1_metadata(s1_tsv_path: str) -> Dict[str, Dict[str, str]]:
    """Load S1 metadata: entity_id -> {country, business_name, etc.}."""
    meta: Dict[str, Dict[str, str]] = {}
    with open(s1_tsv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                meta[eid] = {
                    "country": row.get("country", "UNKNOWN").strip(),
                    "business_name": row.get("business_name", "").strip(),
                }
    return meta


def load_ground_truth(gt_tsv_path: str) -> Dict[str, List[str]]:
    """Load ground truth matches: s1_id -> [s2/s3 ids]."""
    gt: Dict[str, List[str]] = {}
    with open(gt_tsv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            if s1_id:
                gt[s1_id] = parse_id_list(row.get("matched_entity_ids", ""))
    return gt


def build_connected_components(
    s1_ids: Set[str],
    gt_matches: Dict[str, List[str]],
    other_entity_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Build connected components using DSU across ground truth matches."""
    dsu = DisjointSetUnion()

    # Ensure all S1 entities exist in DSU
    for s1_id in s1_ids:
        dsu.find(s1_id)

    # Union S1 entity with all its matched records
    for s1_id, matches in gt_matches.items():
        if s1_id not in s1_ids:
            continue
        for m_id in matches:
            dsu.union(s1_id, m_id)

    # Unmatched S2/S3 records in training set
    if other_entity_ids:
        for eid in other_entity_ids:
            dsu.find(eid)

    return dsu.get_components()


def partition_components(
    components: Dict[str, Set[str]],
    s1_meta: Dict[str, Dict[str, str]],
    gt_matches: Dict[str, List[str]],
    n_splits: int = 3,
    seed: int = 42,
) -> Dict[str, int]:
    """Partition connected components across n_splits folds balancing country & match counts.

    Returns:
        Mapping of component_root_id -> fold_id (0 .. n_splits-1)
    """
    rng = random.Random(seed)

    # Compute component statistics
    comp_stats = []
    for root, members in components.items():
        s1_in_comp = [m for m in members if m.startswith("S1-") or m in s1_meta]
        num_s1 = len(s1_in_comp)

        # Count matches originating from S1s in this component
        match_count = 0
        countries = []
        for s1 in s1_in_comp:
            match_count += len(gt_matches.get(s1, []))
            country = s1_meta.get(s1, {}).get("country", "UNKNOWN")
            countries.append(country)

        primary_country = max(set(countries), key=countries.count) if countries else "UNKNOWN"
        is_singleton = (match_count == 0)

        comp_stats.append({
            "root": root,
            "members": members,
            "num_s1": num_s1,
            "match_count": match_count,
            "country": primary_country,
            "is_singleton": is_singleton,
            "total_entities": len(members),
        })

    # Group components by country and singleton status for stratified greedy bin-packing
    groups = defaultdict(list)
    for c in comp_stats:
        key = (c["country"], c["is_singleton"])
        groups[key].append(c)

    fold_assignments: Dict[str, int] = {}
    fold_loads = [{
        "s1_count": 0,
        "match_count": 0,
        "country_s1": defaultdict(int),
        "country_matches": defaultdict(int),
    } for _ in range(n_splits)]

    # Assign each group
    for (country, is_singleton), items in sorted(groups.items(), key=lambda x: str(x[0])):
        # Sort items descending by match_count and num_s1 with tie-break randomness
        rng.shuffle(items)
        items.sort(key=lambda x: (x["match_count"], x["num_s1"], x["total_entities"]), reverse=True)

        for item in items:
            # Pick fold with minimum weighted load for this specific stratum and overall
            best_fold = 0
            best_score = float("inf")

            for f_idx in range(n_splits):
                load = fold_loads[f_idx]
                if not is_singleton:
                    # Score based on country match count & s1 count
                    score = (
                        load["country_matches"][country] * 3.0
                        + load["country_s1"][country] * 1.0
                        + load["match_count"] * 0.5
                    )
                else:
                    # Score based on country s1 count
                    score = (
                        load["country_s1"][country] * 2.0
                        + load["s1_count"] * 1.0
                    )

                if score < best_score:
                    best_score = score
                    best_fold = f_idx

            fold_assignments[item["root"]] = best_fold
            fold_loads[best_fold]["s1_count"] += item["num_s1"]
            fold_loads[best_fold]["match_count"] += item["match_count"]
            fold_loads[best_fold]["country_s1"][country] += item["num_s1"]
            fold_loads[best_fold]["country_matches"][country] += item["match_count"]

    return fold_assignments


def create_folds(
    s1_tsv_path: str,
    gt_tsv_path: str,
    s2_tsv_path: Optional[str] = None,
    s3_tsv_path: Optional[str] = None,
    n_splits: int = 3,
    seed: int = 42,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Create entity-to-fold mapping and summary statistics."""
    s1_meta = load_s1_metadata(s1_tsv_path)
    s1_ids = set(s1_meta.keys())
    gt_matches = load_ground_truth(gt_tsv_path)

    other_entity_ids: Set[str] = set()

    def _collect_ids(path: Optional[str]):
        if path and os.path.exists(path):
            with open(path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    eid = row.get("entity_id", "").strip()
                    if eid:
                        other_entity_ids.add(eid)

    _collect_ids(s2_tsv_path)
    _collect_ids(s3_tsv_path)

    components = build_connected_components(s1_ids, gt_matches, other_entity_ids)
    comp_to_fold = partition_components(components, s1_meta, gt_matches, n_splits=n_splits, seed=seed)

    # Assign every individual entity to its component's fold
    entity_to_fold: Dict[str, int] = {}
    for root, members in components.items():
        assigned_fold = comp_to_fold[root]
        for member in members:
            entity_to_fold[member] = assigned_fold

    # Generate summary report
    summary = generate_split_summary(entity_to_fold, s1_meta, gt_matches, n_splits)
    return entity_to_fold, summary


def generate_split_summary(
    entity_to_fold: Dict[str, int],
    s1_meta: Dict[str, Dict[str, str]],
    gt_matches: Dict[str, List[str]],
    n_splits: int = 3,
) -> Dict[str, Any]:
    """Compute detailed summary of split distribution across folds."""
    fold_stats = {
        f: {
            "total_entities": 0,
            "s1_count": 0,
            "s2_count": 0,
            "s3_count": 0,
            "singletons": 0,
            "non_singletons": 0,
            "total_matches": 0,
            "countries": defaultdict(lambda: {"s1": 0, "matches": 0, "singletons": 0}),
        }
        for f in range(n_splits)
    }

    for eid, fold in entity_to_fold.items():
        stats = fold_stats[fold]
        stats["total_entities"] += 1
        if eid.startswith("S1-") or eid in s1_meta:
            stats["s1_count"] += 1
            matches = gt_matches.get(eid, [])
            match_count = len(matches)
            country = s1_meta.get(eid, {}).get("country", "UNKNOWN")
            stats["total_matches"] += match_count
            stats["countries"][country]["s1"] += 1
            stats["countries"][country]["matches"] += match_count
            if match_count == 0:
                stats["singletons"] += 1
                stats["countries"][country]["singletons"] += 1
            else:
                stats["non_singletons"] += 1
        elif eid.startswith("S2-"):
            stats["s2_count"] += 1
        elif eid.startswith("S3-"):
            stats["s3_count"] += 1

    # Format for JSON serialization
    serialized_stats = {}
    for f in range(n_splits):
        serialized_stats[f"fold_{f}"] = {
            "total_entities": fold_stats[f]["total_entities"],
            "s1_count": fold_stats[f]["s1_count"],
            "s2_count": fold_stats[f]["s2_count"],
            "s3_count": fold_stats[f]["s3_count"],
            "singletons": fold_stats[f]["singletons"],
            "non_singletons": fold_stats[f]["non_singletons"],
            "total_matches": fold_stats[f]["total_matches"],
            "countries": {k: dict(v) for k, v in fold_stats[f]["countries"].items()},
        }

    return {
        "n_splits": n_splits,
        "total_unique_entities": len(entity_to_fold),
        "total_s1": len(s1_meta),
        "folds": serialized_stats,
    }


def verify_split_integrity(
    entity_to_fold: Dict[str, int],
    gt_matches: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """Verify that no matches cross fold boundaries and no held-out entities leak."""
    issues = []
    for s1_id, matches in gt_matches.items():
        if s1_id not in entity_to_fold:
            continue
        s1_fold = entity_to_fold[s1_id]
        for m_id in matches:
            if m_id in entity_to_fold:
                m_fold = entity_to_fold[m_id]
                if m_fold != s1_fold:
                    issues.append(
                        f"Cross-fold match leak: {s1_id} (fold {s1_fold}) matched with {m_id} (fold {m_fold})"
                    )

    passed = len(issues) == 0
    return passed, issues


def get_train_val_entities(
    entity_to_fold: Dict[str, int],
    val_fold: int,
) -> Tuple[Set[str], Set[str]]:
    """Return strictly disjoint (train_entity_ids, val_entity_ids) for a given val fold.

    Ensures held-out entities are excluded from training pairs / candidate generation / negatives.
    """
    train_ids = {eid for eid, fold in entity_to_fold.items() if fold != val_fold}
    val_ids = {eid for eid, fold in entity_to_fold.items() if fold == val_fold}
    assert train_ids.isdisjoint(val_ids), "Train and validation entity sets must be strictly disjoint!"
    return train_ids, val_ids


def save_folds_tsv(entity_to_fold: Dict[str, int], out_path: str) -> None:
    """Save entity -> fold assignments to TSV."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["entity_id", "fold"])
        for eid in sorted(entity_to_fold.keys()):
            writer.writerow([eid, entity_to_fold[eid]])


def load_folds_tsv(folds_path: str) -> Dict[str, int]:
    """Load entity -> fold assignments from TSV."""
    mapping: Dict[str, int] = {}
    with open(folds_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[row["entity_id"]] = int(row["fold"])
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Create balanced validation folds for Entity Resolution.")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory containing dataset/train files")
    parser.add_argument("--s1", type=str, default=None, help="Path to train_source1.tsv")
    parser.add_argument("--s2", type=str, default=None, help="Path to train_source2.tsv (optional)")
    parser.add_argument("--s3", type=str, default=None, help="Path to train_source3.tsv (optional)")
    parser.add_argument("--gt", type=str, default=None, help="Path to train_ground_truth.tsv")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of folds (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out-folds", type=str, default="configs/folds.tsv", help="Path to save fold assignments")
    parser.add_argument("--out-summary", type=str, default="reports/split_summary.json", help="Path to save summary report")
    parser.add_argument("--verify", action="store_true", help="Verify fold isolation and integrity")

    args = parser.parse_args()

    # Resolve paths if data-dir provided
    s1_path = args.s1 or (os.path.join(args.data_dir, "train_source1.tsv") if args.data_dir else None)
    gt_path = args.gt or (os.path.join(args.data_dir, "train_ground_truth.tsv") if args.data_dir else None)
    s2_path = args.s2 or (os.path.join(args.data_dir, "train_source2.tsv") if args.data_dir else None)
    s3_path = args.s3 or (os.path.join(args.data_dir, "train_source3.tsv") if args.data_dir else None)

    if not s1_path or not gt_path:
        print("Error: Both S1 path and Ground Truth path are required (use --s1/--gt or --data-dir).")
        parser.print_help()
        sys.exit(1)

    print(f"Generating {args.n_splits}-fold split from {s1_path} and {gt_path}...")
    entity_to_fold, summary = create_folds(
        s1_tsv_path=s1_path,
        gt_tsv_path=gt_path,
        s2_tsv_path=s2_path,
        s3_tsv_path=s3_path,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    if args.verify:
        gt_matches = load_ground_truth(gt_path)
        passed, issues = verify_split_integrity(entity_to_fold, gt_matches)
        if passed:
            print("Verification PASSED: All matched records strictly preserved in the same fold!")
        else:
            print(f"Verification FAILED: {len(issues)} cross-fold leaks found!")
            for issue in issues[:10]:
                print(f"  - {issue}")
            sys.exit(1)

    save_folds_tsv(entity_to_fold, args.out_folds)
    print(f"Saved fold assignments to: {args.out_folds}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split summary to: {args.out_summary}")

    print("\n--- Fold Distribution Summary ---")
    for f_name, stats in summary["folds"].items():
        print(f"\n[{f_name}]")
        print(f"  Total S1 Entities: {stats['s1_count']}")
        print(f"  Total Matches:     {stats['total_matches']}")
        print(f"  Singletons:        {stats['singletons']}")
        print(f"  Non-Singletons:    {stats['non_singletons']}")
        for ctry, cstats in stats["countries"].items():
            print(f"    - {ctry:<8} | S1: {cstats['s1']:<6} | Matches: {cstats['matches']:<6} | Singletons: {cstats['singletons']:<6}")


if __name__ == "__main__":
    main()
