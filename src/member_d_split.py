"""Create deterministic, component-safe Member D fusion/calibration/selection partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class DSU:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}
        self.r: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.p:
            self.p[x] = x
            self.r[x] = 0
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _ids(raw: str) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _load(
    query_path: Path,
    truth_path: Path,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    queries: list[str] = []
    countries: dict[str, str] = {}

    with query_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (
                row.get("entity_id")
                or row.get("source1_entity_id")
                or ""
            ).strip()
            if s1:
                queries.append(s1)
                countries[s1] = (
                    (row.get("country") or "UNKNOWN").strip()
                    or "UNKNOWN"
                )

    if len(queries) != len(set(queries)):
        raise ValueError("Threshold query file contains duplicate S1 IDs")

    truth = {s1: [] for s1 in queries}
    with truth_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (row.get("source1_entity_id") or "").strip()
            if s1 in truth:
                truth[s1] = _ids(row.get("matched_entity_ids", ""))

    return queries, countries, truth


def _components(
    queries: list[str],
    truth: dict[str, list[str]],
) -> list[list[str]]:
    dsu = DSU()
    for s1 in queries:
        dsu.find(s1)
        for candidate in truth[s1]:
            dsu.union(s1, candidate)

    groups: dict[str, list[str]] = defaultdict(list)
    for s1 in queries:
        groups[dsu.find(s1)].append(s1)
    return [sorted(v) for v in groups.values()]


def _component_stats(
    comp: list[str],
    countries: dict[str, str],
    truth: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "s1": len(comp),
        "singletons": sum(1 for s in comp if len(truth[s]) == 0),
        "gold_edges": sum(len(truth[s]) for s in comp),
        "countries": Counter(
            countries.get(s, "UNKNOWN") for s in comp
        ),
    }


def _assign(
    comps: list[list[str]],
    countries: dict[str, str],
    truth: dict[str, list[str]],
    proportions: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign connected components while minimizing GLOBAL partition imbalance.

    Important:
    The old implementation compared only the candidate partition's own loss.
    With 40/30/30 targets this can prefer the smaller 30% partitions and leave
    fusion_fit empty. Here each hypothetical assignment is evaluated against
    the total loss across *all* partitions.

    Components are indivisible, so exact proportions are not always possible,
    but S1 count, singleton count, gold-edge count and country mix are balanced
    as closely as component constraints allow.
    """
    rng = random.Random(seed)
    parts = list(proportions)

    comp_rows = [
        (comp, _component_stats(comp, countries, truth), rng.random())
        for comp in comps
    ]
    # Harder/larger components first; seeded random breaks exact ties
    # deterministically.
    comp_rows.sort(
        key=lambda x: (
            -x[1]["s1"],
            -x[1]["gold_edges"],
            -x[1]["singletons"],
            x[2],
        )
    )

    totals = {
        "s1": sum(st["s1"] for _, st, _ in comp_rows),
        "singletons": sum(
            st["singletons"] for _, st, _ in comp_rows
        ),
        "gold_edges": sum(
            st["gold_edges"] for _, st, _ in comp_rows
        ),
    }
    all_countries = sorted(
        {
            country
            for _, st, _ in comp_rows
            for country in st["countries"]
        }
    )
    country_totals = {
        country: sum(
            st["countries"].get(country, 0)
            for _, st, _ in comp_rows
        )
        for country in all_countries
    }

    state = {
        part: {
            "s1": 0,
            "singletons": 0,
            "gold_edges": 0,
            "countries": Counter(),
        }
        for part in parts
    }

    def global_loss(
        hypothetical_part: str,
        component_stats: dict[str, Any],
    ) -> float:
        loss = 0.0

        for part in parts:
            add = component_stats if part == hypothetical_part else None

            s1_value = state[part]["s1"] + (
                add["s1"] if add else 0
            )
            singleton_value = state[part]["singletons"] + (
                add["singletons"] if add else 0
            )
            gold_value = state[part]["gold_edges"] + (
                add["gold_edges"] if add else 0
            )

            for key, value in (
                ("s1", s1_value),
                ("singletons", singleton_value),
                ("gold_edges", gold_value),
            ):
                # If a statistic is absent globally (e.g. zero gold edges),
                # it contributes no balancing signal.
                if totals[key] == 0:
                    continue
                target = totals[key] * proportions[part]
                target = max(target, 1.0)
                loss += ((value - target) / target) ** 2

            for country in all_countries:
                value = state[part]["countries"].get(country, 0)
                if add:
                    value += add["countries"].get(country, 0)
                total_country = country_totals[country]
                if total_country == 0:
                    continue
                target = max(
                    total_country * proportions[part],
                    1.0,
                )
                # Country is a secondary balancing criterion.
                loss += 0.5 * ((value - target) / target) ** 2

        return loss

    assignment: dict[str, str] = {}

    for comp, st, _ in comp_rows:
        chosen = min(
            parts,
            key=lambda part: (
                global_loss(part, st),
                state[part]["s1"],
                parts.index(part),
            ),
        )

        state[chosen]["s1"] += st["s1"]
        state[chosen]["singletons"] += st["singletons"]
        state[chosen]["gold_edges"] += st["gold_edges"]
        state[chosen]["countries"].update(st["countries"])

        for s1 in comp:
            assignment[s1] = chosen

    missing_parts = [part for part in parts if state[part]["s1"] == 0]
    if missing_parts:
        raise ValueError(
            "Split assignment produced empty partition(s): "
            f"{missing_parts}. Check component sizes/proportions."
        )

    return assignment


def build_member_d_split(config: dict[str, Any]) -> dict[str, Any]:
    q = Path(config["threshold_queries_path"])
    t = Path(config["truth_path"])
    out = Path(config.get("output_dir", ".")) / "configs" / "member_d"
    out.mkdir(parents=True, exist_ok=True)

    proportions = config.get(
        "proportions",
        {
            "fusion_fit": 0.4,
            "calibration": 0.3,
            "selection": 0.3,
        },
    )
    if abs(sum(float(x) for x in proportions.values()) - 1.0) > 1e-9:
        raise ValueError("Partition proportions must sum to 1")
    if any(float(v) <= 0 for v in proportions.values()):
        raise ValueError("Every partition proportion must be positive")

    seed = int(config.get("seed", 42))
    queries, countries, truth = _load(q, t)
    comps = _components(queries, truth)

    if len(comps) < len(proportions):
        raise ValueError(
            "Too few connected components to populate all partitions: "
            f"{len(comps)} components for {len(proportions)} partitions"
        )

    assignment = _assign(
        comps,
        countries,
        truth,
        proportions,
        seed,
    )

    split_path = out / "split.tsv"
    with split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["source1_entity_id", "member_d_partition"]
        )
        for s1 in queries:
            writer.writerow([s1, assignment[s1]])

    # Compatibility alias for older D code.
    (out / "selection_split.tsv").write_text(
        split_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    excluded = set(queries)
    for vals in truth.values():
        excluded.update(vals)

    exclusion_path = out / "base_model_exclusion_ids.tsv"
    with exclusion_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["entity_id"])
        for eid in sorted(excluded):
            writer.writerow([eid])

    part_counts = Counter(assignment.values())
    singleton_counts = Counter(
        assignment[s] for s in queries if not truth[s]
    )
    edge_counts = Counter()
    country_counts: dict[str, Counter] = defaultdict(Counter)

    for s1 in queries:
        part = assignment[s1]
        edge_counts[part] += len(truth[s1])
        country_counts[part][countries.get(s1, "UNKNOWN")] += 1

    component_payload = json.dumps(
        sorted(sorted(c) for c in comps),
        separators=(",", ":"),
    )

    manifest = {
        "seed": seed,
        "git_commit": _git(),
        "threshold_queries_sha256": _sha(q),
        "evaluation_truth_sha256": _sha(t),
        "component_hash": hashlib.sha256(
            component_payload.encode()
        ).hexdigest(),
        "component_count": len(comps),
        "partition_counts": {
            part: int(part_counts.get(part, 0))
            for part in proportions
        },
        "country_counts": {
            part: dict(country_counts.get(part, Counter()))
            for part in proportions
        },
        "singleton_counts": {
            part: int(singleton_counts.get(part, 0))
            for part in proportions
        },
        "gold_edge_counts": {
            part: int(edge_counts.get(part, 0))
            for part in proportions
        },
        "proportions": proportions,
    }

    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "assignment": assignment,
        "manifest": manifest,
        "split_path": str(split_path),
        "exclusion_path": str(exclusion_path),
    }


def assert_no_excluded_entities(
    training_manifest_path: str | Path,
    exclusion_ids_path: str | Path,
) -> None:
    with Path(exclusion_ids_path).open(
        newline="", encoding="utf-8"
    ) as f:
        excluded = {
            row["entity_id"].strip()
            for row in csv.DictReader(f, delimiter="\t")
            if row.get("entity_id", "").strip()
        }

    path = Path(training_manifest_path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = set(map(str, data.get("entity_ids", [])))
    else:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            ids = {
                (
                    row.get("entity_id")
                    or row.get("source1_entity_id")
                    or ""
                ).strip()
                for row in reader
            }

    overlap = sorted((ids - {""}) & excluded)
    if overlap:
        raise ValueError(
            f"Base-model training overlaps D holdouts: {overlap[:10]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build_member_d_split(
        json.loads(
            Path(args.config).read_text(encoding="utf-8")
        )
    )


if __name__ == "__main__":
    main()
