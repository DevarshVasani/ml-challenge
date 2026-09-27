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
            self.p[x] = x; self.r[x] = 0
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.r[a] < self.r[b]: a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]: self.r[a] += 1


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _ids(raw: str) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _load(query_path: Path, truth_path: Path) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    queries: list[str] = []
    countries: dict[str, str] = {}
    with query_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (row.get("entity_id") or row.get("source1_entity_id") or "").strip()
            if s1:
                queries.append(s1); countries[s1] = (row.get("country") or "UNKNOWN").strip() or "UNKNOWN"
    truth = {s1: [] for s1 in queries}
    with truth_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (row.get("source1_entity_id") or "").strip()
            if s1 in truth:
                truth[s1] = _ids(row.get("matched_entity_ids", ""))
    return queries, countries, truth


def _components(queries: list[str], truth: dict[str, list[str]]) -> list[list[str]]:
    dsu = DSU()
    for s1 in queries:
        dsu.find(s1)
        for candidate in truth[s1]:
            dsu.union(s1, candidate)
    groups: dict[str, list[str]] = defaultdict(list)
    for s1 in queries:
        groups[dsu.find(s1)].append(s1)
    return [sorted(v) for v in groups.values()]


def _component_stats(comp: list[str], countries: dict[str, str], truth: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "s1": len(comp),
        "singletons": sum(1 for s in comp if len(truth[s]) == 0),
        "gold_edges": sum(len(truth[s]) for s in comp),
        "countries": Counter(countries.get(s, "UNKNOWN") for s in comp),
    }


def _assign(
    comps: list[list[str]], countries: dict[str, str], truth: dict[str, list[str]], proportions: dict[str, float], seed: int
) -> dict[str, str]:
    rng = random.Random(seed)
    parts = list(proportions)
    comp_rows = [(c, _component_stats(c, countries, truth), rng.random()) for c in comps]
    comp_rows.sort(key=lambda x: (-x[1]["s1"], -x[1]["gold_edges"], x[2]))

    totals = {
        "s1": sum(x[1]["s1"] for x in comp_rows),
        "singletons": sum(x[1]["singletons"] for x in comp_rows),
        "gold_edges": sum(x[1]["gold_edges"] for x in comp_rows),
    }
    all_countries = sorted({c for _, st, _ in comp_rows for c in st["countries"]})
    country_totals = {c: sum(st["countries"].get(c, 0) for _, st, _ in comp_rows) for c in all_countries}
    state = {p: {"s1": 0, "singletons": 0, "gold_edges": 0, "countries": Counter()} for p in parts}

    def score(part: str, st: dict[str, Any]) -> float:
        after = dict(state[part])
        after["countries"] = state[part]["countries"].copy()
        for key in ("s1", "singletons", "gold_edges"):
            after[key] += st[key]
        after["countries"].update(st["countries"])
        loss = 0.0
        for key in ("s1", "singletons", "gold_edges"):
            target = max(1.0, totals[key] * proportions[part])
            loss += ((after[key] - target) / target) ** 2
        for country in all_countries:
            target = max(1.0, country_totals[country] * proportions[part])
            loss += 0.5 * ((after["countries"].get(country, 0) - target) / target) ** 2
        return loss

    assignment: dict[str, str] = {}
    for comp, st, _ in comp_rows:
        chosen = min(parts, key=lambda p: (score(p, st), state[p]["s1"]))
        state[chosen]["s1"] += st["s1"]
        state[chosen]["singletons"] += st["singletons"]
        state[chosen]["gold_edges"] += st["gold_edges"]
        state[chosen]["countries"].update(st["countries"])
        for s1 in comp:
            assignment[s1] = chosen
    return assignment


def build_member_d_split(config: dict[str, Any]) -> dict[str, Any]:
    q = Path(config["threshold_queries_path"]); t = Path(config["truth_path"])
    out = Path(config.get("output_dir", ".")) / "configs" / "member_d"; out.mkdir(parents=True, exist_ok=True)
    proportions = config.get("proportions", {"fusion_fit": .4, "calibration": .3, "selection": .3})
    if abs(sum(float(x) for x in proportions.values()) - 1.0) > 1e-9:
        raise ValueError("Partition proportions must sum to 1")
    seed = int(config.get("seed", 42))
    queries, countries, truth = _load(q, t)
    comps = _components(queries, truth)
    assignment = _assign(comps, countries, truth, proportions, seed)

    split_path = out / "split.tsv"
    with split_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["source1_entity_id", "member_d_partition"])
        for s1 in queries: w.writerow([s1, assignment[s1]])
    # Compatibility alias for older code on Daksh.
    (out / "selection_split.tsv").write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")

    excluded = set(queries)
    for vals in truth.values(): excluded.update(vals)
    with (out / "base_model_exclusion_ids.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["entity_id"])
        for eid in sorted(excluded): w.writerow([eid])

    part_counts = Counter(assignment.values())
    singleton_counts = Counter(assignment[s] for s in queries if not truth[s])
    edge_counts = Counter()
    country_counts: dict[str, Counter] = defaultdict(Counter)
    for s in queries:
        p = assignment[s]; edge_counts[p] += len(truth[s]); country_counts[p][countries.get(s, "UNKNOWN")] += 1
    component_payload = json.dumps(sorted(sorted(c) for c in comps), separators=(",", ":"))
    manifest = {
        "seed": seed,
        "git_commit": _git(),
        "threshold_queries_sha256": _sha(q),
        "evaluation_truth_sha256": _sha(t),
        "component_hash": hashlib.sha256(component_payload.encode()).hexdigest(),
        "component_count": len(comps),
        "partition_counts": dict(part_counts),
        "country_counts": {p: dict(c) for p, c in country_counts.items()},
        "singleton_counts": dict(singleton_counts),
        "gold_edge_counts": dict(edge_counts),
        "proportions": proportions,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"assignment": assignment, "manifest": manifest}


def assert_no_excluded_entities(training_manifest_path: str | Path, exclusion_ids_path: str | Path) -> None:
    with Path(exclusion_ids_path).open(newline="", encoding="utf-8") as f:
        excluded = {row["entity_id"].strip() for row in csv.DictReader(f, delimiter="\t") if row.get("entity_id", "").strip()}
    path = Path(training_manifest_path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8")); ids = set(map(str, data.get("entity_ids", [])))
    else:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t"); ids = {(r.get("entity_id") or r.get("source1_entity_id") or "").strip() for r in reader}
    overlap = sorted((ids - {""}) & excluded)
    if overlap: raise ValueError(f"Base-model training overlaps D holdouts: {overlap[:10]}")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); args = p.parse_args()
    build_member_d_split(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__": main()
