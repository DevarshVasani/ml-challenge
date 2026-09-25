"""Future compute experiment for per-channel K and optional pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .candidate_index import ExactMatchIndex, SourceTfidfIndex
from .candidates import candidate_diagnostics, generate_candidates, load_ground_truth, load_source, prune_candidates, union_candidates
from .neural_contracts import config_hash
from .neural_data import select_query_ids
from .split import load_folds_tsv


def slice_candidates(cands: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("K must be positive")
    name_rank = cands["name_rank"].astype(int) if "name_rank" in cands else pd.Series(0, index=cands.index)
    address_rank = cands["address_rank"].astype(int) if "address_rank" in cands else pd.Series(0, index=cands.index)
    name = (name_rank > 0) & (name_rank <= k)
    address = (address_rank > 0) & (address_rank <= k)
    exact = cands.get("retrieved_by_exact", pd.Series(0, index=cands.index)).astype(int).eq(1)
    result = cands[name | address | exact].copy()
    result["retrieved_by_name"] = name[result.index].astype(int)
    result["retrieved_by_address"] = address[result.index].astype(int)
    return result


def plan(config: dict[str, Any]) -> dict[str, Any]:
    return {"command": "experiment_c_k_tuning", "configuration_hash": config_hash(config), "execute_required": True, "k_values": config.get("k_values", [20, 40, 80]), "selection": "deterministic stratified query manifest; no first-N sampling", "budget_semantics": "K is per channel/source, not a total candidate cap", "network": "disabled"}


def execute(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    s1, s2, s3 = load_source(paths["s1"]), load_source(paths["s2"]), load_source(paths["s3"])
    gt = load_ground_truth(paths["ground_truth"])
    if config.get("query_manifest"):
        manifest = json.loads(Path(config["query_manifest"]).read_text(encoding="utf-8"))
        selected = set(map(str, manifest["query_ids"][config.get("query_subset", "threshold")]))
        s1 = s1[s1["entity_id"].astype(str).isin(selected)].copy()
        gt = {key: value for key, value in gt.items() if key in selected}
        selection_report = {"selection": "explicit query manifest", "queries": len(selected)}
    elif config.get("folds"):
        folds = load_folds_tsv(config["folds"])
        ids, selection_report = select_query_ids(s1, gt, fold_map=folds, fold=int(config.get("fold", 0)), requested=int(config.get("query_count", len(s1))), seed=int(config.get("seed", 42)))
        selected = set(ids)
        s1 = s1[s1["entity_id"].astype(str).isin(selected)].copy()
        gt = {key: value for key, value in gt.items() if key in selected}
    else:
        selection_report = {"selection": "explicit input query set"}
    k_values = [int(value) for value in config.get("k_values", [20, 40, 80])]
    if config.get("source_index_location"):
        root = Path(config["source_index_location"])
        channels = {}
        for source, frame in (("S2", s2), ("S3", s3)):
            for field, short in (("business_name", "name"), ("business_address", "address")):
                location = root / f"{source}_{field}"
                channels[f"{source}_{short}"] = SourceTfidfIndex.load(location).query(s1, candidate_source=source, top_k=max(k_values))
                channels[f"{source}_exact_{short}"] = ExactMatchIndex.load(location).query(s1, field=field)
        pool = union_candidates(channels)
    else:
        pool = generate_candidates(s1, s2, s3, top_k_name=max(k_values), top_k_address=max(k_values), batch_size=int(config.get("batch_size", 2048)))
    report = {"selection": selection_report, "pool": candidate_diagnostics(pool, s1, s2, s3, gt), "by_k": {}}
    for value in k_values:
        report["by_k"][str(value)] = candidate_diagnostics(slice_candidates(pool, value), s1, s2, s3, gt)
    report["by_total_budget"] = {}
    for value in map(int, config.get("total_budgets", [20, 40, 80])):
        pruned = prune_candidates(pool, value, exact_overflow=str(config.get("exact_overflow", "allow")))
        report["by_total_budget"][str(value)] = candidate_diagnostics(pruned, s1, s2, s3, gt)
    output = Path(config.get("output", "artifacts/c_experiments/k_tuning_report.json")); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output": str(output), "k_values": k_values}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune candidate K using a reusable top-max-K pool.")
    parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int); parser.add_argument("--query-manifest"); parser.add_argument("--k", type=int, nargs="+"); parser.add_argument("--source-index-location"); parser.add_argument("--output")
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for name in ("seed", "query_manifest", "source_index_location", "output"):
        value = getattr(args, name)
        if value is not None: config[name] = value
    if args.k is not None: config["k_values"] = args.k
    if args.dry_run and args.execute: parser.error("--dry-run and --execute are mutually exclusive")
    print(json.dumps(plan(config) if not args.execute else execute(config), indent=2))


if __name__ == "__main__":
    main()
