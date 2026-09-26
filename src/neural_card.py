"""Member A checkpoint card: what D needs to trust and place the neural scores.

Writes ``checkpoint_card.json`` (checkpoint ID, input format, score meaning,
training inputs verified against the checkpoint's own record) and
``training_exposure.parquet``: every entity ID the matcher saw in training,
as a query or as a candidate. D must keep fusion/calibration components
disjoint from these entities; any component containing one is not clean for
neural scores.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .neural_contracts import atomic_write_json, file_identity, git_commit, manifest_shard_paths
from .score_neural import checkpoint_identity

EXPOSURE_COLUMNS = ["entity_id", "role", "candidate_source", "fold", "seen_as_positive"]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pair_paths(manifest_path: str, subdir: str) -> list[str]:
    return manifest_shard_paths(manifest_path, _read_json(manifest_path).get("shard_order", []), subdir)


def training_exposure(pair_paths: Sequence[str]) -> pd.DataFrame:
    """Distinct (entity, role) pairs across the training pair shards, streamed shard by shard."""
    queries: dict[str, tuple[int, bool]] = {}
    candidates: dict[tuple[str, str], tuple[int, bool]] = {}
    for path in pair_paths:
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "source1_fold", "candidate_fold", "label"]
        frame = pd.read_parquet(path, columns=columns)
        for sid, fold, label in zip(frame["source1_entity_id"].astype(str), frame["source1_fold"], frame["label"]):
            prev = queries.get(sid, (int(fold), False)); queries[sid] = (prev[0], prev[1] or bool(label))
        for cid, src, fold, label in zip(frame["candidate_entity_id"].astype(str), frame["candidate_source"].astype(str), frame["candidate_fold"], frame["label"]):
            prev = candidates.get((cid, src), (int(fold), False)); candidates[(cid, src)] = (prev[0], prev[1] or bool(label))
    rows = [(sid, "train_query", "S1", fold, pos) for sid, (fold, pos) in queries.items()]
    rows += [(cid, "train_candidate", src, fold, pos) for (cid, src), (fold, pos) in candidates.items()]
    return pd.DataFrame(rows, columns=EXPOSURE_COLUMNS).sort_values(["role", "entity_id"], kind="stable").reset_index(drop=True)


def build_card(checkpoint: str, train_config_path: str, output_dir: str, *, check_pair_manifests: Sequence[str] = ()) -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    adapter_path, checkpoint_id, identity = checkpoint_identity(checkpoint)
    adapter_config = _read_json(adapter_path / "adapter_config.json")
    state_path = Path(checkpoint) / "checkpoint.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    train_config = _read_json(train_config_path)
    training = train_config.get("training", train_config)
    pair_manifest = training.get("pair_manifest") or train_config.get("pair_manifest")
    query_manifest_path = training.get("query_manifest") or train_config.get("query_manifest")
    subdir = str(training.get("pair_subdir", train_config.get("pair_subdir", "train_pairs")))

    # The checkpoint recorded the hashes of what it was trained on; confirm today's files are those.
    recorded = state.get("manifest_identities", {})
    checks = {}
    for key, path in (("pair_manifest", pair_manifest), ("query_manifest", query_manifest_path)):
        current = file_identity(path, hash_content=True).get("sha256") if path and Path(path).is_file() else None
        expected = recorded.get(key, {}).get("sha256")
        checks[key] = {"path": path, "recorded_sha256": expected, "current_sha256": current, "match": bool(expected) and expected == current}

    query_manifest = _read_json(query_manifest_path) if query_manifest_path else {}
    subsets = {name: list(map(str, ids)) for name, ids in query_manifest.get("query_ids", {}).items()}
    metadata = query_manifest.get("query_metadata", {})
    subset_folds = {name: dict(Counter(str(metadata.get(sid, {}).get("fold", "unknown")) for sid in ids)) for name, ids in subsets.items()}

    exposure = training_exposure(_pair_paths(pair_manifest, subdir)) if pair_manifest else pd.DataFrame(columns=EXPOSURE_COLUMNS)
    exposure.to_parquet(output / "training_exposure.parquet", index=False)
    exposed = set(exposure["entity_id"])
    leakage = {name: sum(sid in exposed for sid in ids) for name, ids in subsets.items() if name != "train"}
    candidate_overlap = {}
    for manifest in check_pair_manifests:
        seen = set()
        for path in _pair_paths(manifest, Path(manifest).stem.replace("_manifest", "")):
            seen.update(pd.read_parquet(path, columns=["candidate_entity_id"])["candidate_entity_id"].astype(str))
        candidate_overlap[manifest] = {"distinct_candidates": len(seen), "seen_in_training": len(seen & exposed)}

    serialization = adapter_config.get("serialization", {})
    card = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint),
        "checkpoint_identity_hash": identity["identity_hash"],
        "checkpoint_files": [{k: f[k] for k in ("path", "size", "sha256") if k in f} for f in identity["files"]],
        "base_model": adapter_config.get("provenance", {}),
        "input_format": {
            "adapter": adapter_config.get("adapter", {}).get("adapter_type"),
            "max_length": adapter_config.get("adapter", {}).get("max_length"),
            "segment_a": serialization.get("side_a"), "segment_b": serialization.get("side_b"),
            "field_separator": serialization.get("separator"), "truncation": serialization.get("truncation"),
            "serialization_version": serialization.get("version"),
            "text_source": "source_store.record_text: business_name|name, business_address|address, country; whitespace collapsed, missing -> ''",
        },
        "score_definition": "neural_logit = logit[match] - logit[no_match]; sigmoid(neural_logit) = P(match). Uncalibrated; D calibrates.",
        "inference_precision": "fp16 on compute capability 7.x (T4), bf16 on >= 8.0, fp32 on CPU; non-finite fp16 rows are rescored in fp32",
        "training": {
            "step": state.get("step"), "epoch": state.get("epoch"),
            "config": {k: v for k, v in state.get("config", training).items() if k not in {"run_dir"}},
            "inputs_verified": all(c["match"] for c in checks.values()), "input_checks": checks,
            "train_queries": len(subsets.get("train", [])),
        },
        "query_subsets": {name: {"queries": len(ids), "folds": subset_folds[name]} for name, ids in subsets.items()},
        "exposure": {
            "file": "training_exposure.parquet", "entities": int(len(exposure)),
            "by_role": exposure["role"].value_counts().to_dict(),
            "by_fold": {str(k): int(v) for k, v in exposure["fold"].value_counts().to_dict().items()},
            "heldout_queries_seen_in_training": leakage,
            "heldout_candidates_seen_in_training": candidate_overlap,
            "rule_for_D": "a component is clean for neural fusion/calibration only if none of its entities appears in training_exposure.parquet",
        },
        "never_tune_on": ["final"],
        "git_commit": git_commit(),
    }
    atomic_write_json(output / "checkpoint_card.json", card)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the neural checkpoint card and training-exposure table for D.")
    parser.add_argument("--checkpoint", required=True, help="training run dir (with adapter/) or adapter dir")
    parser.add_argument("--train-config", required=True, help="the config the checkpoint was trained with")
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-pairs", nargs="*", default=[], help="pair manifests whose candidates to cross-check against training exposure")
    args = parser.parse_args()
    card = build_card(args.checkpoint, args.train_config, args.output, check_pair_manifests=args.check_pairs)
    summary = {k: card[k] for k in ("checkpoint_id", "score_definition")}
    summary.update(inputs_verified=card["training"]["inputs_verified"], exposure=card["exposure"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
