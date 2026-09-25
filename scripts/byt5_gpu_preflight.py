"""Preflight checks before spending GPU time on the ByT5 baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _pair_manifest_check(manifest_path: Path, pair_subdir: str) -> dict:
    result = {"path": str(manifest_path), "exists": manifest_path.is_file(), "ok": False}
    if not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest.get("shard_order", []))
    base = manifest_path.parent / pair_subdir
    missing = [str(base / name) for name in names if not (base / name).is_file()]
    result.update({
        "shards": len(names),
        "pair_dir": str(base),
        "missing_shards": missing[:20],
        "ok": bool(names) and not missing,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CUDA, BF16, shared artifacts, and optionally ByT5 forward pass.")
    parser.add_argument("--config", default="configs/neural/byt5_l40s_short_pilot.json")
    parser.add_argument("--load-model", action="store_true", help="Download/load ByT5 and execute a small BF16 CUDA forward pass.")
    parser.add_argument("--allow-training-only", action="store_true", help="Do not fail if threshold/final artifacts are not ready yet.")
    parser.add_argument("--output", default="artifacts/byt5-preflight.json")
    args = parser.parse_args()

    import torch
    import transformers

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    training = config.get("training", config)
    result = {
        "config": args.config,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "device_count": torch.cuda.device_count(),
        "model_check_requested": args.load_model,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_vram_gb": props.total_memory / (1024 ** 3),
            "capability": list(torch.cuda.get_device_capability(0)),
        }

    disk = shutil.disk_usage(Path.cwd())
    result["disk_free_gb"] = disk.free / (1024 ** 3)

    artifacts = {
        "train": _pair_manifest_check(Path("artifacts/neural-data/train_pairs_manifest.json"), "train_pairs"),
        "threshold": _pair_manifest_check(Path("artifacts/neural-data/threshold_pairs_manifest.json"), "threshold_pairs"),
        "final": _pair_manifest_check(Path("artifacts/neural-data/final_pairs_manifest.json"), "final_pairs"),
        "query_manifest": {"path": "artifacts/neural-data/query_manifest.json", "ok": Path("artifacts/neural-data/query_manifest.json").is_file()},
        "candidate_manifest": {"path": "artifacts/neural-candidates/manifest.json", "ok": Path("artifacts/neural-candidates/manifest.json").is_file()},
    }
    result["artifacts"] = artifacts

    hard_failures = []
    if not result["cuda_available"]:
        hard_failures.append("CUDA is not available")
    if str(training.get("precision", "")).lower() == "bf16" and not result["bf16_supported"]:
        hard_failures.append("BF16 was requested but this GPU/driver does not report BF16 support")
    if not artifacts["train"]["ok"]:
        hard_failures.append("training pair manifest/shards are incomplete")
    if not artifacts["query_manifest"]["ok"]:
        hard_failures.append("query manifest is missing")
    if not args.allow_training_only:
        for key in ("threshold", "final", "candidate_manifest"):
            if not artifacts[key]["ok"]:
                hard_failures.append(f"required shared artifact is incomplete: {key}")

    if args.load_model and not hard_failures:
        from src.neural_adapters import AdapterConfig, get_adapter_class
        from src.neural_contracts import seed_everything

        adapter_type = str(training.get("adapter_type", config.get("adapter_type", "byt5")))
        adapter_config = AdapterConfig(
            adapter_type=adapter_type,
            checkpoint=str(training.get("checkpoint", config.get("checkpoint", "google/byt5-small"))),
            revision=training.get("revision", config.get("revision")),
            max_length=int(training.get("max_length", config.get("max_length", 512))),
            device=str(training.get("device", "cuda")),
        )