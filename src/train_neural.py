"""Shared neural trainer CLI; adapter/model loading is explicit runtime work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .neural_adapters import AdapterConfig, get_adapter_class
from .neural_contracts import atomic_write_text, file_identity
from .neural_models import NeuralTrainingConfig, load_training_state, train_neural


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    return config


def plan(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", config)
    return {"command": "train_neural", "adapter_type": training.get("adapter_type", config.get("adapter_type")), "execute_required": True, "model_initialization": "deferred", "dataset_iteration": "deferred", "precision": training.get("precision", "fp32"), "max_steps": training.get("max_steps")}


def _pair_inputs(config: dict[str, Any], training: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    manifest_path = training.get("pair_manifest") or config.get("pair_manifest")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        base = Path(manifest_path).parent / str(training.get("pair_subdir", config.get("pair_subdir", "train_pairs")))
        paths = [str(base / name) for name in manifest.get("shard_order", [])]
        identities = {"pair_manifest": file_identity(manifest_path, hash_content=True)}
    else:
        raw = training.get("pairs") or config.get("pairs")
        if isinstance(raw, list):
            paths = list(map(str, raw))
        elif raw and Path(raw).is_dir():
            paths = [str(path) for path in sorted(Path(raw).glob("*.parquet"))]
        elif raw:
            paths = [str(raw)]
        else:
            paths = []
        identities = {f"pair_{index:06d}": file_identity(path, hash_content=True) for index, path in enumerate(paths)}
    if not paths or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("configured training pair shards are missing")
    query_manifest = training.get("query_manifest") or config.get("query_manifest")
    if query_manifest:
        identities["query_manifest"] = file_identity(query_manifest, hash_content=True)
    return paths, identities


def execute(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", config)
    adapter_type = str(training.get("adapter_type", config.get("adapter_type", "")))
    if adapter_type.lower() in {"fake", "smoke"} and not bool(config.get("smoke", False)):
        raise RuntimeError("fake adapter requires explicit smoke=true and must not create production predictions")
    pairs, manifest_identities = _pair_inputs(config, training)
    manifest_identities["query_manifest_id"] = str(manifest_identities.get("query_manifest", {}).get("sha256", ""))
    adapter_cls = get_adapter_class(adapter_type)
    adapter_cfg = AdapterConfig(adapter_type=adapter_type, checkpoint=str(training.get("checkpoint", config.get("checkpoint", ""))), revision=training.get("revision", config.get("revision")), max_length=int(training.get("max_length", config.get("max_length", 256))), device=str(training.get("device", "cpu")))
    run_cfg = NeuralTrainingConfig(**{key: value for key, value in training.items() if key in NeuralTrainingConfig.__dataclass_fields__})
    resume_dir = training.get("resume") or config.get("resume")
    resume_state = None
    if resume_dir:
        adapter = adapter_cls.load_pretrained(str(Path(resume_dir) / "adapter"), map_location=run_cfg.device)
        resume_state = load_training_state(resume_dir, expected_config=run_cfg, expected_manifest_identities=manifest_identities, require_resumable=True)
    else:
        run_dir = Path(run_cfg.run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty training run directory: {run_dir}")
        adapter = adapter_cls.from_config(adapter_cfg, execute=True)
    result = train_neural(adapter, pairs, run_cfg, query_manifest_id=manifest_identities["query_manifest_id"], checkpoint_manifest=manifest_identities, resume_state=resume_state, save_final_checkpoint=True)
    run_dir = Path(run_cfg.run_dir)
    if bool(config.get("smoke", False)):
        atomic_write_text(run_dir / "SMOKE_ONLY", "This checkpoint is for offline smoke testing; never use it as production output.\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shared pair-classification loop with a configured adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if not args.execute:
        print(json.dumps(plan(config), indent=2))
        return
    print(json.dumps(execute(config), indent=2))


if __name__ == "__main__":
    main()
