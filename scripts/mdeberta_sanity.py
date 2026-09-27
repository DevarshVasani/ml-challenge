"""Phase-2 sanity checks for the mDeBERTa adapter, built on the shared trainer.

Checks, in order: pretrained load, token-length/truncation profile, tiny
overfit (loss must fall), checkpoint save -> reload -> identical scores, and
inference throughput / peak VRAM at a few batch sizes. Writes a JSON report.

    python scripts/mdeberta_sanity.py --synthetic --device cuda
    python scripts/mdeberta_sanity.py --pairs artifacts/neural-data/train_pairs --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neural_adapters import AdapterConfig  # noqa: E402
from src.neural_adapters.mdeberta import DEFAULT_CHECKPOINT, MDebertaPairAdapter  # noqa: E402
from src.neural_models import NeuralTrainingConfig, predict_pairs, save_checkpoint, train_neural  # noqa: E402

SYNTHETIC = [  # (S1 name, address, country, candidate name, address, country, label)
    ("Café Bleu", "12 Rue de Rivoli, Paris", "France", "Cafe Bleu", "12 rue Rivoli Paris", "France", 1),
    ("Café Bleu", "12 Rue de Rivoli, Paris", "France", "Café Bleu", "21 Rue de Rivoli, Paris", "France", 0),
    ("Sharma Traders", "45 MG Road, Bengaluru", "India", "Sharma Trading Co", "45 M.G. Road Bangalore", "India", 1),
    ("Sharma Traders", "45 MG Road, Bengaluru", "India", "Verma Traders", "45 MG Road, Bengaluru", "India", 0),
    ("Joe's Pizza", "100 Main Street, Springfield", "United States", "Joes Pizza", "100 Main St Springfield", "United States", 1),
    ("Joe's Pizza", "100 Main Street, Springfield", "United States", "Joe's Pizza", "100 Main Street, Shelbyville", "United States", 0),
    ("Bäckerei Müller", "Hauptstraße 7, Berlin", "Germany", "Baeckerei Mueller", "Hauptstr. 7 Berlin", "Germany", 1),
    ("Bäckerei Müller", "Hauptstraße 7, Berlin", "Germany", "Metzgerei Müller", "Hauptstraße 7, Berlin", "Germany", 0),
    ("श्री गणेश स्वीट्स", "", "India", "Shree Ganesh Sweets", "Station Road, Pune", "India", 1),
    ("Shree Ganesh Sweets", "Station Road, Pune", "India", "Shree Ganesh Hardware", "Station Road, Pune", "India", 0),
    ("Blue Bottle Coffee", "", "United States", "Blue Bottle Coffee Co.", "66 Mint St, San Francisco", "United States", 1),
    ("Blue Bottle Coffee", "", "United States", "Blue Door Coffee", "66 Mint St, San Francisco", "United States", 0),
    ("", "Plot 14, Sector 18, Noida", "India", "Kumar Electronics", "Plot 14 Sector-18 Noida", "India", 1),
    ("", "Plot 14, Sector 18, Noida", "India", "Kumar Electronics", "Plot 41, Sector 81, Noida", "India", 0),
    ("Le Petit Zinc", "5 Place du Marché", "France", "Le Petit Zinc", "5 pl. du Marche", "France", 1),
    ("Le Petit Zinc", "5 Place du Marché", "France", "Le Grand Zinc", "5 Place du Marché", "France", 0),
]


def synthetic_pairs() -> pd.DataFrame:
    rows = []
    for index, (na, aa, ca, nb, ab, cb, label) in enumerate(SYNTHETIC):
        rows.append({"source1_entity_id": f"S1-syn-{index // 2}", "candidate_entity_id": f"S2-syn-{index}", "candidate_source": "S2",
                     "source1_fold": 1, "candidate_fold": 1, "name_a": na, "address_a": aa, "country_a": ca,
                     "name_b": nb, "address_b": ab, "country_b": cb, "label": label})
    return pd.DataFrame(rows)


def load_pairs(path: str, limit: int, seed: int) -> pd.DataFrame:
    source = Path(path)
    shards = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
    if not shards:
        raise FileNotFoundError(f"no parquet pair shards under {source}")
    frame = pd.read_parquet(shards[0])
    return frame.sample(n=min(limit, len(frame)), random_state=seed).reset_index(drop=True)


def length_profile(adapter: MDebertaPairAdapter, frame: pd.DataFrame) -> dict:
    lengths = np.array(adapter.pair_lengths(frame.to_dict(orient="records")))
    return {
        "pairs": int(len(lengths)),
        "percentiles": {f"p{q}": float(np.percentile(lengths, q)) for q in (50, 90, 95, 99)},
        "max": int(lengths.max()),
        "truncation_rate_at": {str(limit): float((lengths > limit).mean()) for limit in (128, 192, 256, 384, 512)},
        "configured_max_length": adapter.config.max_length,
    }


def balanced(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    positives = frame[frame["label"].astype(int) == 1]
    negatives = frame[frame["label"].astype(int) == 0]
    half = size // 2
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("overfit sample needs both positive and negative pairs")
    picked = pd.concat([positives.sample(n=min(half, len(positives)), random_state=seed), negatives.sample(n=min(size - half, len(negatives)), random_state=seed)])
    return picked.reset_index(drop=True)


def throughput(adapter: MDebertaPairAdapter, frame: pd.DataFrame, batch_sizes: list[int], device: str) -> dict:
    import torch

    unlabeled = frame.drop(columns=["label"], errors="ignore")
    results = {}
    for batch_size in batch_sizes:
        if device.startswith("cuda"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        predict_pairs(adapter, unlabeled.head(batch_size * 2), batch_size=batch_size, device=device)  # warm-up
        _, metrics = predict_pairs(adapter, unlabeled, batch_size=batch_size, device=device, return_metrics=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        results[str(batch_size)] = {"pairs_per_second": metrics["pairs_per_second"], "collate_seconds": metrics["collate_seconds"], "forward_seconds": metrics["forward_seconds"],
                                    "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.startswith("cuda") else None}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", help="labelled pair shard (.parquet) or a directory of shards, e.g. artifacts/neural-data/train_pairs")
    source.add_argument("--synthetic", action="store_true", help="use 16 built-in pairs; for checks before Member A's data exists")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default=None, help="training precision; default bf16 on A10G/L4/A100, fp16 on T4, fp32 on CPU")
    parser.add_argument("--sample", type=int, default=5000, help="pairs used for length profile and throughput")
    parser.add_argument("--overfit-rows", type=int, default=16)
    parser.add_argument("--overfit-epochs", type=int, default=30)
    parser.add_argument("--overfit-lr", type=float, default=5e-5)
    parser.add_argument("--batch-sizes", default="32,64,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/mdeberta-sanity")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty {output}; pass a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)

    tick = time.perf_counter()
    adapter = MDebertaPairAdapter(AdapterConfig("mdeberta", checkpoint=args.checkpoint, max_length=args.max_length, device=args.device))
    precision = args.precision or adapter.inference_precision
    report: dict = {"checkpoint": args.checkpoint, "device": args.device, "precision": precision, "inference_precision": adapter.inference_precision, "max_length": args.max_length, "checks": {}}
    report["load_seconds"] = time.perf_counter() - tick
    report["provenance"] = adapter.provenance
    report["parameters_millions"] = sum(p.numel() for p in adapter.parameters()) / 1e6
    report["checks"]["pretrained_loads"] = True

    frame = synthetic_pairs() if args.synthetic else load_pairs(args.pairs, args.sample, args.seed)
    report["example_serialization"] = adapter.serialize_pair(frame.iloc[0].to_dict())
    report["length_profile"] = length_profile(adapter, frame)

    tiny = frame if args.synthetic else balanced(frame, args.overfit_rows, args.seed)
    run_cfg = NeuralTrainingConfig(adapter_type="mdeberta", microbatch=8, gradient_accumulation=1, learning_rate=args.overfit_lr, epochs=args.overfit_epochs,
                                   warmup_fraction=0.0, log_every=1, precision=precision, device=args.device, seed=args.seed, run_dir=str(output / "overfit"))
    train = train_neural(adapter, tiny, run_cfg)
    losses = [entry["loss"] for entry in train["log_history"]]
    head, tail = float(np.mean(losses[:3])), float(np.mean(losses[-3:]))
    report["overfit"] = {"rows": len(tiny), "steps": train["steps"], "first_losses": losses[:5], "last_losses": losses[-5:], "peak_vram_gb": (train["peak_gpu_memory_bytes"] or 0) / 1e9, "train_pairs_per_second": train["pairs_per_second"]}
    report["checks"]["loss_decreases"] = tail < 0.5 * head

    unlabeled = tiny.drop(columns="label")
    before = predict_pairs(adapter, unlabeled, batch_size=16, device=args.device)
    save_checkpoint(adapter, output / "overfit", run_cfg, step=train["steps"])
    reloaded = MDebertaPairAdapter.load_pretrained(str(output / "overfit" / "adapter"), map_location=args.device)
    after = predict_pairs(reloaded, unlabeled, batch_size=16, device=args.device)
    max_diff = float((before["score"] - after["score"]).abs().max())
    accuracy = float(((after["score"].to_numpy() >= 0.5).astype(int) == tiny["label"].astype(int).to_numpy()).mean())
    report["reload"] = {"max_score_difference": max_diff, "tiny_set_accuracy_after_reload": accuracy}
    report["checks"]["reload_consistent"] = max_diff < 1e-3
    report["checks"]["tiny_set_memorized"] = accuracy >= 0.9
    del adapter

    report["inference"] = throughput(reloaded, frame, [int(value) for value in args.batch_sizes.split(",")], args.device)
    report["all_checks_passed"] = all(report["checks"].values())
    (output / "sanity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_checks_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
