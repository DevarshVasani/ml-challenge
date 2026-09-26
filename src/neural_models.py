"""Optional-Torch shared training, thresholding, checkpoint and scoring code."""

from __future__ import annotations

import contextlib
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .evaluate import evaluate_predictions
from .neural_contracts import PREDICTION_COLUMNS, atomic_write_json, seed_everything, validate_pair_frame, validate_prediction_frame


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed; install the future GPU extra on the compute machine") from exc
    return torch


@dataclass
class NeuralTrainingConfig:
    adapter_type: str
    microbatch: int = 8
    gradient_accumulation: int = 8
    learning_rate: float = 2e-5
    epochs: int = 1
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    max_steps: int | None = None
    log_every: int = 25
    checkpoint_every: int = 0
    seed: int = 42
    device: str = "cpu"
    run_dir: str = "artifacts/neural-run"
    num_workers: int = 0


def positive_probability(logits: Any):
    torch = require_torch()
    if getattr(logits, "ndim", None) != 2 or logits.shape[1] != 2:
        raise ValueError(f"adapter logits must have shape [batch, 2], got {tuple(logits.shape)}")
    if not logits.dtype.is_floating_point:
        raise ValueError("adapter logits must be floating point")
    return torch.softmax(logits.float(), dim=1)[:, 1]


def _read_pairs(path: str | os.PathLike[str]) -> pd.DataFrame:
    path = str(path)
    frame = pd.read_parquet(path) if Path(path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "label" in frame:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    return frame


def _source_frames(pairs: pd.DataFrame | Sequence[str | os.PathLike[str]]) -> tuple[list[pd.DataFrame] | list[str], list[int]]:
    if isinstance(pairs, pd.DataFrame):
        validate_pair_frame(pairs, labeled=True)
        return [pairs], [len(pairs)]
    paths = [str(path) for path in pairs]
    if not paths:
        raise ValueError("training requires at least one pair shard")
    counts: list[int] = []
    for path in paths:
        if Path(path).suffix.lower() in {".parquet", ".pq"}:
            try:
                import pyarrow.parquet as pq
                counts.append(int(pq.ParquetFile(path).metadata.num_rows))
                continue
            except ImportError:
                pass
        counts.append(len(_read_pairs(path)))
    return paths, counts


def _epoch_batches(sources: list[pd.DataFrame] | list[str], config: NeuralTrainingConfig, epoch: int):
    rng = np.random.default_rng(config.seed + epoch)
    shard_order = rng.permutation(len(sources)).tolist()
    for shard_position in shard_order:
        source = sources[shard_position]
        frame = source if isinstance(source, pd.DataFrame) else _read_pairs(source)
        validate_pair_frame(frame, labeled=True)
        row_rng = np.random.default_rng(config.seed + epoch * 1_000_003 + shard_position)
        order = row_rng.permutation(len(frame)).tolist()
        for start in range(0, len(order), config.microbatch):
            indices = order[start : start + config.microbatch]
            yield frame.iloc[indices].copy()


def _precision_tools(config: NeuralTrainingConfig):
    torch = require_torch()
    precision = config.precision.lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if precision == "fp32":
        return lambda: contextlib.nullcontext(), None
    if not config.device.startswith("cuda"):
        raise RuntimeError("mixed precision is supported only on an explicitly selected CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was explicitly requested but this GPU does not report BF16 support")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")
    return lambda: torch.autocast(device_type="cuda", dtype=dtype), scaler


def _optimizer_schedule(adapter, config: NeuralTrainingConfig, total_steps: int):
    torch = require_torch()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup = int(total_steps * config.warmup_fraction)
    def scale(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        remaining = max(1, total_steps - warmup)
        return max(0.0, 1.0 - max(0, step - warmup) / remaining)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_neural(
    adapter,
    pairs: pd.DataFrame | Sequence[str | os.PathLike[str]],
    config: NeuralTrainingConfig,
    *,
    query_manifest_id: str = "",
    checkpoint_manifest: Mapping[str, Any] | None = None,
    resume_state: Mapping[str, Any] | None = None,
    save_final_checkpoint: bool = False,
) -> dict[str, Any]:
    """Train with deterministic single-worker shard/row order and exact resume."""
    torch = require_torch()
    if config.microbatch <= 0 or config.gradient_accumulation <= 0 or config.epochs <= 0:
        raise ValueError("microbatch, gradient_accumulation, and epochs must be positive")
    if config.num_workers != 0:
        raise ValueError("exact resume currently requires num_workers=0")
    sources, counts = _source_frames(pairs)
    total_rows = sum(counts)
    if total_rows == 0:
        raise ValueError("cannot train on empty pair shards")
    micros_per_epoch = sum(math.ceil(count / config.microbatch) for count in counts)
    planned_steps = config.epochs * math.ceil(micros_per_epoch / config.gradient_accumulation)
    if config.max_steps is not None:
        planned_steps = min(planned_steps, config.max_steps)
    seed_everything(config.seed)
    adapter.to_device(config.device)
    adapter.set_gradient_checkpointing(config.gradient_checkpointing)
    optimizer, scheduler = _optimizer_schedule(adapter, config, max(1, planned_steps))
    autocast, scaler = _precision_tools(config)
    if resume_state is not None:
        if resume_state.get("config") != asdict(config):
            raise ValueError("resume configuration differs; refusing to duplicate or skip training samples")
        if "optimizer" not in resume_state or "scheduler" not in resume_state:
            raise ValueError("checkpoint is not resumable because optimizer/scheduler state is missing")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if scaler is not None and resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        _restore_rng_state(resume_state)
    loss_fn = torch.nn.CrossEntropyLoss()
    adapter.model.train() if hasattr(adapter, "model") and hasattr(adapter.model, "train") else None
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config.device)
    step = int(resume_state.get("step", 0)) if resume_state else 0
    resume_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    resume_cursor = int(resume_state.get("data_cursor", 0)) if resume_state else 0
    if resume_cursor < 0 or resume_cursor > micros_per_epoch:
        raise ValueError("resume data cursor is outside the deterministic epoch order")
    losses: list[float] = []
    log_history: list[dict[str, Any]] = []
    processed_pairs = 0
    truncated = 0
    collate_seconds = 0.0
    forward_seconds = 0.0
    next_epoch, next_cursor = resume_epoch, resume_cursor
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    stop = step >= planned_steps
    for epoch in range(resume_epoch, config.epochs):
        cursor_start = resume_cursor if epoch == resume_epoch else 0
        micro_in_window = 0
        window_size = 0
        for cursor, batch_frame in enumerate(_epoch_batches(sources, config, epoch)):
            if cursor < cursor_start:
                continue
            if stop:
                break
            if micro_in_window == 0:
                window_size = min(config.gradient_accumulation, micros_per_epoch - cursor)
            rows = batch_frame.to_dict(orient="records")
            tick = time.perf_counter()
            batch = adapter.collate(rows)
            collate_seconds += time.perf_counter() - tick
            labels = torch.tensor(batch_frame["label"].astype(int).to_numpy(), dtype=torch.long, device=config.device)
            tick = time.perf_counter()
            with autocast():
                logits = adapter.logits(batch)
                loss = loss_fn(logits, labels)
            forward_seconds += time.perf_counter() - tick
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite neural loss at epoch={epoch}, step={step}")
            scaled_loss = loss / window_size
            scaler.scale(scaled_loss).backward() if scaler is not None else scaled_loss.backward()
            processed_pairs += len(batch_frame)
            truncated += int(batch.get("truncated", 0)) if isinstance(batch, Mapping) else 0
            micro_in_window += 1
            next_epoch, next_cursor = epoch, cursor + 1
            if micro_in_window == window_size:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), config.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1; micro_in_window = 0
                losses.append(float(loss.detach().cpu()))
                if config.log_every and step % config.log_every == 0:
                    log_history.append({"step": step, "loss": losses[-1], "pairs_processed": processed_pairs})
                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    save_checkpoint(adapter, config.run_dir, config, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=next_epoch, data_cursor=next_cursor, step=step, manifest_identities={"query_manifest_id": query_manifest_id, **dict(checkpoint_manifest or {})})
                if step >= planned_steps:
                    stop = True
        if stop:
            break
        next_epoch, next_cursor = epoch + 1, 0
    elapsed = time.perf_counter() - started
    identities = {"query_manifest_id": query_manifest_id, **dict(checkpoint_manifest or {})}
    if save_final_checkpoint:
        save_checkpoint(adapter, config.run_dir, config, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=next_epoch, data_cursor=next_cursor, step=step, manifest_identities=identities)
    peak_memory = int(torch.cuda.max_memory_allocated(config.device)) if config.device.startswith("cuda") else None
    return {"steps": step, "resume_epoch": next_epoch, "data_cursor": next_cursor, "mean_loss": float(np.mean(losses)) if losses else None, "last_loss": losses[-1] if losses else None, "seconds": elapsed, "pairs_processed": processed_pairs, "pairs_per_second": processed_pairs / elapsed if elapsed else 0.0, "collate_seconds": collate_seconds, "forward_loss_seconds": forward_seconds, "truncated_pairs": truncated, "truncation_rate": truncated / processed_pairs if processed_pairs else 0.0, "peak_gpu_memory_bytes": peak_memory, "query_manifest_id": query_manifest_id, "single_worker_exact_resume": True, "log_history": log_history}


def predict_pairs(adapter, pairs: pd.DataFrame, *, batch_size: int = 32, device: str = "cpu", return_metrics: bool = False):
    torch = require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_pair_frame(pairs, labeled=False)
    adapter.to_device(device)
    if pairs.empty:
        result = pd.DataFrame(columns=PREDICTION_COLUMNS)
        return (result, {"pairs": 0, "seconds": 0.0, "pairs_per_second": 0.0, "collate_seconds": 0.0, "forward_seconds": 0.0, "truncated_pairs": 0}) if return_metrics else result
    adapter.model.eval() if hasattr(adapter, "model") and hasattr(adapter.model, "eval") else None
    rows: list[dict[str, Any]] = []
    collate_seconds = 0.0; forward_seconds = 0.0; truncated = 0
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_frame = pairs.iloc[start : start + batch_size]
            tick = time.perf_counter(); batch = adapter.collate(batch_frame.to_dict(orient="records")); collate_seconds += time.perf_counter() - tick
            truncated += int(batch.get("truncated", 0)) if isinstance(batch, Mapping) else 0
            tick = time.perf_counter(); scores = positive_probability(adapter.logits(batch)).detach().cpu().numpy(); forward_seconds += time.perf_counter() - tick
            for row, score in zip(batch_frame.itertuples(index=False), scores):
                rows.append({"source1_entity_id": str(row.source1_entity_id), "candidate_entity_id": str(row.candidate_entity_id), "candidate_source": str(row.candidate_source), "score": float(score)})
    result = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    validate_prediction_frame(result)
    elapsed = time.perf_counter() - started
    metrics = {"pairs": len(result), "seconds": elapsed, "pairs_per_second": len(result) / elapsed if elapsed else 0.0, "collate_seconds": collate_seconds, "forward_seconds": forward_seconds, "truncated_pairs": truncated, "truncation_rate": truncated / len(result) if len(result) else 0.0}
    return (result, metrics) if return_metrics else result


def predictions_to_mapping(scores: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    validate_prediction_frame(scores)
    selected = scores[scores["score"].astype(float) >= float(threshold)]
    result: dict[str, list[str]] = {}
    for sid, group in selected.groupby("source1_entity_id", sort=False):
        result[str(sid)] = list(dict.fromkeys(group.sort_values(["score", "candidate_entity_id"], ascending=[False, True])["candidate_entity_id"].astype(str)))
    return result


def _selected_scores(scores: pd.DataFrame, ids: Sequence[str], candidates: pd.DataFrame | None) -> pd.DataFrame:
    selected = scores[scores["source1_entity_id"].astype(str).isin(set(ids))].copy()
    if candidates is None:
        validate_prediction_frame(selected)
    else:
        expected = set(map(tuple, candidates[["source1_entity_id", "candidate_entity_id", "candidate_source"]].astype(str).itertuples(index=False, name=None)))
        validate_prediction_frame(selected, expected)
    return selected


def tune_threshold(scores: pd.DataFrame, ground_truth: Mapping[str, Iterable[str]], query_ids: Iterable[str], candidates: pd.DataFrame | None = None, thresholds: Sequence[float] | None = None) -> dict[str, Any]:
    ids = list(map(str, query_ids)); id_set = set(ids)
    gt = {str(k): list(map(str, v)) for k, v in ground_truth.items() if str(k) in id_set}
    if id_set - set(gt):
        raise ValueError("ground truth is missing selected threshold-tuning queries")
    if candidates is not None and set(candidates["source1_entity_id"].astype(str)) - id_set:
        raise ValueError("threshold candidate frame contains queries outside the tuning subset")
    selected = _selected_scores(scores, ids, candidates)
    grouped = {str(sid): list(zip(group["candidate_entity_id"].astype(str), group["score"].astype(float))) for sid, group in selected.groupby("source1_entity_id", sort=False)}
    values = [float(value) for value in (thresholds or np.linspace(0.05, 0.95, 19))]
    if not values or any(not 0 <= value <= 1 for value in values):
        raise ValueError("threshold search must contain probabilities in [0, 1]")
    best = None
    for threshold in values:
        prediction = {sid: [candidate for candidate, score in rows if score >= threshold] for sid, rows in grouped.items()}
        metrics = evaluate_predictions(gt, prediction)
        candidate = (float(metrics["macro_f05"]), -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold, metrics)
    assert best is not None
    return {"threshold": best[1], "metrics": best[2], "query_count": len(ids), "search_count": len(values)}


def evaluate_scores(scores: pd.DataFrame, ground_truth: Mapping[str, Iterable[str]], query_ids: Iterable[str], threshold: float, *, candidate_pairs: pd.DataFrame | None = None, query_metadata: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    ids = list(map(str, query_ids))
    if set(ids) - set(map(str, ground_truth)):
        raise ValueError("ground truth does not cover every selected query, including zero-candidate queries")
    selected_scores = _selected_scores(scores, ids, candidate_pairs)
    selected_gt = {sid: list(map(str, ground_truth[sid])) for sid in ids}
    prediction = predictions_to_mapping(selected_scores, threshold)
    metrics = evaluate_predictions(selected_gt, prediction)
    metrics["threshold"] = float(threshold)
    metrics["candidate_pairs_scored"] = int(len(selected_scores))
    available: dict[str, set[str]] = {}
    for row in selected_scores.itertuples(index=False):
        available.setdefault(str(row.source1_entity_id), set()).add(str(row.candidate_entity_id))
    oracle = {sid: [candidate for candidate in selected_gt[sid] if candidate in available.get(sid, set())] for sid in ids}
    metrics["candidate_oracle_ceiling"] = float(evaluate_predictions(selected_gt, oracle)["macro_f05"])
    counts = np.array([len(available.get(sid, set())) for sid in ids], dtype=float)
    metrics["candidate_counts"] = {"mean": float(counts.mean()) if len(counts) else 0.0, "p95": float(np.percentile(counts, 95)) if len(counts) else 0.0, "max": int(counts.max()) if len(counts) else 0}
    if query_metadata is not None:
        for field, output_key in (("country", "by_country"), ("match_count", "by_match_count"), ("singleton", "by_singleton")):
            groups: dict[str, list[str]] = {}
            for sid in ids:
                groups.setdefault(str(query_metadata.get(sid, {}).get(field, "UNKNOWN")), []).append(sid)
            metrics[output_key] = {}
            for value, group_ids in sorted(groups.items()):
                detail = evaluate_predictions({sid: selected_gt[sid] for sid in group_ids}, prediction)
                metrics[output_key][value] = {key: val for key, val in detail.items() if key != "per_entity_scores"}
    return metrics


def _capture_rng_state(torch) -> dict[str, Any]:
    state = {"rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    torch = require_torch()
    if "rng_state" in state: torch.set_rng_state(state["rng_state"])
    if "python_rng_state" in state: random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state: np.random.set_state(state["numpy_rng_state"])
    if "cuda_rng_state_all" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def save_checkpoint(adapter, output_dir: str | os.PathLike[str], config: NeuralTrainingConfig, *, optimizer=None, scheduler=None, scaler=None, epoch: int = 0, data_cursor: int = 0, step: int = 0, manifest_identities: Mapping[str, Any] | None = None) -> None:
    torch = require_torch()
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    adapter.save_pretrained(str(path / "adapter"))
    state: dict[str, Any] = {"epoch": epoch, "data_cursor": data_cursor, "step": step, "config": asdict(config), "manifest_identities": dict(manifest_identities or {}), **_capture_rng_state(torch)}
    if optimizer is not None: state["optimizer"] = optimizer.state_dict()
    if scheduler is not None: state["scheduler"] = scheduler.state_dict()
    if scaler is not None: state["scaler"] = scaler.state_dict()
    temporary = path / ".training_state.pt.tmp"
    torch.save(state, temporary); os.replace(temporary, path / "training_state.pt")
    atomic_write_json(path / "checkpoint.json", {key: value for key, value in state.items() if key not in {"rng_state", "python_rng_state", "numpy_rng_state", "cuda_rng_state_all", "optimizer", "scheduler", "scaler"}})


def load_training_state(path: str | os.PathLike[str], *, expected_config: NeuralTrainingConfig | None = None, expected_manifest_identities: Mapping[str, Any] | None = None, require_resumable: bool = False) -> dict[str, Any]:
    torch = require_torch()
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    if expected_config is not None and state.get("config") != asdict(expected_config):
        raise ValueError("resume configuration differs; refusing unsupported resume changes")
    if expected_manifest_identities is not None and state.get("manifest_identities", {}) != dict(expected_manifest_identities):
        raise ValueError("resume manifest identities differ; candidate/query inputs changed")
    if require_resumable and ("optimizer" not in state or "scheduler" not in state):
        raise ValueError("checkpoint is not resumable because optimizer/scheduler state is missing")
    return state
