"""Member A production scorer: selected pair shards -> keyed raw neural logits.

``predict_neural`` stays as the pilot tool (probabilities, fixed batches).
This scorer produces what D fuses:

* ``neural_logit`` is the raw match logit ``logit[1] - logit[0]``; its sigmoid
  equals the softmax match probability. D owns calibration and fusion.
* Every selected input row gets an output row with an explicit
  ``neural_status``. A pair that could not be scored has a NaN logit and
  ``neural_scored=False``; nothing is ever filled with zero.
* Each pair is tokenized once, batches are length-sorted under a token budget,
  and the next shard is read/joined/tokenized on a CPU thread while the GPU
  scores the current one.
* Key-only shards (C's route shards) get text from the shared source store
  through ``record_text``, the same mapping used to build training pairs.
* Shards are atomic and resumable, and a rerun picks up newly published
  shards. Rows whose key, text and model/input version match an earlier run
  can be reused instead of rescored (``reuse_from``).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .neural_adapters import get_adapter_class
from .neural_contracts import TEXT_FIELDS, atomic_write_json, config_hash, directory_identity, file_identity, git_commit, manifest_shard_paths, sha256_file

SCHEMA_VERSION = "neural-scores-1"
KEY = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
SCORED, CACHED, MISSING_RECORD, NONFINITE = "scored", "cached", "missing_record", "nonfinite"
OUTPUT_COLUMNS = KEY + ["neural_logit", "neural_scored", "neural_status", "checkpoint_id", "cache_key", "truncated", "token_length"]
OUTPUT_SUFFIX = ".neural.parquet"
MANIFEST = "neural_manifest.json"


# -- inputs and identities --------------------------------------------------

def discover_inputs(config: Mapping[str, Any]) -> tuple[list[Path], list[Path]]:
    """Return (ready, pending) input shards; pending ones lack a completion sidecar."""
    if config.get("pair_manifest"):
        manifest = json.loads(Path(config["pair_manifest"]).read_text(encoding="utf-8"))
        paths = manifest_shard_paths(config["pair_manifest"], manifest.get("shard_order", []), str(config.get("pair_subdir", "final_pairs")))
    else:
        raw = config.get("pairs")
        if isinstance(raw, list):
            paths = list(map(str, raw))
        elif raw and Path(raw).is_dir():
            paths = sorted(str(p) for p in Path(raw).iterdir() if p.suffix.lower() in {".parquet", ".tsv"})
        elif raw and any(ch in str(raw) for ch in "*?["):
            paths = sorted(glob.glob(str(raw)))
        elif raw:
            paths = [str(raw)]
        else:
            raise ValueError("configure pair_manifest or pairs")
    inputs = [Path(p) for p in paths]
    stems = [_output_name(p) for p in inputs]
    if len(set(stems)) != len(stems):
        raise ValueError("input shard file names must be unique; output shards are named after them")
    if bool(config.get("require_complete_sidecar", False)):
        ready = [p for p in inputs if p.is_file() and Path(f"{p}.complete.json").is_file()]
        return ready, [p for p in inputs if p not in ready]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"configured input shards are missing: {missing[:5]}")
    return inputs, []


def _output_name(path: Path) -> str:
    return path.stem + OUTPUT_SUFFIX


def checkpoint_identity(checkpoint: str | os.PathLike[str]) -> tuple[Path, str, dict[str, Any]]:
    """Hash only the inference weights/tokenizer/config, not optimizer state."""
    root = Path(checkpoint)
    adapter_path = root / "adapter" if (root / "adapter").is_dir() else root
    identity = directory_identity(adapter_path, hash_content=True)
    if not identity.get("files"):
        raise FileNotFoundError(f"checkpoint has no files: {adapter_path}")
    return adapter_path, f"ckpt-{identity['identity_hash'][:16]}", identity


def input_signature(adapter) -> dict[str, Any]:
    return adapter.input_signature() if hasattr(adapter, "input_signature") else {"adapter": adapter.config_dict()}


# -- reuse cache ------------------------------------------------------------

class ReuseCache:
    """Sorted uint64 cache keys -> logits (12 bytes/pair) from completed score shards."""

    def __init__(self, files: Sequence[Path] = ()):
        keys, logits = [], []
        for path in files:
            frame = pd.read_parquet(path, columns=["cache_key", "neural_logit", "neural_status"])
            frame = frame[frame["neural_status"].isin([SCORED, CACHED])]
            keys.append(frame["cache_key"].to_numpy(np.uint64)); logits.append(frame["neural_logit"].to_numpy(np.float32))
        key = np.concatenate(keys) if keys else np.zeros(0, np.uint64)
        value = np.concatenate(logits) if logits else np.zeros(0, np.float32)
        order = np.argsort(key, kind="stable")
        self.keys, self.logits = key[order], value[order]

    @classmethod
    def from_dirs(cls, dirs: Sequence[str | os.PathLike[str]]) -> "ReuseCache":
        return cls([p for d in dirs for p in sorted(Path(d).glob(f"*{OUTPUT_SUFFIX}")) if Path(f"{p}.complete.json").is_file()])

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.keys) or not len(keys):
            return np.zeros(len(keys), bool), np.full(len(keys), np.nan, np.float32)
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        hit = self.keys[pos] == keys
        return hit, np.where(hit, self.logits[pos], np.nan).astype(np.float32)


# -- shard preparation (CPU thread) ----------------------------------------

@dataclass
class Prepared:
    name: str
    keys: pd.DataFrame
    status: np.ndarray
    logits: np.ndarray
    cache_key: np.ndarray
    truncated: np.ndarray
    token_length: np.ndarray
    todo: np.ndarray                      # frame positions still needing the model
    rows: list[dict[str, Any]]            # text rows for ``todo``
    encoded: dict[str, Any] | None = None
    input_identity: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Context:
    adapter: Any
    checkpoint_id: str
    cache_prefix: bytes
    key_columns: dict[str, str]
    route_column: str | None
    route_value: str
    require_route: bool
    source_store: str | None
    reuse: ReuseCache


def read_shard(path: Path, ctx: Context) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        available = set(pq.read_schema(path).names)
    else:
        available = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    wanted = [ctx.key_columns.get(k, k) for k in KEY] + [c for c in TEXT_FIELDS if c in available]
    if ctx.route_column and ctx.route_column in available:
        wanted.append(ctx.route_column)
    elif ctx.require_route:
        raise ValueError(f"{path}: route column {ctx.route_column!r} is required but absent")
    missing = [c for c in wanted if c not in available]
    if missing:
        raise ValueError(f"{path}: missing key columns {missing}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=wanted)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, usecols=wanted)


def _select(frame: pd.DataFrame, ctx: Context, where: str) -> pd.DataFrame:
    frame = frame.rename(columns={v: k for k, v in ctx.key_columns.items() if v != k})
    for column in KEY:
        if pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{where}: {column} is numeric; IDs must be strings so leading zeros survive")
    if ctx.route_column and ctx.route_column in frame.columns:
        frame = frame[frame[ctx.route_column].astype(str) == ctx.route_value]
    frame = frame.reset_index(drop=True)
    keys = frame[KEY].astype(str)
    if keys.isin(["", "nan", "None"]).any().any():
        raise ValueError(f"{where}: empty pair key")
    if keys.duplicated().any():
        raise ValueError(f"{where}: duplicate pair keys")
    frame[KEY] = keys
    return frame


def _join_text(frame: pd.DataFrame, store_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach the six text fields from the source store; returns (frame, missing mask)."""
    from .source_store import SourceStore, record_text

    with SourceStore(store_path) as store:
        s1 = store.lookup("S1", frame["source1_entity_id"].unique().tolist())
        candidates = store.lookup_many({str(src): ids.unique().tolist() for src, ids in frame.groupby("candidate_source", sort=False)["candidate_entity_id"]})
    empty_a, empty_b = record_text({}, "a"), record_text({}, "b")
    left = [record_text(s1[sid], "a") if sid in s1 else None for sid in frame["source1_entity_id"]]
    right = [record_text(candidates[(src, cid)], "b") if (src, cid) in candidates else None for src, cid in zip(frame["candidate_source"], frame["candidate_entity_id"])]
    missing = np.array([a is None or b is None for a, b in zip(left, right)], bool)
    text = pd.DataFrame([{**(a or empty_a), **(b or empty_b)} for a, b in zip(left, right)], index=frame.index, columns=TEXT_FIELDS)
    return pd.concat([frame.drop(columns=[c for c in TEXT_FIELDS if c in frame.columns]), text], axis=1), missing


def _cache_keys(adapter, rows: Sequence[Mapping[str, Any]], prefix: bytes) -> np.ndarray:
    base = hashlib.blake2b(prefix, digest_size=8)
    out = np.empty(len(rows), np.uint64)
    for i, row in enumerate(rows):
        h = base.copy()
        h.update("\x1f".join(str(row[k]) for k in KEY).encode()); h.update(b"\x1e"); h.update(adapter.serialize_pair(row).encode())
        out[i] = int.from_bytes(h.digest(), "little")
    return out


def prepare_frame(frame: pd.DataFrame, ctx: Context, name: str, *, stale: ReuseCache | None = None) -> Prepared:
    timings: dict[str, float] = {}
    tick = time.perf_counter()
    frame = _select(frame, ctx, name)
    n = len(frame)
    missing = np.zeros(n, bool)
    if not all(c in frame.columns for c in TEXT_FIELDS):
        if not ctx.source_store:
            raise ValueError(f"{name}: shard has no text columns and no source_store is configured")
        frame, missing = _join_text(frame, ctx.source_store)
    timings["join_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    present = np.flatnonzero(~missing)
    rows = frame.loc[present, KEY + TEXT_FIELDS].to_dict(orient="records")
    cache_key = np.zeros(n, np.uint64)
    cache_key[present] = _cache_keys(ctx.adapter, rows, ctx.cache_prefix)
    logits = np.full(n, np.nan, np.float32)
    status = np.full(n, SCORED, object); status[missing] = MISSING_RECORD
    hit = np.zeros(n, bool)
    for cache in (ctx.reuse, stale):
        if cache is not None and len(cache):
            found, values = cache.lookup(cache_key[present])
            new = found & ~hit[present]
            logits[present[new]] = values[new]; hit[present[new]] = True
    status[hit] = CACHED
    timings["hash_seconds"] = time.perf_counter() - tick

    todo_mask = ~missing & ~hit
    todo = np.flatnonzero(todo_mask)
    todo_rows = [row for row, keep in zip(rows, todo_mask[present]) if keep]
    tick = time.perf_counter()
    encoded = ctx.adapter.encode(todo_rows) if hasattr(ctx.adapter, "encode") and todo_rows else None
    timings["tokenize_seconds"] = time.perf_counter() - tick
    truncated = np.zeros(n, bool); token_length = np.full(n, -1, np.int32)
    if encoded is not None:
        truncated[todo] = encoded["truncated"]; token_length[todo] = np.diff(encoded["offsets"])
    return Prepared(name=name, keys=frame[KEY].copy(), status=status, logits=logits, cache_key=cache_key, truncated=truncated,
                    token_length=token_length, todo=todo, rows=todo_rows, encoded=encoded, timings=timings)


def prepare_shard(path: Path, ctx: Context, *, stale: Path | None = None) -> Prepared:
    tick = time.perf_counter()
    identity = file_identity(path, hash_content=True)
    frame = read_shard(path, ctx)
    read_seconds = time.perf_counter() - tick
    prepared = prepare_frame(frame, ctx, _output_name(path), stale=ReuseCache([stale]) if stale else None)
    prepared.input_identity = identity
    prepared.timings["read_seconds"] = read_seconds
    return prepared


# -- GPU scoring (main thread) ---------------------------------------------

def token_budget_batches(sorted_lengths: np.ndarray, max_tokens: int, max_rows: int) -> list[tuple[int, int]]:
    """Slice descending lengths so rows x widest row stays within max_tokens."""
    batches, start, n = [], 0, len(sorted_lengths)
    while start < n:
        width = max(int(sorted_lengths[start]), 1)
        size = max(1, min(max_rows, max_tokens // width))
        batches.append((start, min(n, start + size))); start += size
    return batches


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _forward(adapter, prepared: Prepared, indices: np.ndarray):
    if prepared.encoded is not None:
        batch = adapter.pad_encoded(prepared.encoded, indices)
    else:
        batch = adapter.collate([prepared.rows[i] for i in indices])
    logits = adapter.logits(batch)
    return (logits[:, 1] - logits[:, 0]).float()


def score_prepared(adapter, prepared: Prepared, *, max_tokens: int, max_rows: int) -> dict[str, Any]:
    """Fill prepared.logits/status for the todo rows; returns throughput stats."""
    import torch

    stats = {"model_pairs": len(prepared.todo), "forward_seconds": 0.0, "batches": 0, "oom_splits": 0, "fp32_retries": 0, "real_tokens": 0, "padded_tokens": 0}
    if not len(prepared.todo):
        return stats
    if prepared.encoded is not None:
        lengths = np.diff(prepared.encoded["offsets"])
    else:
        lengths = np.array([len(adapter.serialize_pair(row)) for row in prepared.rows], np.int64)
    order = np.argsort(-lengths, kind="stable")
    pending = [order[a:b] for a, b in token_budget_batches(lengths[order], max_tokens, max_rows)]
    done: list[tuple[np.ndarray, Any]] = []
    tick = time.perf_counter()
    with torch.inference_mode():
        while pending:
            indices = pending.pop(0)
            try:
                done.append((indices, _forward(adapter, prepared, indices)))
            except RuntimeError as exc:
                if not _is_oom(exc) or len(indices) == 1:
                    raise
                torch.cuda.empty_cache()
                half = len(indices) // 2
                pending[:0] = [indices[:half], indices[half:]]; stats["oom_splits"] += 1
                continue
            stats["batches"] += 1
            stats["real_tokens"] += int(lengths[indices].sum()); stats["padded_tokens"] += int(lengths[indices].max()) * len(indices)
        values = np.full(len(lengths), np.nan, np.float32)
        for indices, tensor in done:  # one host sync per batch, after all batches were queued
            values[indices] = tensor.cpu().numpy()
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad) and getattr(adapter, "inference_precision", "fp32") != "fp32":
            # fp16 can overflow on DeBERTa; rescore only those pairs in fp32.
            saved, adapter.inference_precision = adapter.inference_precision, "fp32"
            try:
                values[bad] = _forward(adapter, prepared, bad).cpu().numpy()
            finally:
                adapter.inference_precision = saved
            stats["fp32_retries"] = len(bad)
    stats["forward_seconds"] = time.perf_counter() - tick
    prepared.logits[prepared.todo] = values
    nonfinite = prepared.todo[~np.isfinite(values)]
    prepared.status[nonfinite] = NONFINITE; prepared.logits[nonfinite] = np.nan
    return stats


def output_frame(prepared: Prepared, checkpoint_id: str) -> pd.DataFrame:
    frame = prepared.keys.reset_index(drop=True).copy()
    frame["neural_logit"] = prepared.logits.astype(np.float32)
    frame["neural_status"] = prepared.status.astype(str)
    frame["neural_scored"] = np.isin(prepared.status, [SCORED, CACHED])
    frame["checkpoint_id"] = checkpoint_id
    frame["cache_key"] = prepared.cache_key
    frame["truncated"] = prepared.truncated
    frame["token_length"] = prepared.token_length
    frame = frame[OUTPUT_COLUMNS]
    if frame.loc[frame["neural_scored"], "neural_logit"].isna().any() or frame.loc[~frame["neural_scored"], "neural_logit"].notna().any():
        raise AssertionError("neural_scored must agree with logit presence")
    return frame


def _write_atomic(frame: pd.DataFrame, target: Path) -> str:
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False); os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True); raise
    return sha256_file(target)


# -- orchestration ----------------------------------------------------------

def _load(config: Mapping[str, Any]):
    adapter_type = str(config.get("adapter_type", ""))
    if adapter_type.lower() in {"fake", "smoke"} and not bool(config.get("smoke", False)):
        raise RuntimeError("fake adapter requires explicit smoke=true and cannot produce production scores")
    adapter_path, checkpoint_id, identity = checkpoint_identity(config["checkpoint"])
    adapter = get_adapter_class(adapter_type).load_pretrained(str(adapter_path), map_location=str(config.get("device", "cpu")))
    if hasattr(adapter, "model"):
        adapter.model.eval()
    return adapter, checkpoint_id, identity


def _context(config: Mapping[str, Any], adapter, checkpoint_id: str, reuse: ReuseCache) -> tuple[Context, dict[str, Any]]:
    signature = {"checkpoint_id": checkpoint_id, "input_signature": input_signature(adapter), "precision": getattr(adapter, "inference_precision", "fp32")}
    route_column = config.get("route_column", "route")
    ctx = Context(adapter=adapter, checkpoint_id=checkpoint_id, cache_prefix=config_hash(signature).encode(),
                  key_columns={k: str(v) for k, v in dict(config.get("key_columns", {})).items()},
                  route_column=str(route_column) if route_column else None, route_value=str(config.get("route_value", "neural")),
                  require_route=bool(config.get("require_route", False)), source_store=config.get("source_store"), reuse=reuse)
    scoring_identity = {"schema_version": SCHEMA_VERSION, **signature, "route": [ctx.route_column, ctx.route_value], "key_columns": ctx.key_columns,
                        "source_store": file_identity(ctx.source_store) if ctx.source_store else None}
    return ctx, scoring_identity


def plan(config: Mapping[str, Any]) -> dict[str, Any]:
    ready, pending = discover_inputs(config)
    output = Path(config["output_dir"])
    done = [p for p in ready if (output / _output_name(p)).is_file() and Path(f"{output / _output_name(p)}.complete.json").is_file()]
    return {"command": "score_neural", "execute_required": True, "ready_inputs": len(ready), "pending_inputs": len(pending), "already_scored": len(done),
            "output_dir": str(output), "reuse_from": list(config.get("reuse_from", [])), "max_tokens": int(config.get("max_tokens", 16384))}


def execute(config: Mapping[str, Any], *, rescore_changed: bool = False, log=print) -> dict[str, Any]:
    ready, pending = discover_inputs(config)
    adapter, checkpoint_id, ckpt_identity = _load(config)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    reuse = ReuseCache.from_dirs([d for d in config.get("reuse_from", []) if Path(d).resolve() != output.resolve()])
    ctx, scoring_identity = _context(config, adapter, checkpoint_id, reuse)
    scoring_hash = config_hash(scoring_identity)
    manifest_path = output / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if manifest is None and any(output.glob(f"*{OUTPUT_SUFFIX}")):
        raise FileExistsError(f"{output} has score shards but no {MANIFEST}; refusing to mix")
    if manifest is not None and manifest.get("scoring_identity_hash") != scoring_hash:
        raise ValueError(f"{output} was scored with a different checkpoint/input format/route; use a new output_dir "
                         f"(existing {manifest.get('checkpoint_id')}, now {checkpoint_id})")
    manifest = manifest or {"schema_version": SCHEMA_VERSION, "kind": "neural-scores", "checkpoint_id": checkpoint_id, "checkpoint": str(config["checkpoint"]),
                            "checkpoint_identity_hash": ckpt_identity["identity_hash"], "scoring_identity": scoring_identity,
                            "scoring_identity_hash": scoring_hash, "score_definition": "neural_logit = logit[match] - logit[no_match]; sigmoid(neural_logit) = P(match)",
                            "shards": {}}

    work: list[tuple[Path, Path | None]] = []
    resumed = 0
    for path in ready:
        final = output / _output_name(path)
        sidecar = Path(f"{final}.complete.json")
        if final.is_file() and sidecar.is_file():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if saved.get("sha256") != sha256_file(final):
                raise ValueError(f"score shard checksum mismatch: {final}")
            if saved.get("input", {}).get("sha256") == file_identity(path, hash_content=True)["sha256"]:
                resumed += 1; continue
            if not rescore_changed:
                raise ValueError(f"input shard changed since it was scored: {path}; rerun with --rescore-changed to rescore it (unchanged pairs are reused)")
            work.append((path, final))
        else:
            work.append((path, None))
    log(f"[score_neural] {checkpoint_id} precision={scoring_identity['precision']} ready={len(ready)} pending={len(pending)} resumed={resumed} to_score={len(work)} reuse_cache={len(reuse)}")

    max_tokens, max_rows = int(config.get("max_tokens", 16384)), int(config.get("max_batch_rows", 512))
    totals = {"rows": 0, "model_pairs": 0, "cached": 0, "missing_record": 0, "nonfinite": 0, "truncated": 0, "forward_seconds": 0.0, "oom_splits": 0, "fp32_retries": 0, "real_tokens": 0, "padded_tokens": 0}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        submit = lambda item: pool.submit(prepare_shard, item[0], ctx, stale=item[1])
        queue = iter(work)
        head = next(queue, None)
        future = submit(head) if head else None
        while future is not None:
            prepared = future.result()
            following = next(queue, None)
            future = submit(following) if following else None  # prefetch while the GPU works
            stats = score_prepared(adapter, prepared, max_tokens=max_tokens, max_rows=max_rows)
            frame = output_frame(prepared, checkpoint_id)
            final = output / prepared.name
            digest = _write_atomic(frame, final)
            counts = frame["neural_status"].value_counts().to_dict()
            atomic_write_json(Path(f"{final}.complete.json"), {"schema_version": SCHEMA_VERSION, "scoring_identity_hash": scoring_hash, "checkpoint_id": checkpoint_id,
                                                               "input": prepared.input_identity, "rows": len(frame), "status_counts": counts, "sha256": digest})
            manifest["shards"][prepared.name] = {"input": prepared.input_identity["path"], "input_sha256": prepared.input_identity["sha256"], "rows": len(frame), "status_counts": counts, "sha256": digest}
            manifest.update(completion="partial", git_commit=git_commit()); atomic_write_json(manifest_path, manifest)
            for key in ("model_pairs", "forward_seconds", "oom_splits", "fp32_retries", "real_tokens", "padded_tokens"):
                totals[key] += stats[key]
            totals["rows"] += len(frame); totals["cached"] += counts.get(CACHED, 0); totals["missing_record"] += counts.get(MISSING_RECORD, 0)
            totals["nonfinite"] += counts.get(NONFINITE, 0); totals["truncated"] += int(frame["truncated"].sum())
            rate = stats["model_pairs"] / stats["forward_seconds"] if stats["forward_seconds"] else 0.0
            log(f"[score_neural] {prepared.name}: rows={len(frame)} model={stats['model_pairs']} cached={counts.get(CACHED, 0)} missing={counts.get(MISSING_RECORD, 0)} "
                f"nonfinite={counts.get(NONFINITE, 0)} {rate:,.0f} pairs/s forward")
    elapsed = time.perf_counter() - started
    known = {_output_name(p) for p in ready + pending}
    manifest["pending_inputs"] = sorted(str(p) for p in pending)
    manifest["unexpected_shards"] = sorted(set(manifest["shards"]) - known)
    manifest["completion"] = "complete" if not pending and all(_output_name(p) in manifest["shards"] for p in ready) else "partial"
    manifest["git_commit"] = git_commit()
    atomic_write_json(manifest_path, manifest)
    return {"output_dir": str(output), "checkpoint_id": checkpoint_id, "completion": manifest["completion"], "shards_scored": len(work), "shards_resumed": resumed,
            "pending_inputs": len(pending), **totals, "wall_seconds": elapsed, "pairs_per_second_wall": totals["rows"] / elapsed if elapsed else None,
            "padding_efficiency": totals["real_tokens"] / totals["padded_tokens"] if totals["padded_tokens"] else None}


# -- benchmark --------------------------------------------------------------

def _sample_rows(ready: Sequence[Path], ctx: Context, n: int, max_shards: int = 16, seed: int = 0) -> tuple[pd.DataFrame, float, int]:
    """Deterministic sample spread across evenly spaced shards; read time prorated to sampled rows."""
    picks = [ready[int(i)] for i in np.unique(np.linspace(0, len(ready) - 1, min(max_shards, len(ready))).astype(int))]
    per_shard = math.ceil(n / len(picks))
    rng = np.random.default_rng(seed)
    parts, read_seconds, total_rows = [], 0.0, 0
    for path in picks:
        tick = time.perf_counter(); frame = read_shard(path, ctx); elapsed = time.perf_counter() - tick
        if ctx.route_column and ctx.route_column in frame.columns:
            frame = frame[frame[ctx.route_column].astype(str) == ctx.route_value]
        take = frame.iloc[np.sort(rng.choice(len(frame), min(per_shard, len(frame)), replace=False))] if len(frame) else frame
        read_seconds += elapsed * (len(take) / max(len(frame), 1)); total_rows += len(frame)
        parts.append(take)
    sample = pd.concat(parts, ignore_index=True).head(n)
    return sample, read_seconds, total_rows


def benchmark(config: Mapping[str, Any], *, pairs: int, max_tokens_sweep: Sequence[int] | None = None, project_pairs: int | None = None, log=print) -> dict[str, Any]:
    import torch

    ready, _ = discover_inputs(config)
    if not ready:
        raise FileNotFoundError("no ready input shards to benchmark")
    adapter, checkpoint_id, _ = _load(config)
    ctx, scoring_identity = _context(config, adapter, checkpoint_id, ReuseCache())
    cuda = torch.cuda.is_available() and str(config.get("device", "cpu")).startswith("cuda")
    sample, read_seconds, _ = _sample_rows(ready, ctx, pairs)
    prepared = prepare_frame(sample, ctx, "benchmark")
    n = len(prepared.todo)
    sweep = list(max_tokens_sweep or [int(config.get("max_tokens", 16384))])
    max_rows = int(config.get("max_batch_rows", 512))
    # Warm up kernels/allocator on the widest batch; not timed.
    if n:
        lengths = np.diff(prepared.encoded["offsets"]) if prepared.encoded is not None else np.ones(n, np.int64)
        widest = np.argsort(-lengths, kind="stable")[: max(1, min(max_rows, max(sweep) // max(int(lengths.max()), 1)))]
        with torch.inference_mode():
            _forward(adapter, prepared, widest).cpu()
    runs = []
    for budget in sweep:
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        stats = score_prepared(adapter, prepared, max_tokens=budget, max_rows=max_rows)
        run = {"max_tokens": budget, **stats, "model_pairs_per_second": n / stats["forward_seconds"] if stats["forward_seconds"] else None,
               "padding_efficiency": stats["real_tokens"] / stats["padded_tokens"] if stats["padded_tokens"] else None}
        if cuda:
            run["peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            run["peak_gpu_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        runs.append(run); log(f"[benchmark] max_tokens={budget}: {run['model_pairs_per_second'] or 0:,.0f} pairs/s forward, oom_splits={stats['oom_splits']}")
    best = max(runs, key=lambda r: r["model_pairs_per_second"] or 0)
    out_dir = Path(config["output_dir"]) / "benchmark"; out_dir.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter(); _write_atomic(output_frame(prepared, checkpoint_id), out_dir / f"sample{OUTPUT_SUFFIX}"); write_seconds = time.perf_counter() - tick

    cpu = {"read": read_seconds, "join": prepared.timings["join_seconds"], "hash": prepared.timings["hash_seconds"], "tokenize": prepared.timings["tokenize_seconds"], "write": write_seconds}
    rows = len(sample)
    per_pair_cpu = sum(cpu.values()) / rows
    per_pair_gpu = best["forward_seconds"] / rows
    lengths = prepared.token_length[prepared.token_length >= 0]
    report = {
        "checkpoint_id": checkpoint_id, "device": torch.cuda.get_device_name() if cuda else "cpu", "precision": scoring_identity["precision"],
        "sample_pairs": rows, "stage_seconds": {**cpu, "forward": best["forward_seconds"]},
        "pairs_per_second_serial": 1.0 / (per_pair_cpu + per_pair_gpu),
        "pairs_per_second_overlapped": 1.0 / max(per_pair_cpu, per_pair_gpu),
        "bottleneck": "gpu_forward" if per_pair_gpu >= per_pair_cpu else "cpu_prepare",
        "best_max_tokens": best["max_tokens"], "sweep": runs,
        "token_length": {q: float(np.percentile(lengths, p)) for q, p in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))} if len(lengths) else {},
        "truncation_rate": float(prepared.truncated.mean()) if rows else 0.0,
        "notes": "overlapped assumes CPU prep of the next shard fully hides behind GPU forward (the scorer prefetches one shard); serial is the no-overlap bound",
    }
    if project_pairs:
        report["projection"] = {"pairs": int(project_pairs), "hours_overlapped": project_pairs / report["pairs_per_second_overlapped"] / 3600,
                                "hours_serial": project_pairs / report["pairs_per_second_serial"] / 3600}
    atomic_write_json(out_dir / f"benchmark-{time.strftime('%Y%m%dT%H%M%S')}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score selected pair shards with the neural matcher; writes keyed raw logits.")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--benchmark", type=int, metavar="PAIRS", help="time the full pipeline on a representative sample of PAIRS rows")
    parser.add_argument("--rescore-changed", action="store_true", help="rescore input shards whose content changed; unchanged pairs are reused")
    parser.add_argument("--max-tokens", default=None, help="benchmark: comma-separated token budgets to sweep")
    parser.add_argument("--project-pairs", type=int, default=None, help="benchmark: pair count to project the full run for")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.benchmark:
        sweep = [int(x) for x in args.max_tokens.split(",")] if args.max_tokens else None
        result = benchmark(config, pairs=args.benchmark, max_tokens_sweep=sweep, project_pairs=args.project_pairs)
    elif args.execute:
        result = execute(config, rescore_changed=args.rescore_changed)
    else:
        result = plan(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
