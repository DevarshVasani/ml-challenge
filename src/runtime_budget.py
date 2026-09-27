"""Runtime and resource budget estimator for Member D (§12).

Computes stage ETAs, critical-path ETA, aggregate disk requirements, peak
concurrent memory, and the latest safe start time before the 4-hour reserve.

Does NOT hardcode 64 GiB; reads actual memory limits from the production host.

Adds one concurrency guard: refuses to launch when configured peak_rss_gb *
cpu_workers exceeds the production budget.

Usage
-----
    python -m src.runtime_budget --config configs/member_d/runtime_budget.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Host memory detection
# ---------------------------------------------------------------------------


def get_host_memory_gb() -> float:
    """Detect available memory in GiB from /proc/meminfo or cgroup v1/v2."""
    # cgroup v2
    cgroup_v2 = Path("/sys/fs/cgroup/memory.max")
    if cgroup_v2.exists():
        raw = cgroup_v2.read_text().strip()
        if raw != "max":
            try:
                return int(raw) / (1024 ** 3)
            except ValueError:
                pass

    # cgroup v1
    cgroup_v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_v1.exists():
        try:
            val = int(cgroup_v1.read_text().strip())
            # A very large value means no cgroup limit
            if val < (1 << 60):
                return val / (1024 ** 3)
        except ValueError:
            pass

    # /proc/meminfo
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024 ** 2)

    # Windows fallback via ctypes (unlikely in WSL but defensive)
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass

    # Unknown: return a conservative default and warn
    print("[runtime_budget] WARNING: could not detect host memory; assuming 8 GiB")
    return 8.0


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------


def compute_stage_budget(stage: dict[str, Any]) -> dict[str, Any]:
    """Compute ETA and resource estimates for a single pipeline stage."""
    rows_per_second = float(stage.get("rows_per_second", 1))
    input_row_count = int(stage.get("input_row_count", 0))
    projected_runtime_s = float(stage.get("projected_runtime_s", 0))

    if projected_runtime_s <= 0 and rows_per_second > 0 and input_row_count > 0:
        projected_runtime_s = input_row_count / rows_per_second

    peak_rss_gb = float(stage.get("peak_rss_gb", 0))
    disk_read_gb = float(stage.get("disk_read_gb", 0))
    disk_write_gb = float(stage.get("disk_write_gb", 0))
    artifact_size_gb = float(stage.get("artifact_size_gb", 0))
    cpu_workers = int(stage.get("cpu_worker_count", 1))
    gpu_count = int(stage.get("gpu_count", 0))

    return {
        "name": stage.get("name", "unnamed"),
        "projected_runtime_s": projected_runtime_s,
        "projected_runtime_human": _fmt_duration(projected_runtime_s),
        "peak_rss_gb": peak_rss_gb,
        "disk_read_gb": disk_read_gb,
        "disk_write_gb": disk_write_gb,
        "artifact_size_gb": artifact_size_gb,
        "cpu_worker_count": cpu_workers,
        "gpu_count": gpu_count,
        "concurrent_memory_estimate_gb": peak_rss_gb * cpu_workers,
    }


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


def compute_budget(config: dict[str, Any]) -> dict[str, Any]:
    """Compute the full runtime budget.

    Parameters
    ----------
    config : dict with keys:
        stages : list[dict]  per-stage inputs (see compute_stage_budget)
        recovery_reserve_hours : float  (default 4)
        contest_deadline_epoch : float  (optional UNIX timestamp of contest end)

    Returns
    -------
    dict with stage budgets, critical-path ETA, aggregate disk, peak memory,
    latest safe start time, and a concurrency guard result.
    """
    stages_config = config.get("stages", [])
    recovery_reserve_s = float(config.get("recovery_reserve_hours", 4)) * 3600
    deadline_epoch = config.get("contest_deadline_epoch")

    host_memory_gb = get_host_memory_gb()
    stage_budgets = [compute_stage_budget(s) for s in stages_config]

    critical_path_s = sum(b["projected_runtime_s"] for b in stage_budgets)
    aggregate_disk_gb = sum(
        b["disk_read_gb"] + b["disk_write_gb"] for b in stage_budgets
    )
    peak_concurrent_memory_gb = max(
        (b["concurrent_memory_estimate_gb"] for b in stage_budgets), default=0.0
    )
    artifact_total_gb = sum(b["artifact_size_gb"] for b in stage_budgets)

    # Concurrency guard
    concurrency_ok = peak_concurrent_memory_gb <= host_memory_gb
    concurrency_message = (
        f"OK: peak {peak_concurrent_memory_gb:.1f} GiB <= host {host_memory_gb:.1f} GiB"
        if concurrency_ok
        else (
            f"REFUSED: peak {peak_concurrent_memory_gb:.1f} GiB "
            f"> host {host_memory_gb:.1f} GiB. "
            f"Reduce cpu_worker_count or peak_rss_gb before launching."
        )
    )

    # Latest safe start
    latest_start_epoch = None
    latest_start_human = "unknown"
    if deadline_epoch:
        latest_start_epoch = float(deadline_epoch) - critical_path_s - recovery_reserve_s
        import datetime
        dt = datetime.datetime.fromtimestamp(latest_start_epoch)
        latest_start_human = dt.isoformat()

    # Cut optional experiments
    optional_stages = [s for s in stage_budgets if stages_config[stage_budgets.index(s)].get("optional", False)]
    required_path_s = sum(
        b["projected_runtime_s"]
        for b, sc in zip(stage_budgets, stages_config)
        if not sc.get("optional", False)
    )
    recovery_safe = (required_path_s + recovery_reserve_s) <= (
        (float(deadline_epoch) - time.time()) if deadline_epoch else float("inf")
    )

    result: dict[str, Any] = {
        "host_memory_gb": round(host_memory_gb, 2),
        "stage_budgets": stage_budgets,
        "critical_path_s": round(critical_path_s, 1),
        "critical_path_human": _fmt_duration(critical_path_s),
        "aggregate_disk_gb": round(aggregate_disk_gb, 2),
        "artifact_total_gb": round(artifact_total_gb, 2),
        "peak_concurrent_memory_gb": round(peak_concurrent_memory_gb, 2),
        "recovery_reserve_s": recovery_reserve_s,
        "concurrency_guard": {
            "ok": concurrency_ok,
            "message": concurrency_message,
        },
        "latest_safe_start_epoch": latest_start_epoch,
        "latest_safe_start_human": latest_start_human,
        "recovery_reserve_safe": recovery_safe,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Member D runtime and resource budget estimator.")
    parser.add_argument("--config", required=True, help="Path to runtime_budget.json config")
    parser.add_argument("--output", default=None, help="Path to write JSON output (optional)")
    parser.add_argument("--refuse-on-violation", action="store_true",
                        help="Exit with code 1 if concurrency guard fails")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = compute_budget(cfg)

    print("\n" + "=" * 60)
    print("[runtime_budget] Summary")
    print("=" * 60)
    print(f"  Host memory          : {result['host_memory_gb']:.1f} GiB")
    print(f"  Critical-path ETA    : {result['critical_path_human']}")
    print(f"  Aggregate disk       : {result['aggregate_disk_gb']:.1f} GiB")
    print(f"  Peak concurrent mem  : {result['peak_concurrent_memory_gb']:.1f} GiB")
    print(f"  Concurrency guard    : {result['concurrency_guard']['message']}")
    if result["latest_safe_start_human"] != "unknown":
        print(f"  Latest safe start    : {result['latest_safe_start_human']}")
    print()
    for sb in result["stage_budgets"]:
        print(
            f"  [{sb['name']}]  ETA: {sb['projected_runtime_human']}  "
            f"RSS: {sb['peak_rss_gb']:.1f}GB  "
            f"disk R/W: {sb['disk_read_gb']:.1f}/{sb['disk_write_gb']:.1f}GB"
        )
    print("=" * 60 + "\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"[runtime_budget] Written: {args.output}")

    if args.refuse_on_violation and not result["concurrency_guard"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
