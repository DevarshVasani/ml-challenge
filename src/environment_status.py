"""Report host/resource limits without reading dataset files or initializing CUDA."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _cgroup_memory_limit() -> int | None:
    """Return a Linux cgroup memory limit when one is materially bounded."""
    for raw_path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        path = Path(raw_path)
        try:
            value = path.read_text(encoding="ascii").strip()
        except (FileNotFoundError, OSError):
            continue
        if value in {"", "max"}:
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        if limit > 0:
            return limit
    return None


def report(artifact_dir: str | Path = ".", configured: dict | None = None) -> dict:
    artifact_path = Path(artifact_dir)
    disk_probe = artifact_path
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    usage = shutil.disk_usage(disk_probe)
    memory = None
    try:
        import psutil
        memory = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    affinity = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    cgroup = _cgroup_memory_limit()
    usable = min(value for value in (memory, cgroup) if value is not None) if memory is not None or cgroup is not None else None
    defaults = {
        "index_workers": 1, "sparse_threads": 2, "query_batch_size": 128,
        "source_read_chunk_rows": 25000, "pair_join_target_rows": 25000,
        "shard_target_rows": 75000,
    }
    allowed = set(defaults) | {"memory_limit_mb", "memory_budget_mb", "disk_limit_gb", "memory_reserve_fraction", "bootstrap_train_queries", "resume", "strict_country"}
    defaults.update({key: value for key, value in (configured or {}).items() if key in allowed})
    reserve_fraction = float(defaults.get("memory_reserve_fraction", 0.30))
    return {
        "cpu_count": os.cpu_count(), "cpu_affinity_count": affinity,
        "memory_bytes": memory,
        "cgroup_memory_limit_bytes": cgroup,
        "effective_memory_limit_bytes": usable,
        "recommended_working_memory_bytes": int(usable * (1.0 - reserve_fraction)) if usable else None,
        "memory_reserve_fraction": reserve_fraction,
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "artifact_dir": str(artifact_dir),
        "disk_probe_path": str(disk_probe),
        "configured": defaults,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=".")
    parser.add_argument("--config")
    args = parser.parse_args()
    configured = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    print(json.dumps(report(args.artifact_dir, configured), indent=2))


if __name__ == "__main__":
    main()
