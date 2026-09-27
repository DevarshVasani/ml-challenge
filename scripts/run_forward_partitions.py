"""Run and resume bounded parallel S1-forward FTS candidate generation."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.forward_fts import audit_output, index_path, source_path
from src.retrieval_experiments import sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    if args.workers <= 0 or args.partitions <= 0:
        raise ValueError("workers and partitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in ("S2", "S3"):
        index = index_path(Path("artifacts/forward-fts-v2"), args.split, source)
        metadata = json.loads(index.with_suffix(".json").read_text())
        if not metadata.get("complete") or not index.is_file() or sha(index) != metadata["sha256"]:
            raise ValueError(f"forward {source} index failed checksum verification")
    pending = list(range(args.partitions))
    active = {}
    start = time.time()
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                partition = pending.pop(0)
                target = args.output / f"bucket-{partition:02d}"
                target.mkdir(exist_ok=True)
                command = [sys.executable, "-m", "src.forward_fts", "retrieve", "--split", args.split,
                           "--output", str(target), "--k", str(args.k), "--batch-size", str(args.batch_size),
                           "--partition", str(partition), "--partitions", str(args.partitions)]
                log = (args.output / f"bucket-{partition:02d}.log").open("a")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                active[partition] = (process, log)
                print(json.dumps({"started_bucket": partition, "pid": process.pid}), flush=True)
            time.sleep(1)
            for partition, (process, log) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                del active[partition]
                if process.returncode:
                    raise RuntimeError(f"forward bucket {partition} failed; see {args.output / f'bucket-{partition:02d}.log'}")
                print(json.dumps({"completed_bucket": partition}), flush=True)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
    expected = sum(1 for _ in source_path(args.split, "S1").open()) - 1
    audit = audit_output(args.output, args.split, expected)
    if audit["partitions"] != args.partitions or audit["k_per_field"] != args.k:
        raise ValueError("forward output differs from requested partition count or rank budget")
    report = {"complete": True, **audit, "seconds": time.time() - start}
    write_json(args.output / "run_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
