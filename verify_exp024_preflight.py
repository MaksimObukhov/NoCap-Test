#!/usr/bin/env python3
"""Create the immutable source, dataset, GPU, and disk manifest for exp024."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import torch


DATA_MAGIC = 20240520
DATA_VERSION = 1
DATA_HEADER_BYTES = 1024
EXPECTED_TRAIN_TOKENS = 100_000_000
EXPECTED_TRAIN_SHARDS = 50


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_record(path):
    with path.open("rb") as handle:
        prefix = handle.read(12)
    if len(prefix) != 12:
        raise ValueError(f"truncated data header: {path}")
    magic, version, tokens = struct.unpack("<3i", prefix)
    size = path.stat().st_size
    if magic != DATA_MAGIC or version != DATA_VERSION:
        raise ValueError(f"unexpected data header in {path}")
    if size != DATA_HEADER_BYTES + 2 * tokens:
        raise ValueError(f"data size does not match header: {path}")
    print(f"hashing {path.name}", flush=True)
    return {
        "path": str(path),
        "bytes": size,
        "tokens": tokens,
        "sha256": sha256_file(path),
    }


def git_source():
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0 or subprocess.call(
        ["git", "diff", "--cached", "--quiet"]
    ) != 0
    if branch != "exp024/staircase-batch-ramp":
        raise ValueError(f"unexpected branch: {branch}")
    if dirty:
        raise ValueError("tracked worktree changes invalidate exp024")
    return {"branch": branch, "commit": commit, "tracked_dirty": False}


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=25)
    args = parser.parse_args()
    if args.minimum_free_gib < 1:
        parser.error("--minimum-free-gib must be positive")

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    expected_names = [
        f"fineweb_train_{index:06d}.bin"
        for index in range(1, EXPECTED_TRAIN_SHARDS + 1)
    ]
    train_paths = sorted(data_root.glob("fineweb_train_*.bin"))
    if [path.name for path in train_paths] != expected_names:
        raise ValueError("FineWeb train set must be exactly shards 000001 through 000050")
    val_path = data_root / "fineweb_val_000000.bin"
    if not val_path.is_file():
        raise FileNotFoundError(f"missing validation shard: {val_path}")

    train_records = [data_record(path) for path in train_paths]
    if any(record["tokens"] != EXPECTED_TRAIN_TOKENS for record in train_records):
        raise ValueError("every FineWeb train shard must contain exactly 100M tokens")
    val_record = data_record(val_path)

    disk = shutil.disk_usage(output.parent)
    minimum_free_bytes = args.minimum_free_gib * 1024**3
    if disk.free < minimum_free_bytes:
        raise ValueError(
            f"need at least {args.minimum_free_gib} GiB free for "
            f"exp024 checkpoints; found {disk.free / 1024**3:.1f} GiB"
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")

    payload = {
        "experiment": "exp024",
        "status": "valid",
        "source": git_source(),
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "dataset": {
            "root": str(data_root),
            "train_tokens": sum(record["tokens"] for record in train_records),
            "train_shards": train_records,
            "validation_shard": val_record,
        },
    }
    write_json_atomic(output, payload)
    print(f"exp024 preflight valid: {output}")


if __name__ == "__main__":
    main()
