#!/usr/bin/env python3
"""Verify and inventory every immutable input required by exp016-A/B."""

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "61f4afe121c6ee1eb077b0356234cad6d031d7d95ae8ab79f18fcc678cb81cbd"
)
EXPECTED_SOURCE_BYTES = 27_614
EXPECTED_CHECKPOINTS = {
    "proxy": {
        "relative_dir": "proxy-seed-0",
        "sha256": "bb909dfe3d3b99258e1d550a62de098877075887551b2dd14695eb55b98034a8",
        "bytes": 1_482_759_539,
    },
    "full": {
        "relative_dir": "full-seed-0",
        "sha256": "893db0f1dfbb582f0a33da598b0e92de0e56bcac3322a002140c6839d3010b4b",
        "bytes": 1_482_759_539,
    },
}
BASELINE_RELATIVE_ROOT = Path(
    "nocap-runs-backup/baseline-20260718T074451Z"
)
DATA_MAGIC = 20240520
DATA_VERSION = 1
DATA_HEADER_BYTES = 1024
EXPECTED_TRAIN_SHARDS = 50
EXPECTED_TRAIN_TOKENS_PER_SHARD = 100_000_000
MINIMUM_VAL_TOKENS = 1_048_577


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_record(path, include_hash):
    with path.open("rb") as handle:
        prefix = handle.read(12)
    if len(prefix) != 12:
        raise ValueError(f"truncated data header: {path}")
    magic, version, tokens = struct.unpack("<3i", prefix)
    size = path.stat().st_size
    expected_size = DATA_HEADER_BYTES + 2 * tokens
    if magic != DATA_MAGIC:
        raise ValueError(f"bad data magic in {path}: {magic}")
    if version != DATA_VERSION:
        raise ValueError(f"bad data version in {path}: {version}")
    if size != expected_size:
        raise ValueError(
            f"data size mismatch for {path}: expected {expected_size}, got {size}"
        )
    record = {"path": str(path), "bytes": size, "tokens": tokens}
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def git_metadata():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0 or subprocess.call(
        ["git", "diff", "--cached", "--quiet"]
    ) != 0
    if dirty:
        raise ValueError("tracked worktree changes invalidate the input manifest")
    return {"branch": branch, "commit": commit, "tracked_dirty": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-data-hashes",
        action="store_true",
        help="validate every data header/size but omit slow dataset SHA-256 hashes",
    )
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    baseline_root = checkpoint_root / BASELINE_RELATIVE_ROOT

    checkpoints = {}
    for label, expected in EXPECTED_CHECKPOINTS.items():
        directory = baseline_root / expected["relative_dir"]
        checkpoint_path = directory / "checkpoint.pt"
        source_path = directory / "train_gpt2.py"
        for path in (checkpoint_path, source_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing exp016 input: {path}")
        checkpoint_size = checkpoint_path.stat().st_size
        checkpoint_sha = sha256_file(checkpoint_path)
        source_size = source_path.stat().st_size
        source_sha = sha256_file(source_path)
        if checkpoint_size != expected["bytes"]:
            raise ValueError(
                f"{label} checkpoint size: expected {expected['bytes']}, "
                f"got {checkpoint_size}"
            )
        if checkpoint_sha != expected["sha256"]:
            raise ValueError(f"unexpected {label} checkpoint SHA-256: {checkpoint_sha}")
        if source_size != EXPECTED_SOURCE_BYTES:
            raise ValueError(
                f"{label} source size: expected {EXPECTED_SOURCE_BYTES}, got {source_size}"
            )
        if source_sha != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"unexpected {label} source SHA-256: {source_sha}")
        checkpoints[label] = {
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_size,
                "sha256": checkpoint_sha,
            },
            "captured_source": {
                "path": str(source_path),
                "bytes": source_size,
                "sha256": source_sha,
            },
        }

    expected_names = [f"fineweb_train_{index:06d}.bin" for index in range(1, 51)]
    actual_train = sorted(data_root.glob("fineweb_train_*.bin"))
    if [path.name for path in actual_train] != expected_names:
        raise ValueError(
            "FineWeb train shard set must be exactly fineweb_train_000001.bin "
            "through fineweb_train_000050.bin"
        )
    val_path = data_root / "fineweb_val_000000.bin"
    if not val_path.is_file():
        raise FileNotFoundError(f"missing validation shard: {val_path}")

    include_hash = not args.skip_data_hashes
    train_records = [data_record(path, include_hash) for path in actual_train]
    wrong_train_tokens = [
        record for record in train_records if record["tokens"] != EXPECTED_TRAIN_TOKENS_PER_SHARD
    ]
    if wrong_train_tokens:
        raise ValueError("one or more FineWeb train shards do not contain 100M tokens")
    val_record = data_record(val_path, include_hash)
    if val_record["tokens"] < MINIMUM_VAL_TOKENS:
        raise ValueError(
            f"validation shard has {val_record['tokens']} tokens; "
            f"need at least {MINIMUM_VAL_TOKENS}"
        )

    payload = {
        "experiment": "exp016-input-manifest-v5",
        "status": "valid",
        "run_plan": {
            "experiment": "exp016",
            "treatment": "descent-budgeted spectral reweighting of attention-V AdamW updates",
            "launcher": "run_exp016_ab.sh",
            "stages": ["causal-a", "systems-b-if-a-passes"],
            "seed": 0,
            "automatic_paid_training": False,
        },
        "source": git_metadata(),
        "checkpoint_root": str(checkpoint_root),
        "data_root": str(data_root),
        "checkpoints": checkpoints,
        "dataset": {
            "hashes_included": include_hash,
            "train_shards": train_records,
            "train_tokens": sum(record["tokens"] for record in train_records),
            "validation_shard": val_record,
        },
    }
    write_json_atomic(output, payload)
    print(f"exp016 inputs valid; manifest written to {output}")


if __name__ == "__main__":
    main()
