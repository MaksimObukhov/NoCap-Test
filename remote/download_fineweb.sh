#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${REMOTE_PYTHON:-python}"
DATA_DIR="$REPO_ROOT/data/fineweb10B"
EXPECTED_TRAIN_SHARDS=50

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: download_fineweb.sh

Downloads the canonical 50 FineWeb training shards plus the validation shard.
Existing complete files are reused by the repository downloader.

Environment:
  REMOTE_PYTHON  Python executable with huggingface_hub installed.
EOF
  exit 0
fi

echo "FineWeb target: $DATA_DIR"
df -h "$REPO_ROOT"

cd "$REPO_ROOT"
if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "ERROR: huggingface_hub is missing from $PYTHON_BIN." >&2
  echo "Install it with:" >&2
  echo "  uv pip install --python \"$PYTHON_BIN\" huggingface-hub==0.30.2" >&2
  exit 1
fi
"$PYTHON_BIN" -c "import huggingface_hub; print(f'huggingface_hub={huggingface_hub.__version__}')"
"$PYTHON_BIN" data/cached_fineweb10B.py

DATA_DIR="$DATA_DIR" \
EXPECTED_TRAIN_SHARDS="$EXPECTED_TRAIN_SHARDS" \
"$PYTHON_BIN" - <<'PY'
import glob
import os
import struct

data_dir = os.environ["DATA_DIR"]
expected_train_shards = int(os.environ["EXPECTED_TRAIN_SHARDS"])
train_files = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
val_files = sorted(glob.glob(os.path.join(data_dir, "fineweb_val_*.bin")))

if len(train_files) != expected_train_shards:
    raise RuntimeError(
        f"expected {expected_train_shards} train shards, found {len(train_files)}"
    )
if len(val_files) != 1:
    raise RuntimeError(f"expected 1 validation shard, found {len(val_files)}")

total_tokens = 0
total_bytes = 0
for path in val_files + train_files:
    with open(path, "rb") as f:
        magic, version, token_count = struct.unpack("<3i", f.read(12))
    if magic != 20240520:
        raise RuntimeError(f"bad magic number in {path}")
    if version != 1:
        raise RuntimeError(f"unsupported version in {path}: {version}")

    expected_size = 256 * 4 + token_count * 2
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise RuntimeError(
            f"size mismatch in {path}: expected {expected_size}, got {actual_size}"
        )

    total_tokens += token_count
    total_bytes += actual_size

print(f"validated train shards: {len(train_files)}")
print(f"validated val shards:   {len(val_files)}")
print(f"total tokens:           {total_tokens:,}")
print(f"total size:             {total_bytes / 1024**3:.2f} GiB")
print("PASS: FineWeb download is complete and shard headers are valid.")
PY
