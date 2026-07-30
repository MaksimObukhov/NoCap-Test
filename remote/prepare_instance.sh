#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SKIP_DATASET=0

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: prepare_instance.sh [--skip-dataset]

Runs, in order:
  1. Instance, repository, Python, CUDA, and disk checks.
  2. Install the pinned W&B client when needed.
  3. BF16 GPU throughput gate.
  4. FineWeb download and validation, unless --skip-dataset is supplied.

Useful overrides:
  REMOTE_PYTHON=/venv/main/bin/python
  GPU_BENCH_SECONDS=600
  GPU_MIN_TFLOPS=160
  GPU_MIN_SM_CLOCK_MHZ=2350
  MIN_FREE_GIB=20
  REMOTE_LOG_DIR=/path/to/log/directory
EOF
  exit 0
fi

if [[ "${1:-}" == "--skip-dataset" ]]; then
  SKIP_DATASET=1
elif [[ $# -gt 0 ]]; then
  echo "ERROR: unknown argument: $1" >&2
  exit 2
fi

LOG_DIR="${REMOTE_LOG_DIR:-$REPO_ROOT/runs/remote-prep-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"

echo "=== 1/4 Instance checks ==="
"$SCRIPT_DIR/check_instance.sh" 2>&1 | tee "$LOG_DIR/check_instance.log"

echo
echo "=== 2/4 W&B client ==="
"$SCRIPT_DIR/ensure_wandb.sh" 2>&1 | tee "$LOG_DIR/ensure_wandb.log"

echo
echo "=== 3/4 BF16 GPU benchmark ==="
"$SCRIPT_DIR/benchmark_gpu.sh" 2>&1 | tee "$LOG_DIR/benchmark_gpu.log"

if (( SKIP_DATASET == 1 )); then
  echo
  echo "=== 4/4 FineWeb download skipped ==="
else
  echo
  echo "=== 4/4 FineWeb download ==="
  "$SCRIPT_DIR/download_fineweb.sh" 2>&1 | tee "$LOG_DIR/download_fineweb.log"
fi

echo
echo "Remote instance is ready."
echo "No training run was started."
echo "Logs saved in: $LOG_DIR"
echo
echo "Next profiling command:"
echo "  ./run_profiler.sh 20"
