#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${REMOTE_PYTHON:-python}"
DURATION_SECONDS="${1:-${GPU_BENCH_SECONDS:-60}}"
MIN_TFLOPS="${2:-${GPU_MIN_TFLOPS:-120}}"
MATRIX_SIZE="${GPU_BENCH_MATRIX_SIZE:-8192}"
INNER_ITERS="${GPU_BENCH_INNER_ITERS:-100}"
WARMUP_ITERS="${GPU_BENCH_WARMUP_ITERS:-20}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: benchmark_gpu.sh [duration_seconds] [minimum_median_tflops]

Defaults:
  duration_seconds       60
  minimum_median_tflops  120

Environment overrides:
  REMOTE_PYTHON
  GPU_BENCH_SECONDS
  GPU_MIN_TFLOPS
  GPU_BENCH_MATRIX_SIZE
  GPU_BENCH_INNER_ITERS
  GPU_BENCH_WARMUP_ITERS

Example matching the longer diagnostic:
  GPU_BENCH_SECONDS=600 ./remote/benchmark_gpu.sh
EOF
  exit 0
fi

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: duration must be a positive number." >&2
  exit 2
fi
if ! [[ "$MIN_TFLOPS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: minimum TFLOPS must be a positive number." >&2
  exit 2
fi
for value in "$MATRIX_SIZE" "$INNER_ITERS" "$WARMUP_ITERS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: benchmark sizes and iterations must be positive integers." >&2
    exit 2
  fi
done

BENCH_DURATION_SECONDS="$DURATION_SECONDS" \
BENCH_MIN_TFLOPS="$MIN_TFLOPS" \
BENCH_MATRIX_SIZE="$MATRIX_SIZE" \
BENCH_INNER_ITERS="$INNER_ITERS" \
BENCH_WARMUP_ITERS="$WARMUP_ITERS" \
"$PYTHON_BIN" - <<'PY'
import os
import statistics
import subprocess
import sys
import time

import torch

duration_seconds = float(os.environ["BENCH_DURATION_SECONDS"])
minimum_tflops = float(os.environ["BENCH_MIN_TFLOPS"])
matrix_size = int(os.environ["BENCH_MATRIX_SIZE"])
inner_iters = int(os.environ["BENCH_INNER_ITERS"])
warmup_iters = int(os.environ["BENCH_WARMUP_ITERS"])

assert duration_seconds > 0
assert minimum_tflops > 0
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"

device_name = torch.cuda.get_device_name(0)
print(f"GPU: {device_name}")
print(
    f"BF16 GEMM: {matrix_size}x{matrix_size}, "
    f"warmup={warmup_iters}, batch={inner_iters}, "
    f"duration={duration_seconds:.0f}s, gate={minimum_tflops:.1f} TFLOPS",
    flush=True,
)

a = torch.randn(
    matrix_size,
    matrix_size,
    device="cuda",
    dtype=torch.bfloat16,
)
b = torch.randn(
    matrix_size,
    matrix_size,
    device="cuda",
    dtype=torch.bfloat16,
)

for _ in range(warmup_iters):
    c = a @ b
torch.cuda.synchronize()

samples = []
deadline = time.perf_counter() + duration_seconds
while time.perf_counter() < deadline:
    start = time.perf_counter()
    for _ in range(inner_iters):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = 2 * matrix_size**3 * inner_iters / elapsed / 1e12
    samples.append(tflops)

    telemetry = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"{tflops:6.1f} TFLOPS | {telemetry}", flush=True)

median_tflops = statistics.median(samples)
print(
    f"summary: samples={len(samples)} "
    f"median={median_tflops:.1f} "
    f"min={min(samples):.1f} "
    f"max={max(samples):.1f} TFLOPS"
)

if median_tflops < minimum_tflops:
    print(
        f"FAIL: median {median_tflops:.1f} < "
        f"{minimum_tflops:.1f} TFLOPS; reject this host.",
        file=sys.stderr,
    )
    sys.exit(3)

print(
    f"PASS: median {median_tflops:.1f} >= "
    f"{minimum_tflops:.1f} TFLOPS."
)
PY
