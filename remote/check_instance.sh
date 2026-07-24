#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${REMOTE_PYTHON:-python}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: check_instance.sh

Environment:
  REMOTE_PYTHON  Python executable from the remote CUDA environment.
  MIN_FREE_GIB   Minimum free disk space required. Default: 20.
EOF
  exit 0
fi

for command_name in git nvidia-smi "$PYTHON_BIN" torchrun; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command is missing: $command_name" >&2
    exit 1
  fi
done

if command -v uv >/dev/null 2>&1; then
  echo "uv: $(uv --version)"
else
  echo "WARNING: uv is not installed; use uv for dependency changes." >&2
fi

echo "repository:"
git -C "$REPO_ROOT" status --short --branch
git -C "$REPO_ROOT" log -1 --oneline

echo
echo "gpu:"
nvidia-smi -L
echo "name, driver, memory, pstate, current SM clock, max SM clock, power, limit, temperature:"
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,pstate,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu \
  --format=csv,noheader

echo
echo "vast lifecycle:"
if [[ -f /etc/vast-agents-guide.md ]]; then
  echo "guide=/etc/vast-agents-guide.md"
else
  echo "WARNING: /etc/vast-agents-guide.md is missing." >&2
fi
if [[ -n "${CONTAINER_ID:-}" ]]; then
  echo "CONTAINER_ID=present"
else
  echo "CONTAINER_ID=missing"
fi
if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
  echo "CONTAINER_API_KEY=present"
else
  echo "CONTAINER_API_KEY=missing"
fi

echo
echo "python/cuda:"
"$PYTHON_BIN" - <<'PY'
import sys

import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"

gpu_name = torch.cuda.get_device_name(0)
assert "4090" in gpu_name, f"expected RTX 4090, got {gpu_name}"

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu={gpu_name}")
PY

FREE_KIB="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
MIN_FREE_KIB=$((MIN_FREE_GIB * 1024 * 1024))

echo
echo "disk:"
df -h "$REPO_ROOT"
if (( FREE_KIB < MIN_FREE_KIB )); then
  echo "ERROR: less than ${MIN_FREE_GIB} GiB is free." >&2
  exit 1
fi

echo
echo "PASS: instance health checks succeeded."
