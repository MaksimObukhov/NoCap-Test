#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${REMOTE_PYTHON:-python}"
WANDB_VERSION="${WANDB_VERSION:-0.19.9}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ensure_wandb.sh

Installs the pinned W&B client into REMOTE_PYTHON when it is missing or has a
different version.

Environment:
  REMOTE_PYTHON  Python executable for the training environment.
  WANDB_VERSION  Required W&B version. Default: 0.19.9.
EOF
  exit 0
fi

if "$PYTHON_BIN" -c \
  "import importlib.metadata; assert importlib.metadata.version('wandb') == '$WANDB_VERSION'" \
  >/dev/null 2>&1; then
  echo "wandb=$WANDB_VERSION already installed in $PYTHON_BIN"
  exit 0
fi

echo "Installing wandb==$WANDB_VERSION into $PYTHON_BIN"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON_BIN" "wandb==$WANDB_VERSION"
else
  "$PYTHON_BIN" -m pip install "wandb==$WANDB_VERSION"
fi

"$PYTHON_BIN" -c \
  "import importlib.metadata; print('wandb=' + importlib.metadata.version('wandb'))"
