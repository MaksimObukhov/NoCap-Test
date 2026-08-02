#!/usr/bin/env bash
# NoCap proxy suite v3 -- the current entry point.
#
# Defaults to a dry run. Nothing costs money until --run-paid is passed.
# There is deliberately no full stage anywhere in this suite: the manifest
# contains none, and the guard below refuses to start if one ever appears.
#
# Typical use on the GPU host, inside tmux so a dropped SSH connection does
# not take the night with it:
#
#     tmux new -s nocap
#     ./run_proxy_suite_v3.sh --dry-run
#     ./run_proxy_suite_v3.sh --run-paid 2>&1 | tee -a suite.console.log
#     # detach with ctrl-b d, reconnect later with: tmux attach -t nocap

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-$REPO_ROOT/suite_v3/manifest.json}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/nocap-results/proxy-suite-v3}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/workspace/nocap-worktrees}"
BASE_WORKTREE="${BASE_WORKTREE:-$REPO_ROOT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/workspace/nocap-control/data/fineweb10B}"
TRAIN_GLOB="${TRAIN_GLOB:-$DATA_ROOT/fineweb_train_*.bin}"
VAL_GLOB="${VAL_GLOB:-$DATA_ROOT/fineweb_val_*.bin}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

RUN_PAID=0
EXTRA_ARGS=()
for argument in "$@"; do
  case "$argument" in
    --run-paid) RUN_PAID=1 ;;
    --dry-run)  RUN_PAID=0 ;;
    *)          EXTRA_ARGS+=("$argument") ;;
  esac
done

# Structural guard, belt-and-braces with the same assertion inside
# suite_v3.suite. Cheap, and the failure it prevents costs five GPU-hours and
# an unapproved charge. Matches only an argument pair that would actually
# launch a full run: the manifest legitimately names exp000's full-seed-0
# checkpoint as a read-only replay input for exp016, and reading a full run's
# checkpoint is not launching one.
if grep -qE '"--run_mode"[[:space:]]*,[[:space:]]*$' -A1 "$MANIFEST" | grep -q '"full"' \
   || grep -qE '"--run_mode"[[:space:]]*,[[:space:]]*"full"' "$MANIFEST"; then
  echo "REFUSING TO START: manifest $MANIFEST would launch a full run." >&2
  exit 65
fi

COMMON=(
  "$PYTHON_BIN" -m suite_v3.suite
  --manifest "$MANIFEST"
  --results-root "$RESULTS_ROOT"
  --worktree-root "$WORKTREE_ROOT"
  --base-worktree "$BASE_WORKTREE"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --train-glob "$TRAIN_GLOB"
  --val-glob "$VAL_GLOB"
  --python "$PYTHON_BIN"
)

if [[ "$RUN_PAID" -eq 1 ]]; then
  echo "Starting PAID run. Results: $RESULTS_ROOT"
  exec "${COMMON[@]}" --log-wandb --run-paid "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
  exec "${COMMON[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
