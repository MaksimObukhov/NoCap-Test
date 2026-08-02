#!/usr/bin/env bash
set -uo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---dry-run}"
REPO_ROOT="${REPO_ROOT:-/workspace/nocap-control}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/workspace/nocap-worktrees-v4}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/nocap-results/overnight-v4-20260802}"
DATA_ROOT="${NOCAP_DATA_DIR:-/workspace/nocap-control/data/fineweb10B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-nocap-baseline}"
WANDB_GROUP="${WANDB_GROUP:-overnight-v4-20260802}"
WORKTREES_JSON="$WORKTREE_ROOT/worktrees.json"
STATUS_FILE="$RESULTS_ROOT/suite_status.json"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--run-paid" ]]; then
  echo "usage: bash run_overnight_v4.sh {--dry-run|--run-paid}" >&2
  exit 2
fi

echo "Suite order:"
echo "  1. exp022 proxy seed 0"
echo "  2. select exp021 or exp022"
echo "  3. exp023 FP8 integration benchmark (no FP8 training)"
echo "  4. exp016 causal A, then systems B only if A passes"
echo "  5. selected BF16 full seed 0 at 2,700,083,200 tokens"
echo "repo=$REPO_ROOT"
echo "worktrees=$WORKTREE_ROOT"
echo "results=$RESULTS_ROOT"
echo "data=$DATA_ROOT"

if [[ "$MODE" == "--dry-run" ]]; then
  exit 0
fi

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$RESULTS_ROOT/suite.console.log") 2>&1

record_stage() {
  "$PYTHON_BIN" "$SCRIPT_ROOT/overnight_v4/stage_status.py" \
    --file "$STATUS_FILE" --stage "$1" --status "$2" \
    --exit-code "${3:-0}" --detail "${4:-}"
}

worktree_path() {
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"])' \
    "$WORKTREES_JSON" "$1"
}

upload_stage() {
  local directory="$1"
  local name="$2"
  if [[ ! -d "$directory" ]]; then
    return 0
  fi
  "$PYTHON_BIN" "$SCRIPT_ROOT/overnight_v4/upload_stage.py" \
    --directory "$directory" --name "$name" --group "$WANDB_GROUP" \
    --project "$WANDB_PROJECT" || echo "WARNING: W&B evidence upload failed: $name"
}

if [[ ! -f "$WORKTREES_JSON" ]]; then
  echo "missing $WORKTREES_JSON; run prepare_worktrees.py first" >&2
  exit 2
fi
if ! compgen -G "$DATA_ROOT/fineweb_train_*.bin" >/dev/null \
   || ! compgen -G "$DATA_ROOT/fineweb_val_*.bin" >/dev/null; then
  echo "FineWeb train/validation shards not found under $DATA_ROOT" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import torch,wandb,torchao; assert torch.cuda.is_available()'; then
  echo "CUDA, wandb and torchao must be importable before the paid suite" >&2
  exit 2
fi

EXP021_WORKTREE="$(worktree_path exp021)"
EXP022_WORKTREE="$(worktree_path exp022)"
EXP023_WORKTREE="$(worktree_path exp023)"
EXP016_WORKTREE="$(worktree_path exp016)"

EXP022_OUT="$RESULTS_ROOT/exp022/proxy-seed-0"
if [[ -f "$EXP022_OUT/summary.json" ]] \
   && "$PYTHON_BIN" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("status") == "complete"' "$EXP022_OUT/summary.json"; then
  record_stage exp022-proxy skipped 0 "existing complete summary"
else
  RESUME_ARGS=()
  if [[ -f "$EXP022_OUT/checkpoints/latest.pt" ]]; then
    RESUME_ARGS=(--resume "$EXP022_OUT")
  fi
  (
    cd "$EXP022_WORKTREE" || exit 1
    export NOCAP_DATA_DIR="$DATA_ROOT"
    export WANDB_PROJECT WANDB_GROUP
    export TORCHINDUCTOR_CACHE_DIR="$RESULTS_ROOT/compile-cache/exp022-proxy"
    bash run.sh proxy 0 "$EXP022_OUT" "${RESUME_ARGS[@]}"
  )
  EXP022_RC=$?
  if [[ "$EXP022_RC" -eq 0 ]]; then
    record_stage exp022-proxy complete 0
  else
    record_stage exp022-proxy failed "$EXP022_RC" "falling back to exp021"
  fi
fi

"$PYTHON_BIN" "$SCRIPT_ROOT/overnight_v4/select_winner.py" \
  --summary "$EXP022_OUT/summary.json" \
  --metrics "$EXP022_OUT/metrics.jsonl" \
  --output "$RESULTS_ROOT/winner.json"
WINNER="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$RESULTS_ROOT/winner.json")"
record_stage select-winner complete 0 "$WINNER"
if [[ "$WINNER" == "exp022" ]]; then
  WINNER_WORKTREE="$EXP022_WORKTREE"
else
  WINNER_WORKTREE="$EXP021_WORKTREE"
fi

FP8_OUT="$RESULTS_ROOT/exp023"
(
  cd "$EXP023_WORKTREE" || exit 1
  export PYTHON_BIN
  bash run_fp8_benchmark.sh "$WINNER" "$WINNER_WORKTREE" "$FP8_OUT"
)
FP8_RC=$?
if [[ "$FP8_RC" -eq 0 ]]; then
  record_stage exp023-fp8-benchmark complete 0
else
  record_stage exp023-fp8-benchmark failed "$FP8_RC" "BF16 full remains authorised"
fi
upload_stage "$FP8_OUT" "exp023-${WINNER}-fp8-integration"

EXP016_OUT="$RESULTS_ROOT/exp016-track"
(
  cd "$EXP016_WORKTREE" || exit 1
  export PYTHON_BIN
  bash run_exp016_ab.sh "$EXP016_OUT" "$CHECKPOINT_ROOT" "$DATA_ROOT"
)
EXP016_RC=$?
if [[ "$EXP016_RC" -eq 0 ]]; then
  record_stage exp016-ab complete 0
else
  record_stage exp016-ab failed "$EXP016_RC" "full continues independently"
fi
upload_stage "$EXP016_OUT/exp016/causal-a" "exp016-causal-a"
upload_stage "$EXP016_OUT/exp016/systems-b" "exp016-systems-b"

FULL_OUT="$RESULTS_ROOT/$WINNER/full-seed-0"
RESUME_ARGS=()
if [[ -f "$FULL_OUT/checkpoints/latest.pt" ]]; then
  RESUME_ARGS=(--resume "$FULL_OUT")
fi
(
  cd "$WINNER_WORKTREE" || exit 1
  export NOCAP_DATA_DIR="$DATA_ROOT"
  export WANDB_PROJECT WANDB_GROUP
  export TORCHINDUCTOR_CACHE_DIR="$RESULTS_ROOT/compile-cache/${WINNER}-full"
  bash run.sh full 0 "$FULL_OUT" "${RESUME_ARGS[@]}"
)
FULL_RC=$?
if [[ "$FULL_RC" -eq 0 ]]; then
  record_stage "$WINNER-full" complete 0 "2,700,083,200 tokens"
  echo "Suite complete. BF16 full winner: $WINNER"
  exit 0
fi
record_stage "$WINNER-full" failed "$FULL_RC" "instance left running for inspection"
echo "Full run failed; inspect $FULL_OUT and do not improvise a replacement run." >&2
exit "$FULL_RC"
