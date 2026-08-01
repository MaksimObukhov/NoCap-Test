#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 EXP011_B_SUMMARY [RUN_ROOT]" >&2
  exit 2
fi

b_summary=$1
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${2:-runs/exp011-c-proxy-${timestamp}}
expected_branch=exp011/attn-v-spectral-cap
actual_branch=$(git branch --show-current)

[[ "$actual_branch" == "$expected_branch" ]] || {
  echo "expected branch ${expected_branch}, got ${actual_branch}" >&2
  exit 1
}
git diff --quiet && git diff --cached --quiet || {
  echo "tracked worktree changes invalidate exp011-C" >&2
  exit 1
}
[[ ! -e "$run_root" ]] || { echo "run root exists: $run_root" >&2; exit 1; }
[[ -f "$b_summary" ]] || { echo "missing B summary: $b_summary" >&2; exit 1; }
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["decision"] == "pass", "exp011-B did not pass"' "$b_summary"
[[ -n "${WANDB_API_KEY:-}" ]] || {
  echo "WANDB_API_KEY is required for exp011-C" >&2
  exit 1
}
compgen -G 'data/fineweb10B/fineweb_train_*.bin' >/dev/null || {
  echo "missing FineWeb shards" >&2
  exit 1
}

echo "experiment=exp011-C"
echo "treatment=attention-V selective spectral AdamW"
echo "mode=proxy"
echo "branch=${actual_branch}"
echo "sha=$(git rev-parse HEAD)"
echo "launcher=run_exp011_c.sh"
echo "updates=1788"
echo "seed=0"
echo "wandb_group=exp011-attn-v-spectral-cap"
echo "auto_stop=disabled"
echo "run_root=${run_root}"
echo "b_summary=${b_summary}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export WANDB_ENABLED=1
export WANDB_MODE=online
export WANDB_PROJECT=${WANDB_PROJECT:-nocap-baseline}
export WANDB_GROUP=exp011-attn-v-spectral-cap

mkdir -p "$(dirname "$run_root")"

bash run.sh proxy 0 "$run_root" \
  --run_name exp011-c-proxy-seed-0 \
  --spectral_attn_v \
  2>&1 | tee "${run_root}-stdout.log"
mv "${run_root}-stdout.log" "${run_root}/stdout.log"

python - "$run_root/summary.json" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path))
loss = summary["final_val_loss"]
if loss <= 3.593773:
    decision = "pass"
elif loss >= 3.601773:
    decision = "kill"
else:
    decision = "inconclusive"
print(json.dumps({"experiment": "exp011-C", "decision": decision, "final_val_loss": loss}, indent=2))
PY

sync
echo "exp011-C artifacts: ${run_root}"
echo "Instance remains running for artifact review."
