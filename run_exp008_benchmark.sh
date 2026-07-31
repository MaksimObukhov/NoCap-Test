#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

IMPLEMENTATION_COMMIT="a40345e42b2dbfe29e9044a29568bb127397b5e6"
EXPECTED_BRANCH="exp008/gqa-4kv"
RUN_ID="${1:-exp008-gqa-4kv-profile-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="runs/$RUN_ID"

# Complete every fallible preflight before enabling any instance shutdown.
actual_branch="$(git branch --show-current)"
actual_sha="$(git rev-parse HEAD)"

if [[ "$actual_branch" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch $EXPECTED_BRANCH, got $actual_branch." >&2
  exit 2
fi
if ! git merge-base --is-ancestor "$IMPLEMENTATION_COMMIT" HEAD; then
  echo "ERROR: the reviewed GQA implementation is not in HEAD $actual_sha." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree changes detected." >&2
  git status --short >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "ERROR: run directory already exists: $RUN_DIR" >&2
  exit 2
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  read -r -s -p "Paste WANDB_API_KEY: " WANDB_API_KEY
  echo
  export WANDB_API_KEY
fi
if [[ -z "$WANDB_API_KEY" ]]; then
  echo "ERROR: WANDB_API_KEY is empty." >&2
  exit 2
fi
if [[ -z "${CONTAINER_ID:-}" || -z "${CONTAINER_API_KEY:-}" ]]; then
  echo "ERROR: CONTAINER_ID and CONTAINER_API_KEY are required for auto-stop." >&2
  exit 2
fi
if ! command -v vastai >/dev/null 2>&1; then
  echo "ERROR: vastai CLI is missing." >&2
  exit 2
fi

python - <<'PY'
import os
import wandb

if not wandb.login(key=os.environ["WANDB_API_KEY"], verify=True):
    raise SystemExit("W&B authentication failed")
print("W&B authentication verified")
PY

export WANDB_ENABLED=1
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-nocap-baseline}"
export WANDB_GROUP="${WANDB_GROUP:-exp008-gqa-4kv-systems}"

mkdir -p "$RUN_DIR"

echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "git_sha=$actual_sha"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_group=$WANDB_GROUP"
echo "Starting 50-update GQA systems benchmark."

# Do not use run_profiler.sh here: it deliberately disables W&B.
bash run.sh smoke 0 "$RUN_DIR" \
  --run_mode custom \
  --run_name "$RUN_ID" \
  --num_iterations 50 \
  --warmup_iters 0 \
  --warmdown_iters 0 \
  --val_loss_every 0 \
  --save_every 0 \
  --profile \
  --profile_wait_steps 48 \
  --profile_warmup_steps 1 \
  --profile_active_steps 1 \
  2>&1 | tee "$RUN_DIR/stdout.log"

python - "$RUN_DIR" <<'PY'
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in (run_dir / "metrics.jsonl").read_text().splitlines()
]
times = [
    row["step_time_ms"]
    for row in records
    if row.get("event") == "train" and 11 <= row["step"] <= 48
]
if len(times) != 38:
    raise SystemExit(f"expected 38 steady updates, got {len(times)}")

result = {
    "steady_range": "updates 11-48",
    "steady_updates": len(times),
    "steady_median_ms": statistics.median(times),
    "steady_min_ms": min(times),
    "steady_max_ms": max(times),
}
(run_dir / "benchmark_metrics.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
PY

for artifact in \
  checkpoint.pt \
  config.json \
  metrics.jsonl \
  summary.json \
  stdout.log \
  benchmark_metrics.json \
  profile/rank0_key_averages.txt \
  profile/rank0_trace.json; do
  if [[ ! -s "$RUN_DIR/$artifact" ]]; then
    echo "ERROR: required local artifact is missing: $RUN_DIR/$artifact" >&2
    echo "Instance remains running for inspection." >&2
    exit 1
  fi
done

python - "$RUN_DIR" "$RUN_ID" "$actual_sha" <<'PY'
import os
import sys
from pathlib import Path
import wandb

run_dir = Path(sys.argv[1])
run_id = sys.argv[2]
git_sha = sys.argv[3]

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY") or None,
    project=os.environ["WANDB_PROJECT"],
    group=os.environ["WANDB_GROUP"],
    name=f"{run_id}-profiler-artifact",
    job_type="profiler-artifact",
)
artifact = wandb.Artifact(
    name=f"{run_id}-files",
    type="profiler",
    metadata={"git_commit": git_sha},
)
for relative in (
    "config.json",
    "metrics.jsonl",
    "summary.json",
    "stdout.log",
    "benchmark_metrics.json",
    "train_gpt2.py",
):
    artifact.add_file(str(run_dir / relative), name=relative)
artifact.add_dir(str(run_dir / "profile"), name="profile")
run.log_artifact(artifact)
run.finish(exit_code=0)
print(f"W&B profiler artifact uploaded: {run_id}-files")
PY

sync
echo "Benchmark complete and artifacts verified locally and in W&B."
echo "Stopping Vast instance $CONTAINER_ID to end GPU billing."

for attempt in 1 2 3; do
  if vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
    echo "Vast stop request succeeded."
    exit 0
  fi
  echo "WARNING: Vast stop failed (attempt $attempt); retrying in 15s." >&2
  sleep 15
done

echo "ERROR: auto-stop failed; stop instance $CONTAINER_ID manually." >&2
exit 1
