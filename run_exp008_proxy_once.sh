#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

IMPLEMENTATION_COMMIT="a40345e42b2dbfe29e9044a29568bb127397b5e6"
EXPECTED_BRANCH="exp008/gqa-4kv"
RUN_ID="${1:-exp008-gqa-4kv-proxy-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="runs/$RUN_ID/proxy-seed-0"

actual_branch="$(git branch --show-current)"
actual_sha="$(git rev-parse HEAD)"

# Failures in this preflight never stop the instance.
if [[ "$actual_branch" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch $EXPECTED_BRANCH, got $actual_branch." >&2
  exit 2
fi
if ! git merge-base --is-ancestor "$IMPLEMENTATION_COMMIT" HEAD; then
  echo "ERROR: reviewed GQA implementation is not in HEAD $actual_sha." >&2
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

vastai start instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >/dev/null
echo "Vast lifecycle authentication verified"

export WANDB_ENABLED=1
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-nocap-baseline}"
export WANDB_GROUP="${WANDB_GROUP:-exp008}"

mkdir -p "$RUN_DIR"

echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "git_sha=$actual_sha"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_group=$WANDB_GROUP"
echo "Starting exp008 GQA proxy seed 0: 1,788 optimizer updates."

# If training or artifact verification fails, set -e exits here and deliberately
# leaves the instance running for inspection. Auto-stop exists only below the
# successful completion path.
bash run.sh proxy 0 "$RUN_DIR" 2>&1 | tee "$RUN_DIR/stdout.log"

for artifact in checkpoint.pt metrics.jsonl summary.json stdout.log config.json; do
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
    name=f"{run_id}-result-artifact",
    job_type="result-artifact",
)
artifact = wandb.Artifact(
    name=f"{run_id}-results",
    type="proxy-result",
    metadata={"git_commit": git_sha, "seed": 0, "updates": 1788},
)
for relative in (
    "config.json",
    "metrics.jsonl",
    "summary.json",
    "stdout.log",
    "train_gpt2.py",
):
    artifact.add_file(str(run_dir / relative), name=relative)
run.log_artifact(artifact)
run.finish(exit_code=0)
print(f"W&B result artifact uploaded: {run_id}-results")
PY

sync
echo "Proxy complete; local files and W&B result artifact verified."
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
