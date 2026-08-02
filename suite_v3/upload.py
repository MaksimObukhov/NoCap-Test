"""W&B upload for a finished stage.

Two rules the v2 uploader broke:

1. Upload only after the stage's files are closed. The v2 artifact included
   the uploader's own live log file, so an artifact member was being written
   while it was being read.
2. Never let an upload failure invalidate finished training. Training is
   expensive and the local artifacts are the source of truth; the upload is
   a backup and a viewer. A failed upload writes a receipt saying so and the
   suite carries on.

Checkpoints go up as a separate artifact so a lost instance does not mean
lost weights, and so the metrics artifact stays small enough to fetch.
"""

import json
import os
import time

MAX_ATTEMPTS = 3
RETRY_SECONDS = 15

# Written while the stage is running, or already captured elsewhere.
EXCLUDED_NAMES = {"upload.log", "wandb_upload.json"}
EXCLUDED_DIRS = {"wandb", "checkpoints"}


def collect_files(stage_dir):
    collected = []
    for root, directories, files in os.walk(stage_dir):
        directories[:] = [d for d in directories if d not in EXCLUDED_DIRS]
        for name in sorted(files):
            if name in EXCLUDED_NAMES or name.endswith(".tmp"):
                continue
            absolute = os.path.join(root, name)
            collected.append((absolute, os.path.relpath(absolute, stage_dir)))
    return sorted(collected)


def upload_stage(project, group, run_name, stage_dir, upload_checkpoints=True):
    import wandb

    receipt_path = os.path.join(stage_dir, "wandb_upload.json")
    files = collect_files(stage_dir)
    receipt = {
        "run_name": run_name,
        "group": group,
        "project": project,
        "files": [relative for _, relative in files],
        "attempts": [],
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            run = wandb.init(
                project=project,
                group=group,
                name=f"{run_name}-artifacts",
                job_type="stage-artifacts",
                reinit=True,
            )
            artifact = wandb.Artifact(name=run_name.replace("/", "-"), type="stage")
            for absolute, relative in files:
                artifact.add_file(absolute, name=relative)
            logged = run.log_artifact(artifact)
            logged.wait()
            receipt["artifact_id"] = getattr(logged, "id", None)
            receipt["artifact_name"] = getattr(logged, "name", None)
            receipt["run_url"] = run.url

            if upload_checkpoints:
                receipt["checkpoints"] = _upload_checkpoints(run, run_name, stage_dir)

            run.finish()
            receipt["status"] = "uploaded"
            receipt["attempts"].append({"attempt": attempt, "status": "uploaded"})
            _write(receipt_path, receipt)
            return receipt
        except Exception as error:  # noqa: BLE001
            receipt["attempts"].append(
                {"attempt": attempt, "error": f"{type(error).__name__}: {error}"}
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SECONDS * attempt)

    receipt["status"] = "failed"
    receipt["note"] = (
        "Local artifacts are complete and authoritative. Re-run "
        "suite_v3/upload.py against this directory to retry; no training "
        "needs to be repeated."
    )
    _write(receipt_path, receipt)
    return receipt


def _upload_checkpoints(run, run_name, stage_dir):
    import wandb

    checkpoint_dir = os.path.join(stage_dir, "checkpoints")
    final = os.path.join(checkpoint_dir, "final.pt")
    if not os.path.exists(final):
        return {"status": "absent"}
    artifact = wandb.Artifact(
        name=f"{run_name.replace('/', '-')}-final-checkpoint", type="model"
    )
    artifact.add_file(final, name="final.pt")
    logged = run.log_artifact(artifact)
    logged.wait()
    return {
        "status": "uploaded",
        "artifact_name": getattr(logged, "name", None),
        "bytes": os.path.getsize(final),
    }


def _write(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retry a stage upload")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    receipt = upload_stage(
        arguments.project, arguments.group, arguments.run_name, arguments.stage_dir
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
