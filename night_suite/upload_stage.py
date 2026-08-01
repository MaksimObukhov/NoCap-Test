#!/usr/bin/env python3
"""Upload a completed stage's durable logs to W&B and verify the artifact."""

import argparse
import json
import os
from pathlib import Path


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--project", default="nocap-baseline")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).resolve()
    required = [
        stage_dir / "manifest.json",
        stage_dir / "summary.json",
        stage_dir / "gate_decision.json",
        stage_dir / "stdout.log",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"refusing incomplete artifact; missing: {missing}")

    import wandb

    run = wandb.init(
        project=args.project,
        group=args.group,
        name=args.run_name,
        job_type="stage-artifact",
        config={"stage_dir": str(stage_dir)},
    )
    artifact = wandb.Artifact(args.artifact_name, type="nocap-night-stage")
    included = []
    for path in sorted(stage_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(stage_dir)
        if path.suffix == ".pt" or "wandb" in relative.parts:
            continue
        artifact.add_file(str(path), name=str(relative))
        included.append(str(relative))
    if not included:
        raise RuntimeError("stage artifact would be empty")
    logged = run.log_artifact(artifact)
    logged.wait()
    payload = {
        "status": "uploaded",
        "artifact_name": logged.name,
        "artifact_version": logged.version,
        "artifact_id": logged.id,
        "files": included,
        "run_url": run.url,
    }
    write_json(stage_dir / "wandb_upload.json", payload)
    run.summary.update(payload)
    run.finish()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
