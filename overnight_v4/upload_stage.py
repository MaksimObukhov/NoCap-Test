#!/usr/bin/env python3
"""Upload small offline-stage evidence to one W&B run and artifact."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--project", default="nocap-baseline")
    args = parser.parse_args()
    import wandb

    directory = Path(args.directory).resolve()
    run = wandb.init(
        project=args.project,
        group=args.group,
        name=args.name,
        job_type="offline-evidence",
    )
    artifact = wandb.Artifact(args.name, type="experiment-evidence")
    uploaded = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".pt" or "compile-cache" in path.parts:
            continue
        if path.name.endswith("trace.json") or path.stat().st_size > 32 * 1024 * 1024:
            continue
        artifact.add_file(str(path), name=str(path.relative_to(directory)))
        uploaded.append(str(path.relative_to(directory)))
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                run.summary[key] = value
    run.summary["uploaded_files"] = len(uploaded)
    run.log_artifact(artifact)
    run.finish()
    receipt = {"status": "uploaded", "files": uploaded, "run_url": run.url}
    (directory / "wandb_upload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
