"""Reconstruct a run's metrics.jsonl and summary.json from W&B.

The local files are the canonical record, but when an instance is stopped and its
GPU gets rented out from under you, W&B is the only copy left. Every event the
training loop writes to metrics.jsonl is also logged to W&B with the same keys,
so the local files can be rebuilt exactly.

usage: pull_from_wandb.py PROJECT GROUP OUTPUT_DIR [ENTITY]
"""

import json
import os
import sys

import wandb


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2

    project, group, output_dir = sys.argv[1:4]
    entity = sys.argv[4] if len(sys.argv) == 5 else None

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = [r for r in api.runs(path) if r.group == group]

    if not runs:
        print(f"no runs in {path} with group {group}", file=sys.stderr)
        print("available groups:", file=sys.stderr)
        for g in sorted({r.group for r in api.runs(path) if r.group}):
            print(f"  {g}", file=sys.stderr)
        return 3

    for run in runs:
        run_dir = os.path.join(output_dir, run.name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"{run.name}  state={run.state}  id={run.id}")

        rows = list(run.scan_history())
        events = {}
        with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
            for row in rows:
                record = {k: v for k, v in row.items() if not k.startswith("_")}
                if not record.get("event"):
                    continue
                events[record["event"]] = events.get(record["event"], 0) + 1
                f.write(json.dumps(record, sort_keys=True) + "\n")

        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            summary = {
                k: v for k, v in dict(run.summary).items() if not k.startswith("_")
            }
            json.dump(summary, f, indent=2, sort_keys=True, default=str)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"args": dict(run.config)}, f, indent=2, sort_keys=True, default=str)

        print(f"  {len(rows)} logged rows -> {run_dir}")
        for event, count in sorted(events.items()):
            print(f"    {event}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
