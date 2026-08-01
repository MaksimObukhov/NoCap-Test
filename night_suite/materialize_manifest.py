#!/usr/bin/env python3
"""Materialize a clean runtime manifest after the control commit is known."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="night_suite/manifest.template.json")
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tracked_dirty = (
        subprocess.call(["git", "diff", "--quiet"]) != 0
        or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0
    )
    if branch != "exp/night-suite-harness":
        raise RuntimeError(f"expected exp/night-suite-harness, got {branch!r}")
    if tracked_dirty:
        raise RuntimeError("tracked control checkout changes invalidate the manifest")
    if not args.suite_id or any(character.isspace() for character in args.suite_id):
        raise ValueError("suite-id must be a non-empty token without whitespace")

    payload = json.loads(Path(args.template).read_text())
    payload["suite_id"] = args.suite_id
    payload["control"]["sha"] = sha
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
