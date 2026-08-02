#!/usr/bin/env python3
"""Append one atomic stage status record to the suite status file."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--detail", default="")
    args = parser.parse_args()
    path = Path(args.file)
    payload = json.loads(path.read_text()) if path.is_file() else {"stages": []}
    payload["stages"].append(
        {
            "stage": args.stage,
            "status": args.status,
            "exit_code": args.exit_code,
            "detail": args.detail,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
