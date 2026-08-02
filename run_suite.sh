#!/usr/bin/env bash
set -euo pipefail

echo "run_suite.sh is disabled on exp021; the overnight orchestrator owns stage order." >&2
echo "Run a single authorised stage with: bash run.sh {proxy|full} 0" >&2
exit 2
