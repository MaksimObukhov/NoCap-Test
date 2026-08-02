#!/usr/bin/env bash
# LEGACY LAUNCHER -- DISABLED ON PURPOSE.
#
# The original run_suite.sh ran `proxy 0`, `proxy 1`, `proxy 2` and then
# `full 0` with no confirmation. During the night-20260802-v2 review an
# operator came close to launching an unapproved full run with it, because
# the name looks like a current entry point and nothing in it hints that the
# last line costs five GPU-hours.
#
# It is kept as a hard failure rather than deleted so that any stale script,
# note, or muscle-memory invocation stops loudly instead of silently finding
# nothing -- or worse, finding some newer file that happens to reuse the name.
#
# The current entry point is:
#
#     ./run_proxy_suite_v3.sh --dry-run
#
# which defaults to a dry run and contains no full stage at all. Full runs
# require Max's explicit approval and are launched deliberately, one at a
# time, never from a suite script.

set -euo pipefail

cat >&2 <<'MESSAGE'
run_suite.sh is the LEGACY v1 launcher and is disabled.

It contained an unguarded `full 0` stage. Use the current launcher instead:

    ./run_proxy_suite_v3.sh --dry-run     # plan only, no compute
    ./run_proxy_suite_v3.sh --run-paid    # requires explicit approval

To recover the original v1 script for historical reference:

    git show exp/night-suite-harness:run_suite.sh
MESSAGE

exit 64
