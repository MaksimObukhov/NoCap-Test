# NoCap nightly suite

This harness runs only the pre-registered benchmark, health-diagnostic, and
proxy stages. It never launches a full run.

## Safety model

- The instance-provided tmux session is required. None of these scripts creates,
  attaches, or replaces a tmux session.
- `manifest.json` pins every branch to a full pushed SHA. Experiment worktrees
  are detached and never switched during the queue.
- A valid scientific `kill` writes `gate_decision.json`, uploads the stage
  artifact to W&B, and continues with the next experiment.
- A crash, wrong/dirty SHA, data/checkpoint mismatch, missing summary, or failed
  W&B artifact upload is an infrastructure failure. The queue stops and leaves
  the instance running for inspection.
- There are no `EXIT`, `ERR`, `INT`, or `TERM` lifecycle traps. The checked-in
  manifest uses `auto_stop=disabled`; stopping the instance remains a separate
  explicit action after the live Vast guide has been reviewed.

## Durable outputs

Each stage writes outside its Git worktree: `manifest.json`, `config.json` when
applicable, `metrics.jsonl`, diagnostic or benchmark JSON, `gate_decision.json`,
`summary.json`, `stdout.log`, profiler files, source snapshots, and a verified
`wandb_upload.json`. Checkpoints and compiler caches are excluded from W&B
artifacts.

The remote workflow is:

1. clone the control branch and inspect `/etc/vast-agents-guide.md`;
2. prepare all detached pinned worktrees with `prepare_worktrees.py`;
3. run `run_night_suite.py --preflight-only --probe-wandb`;
4. review the single reconciled state, then run the same command with `--run`.

