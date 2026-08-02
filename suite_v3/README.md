# Proxy suite v3 — operator runbook

Start it, disconnect, sleep. Everything below assumes a single RTX 4090
Vast.ai instance and the layout the v2 night used.

**No stage in this suite is a full run.** The manifest contains none, the
launcher refuses to start if one appears, and `test_suite.py` asserts both.
A full run is launched deliberately, alone, with Max's explicit approval.

## What it runs

| Order | ID | Stages | Worst case |
|---|---|---|---|
| 1 | exp018 | `fp8_microbench` | 5 min |
| 2 | exp019 | `accounting → benchmark → proxy` | 140 min |
| 3 | exp017 | `accounting → proxy` | 118 min |
| 4 | exp016 | `checkpoint_validation → causal_replay → proxy` | 155 min |
| 5 | exp020 | `accounting → proxy` | 126 min |

Worst case ≈ 9.1 h. Realistically shorter: exp018 either passes in five
minutes or dies, exp019's benchmark can kill it after ~18 min, and exp016's
causal replay can kill it after ~15 min.

`accounting` costs about 30 seconds and runs no training. It is the cheapest
place to catch a model that is not the model that was preregistered.

There is no separate health stage. Gradient health rides inside the proxy,
tagged per schedule phase, with two tripwires: non-finite values, and a
validation loss above a per-experiment envelope. See the "Proxy suite v3 —
funnel design" section of `EXPERIMENTS.md` for why.

## Before you start

1. **Push the branches** (from the Mac; nothing is pushed automatically):

   ```bash
   git push origin exp/proxy-suite-v3 exp017/no-wd-tied-embedding \
       exp019/swiglu-fused-ramp exp020/cosine-schedule research/experiments
   ```

2. **Upload exp016's canonical checkpoints** — about 3 GB total. Without
   them exp016 stops at `checkpoint_validation` and the suite continues to
   exp020, which is a valid outcome, just a wasted opportunity.

   ```bash
   rsync -avP --progress \
     nocap-runs-backup/baseline-20260718T074451Z/proxy-seed-0/checkpoint.pt \
     nocap-runs-backup/baseline-20260718T074451Z/full-seed-0/checkpoint.pt \
     "$VAST_SSH":/workspace/nocap-runs-backup/baseline-20260718T074451Z/
   ```

   Their SHA-256 hashes are pinned in `manifest.json` and verified before
   use, so a truncated transfer fails loudly rather than replaying garbage.

3. **Prepare worktrees** on the instance:

   ```bash
   /venv/main/bin/python -m suite_v3.prepare_worktrees \
     --repo /workspace/nocap-control \
     --manifest /workspace/nocap-control/suite_v3/manifest.json \
     --worktree-root /workspace/nocap-worktrees \
     --base-worktree /workspace/nocap-control \
     --output /workspace/nocap-results/prepared.json
   ```

   Every worktree is detached and must be clean; the suite refuses to start
   otherwise.

## Start it

```bash
tmux new -s nocap
cd /workspace/nocap-control
./run_proxy_suite_v3.sh --dry-run          # plan only, costs nothing
./run_proxy_suite_v3.sh --run-paid 2>&1 | tee -a /workspace/suite.console.log
```

Detach with `ctrl-b d`. Reattach later with `tmux attach -t nocap`.

Read the dry-run output before passing `--run-paid`. It prints every
experiment, its stages, its worktree and the worst-case GPU time.

## While it runs, or after

```bash
# current state, one line per experiment
python -c "import json;print(json.dumps(json.load(open('/workspace/nocap-results/proxy-suite-v3/suite_status.json')),indent=2))"

# full history, append-only, never rewritten
tail -f /workspace/nocap-results/proxy-suite-v3/suite_events.jsonl
```

Per stage, on local disk:

- `stage.json` — identity: manifest hash, SHA, exact commands
- `gate.json` — the decision and all evidence behind it
- `summary.json`, `metrics.jsonl`, `stdout.log`
- `checkpoints/final.pt` for proxy stages, hashed into `summary.json`
- `wandb_upload.json` — upload receipt, including failures

Local artifacts are authoritative. W&B is a backup and a viewer.

## Decisions you will see

| Decision | Meaning | Next stage runs? |
|---|---|---|
| `pass` | proceed | yes |
| `warning` | proceed, but say why in the report | yes |
| `inconclusive` | no claim either way | no |
| `kill` | negative result | no |
| `invalid` | the measurement is untrustworthy; **not** a result | no |

`invalid` is not a scientific finding. It means the thing measured was not
the thing described — wrong parameter count, drifted host, missing final
checkpoint — and the experiment must be fixed and re-run, not written up.

## If something goes wrong

**A stage crashed.** The experiment is recorded and the suite moves on. Look
at that stage's `gate.json` for the exception and `stdout.log` for the trace.

**The instance died mid-proxy.** Re-run the same command. Completed stages
are reused only when their identity hash still matches the manifest; the
interrupted proxy resumes from `checkpoints/latest.pt` with its optimizer,
loader cursor, RNG state and token cursor intact, and its `metrics.jsonl` is
truncated to the checkpoint's token cursor so no update is logged twice.

**An upload failed.** Training is not repeated. Retry just the upload:

```bash
/venv/main/bin/python -m suite_v3.upload \
  --stage-dir /workspace/nocap-results/proxy-suite-v3/exp019/proxy \
  --project nocap-baseline --group exp019 \
  --run-name proxy-suite-v3-exp019-proxy-seed-0
```

**A proxy aborted on a tripwire.** `summary.json` carries
`status: aborted-nonfinite` or `aborted-envelope` plus the detail. Exit
codes: 3 non-finite, 4 envelope, 5 no final checkpoint.

**You want to re-run one experiment.** `--only exp019 --force`.

## Tests

```bash
python -m suite_v3.test_suite
```

48 tests, no GPU, no torch, under a second. Several encode a specific v2
defect and fail against that code: the pooled warmup/steady median that
killed exp015, the rotating single checkpoint that lost exp012–exp014's
final weights, the filename-only stage reuse that let a stale control SHA
through, and the artifact that included the uploader's own live log.
