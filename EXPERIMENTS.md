# Experiment log

One row per experiment, one change per experiment. Hypothesis is always written before running.

Baseline reference (suite `baseline-20260718T074451Z`, commit `bd681a3`, torch 2.11, RTX 4090):
- full (seed 0): 3.37770
- proxy mean (seeds 0/1/2): 3.59935, sigma = 0.0018, MDE (3v3) ~ 0.004
- proxy seed 0 alone: 3.59777 — use this for paired single-seed comparison
- steady step 4.14s, ~126.7k tok/s, peak VRAM 9825 MiB

Funnel (costs at $0.36/hr): smoke ~$0.25 (41 min) -> proxy seed 0 ~$0.75 (2.1h) -> proxy seeds 1/2 -> full ~$2 (5.6h).
Rules:
- smoke only checks "not broken" (compiles, loss falls, no NaN, step time sane). no quality verdicts from smoke.
- proxy seed 0: worse than baseline seed 0 by >0.004 -> kill. better by >0.004 -> run seeds 1/2. gray zone -> judge by hypothesis.
- proxy 3-seed mean better by >0.004 -> full run.
- always commit before running; summary.json records the hash.

| id | date | commit | change (one line) | hypothesis / expected effect | stage reached | proxy s0 | proxy mean | delta vs baseline | step time | verdict | cost |
|----|------|--------|-------------------|------------------------------|---------------|----------|------------|-------------------|-----------|---------|------|
| exp000 | 18.7 | bd681a3 | baseline, no changes | — | full | 3.59777 | 3.59935 | — | 4.14s | reference | ~$4 |

[TODO] Add into table:
commit; - why we need `id` colomn?
validation loss;
wall-clock;
tokens/sec;
VRAM;
стабильность;
convergence per token;
