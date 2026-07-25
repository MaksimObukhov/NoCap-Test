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
| exp001 | 24.7 | `e3c9ed0` | effective batch 131,072→524,288 over first 50% of tokens; LR ∝ √B | same token budget/data order; proxy s0 improves by ≥0.004 with ≤1% wall-time overhead | proxy s0 | 3.56079 | — | **−0.03698** | +0.83% algo overhead | criterion met, but effect decays — do not promote to full yet | ~$0.75 |
| exp002 | 25.7 | pending | measure the gradient noise scale B_simple along the baseline trajectory (no training change) | B_simple ≪ 524,288 early and crosses it well before 50% of tokens; see predictions below | planned | n/a (measurement) | — | — | — | pending | ~$0.25 |

## exp002 — gradient noise scale measurement (25.7.2026)

**This is a measurement, not an intervention.** Training is unchanged; the run only instruments the baseline trajectory. It exists because exp001's ramp schedule was guessed (8→32 over 50% of tokens) and its biggest open risk — proportional vs absolute indexing of the ramp — is a question about B_crit, which is measurable rather than arguable.

**Method.** McCandlish et al. 2018, *An Empirical Model of Large-Batch Training*, Appendix A. During gradient accumulation we already compute one gradient per micro-batch, so both quantities the estimator needs are almost free:

- `|G_small|²` = mean over micro-batches of `|g_i|²`, at B_small = 16,384 tokens
- `|G_big|²`   = `|mean_i g_i|²`, at B_big = accumulation × 16,384 tokens

then, with `B` in tokens:

```
|G|²      = (B_big·|G_big|² − B_small·|G_small|²) / (B_big − B_small)     unbiased |true gradient|²
tr(Σ)     = (|G_small|² − |G_big|²) / (1/B_small − 1/B_big)               unbiased gradient variance
B_simple  = tr(Σ) / |G|²
```

`B_simple` is the batch size at which gradient noise equals gradient signal. Below it, doubling the batch nearly doubles progress per step; above it, doubling the batch buys almost nothing while costing 2× compute. It is expected to *grow* during training as the easy, mutually-consistent gradient directions get used up.

**Predictions, written before the run** (the point is to be wrong in an informative way, not to pass a threshold):

1. `B_simple` in the first ~10M tokens is of order 10⁴–10⁵ tokens, i.e. **at least 5× below the baseline's 524,288** — this is what exp001's −0.75 nats at 7% of the budget implies, and it would confirm the early phase is over-batched rather than the improvement being an LR artefact.
2. `B_simple` crosses 524,288 **before 50% of the proxy budget (0.47B tokens)**. If it crosses much earlier — say around 0.15B — then exp001's ramp held the batch small roughly 2–3× too long, and the fraction-of-budget indexing is the wrong parameterisation.
3. `B_simple` grows monotonically apart from estimator noise (it is a ratio of two noisy quantities from ~32 samples, so single-step estimates will be jumpy; judge the smoothed trend, not individual points).

If (1) is false — `B_simple` already near or above 524,288 at the start — then the batch story is dead, exp001's gain was an LR-schedule effect, and the next run should be the LR-only control instead.

**Success criterion.** Not a loss number. The run succeeds if it produces a `B_simple(tokens)` curve whose smoothed trend is stable enough to read a crossing point off it. That crossing point then parameterises any future batch ramp, and the same instrumentation is reusable for the sequence-length curriculum.

**Cross-instance reproducibility floor (found by the A/A gate, 25.7.2026).** The instrumented binary with measurement disabled does *not* reproduce the baseline bitwise on a different rented card. The first update differs by 0.35 ulp — the smallest representable non-zero difference, from a different cuBLAS reduction order — and training amplifies it roughly tenfold every ten updates, reaching 0.059 by step 47. The sign stays balanced (43% positive) and the deviation stays at 1.03× the baseline's own step-to-step jitter, so this is chaos, not a code change.

Two things follow. Any paired single-seed comparison against a baseline recorded on a *different* instance carries this floor, which should be of the same order as the 0.0018 seed-to-seed sigma — the 1788-update run tests this directly, since its final val loss should land within roughly ±0.004 of 3.59777. And exp001's −0.03698 is unaffected: it was also measured across cards, but the effect is ~20× the floor.

**Cost.** ~400 updates ≈ 30 min plus a 50-update A/A gate ≈ 5 min, single 4090 ≈ $0.25.

## exp001 — batch ramp, proxy seed 0 (24.7.2026)

Run artifacts: `nocap-runs-backup/dev-exp001-proxy-seed-0-20260724T190552Z` (on branch `exp001/batch-ramp`, commit `33cdf3b`).
The run itself was launched from a dirty tree; `summary.json` records `a331371` + `git_dirty=true`, but its `train_gpt2.py` snapshot is byte-identical to `e3c9ed0`, so the run is reconstructible.

**Result.** final val 3.560791 vs baseline s0 3.597773 → **−0.03698** (≈20σ, 9× the 0.004 MDE). 2549 optimizer updates vs 1788 (+42.6%) at an identical 937,426,944-token budget. Peak VRAM 9826 vs 9821 MiB. No NaN; update-to-update train-loss std 0.0793 vs 0.0633 and max jump +0.42 vs +0.37 — expected from smaller batches, not instability.

**Wall-clock — the reported −1.9% is hardware, not algorithm.** Median full step at the *identical* accum=32 config: 4026.7 ms (exp001) vs 4106.2 ms (baseline). exp001 ran on the same faster instance as the 24.7 profiling session (4.030 s/step); the baseline suite ran on instance 45221458 at 4.14 s. Normalising per micro-batch *within* the exp001 run isolates the real cost: 126.33 ms at accum=8 vs 125.83 ms at accum=32 → **+0.83% over the whole run**. Inside the ≤1% budget, but this is the number that counts.
Rule: never compare step time across rented instances. Compare on one card, or normalise per micro-batch inside a single run.

**The effect decays, and keeps decaying after the ramp ends.** Δ by token fraction: 0.752 @7% → 0.239 @29% → 0.084 @50% → 0.043 @86% → 0.037 @100%. The ramp ends at update 1637 (49% of tokens); past that both runs are physically identical (accum=32, `lr_batch_scale`=1.0, same base LR) and the baseline still closes the gap from 0.084 to 0.037. Token advantage falls 1.92× → ~1.05×.
Post-ramp power fit `Δ(f) ≈ 0.036·f^(−1.25)` extrapolates to **Δ ≈ 0.011 at the full 2.5B budget**. Converting via the schedule-family slope (proxy 0.94B→3.59777, full 2.5B→3.37770 ⇒ 0.1555 nats/doubling): 0.011 nats ≈ 4.6% fewer tokens ≈ ~15 min off a 5.48 h full run, minus 0.83% overhead. Baseline crosses the challenge target 3.3821 at 98.9% of its budget, so there is no slack to absorb an overestimate.

**Methodological finding, wider than exp001.** The `proxy → full` funnel systematically *overestimates* interventions that accelerate the early phase (batch/LR/curriculum). The proxy was calibrated for σ and MDE but never validated as a predictor of full-run ranking. Interventions that change capacity or architecture are less exposed to this.

**Not yet controlled (two open confounds).**
1. Two things changed at once: the batch schedule *and* the shape of the LR trajectory (`lr_batch_scale` 0.5→1.0, so peak LR during warmup is also halved). Isolating ablation: keep accum=32 constant, apply the same time-varying LR multiplier. Costs one proxy; if it recovers most of −0.037, the win is an LR-schedule fix at 0% overhead.
2. No A/A control for the loop rewrite (step-indexed → token-indexed). Ablation: new code with `--batch_ramp_start_accumulation_steps 0`, compare early train_loss against the baseline logs.

**Design risk for the full run.** `ramp_tokens = target_tokens × fraction` is *proportional* to the budget, but B_crit tracks the gradient noise scale, i.e. the loss level, not the fraction of the run. On proxy the ramp ends at 0.47B tokens / loss ≈3.87; on full it would end at 1.25B, where the baseline is already at ≈3.65 — holding the batch small ~0.8B tokens past the point of usefulness. Re-index the ramp on absolute tokens (or loss) before spending on a full run.

[TODO] Add into table:
commit;
validation loss;
wall-clock;
tokens/sec;
VRAM;
стабильность;
convergence per token;

[TODO] restructure this doc, add H2 for baseline, H2 for baseline profiler results and intermediate суждения, после этого вести таблицу с экспериментами в отдельном md файле, а в experiments.md записывать только под новым хедерем
