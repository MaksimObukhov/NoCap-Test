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
| exp002 | 25.7 | `9185772` | measure the gradient noise scale B_simple along the baseline trajectory (no training change) | B_simple ≪ 524,288 early and crosses it well before 50% of tokens; see predictions below | measurement, 1788 updates | 3.59701 (baseline reproduction) | — | −0.00077 vs baseline s0 | 4.01s | prediction 1 confirmed, prediction 2 falsified — the baseline is over-batched ~4.9× for 97% of the run | ~$0.75 |
| exp003 | 25.7 | `6ae36b8` | flat effective batch 16,384 tokens for the whole run (accum=1), LR ∝ √B = 3.182e-4 | proxy s0 = **3.4256** (−0.1722), predicted before the run from the exp002 B_crit curve; see predictions below | proxy s0, stopped at 22% of budget | — (stopped) | — | −0.3847 @ 21.5%, extrapolates to −0.05…−0.13 at 100% | +17.6% algo overhead | direction works but is a hyperparameter, not an algorithm — closed by scope, see below | ~$0.35 |

## exp003 — flat small batch (25.7.2026)

**One sentence.** exp002 measured that the baseline's 524,288-token batch is ~4.9× larger than the gradient noise scale for 97% of the run; exp003 stops paying for that by updating every 16,384 tokens instead, over the identical token budget and identical data order.

**Why flat and not a ramp.** Token cost per unit of optimization progress goes as `B + B_crit`. That is monotonically decreasing in `B` — there is no interior optimum, so the best schedule is "as small as overhead allows", and any ramp loses because it spends most of its time at a large batch. This is what retired the exp001 shape; exp003 is the shape the measurement actually points at.

**What changes and what does not.** The micro-batch stays 16×1024, so the data order is bit-identical to the baseline and to exp001 — only the grouping of micro-batches into updates differs. The token budget stays pinned at 937,426,944 via the token-indexed loop from exp001 (`e3c9ed0`), so warmup/warmdown/validation land at the same token positions. Number of optimizer updates goes 1,788 → 57,216.

### Sizing, computed before the run

`ceiling_from_noise_scale.py` at `1f90719` on `exp002/noise-scale`, run against `nocap-runs-backup/exp002-noise-scale/exp002-noise-scale-seed-0/metrics.jsonl`:

| schedule | calibrated token saving | implied val loss |
|---|---|---|
| **accum=1 flat (16,384)** | **53.6%** | **−0.1722** |
| accum=2 flat (32,768) | 50.2% | −0.1562 |
| accum=4 flat (65,536) | 44.0% | −0.1299 |
| accum=8 flat (131,072) | 33.6% | −0.0918 |
| track B_crit exactly | 33.3% | −0.0909 |
| 16k → 524k ramped over 97% | 17.0% | −0.0419 |
| exp001 as run | 6% (measured) | −0.037 (measured) |

Token saving converts to loss through the schedule family's own slope, 0.1555 nats per doubling of the token budget (proxy 0.94B → 3.59777, full 2.5B → 3.37770): saving a fraction `X` buys `log2(1/(1−X))` extra doublings.

**Why accum=1 and not the safer accum=4.** Two reasons, and neither is "the number is bigger".

1. *The LR rule is least ambiguous at accum=1.* We scale LR ∝ √B, but McCandlish's own theory says the optimum is `ε_max/(1 + B_crit/B)`, which is linear in B when `B ≪ B_crit` and flat when `B ≫ B_crit`. Evaluating that against the measured B_crit curve and weighting by tokens gives an optimal LR ratio of 0.2031 at accum=1 versus the √B rule's 0.1768 — 13% apart. At accum=4 the same comparison is 0.4863 vs 0.3536, **38% apart**. Running accum=4 with the √B rule would under-drive the LR by a quarter, and a miss would be unattributable: model too optimistic, or LR too low? At accum=1 that fork mostly closes. (The near-agreement at accum=1 is a property of this curve, not a law — the two rules cross somewhere, and here the crossing lands near accum=1–2.)
2. *accum=4 does not remove the Adam risk, it divides it by four.* 14,304 updates is still an 8× compression of every moment window relative to the baseline. A half-sized result at accum=4 would not distinguish "the cost model overstates" from "Adam is the binding constraint". accum=1 tests the ceiling and the main risk in the same $0.75.

**Learning rate.** `0.0018 × √(16,384/524,288) = 3.182e-4`, applied through the existing `lr_batch_scale`, i.e. the same coupling rule exp001 used. Kept deliberately at the √B rule rather than the noise-scale-optimal 3.656e-4, so that exp003 and exp001 remain comparable and the batch is the only knob that moved.

### Predictions, written before the run

1. **Point prediction: proxy s0 = 3.4256, i.e. −0.1722 vs the baseline's 3.59777.** This is the least trustworthy line here and it is stated anyway so the miss is measurable: the 2.9× calibration factor was fitted on a 6% effect at a 1.6× batch change and is being applied to a 53.6% effect at a 32× batch change. The interval I would actually bet on is **−0.08 to −0.17**.
2. **Retention, not just size.** exp001's advantage decayed because after the ramp ended both runs were physically identical (Δ 0.084 at 50% → 0.037 at 100%, 44% retained). exp003 keeps the small batch to the last token, so that mechanism is absent. Prediction: **Δ(100%) / Δ(50%) > 0.70**. If retention is instead ~0.44 again, the decay is not about "when the intervention stops" and the exp001 explanation was wrong.
3. **Stability without divergence.** No `clip_grad_norm_` exists anywhere in `train_gpt2.py`, so small-batch gradient noise is not distorted by clipping. Update-to-update train-loss std should rise roughly as 1/√B, i.e. ×√32 ≈ 5.7 over the baseline's 0.0633, to **~0.36**. No NaN, no divergence.
4. **Wall-clock overhead +4.1%.** The fixed cost of one optimizer update, fitted from exp001's two in-run points (126.33 ms/micro-batch at accum=8 vs 125.83 at accum=32), is C ≈ 5.3 ms; at accum=1 that is paid every micro-batch. This is the number that decides whether a token saving becomes a wall-clock win, and the fit is a two-point extrapolation from 8 down to 1, so it is the prediction most likely to be off. Do not compare against any step time recorded on another instance.
5. **The falsifier that matters.** The cost model gives accum=1 a 2.15× token advantage against exp001's 1.06×. If proxy s0 lands **worse than exp001's 3.56079**, the batch direction has hit the Adam-timescale ceiling rather than a modelling error, and exp004 should be β2 (0.95 → 0.99) at fixed accum=1 — a clean single-knob test — not a different batch size.

**Known caveats, unchanged from exp002.** B_crit was measured on raw gradients while AdamW is adaptive, so the curve may be systematically biased for this optimizer. β1=0.9 / β2=0.95 are timescales in steps: at accum=1 the β2 window covers 327,680 tokens instead of 10,485,760, and β2=0.95 is a large-batch convention in the first place. The token-indexed loop does put warmup on tokens (50.3M ⇒ 3,072 updates of warmup at accum=1), so the moment estimates are at least not being asked to converge in 96 steps.

**Cost.** Straight to proxy seed 0, smoke skipped by decision: 2.1 h + ~4% overhead, single 4090, ≈ $0.75. Skipping smoke trades a $0.25 insurance premium against a $0.75 exposure; accepted because there is no clipping, the LR coupling is the one already validated in exp001, and neither model nor data changed.

### Result — stopped at 22% of budget (25.7.2026)

**Stopped deliberately, not because it was failing.** Re-reading the README mid-run settled a scope question that should have been settled before exp001: the challenge explicitly does not want hyperparameter work ("We're not here to optimize learning rates and torch.compile flags"; the *not expected to* list names LR bumps, hyperparameter magic, and grid-searching Adam betas). Its Technical Notes go further and treat batch size and gradient accumulation as the knobs you turn to fit your GPU. However carefully exp003's batch was derived from measurement, the deliverable reads as "effective batch 16,384 instead of 524,288, LR × √B" — a hyperparameter. The run was killed at update ~12,810 and the record reconstructed from W&B by `pull_from_wandb.py` (`nocap-runs-backup/exp003-flat-small-batch`, W&B state `killed`).

**What the partial run does establish.** Four validations landed before the stop, against the baseline at identical token counts:

| tokens | % of budget | exp003 val | baseline val | Δ |
|---|---|---|---|---|
| 0 | 0% | 10.9418 | 10.9418 | 0.0000 |
| 67,108,864 | 7.2% | 4.9915 | 5.9149 | **−0.9234** |
| 134,217,728 | 14.3% | 4.4403 | 5.1016 | −0.6613 |
| 201,326,592 | 21.5% | 4.2709 | 4.6557 | −0.3847 |

Against the five pre-registered predictions:

1. **Point prediction — untestable, but trending below the bet interval.** A power law fit through the three non-trivial Δ points extrapolates to Δ(100%) ≈ 0.13; fit through the last two only (the decay is steepening, exponent −0.76 → −1.34) gives ≈ 0.05. Both sit at or under the low end of the −0.08…−0.17 interval, and the honest reading is the pessimistic one, since exp001 established that this decay keeps accelerating. **The 2.9× calibration factor still overstated the effect.**
2. **Retention — not measurable, but the early decay already argues against it.** Δ halved between 7.2% and 21.5% while the intervention was still fully active. exp001's decay was blamed on the ramp ending; here nothing ends, and it decays anyway. That weakens the exp001 explanation independently of where exp003 would have finished.
3. **Stability — confirmed as "no divergence", falsified as a number.** No NaN, no blowup. Update-to-update train-loss std over the last 500 updates was 0.1816 vs the baseline's 0.0866 — ×2.1, not the predicted ×5.7. Small-batch noise does not propagate to the loss as 1/√B here.
4. **Wall-clock overhead — badly missed. +17.6%, predicted +4.1%.** 1,927.0 s of train time over 12,810 updates = 150.4 ms per micro-batch, against the baseline's 128.3 ms (4,106 ms / 32). The two-point fit of C ≈ 5.3 ms from accum=8→32 does not extrapolate to accum=1; the real fixed cost per optimizer update is ~22 ms. This was flagged before the run as the prediction most likely to be off, and it was.
5. **Falsifier — not reached.** The run never produced a final val loss, so the Adam-timescale ceiling was neither confirmed nor ruled out.

**Net effect, priced properly.** Converting through the schedule-family slope of 0.1555 nats per doubling, and paying the *measured* 17.6% overhead:

| Δ at 100% | token saving | wall-clock vs baseline |
|---|---|---|
| 0.05 (pessimistic fit) | 20.0% | −5.6% |
| 0.11 | 38.8% | −27.7% |
| 0.1722 (predicted) | 53.6% | −45.2% |

So the direction is real but the plausible band is wide and its floor is nearly nothing — and this is a *proxy* number, which exp001 showed systematically overstates early-phase interventions on the full budget. A −5.6% full-run result built on a hyperparameter is not what the challenge is asking for.

**Verdict: closed by scope, not by failure.** The measurement stands (the baseline genuinely is ~4.9× over-batched for 97% of the run), the intervention genuinely helps, and it is still a hyperparameter. Budget moves to an algorithmic change. exp004 is multi-token prediction heads — explicitly on the README's suggested list, and in the same family as both entries that currently beat the baseline on the leaderboard.

**Carried forward into `IDEA.md`.** Worth writing up as a scoped-out negative result: the noise scale was measured rather than guessed, the ceiling was sized before spending, and the direction was dropped once it was clear what class of change it belonged to. The two reusable facts for later runs are the measured per-update fixed cost (~22 ms, i.e. small batches are not free) and the decay behaviour of early-phase interventions.

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

**Cost.** Extended to the full 1788-update proxy length before launching, because a 400-update run only reaches 0.21B tokens and the crossing point might sit beyond that. 2.1 h, single 4090, ≈ $0.75. Artifacts: `nocap-runs-backup/exp002-noise-scale` (commit `fdb13df` on `exp002/noise-scale`).

### Result (25.7.2026)

Run completed all 1788 updates from a clean tree (`git_dirty: false`), 112 measurement steps, and every one of them produced a usable estimate — no negative variance or signal terms, so the estimator is well conditioned on this problem.

**In plain terms: the baseline trains with a batch about five times larger than it needs, and it does so for essentially the whole run.**

| what we measured | number |
|---|---|
| B_crit at the very first update | **3,283 tokens** — 160× smaller than the baseline's 524,288 |
| B_crit through mid-training (0.1–0.7B tokens) | ~106,000 tokens — the baseline is running **4.9× larger** |
| where B_crit finally reaches 524,288 | **~910M tokens = 97% of the proxy budget** |
| final val loss | 3.597008 vs baseline 3.597773 (−0.00077) |

Against the three predictions written before the run:

1. **Confirmed, and then some.** B_crit early is 3,283 tokens, an order of magnitude below even the 10⁴ end of the predicted range. The early phase is not mildly over-batched, it is over-batched by more than two orders of magnitude.
2. **Falsified.** The crossing was predicted before 50% of the budget; it happens at 97%. The reasoning behind the prediction was simply wrong — B_crit does not climb fast enough to meet 524,288 mid-run, it sits 4–8× below it almost the whole way.
3. **Confirmed.** Growth is monotonic under smoothing; individual points swing by up to 3× as expected from a ratio of two quantities estimated from 32 samples.

**Why the falsified prediction is the valuable one.** The crossing point bounds how much this whole direction can win, because the baseline only overpays where B_crit is below its batch. Predicting a crossing at 50% meant the direction was capped at a few percent and should be dropped. Measuring it at 97% means nearly the entire budget is overpaid, and the direction is the most valuable one on the board.

**Predicted token savings** (`ceiling_from_noise_scale.py`, integrating cost ∝ `B + B_crit` over the measured curve, with the model calibrated against exp001's measured 1.06× — the raw model overstates gains by 2.9× on this problem):

| schedule | calibrated token saving |
|---|---|
| accum=1 flat (16,384 tokens) | **53.6%** |
| track B_crit exactly | 33.3% |
| 16k → 524k ramped over 97% | 17.0% |
| 16k → 524k ramped over 90% | 15.4% |
| exp001 as run | 6% (measured) |

**This retires the ramp idea.** Token cost goes as `B + B_crit`, which falls monotonically as B falls — there is no interior optimum, so a flat small batch beats any ramp, and beats even tracking B_crit exactly (at `B = B_crit` you pay `2·B_crit`; at `B → 0` you pay `B_crit`). A ramp loses simply because it spends most of its time at a large batch. The GPT-3-style ramp premise behind exp001 is not what this problem wants.

So the measurement's real payoff was not "derive the schedule" — the derived schedule is trivially "as small as overhead allows". It was sizing the prize: 4.9× over-batched for 97% of the run.

**Caveats, in order of how much they could matter.**
1. The estimator uses raw gradients, but AdamW is adaptive, and McCandlish notes the relevant noise scale for an adaptive optimizer should be computed on the preconditioned gradient. The measured B_crit may be systematically biased for this optimizer.
2. The calibration factor of 2.9× comes from a 6% effect at a 1.6× batch change, and is being applied to a 32× batch change. Treat the savings table as an ordering, not a forecast.
3. Adam's `β1=0.9, β2=0.95` are timescales in *steps*, not tokens. At accum=1 the run does 57,216 updates instead of 1,788, so the second-moment estimate averages over 32× less data. The model cannot see this, and it is the most likely way an aggressive schedule fails.
4. The sharp rise at the end (150k → 2.26M over the last 15%) coincides with warmdown: near the schedule's minimum `|G|²` shrinks while `tr(Σ)` does not, so the ratio climbs. This is a real effect — the end of training genuinely wants a large batch — but it means the 97% crossing is partly a warmdown artefact and should not be read as "the batch was only correct at the very end".

**Side results.**
- Final val loss landed 0.00077 from the baseline on a different rented card, i.e. the cross-instance reproducibility floor is roughly half the 0.0018 seed-to-seed sigma. Single-seed paired comparisons across instances stay valid, and exp001's −0.037 is ~48× this floor.
- Measurement overhead: peak VRAM 10,777 MiB against the baseline's 9,821 (the fp32 gradient buffer), throughput 129,744 tok/s.
- Operational: the instance auto-stopped as designed, then its GPU was rented away before the files were fetched, which on Vast blocks a restart for hours to weeks. W&B held the full record and `pull_from_wandb.py` rebuilt the local files exactly. Fetch before stopping, or treat W&B as the real backup.

**Next.** exp003 should test a *flat small batch*, not a ramp. Recommended starting point is accum=4 (65,536 tokens) rather than accum=1: mid-training B_crit is 106,000, so accum=4 is already below the noise scale and inside the target regime, capturing roughly 71% of the modelled benefit at a quarter of the extra optimizer steps — and therefore a quarter of the exposure to caveat 3.

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
