# Experiment journal

This is the readable record of the NoCap research funnel. It distinguishes
measured results from hypotheses and plans. A planned experiment is not
evidence; a partial run is not a final proxy result.

## How to read and extend this journal

- `exp000` is the immutable reference. Every quality delta is against the
  appropriate baseline at the same token count unless stated otherwise.
- Before implementation, record one hypothesis, one change, and a success or
  stopping criterion here on `research/experiments` in a docs-only commit.
- The run funnel is `smoke -> proxy seed 0 -> proxy seeds 1/2 -> full`.
  Smoke establishes only that a run is healthy; it makes no quality claim.
- Do not compare wall-clock measurements between rented instances. Compare on
  one card, or normalise within one run. Git SHA is the source of truth; W&B is
  visualisation and backup.
- `research/experiments` contains this journal only. Experiment code stays on
  `expNNN/...`; data-classification code stays on its data branch.

## Baseline — exp000

**Status:** completed reference.
**Commit:** `bd681a3`; suite `baseline-20260718T074451Z`; PyTorch 2.11; RTX 4090.

| Metric | Reference value |
|---|---:|
| Full validation loss, seed 0 | 3.37770 |
| Proxy validation loss, seed 0 | 3.59777 |
| Proxy mean, seeds 0/1/2 | 3.59935 |
| Proxy seed-to-seed sigma | 0.0018 |
| Proxy 3v3 minimum detectable effect | about 0.004 |
| Proxy training budget | 937,426,944 tokens / 1,788 updates |
| Full training budget | 2.5B tokens |
| Steady step / throughput | 4.14 s / about 126.7k tok/s |
| Peak VRAM | 9,825 MiB |
| Approximate full-run cost | $4 |

At $0.36/hour, the planning costs are approximately: smoke $0.25 (41 min),
proxy seed 0 $0.75 (2.1 h), and full $2 (5.6 h). For a paired proxy seed-0
comparison, use 3.59777. A seed-0 result worse by more than 0.004 is normally a
kill; better by more than 0.004 earns seeds 1 and 2; the grey zone is judged by
the pre-registered mechanism.

## Baseline profiler context

**Measurement:** 24 July 2026, RTX 4090, after warm-up: 4.030 s for a complete
optimizer step, about 130k tok/s. GPU utilisation was 99.28%; only 29 ms had no
CUDA work. This is not a CPU or data-loading bottleneck.

`aten::mm` consumed 2.685 s (67.07%) of CUDA time. FlashAttention forward plus
backward was about 355 ms (8.9%); AdamW was 11 ms. The baseline is therefore
GEMM-bound. For any performance intervention, inspect full-step time,
throughput, `aten::mm`, attention, and CUDA idle; the profile estimates a
proposal's cost but does not predict a loss improvement.

## Experiment overview

| ID | Status | What changed or was measured | Key outcome | Decision / next step |
|---|---|---|---|---|
| exp000 | completed | Frozen baseline | Proxy mean 3.59935; full s0 3.37770 | Reference |
| exp001 | completed proxy s0 | Batch 131,072 -> 524,288 over the first 50% of tokens; LR proportional to sqrt(B) | 3.56079, delta -0.03698; +0.83% normalised overhead | Direction real, but the effect decays; do not promote unchanged ramp |
| exp002 | completed measurement | Measured gradient noise scale along unchanged baseline | Mid-run B_crit about 106k vs 524,288 baseline batch; crossing about 910M tokens | Ramp premise retired; measurement informed exp003 |
| exp003 | closed, partial proxy | Flat 16,384-token effective batch; LR 3.182e-4 | -0.3847 at 21.5% of budget; +17.6% overhead | Helpful direction, closed as out-of-scope hyperparameter work |
| exp004-A | completed data analysis | Classified exact baseline FineWeb subset with FineWeb-Edu classifier | 8.2418% of selected tokens are in documents with rounded score >=3 | Evidence for a controlled ordering experiment, not evidence that it helps training |
| exp004-B | planned proxy | Same documents and token count; place high-quality documents at the end | No run or implementation yet | Specify stream construction and success criterion before implementation |

## Completed experiments

### exp001 — batch ramp

**Status:** completed through proxy seed 0 (24 July 2026).
**Implementation:** `e3c9ed0` on `exp001/batch-ramp`; run artifacts
`nocap-runs-backup/dev-exp001-proxy-seed-0-20260724T190552Z` (`33cdf3b`). The
launch tree was dirty (`a331371` in `summary.json`), but the captured
`train_gpt2.py` is byte-identical to `e3c9ed0`, so the result is reconstructible.

**Hypothesis.** With identical data order and token budget, ramping effective
batch 131,072 -> 524,288 during the first half of training, coupled with
LR proportional to sqrt(B), would improve proxy seed 0 by at least 0.004 with
no more than 1% wall-clock overhead.

**What was tested.** Micro-batches remained 16 x 1024 tokens. The token-indexed
loop fixed the budget at 937,426,944 tokens while accumulation changed; this
produced 2,549 optimizer updates rather than 1,788.

| Result | Value |
|---|---:|
| Final proxy validation loss | 3.560791 |
| Delta vs baseline seed 0 | **-0.03698** (about 20 sigma; 9x MDE) |
| Peak VRAM | 9,826 MiB vs 9,821 MiB baseline |
| Normalised algorithmic overhead | +0.83% |
| Stability | No NaN; train-loss std 0.0793 vs 0.0633; max jump +0.42 vs +0.37 |

The reported 1.9% wall-clock advantage is hardware, not algorithm: median
accum=32 step was 4,026.7 ms on the faster exp001 card versus 4,106.2 ms on the
baseline card. Within exp001, micro-batch time was 126.33 ms at accum=8 and
125.83 ms at accum=32, which gives the 0.83% cost.

**Interpretation and caveats.** The advantage decayed despite both runs becoming
physically identical after the ramp ended at 49% of tokens: delta was 0.752 at
7%, 0.239 at 29%, 0.084 at 50%, 0.043 at 86%, and 0.037 at 100%. A post-ramp
fit, `Delta(f) ~= 0.036 f^-1.25`, projects only about 0.011 at 2.5B tokens.
Using the 0.1555-nat-per-doubling proxy/full slope, that is about 4.6% fewer
tokens (roughly 15 minutes of a 5.48 h full run) before overhead. The challenge
target 3.3821 is reached only at 98.9% of baseline budget, leaving little slack.

The proxy-to-full funnel therefore overstates interventions that mostly advance
the early phase. Two confounds remain untested: the ramp also changes the LR
shape (`lr_batch_scale` 0.5 -> 1.0), and no A/A control isolated the
step-indexed -> token-indexed loop rewrite. A proportional ramp is also tied to
budget fraction rather than the loss/noise-scale trajectory; on a full run it
would stay small about 0.8B tokens beyond the analogous proxy point.

**Conclusion.** The proxy criterion was met, but the unmodified ramp is not a
full-run candidate. exp002 measured the underlying premise instead of extending
this schedule.

### exp002 — gradient-noise-scale measurement

**Status:** completed (25 July 2026).
**Implementation:** `9185772` / `exp002/noise-scale`; artifacts
`nocap-runs-backup/exp002-noise-scale` (reconstructed from W&B after the Vast
instance was unavailable).

**Hypothesis and method.** This was not a training intervention. During the
unchanged baseline trajectory, it estimated McCandlish et al.'s `B_simple` from
the mean squared micro-batch gradient at B_small = 16,384 and the squared mean
gradient at B_big = accumulation x 16,384:

```text
|G|^2     = (B_big |G_big|^2 - B_small |G_small|^2) / (B_big - B_small)
tr(Sigma) = (|G_small|^2 - |G_big|^2) / (1/B_small - 1/B_big)
B_simple  = tr(Sigma) / |G|^2
```

The success criterion was a stable enough `B_simple(tokens)` curve to locate
the baseline-batch crossing. Predictions were: early `B_simple` at least 5x
below 524,288, crossing before 0.47B proxy tokens, and a monotonic smoothed
trend.

| Measurement | Result |
|---|---:|
| Measurement steps / usable estimates | 112 / 112 |
| B_crit at first update | 3,283 tokens |
| B_crit through 0.1–0.7B tokens | about 106,000 tokens |
| Baseline / mid-run B_crit | about 4.9x |
| 524,288 crossing | about 910M tokens (97% of proxy) |
| Final validation loss | 3.597008 vs 3.597773 baseline (-0.00077) |
| Measurement overhead | 10,777 MiB peak VRAM; 129,744 tok/s |

Early over-batching and monotonic smoothed growth were confirmed. The predicted
mid-run crossing was falsified: it happens at 97%, not before 50%. Individual
estimates swung by up to 3x, as expected for a ratio from roughly 32 samples.

**What the measurement supports.** Integrating the measured curve with cost
proportional to `B + B_crit`, then applying the 2.9x calibration from exp001,
ranked flat small batches above a ramp: accum=1 (16,384) 53.6% modelled token
saving; track B_crit 33.3%; 16k -> 524k ramp over 97% 17.0%; exp001 as run 6%
measured. The raw model is an ordering aid, not a forecast. Since `B + B_crit`
falls with smaller B, it retired the ramp premise: absent overhead, smaller is
better rather than an interior optimum.

**Caveats retained.** The estimate uses raw rather than AdamW-preconditioned
gradients. The 2.9x calibration extrapolates a 6% effect at 1.6x batch change to
a 32x change. Adam beta1=0.9 and beta2=0.95 are step-timescales: accum=1 would
make the beta2 window cover 32x fewer tokens. Finally, the late rise from 150k
to 2.26M during warmdown partly reflects shrinking signal, so the 97% crossing
is not proof that only the final batch is correctly sized.

An A/A gate also found a cross-instance reproducibility floor: a 0.35-ulp first
update difference from cuBLAS reduction order grew to 0.059 by step 47, with
balanced sign and 1.03x baseline step jitter. Treat approximately +/-0.004 as
the paired cross-card floor. It does not threaten exp001's -0.03698 effect.

**Conclusion.** The baseline is substantially over-batched for almost the full
proxy, but exp003 showed that exploiting this is not free and is outside the
challenge's intended algorithmic scope.

### exp003 — flat small batch

**Status:** deliberately stopped at 22% of proxy budget; closed by scope, not
by failure (25 July 2026).
**Implementation:** `6ae36b8` on `exp003/flat-small-batch`; result rebuilt from
W&B in `nocap-runs-backup/exp003-flat-small-batch` (run state `killed`).

**Hypothesis.** Keep the same micro-batch (16 x 1024) and exact data order, but
update after every micro-batch: effective batch 16,384 for all 937,426,944
tokens, with LR `0.0018 * sqrt(16,384 / 524,288) = 3.182e-4`. The pre-registered
point prediction was proxy seed 0 = 3.4256 (-0.1722), with an honest interval of
-0.08 to -0.17; it also predicted retention Delta(100%)/Delta(50%) > 0.70,
no divergence, train-loss std about 0.36, and +4.1% wall-clock overhead.

| Tokens | Budget | exp003 val | Baseline val | Delta |
|---:|---:|---:|---:|---:|
| 0 | 0% | 10.9418 | 10.9418 | 0.0000 |
| 67,108,864 | 7.2% | 4.9915 | 5.9149 | -0.9234 |
| 134,217,728 | 14.3% | 4.4403 | 5.1016 | -0.6613 |
| 201,326,592 | 21.5% | 4.2709 | 4.6557 | **-0.3847** |

The run did not diverge, but train-loss std was 0.1816 vs 0.0866 baseline
(2.1x, not the predicted 5.7x). Its 1,927.0 s over 12,810 updates was 150.4 ms
per micro-batch versus 128.3 ms baseline: +17.6%, not +4.1%. The two-point
fixed-cost estimate from accum=8 -> 32 did not extrapolate to accum=1; the
real per-update cost is about 22 ms.

**Interpretation.** The early advantage decayed while the intervention remained
active, weakening exp001's explanation that decay came from ending the ramp. A
full loss was never observed: a fit to all three nontrivial points predicts
Delta(100%) about 0.13, while the last two predict about 0.05. The pessimistic
reading is appropriate because exp001's decay accelerated. With the measured
overhead, those scenarios imply 20.0% / 38.8% / 53.6% token saving for final
deltas 0.05 / 0.11 / 0.1722, or -5.6% / -27.7% / -45.2% wall-clock respectively.

**Conclusion.** The measured direction helps but is a batch/LR hyperparameter,
which the challenge explicitly does not seek. The Adam-timescale falsifier was
not reached. Record it as a scoped-out negative result; carry forward only the
rough 22 ms update cost and the observation that early-phase gains decay.

### exp004-A — FineWeb-Edu classification of the baseline data

**Status:** completed data analysis (29 July 2026); no training intervention.
**Implementation provenance:** `9801b8c` (initial CPU classifier) and `d17a79f`
(CUDA implementation) on `data/fineweb-edu-classification`. The classifier code
is intentionally separate from this documentation branch. Local archived
artifact: `data/nocap-fineweb-edu/fineweb-edu-results.tar.gz`; SHA-256
`425b803e04909c70d694e0719303070d40103ba42eabd83bdc8ff73bb6ae86a7` matches its
recorded checksum.

**Question.** Can the exact FineWeb token subset consumed by the baseline be
partitioned reproducibly into lower- and higher-quality documents, without
silently changing its data volume?

**Method.** The analysis reconstructs the loader selection rather than taking a
global shard prefix. It uses the baseline configuration: 4,768 iterations,
batch 16, sequence length 1,024, and accumulation 32. For each ordered shard it
takes whole fixed-size micro-batches, counts shifted target positions `[1, N+1)`,
and drops a shard tail that cannot form another micro-batch. This selects
2,499,805,184 target tokens across 26 shards (25 near-100M spans plus a final
16,384-token span), exactly 4,768 x 16 x 1,024 x 32.

Documents are reconstructed from GPT-2 EOT boundaries, including a document that
crosses a shard or selected-span boundary so its text can be scored once. Only
its overlap with the selected spans contributes to the reported token totals.
The text is decoded with the GPT-2 tokenizer and scored by
`HuggingFaceFW/fineweb-edu-classifier` in evaluation mode. The remote completed
run used one CUDA GPU, bfloat16, classifier batch 256, and classifier maximum
length 512. Scores are rounded after clipping to [0, 5]; `int_score >= 3` marks a
document high-quality. The manifest pins the selection with SHA-256
`7cacfa8d4fca62f11ffb823cdb8ea547d8aa8d5a7c45e63b92e5edd0ccb241fa`.

| Result | Value |
|---|---:|
| Complete | yes |
| Documents scored | 3,613,954 |
| Selected tokens scored | 2,499,805,184 |
| High-quality selected tokens | 206,027,733 |
| High-quality token fraction | **8.2418%** |
| Score mean / range | 1.1956 / -0.8594 to 5.0625 |
| Elapsed time / throughput | 6,009.7 s / 601.35 documents/s |

**What this establishes.** There is a reproducible, auditable partition for a
same-data ordering experiment. It does *not* establish that the classifier's
score is a causal measure of usefulness for this model, that 8.24% is an ideal
mixture, or that a classifier-selected curriculum improves validation loss.

**Limitations.** This is document-level classification, not token-level quality:
all selected tokens of an accepted document count as high-quality. Each document
is truncated to 512 classifier tokens, so the score may not represent its later
content. The classifier's rounded threshold is an operational partition, not a
calibrated quality boundary. Scores and text hashes make the pass reproducible,
but the current artifact proves the analysis, not a new training stream. Any
future stream builder must preserve each selected token exactly once, handle
documents that overlap selected-span boundaries explicitly, and avoid accidental
duplication or omission at EOT boundaries.

## Planned work

### exp004-B — late high-quality-data proxy

**Status:** TODO — not implemented, not run, and no dataset has been created.

**Hypothesis.** At the same proxy token budget and with the same selected FineWeb
documents used exactly once, training on lower-quality documents first and the
exp004-A high-quality partition last will improve final proxy validation loss
relative to the baseline ordering. The mechanism to test is late-stage data
mixture, not data removal or an increase in high-quality-token count.

**Planned intervention.** Keep the exact 2,499,805,184 selected-token set and
the baseline volume. Preserve document-internal token order; emit the
lower-quality partition first and the 206,027,733 high-quality selected tokens
at the end. The high-quality tail is therefore about 8.24% of the token budget.
This is a provisional design statement, not an implementation specification.

**TODO before implementation.** On `research/experiments`, add a docs-only
pre-registration that fixes: (1) the exact stream-construction rule for
documents crossing selected spans and for shifted targets; (2) a hash, token
count, document count, and per-partition accounting proving same-data-once;
(3) whether the high-quality tail is contiguous or has a pre-registered
transition; (4) the proxy success threshold, timing/VRAM guardrails, baseline
comparison, and kill rule. Then create `exp004/late-quality-tail` from that
commit. Do not start paid training without explicit approval.

**Success criterion to pre-register.** The future run must first demonstrate
identical selected-token accounting. Only then can proxy seed 0 be judged by
final validation loss against 3.59777 and by wall-clock/VRAM against exp000. A
better loss alone would not justify a conclusion if the data set, token count,
or loader semantics differ.
