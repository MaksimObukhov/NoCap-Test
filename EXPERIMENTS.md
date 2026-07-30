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
| exp004-B | planned proxy | Same proxy tokens exactly once; move all score >=3 tokens into a warmdown-aligned enriched mixture | No run or implementation yet | Build and audit the stream, then smoke and proxy seed 0 |
| exp005 | planned proxy | T=512 for updates 0-895, then T=1024; keep tokens/update, token order, LR, and validation fixed | No run or implementation yet | Measure compile/shape cost on the target 4090 before proxy |

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

### exp004-B — warmdown-aligned high-quality mixture

**Status:** planned on 30 July 2026; not implemented, not run, and no training
stream has been created.

**Hypothesis.** At the same 937,426,944-token proxy budget, with every target
token occurrence from the baseline proxy prefix used exactly once and with the
model, LR schedule, effective batch, update count, and validation unchanged,
moving all FineWeb-Edu `int_score >= 3` tokens into an enriched mixture spanning
the complete warmdown will improve final proxy seed-0 validation loss by at
least 0.004. The mechanism under test is *when* scarce high-scoring data is
consumed, not filtering, repetition, or adding data.

**Why this is not a pure high-quality tail.** The proxy prefix contains
77,677,516 score >=3 target tokens, or 8.2862% of the budget. A contiguous tail
would start around update 1,640, where LR has already fallen below 39% of peak,
and those tokens would see only about 19% of peak LR on average. Distributing
them over the full warmdown gives them about 50% of peak LR on average while
still making the late mixture substantially richer. This avoids changing the
baseline LR schedule to rescue the curriculum.

**Pre-registered stream.**

| Stage | Updates | Target tokens | Contents |
|---|---:|---:|---|
| general | 0-1403 | 736,100,352 | score 0-2 target occurrences in original order |
| enriched warmdown | 1404-1787 | 201,326,592 | all 77,677,516 score >=3 tokens plus 123,649,076 remaining score 0-2 tokens |

The warmdown mixture is therefore 38.5828% score >=3 and 61.4172% score 0-2.
The builder will preserve order within the high and non-high queues and merge
the two queues deterministically in proportion to their final token totals.
It may split only the one non-high document required to land exactly on the
update-1404 boundary, plus the already partial document at the end of the
baseline proxy prefix. Document-internal order within every emitted segment is
preserved. No selected target occurrence may be duplicated or omitted.

The output will be ordinary pre-materialised `.bin` shards consumed by the
unchanged fixed-shape loader. Each output shard will contain one context-only
prefix followed by a multiple of the 16,384-token micro-batch size, so shard
tails cannot silently lose targets. The target stream remains exactly
937,426,944 tokens / 1,788 updates. A manifest must record the source selection
hash, score-file hash, per-stage and per-score counts, split-document details,
source-span coverage, output shard hashes, and the exact transition.

**Controlled variables.** Training remains B=16, T=1024, accumulation=32,
524,288 tokens/update, 1,788 updates, warmup=96, warmdown=384, peak LR=0.0018,
seed 0, and the unchanged 1,048,576-token T=1024 validation. The data shape does
not change at the stage boundary, so any `torch.compile` recompile is a failure,
not an expected cost.

**Expected dynamics and falsifiers.** Validation may be worse before update
1,404 because all high-scoring data is deliberately withheld; that is not an
early stopping signal. Supporting evidence is a closing or crossing validation
gap during warmdown. The sign of the train-loss jump at the transition is not
pre-registered because FineWeb-Edu score is not an example-difficulty measure.
The mechanism is weakened if the gain appears before the enriched stage, if a
recompile or loader stall occurs at the boundary, or if accounting differs from
the baseline proxy prefix.

**Success and stopping criteria.**

- Accounting must pass before loss is interpreted.
- Final seed-0 validation `<= 3.593773` (delta <= -0.004 vs 3.597773) passes
  the quality gate and earns proxy seeds 1/2.
- Final validation `>= 3.601773` (delta >= +0.004) kills the hypothesis.
- The interval between those thresholds is inconclusive and does not
  automatically earn more seeds.
- No NaN/OOM, no unexpected compile, and no more than 1% normalised runtime
  overhead from the pre-materialised data path.

**Branch and run identity.** Implementation branch
`exp004/late-quality-mixture`; W&B group, run directory, manifest, and result
record use `exp004-B`. Paid smoke or proxy training still requires explicit
approval after code review.

### exp005 — staged sequence length, 512 to 1024

**Status:** planned on 30 July 2026; not implemented or run.

**Hypothesis.** With identical target-token order, token budget, effective
batch, update count, LR schedule, and T=1024 validation, using T=512 for the
first half of the proxy and T=1024 for the second half will reduce normalised
end-to-end wall-time by at least 1% without worsening final proxy seed-0
validation by more than 0.004.

**Pre-registered schedule.**

| Stage | Updates | T | Micro-batch B | Accumulation | Tokens/update |
|---|---:|---:|---:|---:|---:|
| short | 0-895 | 512 | 32 | 32 | 524,288 |
| long | 896-1787 | 1024 | 16 | 32 | 524,288 |

In both stages `B * T = 16,384`, so the loader advances by the same number of
flat source tokens per micro-batch and the target-token order is identical to
exp000. Optimizer updates, AdamW timescales, peak LR, warmup/warmdown positions,
and total tokens are unchanged. Validation always uses the original T=1024
layout; the transition at update 896 coincides with an existing validation
boundary.

**Profiler prior, not a result.** At T=1024 the measured full update is 4.030 s,
of which FlashAttention forward+backward is about 355 ms. At fixed B*T, halving
T approximately halves attention work but leaves the dominant token-linear
GEMMs unchanged. The resulting prior is about 3.85 s/update at T=512, +4.6%
steady throughput, and roughly 159 seconds gross proxy saving for 896 short
updates. One shape recompile could cost around one minute and consume much of
that saving.

**Mandatory pre-proxy systems gate.** On the target RTX 4090, a short same-host
FineWeb run must measure cold first-call/compile latency, transition recompile
latency, steady T=512 and T=1024 medians, peak VRAM, graph breaks, and recompile
logs. The run records `torch._dynamo.utils.compile_times()` and
`TORCH_LOGS=recompiles`. Kill before proxy if the measured transition cost
consumes the projected saving, T=512 is less than 3% faster in steady state,
the normalised projected net saving is below 1%, or compilation falls back to
eager execution.

**Wall-time definition.** Primary time starts immediately before the first
compiled model call and ends after final validation. It includes lazy initial
compile, the T=512 -> T=1024 recompile, optimizer steps, validation, data
transitions, scheduled checkpoints, and in-loop logging. Offline setup and data
download are excluded. Steady per-stage medians and compile latency are reported
separately. Because two rented cards remain different systems even when BF16
TFLOPS and clocks match, the speed claim uses a same-host timing calibration
and a within-run T=1024 counterfactual; cross-instance wall-time is supporting
context only.

**Success and stopping criteria.**

- Exact 937,426,944 tokens, 1,788 updates, unchanged target order, LR, and
  validation accounting are mandatory.
- Final validation must be `<= 3.601773`, the +0.004 non-inferiority boundary
  versus baseline seed 0.
- Normalised end-to-end wall-time, including compile/recompile, must improve by
  at least 1% against the same-host counterfactual.
- Loss worse than 3.601773, net speedup below 1%, OOM/NaN, recompile storm, or
  eager fallback kills the schedule.
- Passing seed 0 earns proxy seeds 1/2 before any full run.

**Branch and sequencing.** Implementation branch
`exp005/sequence-length-512-1024`. It is independent of exp004-B and may run on
a second instance at the same time, but it is implemented only after exp004-B
has been reviewed, committed, and launched. The two modifications are never
combined in one training run. Any paid gate, smoke, or proxy requires explicit
approval after code review.
