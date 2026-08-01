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

### Frozen baseline components

The reference model is `d12`: vocabulary 50,257, 12 transformer blocks, 12
attention heads, and embedding width 768. Each block is pre-norm residual:
RMSNorm (epsilon `1e-6`) -> causal self-attention -> scaled residual, then
RMSNorm -> GELU MLP (`D -> 4D -> D`) -> residual. The attention projections are
QKV and output linear layers without bias. Attention uses PyTorch SDPA with
`is_causal=True`; on the measured CUDA stack the profiler reports its
FlashAttention forward/backward kernels.

Position information comes from RoPE on Q and K, not learned positional
embeddings. The token embedding and bias-free `lm_head` share one weight matrix.
There are no linear biases, dropout, LayerNorm affine parameters, or learned
position embeddings. Therefore it is GPT-2-sized (the d12 124M-scale shape),
but not a literal GPT-2 implementation.

The optimizer is one AdamW over `self.parameters()` -- no separate decay/no-decay
parameter groups. The full baseline launch uses peak LR `0.0018`, weight decay
`0.1`, and betas `(0.9, 0.95)`. Its fixed training layout is micro-batch
`B=16`, sequence length `T=1024`, accumulation 32, or 524,288 target tokens per
optimizer update.

### RTX 4090 capability and measured operating point

The rented card class is GeForce RTX 4090 (Ada, CUDA capability 8.9) with 24 GB
GDDR6X VRAM. Ada Tensor Cores support BF16, TF32 and FP8 operations; see NVIDIA's
[Ada tuning guide](https://docs.nvidia.com/cuda/archive/13.0.3/ada-tuning-guide/index.html)
and [RTX precision support matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix-1/1.2.html).
Capability is not the baseline configuration: this study uses BF16 autocast;
TF32 and FP8 are not enabled or validated training configurations.

What is measured for the reference run is 9,825 MiB peak allocated memory at
`B=16, T=1024, accum=32`. This leaves apparent VRAM headroom, but it does **not**
establish a maximum feasible micro-batch: that limit has not been measured on the
frozen stack and must not be inferred from `24 GB - 9,825 MiB`. The baseline
uses PyTorch SDPA; FlashAttention is a selected measured kernel path, not a
hardware feature implied merely by the word "Ada".

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
| exp004-B | completed proxy s0, negative | Same proxy tokens exactly once; move all score >=3 tokens into a warmdown-aligned enriched mixture | 3.612175, delta +0.014402; accounting passed | Kill the hypothesis; do not run seeds 1/2 |
| exp005 | stopped at systems gate | T=512 for updates 0-895, then T=1024; keep tokens/update, token order, LR, and validation fixed | T=512 was 2.59% faster steady-state, but compile overhead projected -1.21% net proxy speedup | Systems hypothesis failed; no exp005 proxy |
| exp006 | completed proxy s0 | T=512 for updates 0-383, then T=1024; same tokens/update, source order, LR, and T=1024 validation | 3.600870, delta +0.003097 vs baseline s0; inside the pre-registered grey zone | No positive loss-per-token evidence; do not run seeds 1/2 |
| exp007 | completed proxy s0, grey | Reduce the MLP expansion ratio from 4D to 3D; keep the rest of training fixed | 3.616361, delta +0.018588 vs baseline s0: inside the pre-registered grey zone; valid systems gain remains 10.54% | Do not promote automatically; seeds 1/2 and full remain unapproved pending reassessment of quality and time-to-target |
| exp008 | completed proxy s0, quality killed | Replace 12-head MHA with 12-query-head, 4-KV-head GQA; keep model width and the rest of training fixed | 3.617567, delta +0.019794 vs baseline s0: 0.015794 above the pre-registered KILL threshold | Stop unchanged GQA; no seeds 1/2. Same-host MHA-vs-GQA timing remains unmeasured, so no speedup claim |
| exp009 | completed offline measurement, killed | Residual-Complement Attention: measure and locally perturb the residual-aligned part of each attention update | No layer passed the full checkpoint because the four-batch finite-difference slope was not significant | Stop RCA unchanged; no training experiment |
| exp010 | completed offline measurement, mechanism passed | Selective Spectral AdamW: diagnose persistent dominant directions in AdamW-preconditioned hidden-matrix updates | Shared passing families: attention output, attention V, MLP down, and MLP up | Premise survives; causal top-mode attribution and systems cost remain unresolved before implementation |
| exp011 | stopped at A, treatment killed | Selectively cap only the excessive leading mode of the AdamW-preconditioned attention-V update | A reduced functional concentration but lost too much first-order descent and over-amplified weaker proxy modes | Do not run B or C unchanged; any partial/descent-budgeted cap is a new experiment |
| exp012 | planned nightly funnel | Joint treatment: uniform 3D MLP plus the exact exp001 batch ramp; fixed Adam betas and current WSD | Awaiting same-host benchmark, 256-update health diagnostic, and proxy seed 0 | No full run tonight; promote only after the pre-registered proxy gate |

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

## Current and planned work

### exp004-B — warmdown-aligned high-quality mixture

**Status:** completed proxy seed 0 on 30 July 2026; hypothesis killed.

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

**Implementation and provenance.** The pre-materialised stream builder,
manifest validation, and guarded launcher were implemented from `603f5c0` on
`exp004/late-quality-mixture`. The run used clean commit `d1205aa`, which
contains that implementation. Its input-manifest file SHA-256 was
`640bb47b146fea8207296ad6641e51e157953c24e6c8f66e6cc88e77142360a2`;
the manifest payload SHA-256 was
`2ca0d03b4040b3255a73a90099663bd6416c32c77c460343d5eac5059be069dc`.
Local artifacts are in
`nocap-runs-backup/exp004-B-results/runs/exp004-B-20260730T115702Z/`
and the W&B run is
`https://wandb.ai/m-obukhov-home/nocap-exp004/runs/upadrpkl`.

**Result.** All accounting gates passed: 937,426,944 source and output target
tokens, 1,788 updates, 77,677,516 score >=3 tokens, ten verified output shards,
and the logged transition at update 1,404. The run completed without NaN/OOM.

| Metric | exp004-B | Baseline seed 0 | Difference |
|---|---:|---:|---:|
| Final validation loss | **3.612175** | 3.597773 | **+0.014402** |
| Training time | 7,370.19 s | 7,403.11 s | -0.44% cross-host |
| Throughput | 127,192 tok/s | 126,626 tok/s | +0.45% cross-host |
| Peak VRAM | 9,825 MiB | 9,821 MiB | +4 MiB |
| End-to-end wall time | 7,548.94 s | 7,601.37 s | -0.69% cross-host |

At $0.36/hour, the measured proxy wall-time cost was $0.755 (about $0.75);
offline instance preparation and curriculum construction are excluded. The
runtime differences are hardware context, not algorithmic speedups.

**Dynamics and verdict.** Withholding score >=3 documents did not create a
stable early gap: the validation delta versus baseline was +0.0685 at update
128, approximately zero at 384, -0.0026 at 1,024, and +0.0283 at 1,280.
Immediately after the enriched stage began it was +0.0179 at update 1,408; it
remained worse by +0.0223 at 1,536, +0.0159 at 1,664, and +0.0144 at the end.
The enriched warmdown therefore did not close the gap. Final loss exceeded the
3.601773 kill threshold by 0.010402 and was 3.6 times the pre-registered 0.004
decision margin worse than baseline. Do not run seeds 1/2 or promote this
curriculum. This falsifies the tested timing and mixture, not the general claim
that FineWeb-Edu scores may be useful in another sampling or filtering design.

### exp005 — staged sequence length, 512 to 1024

**Status:** stopped at the mandatory systems gate on 30 July 2026; no proxy
quality run.

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

**Measured systems-gate result.** Commit `4b5d6f9`, PyTorch 2.11.0+cu128,
RTX 4090; artifact
`runs/exp005-shape-gate-host2-static/gate-summary.json`. Both the fixed-T=1024
control and the scheduled run completed 20 updates. The gate measured:

| Metric | Result |
|---|---:|
| Fixed T=1024 steady update | 3,967.44 ms |
| Scheduled T=512 steady update | 3,864.54 ms |
| Scheduled T=1024 steady update | 3,964.76 ms |
| T=512 steady speedup | 2.59% |
| Additional compile/recompile cost | 175.93 s |
| Gross projected proxy saving | 89.79 s |
| Net projected proxy saving | -86.14 s |
| Normalised projected net speedup | -1.21% |

The long-stage control drift was only -0.07%, so the steady timing comparison
was internally consistent. The gate failed both the pre-registered 3%
steady-speedup threshold and the 1% net wall-time threshold. This falsifies the
systems claim that this discrete schedule is a proxy wall-time optimisation.
It does **not** test whether changing sequence length improves loss per target
token: the 20-update gate is too short for a quality conclusion, and the
pre-registered quality condition was non-inferiority rather than improvement.

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

**Verdict and branch.** Implementation branch
`exp005/sequence-length-512-1024`. It is independent of exp004-B and may run on
a second instance at the same time. Stop exp005 at the failed systems gate;
do not reinterpret a future quality run as a continuation of this hypothesis.
The separate loss-per-token question is pre-registered below as exp006.

### exp006 — sequence-length curriculum for loss per token

**Status:** proxy seed 0 completed on 30 July 2026; stopped as inconclusive
without advancing to seeds 1/2.

**Question separated from exp005.** The exp005 systems gate established that
the discrete shape schedule is not a net wall-time optimisation at proxy scale.
It did not establish whether early short-context training changes optimisation
quality. Exp006 therefore treats loss versus target tokens as primary and
records compile-aware wall-time only as a secondary cost.

**Hypothesis.** With identical target-token order, total token budget, effective
batch, optimizer-update count, LR schedule, and fixed T=1024 validation,
training approximately the first quarter of the proxy at T=512 and the
remainder at T=1024 will improve final seed-0 validation loss by at least 0.004
relative to the fixed-T=1024 baseline.

**Literature-grounded schedule choice.** Shortformer kept tokens per batch
constant and trained every model for the same 205 epochs. Its best staged
result used the short length for 50 epochs, about 24% of training; nearby
durations were broadly competitive, while very long short-context stages
degraded. Sequence Length Warmup found gradual growth more robust for aggressive
large-model recipes and warned that remaining short for too long can create a
transition mismatch. A continuously changing T would introduce many compiled
shapes in this single-GPU proxy, so exp006 uses the closest low-complexity test:
one transition after approximately 25% of updates.

The proxy validates every 128 updates. The nearest clean boundary on the early
side of the literature prior is update 384 (21.5% of the run). Using it gives a
T=1024 validation immediately before the transition; update 447 would be
mathematically exact but would make the transition trajectory harder to
interpret.

| Stage | Updates | T | Micro-batch B | Accumulation | Tokens/update |
|---|---:|---:|---:|---:|---:|
| short | 0-383 | 512 | 32 | 32 | 524,288 |
| long | 384-1787 | 1024 | 16 | 32 | 524,288 |

This is 384 short updates and 1,404 long updates. In both stages
`B * T = 16,384`; therefore each micro-batch and optimizer update consumes the
same count of flat source targets as baseline. Targets remain in baseline
order, but their conditioning context is intentionally different: T=512
introduces more context resets. That difference is the treatment, not evidence
that the examples are intrinsically easier.

**Metrics and interpretation.**

- Primary: T=1024 validation loss versus target tokens, including final proxy
  loss after exactly 937,426,944 targets and 1,788 updates.
- Secondary: earliest token count and end-to-end wall-time at which the run
  reaches baseline final loss 3.597773. Wall-time includes lazy compile and
  shape recompilation.
- Diagnostic: validation trajectory before and after update 384, transition
  loss spike, per-stage training loss, gradient norm if already available,
  peak VRAM, compile/recompile report, and W&B continuity.
- The exp005 20-update loss difference is not prior quality evidence and is not
  used in the exp006 decision.

**Success and stopping criteria.**

- `final val <= 3.593773` (at least 0.004 better than baseline seed 0) advances
  to proxy seeds 1 and 2.
- `final val >= 3.601773` kills the quality hypothesis.
- The interval `(3.593773, 3.601773)` is inconclusive and does not
  automatically earn more seeds.
- OOM, NaN, unrecovered loss spike at the transition, incorrect token/order/LR
  accounting, eager fallback, or missing durable metrics invalidates the run.
- A quality win does not retroactively make exp005 a systems win. Report both
  loss-versus-tokens and compile-inclusive time-to-target.

**Branch and run identity.** Implementation branch
`exp006/sequence-length-quality`; W&B group and run directories use `exp006`.
Reuse the reviewed two-shape mechanism, but set the transition explicitly to
update 384. Exp006 remains independent of exp004-B and uses the unchanged
baseline FineWeb stream. Paid proxy seed 0 requires a clean pushed SHA,
recoverable local artifacts plus W&B, and explicit launch approval.

**Proxy seed-0 result.** The run completed from clean commit `efe1d89` with
the exact pre-registered 937,426,944 target tokens and 1,788 optimizer updates.
The uploaded summary, stdout, and metrics agree on the run identity and final
measurements.

| Metric | exp006 seed 0 |
|---|---:|
| Final T=1024 validation loss | 3.600870 |
| Delta versus baseline seed 0 (3.597773) | +0.003097 |
| Delta versus baseline proxy mean (3.599351) | +0.001520 |
| Training time | 7,219.89 s (2.006 h) |
| End-to-end wall-time | 7,395.88 s (2.054 h) |
| Throughput, training-time denominator | 129,839 tok/s |
| Peak allocated memory | 9,820 MiB |
| TorchDynamo unique graphs | 6 |
| Recorded cost | not present in the uploaded artifacts |

The fixed-T=1024 validation trajectory was:
`10.941801` at update 0, `5.891355` at 128, `5.123504` at 256,
`4.797851` immediately before the transition at 384, `4.285623` at 512,
`4.127798` at 640, `4.035853` at 768, `3.959792` at 896, `3.907623` at
1,024, `3.855236` at 1,152, `3.816961` at 1,280, `3.786529` at 1,408,
`3.715722` at 1,536, `3.645185` at 1,664, and `3.600870` at 1,788.
There was no unrecovered validation-loss spike: the first post-transition
validation was lower. Because validation is spaced 128 updates apart, this
cannot exclude a brief transition transient.

The median steady optimizer step was 3,880.88 ms at T=512 and 3,977.19 ms at
T=1024. The first T=1024 training step took 48.82 s, about 44.84 s above its
steady-state median, consistent with the expected new-shape compilation. The
primary 7,395.88 s wall-time includes initial lazy compilation, transition
recompilation, validation, and training. It must not be presented as a speedup
against a baseline run from another rented host.

**Verdict.** The final loss is inside the pre-registered inconclusive interval
`(3.593773, 3.601773)`, but the observed direction is worse rather than better.
The run did not reach the baseline seed-0 final loss at any logged validation
point. Therefore seed 0 provides no positive evidence for the hypothesised
loss-per-token improvement of at least 0.004, and exp006 stops without spending
compute on seeds 1/2. This quality result does not change exp005's failed
systems verdict.

**Artifacts.** Local uploaded bundle:
`nocap-runs-backup/exp006-proxy-20260730/proxy-seed-0/`. W&B:
<https://wandb.ai/m-obukhov-home/nocap-baseline/runs/k8v23t2i>.

### exp007 — narrower MLP (4D -> 3D)

**Status:** systems benchmark completed on 31 July 2026 and passed. Proxy seed
0 completed from the clean branch head on 31 July 2026, but fell in the
pre-registered grey zone. Seeds 1/2 and a full run remain unapproved.

**Hypothesis.** On the same calibrated RTX 4090, with `B=16`, `T=1024`,
accumulation 32, data, optimizer, LR schedule, BF16, `torch.compile`, and all
other architectural components held fixed, reducing the MLP expansion ratio
from 4D to 3D will reduce the median steady-state time of a complete training
update by at least 3% relative to the fixed 4D baseline.

**Causal mechanism and headroom.** Each transformer block contains two MLP
linear projections, `D -> 4D` and `4D -> D`. For `D=768`, exp007 changes the
intermediate width from 3,072 to 2,304, so both projections perform 25% less
forward arithmetic; their input- and weight-gradient matrix multiplications
shrink correspondingly. Across 12 blocks this removes 14,155,776 parameters.
The existing profile attributes 67.07% of CUDA time to `aten::mm`, but does not
identify all matrix shapes. A FLOP-weighted Amdahl estimate gives an optimistic
full-update speedup ceiling of about 8.3%; this is a sizing estimate, not a
prediction.

**Controlled change.** Only the two MLP projection dimensions change. Attention,
GELU, residual scaling, RMSNorm, tied embeddings and `lm_head`, initialization
policy, optimizer settings, effective batch, token order, validation, seed,
compilation policy, and software stack remain unchanged. The variant starts
from a fresh initialization because a 4D checkpoint is shape-incompatible.

**Systems benchmark.** Compare fresh 4D and 3D processes on one sustained-clock
RTX 4090 at the exact baseline shapes. Measure compile latency separately, then
compare post-warm-up complete training updates, each comprising 32
forward/backward microsteps plus `optimizer.step()`. Record median update time,
tokens/s, `aten::mm`, peak allocated memory, compilation/recompilation evidence,
and the presence of the expected CUDA kernels. Verify the parameter-count delta
before interpreting timing.

**Success and stopping criteria.**

- Median steady-state full-update time must improve by at least 3% against the
  same-host 4D control.
- Extra compile cost must not erase the projected saving at proxy or full-run
  scale.
- Eager fallback, a recompile storm, incorrect parameter shapes/counts, OOM, or
  NaN invalidates the benchmark.
- Finite, same-order-of-magnitude loss and gradients on the first identical
  batches are a health gate. Failure kills the architecture candidate, but does
  not by itself falsify the narrower-GEMM timing mechanism.
- A passing benchmark promotes exp007 only to a proxy candidate. Before paid
  proxy seed 0, separately pre-register the loss/non-inferiority and
  compile-inclusive time-to-target criteria.

**Measured systems result.** Implementation commit `bc8ebb5` was compared with
its 4D parent `0430323` on one RTX 4090 using PyTorch 2.11.0+cu128. Each timing
run used 50 updates at the exact baseline shape. After excluding the first five
updates, the median full-update time was 3,984.11 ms for 4D and 3,604.36 ms for
3D: a 10.54% same-host steady-state speedup, comfortably above the 3% gate.
Peak allocated memory fell from 9,822 MiB to 9,030 MiB.

The estimated compile/warm-up excess was 152.03 s for baseline and 165.75 s for
exp007, so the variant paid 13.72 s of additional one-time overhead. Applying
the measured update-time difference to 1,788 proxy updates gives 678.99 s of
gross steady saving and 665.27 s of projected net training-time saving after
that extra overhead. This projection excludes validation/checkpoint costs and
does not assume that 3D reaches the same loss in the same number of tokens.

One post-warm-up complete update from each run was then profiled. Profiler step
time fell from 4.006 s to 3.624 s (9.54%). `aten::mm` fell from 2.626 s to
2.355 s (10.32%, or 271 ms) with the same 4,704 calls; aggregated GELU kernels
fell from 199.39 ms to 147.36 ms, while FlashAttention forward plus backward
was essentially unchanged at 348.47 ms versus 346.44 ms. The narrower GEMMs
and activations therefore explain most of the observed full-step improvement.
Both runs retained FlashAttention, showed the same expected one-time Rotary
cache recompile, and had no eager fallback, recompile storm, OOM, NaN, or
unhealthy early loss.

**Verdict.** The systems hypothesis passed and its causal mechanism is supported
by the profile. The only principal unresolved question is quality retention:
whether the smaller MLP needs few enough additional tokens that its roughly
9-10% systems advantage still improves compile-inclusive time-to-target. Do not
launch a proxy until that claim and its stopping criteria are separately
pre-registered and explicitly approved.

**Proxy seed-0 hypothesis.** At the same 937,426,944 target tokens and 1,788
updates as baseline proxy seed 0, with seed, FineWeb token order, effective
batch, sequence length, optimizer, LR schedule, validation, BF16, compilation
policy, and all non-MLP architecture held fixed, the 3D MLP will finish with
validation loss no more than 0.018 above baseline seed 0 (`3.597773`). Combined
with the measured compile-inclusive systems saving, that is strong enough to
retain the hypothesis that 3D improves expected time-to-target.

The tolerance is an economic gate, not a claim of equal quality per token. The
measured 10.54% steady speedup allows approximately 10% more training time at a
fixed token budget. As a rough translation only, the baseline proxy-to-full
slope of about 0.1555 loss per token doubling maps that margin to roughly 0.022
loss. Because that estimate comes from only two terminal budgets, the pass
boundary keeps about 0.004 loss of protection rather than using 0.022 directly.

**Proxy controls and decision rule.** Run only seed 0 from the clean pushed
`exp007/mlp-3d` SHA. The primary metric is final validation loss at exactly
937,426,944 target tokens; wall time, tokens/s, peak VRAM, compile behaviour,
loss trajectory, gradients, checkpoint integrity, and W&B/run identity are
secondary health and provenance checks.

- Pass: final validation loss <= `3.615773` (delta <= `+0.018000`). This earns
  proxy seeds 1/2, not a full run.
- Grey zone: `3.615773 < loss < 3.623773`. Stop and reassess the trajectory and
  time-to-target model; do not promote automatically.
- Kill: final validation loss >= `3.623773` (delta >= `+0.026000`). Do not run
  seeds 1/2 unchanged.
- Wrong SHA, changed token budget/order, changed schedule or validation,
  fallback/recompile pathology, corrupt or missing checkpoint/summary, OOM,
  NaN/Inf, or clearly unhealthy loss/gradients invalidates the run.

Passing this proxy does not establish late-target benefit: reduced capacity can
hurt more near the full target. Seeds 1/2 are required before deciding whether
the remaining transfer risk deserves a full run.

**Proxy seed-0 result (31 July 2026).** The clean branch-head SHA
`e7c3e9228107f14559bec6c079c83be489b221a7` ran seed 0 for the exact
937,426,944 target tokens and 1,788 updates. The local summary, W&B summary,
and stdout agree on completion and the following values:

| Metric | Result |
|---|---:|
| Final validation loss | 3.616361 |
| Delta vs baseline seed 0 (3.597773) | **+0.018588** |
| Delta vs baseline three-seed mean (3.599351) | **+0.017011** |
| Training time | 6,552.268 s |
| Compile-inclusive wall time | 6,750.093 s |
| Reported throughput | 143,069 tok/s |
| Peak allocated memory | 9,027 MiB |
| `git_dirty` / status | `false` / `complete` |

The final loss is 0.000588 above the PASS boundary (3.615773) and 0.007412
below the KILL boundary (3.623773), therefore this is a **grey-zone result**.
It does not earn proxy seeds 1/2 or a full run automatically. Against baseline
seed 0, validation was lower through step 640, approximately tied at step 768,
and higher from step 896 through the final validation; this is no evidence that
the early advantage persists near the challenge target.

The archived baseline proxy timing is from another rented host. Consequently,
the proxy's training time, wall time, and reported throughput above are
provenance observations only, not a measured exp007 speedup. The valid
same-host systems benchmark and profiler remain the evidence for the 10.54%
steady-update improvement. No monetary cost or durable W&B URL/export was
captured in the local backup.

**Artifacts.** Local profiler copies are under `profiles/exp007-gate/`; remote
timing and profiler runs used `runs/exp007-gate/`. Recorded monetary cost is not
available in the supplied artifacts.

**Branch and run identity.** Implementation belongs on `exp007/mlp-3d`; future
run names, W&B group, and result directories use `exp007`. Do not combine GQA,
activation changes, LR changes, or compensating width changes in this first
causal test.

### exp008 — grouped-query attention (12 query heads, 4 KV heads)

**Status:** proxy seed 0 completed on 31 July 2026 despite the missing valid
systems comparison. Its pre-registered quality gate was killed. The first
systems attempt did not execute the exp008 commit and remains invalid as GQA
evidence.

**Hypothesis.** On the same calibrated RTX 4090, with `B=16`, `T=1024`,
accumulation 32, data, optimizer, LR schedule, BF16, `torch.compile`, and all
other architectural components held fixed, replacing 12-head multi-head
attention with 12 query heads and 4 shared key/value heads will reduce the
median steady-state time of a complete training update by at least 3% relative
to the fixed 12-KV-head baseline.

**Causal mechanism and measured headroom.** GQA groups three query heads around
each key/value head. Query width remains 768, while key and value widths each
fall from 768 to 256. The combined QKV projection therefore narrows from 2,304
to 1,280 outputs, a 44.4% reduction in that projection's forward arithmetic and
corresponding backward matrix multiplications. Across 12 blocks this removes
9,437,184 parameters. It does not shrink the attention output projection,
`lm_head`, MLPs, or the number of query-by-key attention scores. The baseline
profile places only about 8.9% of CUDA time in FlashAttention and does not
isolate QKV projection GEMMs, so the 3% full-update gain remains an empirical
gate rather than a FLOP-derived expectation.

**Controlled change.** Only the number of key/value heads and the resulting QKV
projection width change. Model width, 12 query heads, head dimension, attention
output shape, MLP, normalization, embeddings and `lm_head`, initialization
policy, effective batch, token order, validation, seed, optimizer, LR schedule,
and compilation policy remain unchanged. PyTorch SDPA receives
`enable_gqa=True`; the benchmark must verify that it retains the fused CUDA
attention path without materialized K/V repeats or hidden copy overhead.

**Systems benchmark and falsifier.** Run fresh baseline and GQA processes on
one sustained-clock RTX 4090 at the exact baseline shapes. For exp008, first
verify a clean checkout of full SHA
`3da417ee1ff9e2f965584adab84ae4ea23e3f3a9`. Use 50 updates, measure initial
compile/warm-up separately, and compare the median of complete optimizer updates
11-48 with the same-host baseline. Profile a post-warm-up full update and record
tokens/s, `aten::mm`, FlashAttention forward/backward, any repeat/copy kernels,
peak allocated memory, compilation/recompilation evidence, and early loss
health.

**Success and stopping criteria.**

- Median steady-state full-update time must improve by at least 3% against the
  same-host MHA control.
- Extra compilation cost must not erase the projected saving at proxy or
  full-run scale.
- The expected fused attention backend must remain active without materialized
  K/V replication large enough to erase the projection saving.
- Wrong SHA, dirty source, eager fallback, a recompile storm, incorrect tensor
  or parameter shapes, OOM, NaN, or unhealthy early loss invalidates the run.
- A systems pass would normally be required before proxy promotion. The
  explicitly authorised exception below does not repair the missing speed
  evidence or permit a time-to-target win to be claimed from this run alone.

**Proxy seed-0 hypothesis and decision rule.** With the baseline proxy token
budget of 937,426,944 targets, unchanged data order, optimizer, LR schedule,
effective batch, validation, and seed 0, 12Q/4KV GQA will retain enough learning
efficiency to remain competitive with the baseline seed-0 final validation loss
of 3.597773. The explicitly approved run bypasses the unresolved systems gate;
it measures quality and produces GQA timing, but cannot by itself establish a
same-host speedup against MHA.

- Final validation loss `<= 3.593773` passes the seed-0 quality gate and permits
  consideration of seeds 1/2 after the missing systems comparison is resolved.
- Final validation loss `>= 3.601773` kills the quality hypothesis and stops
  exp008 without more seeds.
- The interval `(3.593773, 3.601773)` is inconclusive and does not automatically
  earn more compute.
- Wrong SHA, dirty tracked source, token/order/LR mismatch, OOM, NaN, missing
  durable artifacts, or failed W&B completion invalidates the proxy.
- Compile-inclusive time-to-target remains unproven unless GQA is also compared
  with MHA on the same calibrated host.

**Proxy seed-0 result (31 July 2026).** The clean branch-head SHA
`2a5cf204f538bf0592f94defaca983c9f162876e` ran seed 0 for the exact
937,426,944 target tokens and 1,788 updates. The local summary, W&B summary,
and stdout agree on completion and the following values:

| Metric | Result |
|---|---:|
| Final validation loss | 3.617567 |
| Delta vs baseline seed 0 (3.597773) | **+0.019794** |
| Delta vs baseline three-seed mean (3.599351) | **+0.018216** |
| Training time | 6,866.870 s |
| Compile-inclusive wall time | 7,185.187 s |
| Reported throughput | 136,514 tok/s |
| Peak allocated memory | 9,124 MiB |
| `git_dirty` / status | `false` / `complete` |

The final loss is 0.015794 above the KILL boundary (3.601773). The quality
hypothesis is therefore **killed**: do not run exp008 seeds 1/2 unchanged.
After the early step-0 difference, every matched validation from step 128
through the final validation is worse than baseline seed 0, so the trajectory
provides no late-quality rescue.

The archived baseline proxy and exp008 proxy ran on different rented hosts.
Their timings and throughputs are not a causal MHA-versus-GQA comparison and
must not be reported as a speedup. A valid same-host MHA control, a full-model
GQA profiler, and isolated total compile time are still missing. The local
backup also lacks the proxy `config.json`, `metrics.jsonl`, checkpoint, and a
durable W&B URL/export; no monetary cost was recorded.

**Invalid first benchmark attempt.** The directory named
`runs/exp008-gqa-4kv-profile-20260731T123134Z` reported a 3,607.49 ms median for
updates 11-48, a 176,651.75 ms first update, 9,448 MiB peak memory, 2.372 s of
`aten::mm`, and fused FlashAttention kernels. However, its canonical
`summary.json` records clean commit
`bc8ebb536e9af8813a1ba003170936e6bd79b058`, which is the exp007 3D-MLP
implementation. The run is therefore a repeat measurement of exp007 despite
its directory name. None of these timing, memory, profiler, or loss values are
evidence about GQA and they must not be used to promote exp008.

A separate forced fused-SDPA GQA smoke test passed, which is limited evidence
that the tensor shapes and backend call are compatible. It is not a compiled
full-model timing result and does not satisfy the systems gate.

**Branch and run identity.** The reviewed GQA implementation is commit
`a40345e`; runnable branch head
`2a5cf204f538bf0592f94defaca983c9f162876e` adds dedicated benchmark and
one-shot proxy launchers without a pre-run lifecycle mutation. All are on
`exp008/gqa-4kv` and pushed to origin. The
invalid downloaded artifacts are retained locally under
`profiles/exp008-gate/exp008/` for provenance. The proxy must record the full
branch-head SHA, not merely an exp008 directory name.

## exp009 — Residual-Complement Attention offline gate

**Status:** completed offline measurement; mechanism killed. No optimizer
update or proxy training was performed or authorised.

**Hypothesis.** At the canonical exp000 proxy and full seed-0 checkpoints, the
post-`c_proj` attention update contains a non-random component parallel to the
block residual stream. Removing a small learned-sign amount of only that
component will have a locally favourable loss derivative in at least one
consistent transformer layer. This is a prerequisite for Residual-Complement
Attention; it is not evidence that a trained gate improves time-to-target.

**One measured intervention.** Freeze all model weights. For each attention
block, decompose its output `a` against that block's pre-attention residual
input `u` and evaluate `a - tanh(alpha_l) * proj_u(a)`, with all scalar layer
gates initialized to zero. Use fixed held-out training batches continued from
each checkpoint. Record residual alignment, a sequence-shifted residual null,
the exact gate gradient at zero, and a centred finite-difference slope. No
checkpoint state is written and no optimizer step is taken.

**Success and stopping criteria.** Evaluate 16 fixed micro-batches at each of
the canonical exp000 proxy and full seed-0 checkpoints; use the first four for
the centred perturbation check at gate values `+/-0.05`.

- A layer passes one checkpoint when mean alignment excess over the shifted
  null is positive by more than two standard errors, mean gate gradient plus
  two standard errors is below zero, at least 75% of batch gradients are
  negative, and the centred finite-difference slope plus two standard errors
  is below zero.
- The mechanism gate passes only if at least one identical layer passes at
  both proxy and full checkpoints. A full-only pass is weak/inconclusive.
- If no layer passes at the full checkpoint, stop RCA unchanged. Do not build
  a training experiment from alignment alone.
- Wrong checkpoint identity or baseline arguments, changed batch continuation,
  NaN/Inf, a non-zero weight delta, or disagreement in sign between autograd
  and finite differences invalidates the measurement.

Primary artifacts are the full branch SHA, checkpoint identities and arguments,
per-batch/per-layer measurements, aggregate decision, elapsed time, GPU/runtime
metadata, and stdout. Any later trainable RCA experiment requires a new
pre-registration and a separate paid-compute approval.

**Offline result (1 August 2026).** Clean branch SHA
`44eba56557e772d7221f0592326a8b7ce82854a5` evaluated the canonical exp000
proxy checkpoint at step 1,788 and full checkpoint at step 4,768 on one RTX
4090 with PyTorch 2.11/CUDA 12.8. Both checkpoints retained exact parameter
equality. The complete measurement took 42.60 seconds and returned **KILL**:
neither checkpoint had a passing layer, so there was no shared passing layer.

The result is more specific than “no signal.” Later layers often had positive
alignment excess and negative mean autograd gate gradients. For example, layer
7 passed the alignment, gradient-significance, and negative-fraction checks at
both checkpoints; its mean finite-difference slope was also negative at both.
However, no layer's four-batch centred finite-difference slope was below zero
by two standard errors. That pre-registered confirmatory check failed for every
layer and therefore controls the verdict. The estimated local effects were also
small: layer-7 mean gate gradients were about `-4.15e-4` at proxy and `-6.00e-4`
at full.

This is a valid formal kill, not proof that residual-aligned components never
matter. It says the unchanged RCA proposal did not produce a sufficiently
large, independently confirmed local loss benefit to justify implementation or
paid training. Do not increase the finite-difference sample after seeing this
result and retroactively relabel exp009 as passing.

Artifacts are retained under
`profiles/exp009-gate/exp009-rca-gate-20260801T125454Z/` and remain untracked.

## exp010 — Selective Spectral AdamW offline gate

**Status:** completed offline measurement; mechanism gate passed. No capped
optimizer or proxy training was performed or authorised.

**Hypothesis.** At the canonical exp000 proxy and full seed-0 checkpoints,
AdamW's bias-corrected preconditioned update matrices for at least one repeated
hidden-matrix family are dominated by a leading singular direction, and the
weaker right-singular subspace persists across adjacent data updates. This is a
prerequisite for selectively limiting only the dominant mode while retaining
weaker directions; it is not evidence that spectral capping trains better.

**One measured intervention.** Load the exact model and AdamW states, freeze
parameter values, and replay eight consecutive baseline effective batches per
checkpoint with learning rate and weight decay set to zero. Reconstruct the
bias-corrected preconditioned AdamW matrix before each zero-LR optimizer step.
Measure hidden attention and MLP matrices only; split the fused attention input
projection into Q, K, and V, and exclude embeddings and the tied `lm_head`.
Record spectral energy, cap-fraction proxy `(sigma1 - sigma2) / ||A||_F`, the
same statistics after applying the update to sampled module inputs, and overlap
of singular modes 2 through 8 between adjacent effective batches.

**Success and stopping criteria.** Use the baseline `B=16`, `T=1024`, and 32
micro-steps per effective update. Capture at most 2,048 module-input rows per
effective update. Estimate eight singular modes with a deterministic block
power method and retain exact Frobenius norms.

- A matrix family passes one checkpoint when its median cap-fraction is at
  least 0.05, its median functional leading-mode energy is at least 0.10, and
  its median adjacent-update overlap for modes 2--8 is at least the larger of
  0.05 and three times the dimensional random-subspace expectation.
- The mechanism gate passes only if an identical matrix family passes at both
  proxy and full checkpoints. A full-only pass is weak/inconclusive.
- If no family passes at the full checkpoint, stop this optimizer idea
  unchanged. Do not infer usefulness from singular values without the
  functional and temporal checks.
- Wrong checkpoint identity or baseline arguments, changed batch continuation,
  parameter movement above numerical equality, NaN/Inf, missing optimizer
  state, or a failed spectral self-test invalidates the measurement.

Primary artifacts are the full branch SHA, checkpoint identities and arguments,
per-update/per-matrix metrics, family aggregates and decision, elapsed time,
GPU/runtime metadata, and stdout. Any capped-update implementation requires a
new pre-registration and a separate paid-compute approval.

**Offline result (1 August 2026).** Clean branch SHA
`f08a754240a0f3618ee4f96754de60d718f23e29` replayed eight frozen-weight
effective updates at each canonical exp000 checkpoint on the same RTX 4090 and
runtime as exp009. The self-test passed, all optimizer states were present,
checkpoint parameters remained exactly unchanged, and the complete measurement
took 130.97 seconds. The pre-registered mechanism gate **passed** with four
shared matrix families:

| Family | Median cap fraction, proxy / full | Functional leading energy, proxy / full | Weak-subspace overlap, proxy / full |
|---|---:|---:|---:|
| Attention output | 0.0587 / 0.1206 | 0.2451 / 0.6230 | 0.7913 / 0.7520 |
| Attention V | 0.2283 / 0.1009 | 0.8824 / 0.8001 | 0.7704 / 0.7423 |
| MLP down | 0.1210 / 0.2619 | 0.5615 / 0.8408 | 0.6609 / 0.6246 |
| MLP up | 0.1029 / 0.0832 | 0.7735 / 0.8157 | 0.7020 / 0.7042 |

Attention Q passed proxy but missed the full cap-fraction threshold narrowly
(`0.0495 < 0.05`); attention K failed that threshold at both checkpoints. Of
the shared families, attention V combines the strongest proxy cap fraction,
strong functional concentration at both checkpoints, stable weak modes, and
the smallest candidate matrices. It is therefore the leading family for a
subsequent isolated falsifier, not yet an approved optimizer treatment.

**Interpretation limit.** The gate measured the leading spectrum of the AdamW
update `A` and, separately, the leading spectrum of its sampled functional
effect `X A^T`. It did not measure how much of `X A^T` is caused specifically
by the rank-one component that the proposal would remove, nor whether that
component has favourable or harmful alignment with the current loss gradient.
Activation covariance can make `X A^T` concentrated even when the candidate
rank-one cap is not the cause. Thus the pass establishes spectral dominance and
temporal persistence, but not that capping improves descent, loss per token, or
time-to-target.

Before implementing a capped optimizer, a new offline falsifier should measure
the functional contribution and first-order loss contribution of
`(sigma_1 - sigma_2) u_1 v_1^T` itself, beginning with attention V. Any later
implementation must then pass an exact-shape full-update systems gate with no
more than roughly 1% compile-inclusive overhead. No proxy or training run is
authorised by this result.

Artifacts are retained under
`profiles/exp010-gate/exp010-spectral-adamw-gate-20260801T125712Z/` and remain
untracked.

## exp011 — Attention-V Selective Spectral AdamW

**Status:** stopped at A; the full-gap attention-V treatment is killed. B and C
were not run and remain unauthorised. Any partial or descent-budgeted cap is a
new treatment requiring a new experiment identity and pre-registration.

**Treatment and causal hypothesis.** For each layer's V rows inside the fused
attention input projection, reconstruct AdamW's bias-corrected preconditioned
update `A`. Estimate its first two singular modes and define

```text
C = (sigma_1 - sigma_2) * u_1 v_1^T
A_cap = ||A||_F / ||A - C||_F * (A - C)
```

All Q/K rows, attention output matrices, MLP matrices, embeddings, AdamW
moments, weight decay, LR schedule, data order, model, and evaluation remain
unchanged. The hypothesis is that the excessive V leading mode causes a
disproportionate functional movement while contributing little unique
first-order descent; removing only its excess and matching update norm will
improve loss per token without Muon's all-mode flattening.

**Main transfer risks.** The leading mode may be useful signal, not domination;
activation covariance may have caused exp010's concentrated `X A^T`; Frobenius
matching may amplify weaker noisy modes; a warm-started rank-two estimate may
lag or misorder close modes; modifying only a slice of fused QKV state may be
incorrect; and extra optimizer kernels may erase any quality gain in the
compile-inclusive objective.

### exp011-A — causal component attribution

Freeze weights and replay the same eight effective batches from both canonical
exp000 proxy and full seed-0 checkpoints. For every layer's attention-V update,
compute converged rank-two `C`, norm-matched `A_cap`, and sampled functional
effects before and after capping. Record:

- relative reduction in leading energy of `X A^T`;
- first-order descent retention `<G, A_cap> / <G, A>`;
- Frobenius rescale factor and the fraction of records retaining at least 95%
  of predicted descent.

A passes one checkpoint only when the median functional leading-energy
reduction is at least 10%, median descent retention is at least 98%, at least
75% of layer/update records retain 95% of descent, and the median norm rescale
is no greater than 1.05. It passes overall only at both proxy and full. A
non-positive total descent denominator, parameter movement, missing state,
NaN/Inf, wrong checkpoint provenance, or failed synthetic cap test invalidates
the measurement. Failure stops exp011 before implementation.

**A result (1 August 2026).** Clean branch SHA
`b972f54b63bc3d5c47ab532e9aebbc7d2c5405f0` replayed eight effective batches
at each canonical exp000 checkpoint on one RTX 4090 with PyTorch 2.11/CUDA
12.8. The synthetic cap test passed, both checkpoint parameter sets remained
exactly unchanged, and 96 layer/update records were measured per checkpoint in
114.63 seconds. The overall decision was **KILL**.

| A metric | Proxy checkpoint | Full checkpoint | Required |
|---|---:|---:|---:|
| Median functional leading-energy reduction | 0.2794 | 0.1407 | >= 0.10 |
| Median first-order descent retention | 0.8817 | 0.9583 | >= 0.98 |
| Fraction retaining at least 95% descent | 0.1250 | 0.6458 | >= 0.75 |
| Median Frobenius norm rescale | 1.0911 | 1.0276 | <= 1.05 |

The cap did reduce functional concentration, so exp010's spectral observation
was real. However, at the proxy checkpoint it removed a median 11.8% of
predicted descent and norm matching amplified the remaining modes by 9.1%; only
12 of 96 records retained at least 95% descent. At the full checkpoint it still
removed a median 4.2% of descent, and only 62 of 96 records met the 95%
retention threshold. The leading mode therefore contains substantial useful
signal rather than being an almost-free excess component.

This falsifies the pre-registered full-gap rule `sigma_1 -> sigma_2`. Do not run
the implementation benchmark B or proxy C unchanged. A proxy could in principle
show a longer-horizon regularisation effect despite this local result, but that
was not the exp011 hypothesis and overriding the stopping gate after observing
A would be exploratory post-hoc work. A partial or descent-budgeted cap may be
formulated separately, but cannot be relabelled as a continuation or pass of
exp011.

Artifacts are retained under
`profiles/exp011-gate/exp011-a-20260801T133033Z/` and remain untracked.

### exp011-B — implementation correctness and systems gate

Implement the same cap with a persistent warm-started rank-two subspace: four
block-power iterations on its first use and one iteration on later optimizer
updates. Store the subspace in optimizer state so checkpoint resume is exact.
Apply ordinary AdamW first, then correct only V rows to the norm-matched capped
preconditioned update; verify all non-V updates match the AdamW control.

Run fresh control and treatment processes for 50 exact-shape baseline optimizer
updates on one sustained-clock RTX 4090. Measure initial compile/warm-up,
complete-update times, updates 11--48, peak memory, compile/recompile evidence,
early loss health, and the treatment's cap/rescale diagnostics.

- Median complete-update overhead for updates 11--48 must be at most 1.0%.
- Projected total optimizer overhead across the 1,788-update proxy must be at
  most 1.0%; no hidden model recompile is allowed.
- Synthetic update equivalence, zero-gap identity, V-only correction, optimizer
  state round-trip, and deterministic resume tests must pass.
- OOM, NaN/Inf, non-V mismatch, dirty/wrong SHA, or timing noise too large to
  resolve a 1% boundary invalidates B. Failure stops before C.

### exp011-C — proxy seed 0

Only after explicit review and approval of A and B, train from scratch for the
exact exp000 proxy budget: seed 0, 1,788 updates, 937,426,944 target tokens,
unchanged FineWeb order, `B=16`, `T=1024`, accumulation 32, LR schedule,
validation, checkpointing, and W&B project. The sole treatment is attention-V
top-mode capping as implemented and frozen after B.

- Final validation loss `<= 3.593773` passes the seed-0 quality gate and permits
  consideration of seeds 1/2.
- Final validation loss `>= 3.601773` kills exp011 unchanged.
- The interval between those thresholds is inconclusive.
- Compile-inclusive time-to-target remains unproven until an eventual full run
  is compared with a same-host baseline. C cannot by itself establish the
  challenge objective `val_loss <= 3.3821`.
- Wrong SHA, changed treatment after B, token/order/LR mismatch, OOM, NaN,
  missing durable local artifacts, or failed W&B completion invalidates C.

## exp012 — 3D MLP plus batch-ramp combination

**Status:** planned. This is an intentionally joint treatment, not a clean
ablation. No full run is authorised by this row.

**Hypothesis.** At the exact exp000 proxy token prefix, combining the measured
3D-MLP systems saving with the exact exp001 `131,072 -> 524,288` effective-batch
ramp over the first 50% of tokens will preserve enough of exp001's loss/token
advantage to offset exp007's capacity loss and finish at least `0.004` below
the baseline proxy seed-0 loss, while retaining a material same-host throughput
advantage. Adam betas remain `(0.9, 0.95)` in step time and the LR remains the
current WSD with exp001's square-root batch scaling.

**Night funnel.** First run a 50-update exact-shape systems benchmark and a
separate 256-update no-clipping health diagnostic beginning at accumulation 8.
The diagnostic records global and maximum gradient norm, non-finite counts,
parameter norm, sampled update/weight ratio, and loss jumps locally and in
W&B. It is not timing evidence. If both gates are valid, run proxy seed 0 for
exactly `937,426,944` target tokens; the changing batch determines the number
of optimizer updates.

- Compile/eager failure, OOM, NaN/Inf, a non-finite gradient, or a reproducible
  loss/gradient spike more than `10x` the diagnostic median kills the treatment
  before proxy. A lone finite outlier is reported but does not auto-kill.
- Proxy loss `<= 3.593773` passes the seed-0 quality gate. Loss
  `>= 3.601773` kills the combination; the interval is inconclusive.
- Same-host timing must be reported separately from quality. The exp007
  `10.54%` result is a prior, not evidence for this new SHA.
- Wrong SHA, dirty tracked files, changed token prefix, missing artifacts or
  failed W&B artifact upload invalidates a stage and stops the nightly suite.
