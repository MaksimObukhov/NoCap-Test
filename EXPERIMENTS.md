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
| exp012 | completed proxy s0, pass | Joint treatment: uniform 3D MLP plus the exact exp001 batch ramp; fixed Adam betas and current WSD | 3.581923, delta -0.015850 vs baseline s0 (-8.8 sigma); +10.18% same-host benchmark speedup; effects independent of exp001/exp007 (interaction +0.0025, 1.4 sigma) | Passed the pre-registered proxy gate; leads baseline at every validation checkpoint through 937M tokens; promoted to full run seed 0, pending Max's approval before paid compute |
| exp013 | completed proxy s0, null result | Replace GELU with squared ReLU in the otherwise unchanged 4D MLP | 3.597677, delta -0.000096 vs baseline s0 (-0.05 sigma, exact null); systems speedup +0.34%, `aten::mm` time unchanged (-0.5%) | No loss/token or systems benefit; the GELU activation kernel is already fully fused by Inductor, so the model stays GEMM-bound. Close the hypothesis; do not run seeds 1/2 |
| exp014 | completed proxy s0, killed | Depth-shaped MLP: 2.5D in layers 0–3, 3D in 4–7, 3.5D in 8–11 | 3.624233, delta +0.026460 vs baseline s0 (+14.7 sigma), +0.007872 worse than exp007 uniform 3D (+4.4 sigma); benchmark +0.17% vs uniform-3D control | Killed by the pre-registered proxy gate. Per-layer gradient RMS at update 256 shows the allocation prior was backwards: layers 8-11 (given 3.5D) have below-average grad RMS while layers 0-3 (given 2.5D) have 5-8x the mean. Do not run seeds 1/2 |
| exp015 | health-killed, kill contested | Replace the 3D GELU MLP with a 2D SwiGLU MLP at equal leading MLP parameters/FLOPs | Health diagnostic killed on 3 gradient spikes and 2 loss spikes (pool median over warmup+post-warmup); benchmark -1.225% vs uniform-3D control, within the -2% threshold; proxy never ran | All spikes (updates 80, 86, 88) fall inside the 96-update warmup and fully recover within one update; post-warmup max grad norm (1.92) is in line with exp012-014. The pooled-median health gate conflates warmup transients with steady-state instability. See exp015 Results below for the reanalysis and the reopening condition |
| exp016 | completed A, mechanism killed | Descent-Budgeted Multi-Directional AdamW on attention-V updates | Exact frozen-weight replay passed provenance, immutability, descent, norm, activity, and positive-descent checks, but median functional leading-energy reduction was only 0.6456% at the proxy checkpoint and 1.7315% at full versus the registered 10% minimum | Scientific kill at A; B and C were correctly skipped. The 1% descent budget makes the treatment safe but too weak to materially change functional concentration |
| exp017 | completed proxy s0, inconclusive | Selective weight decay: exempt only the tied `wte`/`lm_head` matrix from WD 0.1, on the full exp012 stack | Accounting passed; 3.580403, delta -0.001520 vs exp012 (-0.85 sigma); 937,426,944 tokens and 2,549 optimizer updates | Inside the pre-registered inconclusive band. Do not promote or rerun alone; retain as a possible interaction term for a future small-batch combination |
| exp018 | completed forward feasibility measurement, pass | FP8 compute path for the `lm_head` vocab projection | BF16 forward median 10.724 ms vs FP8 cast+GEMM 4.572 ms; forward-only projected full-step gain 5.28% on RTX 4090 | Authorises a separate integration benchmark only after the architecture winner. Backward, tied-weight integration, padded-vocabulary masking and training quality remain unvalidated |
| exp019 | attempted v3, infrastructure-invalid | Fused 2D SwiGLU (one `Linear(768->3072)` + `chunk(2)`) plus the exact exp012 batch ramp, with gradient clipping at 10 | Accounting passed at 109,376,256 parameters and exact fused/reference equivalence; treatment benchmark failed before training because its validation path remained relative | No systems or proxy result. Retry later with absolute train and validation globs; this remains the next architecture candidate if exp021 does not justify a new combination |
| exp020 | completed proxy s0, killed | Cosine LR schedule against the baseline trapezoid WSD, unchanged 4D architecture and flat 524,288 batch | 3.676178, delta +0.078405 vs baseline (+43.6 sigma); exact 937,426,944-token budget | Strong negative result for this horizon: close cosine unchanged and do not run seeds 1/2 |
| exp021 | completed proxy and full s0, full target miss | Uniform 3D GELU MLP plus an aggressive token-linear effective-batch ramp from 16,384 to 262,144 over the first 50% of tokens; LR scaled against the fixed 524,288-token reference batch | Proxy 3.567184; full 3.387356 at 2,700,083,200 tokens in 19,500.15 s training / 19,966.26 s wall time; observed 138,465 tok/s | The full missed the 3.3821 target by 0.005256 despite 8.01% more tokens than baseline. The early advantage vanished near 1.1B tokens; do not rerun this schedule unchanged |
| exp022 | completed proxy s0, killed | On the exact exp021 schedule combine fused 2D SwiGLU, global clip 10, and selective no-WD for the tied embedding/head | 3.607177, +0.039993 worse than exp021; 1.52% more training time and 1.50% lower throughput; clip 10 never activated | Failed both winner routes. Do not run full or promote SwiGLU/selective-no-WD on this stack; the joint run cannot attribute the loss regression between its additions |
| exp023 | attempted integration benchmark, infrastructure-invalid | FP8 compute only for the padded tied lm_head of the selected exp021 architecture | No benchmark result: the runner rejected the valid `exp021` winner before measurement; TorchAO also emitted binary-load warnings | Fix winner transport and the pinned TorchAO runtime, then repeat the isolated integration benchmark after the final architecture is selected. No FP8 quality or speed claim exists yet |
| exp024 | implementation ready; full s0 authorised | Uniform 3D GELU MLP with an absolute-token staircase batch schedule 64K -> 128K -> 256K -> 524K; fixed 524K LR reference | Branch `exp024/staircase-batch-ramp` at `9e86007` freezes one 2,700,083,200-token full; the switch to 524K is fixed at 939,524,096 tokens | Max authorised the direct paid full on 3 August 2026. Run once with W&B; do not tune boundaries or launch another algorithmic full afterward if it fails |

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

**Results (night suite `night-20260802-v2`, 2 August 2026).** SHA
`7423fbc8`, clean tree, provenance and W&B artifacts complete for benchmark,
health, and proxy stages.

| Result | Value |
|---|---:|
| Benchmark speedup vs baseline (median steady-state tok/s) | +10.18% |
| Health diagnostic | pass; 0 grad spikes, 0 loss spikes, max grad norm 4.72 |
| Final proxy validation loss | **3.581923** |
| Delta vs baseline seed 0 (3.597773) | **-0.015850** (-8.8 sigma) |
| Proxy time-to-baseline-final-loss | 6582.7s vs baseline's 7403.1s (-11.1% train time) |

The benchmark and health gates passed as pre-registered; the proxy loss is
well below the `3.593773` pass threshold. exp012 led the baseline learning
curve at all 14 validation checkpoints from 67M to 937M tokens, though the
absolute gap decayed from 0.71 at 67M tokens to 0.016 at 937M (see the
factor decomposition below for why this decay is expected).

**Factor decomposition.** At the identical 937M-token proxy checkpoint:

```text
baseline 4D, flat batch     3.597773
exp001  4D + ramp alone     3.560791   delta -0.036982 (-20.6 sigma)
exp007  3D, flat batch      3.616361   delta +0.018588 (+10.3 sigma)
exp012  3D + ramp (joint)   3.581923   delta -0.015850 (-8.8 sigma)
additive prediction         3.579379   delta -0.018394
interaction (obs-additive)  +0.002544  (+1.4 sigma)
```

The ramp and 3D-MLP effects are statistically independent (interaction is
within 1.5 sigma of zero): the batch ramp buys back exp007's capacity loss
without a joint-treatment penalty or bonus. The ramp's own systems cost is
small: bucketing throughput by accumulation level during the ramped proxy
run gives +1.59% wall-time overhead versus running the entire proxy at the
terminal accumulation=32, against 2549 optimizer updates instead of 1788
(+42.6%).

**Promotion decision.** exp012 is the leading full-run candidate. The proxy
gap versus baseline decays roughly as a power law in tokens (exponent
estimates from 1.6 on the full 14-point curve to 3.0 on the last 4
warmdown-aligned points), so the proxy result does not by itself establish
that exp012 beats the `3.3821` full-run target: extrapolated gap-at-2.5B
ranges from 0.0008 to 0.0047, i.e. plausibly inside or outside the +0.0044
margin baseline carries at 2.5B tokens. Only a full run resolves this. No
full run has been authorised or started as of this entry; it requires Max's
explicit approval per the paid-compute rule in `AGENTS.md`.

## exp013 — squared ReLU activation

**Status:** completed through proxy seed 0 on 2 August 2026; null result. No
full run is authorised by this row.

**Hypothesis.** Replacing only `GELU(x)` with `ReLU(x)^2` in every unchanged
4D MLP will either improve loss per token through a sparse, high-amplitude
nonlinearity or reduce activation-kernel time enough to improve proxy
time-to-quality. The two claims are evaluated separately: the baseline
profiler's roughly 5% GELU share is only an Amdahl ceiling, not a predicted
speedup.

**Night funnel.** Run a fresh same-host baseline calibration, then a 50-update
exact-shape treatment benchmark. Unless it is invalid or more than 3% slower,
run a separate 256-update no-clipping diagnostic and then proxy seed 0 with the
baseline token order, batch, WSD, AdamW, and token budget.

- Benchmark speed alone does not pass or kill the quality mechanism. A valid
  slowdown above 3%, compile/eager failure, or non-finite values stops before
  proxy unless the 256-update diagnostic shows a pre-registered quality signal.
- Proxy loss `<= 3.593773` passes the quality gate. A loss within the baseline
  grey zone may remain interesting only if the same-host complete-step gain is
  at least 2%; loss `>= 3.601773` kills the unchanged treatment.
- The health gate and provenance/W&B invalidation rules are identical to
  exp012. The diagnostic timing is never used as systems evidence.

**Results (night suite `night-20260802-v2`, 2 August 2026).** SHA
`be085f3f`, clean tree, provenance and W&B artifacts complete.

| Result | Value |
|---|---:|
| Benchmark speedup vs baseline (median steady-state tok/s) | +0.34% |
| Health diagnostic | pass; 1 grad spike (step 1, warmup), 0 loss spikes |
| Final proxy validation loss | 3.597677 |
| Delta vs baseline seed 0 (3.597773) | **-0.000096** (-0.05 sigma) |

This is a measured null on both claims, not an inconclusive result: the
delta is two orders of magnitude below the seed-to-seed noise floor
(sample std 0.0018). The profiler explains the systems null directly —
`aten::mm` self-CUDA time is unchanged from baseline (2.739s vs 2.753s,
-0.5%), because the GELU-vs-ReLU^2 activation is already fully fused into a
single Triton kernel by `torch.compile`, and the model spends 67% of its
step in GEMM regardless. The pre-registered 5% GELU-kernel Amdahl ceiling
was never reachable at this compile configuration. Close the hypothesis;
do not run seeds 1/2 or a full run.

## exp014 — depth-shaped 3D-average MLP

**Status:** planned. No full run is authorised by this row.

**Hypothesis.** Holding the mean expansion at 3D, allocating widths
`2.5D / 3D / 3.5D` to layers `0–3 / 4–7 / 8–11` will retain nearly the full
systems gain of uniform 3D while recovering at least `0.004` proxy loss through
more capacity in later blocks, where the residual representation is more
task-specific. The shallow-to-deep increase is a testable allocation prior,
not an established transformer law; per-layer gradient and update ratios are
recorded so the result can challenge that prior.

**Night funnel.** Run exact parameter/FLOP accounting, a 50-update benchmark,
and a 256-update no-clipping diagnostic with per-layer normalized gradient and
update/weight ratios. If valid, run proxy seed 0. Uniform 3D is the principal
capacity- and average-FLOP control; exp000 remains the quality reference.

- The benchmark passes when complete-step throughput is within 2% of the
  same-host uniform-3D control. A larger regression is a systems kill unless
  shape transitions or host drift make the comparison invalid.
- Proxy success requires loss at least `0.004` better than exp007's
  `3.616361`, i.e. `<= 3.612361`; `<= 3.601773` is the stronger threshold for
  remaining competitive with the 4D baseline grey zone.
- Loss worse than exp007 by `0.004` or more kills the allocation. Values between
  the thresholds are reported as inconclusive, not promoted by narrative.
- Non-finite/pathological health, wrong SHA, accounting mismatch, dirty tracked
  files, missing artifacts, or failed W&B upload invalidates the stage.

**Results (night suite `night-20260802-v2`, 2 August 2026).** SHA
`cd87f73e`, clean tree, provenance and W&B artifacts complete.

| Result | Value |
|---|---:|
| Benchmark vs uniform-3D control (median steady-state tok/s) | +0.17% (within 2%) |
| Health diagnostic | pass; 0 grad spikes, 0 loss spikes |
| Final proxy validation loss | 3.624233 |
| Delta vs baseline seed 0 (3.597773) | +0.026460 (+14.7 sigma) |
| Delta vs exp007 uniform-3D (3.616361) | **+0.007872** (+4.4 sigma, wrong direction) |

Killed by the pre-registered gate: loss is worse than exp007 by more than
the `0.004` kill margin, in the opposite direction from the hypothesis.
Per-layer normalized gradient RMS at update 256 shows why: layers 8-11
(allocated 3.5D, the "more capacity where it matters" layers) have grad RMS
0.93-1.14e-5, below the mean and below their 4D-uniform counterparts
(1.30-1.66e-5), while layers 0-3 (allocated the narrower 2.5D) have RMS
5.71-1.49e-5, 5-8x the mean. The shallow-to-deep allocation prior in the
hypothesis was backwards for this model: gradient magnitude, and by
extension apparent capacity demand, is concentrated in the early layers,
not the late ones. Do not run seeds 1/2 or revisit this allocation without
first re-deriving the prior from the per-layer diagnostic.

## exp015 — 2D SwiGLU versus 3D GELU

**Status:** planned. No full run is authorised by this row.

**Hypothesis.** A SwiGLU MLP with hidden width 2D has three D-by-hidden matrix
products, or `6D^2` leading parameters/FLOPs, matching a 3D GELU MLP's two
matrix products. Its learned multiplicative gate may recover at least `0.004`
of exp007's proxy loss while preserving the uniform-3D systems gain. This is
why the comparison is **2D SwiGLU versus 3D GELU**, not 2D versus 2D.

**Night funnel.** Verify exact model and per-layer parameter accounting, run a
50-update same-host benchmark, then a separate 256-update no-clipping health
diagnostic. Run proxy seed 0 only if the compiled path is valid and the median
complete-step regression versus uniform 3D is no more than 2%.

- Proxy loss `<= 3.612361` recovers the minimum detectable `0.004` from
  exp007; `<= 3.601773` is the stronger baseline-competitive threshold.
- Loss `>= 3.620361` or a valid systems regression above 2% kills the unchanged
  candidate. Intermediate quality is inconclusive.
- The activation/gating kernel must remain compiled; eager fallback, recompile
  storms, non-finite/pathological health, accounting mismatch, wrong SHA,
  missing artifacts, or failed W&B upload invalidates the stage.

**Results (night suite `night-20260802-v2`, 2 August 2026).** SHA
`2008e2d1`, clean tree. Benchmark and health stages complete and uploaded;
proxy was skipped by the harness because the health stage returned `kill`.

| Result | Value |
|---|---:|
| Benchmark vs uniform-3D control (median steady-state tok/s) | -1.225% (within the -2% threshold) |
| Health diagnostic decision | **kill**: 3 grad spikes, 2 loss spikes (`>=2` of either kills) |
| Max global grad norm | 63.94 (vs 4.15-4.72 for exp012/13/14) |
| Median global grad norm | 0.5044 (in line with exp012/13/14: 0.42-0.53) |
| Nonfinite count | 0 |

**Reanalysis.** The three grad spikes (updates 80, 86, 88) and two loss
spikes (80, 81) all fall inside the 96-update warmup window; none recur
after warmup. The step-80 event (`|g|`=63.94, loss jump +0.968) fully
reverses at step 81 (loss jump -0.946) with zero non-finite values and a
smoothly increasing parameter norm (224.5 -> 224.8 -> 225.2) throughout.
Splitting the diagnostic by phase: warmup max `|g|` = 63.94 but
post-warmup max `|g|` = 1.92, median 0.4175 — statistically indistinguishable
from exp012/13/14's post-warmup medians (0.33-0.42). The health gate computes
its 10x-median spike threshold over the pooled 256 updates (160 of them
post-warmup, median ~0.42), so an ordinary warmup transient in a
multiplicative-gate architecture crossed a threshold calibrated on a mostly
non-warmup population. exp013 shows the same effect at smaller scale (1
spike, at update 1, its only warmup-adjacent update).

This does not establish that SwiGLU is stable — a max grad norm 13.6x every
other candidate's maximum is a real signal that a multiplicative gate is
more warmup-sensitive here than an additive GELU MLP, and is worth
respecting rather than dismissing. It does establish that the specific gate
rule (pooled-median threshold, unconditional on phase) is not equipped to
distinguish "sharp but self-correcting warmup transient" from "persistent
instability" for this architecture family.

**Separately, from the benchmark profiler:** the 2D SwiGLU MLP's `c_gate`
and `c_value` are implemented as two separate `Linear(768, 1536)` layers
(`night-20260802-v2/exp015/benchmark/train_gpt2.py:106-107`) rather than one
fused `Linear(768, 3072)` plus a chunk. This produces 5856 `aten::mm` calls
per step versus 4704 for the other candidates, and is the most likely
source of the -1.225% systems regression relative to uniform 3D despite
`aten::mm` self-CUDA time being the lowest of all six benchmarked
configurations (2.457s).

**Reopening condition.** Do not reopen exp015 with an unmodified gate rule.
A reopened attempt should (a) fuse the gate/value projection into a single
matmul before re-benchmarking, and (b) recompute the health gate's spike
threshold from the post-warmup population only, or run it as two windows.
If clipping is used to hedge the warmup sensitivity, cap it high enough
(8-10) to be a no-op for exp012/13/14 (whose max grad norm is 4.15-4.72),
preserving a fair comparison; this satisfies, rather than contradicts, the
suite-wide rule against clipping without measured pathology, since the
pathology is now measured and specific to this architecture.

## exp016 — Descent-Budgeted Multi-Directional AdamW

**Status:** completed A on 3 August 2026; mechanism killed. B and C were not
run, as required by the staged gate.

**Treatment.** For each attention-V AdamW-preconditioned direction `P`, estimate
the leading two singular triplets and define

```text
gap = max(sigma_1 - sigma_2, 0)
Q(alpha) = P - alpha * gap * u_1 v_1^T,  0 <= alpha <= 1
alpha* = max alpha such that <G,Q(alpha)> / <G,P> >= 0.99
```

Apply `Q(alpha*)` without Frobenius renormalisation; all weaker directions are
retained. This directly addresses exp011's failure mode: it spends at most 1%
of predicted first-order descent and never amplifies the remainder merely to
restore update norm. The scientific claim is an internally derived
descent-constrained spectral reweighting variant of AdamW. It is not claimed as
proven novel; SPECTRA is close prior art and must be disclosed.

**A — canonical-checkpoint causal replay.** Replay the same eight frozen-weight
effective batches at exp000 proxy and full seed-0 checkpoints. A passes only if
both checkpoints have median descent retention at least 0.99, median update-norm
retention at least 0.98, median functional leading-energy reduction at least
10%, and `alpha* > 0` for at least 25% of layer/update records. Missing or wrong
checkpoint state, parameter movement, non-positive descent, failed synthetic
self-test, or NaN/Inf invalidates A; a valid failure is a scientific kill.

**B — correctness and systems.** Verify zero-gap identity, V-only correction,
descent-budget enforcement, optimizer-state round-trip, and deterministic
resume. On one host compare control-before, treatment, and control-after for 50
exact-shape updates with isolated compile caches. Median treatment complete-step
overhead and projected proxy overhead must both be at most 1%, with no eager
fallback, hidden recompile, OOM, or non-V mismatch.

**C — conditional training.** Only if A and B pass, run a separate 128-update
no-clipping health diagnostic and then proxy seed 0. Loss `<= 3.593773` passes,
loss `>= 3.601773` kills, and the interval is inconclusive. Any provenance,
token/order, artifact, or W&B failure invalidates the stage and stops the suite.

## Proxy suite v3 — funnel design

**Status:** pre-registered 2 August 2026, before implementation. This section
fixes the gate structure for exp017-exp020 and the re-run of exp016.

**What changed versus the night-20260802-v2 funnel and why.**

The v2 funnel was `benchmark -> 256-update health -> proxy` for every
experiment. Its health stage produced one confirmed false kill (exp015) and
cost roughly 16 GPU-minutes per experiment for a statistic that could not
observe anything past update 256. v3 removes the separate health stage and
replaces it with in-run instrumentation plus two tripwires inside the proxy
itself:

- every logged update carries the pre-clip global gradient norm, the clipping
  coefficient, whether clipping activated, and a phase tag
  (`warmup` / `steady` / `warmdown`) derived from the production token clock;
- a non-finite loss or gradient aborts the stage immediately;
- a validation loss above a pre-registered per-experiment envelope aborts the
  stage.

This is strictly more information than the v2 health probe (whole run rather
than 256 updates, real scheduler and real ramp rather than a substitute
regime), it removes the pooled warmup/steady median that caused the exp015
false kill, and it removes one stage of harness surface. Health statistics
are reported per phase and are never pooled across warmup and steady state.

**Envelope definition.** The envelope is the appropriate reference run's own
validation curve plus a fixed margin of 0.25 nats, evaluated at the matching
validation checkpoint. exp017 and exp019 use the exp012 proxy curve (they
share its ramp, so their validation checkpoints land on the same token grid);
exp020 uses the baseline proxy curve. The margin is deliberately loose: the
envelope exists to stop divergence, not to pre-judge quality. A candidate
that is merely worse must still finish, because a finished worse run is
evidence and an aborted one is not.

**Benchmark stages run only where a systems claim exists** — exp019 and
exp018. exp017 changes only optimizer parameter grouping and exp020 changes
only the learning-rate schedule shape; neither can move throughput, so
neither gets a benchmark stage. Where a benchmark does run it is bracketed
control-before / treatment / control-after on the same host, and the reported
statistic is the paired median steady-state complete-step time plus a
separate compile-inclusive end-to-end number.

**Accounting is a mandatory first stage for every experiment.** It costs
about 30 seconds, runs no training, and checks parameter counts against the
declared reference, weight-tying identity, optimizer-group coverage (every
parameter in exactly one group), and, where a reference implementation
exists, numerical equivalence against it. An accounting failure kills the
experiment before any GPU-hour is spent.

**Suite-level rules.** A failure inside one experiment is recorded and the
suite continues to the next experiment; only preflight, disk, or GPU
failures stop the suite. No stage in this suite may run a full run, and the
launcher is required to contain no full stage. Proxy stages may not report
`complete` without a validated `checkpoints/final.pt`.

## exp016 — canonical checkpoint manifest

**Status:** attempted in proxy suite v3 on 2 August 2026; checkpoint validation
was infrastructure-invalid, so no scientific stage ran.

Stage A requires frozen-weight replay at the exp000 proxy and full seed-0
states. Both exist in `nocap-runs-backup/baseline-20260718T074451Z/` and were
verified to be final rather than mid-run:

| State | `next_step` | Tokens | Final validation | `checkpoint.pt` SHA-256 | Bytes |
|---|---:|---:|---:|---|---:|
| exp000 proxy seed 0 | 1788 | 937,426,944 | 3.597773 | `bb909dfe3d3b99258e1d550a62de098877075887551b2dd14695eb55b98034a8` | 1,482,759,539 |
| exp000 full seed 0 | 4768 | 2,499,805,184 | 3.377696 | `893db0f1dfbb582f0a33da598b0e92de0e56bcac3322a002140c6839d3010b4b` | 1,482,759,539 |

Both carry `model`, `optimizer`, `next_step`, `train_loader`, `next_x`/`next_y`
and full `rng_state`, but the legacy schema does **not** store `tokens_seen`.
The token counts in the table are exactly derivable as `next_step * 524,288`.
Both were produced by the same captured source, `train_gpt2.py` SHA-256
`61f4afe121c6ee1eb077b0356234cad6d031d7d95ae8ab79f18fcc678cb81cbd`
(27,614 bytes).

**Canonicalisation note.** The recorded git commit is `bd681a33` with
`git_dirty: true`, so the commit alone does not identify the source. Identity
is therefore established by the SHA-256 of the captured `train_gpt2.py`, the
same reconstructibility argument already accepted for exp001. The v3
validator correctly refused to silently replace the missing field and marked
both checkpoints invalid. Before retrying A, amend validation only for these
exact checkpoint and captured-source hashes: accept the legacy schema when
`tokens_seen` is absent and derive it from the pinned `next_step` and 524,288
tokens/update. Continue to require `args.run_mode`, `args.seed`, and model,
optimizer, loader and RNG state. Any other mismatch is an invalidation, not a
scientific result. exp012-exp015 checkpoints must never be substituted.
W&B stage artifact: https://wandb.ai/m-obukhov-home/nocap-baseline/runs/ti786dan

**Overnight-v4 retry.** The v4 branch added the narrowly pinned legacy-schema
exception, but the suite copied only the remote checkpoints and not the
captured source required by the validator. A therefore stopped before replay
with:

```text
missing captured source beside checkpoint:
/workspace/nocap-runs-backup/baseline-20260718T074451Z/proxy-seed-0/train_gpt2.py
```

No spectral records, timing result or parameter update was produced, and B did
not start. This second attempt is also infrastructure-invalid rather than a
scientific failure. The local backup now contains both canonical checkpoints
and their captured `train_gpt2.py`; before the next attempt, generate one
immutable input manifest covering all three SHA-256 values and stage them as a
single unit. Then rerun A unchanged and continue to B/C only through the
already registered gates. W&B failure artifact:
https://wandb.ai/m-obukhov-home/nocap-baseline/runs/rkqpr1ew

**v5 causal result.** Commit
`fc4d9df58e86fac22231c0a9350339bd124f76af` reran A on one RTX 4090 after an
immutable input manifest validated both exact checkpoint hashes, both captured
source hashes, all 50 FineWeb train shards (5B tokens), and the validation
shard. The worktree was clean, the synthetic self-test passed, and parameters
remained bitwise unchanged for both frozen-weight replays. Each checkpoint
produced 96 layer/update records over eight exact effective batches.

| Checkpoint | Descent retention median | Norm retention median | Active fraction | Functional leading-energy reduction median | A decision |
|---|---:|---:|---:|---:|---|
| exp000 proxy | 0.990000 | 0.994261 | 100% | 0.006456 (0.6456%) | kill |
| exp000 full | 0.990000 | 0.995018 | 100% | 0.017315 (1.7315%) | kill |

The treatment passed every registered check except the required functional
reduction of at least 10%. Observed reductions were 0.24-2.29% across proxy
records and 0.79-3.31% across full records. Thus the descent constraint behaved
exactly as intended, but the alpha permitted by a 1% first-order descent budget
was too small to materially flatten the dominant functional mode. This is a
valid scientific negative result rather than another infrastructure failure.
The launcher reported `A=kill` and intentionally skipped B; do not run the
systems benchmark, health diagnostic, or proxy for this formulation.

## exp017 — selective no-weight-decay on the tied embedding

**Status:** completed through proxy seed 0 on 2 August 2026; inconclusive. No
full run is authorised by this row.

**Hypothesis.** The baseline constructs its optimizer as
`AdamW(self.parameters(), weight_decay=0.1)` with no parameter groups, so
weight decay is applied uniformly to every parameter. Because `wte.weight`
and `lm_head.weight` are tied, that single 50,257 x 768 matrix is 38,597,376
parameters, or 35.3% of the 109,376,256-parameter 3D stack, and it is decayed
at the same rate as the transformer blocks. Decaying the shared
embedding/unembedding matrix shrinks logit scale and disproportionately
erodes rare-token rows that receive gradient on few steps. Exempting only
that matrix, while leaving weight decay at 0.1 everywhere else, should
improve loss per token at exactly zero systems cost.

This is an experimental regularization choice, not a bug fix. Blanket weight
decay is a defensible default; the claim under test is only that this model
at this scale does better without it on the tied matrix.

**What is held fixed.** The complete exp012 stack: uniform 3D MLP, the exact
exp001 batch ramp 131,072 -> 524,288 over the first 50% of tokens, LR scaled
by `sqrt(batch / 524288)`, peak LR 0.0018, betas (0.9, 0.95), token-indexed
WSD with warmup 96 and warmdown 384 final-batch-equivalent updates, seed 0,
identical token order, and the 937,426,944-token budget.

**Funnel.** Accounting gate, then proxy seed 0. No benchmark stage: parameter
grouping cannot change throughput.

- Accounting passes only if the tied matrix is a single shared tensor present
  in exactly one optimizer group with `weight_decay == 0.0`, every other
  parameter is in exactly one group with `weight_decay == 0.1`, the union of
  groups covers every parameter exactly once, and the total parameter count is
  unchanged at 109,376,256.
- Proxy loss `<= 3.577923` (at least 0.004 better than exp012's 3.581923)
  passes and makes exp017 the new combo reference.
- Proxy loss `>= 3.585923` kills the treatment.
- The interval `(3.577923, 3.585923)` is inconclusive and does not promote.

**Result.** Optimizer accounting passed: the tied matrix is one shared tensor
with 38,597,376 parameters in the only zero-decay group; the other 70,778,880
parameters remain in the WD 0.1 group; tying and total parameter count
109,376,256 are preserved. The proxy completed the exact 937,426,944-token
budget in 2,549 optimizer updates with final validation loss `3.580403`, delta
`-0.001520` against exp012 (`-0.85 sigma`). This lies inside the registered
inconclusive interval. The run was finite throughout and retained a final
checkpoint (`cb3b2336...b7e9b3b`). Cross-session throughput is not used as a
systems claim because optimizer grouping has no intended compute effect.
W&B stage artifact: https://wandb.ai/m-obukhov-home/nocap-baseline/runs/eujpxoj7

**Decision.** Do not promote and do not spend seeds 1/2 on exp017 alone. The
small sign-consistent delta is retained only as a possible interaction term
for a future small-batch combination, where more optimizer updates per token
also mean more AdamW decay applications.

## exp018 — FP8 lm_head feasibility measurement

**Status:** completed forward feasibility measurement on 2 August 2026; gate
passed. This row authorises no training run.

**Motivation from the night-20260802-v2 profiler.** Two 32-call cutlass
kernels — one call per micro-batch — account for 386.8 ms and 315.3 ms of
every optimizer step, and they are invariant to within 0.2 ms across all six
profiled variants (baseline 4D, uniform 3D, exp012, exp013, exp014, exp015).
That identifies them as the vocab projection, untouched by every MLP
treatment tried so far. With the log-softmax block they are 816.3 ms of the
4,113 ms 4D step (19.85%) and 874.1 ms of the 3,731 ms uniform-3D step
(23.43%). After 3D narrowing the share rises, because the denominator fell
and the numerator did not. It is the largest remaining systems target and is
orthogonal to everything in the current queue.

**Scope this round.** Standalone shape-exact microbenchmark only. No training
integration, no proxy, no interaction with exp017 or exp019.

**What is measured.** On the target RTX 4090 (sm_89), at the exact production
shapes — activations `(16, 1024, 768)` flattened to `(16384, 768)` against a
`(768, 50257)` projection — time the BF16 baseline and the FP8
`torch._scaled_mm` path for: forward, grad-input, grad-weight, and the cast
and scale-computation overhead separately from the GEMM itself. `_scaled_mm`
requires 16-divisible dimensions and 50,257 is not, so the padded compute
shape 50,272 must be measured explicitly; padding is a compute-shape device
only and cross-entropy is always taken over the true 50,257 columns.

**Gate.** The reported statistic is projected full-optimizer-step gain, that
is, measured saving on the vocab-projection block divided by the measured
uniform-3D step time of 3,731 ms.

- Projected full-step gain `>= 5%` authorises a separate integration
  experiment, which must then preserve `wte`/`lm_head` tying exactly and pass
  its own numerical-health and proxy gates.
- Projected full-step gain `< 5%` stops FP8 here and is recorded as a measured
  negative, not as an untested idea.
- Broken tying, a required change to the true vocabulary, or non-finite
  outputs invalidate the measurement.

A realistic prior is 4.7-7.5%, straddling the threshold, which is why the
measurement is worth five GPU-minutes and the integration is not yet worth
its implementation risk.

**Result.** At the registered production forward shape on the RTX 4090, BF16
forward median was `10.724 ms`; FP8 cast-plus-GEMM median was `4.572 ms`
(`4.441 ms` GEMM-only), with finite output. Projecting only this measured
forward saving onto the 3,731 ms uniform-3D reference step gives `5.28%`, just
above the 5% gate.

**Decision and limit.** This is a feasibility pass, not an integrated training
speedup. The measurement did not validate the two backward GEMMs, master-weight
tying, masking the padded 50,272 compute columns back to the true 50,257-token
vocabulary, cross-entropy equivalence, or loss. Run an integration benchmark
only after selecting the winning architecture.
W&B stage artifact: https://wandb.ai/m-obukhov-home/nocap-baseline/runs/qgr9w0e7

## exp019 — fused 2D SwiGLU with the exp012 batch ramp

**Status:** attempted in proxy suite v3 on 2 August 2026; infrastructure-invalid
before the treatment benchmark, so no proxy result exists. Reopens exp015
under the condition recorded in that row.

**Hypothesis.** exp015's -1.225% benchmark result against the uniform-3D
control was kernel-launch overhead, not arithmetic. Its `aten::mm` self-CUDA
time was 2.457 s, the lowest of all six profiled variants and below the
uniform-3D control's 2.469 s, while it issued 5,856 `aten::mm` calls against
4,704 — an extra 1,152 launches per optimizer step, exactly 36 per micro-batch,
produced by implementing the gate as two separate `Linear(768, 1536)` modules.
Merging them into one `Linear(768, 3072)` followed by `chunk(2, dim=-1)`
should remove those launches at identical parameter count and identical
arithmetic. If fused SwiGLU then matches uniform-3D throughput while
recovering part of the +0.018588 capacity penalty that 3D narrowing pays,
it replaces uniform 3D inside the exp012 combination.

**Treatment.** Fused 2D SwiGLU (`hidden_width = 2 * n_embd = 1536`, one
`Linear(768, 3072, bias=False)` split by `chunk(2, dim=-1)` into gate and
value, `silu(gate) * value`, then `Linear(1536, 768, bias=False)`), plus the
exact exp012 batch ramp and LR scaling, plus global gradient clipping at
threshold 10.

**Why clipping is part of the treatment and why 10.** exp015's health kill was
a false positive of the pooled-median gate, but the underlying transient was
real and larger than anything the other candidates produced: at update 80,
inside the 96-update warmup, the global gradient norm reached 63.94 against a
maximum of 4.72 across exp012, exp013 and exp014. It self-corrected in a
single update (loss +0.968 then -0.946, parameter norm continuing smooth at
224.5 -> 224.8 -> 225.2, zero non-finite values), and with betas (0.9, 0.95)
AdamW's second moment forgets it within roughly 40 updates. Post-warmup,
exp015's maximum gradient norm was 1.92 against exp013's 0.81 and exp014's
0.96. The correct response to a real but self-correcting multiplicative-gate
transient is to truncate it, not to abandon the architecture. Threshold 10 is
above every gradient norm ever observed in exp012, exp013 and exp014, so it is
inert for the controls and fires only on the pathological event; clipping
activation count and the pre-clip norm distribution are logged per phase and
reported, so a claim that clipping did the work is falsifiable.

The comparison is therefore fused-SwiGLU-plus-clip against exp012, and the
report must state that clipping is confounded with the architecture change.
If exp019 passes, a clip-only control on the exp012 stack is required before
the result is attributed to SwiGLU.

**Funnel.** Accounting, then bracketed benchmark, then conditional proxy.

- Accounting passes only if leading MLP parameters and FLOPs match uniform 3D
  exactly (3,538,944 MLP parameters per layer, against uniform 3D's
  2 x 768 x 2304 = 3,538,944), total parameters equal 109,376,256, and the
  fused implementation is numerically equivalent to the two-Linear exp015 form
  given the same weights, to within bf16 tolerance.
- Benchmark against exp012 as the control, bracketed control-before /
  treatment / control-after: regression `<= 1%` passes; 1-2% is a warning and
  the proxy may still proceed; regression `> 2%` kills. Eager fallback, graph
  break, unexpected recompilation, or an accounting mismatch invalidates the
  stage rather than producing a verdict.
- Proxy loss `<= 3.577923` passes and makes exp019 the new combo reference;
  `>= 3.585923` kills it as an exp012 replacement unless time-to-target is
  materially better; the interval is inconclusive.

**Attempt result.** Accounting passed: fused SwiGLU exactly matched the
two-Linear reference under the registered tolerance, leading MLP parameters
were 3,538,944 per layer, total parameters were 109,376,256, and embedding /
head tying was preserved. The bracketed benchmark then became invalid before
any treatment update: the control received the absolute training glob, but the
treatment retained relative `data/fineweb10B/fineweb_val_*.bin` and could not
find validation data from its worktree. This is an orchestration bug, not a
health failure or a negative SwiGLU result. Retry later with both train and
validation globs absolute; do not reuse the partial benchmark as evidence.
W&B benchmark artifact: https://wandb.ai/m-obukhov-home/nocap-baseline/runs/ly4o0mya

## exp020 — cosine schedule versus baseline WSD

**Status:** completed proxy seed 0 on 2 August 2026; killed. Scheduler research
with acknowledged scope risk; not the main contribution of this project.

**Hypothesis, stated honestly.** The prior is weak and points at a null: the
trapezoid warmup-stable-decay schedule used by the baseline is generally at
least as good as cosine for short-horizon speedruns, which is why
modded-nanogpt uses it. This row exists so the queue closes the question with
a measurement on this exact stack rather than by appeal to other people's
results.

**What is held fixed.** Unchanged 4D baseline architecture, flat 524,288-token
effective batch with no ramp, peak LR 0.0018, betas (0.9, 0.95), weight decay
0.1 applied as in the baseline, identical data and token order, identical
937,426,944-token budget, seed 0. Only the schedule shape changes. Cosine is
never combined with the batch ramp in this experiment.

**Complete cosine definition, fixed before implementation.** Let `T` be the
937,426,944-token budget and `W` the warmup token count, equal to the WSD
baseline's 96 final-batch-equivalent updates, that is 50,331,648 tokens. For
tokens `t`:

- `t < W`: `lr(t) = 0.0018 * (t + update_tokens) / W`, capped at 0.0018,
  identical to the WSD warmup already implemented;
- `t >= W`: `lr(t) = 0.0018 * 0.5 * (1 + cos(pi * p))` with
  `p = (t - W) / (T - W)`.

No minimum-LR floor, no restarts, no separate decay horizon. `lr(T) = 0`
exactly, matching the WSD baseline's terminal LR.

**Funnel.** Accounting, then proxy seed 0. No benchmark stage: schedule shape
cannot move throughput.

- Proxy loss `<= 3.593773` (at least 0.004 better than baseline seed 0's
  3.597773) passes.
- Proxy loss `>= 3.601773` kills cosine for this stack.
- The interval `(3.593773, 3.601773)` is inconclusive, which given the stated
  prior is the most likely outcome and is a legitimate reportable result.

**Result.** The run completed the exact 937,426,944-token budget in 1,788
updates and retained its final checkpoint. Final validation loss was
`3.676178`, delta `+0.078405` against baseline seed 0 (`+43.6 sigma`), far
beyond the kill threshold. Gradients remained finite; the failure is quality,
not numerical instability.

**Decision.** Close cosine unchanged and do not run seeds 1/2. On this short
horizon, continuously spending the LR decay across training was much worse at
the terminal proxy checkpoint than the baseline WSD schedule that reserves a
concentrated warmdown.
W&B stage artifact: https://wandb.ai/m-obukhov-home/nocap-baseline/runs/o7u2wfwq

## exp021 — aggressive small-batch ramp on uniform 3D

**Status:** completed direct proxy seed 0 on 2 August 2026 and BF16 full seed 0
on 3 August 2026. The proxy passed on quality, but the full missed the challenge
target.

**Question.** exp002 measured a mid-run critical batch near 106k, while exp012
ramps from 131,072 to 524,288 and spends the second half of the proxy at the
largest batch. At a fixed token budget, a smaller effective batch performs
more optimizer updates and injects more gradient noise; it does not make each
individual update larger. The hypothesis is that these more frequent early
updates improve loss per token enough to repay their fixed optimizer overhead,
while the already measured 3D MLP saving keeps total training time competitive.

**Treatment fixed before implementation.** Keep the exp012 uniform 3D GELU
architecture, AdamW betas `(0.9, 0.95)`, WD 0.1, seed 0, data order, validation,
token-indexed WSD shape and exact `937,426,944`-token proxy budget. A microbatch
contains `16 * 1024 = 16,384` tokens. Over the first 50% of training tokens,
linearly interpolate gradient accumulation from 1 to 16 and round to the
nearest integer; after that use 16. Effective batch therefore moves in
fine-grained discrete steps from 16,384 to 262,144 tokens and never reaches
524,288.

The LR reference remains fixed at 524,288 tokens, rather than changing with the
new maximum batch:

```text
lr(t) = wsd_base_lr(t) * sqrt(effective_batch_tokens(t) / 524288)
```

Thus the maximum batch uses peak LR `0.0018 * sqrt(1/2) = 0.001272792`; the
smallest batch uses `0.0018 * sqrt(1/32) = 0.000318198`, before the token-indexed
warmup multiplier. Warmup and warmdown retain exp012's exact token spans:
50,331,648 and 201,326,592 tokens. Expressed at the new final batch these are
192 and 768 final-batch-equivalent updates; the full budget is 3,576 such
updates. Validation remains every 67,108,864 tokens.

**Single-stage run.** Launch proxy seed 0 directly. There is no benchmark,
smoke, standalone health stage, validation-envelope kill or automatic next
experiment. Abort only on infrastructure failure or non-finite loss/gradient.
Log every update and validation to local JSONL and W&B, including tokens,
effective batch, accumulation, base/effective LR, train loss, step/train/wall
time, throughput and phase-aware gradient norm. Retain distinct `latest.pt`
and `final.pt`; a proxy without a final checkpoint is incomplete.

**Pre-registered interpretation.** Compare against exp012 seed 0
(`3.581923`, 6,754.76 s training time) and its time to baseline-final loss
(`3.597773` at 6,582.7 s).

- Clear pass: final loss `<= 3.581923` with training time `<= 6,754.76 s`.
  This authorises the next combination suite on the exp021 ramp.
- Quality-only signal: final loss at least 0.004 better than exp012 but misses
  the time bar. Retain the direction, but first move the ramp start upward or
  reach the maximum batch earlier rather than stacking more methods blindly.
- Kill as a replacement: final loss `>= 3.585923` with no faster crossing of
  `3.597773`, or no loss improvement with higher training time. Return to the
  exp012 ramp and retry fused SwiGLU there.
- Anything else is inconclusive and is reviewed from the full loss-vs-token
  and loss-vs-time curves before choosing the next suite.

**Result.** The proxy completed the exact `937,426,944`-token budget in 7,224
optimizer updates. Final validation loss was `3.567184`, improving on exp012 by
`-0.014740` (about `-8.2 sigma`) and on the baseline by `-0.030589`. Training
time was `6,782.06 s`, 27.30 seconds (`0.40%`) above exp012, so the literal
clear-pass time condition was missed and the registered label is
quality-only. The more relevant time-to-quality curve is positive: linear
interpolation between the fixed validation points places the crossing of
exp012's final `3.581923` about 137 seconds and 23M tokens earlier than exp012.
The treatment led exp012 at every non-zero validation checkpoint.

The run was finite throughout. Warmup / steady / warmdown maximum pre-clip
gradient norms were `7.24 / 1.32 / 0.43`, no clipping was enabled, peak memory
was 9,029 MiB, and the final checkpoint covers the exact token budget
(`f7087ab3...22bb59`). W&B:
https://wandb.ai/m-obukhov-home/nocap-baseline/runs/50arvf1s

**Decision.** Promote the complete schedule package (small-batch ramp plus its
registered `sqrt(B / 524288)` LR coupling) as the base for exp022. This does
not identify smaller batch independently of LR scaling. Keep exp021 as the
automatic BF16 full fallback if exp022 fails its winner rule.

**Full result.** exp022 failed its winner rule, so the overnight-v4 selector
correctly launched exp021. The clean implementation SHA was
`4c072f0ed2ba136026d017dda6ec0174bb19cbb9`. The run completed the exact
`2,700,083,200`-token budget in 20,805 optimizer updates. Final validation loss
was `3.387356`, training time was `19,500.15 s`, compile-inclusive wall time was
`19,966.26 s`, observed throughput was 138,465 tokens/s, and peak memory was
9,027 MiB. W&B:
https://wandb.ai/m-obukhov-home/nocap-baseline/runs/aobxji7s

The canonical baseline completed 2,499,805,184 tokens at `3.377696`; exp021
therefore processed 8.01% more tokens but finished `+0.009661` worse and missed
the challenge target `3.3821` by `0.005256`. At exp021's final training time,
linear interpolation of the baseline validation curve gives approximately
`3.382400`. Raw timing comes from separate run sessions and is reported as
operational context, not a new same-host causal speedup claim.

The full loss-vs-token comparison explains the failed transfer. Relative to
baseline, exp021 was ahead by `-0.0312` at 671M tokens, `-0.0264` at 805M,
`-0.0134` at 940M and only `-0.0002` at 1.074B; the interpolated crossing is
about 1.17B tokens. The exp021 ramp did not reach its 262,144-token maximum
until about 1.35B tokens, after the advantage had disappeared. Thus the full
confirms faster early optimisation from smaller batches/LR coupling but kills
the horizon-scaled 16K-to-256K schedule as a target-reaching recipe.

Nine immutable milestones from 2.120B through 2.657B tokens plus distinct
`latest.pt` and `final.pt` were downloaded after the instance was destroyed.
All eleven files deserialize locally with model, optimizer, loader, RNG,
source and explicit token metadata; `final.pt` records exactly 2,700,083,200
tokens and SHA-256
`465b833ef2a76b9052fd05b51c3a66348962903971230a090a1e3c3a1548fc84`.
Checkpoint averaging remains an unevaluated offline rescue: the local dataset
copy lacks the validation shard, so file integrity does not establish an
averaged validation loss.

**Final decision.** Do not repeat exp021 unchanged. Use its full curve to place
an absolute-token return to the baseline-sized batch in exp024. Retain exp021
as a strong systems/early-optimisation result and an honest negative
proxy-to-full transfer result.

## exp022 — fused SwiGLU small-batch combination

**Status:** completed proxy seed 0 on 2 August 2026; killed by the registered
winner rule. No full run is authorised.

**Hypothesis.** exp021 showed that frequent early updates recover quality at
small wall-time cost, while exp015 showed that equal-leading-FLOP 2D SwiGLU has
competitive GEMM time but suffered avoidable kernel launches and a single
large warmup gradient transient. Fusing the gate/value projection removes the
launch defect. Clip 10 truncates only the measured SwiGLU pathology, and
exempting the tied embedding/head from weight decay targets the stronger
cumulative decay exposure created by exp021's 7,224 updates. The experiment
tests the complete package; it cannot attribute a passing result to any one of
the three additions.

**Treatment.** Start from exp021 exactly: token-linear accumulation ramp
`1 -> 16` over the first 50% of tokens, final effective batch 262,144, LR
`wsd_base_lr * sqrt(effective_batch_tokens / 524288)`, betas `(0.9, 0.95)`,
token-indexed warmup/warmdown, seed 0 and exact `937,426,944`-token budget.
Replace each 3D GELU MLP with fused 2D SwiGLU: one bias-free
`Linear(768, 3072)`, `chunk(2)`, `silu(gate) * value`, then bias-free
`Linear(1536, 768)`. Keep WD 0.1 on all other parameters and WD 0 only on the
single tied `wte/lm_head` parameter. Apply global gradient clipping at 10 and
log the pre-clip norm and clipping coefficient.

**Run and winner rule.** Launch proxy seed 0 directly, without a separate
benchmark, smoke, health probe or validation-envelope stop. Abort only on an
infrastructure error or non-finite loss/gradient. exp022 replaces exp021 for
the BF16 full if either:

- final loss is at most `3.563184` (0.004 better) with training time no more
  than 2% above exp021's `6,782.06 s`; or
- interpolated training time to loss `3.567184` is at least 1% below
  `6,782.06 s`, while final loss is no worse than `3.567184`.

Otherwise exp021 remains the full winner. The selector and all interpolation
inputs are saved as JSON. A completed proxy must retain `final.pt` and its W&B
artifact.

**Result.** The clean implementation SHA
`83fb35fadc625cdd56bc4f076cb9773b6051e0fd` completed the exact
937,426,944-token proxy in 7,224 optimizer updates. Final loss was `3.607177`,
which is `+0.039993` worse than exp021 and `+0.009403` worse than baseline.
Training time was `6,885.03 s` and throughput was 136,154 tokens/s: respectively
1.52% more time and 1.50% less throughput than exp021. It never reached
exp021's final proxy loss, so neither the quality nor time-to-quality winner
route passed. W&B:
https://wandb.ai/m-obukhov-home/nocap-baseline/runs/qc5luw25

Gradient clipping is not an explanation: maximum pre-clip norm was 6.74 in
warmup and 4.71 in steady state, below threshold 10, so clipping never changed
an update. Because fused SwiGLU and selective no-WD were added together, the
remaining quality regression cannot be attributed between them from this run.

**Decision.** Keep exp021 as the selector winner, close exp022, and do not run
its full or seeds 1/2. Do not promote either SwiGLU or selective no-WD on this
stack.

**Conditional full outcome.** The selector chose exp021, which then completed
the registered `2,700,083,200`-token BF16 full described above. No exp022 full
ran. FP8 remained excluded from that run; any later hardware-optimised full
still requires a valid exp023 integration result and separate explicit user
approval.

## exp023 — FP8 tied-lm-head integration benchmark

**Status:** attempted in overnight-v4 on 2 August 2026; infrastructure-invalid
before the first benchmark update. No FP8 proxy or full training is authorised.

**Treatment.** After exp022 selects the architecture, instantiate that winner
with one FP32 master tied embedding/head weight padded from 50,257 to 50,304
rows. Both control and FP8 treatment use the identical padded parameter and
slice logits back to the true first 50,257 classes before cross-entropy. The
padding is therefore a compute-shape device, not a vocabulary or objective
change. Convert only the inner lm-head linear to TorchAO FP8 training; the
embedding lookup uses the shared FP32 master under the existing BF16 autocast,
and the optimizer master weight remains FP32.

**Measurement.** On the same RTX 4090, run control-before, FP8 treatment and
control-after for identical fixed inputs and exact production full updates.
Use isolated compile caches and report compile-inclusive time separately from
the median steady complete-step time. Before timing, require finite forward,
backward and optimizer results, identical weight tying, output shape ending in
50,257, every target below 50,257, and a recorded one-step loss/gradient delta
against BF16.

**Decision.** Median FP8 complete-step gain at least 3%, control drift at most
2%, no graph break/eager fallback, preserved tying and finite numerics pass the
integration benchmark and authorise a separate clean FP8 proxy on the winner.
Anything else closes or invalidates the hardware path as appropriate. This
suite never chains an FP8 proxy or FP8 full automatically.

**Attempt result.** `winner.json` validly selected `exp021`, but
`run_fp8_benchmark.sh` immediately exited 2 with `winner must be exp021 or
exp022`, before creating a benchmark result. Suite preflight also emitted load
failures for TorchAO `_C_cutlass_90a` and a CPython-3.10 `_C_mxfp8` extension
inside the Python-3.12 environment. These are orchestration/runtime failures,
not negative FP8 evidence.

**Next step.** After the final architecture is selected, reproduce the winner
transport failure, pin a TorchAO build compatible with the actual Python/CUDA
runtime, and repeat only the registered integration benchmark. Measure the
complete production forward/backward path before considering any FP8 full.

## exp024 — absolute-token staircase batch schedule on uniform 3D

**Status:** pre-registered on 3 August 2026 from the completed exp021 full
curve. Implementation `9e86007` is ready and Max subsequently authorised one
direct BF16 full seed-0 run. No FP8 treatment is part of exp024.

**Question.** exp021's horizon-scaled 16K-to-256K ramp created a large early
loss advantage but kept increasing the batch too slowly. Against the baseline
full curve, its loss delta shrank from `-0.0312` at 671M tokens to `-0.0264` at
805M, `-0.0134` at 940M, `-0.0088` at 1.007B and approximately zero at 1.074B;
the interpolated crossing is about 1.17B. At those points exp021 was still only
at approximately 131K, 164K, 180K, 197K and 213K effective batches. It reached
256K only near 1.35B, after the advantage had disappeared.

The hypothesis is that a small number of absolute-token staircase phases can
retain useful early update frequency while returning to the baseline 524K
batch and peak LR before the observed advantage is exhausted. Unlike exp021,
the transition tokens do not scale with the run horizon, so a later budget
change cannot silently retime the intervention.

**Treatment fixed before implementation.** Keep exp021's uniform 3D GELU
architecture, token order, seed 0, AdamW betas `(0.9, 0.95)`, weight decay 0.1,
no clipping, sequence length 1,024, BF16 compute and 524,288-token LR reference.
Use a 16,384-token microbatch and the following exact schedule:

| Token interval | Effective batch | Accumulation | Updates in phase | Cumulative update at boundary |
|---|---:|---:|---:|---:|
| `[0, 201,326,592)` | 65,536 | 4 | 3,072 | 3,072 |
| `[201,326,592, 469,762,048)` | 131,072 | 8 | 2,048 | 5,120 |
| `[469,762,048, 939,524,096)` | 262,144 | 16 | 1,792 | 6,912 |
| `[939,524,096, 2,700,083,200)` | 524,288 | 32 | 3,358 | 10,270 |

The 524K transition is at the 14th 67,108,864-token validation boundary, just
below 1B tokens. It is deliberately earlier than exp021's observed crossover:
at 940M the remaining loss buffer was `0.0134`, while by 1.074B it was gone.
The 256K phase itself is not validated by exp021 at these tokens—exp021 was
still below 200K—so this direct full remains a genuine risk rather than a
curve-derived certainty.

At each stage use

```text
lr(tokens) = wsd_base_lr(tokens) * sqrt(effective_batch_tokens / 524288)
```

with peak multipliers `sqrt(1/8)`, `1/2`, `sqrt(1/2)` and `1`, respectively.
Keep exp021 full's exact WSD token spans: 144,965,632 warmup tokens and
579,862,528 warmdown tokens, with base peak LR 0.0018. Batch/LR jumps are part
of the treatment; log validation immediately at every phase boundary and the
first regular validation after it.

**Why direct full instead of the standard proxy.** The standard proxy ends at
937,426,944 tokens, about 2.1M tokens before the decisive 524K transition. It
would measure only the three early phases and could not answer whether
returning to the baseline batch arrests the full-run loss-gap decay. exp021 has
already supplied a complete full causal pilot for locating this transition.
Running another standard proxy and promoting it by terminal loss would repeat
the proxy-to-full mistake exp021 exposed. This is an explicit, final-candidate
exception to the usual funnel, made to reduce rather than expand the number of
remaining ablations.

**Run and artifacts.** If Max later authorises paid compute, run seed 0 once to
the exact 2,700,083,200-token budget. Start from a clean exp024 branch and
record the exact dataset manifest, Git SHA and source snapshot. Count
compile-inclusive wall time as primary timing; report per-shape compilation
spikes separately. Save immutable checkpoints at all three phase boundaries,
the final checkpoint, and the last 8-10 warmdown milestones. Abort only on
non-finite loss/gradient or an infrastructure/provenance failure; do not tune a
boundary during the run.

**Implementation.** Branch `exp024/staircase-batch-ramp`, commit `9e86007`,
uses a 262,144-token clock to preserve exp021's exact warmup, warmdown,
validation, save and milestone spans while allowing the final effective batch
to be 524,288. Local accounting verifies 10,270 optimizer updates, exact phase
boundaries and 109,376,256 parameters. The launcher accepts only full seed 0,
requires online W&B logging, creates a hashed 50-shard dataset manifest before
training, saves immutable phase and warmdown checkpoints, and uploads the final
checkpoint as a W&B artifact.

**Pre-registered interpretation.** Primary success is validation loss at most
`3.3821`, with time-to-target compared against the same baseline timing
definition. Also report final loss at the exact token budget, loss-vs-token,
loss-vs-training-time, observed throughput, optimizer updates, peak memory and
the loss response around every batch transition. A final loss above `3.3821`
kills this schedule as a challenge winner. If exp024 fails, do not launch
another algorithmic full from the current backlog; proceed only with the
separately gated hardware track or finish the negative-result submission.
