# Experiment record

The point of this page is to show that the experiments were not a random grab
bag. Each one starts from a specific piece of evidence — the baseline profiler,
a measurement, or the failure mode of an earlier run — so the **Hypothesis**
column below begins, where applicable, with `<- expNNN` naming the run that
prompted it. Read top to bottom and the reasoning chain should be visible.

## Baseline profiler snapshot (exp000)

Measured on the RTX 4090 after warm-up: one complete optimizer step ≈ **4.03 s**,
≈127k tok/s, GPU utilisation 99.28%, only 29 ms of CUDA idle — so this is **not**
a CPU or data-loading bottleneck. Grouping the profiled kernels by function:

| Step component | Self CUDA time | Share of step |
|---|---:|---:|
| `aten::mm` — all matrix multiplies | 2.685 s | 67.1% |
| &nbsp;&nbsp;of which: MLP up/down projections (+ GELU) | — | ~28.5% |
| &nbsp;&nbsp;of which: tied vocabulary projection (`lm_head`) | — | ~23.5% |
| &nbsp;&nbsp;of which: QKV projections | — | ~11.0% |
| FlashAttention forward + backward | 0.355 s | 8.9% |
| AdamW step | 0.011 s | 0.3% |
| CUDA idle | 0.029 s | 0.7% |

**The workload is firmly GEMM-bound.** This profile is the lever for most of the
architecture work: the three biggest matrix multiplies are the MLP projections,
the tied vocabulary projection, and the QKV projections. From **exp007 onward**
the MLP width is the primary target (28.5% of the step), and the vocabulary
projection (23.5%) is the largest remaining one after that (exp018).

**Baseline reference losses** (every quality delta below is against these, at the
matching token count): proxy seed 0 = `3.597773`; three-seed proxy mean
`3.59935`, sample σ `0.0018`; full seed 0 = `3.377696`. Challenge target ≤
`3.3821`.

## Reading the verdicts

**PASS** met its pre-registered gate · **GREY** landed in the pre-registered
inconclusive band · **NULL** no effect either way · **KILL** failed its gate ·
**MEASURE** a diagnostic, no training claim · **INFRA** stopped by an
orchestration bug, not a scientific result.

## All experiments

| ID | What changed / was measured | Hypothesis (and what prompted it) | Result & verdict |
|---|---|---|---|
| **exp000** | Frozen baseline reference | Establish an immutable, reproducible reference and its proxy/full losses before touching anything. | Proxy s0 `3.597773`; full s0 `3.377696`. **Reference.** |
| **exp001** | Effective-batch ramp `131K → 524K` over first 50% of tokens; LR ∝ √B | ← profiler + prior that the baseline batch is oversized early: more early updates per token should improve loss/token. | Proxy `3.560791`, **−0.03698**. **PASS**, but the gain decays late; unmodified ramp is not a full-run recipe. |
| **exp002** | Gradient-noise-scale measurement along the unchanged baseline | ← exp001: its early gain decayed — measure the *actual* critical batch (McCandlish `B_simple`) to see how over-batched the run really is. | Mid-run `B_crit` ≈106k vs 524K batch; crossing at ≈910M tokens (97% of proxy). **MEASURE**; retires the fixed-ramp premise. |
| **exp003** | Flat `16,384` effective batch for the whole proxy; LR `3.182e-4` | ← exp002: if the baseline is over-batched almost throughout, a flat small batch should help most. | −0.3847 at 21.5% of budget but **+17.6%** step overhead. **KILL** by scope — this is batch/LR tuning, which the challenge excludes. |
| **exp004-A** | FineWeb-Edu classification of the exact baseline token subset | Can the baseline data be partitioned by document quality, reproducibly and without changing token volume? | 8.2418% of selected tokens (206,027,733) are in score-≥3 documents. **MEASURE**; enables a curriculum test. |
| **exp004-B** | Move all score-≥3 tokens into a warmdown-aligned enriched mixture | ← exp004-A: concentrating scarce high-quality data under low LR (warmdown) should improve final loss. | `3.612175`, **+0.014402** worse. **KILL**; reordering a fixed dataset introduced harmful distribution shift. |
| **exp005** | Staged sequence length `T=512 → 1024` (systems gate only) | ← profiler (attention ≈8.9%): halving early `T` at fixed `B·T` should cut wall-time. | `T=512` was 2.59% faster steady-state, but compile cost projected **−1.21% net**. **KILL** at the systems gate; no proxy. |
| **exp006** | Same `T=512 → 1024` curriculum, quality run | ← exp005: the systems gate didn't test *loss per token* — does early short context help learning? | `3.600870`, **+0.003097** (worse direction). **GREY / inconclusive**; not promoted. |
| **exp007** | Narrower MLP `4D → 3D` | ← profiler: MLP is the largest GEMM (~28.5%) — narrowing it should cut step time. | **10.54% faster** step (same-host); proxy `3.616361`, **+0.018588**. **GREY** on quality, but the key systems win the rest of the work builds on. |
| **exp008** | Grouped-query attention (GQA), 12 query / 4 KV heads | ← profiler (QKV ~11%): sharing KV heads should shrink the QKV projection. | `3.617567`, **+0.019794**. **KILL** on quality; no valid same-host speedup measured. |
| **exp009** | Residual-Complement Attention — offline gate | Does the attention output carry a removable residual-aligned component with a favourable loss derivative? | No layer passed the confirmatory slope check at the full checkpoint. **KILL** (offline); no training run. |
| **exp010** | Selective Spectral AdamW — offline diagnostic | Do AdamW-preconditioned update matrices have persistent dominant singular directions? | Passed for 4 matrix families (attn-out, attn-V, MLP down/up). **MEASURE / mechanism present**, but causality unproven. |
| **exp011** | Cap the full σ₁−σ₂ gap of the attention-V update | ← exp010: if the leading mode is redundant, capping it should keep descent while flattening the update. | Descent retention `0.8817` / `0.9583` (below the 0.98 gate). **KILL**; the leading mode carries real gradient signal. |
| **exp012** | `3D` MLP **+** exp001 batch ramp (factored combination) | ← exp007 (systems win, quality loss) + exp001 (quality win): the ramp should buy back the 3D capacity deficit. | `3.581923`, **−0.015850**; +10.18% speedup; effects independent (interaction 1.4σ). **PASS**; leading full-run candidate. |
| **exp013** | Squared-ReLU activation in the `4D` MLP | ← profiler (~5% GELU): a sparse activation might help loss or cut activation time. | `3.597677`, **−0.000096** (exact null); +0.34% throughput. **NULL** — GELU is already fused; still GEMM-bound. |
| **exp014** | Depth-shaped MLP `2.5D / 3D / 3.5D` (mean = 3D) | ← exp007: keep 3D on average but give later layers more width, where representations are more task-specific. | `3.624233`, **+0.007872** vs uniform 3D. **KILL**; per-layer gradients show the allocation prior was backwards. |
| **exp015** | `2D` SwiGLU vs `3D` GELU at equal leading FLOPs | ← exp007: a gated MLP might recover 3D's capacity loss at matched cost. | Health-killed (contested false positive); −1.225% benchmark from unfused kernels. **KILL / contested**; proxy never ran. |
| **exp016** | Descent-Budgeted Multi-Directional AdamW (1% budget) | ← exp011: instead of the full gap, remove only what a strict 1% descent budget allows. | Descent preserved, but functional-energy reduction only 0.65% / 1.73% (vs 10% needed). **KILL**; safe but too weak. |
| **exp017** | Exempt only the tied `wte`/`lm_head` matrix from weight decay | ← exp012 stack: the tied 38.6M-param matrix is 35% of weights — decaying it may erode rare-token rows. | `3.580403`, **−0.001520** vs exp012. **GREY / inconclusive**; retained as a possible small-batch interaction term. |
| **exp018** | FP8 `lm_head` — forward feasibility microbenchmark | ← profiler: vocab projection (~23.5%) is the largest untouched GEMM after 3D narrowing. | Forward `10.724 → 4.572 ms`; projected 5.28% full-step gain. **PASS (feasibility only)**; backward/tying/quality unvalidated. |
| **exp019** | Fused `2D` SwiGLU + exp012 ramp + clip 10 | ← exp015: its slowdown was kernel-launch overhead — fusing the gate/value projection should remove it. | Accounting passed; benchmark stopped by a relative-path bug. **INFRA**; no SwiGLU result. |
| **exp020** | Cosine LR schedule vs baseline WSD | Close the LR-schedule question by measurement on this exact stack, rather than by appeal to others. | `3.676178`, **+0.078405**. **KILL**; WSD's reserved warmdown is much better on this short horizon. |
| **exp021** | Aggressive token-linear ramp `16K → 256K` on `3D` MLP (direct full) | ← exp012 + exp002: smaller early batches help most, so push the ramp harder and take it to a full run. | Proxy `3.567184` (**PASS**); **full `3.387356` missed target by 0.005256**. Early lead vanished ≈1.1B tokens. **KILL as a full recipe** — the decisive negative result. |
| **exp022** | Fused SwiGLU + selective no-WD on the exp021 schedule | ← exp021 winner + exp015 + exp017: stack the three onto the best schedule. | `3.607177`, **+0.039993** worse than exp021. **KILL**; regression can't be attributed between the additions. |
| **exp023** | FP8 tied-`lm_head` integration benchmark | ← exp018: promote the FP8 feasibility pass to a real integration measurement. | Runner rejected the valid winner before measuring; TorchAO build mismatch. **INFRA**; no FP8 claim yet. |
| **exp024** | Absolute-token batch staircase `64K → 128K → 256K → 524K` on `3D` MLP | ← exp021: it returned to the large batch *too late*; fix the transition tokens absolutely so the run reaches 524K before the early advantage is exhausted (≈940M tokens). | **Final `3.382004976`, passes the `3.3821` target by `0.000095`** at 2.700B tokens; 19,330.65 s training (5.370 h). **PASS (borderline, single seed).** ← **submission.** |

## The through-line

- **Systems track** (profiler-driven): exp007 (`3D` MLP) is the win; exp008/013/014/015 are the ablations that show *why* it isn't free, and exp018 points at the next target (vocab projection).
- **Optimization track** (batch/LR): exp001 → exp002 → exp012 build the case that early small batches help; exp021 exposes that a proxy win can die over the full horizon; exp024 fixes it with absolute-token transitions.
- **Optimizer track** (spectral): exp010 → exp011 → exp016 is a mechanism-first negative result — a real spectral phenomenon that no safe intervention could exploit.
- **Data track**: exp004-A → exp004-B tests, and rejects, a specific FineWeb-Edu late-concentration curriculum.

The single submission (exp024) is the endpoint of the optimization track, standing
on the exp007 systems win and the exp021 full-run failure.
