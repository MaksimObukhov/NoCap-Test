"""exp018: is an FP8 lm_head worth integrating?

Measurement only. No training, no model, no proxy. This exists because the
night-20260802-v2 profiler showed the vocab projection costs 702.2 ms of
every optimizer step and is invariant to within 0.2 ms across all six
profiled variants -- it is untouched by every MLP change tried so far, and
after 3D narrowing it is 23.4% of the step.

Integrating FP8 properly is real work: vocab 50257 is not divisible by 16, so
_scaled_mm needs a padded compute shape, and the weight is tied to wte, so
padding must not break tying. That work is only justified if the arithmetic
actually pays. Five GPU-minutes here decides it.

The reported number is projected FULL-STEP gain, not GEMM speedup. A 2x GEMM
that only touches 23% of the step is a 12% step gain at best, and the casts
are not free.
"""

import argparse
import json
import os
import statistics
import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(fn, warmup=10, iterations=50):
    """Median wall time of `fn` in milliseconds, warm caches, synchronized."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iterations):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append(1000.0 * (time.perf_counter() - start))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
    }


def bf16_measurements(tokens, n_embd, vocab, device):
    activations = torch.randn(tokens, n_embd, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n_embd, vocab, device=device, dtype=torch.bfloat16)
    grad_logits = torch.randn(tokens, vocab, device=device, dtype=torch.bfloat16)

    forward = time_callable(lambda: activations @ weight)
    # dgrad: (tokens, vocab) @ (vocab, n_embd)
    grad_input = time_callable(lambda: grad_logits @ weight.t())
    # wgrad: (n_embd, tokens) @ (tokens, vocab)
    grad_weight = time_callable(lambda: activations.t() @ grad_logits)
    return {
        "forward": forward,
        "grad_input": grad_input,
        "grad_weight": grad_weight,
        "total_median_ms": (
            forward["median_ms"]
            + grad_input["median_ms"]
            + grad_weight["median_ms"]
        ),
    }


def fp8_measurements(tokens, n_embd, vocab_padded, device):
    """FP8 forward via _scaled_mm, plus the cast/scale overhead it implies.

    Only the forward projection is measured in FP8. That is the honest scope:
    an FP8 backward needs a transposed-layout second copy of the weight, and
    if the forward alone does not pay there is no reason to price the rest.
    """
    if not hasattr(torch, "_scaled_mm"):
        return {"supported": False, "reason": "torch._scaled_mm is unavailable"}

    e4m3 = getattr(torch, "float8_e4m3fn", None)
    if e4m3 is None:
        return {"supported": False, "reason": "torch.float8_e4m3fn is unavailable"}

    if n_embd % 16 or vocab_padded % 16:
        return {
            "supported": False,
            "reason": (
                f"padded shape ({n_embd}, {vocab_padded}) is not 16-divisible"
            ),
        }

    activations = torch.randn(tokens, n_embd, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n_embd, vocab_padded, device=device, dtype=torch.bfloat16)

    scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)

    # _scaled_mm wants the right operand in column-major layout.
    weight_fp8 = weight.t().contiguous().t().to(e4m3)
    activations_fp8 = activations.to(e4m3)

    def gemm_only():
        return torch._scaled_mm(
            activations_fp8,
            weight_fp8,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    def cast_and_gemm():
        # What a real integration pays every micro-batch: the activation cast
        # and its amax reduction. The weight cast is amortized once per step,
        # so it is deliberately excluded here.
        amax = activations.abs().amax().to(torch.float32)
        dynamic_scale = (448.0 / amax.clamp(min=1e-12)).to(torch.float32)
        casted = (activations * dynamic_scale).to(e4m3)
        return torch._scaled_mm(
            casted,
            weight_fp8,
            scale_a=1.0 / dynamic_scale,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    try:
        gemm_only()
    except Exception as error:  # noqa: BLE001 - report, do not crash the suite
        return {"supported": False, "reason": f"{type(error).__name__}: {error}"}

    result = {"supported": True, "gemm_only": time_callable(gemm_only)}
    try:
        cast_and_gemm()
        result["cast_and_gemm"] = time_callable(cast_and_gemm)
    except Exception as error:  # noqa: BLE001
        result["cast_and_gemm"] = {"error": f"{type(error).__name__}: {error}"}

    reference = gemm_only()
    result["output_finite"] = bool(torch.isfinite(reference).all().item())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_per_microbatch", type=int, default=16384)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--padded_vocab_size", type=int, default=50272)
    parser.add_argument("--microbatches_per_step", type=int, default=32)
    parser.add_argument("--reference_step_time_ms", type=float, default=3731.0)
    parser.add_argument(
        "--minimum_projected_full_step_gain_pct", type=float, default=5.0
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "device": device,
        "torch_version": torch.__version__,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "shape": {
            "tokens_per_microbatch": args.tokens_per_microbatch,
            "n_embd": args.n_embd,
            "vocab_size": args.vocab_size,
            "padded_vocab_size": args.padded_vocab_size,
        },
        "reference_step_time_ms": args.reference_step_time_ms,
        "minimum_projected_full_step_gain_pct": (
            args.minimum_projected_full_step_gain_pct
        ),
    }

    if device == "cpu":
        report["decision"] = "invalid"
        report["reason"] = "no CUDA device; this measurement is GPU-only"
        _write(args.output, report)
        return

    report["bf16"] = bf16_measurements(
        args.tokens_per_microbatch, args.n_embd, args.vocab_size, device
    )
    report["fp8"] = fp8_measurements(
        args.tokens_per_microbatch, args.n_embd, args.padded_vocab_size, device
    )

    if not report["fp8"].get("supported"):
        report["decision"] = "kill"
        report["reason"] = (
            "FP8 path unavailable on this host: "
            f"{report['fp8'].get('reason')}"
        )
        _write(args.output, report)
        return

    bf16_forward = report["bf16"]["forward"]["median_ms"]
    fp8_forward = (
        report["fp8"].get("cast_and_gemm", {}).get("median_ms")
        or report["fp8"]["gemm_only"]["median_ms"]
    )
    saving_per_microbatch_ms = bf16_forward - fp8_forward
    saving_per_step_ms = saving_per_microbatch_ms * args.microbatches_per_step
    projected_gain_pct = 100.0 * saving_per_step_ms / args.reference_step_time_ms

    report["bf16_forward_median_ms"] = bf16_forward
    report["fp8_forward_median_ms"] = fp8_forward
    report["forward_speedup_pct"] = 100.0 * (bf16_forward / fp8_forward - 1.0)
    report["saving_per_step_ms"] = saving_per_step_ms
    report["projected_full_step_gain_pct"] = projected_gain_pct
    report["note"] = (
        "Forward projection only. A full integration would also convert the "
        "two backward GEMMs, so this is a lower bound on the achievable gain "
        "and an upper bound on the confidence that it is achievable."
    )

    if not report["fp8"].get("output_finite", False):
        report["decision"] = "invalid"
        report["reason"] = "FP8 output contained non-finite values"
    elif projected_gain_pct >= args.minimum_projected_full_step_gain_pct:
        report["decision"] = "pass"
        report["reason"] = (
            f"projected full-step gain {projected_gain_pct:.2f}% meets the "
            f"{args.minimum_projected_full_step_gain_pct:.1f}% bar; "
            "integration is authorised as a separate experiment"
        )
    else:
        report["decision"] = "kill"
        report["reason"] = (
            f"projected full-step gain {projected_gain_pct:.2f}% is below the "
            f"{args.minimum_projected_full_step_gain_pct:.1f}% bar; recorded "
            "as a measured negative"
        )

    _write(args.output, report)


def _write(path, report):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
