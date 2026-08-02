"""Stage A: report a model's actual shape, tying and optimizer grouping.

Runs inside the experiment worktree and imports that worktree's own
train_gpt2.py, so what it measures is exactly what will train. It builds the
model on CPU, reports facts, and exits; the gate comparison against the
preregistered expectations happens in gates.accounting_gate.

This costs about thirty seconds and no GPU time. It exists because the two
most expensive kinds of mistake in the v2 suite were silent: a model whose
parameter count did not match what was preregistered, and an optimizer whose
parameter groups did not cover what was claimed. Both are cheap to check
before spending two GPU-hours finding out.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from train_gpt2 import GPT, GPTConfig, mlp_hidden_width  # noqa: E402


def build_report(args):
    config = GPTConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        mlp_ratio=args.mlp_ratio,
    )
    model = GPT(config)

    hidden_width = mlp_hidden_width(config)
    block = model.transformer.h[0]
    mlp_parameters = sum(p.numel() for p in block.mlp.parameters())

    # Tying is an identity property, not an equality property: the embedding
    # and the output projection must be the same tensor object, otherwise
    # they will silently diverge under weight decay or an optimizer that
    # keeps per-parameter state.
    tying_preserved = model.transformer.wte.weight is model.lm_head.weight

    report = {
        "mlp_ratio": args.mlp_ratio,
        "mlp_hidden_width": hidden_width,
        "mlp_parameters_per_layer": mlp_parameters,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "distinct_parameter_tensors": len(list(model.parameters())),
        "tying_preserved": tying_preserved,
        "embedding_parameter_count": model.transformer.wte.weight.numel(),
        "mlp_class": type(block.mlp).__name__,
    }

    optimizer_kwargs = {
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "betas": (0.9, 0.95),
        "device_type": "cpu",
    }
    if args.no_wd_tied_embedding:
        optimizer_kwargs["no_wd_tied_embedding"] = True
    optimizer = model.configure_optimizers(**optimizer_kwargs)

    grouped = []
    seen = {}
    duplicates = []
    for index, group in enumerate(optimizer.param_groups):
        count = sum(p.numel() for p in group["params"])
        grouped.append(
            {
                "index": index,
                "weight_decay": group.get("weight_decay"),
                "tensors": len(group["params"]),
                "parameters": count,
            }
        )
        for parameter in group["params"]:
            key = id(parameter)
            if key in seen:
                duplicates.append((seen[key], index))
            seen[key] = index

    model_parameter_ids = {id(p) for p in model.parameters()}
    covered = set(seen)
    report["optimizer_groups"] = grouped
    report["optimizer_group_count"] = len(grouped)
    report["optimizer_groups_cover_all_parameters"] = (
        covered == model_parameter_ids and not duplicates
    )
    report["optimizer_uncovered_tensors"] = len(model_parameter_ids - covered)
    report["optimizer_duplicated_tensors"] = len(duplicates)

    no_decay = [g for g in grouped if (g["weight_decay"] or 0.0) == 0.0]
    report["no_decay_parameter_count"] = sum(g["parameters"] for g in no_decay)
    report["no_decay_tensor_count"] = sum(g["tensors"] for g in no_decay)

    if args.reference_equivalence == "two_linear_swiglu":
        report["reference_equivalence"] = two_linear_swiglu_equivalence(block.mlp)

    return report


def two_linear_swiglu_equivalence(fused_mlp, tolerance=2e-2):
    """Check the fused SwiGLU against the two-Linear form it replaces.

    exp019's whole premise is that fusing the gate is a pure kernel-launch
    change and not a semantic one. That claim is cheap to falsify here: build
    the exp015-style two-Linear module, copy the fused weights into it by
    splitting c_fc, and compare outputs on random input.

    The tolerance is loose because both paths run in bf16 downstream; the
    check is for a wrong split or a swapped gate/value, which produce errors
    orders of magnitude larger than accumulation noise.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    hidden_width = fused_mlp.hidden_width
    n_embd = fused_mlp.c_proj.out_features

    gate = nn.Linear(n_embd, hidden_width, bias=False)
    value = nn.Linear(n_embd, hidden_width, bias=False)
    projection = nn.Linear(hidden_width, n_embd, bias=False)

    with torch.no_grad():
        fused_weight = fused_mlp.c_fc.weight
        gate.weight.copy_(fused_weight[:hidden_width])
        value.weight.copy_(fused_weight[hidden_width:])
        projection.weight.copy_(fused_mlp.c_proj.weight)

        sample = torch.randn(4, 128, n_embd)
        fused_out = fused_mlp(sample)
        reference_out = projection(F.silu(gate(sample)) * value(sample))
        difference = (fused_out - reference_out).abs().max().item()
        scale = reference_out.abs().max().item()

    return {
        "max_abs_difference": difference,
        "reference_max_abs": scale,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=0.0018)
    parser.add_argument("--no_wd_tied_embedding", action="store_true")
    parser.add_argument("--reference_equivalence", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    report = build_report(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = f"{args.output}.tmp"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
