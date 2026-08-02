#!/usr/bin/env python3
"""exp023: bracketed FP8 tied-lm-head full-update integration benchmark."""

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


TRUE_VOCAB_SIZE = 50_257
PADDED_VOCAB_SIZE = 50_304
MODEL_DIM = 768
MICRO_BATCH_SIZE = 16
SEQUENCE_LENGTH = 1024
ACCUMULATION_STEPS = 16


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def git_metadata(worktree):
    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_dirty": subprocess.call(
            ["git", "-C", str(worktree), "diff", "--quiet"]
        )
        != 0
        or subprocess.call(
            ["git", "-C", str(worktree), "diff", "--cached", "--quiet"]
        )
        != 0,
    }


def load_winner_module(worktree):
    train_path = Path(worktree).resolve() / "train_gpt2.py"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    module_name = f"winner_train_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    original_argv0 = sys.argv[0]
    try:
        # The baseline snapshots argv[0] as provenance while importing.
        sys.argv[0] = str(train_path)
        spec.loader.exec_module(module)
    finally:
        sys.argv[0] = original_argv0
    return module


class TrueVocabHead(nn.Module):
    """Keep a 50,304-row compute weight while exposing 50,257 logits."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, inputs):
        return self.linear(inputs)[..., :TRUE_VOCAB_SIZE]


def build_model(winner, use_fp8):
    config = winner.GPTConfig(
        vocab_size=PADDED_VOCAB_SIZE,
        n_layer=12,
        n_head=12,
        n_embd=MODEL_DIM,
    )
    model = winner.GPT(config).train().cuda()
    original_linear = model.lm_head
    master_weight = model.transformer.wte.weight
    if original_linear.weight is not master_weight:
        raise RuntimeError("winner model did not preserve initial weight tying")
    model.lm_head = TrueVocabHead(original_linear)

    converted_module = None
    if use_fp8:
        try:
            from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        except ImportError as error:
            raise RuntimeError(
                "torchao 0.17.0 is required; install requirements-fp8.txt first"
            ) from error
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            model,
            config=config,
            module_filter_fn=lambda module, fqn: fqn == "lm_head.linear",
        )
        converted_module = type(model.lm_head.linear).__name__
        if "Float8Linear" not in converted_module:
            raise RuntimeError(f"lm_head was not converted: {converted_module}")
        # Conversion is expected to preserve the Parameter. Retie explicitly if
        # this TorchAO release swaps it, then verify the single BF16 master.
        if model.lm_head.linear.weight is not master_weight:
            model.lm_head.linear.weight = master_weight

    if model.transformer.wte.weight is not model.lm_head.weight:
        raise RuntimeError("padded embedding/lm_head tying was broken")
    if model.lm_head.weight.shape != (PADDED_VOCAB_SIZE, MODEL_DIM):
        raise RuntimeError(f"unexpected master shape: {model.lm_head.weight.shape}")
    if model.lm_head.weight.dtype != torch.float32:
        raise RuntimeError(f"master weight must remain FP32, got {model.lm_head.weight.dtype}")
    return model, converted_module


def run_internal_stage(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(23000)
    torch.cuda.manual_seed_all(23000)
    torch.backends.cudnn.benchmark = False

    winner_worktree = Path(args.winner_worktree).resolve()
    source = git_metadata(winner_worktree)
    if source["tracked_dirty"]:
        raise RuntimeError("winner worktree has tracked changes")
    winner = load_winner_module(winner_worktree)
    use_fp8 = args.stage == "treatment"
    base_model, converted_module = build_model(winner, use_fp8)
    parameter_ids = [id(parameter) for parameter in base_model.parameters()]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("duplicate Parameter registration after tying")

    optimizer = base_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.0012727922061357855,
        betas=(0.9, 0.95),
        device_type=device,
    )
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if set(optimizer_parameter_ids) != set(parameter_ids):
        raise RuntimeError("optimizer coverage changed after FP8 conversion")

    compiled = torch.compile(base_model)
    model = DDP(compiled, device_ids=[local_rank])
    generator = torch.Generator(device="cuda").manual_seed(23100)
    inputs = torch.randint(
        0,
        TRUE_VOCAB_SIZE,
        (MICRO_BATCH_SIZE, SEQUENCE_LENGTH),
        device="cuda",
        generator=generator,
    )
    targets = torch.randint(
        0,
        TRUE_VOCAB_SIZE,
        (MICRO_BATCH_SIZE, SEQUENCE_LENGTH),
        device="cuda",
        generator=generator,
    )
    if int(targets.max()) >= TRUE_VOCAB_SIZE:
        raise RuntimeError("padded token leaked into targets")

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), autocast:
        shape_logits, shape_loss = base_model(inputs, targets, return_logits=True)
    output_vocab_size = shape_logits.shape[-1]
    if output_vocab_size != TRUE_VOCAB_SIZE or not math.isfinite(float(shape_loss)):
        raise RuntimeError("untimed true-vocab forward check failed")
    del shape_logits, shape_loss
    torch.cuda.empty_cache()

    step_times = []
    first_loss = None
    first_grad_norm = None
    first_head_grad_norm = None
    started = time.perf_counter()
    for step in range(args.steps):
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device="cuda")
        for micro_step in range(ACCUMULATION_STEPS):
            model.require_backward_grad_sync = micro_step == ACCUMULATION_STEPS - 1
            with autocast:
                _logits, loss = model(inputs, targets, return_logits=False)
                loss = loss / ACCUMULATION_STEPS
                accumulated_loss += loss.detach()
            loss.backward()
        gradient_clip = 10.0 if args.winner == "exp022" else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            base_model.parameters(), gradient_clip, error_if_nonfinite=True
        )
        if first_loss is None:
            first_loss = float(accumulated_loss)
            first_grad_norm = float(grad_norm)
            first_head_grad_norm = float(base_model.lm_head.weight.grad.float().norm())
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(1000.0 * (time.perf_counter() - step_started))
        print(
            f"{args.stage} step {step + 1}/{args.steps}: "
            f"loss={float(accumulated_loss):.6f} time={step_times[-1]:.3f}ms",
            flush=True,
        )

    if output_vocab_size != TRUE_VOCAB_SIZE:
        raise RuntimeError(f"loss saw {output_vocab_size} classes")
    if not all(math.isfinite(value) for value in (first_loss, first_grad_norm, first_head_grad_norm)):
        raise RuntimeError("non-finite one-step numerical diagnostic")
    timed = step_times[args.warmup_steps :]
    if not timed:
        raise RuntimeError("no timed steps after warmup")
    graph_breaks = sum(torch._dynamo.utils.counters.get("graph_break", {}).values())
    payload = {
        "stage": args.stage,
        "winner": args.winner,
        "winner_source": source,
        "fp8": use_fp8,
        "converted_module": converted_module,
        "true_vocab_size": TRUE_VOCAB_SIZE,
        "padded_vocab_size": PADDED_VOCAB_SIZE,
        "master_dtype": str(base_model.lm_head.weight.dtype),
        "weight_tied": base_model.transformer.wte.weight is base_model.lm_head.weight,
        "output_vocab_size": output_vocab_size,
        "parameter_count": sum(parameter.numel() for parameter in base_model.parameters()),
        "first_loss": first_loss,
        "first_grad_norm": first_grad_norm,
        "first_head_grad_norm": first_head_grad_norm,
        "step_times_ms": step_times,
        "warmup_steps": args.warmup_steps,
        "median_steady_step_ms": statistics.median(timed),
        "end_to_end_seconds": time.perf_counter() - started,
        "graph_breaks": graph_breaks,
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "torch_version": torch.__version__,
    }
    write_json(args.output, payload)
    torch.distributed.destroy_process_group()


def run_subprocess(command, environment, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"stage subprocess failed with exit {return_code}")


def run_parent(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    stage_summaries = {}
    for stage in ("control-before", "treatment", "control-after"):
        stage_dir = output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(output_dir / "compile-cache" / stage)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            str(script),
            "--stage",
            stage,
            "--winner",
            args.winner,
            "--winner-worktree",
            str(Path(args.winner_worktree).resolve()),
            "--output",
            str(stage_dir / "summary.json"),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
        ]
        run_subprocess(command, environment, stage_dir / "stdout.log")
        stage_summaries[stage] = json.loads((stage_dir / "summary.json").read_text())

    medians = {
        stage: summary["median_steady_step_ms"]
        for stage, summary in stage_summaries.items()
    }
    control_median = statistics.median(
        (medians["control-before"], medians["control-after"])
    )
    gain_pct = 100.0 * (1.0 - medians["treatment"] / control_median)
    control_drift_pct = 100.0 * abs(
        medians["control-after"] / medians["control-before"] - 1.0
    )
    numerical = {
        "loss_abs_delta": abs(
            stage_summaries["treatment"]["first_loss"]
            - stage_summaries["control-before"]["first_loss"]
        ),
        "grad_norm_relative_delta": abs(
            stage_summaries["treatment"]["first_grad_norm"]
            / stage_summaries["control-before"]["first_grad_norm"]
            - 1.0
        ),
        "head_grad_norm_relative_delta": abs(
            stage_summaries["treatment"]["first_head_grad_norm"]
            / stage_summaries["control-before"]["first_head_grad_norm"]
            - 1.0
        ),
    }
    checks = {
        "complete_step_gain_ge_3pct": gain_pct >= 3.0,
        "control_drift_le_2pct": control_drift_pct <= 2.0,
        "weight_tying_preserved": all(
            summary["weight_tied"] for summary in stage_summaries.values()
        ),
        "true_vocab_loss": all(
            summary["output_vocab_size"] == TRUE_VOCAB_SIZE
            for summary in stage_summaries.values()
        ),
        "fp8_module_active": "Float8Linear"
        in (stage_summaries["treatment"]["converted_module"] or ""),
        "no_graph_breaks": all(
            summary["graph_breaks"] == 0 for summary in stage_summaries.values()
        ),
    }
    payload = {
        "experiment": "exp023",
        "winner": args.winner,
        "decision": "pass" if all(checks.values()) else "kill",
        "median_steady_step_ms": medians,
        "control_median_step_ms": control_median,
        "fp8_complete_step_gain_pct": gain_pct,
        "control_drift_pct": control_drift_pct,
        "one_step_numerical_delta": numerical,
        "checks": checks,
        "stages": stage_summaries,
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", required=True, choices=("exp021", "exp022"))
    parser.add_argument("--winner-worktree", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--stage", choices=("control-before", "treatment", "control-after")
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.steps <= args.warmup_steps:
        parser.error("--steps must exceed --warmup-steps")
    if args.stage and not args.output:
        parser.error("internal --stage requires --output")
    if not args.stage and not args.output_dir:
        parser.error("parent mode requires --output-dir")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stage:
        run_internal_stage(arguments)
    else:
        run_parent(arguments)
