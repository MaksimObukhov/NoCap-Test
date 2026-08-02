import os
import sys
import uuid
import math
import glob
import json
import random
import hashlib
import statistics
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # exp021 keeps the measured exp007/exp012 uniform 3D architecture.
        self.c_fc = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(3 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def state_dict(self):
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state):
        self.current_shard = state["current_shard"]
        self.current_position = state["current_position"]
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import argparse
    import time

    def git_metadata():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            return commit, dirty
        except (OSError, subprocess.CalledProcessError):
            return None, None

    def write_json_atomic(path, payload):
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary_path, path)

    print0(f"Running pytorch {torch.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin to train on",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin to eval validation loss on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="directory for this run's logs, summary, and checkpoint",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="custom",
        choices=["full", "proxy", "smoke", "custom"],
    )
    parser.add_argument("--resume", type=str, default="", help="checkpoint to resume")
    parser.add_argument(
        "--model", type=str, default="d12", help="d12|d24|d36|d48"
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="final number of gradient accumulation steps",
    )
    parser.add_argument(
        "--batch_ramp_start_accumulation_steps",
        type=int,
        default=0,
        help="initial accumulation steps; 0 disables the batch ramp",
    )
    parser.add_argument(
        "--batch_ramp_fraction",
        type=float,
        default=0.0,
        help="fraction of training tokens over which accumulation grows linearly",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    parser.add_argument("--seed", type=int, default=0)
    # workload (final-batch-equivalent updates; fixes the token budget)
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="target tokens expressed as updates at the final effective batch",
    )
    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="peak learning rate"
    )
    parser.add_argument(
        "--lr_reference_batch_tokens",
        type=int,
        default=0,
        help=(
            "batch-token denominator for sqrt(batch/reference) LR scaling; "
            "0 uses the final effective batch"
        ),
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=0,
        help="warmup tokens expressed as final-batch-equivalent updates",
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="warmdown tokens expressed as final-batch-equivalent updates",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--abort_on_nonfinite",
        action="store_true",
        help="abort if train/validation loss or global gradient norm is non-finite",
    )
    # evaluation and persistence
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=0,
        help="validate every N final-batch-equivalent updates",
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="validation batch size"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="save latest.pt every N final-batch-equivalent updates; 0 disables",
    )
    parser.add_argument(
        "--milestone_start_iter",
        type=int,
        default=0,
        help="first final-batch-equivalent update eligible for immutable milestones",
    )
    parser.add_argument(
        "--milestone_every",
        type=int,
        default=0,
        help="immutable milestone interval in final-batch-equivalent updates",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture a short CPU/CUDA profiler trace",
    )
    parser.add_argument(
        "--profile_wait_steps",
        type=int,
        default=3,
        help="optimizer steps to skip before profiler warmup",
    )
    parser.add_argument(
        "--profile_warmup_steps",
        type=int,
        default=1,
        help="profiler warmup steps to collect and discard",
    )
    parser.add_argument(
        "--profile_active_steps",
        type=int,
        default=1,
        help="optimizer steps to save in the profiler trace",
    )
    parser.add_argument("--log_wandb", action="store_true", help="log to W&B")
    parser.add_argument("--wandb_project", type=str, default="nocap-baseline")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument(
        "--wandb_checkpoint_artifact",
        action="store_true",
        help="upload final.pt as a W&B model artifact after a completed run",
    )
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
    assert args.num_iterations > 0
    assert args.warmup_iters + args.warmdown_iters <= args.num_iterations
    assert args.grad_accumulation_steps > 0
    assert args.batch_ramp_start_accumulation_steps >= 0
    assert 0.0 <= args.batch_ramp_fraction <= 1.0
    if args.batch_ramp_start_accumulation_steps == 0:
        assert args.batch_ramp_fraction == 0.0, (
            "--batch_ramp_fraction requires "
            "--batch_ramp_start_accumulation_steps"
        )
    else:
        assert args.batch_ramp_fraction > 0.0
        assert (
            args.batch_ramp_start_accumulation_steps
            <= args.grad_accumulation_steps
        ), "batch ramp must grow toward --grad_accumulation_steps"
    assert args.lr_reference_batch_tokens >= 0
    assert args.save_every >= 0
    assert args.milestone_start_iter >= 0
    assert args.milestone_every >= 0
    assert (args.milestone_start_iter == 0) == (args.milestone_every == 0)
    assert args.profile_wait_steps >= 0
    assert args.profile_warmup_steps >= 0
    assert args.profile_active_steps > 0
    profile_schedule_steps = (
        args.profile_wait_steps
        + args.profile_warmup_steps
        + args.profile_active_steps
    )
    if args.profile:
        assert args.num_iterations >= profile_schedule_steps, (
            "--num_iterations must cover profile wait + warmup + active steps"
        )
    assert torch.cuda.is_available(), "CUDA is required"

    # torchrun supplies these variables. WORLD_SIZE=1 is still used for this baseline.
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert args.grad_accumulation_steps % ddp_world_size == 0
    if args.batch_ramp_start_accumulation_steps > 0:
        assert args.batch_ramp_start_accumulation_steps % ddp_world_size == 0
    args.grad_accumulation_steps //= ddp_world_size
    args.batch_ramp_start_accumulation_steps //= ddp_world_size
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # All stochastic sources used by this program start from the recorded seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    resume_checkpoint = None
    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoints", "latest.pt")
        if not os.path.exists(resume_path):
            raise SystemExit(f"resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_checkpoint.get("is_final"):
            raise SystemExit(f"refusing to resume completed run: {resume_path}")
        resume_keys = (
            "seed",
            "run_mode",
            "model",
            "batch_size",
            "grad_accumulation_steps",
            "batch_ramp_start_accumulation_steps",
            "batch_ramp_fraction",
            "sequence_length",
            "num_iterations",
            "learning_rate",
            "lr_reference_batch_tokens",
            "warmup_iters",
            "warmdown_iters",
            "weight_decay",
            "milestone_start_iter",
            "milestone_every",
        )
        for key in resume_keys:
            checkpoint_value = resume_checkpoint["args"].get(key, 0)
            if checkpoint_value != getattr(args, key):
                raise ValueError(f"resume checkpoint does not match --{key}")

    run_id = (
        resume_checkpoint["run_id"]
        if resume_checkpoint is not None
        else uuid.uuid4().hex
    )
    if not args.run_name:
        args.run_name = f"{args.run_mode}-seed-{args.seed}"
    if not args.output_dir:
        args.output_dir = os.path.join("runs", run_id)

    git_commit, git_dirty = git_metadata()
    metadata = {
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "seed": args.seed,
        "gpu_name": torch.cuda.get_device_name(ddp_local_rank),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "world_size": ddp_world_size,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
    final_checkpoint_path = os.path.join(checkpoint_dir, "final.pt")
    summary_path = os.path.join(args.output_dir, "summary.json")
    if master_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        write_json_atomic(
            os.path.join(args.output_dir, "config.json"),
            {"args": vars(args), "metadata": metadata},
        )
        with open(os.path.join(args.output_dir, "train_gpt2.py"), "w") as f:
            f.write(code)
        if resume_checkpoint is None:
            with open(metrics_path, "w"):
                pass
        else:
            resume_tokens = resume_checkpoint.get("tokens_seen", 0)
            kept_records = []
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        if json.loads(line).get("tokens_seen", 0) <= resume_tokens:
                            kept_records.append(line)
            with open(metrics_path, "w") as f:
                f.writelines(kept_records)

    def log_local(payload):
        if master_process:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")

    wandb_run = None
    wandb_run_id = None
    if args.log_wandb and master_process:
        import wandb

        wandb_run_id = (
            resume_checkpoint.get("wandb_run_id")
            if resume_checkpoint is not None
            else None
        ) or wandb.util.generate_id()
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or None,
            name=args.run_name,
            id=wandb_run_id,
            resume="allow",
            config={"args": vars(args), "metadata": metadata},
        )
        wandb.save("train_gpt2.py", policy="now")
        wandb.save("run.sh", policy="now")

    print0(f"using device: {device} ({metadata['gpu_name']})")
    print0(f"run: {args.run_name} | seed: {args.seed} | output: {args.output_dir}")

    micro_batch_tokens = B * T * ddp_world_size
    final_tokens_per_update = micro_batch_tokens * args.grad_accumulation_steps
    target_tokens = args.num_iterations * final_tokens_per_update
    warmup_tokens = args.warmup_iters * final_tokens_per_update
    warmdown_tokens = args.warmdown_iters * final_tokens_per_update
    validation_interval_tokens = args.val_loss_every * final_tokens_per_update
    save_interval_tokens = args.save_every * final_tokens_per_update
    milestone_start_tokens = args.milestone_start_iter * final_tokens_per_update
    milestone_interval_tokens = args.milestone_every * final_tokens_per_update
    ramp_start_accumulation = (
        args.batch_ramp_start_accumulation_steps
        or args.grad_accumulation_steps
    )
    ramp_tokens = int(target_tokens * args.batch_ramp_fraction)
    lr_reference_batch_tokens = (
        args.lr_reference_batch_tokens or final_tokens_per_update
    )
    assert target_tokens % micro_batch_tokens == 0
    assert lr_reference_batch_tokens > 0
    print0(
        f"token budget: {target_tokens:,} | "
        f"microbatch: {micro_batch_tokens:,} tokens | "
        f"final effective batch: {final_tokens_per_update:,} tokens | "
        f"LR reference batch: {lr_reference_batch_tokens:,} tokens"
    )
    if ramp_start_accumulation != args.grad_accumulation_steps:
        print0(
            "batch ramp: "
            f"{ramp_start_accumulation} -> {args.grad_accumulation_steps} "
            f"microbatches over first {args.batch_ramp_fraction:.1%} of tokens"
        )
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val
    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )

    num_vocab = 50257
    model_config = {
        "d12": GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    base_model = GPT(model_config)
    if resume_checkpoint is not None:
        base_model.load_state_dict(resume_checkpoint["model"])
    base_model = base_model.train().cuda()

    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    print0("compiling the model...")
    model = torch.compile(base_model)
    model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = base_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
    )

    completed_steps = 0
    tokens_seen = 0
    last_validation_tokens = -1
    training_time_ms = 0.0
    previous_wall_time_seconds = 0.0
    final_val_loss = None
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        train_loader.load_state_dict(resume_checkpoint["train_loader"])
        x = resume_checkpoint["next_x"].to(device)
        y = resume_checkpoint["next_y"].to(device)
        completed_steps = resume_checkpoint["next_step"]
        tokens_seen = resume_checkpoint["tokens_seen"]
        last_validation_tokens = resume_checkpoint.get(
            "last_validation_tokens", tokens_seen
        )
        training_time_ms = resume_checkpoint["training_time_ms"]
        previous_wall_time_seconds = resume_checkpoint.get("wall_time_seconds", 0.0)
        final_val_loss = resume_checkpoint.get("last_val_loss")
        random.setstate(resume_checkpoint["rng_state"]["python"])
        np.random.set_state(resume_checkpoint["rng_state"]["numpy"])
        torch.set_rng_state(resume_checkpoint["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(resume_checkpoint["rng_state"]["cuda"])
        print0(
            f"resuming from completed update {completed_steps} "
            f"at {tokens_seen:,} tokens"
        )
    else:
        x, y = train_loader.next_batch()

    def accumulation_steps_at(current_tokens):
        if ramp_start_accumulation == args.grad_accumulation_steps:
            return args.grad_accumulation_steps
        if current_tokens >= ramp_tokens:
            return args.grad_accumulation_steps
        progress = current_tokens / ramp_tokens
        accumulation = ramp_start_accumulation + progress * (
            args.grad_accumulation_steps - ramp_start_accumulation
        )
        return math.floor(accumulation + 0.5)

    def base_lr_at(current_tokens, update_tokens):
        if current_tokens < warmup_tokens:
            return args.learning_rate * min(
                (current_tokens + update_tokens) / warmup_tokens, 1.0
            )
        if current_tokens < target_tokens - warmdown_tokens:
            return args.learning_rate
        return args.learning_rate * (
            (target_tokens - current_tokens) / warmdown_tokens
        )

    def phase_at(current_tokens):
        if current_tokens < warmup_tokens:
            return "warmup"
        if current_tokens < target_tokens - warmdown_tokens:
            return "steady"
        return "warmdown"

    session_wall_start = time.perf_counter()

    def elapsed_wall_seconds():
        return previous_wall_time_seconds + time.perf_counter() - session_wall_start

    def save_checkpoint(next_step, kind):
        if not master_process:
            return None
        if kind == "latest":
            destination = latest_checkpoint_path
        elif kind == "milestone":
            destination = os.path.join(
                checkpoint_dir, f"milestone-tokens{tokens_seen:012d}.pt"
            )
            if os.path.exists(destination):
                print0(f"milestone already exists, not rewriting: {destination}")
                return destination
        elif kind == "final":
            destination = final_checkpoint_path
        else:
            raise ValueError(f"unknown checkpoint kind: {kind}")
        checkpoint = {
            "version": 3,
            "run_id": run_id,
            "wandb_run_id": wandb_run_id,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_step": next_step,
            "tokens_seen": tokens_seen,
            "target_tokens": target_tokens,
            "last_validation_tokens": last_validation_tokens,
            "train_loader": train_loader.state_dict(),
            # The loader cursor is already beyond this prefetched batch.
            "next_x": x.cpu(),
            "next_y": y.cpu(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "training_time_ms": training_time_ms,
            "wall_time_seconds": elapsed_wall_seconds(),
            "last_val_loss": final_val_loss,
            "args": vars(args),
            "metadata": metadata,
            "code": code,
            "kind": kind,
            "is_final": kind == "final",
        }
        temporary_path = f"{destination}.tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination)
        print0(
            f"saved {kind} checkpoint: {destination} "
            f"(next update {next_step}, {tokens_seen:,} tokens)"
        )
        return destination

    profiler = None
    if args.profile:
        profile_dir = os.path.join(args.output_dir, "profile")
        os.makedirs(profile_dir, exist_ok=True)

        def save_profile_trace(prof):
            trace_path = os.path.join(profile_dir, f"rank{ddp_rank}_trace.json")
            table_path = os.path.join(profile_dir, f"rank{ddp_rank}_key_averages.txt")
            prof.export_chrome_trace(trace_path)
            with open(table_path, "w") as f:
                f.write(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=30,
                    )
                )
                f.write("\n")
            print0(f"saved profiler trace: {trace_path}")
            print0(f"saved profiler table: {table_path}")

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=args.profile_wait_steps,
                warmup=args.profile_warmup_steps,
                active=args.profile_active_steps,
                repeat=1,
            ),
            on_trace_ready=save_profile_trace,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.start()
        print0(
            "profiler enabled: "
            f"wait={args.profile_wait_steps}, "
            f"warmup={args.profile_warmup_steps}, "
            f"active={args.profile_active_steps}"
        )

    phase_grad_norms = {}

    while True:
        assert tokens_seen <= target_tokens
        last_step = tokens_seen == target_tokens
        validation_due = (
            args.val_loss_every > 0
            and (
                last_validation_tokens < 0
                or tokens_seen // validation_interval_tokens
                > last_validation_tokens // validation_interval_tokens
                or last_step
            )
        )

        if validation_due:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = torch.zeros(1, device=device)
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            final_val_loss = val_loss.item()
            validation_record = {
                "event": "validation",
                "step": completed_steps,
                "tokens_seen": tokens_seen,
                "val_loss": final_val_loss,
                "training_time_seconds": training_time_ms / 1000,
                "wall_time_seconds": elapsed_wall_seconds(),
            }
            print0(
                f"update:{completed_steps} | tokens:{tokens_seen:,}/"
                f"{target_tokens:,} | val loss {final_val_loss:.6f}"
            )
            log_local(validation_record)
            if wandb_run is not None:
                wandb_run.log(validation_record)
            last_validation_tokens = tokens_seen
            if args.abort_on_nonfinite and not math.isfinite(final_val_loss):
                if wandb_run is not None:
                    wandb_run.finish(exit_code=3)
                destroy_process_group()
                raise SystemExit("non-finite validation loss")

        if last_step:
            break

        remaining_microbatches = (
            target_tokens - tokens_seen
        ) // micro_batch_tokens
        current_accumulation_steps = min(
            accumulation_steps_at(tokens_seen), remaining_microbatches
        )
        assert current_accumulation_steps > 0
        update_tokens = current_accumulation_steps * micro_batch_tokens

        torch.cuda.synchronize()
        train_step_start = time.perf_counter()
        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(current_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == current_accumulation_steps - 1
            )
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / current_accumulation_steps
                train_loss += loss.detach()
            x, y = train_loader.next_batch()
            loss.backward()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()
        pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
            base_model.parameters(), float("inf"), error_if_nonfinite=False
        ).item()
        if args.abort_on_nonfinite and not (
            math.isfinite(lossf) and math.isfinite(pre_clip_grad_norm)
        ):
            abort_record = {
                "event": "abort",
                "reason": "nonfinite",
                "step": completed_steps,
                "tokens_seen": tokens_seen,
                "train_loss": lossf,
                "pre_clip_grad_norm": pre_clip_grad_norm,
                "training_time_seconds": training_time_ms / 1000,
                "wall_time_seconds": elapsed_wall_seconds(),
            }
            log_local(abort_record)
            if wandb_run is not None:
                wandb_run.log(abort_record)
                wandb_run.finish(exit_code=3)
            destroy_process_group()
            raise SystemExit("non-finite loss or gradient")

        base_lr = base_lr_at(tokens_seen, update_tokens)
        lr_batch_scale = math.sqrt(
            update_tokens / lr_reference_batch_tokens
        )
        lr = base_lr * lr_batch_scale
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_time_ms = 1000 * (time.perf_counter() - train_step_start)
        training_time_ms += step_time_ms

        previous_tokens_seen = tokens_seen
        phase = phase_at(previous_tokens_seen)
        tokens_seen += update_tokens
        completed_steps += 1
        phase_grad_norms.setdefault(phase, []).append(pre_clip_grad_norm)
        train_record = {
            "event": "train",
            "step": completed_steps,
            "tokens_seen": tokens_seen,
            "train_loss": lossf,
            "base_learning_rate": base_lr,
            "learning_rate": lr,
            "lr_batch_scale": lr_batch_scale,
            "grad_accumulation_steps": current_accumulation_steps,
            "effective_batch_tokens": update_tokens,
            "phase": phase,
            "pre_clip_grad_norm": pre_clip_grad_norm,
            "step_time_ms": step_time_ms,
            "training_time_seconds": training_time_ms / 1000,
            "tokens_per_second": tokens_seen / (training_time_ms / 1000),
            "wall_time_seconds": elapsed_wall_seconds(),
        }
        print0(
            f"update:{completed_steps} | tokens:{tokens_seen:,}/{target_tokens:,} | "
            f"batch:{update_tokens:,} | loss {lossf:.6f} | lr:{lr:.6g} | "
            f"|g|:{pre_clip_grad_norm:.4f} | "
            f"train_time:{training_time_ms/1000:.2f}s | step:{step_time_ms:.2f}ms"
        )
        log_local(train_record)
        if wandb_run is not None:
            wandb_run.log(train_record)

        crossed_save_boundary = (
            args.save_every > 0
            and tokens_seen // save_interval_tokens
            > previous_tokens_seen // save_interval_tokens
        )
        if crossed_save_boundary:
            save_checkpoint(completed_steps, kind="latest")

        crossed_milestone_boundary = False
        if args.milestone_every > 0 and tokens_seen >= milestone_start_tokens:
            if previous_tokens_seen < milestone_start_tokens:
                crossed_milestone_boundary = True
            else:
                crossed_milestone_boundary = (
                    (tokens_seen - milestone_start_tokens)
                    // milestone_interval_tokens
                    > (previous_tokens_seen - milestone_start_tokens)
                    // milestone_interval_tokens
                )
        if crossed_milestone_boundary:
            save_checkpoint(completed_steps, kind="milestone")

        if profiler is not None:
            # One profiler step is one complete optimizer update, including accumulation.
            profiler.step()

    if profiler is not None:
        profiler.stop()

    peak_memory_mib = torch.cuda.max_memory_allocated() // 1024 // 1024
    training_time_seconds = training_time_ms / 1000
    gradient_health_by_phase = {}
    for phase, values in sorted(phase_grad_norms.items()):
        ordered = sorted(values)
        gradient_health_by_phase[phase] = {
            "updates": len(ordered),
            "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": ordered[-1],
        }
    summary = {
        "status": "complete",
        "run_id": run_id,
        "run_name": args.run_name,
        "run_mode": args.run_mode,
        "seed": args.seed,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gpu_name": metadata["gpu_name"],
        "final_val_loss": final_val_loss,
        "num_iterations": args.num_iterations,
        "optimizer_updates": completed_steps,
        "target_tokens": target_tokens,
        "tokens_seen": tokens_seen,
        "training_time_seconds": training_time_seconds,
        "wall_time_seconds": elapsed_wall_seconds(),
        "tokens_per_second": tokens_seen / training_time_seconds,
        "peak_memory_mib": peak_memory_mib,
        "parameter_count": sum(p.numel() for p in base_model.parameters()),
        "mlp_ratio": 3.0,
        "lr_reference_batch_tokens": lr_reference_batch_tokens,
        "gradient_health_by_phase": gradient_health_by_phase,
    }
    print0(f"peak memory consumption: {peak_memory_mib} MiB")
    for phase, phase_stats in gradient_health_by_phase.items():
        print0(
            f"grad health [{phase}]: n={phase_stats['updates']} "
            f"median={phase_stats['median']:.4f} "
            f"p90={phase_stats['p90']:.4f} max={phase_stats['max']:.4f}"
        )
    final_val_text = "n/a" if final_val_loss is None else f"{final_val_loss:.6f}"
    print0(
        f"final val loss: {final_val_text} | tokens: {tokens_seen:,} | "
        f"optimizer updates: {completed_steps:,} | "
        f"train time: {training_time_seconds/3600:.3f}h | "
        f"throughput: {summary['tokens_per_second']:,.0f} tok/s"
    )

    if master_process:
        save_checkpoint(completed_steps, kind="final")
        digest = hashlib.sha256()
        with open(final_checkpoint_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        summary["final_checkpoint"] = {
            "path": final_checkpoint_path,
            "sha256": digest.hexdigest(),
            "bytes": os.path.getsize(final_checkpoint_path),
            "next_step": completed_steps,
            "tokens_seen": tokens_seen,
        }
        summary["milestone_checkpoints"] = [
            {"path": path, "bytes": os.path.getsize(path)}
            for path in sorted(
                glob.glob(os.path.join(checkpoint_dir, "milestone-tokens*.pt"))
            )
        ]
        write_json_atomic(summary_path, summary)
        if wandb_run is not None:
            wandb_run.summary.update(summary)
            wandb.save(summary_path, policy="now")
            if args.wandb_checkpoint_artifact:
                artifact = wandb.Artifact(
                    f"{args.run_name}-final-checkpoint",
                    type="model",
                    metadata={
                        "git_commit": git_commit,
                        "seed": args.seed,
                        "tokens_seen": tokens_seen,
                        "sha256": digest.hexdigest(),
                    },
                )
                artifact.add_file(final_checkpoint_path, name="final.pt")
                wandb_run.log_artifact(artifact)
            wandb_run.finish()

    destroy_process_group()
