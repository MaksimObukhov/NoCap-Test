# exp024 Results and Reproduction

## Result

Exp024 uses a uniform `3D` GELU MLP and an absolute-token effective-batch
staircase of `64K -> 128K -> 256K -> 524K`.

| Metric | Value |
|---|---:|
| Status | Complete, seed 0 |
| Final validation loss | **3.382004976272583** |
| Challenge target | 3.3821 |
| Margin to target | 0.000095024 |
| Training tokens | 2,700,083,200 |
| Optimizer updates | 10,270 |
| Training time | 19,330.6477 s / 5.370 h |
| Compile-inclusive wall time | 19,787.8399 s / 5.497 h |
| Aggregate throughput | 139,678.88 tokens/s |
| Peak allocated GPU memory | 9,027 MiB |
| Parameters | 109,376,256 |
| W&B | [pplwt5ss](https://wandb.ai/m-obukhov-home/nocap-baseline/runs/pplwt5ss) |

The training timer is nominally 0.58% below the published 5.401 h baseline.
This is a challenge-style comparison, not a same-host hardware speedup. The
complete local wall time is reported separately.

Against my own reproduced baseline trajectory, exp024's recorded first target
crossing was 201.53 s earlier. That comparison is descriptive only: the two runs
used different rented hosts and the archived baseline metadata was
`git_dirty=true`.

## Frozen treatment

| Token interval | Effective batch | Accumulation | Updates |
|---|---:|---:|---:|
| `[0, 201326592)` | 65,536 | 4 | 3,072 |
| `[201326592, 469762048)` | 131,072 | 8 | 2,048 |
| `[469762048, 939524096)` | 262,144 | 16 | 1,792 |
| `[939524096, 2700083200)` | 524,288 | 32 | 3,358 |

The microbatch is 16 sequences by 1,024 tokens. The token clock is 262,144
tokens. At every phase, learning rate is the unchanged WSD base learning rate
multiplied by:

```text
sqrt(effective_batch_tokens / 524288)
```

Peak reference LR is 0.0018, AdamW betas are `(0.9, 0.95)`, weight decay is
0.1, and compute is BF16. No gradient clipping or FP8 treatment is enabled.

## Evaluated environment

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| World size | 1 |
| Python | 3.12 path in the rented image |
| PyTorch | `2.11.0+cu128` |
| CUDA reported by PyTorch | `12.8` |
| cuDNN reported by PyTorch | `91900` |
| W&B client | `0.28.1` |
| Git commit | `9e86007ff149f8070741822a9f2b6d65fcc8abfd` |
| Git dirty | `false` |

The run did not capture an installable PyTorch wheel URL or container digest.
`requirements.txt` therefore installs the recorded user-space dependencies but
deliberately leaves PyTorch to the CUDA image. The mathematical treatment and
launcher are reproducible from this branch, but exact timing reproduction
requires a compatible PyTorch 2.11 CUDA 12.8 image. This runtime-manifest gap is
a limitation of the submitted run.

## Run from a compatible RTX 4090 environment

```bash
git clone --branch submission/max-obukhov --single-branch https://github.com/MaksimObukhov/NoCap-Test.git
cd NoCap-Test
# First activate an environment that already provides PyTorch 2.11 + CUDA 12.8.
pip install -r requirements.txt
wandb login
# Optional: import the published BottleCap baseline run into your W&B account
# for a side-by-side reference.
wandb sync wandb/run-20250410_203158-64s1zc1w
python data/cached_fineweb10B.py
./run.sh
```

The bundled baseline run is only a reference; syncing it is not required to
launch exp024. The completed exp024 run is already available at the public W&B
link above.

`./run.sh` is intentionally the only fresh-run command. It selects exp024 full
seed 0, validates and hashes the dataset, refuses to overwrite an existing run,
logs to W&B, and saves immutable phase and warmdown checkpoints. The default
output root is `./runs`; set `NOCAP_OUTPUT_ROOT` to another absolute directory
if at least 25 GiB are available.

The command above follows the challenge's standard interface. For a timing
attempt, first verify that `python -c 'import torch; print(torch.__version__)'`
reports a compatible 2.11 CUDA 12.8 build and record any deviation.

To resume an interrupted run:

```bash
./run.sh full 0 /absolute/output/path --resume /absolute/checkpoint.pt
```

## Timing definitions

- `training_time_seconds` accumulates timed optimizer steps. Lazy compilation
  on the first executed training shape is included.
- `wall_time_seconds` starts before initial validation and includes validation,
  compilation, logging, checkpointing, and shape transitions.
- `tokens_per_second` is total target tokens divided by accumulated training
  time; it is not a same-host comparison with the archived baseline.

## Provenance and hashes

| Artifact | SHA-256 / identity |
|---|---|
| Dataset manifest | `d8b5d102aa39a21048e611d8bb02f12af0eb164d3aab9056fede22a9f9323582` |
| Final checkpoint | `180d25d8c6d79cdab628046359816c09d98dca4762255215ac2d87eeffafd328` |
| Final checkpoint size | 1,312,910,385 bytes |
| Final checkpoint state | next step 10,270; tokens 2,700,083,200 |
| Run ID | `632643a3557f49e08eb5f8e340433719` |
| W&B run ID | `pplwt5ss` |

The dataset manifest covers 50 FineWeb training shards and one validation
shard. The checkpoint itself is stored in the W&B artifact and is not committed
to Git.

## Compact result artifacts

- [Full validation trajectory](img/submission_full_trajectory.svg)
- [Profiler comparison](img/submission_profiler_comparison.svg)
- [Compact reproduced-baseline summary](results/baseline_summary.json)
- [Machine-readable exp024 summary](results/exp024/summary.json)
- [Compact profiler measurements](results/profiler_summary.csv)
- [Per-experiment overview](EXPERIMENTS.md)
- [W&B run and uploaded artifacts](https://wandb.ai/m-obukhov-home/nocap-baseline/runs/pplwt5ss)

Raw profiler traces, experiment W&B caches, night-suite working directories,
and multi-gigabyte checkpoints are deliberately excluded from the submission
branch. Their compact measurements and hashes are reported instead.
