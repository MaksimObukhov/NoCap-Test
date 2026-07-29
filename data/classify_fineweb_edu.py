#!/usr/bin/env python3
"""Score the exact FineWeb token subset consumed by the NoCap baseline.

The training loader does not consume one global prefix of the downloaded data.
It reads fixed-size micro-batches from each shard and drops the short remainder
that cannot form another batch. This script reproduces that selection before
classifying documents with HuggingFaceFW/fineweb-edu-classifier.

The initial version intentionally runs on CPU. A later GPU pass can change the
inference implementation without changing the selection or output schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize
MAGIC = 20240520
VERSION = 1
GPT2_EOT = 50256
MODEL_ID = "HuggingFaceFW/fineweb-edu-classifier"


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    token_count: int


@dataclass(frozen=True)
class SelectedSpan:
    shard_index: int
    path: Path
    start: int
    end: int
    micro_batches: int

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass
class Document:
    start_shard: int
    start_offset: int
    end_shard: int
    end_offset: int
    token_pieces: list[np.ndarray]
    selected_token_count: int
    starts_with_eot: bool

    @property
    def token_count(self) -> int:
        return sum(len(piece) for piece in self.token_pieces)

    def text_tokens(self) -> np.ndarray:
        if not self.token_pieces:
            return np.empty(0, dtype=np.uint16)
        tokens = (
            self.token_pieces[0]
            if len(self.token_pieces) == 1
            else np.concatenate(self.token_pieces)
        )
        if self.starts_with_eot and len(tokens) and tokens[0] == GPT2_EOT:
            return tokens[1:]
        return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify documents overlapping the exact token spans consumed by "
            "the single-GPU NoCap baseline."
        )
    )
    parser.add_argument(
        "--input-bin",
        default="data/fineweb10B/fineweb_train_*.bin",
        help="Glob for ordered training shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fineweb10B/fineweb_edu_scores.jsonl"),
        help="JSONL output. Existing files are not overwritten.",
    )
    parser.add_argument("--num-iterations", type=int, default=4768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--grad-accumulation-steps", type=int, default=32)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Stop after N selected documents (for a local smoke test).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Rounded int_score at or above this value is marked high quality.",
    )
    parser.add_argument(
        "--classifier-max-length",
        type=int,
        default=512,
        help="Maximum classifier-tokenizer length per document.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the exact shard/token selection without loading the model.",
    )
    return parser.parse_args()


def discover_shards(pattern: str) -> list[ShardInfo]:
    import glob

    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no files match --input-bin {pattern!r}")
    return [read_shard_info(path) for path in paths]


def read_shard_info(path: Path) -> ShardInfo:
    with path.open("rb") as file:
        header = np.frombuffer(file.read(HEADER_BYTES), dtype=np.int32)
    if len(header) != HEADER_INTS:
        raise ValueError(f"{path}: truncated header")
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path}: bad magic number {int(header[0])}")
    if int(header[1]) != VERSION:
        raise ValueError(f"{path}: unsupported version {int(header[1])}")

    token_count = int(header[2])
    expected_size = HEADER_BYTES + token_count * np.dtype(np.uint16).itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} bytes from header, found {actual_size}"
        )
    return ShardInfo(path=path, token_count=token_count)


def plan_loader_selection(
    shards: Sequence[ShardInfo],
    *,
    num_iterations: int,
    batch_size: int,
    sequence_length: int,
    grad_accumulation_steps: int,
) -> list[SelectedSpan]:
    values = {
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "grad_accumulation_steps": grad_accumulation_steps,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    tokens_per_micro_batch = batch_size * sequence_length
    remaining_micro_batches = num_iterations * grad_accumulation_steps
    spans: list[SelectedSpan] = []

    for shard_index, shard in enumerate(shards):
        # next_batch() needs B*T input tokens plus one token for shifted targets.
        available_micro_batches = (shard.token_count - 1) // tokens_per_micro_batch
        used_micro_batches = min(remaining_micro_batches, available_micro_batches)
        if used_micro_batches:
            spans.append(
                SelectedSpan(
                    shard_index=shard_index,
                    path=shard.path,
                    # next_batch() trains against y=buf[1:], so the counted
                    # training tokens are target positions [1, N + 1).
                    start=1,
                    end=used_micro_batches * tokens_per_micro_batch + 1,
                    micro_batches=used_micro_batches,
                )
            )
            remaining_micro_batches -= used_micro_batches
        if remaining_micro_batches == 0:
            break

    if remaining_micro_batches:
        raise ValueError(
            "the requested run would wrap around the available shards; "
            f"{remaining_micro_batches:,} micro-batches remain"
        )
    return spans


def selection_summary(
    spans: Sequence[SelectedSpan],
    *,
    num_iterations: int,
    batch_size: int,
    sequence_length: int,
    grad_accumulation_steps: int,
) -> dict:
    selected_tokens = sum(span.token_count for span in spans)
    return {
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "grad_accumulation_steps": grad_accumulation_steps,
        "tokens_per_micro_batch": batch_size * sequence_length,
        "tokens_per_optimizer_update": (
            batch_size * sequence_length * grad_accumulation_steps
        ),
        "selected_tokens": selected_tokens,
        "selected_shards": len(spans),
        "spans": [
            {
                "shard": span.path.name,
                "start": span.start,
                "end": span.end,
                "tokens": span.token_count,
                "micro_batches": span.micro_batches,
            }
            for span in spans
        ],
    }


def iter_selected_documents(
    shards: Sequence[ShardInfo], spans: Sequence[SelectedSpan]
) -> Iterator[Document]:
    """Yield full document text for every document touching selected tokens.

    FineWeb stores EOT at the *start* of each document, and preprocessing may
    split a document across shard boundaries. We therefore keep scanning the
    original stream through the first EOT after the final selected span.
    """

    if not spans:
        return
    selected_ranges = {
        span.shard_index: (span.start, span.end) for span in spans
    }
    final_selected_shard = spans[-1].shard_index

    current: Document | None = None
    can_stop_after_next_boundary = False

    for shard_index, shard in enumerate(shards):
        if shard_index > final_selected_shard and current is None:
            return

        tokens = np.memmap(
            shard.path,
            mode="r",
            dtype=np.uint16,
            offset=HEADER_BYTES,
            shape=(shard.token_count,),
        )
        boundaries = np.flatnonzero(tokens == GPT2_EOT)
        segment_start = 0

        for boundary in boundaries:
            boundary = int(boundary)
            if boundary > segment_start:
                current = append_segment(
                    current,
                    tokens[segment_start:boundary],
                    shard_index,
                    segment_start,
                    boundary,
                    selected_ranges.get(shard_index),
                    starts_with_eot=False,
                )

            if current is not None:
                if current.selected_token_count:
                    yield current
                if can_stop_after_next_boundary:
                    return

            current = append_segment(
                None,
                tokens[boundary : boundary + 1],
                shard_index,
                boundary,
                boundary + 1,
                selected_ranges.get(shard_index),
                starts_with_eot=True,
            )
            segment_start = boundary + 1

            if (
                shard_index == final_selected_shard
                and boundary >= selected_ranges[shard_index][1]
            ):
                can_stop_after_next_boundary = True

        if segment_start < shard.token_count:
            current = append_segment(
                current,
                tokens[segment_start:],
                shard_index,
                segment_start,
                shard.token_count,
                selected_ranges.get(shard_index),
                starts_with_eot=False,
            )

        if shard_index == final_selected_shard:
            can_stop_after_next_boundary = True

    if current is not None and current.selected_token_count:
        yield current


def append_segment(
    document: Document | None,
    tokens: np.ndarray,
    shard_index: int,
    start: int,
    end: int,
    selected_range: tuple[int, int] | None,
    *,
    starts_with_eot: bool,
) -> Document:
    if document is None:
        document = Document(
            start_shard=shard_index,
            start_offset=start,
            end_shard=shard_index,
            end_offset=end,
            token_pieces=[],
            selected_token_count=0,
            starts_with_eot=starts_with_eot,
        )
    document.token_pieces.append(np.asarray(tokens))
    document.end_shard = shard_index
    document.end_offset = end
    if selected_range is not None:
        selected_start, selected_end = selected_range
        document.selected_token_count += max(
            0, min(end, selected_end) - max(start, selected_start)
        )
    return document


def load_cpu_classifier():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "classification dependencies are missing; install them with:\n"
            "  uv pip install --python .venv-dev/bin/python transformers tiktoken"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to("cpu")
    return torch, tokenizer, model


def score_text(
    text: str,
    *,
    torch_module,
    tokenizer,
    model,
    max_length: int,
) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch_module.inference_mode():
        outputs = model(**inputs)
    return float(outputs.logits.squeeze().float().item())


def classify(
    shards: Sequence[ShardInfo],
    spans: Sequence[SelectedSpan],
    *,
    output: Path,
    threshold: int,
    classifier_max_length: int,
    max_documents: int | None,
    plan: dict,
) -> dict:
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; move or remove it before starting a new run"
        )
    if not 0 <= threshold <= 5:
        raise ValueError("--threshold must be between 0 and 5")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("--max-documents must be positive")

    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is missing; install it with:\n"
            "  uv pip install --python .venv-dev/bin/python tiktoken"
        ) from exc

    torch_module, tokenizer, model = load_cpu_classifier()
    decoder = tiktoken.get_encoding("gpt2")
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    documents = 0
    selected_tokens_scored = 0
    high_quality_selected_tokens = 0
    score_sum = 0.0
    score_min = math.inf
    score_max = -math.inf

    with output.open("x", encoding="utf-8") as file:
        for document in iter_selected_documents(shards, spans):
            text_token_ids = document.text_tokens()
            text = decoder.decode(text_token_ids.tolist())
            score = score_text(
                text,
                torch_module=torch_module,
                tokenizer=tokenizer,
                model=model,
                max_length=classifier_max_length,
            )
            int_score = int(round(max(0.0, min(score, 5.0))))
            is_high_quality = int_score >= threshold
            record = {
                "document_index": documents,
                "start_shard": shards[document.start_shard].path.name,
                "start_offset": document.start_offset,
                "end_shard": shards[document.end_shard].path.name,
                "end_offset": document.end_offset,
                "document_token_count": document.token_count,
                "selected_token_count": document.selected_token_count,
                "starts_with_eot": document.starts_with_eot,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "score": score,
                "int_score": int_score,
                "is_high_quality": is_high_quality,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

            documents += 1
            selected_tokens_scored += document.selected_token_count
            if is_high_quality:
                high_quality_selected_tokens += document.selected_token_count
            score_sum += score
            score_min = min(score_min, score)
            score_max = max(score_max, score)

            elapsed = time.monotonic() - started
            print(
                f"\rdocuments={documents:,} "
                f"selected_tokens={selected_tokens_scored:,} "
                f"docs/s={documents / max(elapsed, 1e-9):.2f}",
                end="",
                flush=True,
            )
            if max_documents is not None and documents >= max_documents:
                break
    print()

    elapsed = time.monotonic() - started
    completed = selected_tokens_scored == plan["selected_tokens"]
    summary = {
        "schema_version": 1,
        "model": MODEL_ID,
        "device": "cpu",
        "int_score_threshold": threshold,
        "classifier_max_length": classifier_max_length,
        "complete": completed,
        "documents_scored": documents,
        "selected_tokens_scored": selected_tokens_scored,
        "high_quality_selected_tokens": high_quality_selected_tokens,
        "high_quality_token_fraction": (
            high_quality_selected_tokens / selected_tokens_scored
            if selected_tokens_scored
            else None
        ),
        "score_mean": score_sum / documents if documents else None,
        "score_min": score_min if documents else None,
        "score_max": score_max if documents else None,
        "elapsed_seconds": elapsed,
        "documents_per_second": documents / elapsed if elapsed else None,
        "selection": plan,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    shards = discover_shards(args.input_bin)
    spans = plan_loader_selection(
        shards,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    plan = selection_summary(
        spans,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return 0

    summary = classify(
        shards,
        spans,
        output=args.output,
        threshold=args.threshold,
        classifier_max_length=args.classifier_max_length,
        max_documents=args.max_documents,
        plan=plan,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
