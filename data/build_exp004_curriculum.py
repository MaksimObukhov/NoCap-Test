#!/usr/bin/env python3
"""Build the exact exp004-B proxy stream from FineWeb-Edu document scores.

The builder changes only the order of target-token occurrences. It reconstructs
the same proxy prefix consumed by the baseline loader, emits the first 1,404
updates from score-0/1/2 documents, then proportionally merges every score-3/4/5
document with the remaining score-0/1/2 documents over the 384-update warmdown.

Output shards use the baseline binary format. Each shard contains one
context-only prefix token followed by an exact multiple of B*T target tokens, so
the existing loader neither duplicates nor drops a target at shard boundaries.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize
MAGIC = 20240520
VERSION = 1
DEFAULT_THRESHOLD = 3
DEFAULT_EXPECTED_HIGH_TOKENS = 77_677_516


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


@dataclass(frozen=True)
class SourceSegment:
    shard_index: int
    start: int
    end: int

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class DocumentSlice:
    document_index: int
    int_score: int
    segments: tuple[SourceSegment, ...]

    @property
    def token_count(self) -> int:
        return sum(segment.token_count for segment in self.segments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the pre-materialised exp004-B proxy stream."
    )
    parser.add_argument(
        "--input-bin",
        default="data/fineweb10B/fineweb_train_*.bin",
        help="Glob for the original ordered FineWeb training shards.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "data/nocap-fineweb-edu/data/fineweb10B/classification/"
            "fineweb_edu_scores.jsonl"
        ),
    )
    parser.add_argument(
        "--classification-manifest",
        type=Path,
        default=Path(
            "data/nocap-fineweb-edu/data/fineweb10B/classification/"
            "fineweb_edu_scores.jsonl.manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fineweb10B/exp004-B-proxy"),
    )
    parser.add_argument("--num-iterations", type=int, default=1788)
    parser.add_argument("--warmdown-iters", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--grad-accumulation-steps", type=int, default=32)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--micro-batches-per-output-shard",
        type=int,
        default=6103,
        help="6103 reproduces the baseline's approximately 100M-token shards.",
    )
    parser.add_argument(
        "--expected-high-tokens",
        type=int,
        default=DEFAULT_EXPECTED_HIGH_TOKENS,
        help="Set to -1 to disable the production-artifact assertion.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate scores and print accounting without writing output shards.",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        metavar="MANIFEST",
        help="Verify an existing output manifest and its shard hashes.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_metadata() -> dict:
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
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def atomic_write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


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
    expected_bytes = HEADER_BYTES + token_count * np.dtype(np.uint16).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path}: header describes {expected_bytes} bytes, "
            f"found {path.stat().st_size}"
        )
    return ShardInfo(path=path, token_count=token_count)


def discover_shards(pattern: str) -> list[ShardInfo]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no files match --input-bin {pattern!r}")
    return [read_shard_info(path) for path in paths]


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
        available = (shard.token_count - 1) // tokens_per_micro_batch
        used = min(available, remaining_micro_batches)
        if used:
            spans.append(
                SelectedSpan(
                    shard_index=shard_index,
                    path=shard.path,
                    start=1,
                    end=1 + used * tokens_per_micro_batch,
                    micro_batches=used,
                )
            )
            remaining_micro_batches -= used
        if remaining_micro_batches == 0:
            break
    if remaining_micro_batches:
        raise ValueError(
            "requested proxy would wrap around the source shards; "
            f"{remaining_micro_batches:,} micro-batches remain"
        )
    return spans


def selection_record(
    spans: Sequence[SelectedSpan],
    *,
    num_iterations: int,
    batch_size: int,
    sequence_length: int,
    grad_accumulation_steps: int,
) -> dict:
    record = {
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "grad_accumulation_steps": grad_accumulation_steps,
        "tokens_per_micro_batch": batch_size * sequence_length,
        "tokens_per_optimizer_update": (
            batch_size * sequence_length * grad_accumulation_steps
        ),
        "selected_tokens": sum(span.token_count for span in spans),
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
    record["selection_sha256"] = canonical_sha256(record)
    return record


def record_segments(
    record: dict,
    shards: Sequence[ShardInfo],
    shard_name_to_index: dict[str, int],
    selected_ranges: dict[int, tuple[int, int]],
) -> tuple[SourceSegment, ...]:
    try:
        first = shard_name_to_index[record["start_shard"]]
        last = shard_name_to_index[record["end_shard"]]
    except KeyError as exc:
        raise ValueError(f"score record references unknown shard {exc.args[0]!r}") from exc
    if last < first:
        raise ValueError(f"document {record['document_index']}: reversed shard span")

    segments: list[SourceSegment] = []
    for shard_index in range(first, last + 1):
        document_start = int(record["start_offset"]) if shard_index == first else 0
        document_end = (
            int(record["end_offset"])
            if shard_index == last
            else shards[shard_index].token_count
        )
        selected_range = selected_ranges.get(shard_index)
        if selected_range is None:
            continue
        selected_start, selected_end = selected_range
        start = max(document_start, selected_start)
        end = min(document_end, selected_end)
        if start < end:
            segments.append(SourceSegment(shard_index, start, end))
    return tuple(segments)


def split_segments(
    segments: Sequence[SourceSegment], prefix_tokens: int
) -> tuple[tuple[SourceSegment, ...], tuple[SourceSegment, ...]]:
    total = sum(segment.token_count for segment in segments)
    if not 0 <= prefix_tokens <= total:
        raise ValueError(f"cannot split {total} tokens at {prefix_tokens}")

    prefix: list[SourceSegment] = []
    suffix: list[SourceSegment] = []
    remaining = prefix_tokens
    for segment in segments:
        if remaining == 0:
            suffix.append(segment)
        elif remaining >= segment.token_count:
            prefix.append(segment)
            remaining -= segment.token_count
        else:
            cut = segment.start + remaining
            prefix.append(SourceSegment(segment.shard_index, segment.start, cut))
            suffix.append(SourceSegment(segment.shard_index, cut, segment.end))
            remaining = 0
    return tuple(prefix), tuple(suffix)


def split_document(
    document: DocumentSlice, prefix_tokens: int
) -> tuple[DocumentSlice | None, DocumentSlice | None]:
    prefix, suffix = split_segments(document.segments, prefix_tokens)
    prefix_document = (
        DocumentSlice(document.document_index, document.int_score, prefix)
        if prefix
        else None
    )
    suffix_document = (
        DocumentSlice(document.document_index, document.int_score, suffix)
        if suffix
        else None
    )
    return prefix_document, suffix_document


def proportional_merge(
    high_documents: Sequence[DocumentSlice],
    general_documents: Sequence[DocumentSlice],
) -> Iterator[DocumentSlice]:
    high_total = sum(document.token_count for document in high_documents)
    general_total = sum(document.token_count for document in general_documents)
    high_index = 0
    general_index = 0
    emitted_high = 0
    emitted_general = 0

    while high_index < len(high_documents) or general_index < len(general_documents):
        if high_index == len(high_documents):
            choose_high = False
        elif general_index == len(general_documents):
            choose_high = True
        else:
            # Compare normalised progress without floating-point rounding.
            choose_high = (
                emitted_high * general_total <= emitted_general * high_total
            )

        if choose_high:
            document = high_documents[high_index]
            high_index += 1
            emitted_high += document.token_count
        else:
            document = general_documents[general_index]
            general_index += 1
            emitted_general += document.token_count
        yield document


class SourceTokens:
    def __init__(self, shards: Sequence[ShardInfo]):
        self.shards = shards
        self._maps: dict[int, np.memmap] = {}

    def shard(self, shard_index: int) -> np.memmap:
        if shard_index not in self._maps:
            info = self.shards[shard_index]
            self._maps[shard_index] = np.memmap(
                info.path,
                mode="r",
                dtype=np.uint16,
                offset=HEADER_BYTES,
                shape=(info.token_count,),
            )
        return self._maps[shard_index]

    def segment(self, segment: SourceSegment) -> np.ndarray:
        return self.shard(segment.shard_index)[segment.start : segment.end]

    def preceding_token(self, segment: SourceSegment) -> int:
        if segment.start == 0:
            raise ValueError("selected target segment has no preceding source token")
        return int(self.shard(segment.shard_index)[segment.start - 1])


class OutputShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        total_target_tokens: int,
        targets_per_shard: int,
        prefix: str = "exp004_train",
    ):
        if targets_per_shard <= 0:
            raise ValueError("targets_per_shard must be positive")
        if total_target_tokens % targets_per_shard == 0:
            self.expected_shards = total_target_tokens // targets_per_shard
        else:
            self.expected_shards = total_target_tokens // targets_per_shard + 1
        self.output_dir = output_dir
        self.total_target_tokens = total_target_tokens
        self.targets_per_shard = targets_per_shard
        self.prefix = prefix
        self.total_written = 0
        self.previous_target: int | None = None
        self.first_context: int | None = None
        self._file = None
        self._digest = None
        self._path: Path | None = None
        self._shard_target_capacity = 0
        self._shard_targets_written = 0
        self.shards: list[dict] = []

    def _start_shard(self, first_context_token: int | None) -> None:
        shard_index = len(self.shards) + 1
        remaining = self.total_target_tokens - self.total_written
        self._shard_target_capacity = min(self.targets_per_shard, remaining)
        if self._shard_target_capacity <= 0:
            raise ValueError("attempted to open an output shard after the budget")

        if self.previous_target is None:
            if first_context_token is None:
                raise ValueError("the first output target needs a context token")
            context_token = first_context_token
            self.first_context = context_token
        else:
            context_token = self.previous_target

        self._path = self.output_dir / f"{self.prefix}_{shard_index:06d}.bin"
        self._file = self._path.open("xb")
        self._digest = hashlib.sha256()
        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[0] = MAGIC
        header[1] = VERSION
        header[2] = self._shard_target_capacity + 1
        self._write_bytes(header.tobytes())
        self._write_bytes(np.asarray([context_token], dtype=np.uint16).tobytes())
        self._shard_targets_written = 0

    def _write_bytes(self, payload: bytes) -> None:
        assert self._file is not None
        assert self._digest is not None
        self._file.write(payload)
        self._digest.update(payload)

    def _finish_shard(self) -> None:
        assert self._file is not None
        assert self._digest is not None
        assert self._path is not None
        if self._shard_targets_written != self._shard_target_capacity:
            raise ValueError(
                f"{self._path}: wrote {self._shard_targets_written} targets, "
                f"expected {self._shard_target_capacity}"
            )
        self._file.close()
        expected_bytes = HEADER_BYTES + 2 * (self._shard_target_capacity + 1)
        actual_bytes = self._path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{self._path}: wrote {actual_bytes} bytes, expected {expected_bytes}"
            )
        self.shards.append(
            {
                "file": self._path.name,
                "target_tokens": self._shard_target_capacity,
                "stored_tokens": self._shard_target_capacity + 1,
                "bytes": actual_bytes,
                "sha256": self._digest.hexdigest(),
            }
        )
        self._file = None
        self._digest = None
        self._path = None

    def emit(self, tokens: np.ndarray, *, first_context_token: int | None = None) -> None:
        tokens = np.asarray(tokens, dtype=np.uint16)
        position = 0
        while position < len(tokens):
            if self._file is None:
                self._start_shard(first_context_token)
            available = self._shard_target_capacity - self._shard_targets_written
            count = min(available, len(tokens) - position)
            chunk = np.ascontiguousarray(tokens[position : position + count])
            self._write_bytes(chunk.tobytes())
            self.previous_target = int(chunk[-1])
            self._shard_targets_written += count
            self.total_written += count
            position += count
            if self._shard_targets_written == self._shard_target_capacity:
                self._finish_shard()

    def finish(self) -> list[dict]:
        if self._file is not None:
            self._finish_shard()
        if self.total_written != self.total_target_tokens:
            raise ValueError(
                f"wrote {self.total_written:,} targets, "
                f"expected {self.total_target_tokens:,}"
            )
        if len(self.shards) != self.expected_shards:
            raise ValueError(
                f"wrote {len(self.shards)} shards, expected {self.expected_shards}"
            )
        return self.shards


def emit_document(
    writer: OutputShardWriter,
    source: SourceTokens,
    document: DocumentSlice,
) -> None:
    first_segment = True
    for segment in document.segments:
        first_context = (
            source.preceding_token(segment)
            if writer.total_written == 0 and first_segment
            else None
        )
        writer.emit(source.segment(segment), first_context_token=first_context)
        first_segment = False


def update_emission_digest(
    digest,
    document: DocumentSlice,
    *,
    stage_index: int,
) -> None:
    record = {
        "stage_index": stage_index,
        "document_index": document.document_index,
        "int_score": document.int_score,
        "segments": [
            [segment.shard_index, segment.start, segment.end]
            for segment in document.segments
        ],
    }
    digest.update(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")


def validate_classification_manifest(
    path: Path,
    *,
    threshold: int,
    source_selection: dict,
) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value["int_score_threshold"]) != threshold:
        raise ValueError(
            f"{path}: threshold {value['int_score_threshold']} does not match {threshold}"
        )
    classified_selection = value["selection"]
    proxy_spans = source_selection["spans"]
    classified_by_shard = {
        span["shard"]: (int(span["start"]), int(span["end"]))
        for span in classified_selection["spans"]
    }
    for span in proxy_spans:
        classified_range = classified_by_shard.get(span["shard"])
        if classified_range is None:
            raise ValueError(f"{path}: proxy shard {span['shard']} was not classified")
        if classified_range[0] > span["start"] or classified_range[1] < span["end"]:
            raise ValueError(
                f"{path}: classified range {classified_range} does not cover "
                f"proxy range {(span['start'], span['end'])}"
            )
    return value


def verify_output_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_payload_sha256 = manifest.get("manifest_payload_sha256")
    payload_without_hash = dict(manifest)
    payload_without_hash.pop("manifest_payload_sha256", None)
    if recorded_payload_sha256 != canonical_sha256(payload_without_hash):
        raise ValueError(f"{manifest_path}: manifest payload SHA-256 mismatch")

    output_dir = manifest_path.parent
    total_targets = 0
    for shard in manifest["output"]["shards"]:
        path = output_dir / shard["file"]
        info = read_shard_info(path)
        if info.token_count != int(shard["stored_tokens"]):
            raise ValueError(
                f"{path}: stored token count {info.token_count} does not match manifest"
            )
        actual_sha = sha256_file(path)
        if actual_sha != shard["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch")
        if int(shard["target_tokens"]) != info.token_count - 1:
            raise ValueError(f"{path}: target-token count does not match shard header")
        total_targets += int(shard["target_tokens"])
    if total_targets != int(manifest["accounting"]["output_target_tokens"]):
        raise ValueError("output target total does not match manifest accounting")
    print(
        f"verified {len(manifest['output']['shards'])} shards and "
        f"{total_targets:,} target tokens"
    )
    return manifest


def build_curriculum(
    *,
    input_pattern: str,
    scores_path: Path,
    classification_manifest_path: Path,
    output_dir: Path,
    num_iterations: int,
    warmdown_iters: int,
    batch_size: int,
    sequence_length: int,
    grad_accumulation_steps: int,
    threshold: int,
    micro_batches_per_output_shard: int,
    expected_high_tokens: int | None,
    plan_only: bool,
) -> dict:
    if not 0 < warmdown_iters < num_iterations:
        raise ValueError("warmdown_iters must be between 0 and num_iterations")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    start_time = time.perf_counter()
    shards = discover_shards(input_pattern)
    spans = plan_loader_selection(
        shards,
        num_iterations=num_iterations,
        batch_size=batch_size,
        sequence_length=sequence_length,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    source_selection = selection_record(
        spans,
        num_iterations=num_iterations,
        batch_size=batch_size,
        sequence_length=sequence_length,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    classification_manifest = validate_classification_manifest(
        classification_manifest_path,
        threshold=threshold,
        source_selection=source_selection,
    )

    tokens_per_micro_batch = batch_size * sequence_length
    tokens_per_update = tokens_per_micro_batch * grad_accumulation_steps
    total_target_tokens = num_iterations * tokens_per_update
    general_stage_updates = num_iterations - warmdown_iters
    general_stage_tokens = general_stage_updates * tokens_per_update
    warmdown_tokens = warmdown_iters * tokens_per_update
    if source_selection["selected_tokens"] != total_target_tokens:
        raise ValueError("source-selection token count does not match run budget")

    selected_ranges = {
        span.shard_index: (span.start, span.end) for span in spans
    }
    shard_name_to_index = {shard.path.name: index for index, shard in enumerate(shards)}
    coverage_cursor = {span.shard_index: span.start for span in spans}
    score_target_counts = {score: 0 for score in range(6)}
    high_documents: list[DocumentSlice] = []
    general_tail_documents: list[DocumentSlice] = []
    general_stage_written = 0
    selected_target_count = 0
    expected_document_index = 0
    stage_split: dict | None = None
    proxy_cutoff: dict | None = None
    emission_digest = hashlib.sha256()
    emitted_document_slices = 0
    emitted_segments = 0

    if plan_only:
        writer = None
        source = None
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} is not empty; refusing to overwrite curriculum data"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = OutputShardWriter(
            output_dir,
            total_target_tokens=total_target_tokens,
            targets_per_shard=(
                micro_batches_per_output_shard * tokens_per_micro_batch
            ),
        )
        source = SourceTokens(shards)

    with scores_path.open("r", encoding="utf-8") as scores_file:
        for line_number, line in enumerate(scores_file, start=1):
            record = json.loads(line)
            document_index = int(record["document_index"])
            if document_index != expected_document_index:
                raise ValueError(
                    f"{scores_path}:{line_number}: expected document_index "
                    f"{expected_document_index}, found {document_index}"
                )
            expected_document_index += 1

            segments = record_segments(
                record, shards, shard_name_to_index, selected_ranges
            )
            if not segments:
                if selected_target_count == total_target_tokens:
                    break
                continue

            for segment in segments:
                expected_start = coverage_cursor[segment.shard_index]
                if segment.start != expected_start:
                    raise ValueError(
                        f"source coverage gap/overlap in "
                        f"{shards[segment.shard_index].path.name}: "
                        f"expected {expected_start}, found {segment.start}"
                    )
                coverage_cursor[segment.shard_index] = segment.end

            int_score = int(record["int_score"])
            if not 0 <= int_score <= 5:
                raise ValueError(f"document {document_index}: invalid score {int_score}")
            document = DocumentSlice(document_index, int_score, segments)
            selected_target_count += document.token_count
            score_target_counts[int_score] += document.token_count
            if selected_target_count > total_target_tokens:
                raise ValueError("score records extend past the proxy source selection")

            full_record_selected = int(record["selected_token_count"])
            if document.token_count != full_record_selected:
                proxy_cutoff = {
                    "document_index": document_index,
                    "int_score": int_score,
                    "classified_selected_tokens": full_record_selected,
                    "proxy_selected_tokens": document.token_count,
                }

            if int_score >= threshold:
                high_documents.append(document)
            elif general_stage_written < general_stage_tokens:
                remaining = general_stage_tokens - general_stage_written
                stage_part, tail_part = split_document(
                    document, min(document.token_count, remaining)
                )
                if stage_part is not None:
                    update_emission_digest(
                        emission_digest, stage_part, stage_index=0
                    )
                    emitted_document_slices += 1
                    emitted_segments += len(stage_part.segments)
                    if writer is not None:
                        assert source is not None
                        emit_document(writer, source, stage_part)
                    general_stage_written += stage_part.token_count
                if tail_part is not None:
                    general_tail_documents.append(tail_part)
                    stage_split = {
                        "document_index": document_index,
                        "int_score": int_score,
                        "general_stage_tokens": stage_part.token_count,
                        "warmdown_tokens": tail_part.token_count,
                    }
            else:
                general_tail_documents.append(document)

            if selected_target_count == total_target_tokens:
                break

    if selected_target_count != total_target_tokens:
        raise ValueError(
            f"scores cover {selected_target_count:,} proxy targets, "
            f"expected {total_target_tokens:,}"
        )
    for span in spans:
        if coverage_cursor[span.shard_index] != span.end:
            raise ValueError(
                f"incomplete source coverage in {span.path.name}: "
                f"{coverage_cursor[span.shard_index]} != {span.end}"
            )
    if general_stage_written != general_stage_tokens:
        raise ValueError(
            f"general stage has {general_stage_written:,} tokens, "
            f"expected {general_stage_tokens:,}"
        )

    high_tokens = sum(document.token_count for document in high_documents)
    general_tail_tokens = sum(
        document.token_count for document in general_tail_documents
    )
    if expected_high_tokens is not None and high_tokens != expected_high_tokens:
        raise ValueError(
            f"found {high_tokens:,} high-quality proxy tokens, "
            f"expected {expected_high_tokens:,}"
        )
    if high_tokens + general_tail_tokens != warmdown_tokens:
        raise ValueError(
            "warmdown queues do not fill the pre-registered warmdown budget"
        )

    for document in proportional_merge(high_documents, general_tail_documents):
        update_emission_digest(emission_digest, document, stage_index=1)
        emitted_document_slices += 1
        emitted_segments += len(document.segments)
        if writer is not None:
            assert source is not None
            emit_document(writer, source, document)
    if writer is not None:
        output_shards = writer.finish()
    else:
        output_shards = []

    scores_sha256 = sha256_file(scores_path)
    classification_manifest_sha256 = sha256_file(classification_manifest_path)
    manifest = {
        "schema_version": 1,
        "experiment_id": "exp004-B",
        "created_unix_seconds": time.time(),
        "builder_git": git_metadata(),
        "source": {
            "input_pattern": input_pattern,
            "selection": source_selection,
            "coverage_verified": True,
            "covered_spans": [
                {
                    "shard": span.path.name,
                    "start": span.start,
                    "end": coverage_cursor[span.shard_index],
                    "tokens": coverage_cursor[span.shard_index] - span.start,
                }
                for span in spans
            ],
            "classification_selection_sha256": classification_manifest.get(
                "selection_sha256"
            ),
        },
        "classification": {
            "scores_file": str(scores_path),
            "scores_sha256": scores_sha256,
            "manifest_file": str(classification_manifest_path),
            "manifest_sha256": classification_manifest_sha256,
            "threshold": threshold,
        },
        "curriculum": {
            "merge_rule": (
                "preserve order within high and general queues; emit the queue "
                "with lower normalised token progress"
            ),
            "stages": [
                {
                    "index": 0,
                    "name": "general",
                    "start_update": 0,
                    "end_update_exclusive": general_stage_updates,
                    "target_tokens": general_stage_tokens,
                    "score_lt_threshold_tokens": general_stage_tokens,
                    "score_gte_threshold_tokens": 0,
                },
                {
                    "index": 1,
                    "name": "enriched_warmdown",
                    "start_update": general_stage_updates,
                    "end_update_exclusive": num_iterations,
                    "target_tokens": warmdown_tokens,
                    "score_lt_threshold_tokens": general_tail_tokens,
                    "score_gte_threshold_tokens": high_tokens,
                    "high_fraction": high_tokens / warmdown_tokens,
                },
            ],
            "stage_boundary_split_document": stage_split,
            "proxy_cutoff_document": proxy_cutoff,
        },
        "output": {
            "directory": str(output_dir),
            "pattern": "exp004_train_*.bin",
            "tokens_per_micro_batch": tokens_per_micro_batch,
            "micro_batches_per_shard": micro_batches_per_output_shard,
            "emission_plan_sha256": emission_digest.hexdigest(),
            "emitted_document_slices": emitted_document_slices,
            "emitted_segments": emitted_segments,
            "shards": output_shards,
        },
        "accounting": {
            "complete": True,
            "source_target_tokens": selected_target_count,
            "output_target_tokens": (
                writer.total_written if writer is not None else total_target_tokens
            ),
            "num_iterations": num_iterations,
            "tokens_per_update": tokens_per_update,
            "score_target_tokens": {
                str(score): score_target_counts[score] for score in range(6)
            },
            "high_target_tokens": high_tokens,
            "general_target_tokens": total_target_tokens - high_tokens,
            "high_documents": len(high_documents),
            "general_tail_documents": len(general_tail_documents),
        },
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)

    if not plan_only:
        manifest_path = output_dir / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        print(f"wrote manifest: {manifest_path}")
    print(
        f"exp004-B accounting: {total_target_tokens:,} targets; "
        f"{high_tokens:,} high ({high_tokens / total_target_tokens:.4%}); "
        f"warmdown mixture {high_tokens / warmdown_tokens:.4%} high"
    )
    return manifest


def main() -> None:
    args = parse_args()
    if args.verify_only is not None:
        verify_output_manifest(args.verify_only)
        return
    expected_high_tokens = (
        None if args.expected_high_tokens < 0 else args.expected_high_tokens
    )
    build_curriculum(
        input_pattern=args.input_bin,
        scores_path=args.scores,
        classification_manifest_path=args.classification_manifest,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        warmdown_iters=args.warmdown_iters,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        grad_accumulation_steps=args.grad_accumulation_steps,
        threshold=args.threshold,
        micro_batches_per_output_shard=args.micro_batches_per_output_shard,
        expected_high_tokens=expected_high_tokens,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
