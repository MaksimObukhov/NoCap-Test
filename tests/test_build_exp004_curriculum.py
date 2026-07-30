import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "data" / "build_exp004_curriculum.py"
SPEC = importlib.util.spec_from_file_location("build_exp004_curriculum", MODULE_PATH)
curriculum = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = curriculum
SPEC.loader.exec_module(curriculum)

TRAIN_MODULE_PATH = REPOSITORY_ROOT / "train_gpt2.py"
TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_gpt2_for_manifest_test", TRAIN_MODULE_PATH
)
trainer = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC.loader is not None
sys.modules[TRAIN_SPEC.name] = trainer
ORIGINAL_ARGV_ZERO = sys.argv[0]
try:
    sys.argv[0] = str(TRAIN_MODULE_PATH)
    TRAIN_SPEC.loader.exec_module(trainer)
finally:
    sys.argv[0] = ORIGINAL_ARGV_ZERO


def write_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(curriculum.HEADER_INTS, dtype=np.int32)
    header[0] = curriculum.MAGIC
    header[1] = curriculum.VERSION
    header[2] = len(tokens)
    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def read_targets(path: Path) -> tuple[int, list[int]]:
    with path.open("rb") as file:
        file.seek(curriculum.HEADER_BYTES)
        tokens = np.frombuffer(file.read(), dtype=np.uint16).tolist()
    return tokens[0], tokens[1:]


class CurriculumBuilderTest(unittest.TestCase):
    def test_exact_once_stream_and_shard_contexts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "fineweb_train_000001.bin"
            scores_path = root / "scores.jsonl"
            classification_manifest_path = root / "scores.manifest.json"
            output_dir = root / "output"

            # Token 100 is context only. Targets 1..12 form six B=2, T=1 batches.
            write_shard(source_path, [100, *range(1, 13)])
            score_records = [
                {
                    "document_index": 0,
                    "start_shard": source_path.name,
                    "start_offset": 0,
                    "end_shard": source_path.name,
                    "end_offset": 5,
                    "selected_token_count": 4,
                    "int_score": 0,
                },
                {
                    "document_index": 1,
                    "start_shard": source_path.name,
                    "start_offset": 5,
                    "end_shard": source_path.name,
                    "end_offset": 7,
                    "selected_token_count": 2,
                    "int_score": 3,
                },
                {
                    "document_index": 2,
                    "start_shard": source_path.name,
                    "start_offset": 7,
                    "end_shard": source_path.name,
                    "end_offset": 13,
                    "selected_token_count": 6,
                    "int_score": 1,
                },
            ]
            with scores_path.open("w", encoding="utf-8") as file:
                for record in score_records:
                    file.write(json.dumps(record) + "\n")

            classification_manifest_path.write_text(
                json.dumps(
                    {
                        "int_score_threshold": 3,
                        "selection_sha256": "synthetic-selection",
                        "selection": {
                            "spans": [
                                {
                                    "shard": source_path.name,
                                    "start": 1,
                                    "end": 13,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = curriculum.build_curriculum(
                input_pattern=str(source_path),
                scores_path=scores_path,
                classification_manifest_path=classification_manifest_path,
                output_dir=output_dir,
                num_iterations=6,
                warmdown_iters=3,
                batch_size=2,
                sequence_length=1,
                grad_accumulation_steps=1,
                threshold=3,
                micro_batches_per_output_shard=2,
                expected_high_tokens=2,
                plan_only=False,
            )

            output_targets = []
            contexts = []
            for shard_record in manifest["output"]["shards"]:
                context, targets = read_targets(output_dir / shard_record["file"])
                contexts.append(context)
                output_targets.extend(targets)

            # Stage 0 takes six general targets. Warmdown then proportionally
            # interleaves the two high targets with the four remaining general.
            self.assertEqual(output_targets, [1, 2, 3, 4, 7, 8, 5, 6, 9, 10, 11, 12])
            self.assertEqual(sorted(output_targets), list(range(1, 13)))
            self.assertEqual(contexts, [100, 4, 6])
            self.assertEqual(
                manifest["curriculum"]["stage_boundary_split_document"][
                    "document_index"
                ],
                2,
            )
            self.assertEqual(manifest["accounting"]["output_target_tokens"], 12)
            curriculum.verify_output_manifest(output_dir / "manifest.json")
            loaded_manifest, manifest_sha256 = trainer.load_curriculum_manifest(
                output_dir / "manifest.json",
                input_pattern=str(output_dir / "exp004_train_*.bin"),
                tokens_per_update=2,
                num_iterations=6,
                allow_prefix=False,
            )
            self.assertEqual(loaded_manifest["experiment_id"], "exp004-B")
            self.assertEqual(
                manifest_sha256,
                curriculum.sha256_file(output_dir / "manifest.json"),
            )
            with self.assertRaisesRegex(ValueError, "run needs"):
                trainer.load_curriculum_manifest(
                    output_dir / "manifest.json",
                    input_pattern=str(output_dir / "exp004_train_*.bin"),
                    tokens_per_update=2,
                    num_iterations=5,
                    allow_prefix=False,
                )

    def test_proportional_merge_preserves_each_queue_order(self):
        high = [
            curriculum.DocumentSlice(
                10, 3, (curriculum.SourceSegment(0, 0, 2),)
            ),
            curriculum.DocumentSlice(
                11, 4, (curriculum.SourceSegment(0, 2, 3),)
            ),
        ]
        general = [
            curriculum.DocumentSlice(
                20, 1, (curriculum.SourceSegment(0, 3, 7),)
            ),
            curriculum.DocumentSlice(
                21, 2, (curriculum.SourceSegment(0, 7, 9),)
            ),
        ]

        merged = list(curriculum.proportional_merge(high, general))
        self.assertEqual(
            [document.document_index for document in merged if document.int_score >= 3],
            [10, 11],
        )
        self.assertEqual(
            [document.document_index for document in merged if document.int_score < 3],
            [20, 21],
        )
        self.assertEqual(sum(document.token_count for document in merged), 9)


if __name__ == "__main__":
    unittest.main()
