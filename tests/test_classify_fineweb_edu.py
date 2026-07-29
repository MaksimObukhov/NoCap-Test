import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "data" / "classify_fineweb_edu.py"
SPEC = importlib.util.spec_from_file_location("classify_fineweb_edu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_shard(path: Path, tokens: list[int]) -> MODULE.ShardInfo:
    header = np.zeros(MODULE.HEADER_INTS, dtype=np.int32)
    header[0] = MODULE.MAGIC
    header[1] = MODULE.VERSION
    header[2] = len(tokens)
    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(np.asarray(tokens, dtype=np.uint16).tobytes())
    return MODULE.read_shard_info(path)


class SelectionTest(unittest.TestCase):
    def test_selection_matches_loader_and_drops_shard_tails(self):
        shards = [
            MODULE.ShardInfo(Path("one.bin"), 11),
            MODULE.ShardInfo(Path("two.bin"), 11),
        ]
        spans = MODULE.plan_loader_selection(
            shards,
            num_iterations=3,
            batch_size=2,
            sequence_length=2,
            grad_accumulation_steps=1,
        )
        self.assertEqual(
            [(span.path.name, span.start, span.end) for span in spans],
            [("one.bin", 1, 9), ("two.bin", 1, 5)],
        )

    def test_baseline_token_math(self):
        shards = [
            MODULE.ShardInfo(Path(f"{index}.bin"), 100_000_000)
            for index in range(50)
        ]
        spans = MODULE.plan_loader_selection(
            shards,
            num_iterations=4768,
            batch_size=16,
            sequence_length=1024,
            grad_accumulation_steps=32,
        )
        self.assertEqual(sum(span.token_count for span in spans), 2_499_805_184)
        self.assertEqual(len(spans), 26)
        self.assertEqual(spans[-1].token_count, 16_384)


class DocumentTest(unittest.TestCase):
    def test_documents_cross_shards_but_count_only_selected_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = [
                write_shard(
                    root / "one.bin",
                    [MODULE.GPT2_EOT, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                ),
                write_shard(
                    root / "two.bin",
                    [20, 21, MODULE.GPT2_EOT, 30, 31, 32, 33, 34, 35, 36, 37],
                ),
            ]
            spans = MODULE.plan_loader_selection(
                shards,
                num_iterations=3,
                batch_size=2,
                sequence_length=2,
                grad_accumulation_steps=1,
            )
            documents = list(MODULE.iter_selected_documents(shards, spans))

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].token_count, 13)
        self.assertEqual(documents[0].selected_token_count, 9)
        self.assertEqual(documents[1].selected_token_count, 3)
        self.assertEqual(documents[1].text_tokens().tolist(), [30, 31, 32, 33, 34, 35, 36, 37])


class OutputTest(unittest.TestCase):
    def test_quality_threshold_uses_rounded_int_score(self):
        document = MODULE.Document(
            start_shard=0,
            start_offset=1,
            end_shard=0,
            end_offset=2,
            token_pieces=[np.asarray([10], dtype=np.uint16)],
            selected_token_count=1,
            starts_with_eot=False,
        )
        shards = [MODULE.ShardInfo(Path("one.bin"), 10)]
        record = MODULE.make_record(
            document,
            score=2.51,
            document_index=0,
            shards=shards,
            threshold=3,
            text="example",
        )
        self.assertEqual(record["int_score"], 3)
        self.assertTrue(record["is_high_quality"])

    def test_resume_manifest_and_stats_round_trip(self):
        plan = {"selected_tokens": 1, "spans": []}
        manifest = MODULE.run_manifest(plan, threshold=3, classifier_max_length=512)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.jsonl"
            stats, last_record = MODULE.prepare_output(
                output, resume=False, expected_manifest=manifest
            )
            self.assertEqual(stats.documents, 0)
            self.assertIsNone(last_record)

            record = {
                "document_index": 0,
                "selected_token_count": 1,
                "is_high_quality": True,
                "score": 3.0,
            }
            output.write_text(f"{MODULE.json.dumps(record)}\n", encoding="utf-8")
            stats, last_record = MODULE.prepare_output(
                output, resume=True, expected_manifest=manifest
            )

        self.assertEqual(stats.documents, 1)
        self.assertEqual(stats.selected_tokens, 1)
        self.assertEqual(last_record, record)


if __name__ == "__main__":
    unittest.main()
