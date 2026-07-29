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


if __name__ == "__main__":
    unittest.main()
