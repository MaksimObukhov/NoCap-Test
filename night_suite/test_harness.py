import json
import struct
import tempfile
import unittest
from pathlib import Path

from night_suite.run_night_suite import (
    InfrastructureFailure,
    data_inventory,
    proxy_allowed,
    start_stage,
    steady_token_rate,
)


class HarnessTest(unittest.TestCase):
    def test_steady_token_rate_handles_variable_effective_batch(self):
        records = []
        tokens = 0
        for step in range(1, 51):
            update_tokens = 100 if step < 25 else 200
            tokens += update_tokens
            records.append(
                {
                    "step": step,
                    "tokens_seen": tokens,
                    "effective_batch_tokens": update_tokens,
                    "step_time_ms": update_tokens / 10.0,
                }
            )
        rate, count = steady_token_rate(records)
        self.assertEqual(count, 38)
        self.assertEqual(rate, 10_000.0)

    def test_data_inventory_checks_header_and_records_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fineweb_train_000001.bin"
            header = bytearray(1024)
            struct.pack_into("<iii", header, 0, 20240520, 1, 17)
            path.write_bytes(header + bytes(34))
            inventory = data_inventory(str(Path(directory) / "*.bin"))
            self.assertEqual(inventory[0]["tokens"], 17)
            json.dumps(inventory)

    def test_scientific_kill_is_a_normal_gate_result(self):
        self.assertFalse(
            proxy_allowed({"decision": "kill"}, {"decision": "pass"})
        )
        self.assertTrue(
            proxy_allowed({"decision": "pass"}, {"decision": "pass"})
        )

    def test_partial_stage_is_an_infrastructure_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage"
            stage.mkdir()
            (stage / "stdout.log").write_text("partial\n")
            with self.assertRaises(InfrastructureFailure):
                start_stage(stage, {"stage": "test"})


if __name__ == "__main__":
    unittest.main()
