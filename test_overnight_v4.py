import json
import tempfile
import unittest
from pathlib import Path

from overnight_v4.select_winner import choose


class WinnerSelectorTests(unittest.TestCase):
    def write_metrics(self, directory, final_loss, final_time):
        path = Path(directory) / "metrics.jsonl"
        records = [
            {
                "event": "validation",
                "tokens_seen": 800_000_000,
                "val_loss": 3.60,
                "training_time_seconds": 5800.0,
            },
            {
                "event": "validation",
                "tokens_seen": 937_426_944,
                "val_loss": final_loss,
                "training_time_seconds": final_time,
            },
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def complete_summary(self, final_loss, training_time):
        return {
            "status": "complete",
            "tokens_seen": 937_426_944,
            "final_val_loss": final_loss,
            "training_time_seconds": training_time,
            "final_checkpoint": {"tokens_seen": 937_426_944},
        }

    def test_quality_route_selects_exp022(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = self.write_metrics(directory, 3.560, 6850.0)
            result = choose(self.complete_summary(3.560, 6850.0), metrics)
        self.assertEqual(result["winner"], "exp022")
        self.assertTrue(result["routes"]["quality_route"])

    def test_incomplete_run_falls_back_to_exp021(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = self.write_metrics(directory, 3.55, 6000.0)
            summary = self.complete_summary(3.55, 6000.0)
            summary["tokens_seen"] = 800_000_000
            result = choose(summary, metrics)
        self.assertEqual(result["winner"], "exp021")

    def test_manifest_pins_four_unique_commits(self):
        manifest = json.loads(
            Path(__file__).with_name("overnight_v4").joinpath("manifest.json").read_text()
        )
        commits = [entry["commit"] for entry in manifest["experiments"].values()]
        self.assertEqual(set(manifest["experiments"]), {"exp016", "exp021", "exp022", "exp023"})
        self.assertEqual(len(commits), len(set(commits)))
        self.assertTrue(all(len(commit) == 40 for commit in commits))


if __name__ == "__main__":
    unittest.main()
