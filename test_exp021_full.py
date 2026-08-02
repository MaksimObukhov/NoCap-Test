import re
import unittest
from pathlib import Path


RUN_SCRIPT = Path(__file__).with_name("run.sh").read_text()


class Exp021FullLauncherTests(unittest.TestCase):
    def test_full_budget_and_schedule_are_frozen(self):
        full_case = RUN_SCRIPT.split("full)", 1)[1].split(";;", 1)[0]
        values = {
            key: int(value)
            for key, value in re.findall(r"([A-Z_]+)=([0-9]+)", full_case)
        }
        self.assertEqual(values["NUM_ITERATIONS"] * 262_144, 2_700_083_200)
        self.assertEqual(values["WARMUP_ITERS"], 553)
        self.assertEqual(values["WARMDOWN_ITERS"], 2212)
        self.assertEqual(values["MILESTONE_START_ITERS"], 8088)
        milestones = list(
            range(
                values["MILESTONE_START_ITERS"],
                values["NUM_ITERATIONS"],
                values["MILESTONE_EVERY"],
            )
        )
        self.assertEqual(len(milestones), 9)

    def test_launcher_accepts_only_proxy_or_full_seed_zero(self):
        self.assertIn('"$MODE" != "proxy" && "$MODE" != "full"', RUN_SCRIPT)
        self.assertIn('"$SEED" != "0"', RUN_SCRIPT)


if __name__ == "__main__":
    unittest.main()
