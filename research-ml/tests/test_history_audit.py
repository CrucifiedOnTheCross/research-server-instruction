from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_experiment_history import CURRENT_REQUIRED_ARTIFACTS, audit, seed_from_run


class HistoryAuditTest(unittest.TestCase):
    def make_run(
        self,
        root: Path,
        experiment: str,
        run: str,
        *,
        test_evaluated: bool = False,
        complete: bool = True,
    ) -> None:
        run_dir = root / experiment / run
        run_dir.mkdir(parents=True)
        for name in CURRENT_REQUIRED_ARTIFACTS:
            path = run_dir / name
            if name.endswith(".json"):
                payload = (
                    {"seed": 42, "test_evaluated": test_evaluated}
                    if name == "summary.json"
                    else {}
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
            else:
                path.write_text("x", encoding="utf-8")
        if not complete:
            (run_dir / "sampling_plan.json").unlink()

    def test_passes_complete_modern_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, "stage13b_replay", "run_42")
            records, report = audit(root, modern_stage_min=9)
            self.assertEqual(len(records), 1)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(records[0]["seed"], 42)

    def test_extracts_seed_from_run_name(self) -> None:
        self.assertEqual(seed_from_run("20260729-051104_44"), 44)
        self.assertIsNone(seed_from_run("invalid"))

    def test_fails_open_test_or_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(
                root,
                "stage13b_replay",
                "run_42",
                test_evaluated=True,
                complete=False,
            )
            _, report = audit(root, modern_stage_min=9)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(len(report["modern_protocol_violations"]), 1)


if __name__ == "__main__":
    unittest.main()
