from __future__ import annotations

import unittest

import pandas as pd

from tools.analyze_stage10_results import validate_best_checkpoint


class AnalysisCheckpointMonitorTests(unittest.TestCase):
    def test_accepts_non_f1_max_monitor(self) -> None:
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "val/macro_f1": [0.7, 0.8, 0.9],
                "val/auprc_ovr_macro": [0.6, 0.85, 0.8],
            }
        )
        validate_best_checkpoint(
            history,
            {"best_epoch": 2, "best_metric": 0.85},
            {
                "training": {
                    "monitor": "val/auprc_ovr_macro",
                    "monitor_mode": "max",
                }
            },
        )

    def test_rejects_wrong_best_epoch(self) -> None:
        history = pd.DataFrame(
            {"epoch": [1, 2], "val/auprc_ovr_macro": [0.7, 0.8]}
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_best_checkpoint(
                history,
                {"best_epoch": 1, "best_metric": 0.8},
                {
                    "training": {
                        "monitor": "val/auprc_ovr_macro",
                        "monitor_mode": "max",
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
