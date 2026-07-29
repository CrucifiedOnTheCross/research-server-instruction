from __future__ import annotations

import unittest

import pandas as pd

from tools.check_stage14_readiness import choose_branch


THRESHOLDS = {
    "minimum_seed_wins": 3,
    "melanoma_recall_margin": -0.05,
    "melanoma_auprc_margin": -0.02,
    "ece_worsening_margin": 0.02,
}


def frames(deltas: dict[str, float], wins: dict[str, int], ci: float):
    paired = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean_difference_synthetic_minus_replay": value,
                "synthetic_wins": wins.get(metric, 3),
            }
            for metric, value in deltas.items()
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {"metric": "macro_f1", "ci95_low": ci},
            {"metric": "mcc", "ci95_low": ci},
        ]
    )
    return paired, bootstrap


class Stage14ReadinessTest(unittest.TestCase):
    def base_deltas(self) -> dict[str, float]:
        return {
            "macro_f1": 0.02,
            "mcc": 0.02,
            "auprc_ovr_macro": 0.01,
            "mel_recall": -0.01,
            "mel_auprc": -0.01,
            "ece": 0.01,
        }

    def test_strong_result_selects_external_robustness(self) -> None:
        paired, bootstrap = frames(self.base_deltas(), {}, 0.001)
        branch, _ = choose_branch(paired, bootstrap, THRESHOLDS)
        self.assertEqual(branch, "A_external_robustness")

    def test_ranking_guardrail_failure_selects_generator_pilot(self) -> None:
        deltas = self.base_deltas()
        deltas["auprc_ovr_macro"] = -0.01
        paired, bootstrap = frames(deltas, {"auprc_ovr_macro": 0}, -0.01)
        branch, _ = choose_branch(paired, bootstrap, THRESHOLDS)
        self.assertEqual(branch, "B_generator_pilot")

    def test_null_result_selects_preprocessing(self) -> None:
        deltas = self.base_deltas()
        deltas["macro_f1"] = -0.01
        deltas["mcc"] = -0.01
        deltas["auprc_ovr_macro"] = -0.01
        paired, bootstrap = frames(
            deltas,
            {"macro_f1": 0, "mcc": 0, "auprc_ovr_macro": 0},
            -0.02,
        )
        branch, _ = choose_branch(paired, bootstrap, THRESHOLDS)
        self.assertEqual(branch, "C_preprocessing_qualification")


if __name__ == "__main__":
    unittest.main()
