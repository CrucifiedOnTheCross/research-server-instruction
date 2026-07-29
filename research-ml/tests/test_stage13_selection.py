from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.select_stage13_multiencoder_coverage import (
    greedy_facility_select,
    selection_gate,
)


class Stage13SelectionTests(unittest.TestCase):
    def test_facility_selection_enforces_unique_sources(self) -> None:
        candidates = pd.DataFrame(
            {
                "image_id": ["a1", "a2", "b1"],
                "source_image_id": ["a", "a", "b"],
                "stage13_quality_score": [1.0, 0.9, 0.8],
            }
        )
        similarity = np.asarray(
            [[1.0, 0.0], [0.9, 0.0], [0.0, 1.0]], dtype=float
        )
        chosen = greedy_facility_select(
            candidates, similarity, np.ones(2), dose=2
        )
        sources = candidates.iloc[chosen]["source_image_id"].tolist()
        self.assertEqual(len(set(sources)), 2)
        self.assertEqual(set(sources), {"a", "b"})

    def test_gate_rejects_fidelity_tradeoff(self) -> None:
        comparison = pd.DataFrame(
            {
                "coverage_delta_new_minus_strict": [0.1] * 12,
                "precision_delta_new_minus_strict": [-0.2] * 12,
                "density_delta_new_minus_strict": [0.0] * 12,
            }
        )
        selected = pd.DataFrame(
            {
                "label": sum(([label] * 30 for label in ("mel", "akiec", "bkl")), []),
                "source_image_id": [f"source_{index}" for index in range(90)],
                "stage13_tier": ["A"] * 90,
            }
        )
        result = selection_gate(comparison, selected, frequency_wins=3)
        self.assertFalse(result["gate_open"])
        self.assertFalse(result["checks"]["mean_precision_not_below_margin"])

    def test_gate_accepts_predeclared_improvement(self) -> None:
        comparison = pd.DataFrame(
            {
                "coverage_delta_new_minus_strict": [0.02] * 9 + [-0.01] * 3,
                "precision_delta_new_minus_strict": [-0.01] * 12,
                "density_delta_new_minus_strict": [0.01] * 12,
            }
        )
        selected = pd.DataFrame(
            {
                "label": sum(([label] * 30 for label in ("mel", "akiec", "bkl")), []),
                "source_image_id": [f"source_{index}" for index in range(90)],
                "stage13_tier": ["A"] * 60 + ["B"] * 30,
            }
        )
        result = selection_gate(comparison, selected, frequency_wins=2)
        self.assertTrue(result["gate_open"])


if __name__ == "__main__":
    unittest.main()
