from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.analyze_stage12_failure_modes import (
    frequency_features,
    frequency_bootstrap,
    grouped_bootstrap_mean,
    interpret_failure_modes,
    prdc,
    ranking_tail_diagnostics,
    vendi_score,
)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class Stage12DiagnosticsTest(unittest.TestCase):
    def test_prdc_identical_sets_are_fully_covered(self) -> None:
        rng = np.random.default_rng(42)
        x = normalize(rng.normal(size=(30, 8)))
        result = prdc(x, x.copy(), k=5)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["coverage"], 1.0)
        self.assertGreaterEqual(result["density"], 1.0)

    def test_vendi_detects_collapsed_features(self) -> None:
        collapsed = normalize(np.ones((12, 4)))
        diverse = normalize(np.eye(4).repeat(3, axis=0))
        self.assertTrue(np.isclose(vendi_score(collapsed), 1.0))
        self.assertGreater(vendi_score(diverse), 3.9)

    def test_frequency_fractions_partition_energy(self) -> None:
        rng = np.random.default_rng(7)
        values = frequency_features(rng.normal(size=(64, 64)))
        total = (
            values["fft_low_fraction"]
            + values["fft_mid_fraction"]
            + values["fft_high_fraction"]
        )
        self.assertTrue(np.isclose(total, 1.0, atol=1e-6))
        self.assertTrue(np.isfinite(values["spectral_slope"]))
        self.assertGreater(values["gradient_rms"], 0)

    def test_grouped_bootstrap_is_reproducible(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        groups = np.asarray(["a", "a", "b", "c"])
        first = grouped_bootstrap_mean(values, groups, 100, 11)
        second = grouped_bootstrap_mean(values, groups, 100, 11)
        self.assertEqual(first, second)
        self.assertLessEqual(first[1], first[0])
        self.assertLessEqual(first[0], first[2])

    def test_frequency_bootstrap_keeps_duplicate_source_presentations_paired(self) -> None:
        rows = []
        for index, pair in enumerate(("synth_a", "synth_b")):
            for kind, value in (("synthetic", 2.0 + index), ("source", 1.0)):
                rows.append(
                    {
                        "kind": kind,
                        "pair_id": pair,
                        "label": "mel",
                        "source_group_id": "lesion_1",
                        "fft_low_fraction": value,
                        "fft_mid_fraction": value,
                        "fft_high_fraction": value,
                        "spectral_slope": value,
                        "gradient_rms": value,
                    }
                )
        for label in ("akiec", "bkl"):
            for kind, value in (("synthetic", 2.0), ("source", 1.0)):
                rows.append(
                    {
                        "kind": kind,
                        "pair_id": f"{label}_pair",
                        "label": label,
                        "source_group_id": f"{label}_lesion",
                        "fft_low_fraction": value,
                        "fft_mid_fraction": value,
                        "fft_high_fraction": value,
                        "spectral_slope": value,
                        "gradient_rms": value,
                    }
                )
        result = frequency_bootstrap(pd.DataFrame(rows), 20, 4)
        mel = result[(result["label"] == "mel") & (result["metric"] == "gradient_rms")]
        self.assertAlmostEqual(float(mel.iloc[0]["mean_synthetic_minus_source"]), 1.5)

    def test_ranking_tail_detects_hard_negative_regression(self) -> None:
        replay = pd.DataFrame(
            {
                "target": ["mel", "mel", "nv", "nv"],
                "prob_mel": [0.8, 0.7, 0.2, 0.1],
                "prob_akiec": [0.1, 0.1, 0.1, 0.1],
                "prob_bkl": [0.1, 0.1, 0.1, 0.1],
            }
        )
        synthetic = replay.copy()
        synthetic["prob_mel"] = [0.82, 0.72, 0.9, 0.05]
        ranking, confusers = ranking_tail_diagnostics(replay, synthetic, 42)
        mel = next(row for row in ranking if row["label"] == "mel")
        self.assertLess(mel["auprc_delta"], 0)
        self.assertGreater(mel["negative_q99_delta"], 0)
        self.assertTrue(
            any(row["scored_class"] == "mel" for row in confusers)
        )

    def test_failure_mode_summary_separates_coverage_from_diversity(self) -> None:
        feature_rows = []
        for encoder in ("dino", "convnext_real_seed42"):
            for label in ("mel", "akiec", "bkl"):
                feature_rows.extend(
                    [
                        {
                            "encoder": encoder,
                            "label": label,
                            "arm": "source",
                            "prdc_coverage": 0.5,
                            "prdc_precision": 0.8,
                            "prdc_density": 0.7,
                            "vendi_score": 2.0,
                        },
                        {
                            "encoder": encoder,
                            "label": label,
                            "arm": "synthetic",
                            "prdc_coverage": 0.2,
                            "prdc_precision": 0.4,
                            "prdc_density": 0.3,
                            "vendi_score": 3.0,
                        },
                    ]
                )
        novelty_rows = []
        for label in ("mel", "akiec", "bkl"):
            novelty_rows.extend(
                [
                    {"encoder": "dino", "label": label, "novelty_ratio": 0.7},
                    {
                        "encoder": "convnext_real_seed42",
                        "label": label,
                        "novelty_ratio": 1.3,
                    },
                ]
            )
        frequency = pd.DataFrame(
            [{"ci95_low": 0.1, "ci95_high": 0.2}] * 3
        )
        ranking = pd.DataFrame(
            [
                {
                    "label": "mel",
                    "auprc_delta": -0.1,
                    "negative_q90_delta": 0.2,
                    "positive_q10_delta": -0.1,
                }
            ]
        )
        result = interpret_failure_modes(
            pd.DataFrame(feature_rows),
            pd.DataFrame(novelty_rows),
            frequency,
            ranking,
        )
        self.assertTrue(result["coverage_failure"]["supported"])
        self.assertFalse(result["simple_diversity_collapse"]["supported"])
        self.assertTrue(result["representation_mismatch"]["supported"])


if __name__ == "__main__":
    unittest.main()
