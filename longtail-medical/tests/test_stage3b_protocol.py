import tempfile
import unittest
from pathlib import Path

import numpy as np

from longtail_medical.calibration import apply_temperature, fit_temperature
from longtail_medical.config import load_config
from tools.build_stage3b_splits import SPLIT_SEEDS
from tools.run_stage3b_training_matrix import ARMS, MODEL_SEEDS, experiment_name


class Stage3BProtocolTest(unittest.TestCase):
    def test_matrix_is_exactly_eighteen_trainings(self):
        self.assertEqual(SPLIT_SEEDS, (101, 202, 303))
        self.assertEqual(MODEL_SEEDS, (42, 43, 44))
        self.assertEqual([arm for arm, _ in ARMS], ["ce", "ldam_drw"])
        names = {
            (experiment_name(arm, split_seed), model_seed)
            for split_seed in SPLIT_SEEDS for model_seed in MODEL_SEEDS for arm, _ in ARMS
        }
        self.assertEqual(len(names), 18)

    def test_training_template_cannot_load_test(self):
        project = Path(__file__).resolve().parents[1]
        config = load_config(project / "configs/stage3b_split_confirmation.yaml")
        self.assertNotIn("test_csv", config["data"])
        self.assertFalse(config["evaluation"]["run_test"])
        self.assertEqual(config["training"]["epochs"], 50)
        self.assertEqual(config["checkpoint"]["primary_policy"], "last")
        self.assertEqual(config["checkpoint"]["monitor"], "lesion_mcc")

    def test_temperature_scaling_reduces_nll_and_preserves_predictions(self):
        labels = np.asarray([0, 0, 1, 1, 1, 0])
        probabilities = np.asarray([
            [0.999, 0.001], [0.99, 0.01], [0.95, 0.05],
            [0.01, 0.99], [0.02, 0.98], [0.98, 0.02],
        ])
        fit = fit_temperature(labels, probabilities)
        calibrated = apply_temperature(probabilities, fit["temperature"])
        self.assertLessEqual(fit["nll_after"], fit["nll_before"] + 1e-10)
        np.testing.assert_array_equal(calibrated.argmax(axis=1), probabilities.argmax(axis=1))
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)

    def test_invalid_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_temperature(np.asarray([[0.5, 0.5]]), 0.0)

    def test_locked_analysis_runs_inside_qualified_container(self):
        project = Path(__file__).resolve().parents[1]
        script = (project / "scripts/run_stage3b_locked_analysis.sh").read_text(encoding="utf-8")
        self.assertIn("docker run --rm", script)
        self.assertIn("--gpus all", script)
        self.assertIn("source '$RESEARCH_VENV/bin/activate'", script)
        self.assertNotIn("\nsource \"$RESEARCH_VENV/bin/activate\"", script)


if __name__ == "__main__":
    unittest.main()
