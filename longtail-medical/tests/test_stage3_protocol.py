import unittest
from pathlib import Path

import numpy as np

from longtail_medical.config import load_config
from tools.run_stage3a_logit_adjustment import adjust
from tools.run_stage3a_training_matrix import ARMS, SEEDS


class Stage3ProtocolTest(unittest.TestCase):
    def test_matrix_has_six_unique_training_methods_and_three_seeds(self):
        names = [name for name, _ in ARMS]
        methods = [config["method"] for _, config in ARMS]
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertEqual(len(set(methods)), 6)
        self.assertEqual(SEEDS, (42, 43, 44))

    def test_stage3_config_is_validation_only_and_monitors_lesion_mcc(self):
        project = Path(__file__).resolve().parents[1]
        config = load_config(project / "configs" / "stage3a_lesion_disjoint_screening.yaml")
        self.assertNotIn("test_csv", config["data"])
        self.assertFalse(config["evaluation"]["run_test"])
        self.assertEqual(config["checkpoint"]["monitor"], "lesion_mcc")

    def test_posthoc_logit_adjustment_favors_rare_class(self):
        probabilities = np.asarray([[0.6, 0.4]])
        priors = np.asarray([0.9, 0.1])
        adjusted = adjust(probabilities, priors, tau=1.0)
        self.assertGreater(adjusted[0, 1], adjusted[0, 0])
        self.assertAlmostEqual(float(adjusted.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
