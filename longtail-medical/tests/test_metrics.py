import unittest

import numpy as np

from longtail_medical.metrics import classification_metrics, expected_calibration_error


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.eye(3)[labels] * 0.98 + (1 - np.eye(3)[labels]) * 0.01
        result = classification_metrics(labels, probabilities, ["a", "b", "c"])
        self.assertAlmostEqual(result["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["mcc"], 1.0)
        self.assertAlmostEqual(result["macro_auprc"], 1.0)
        self.assertAlmostEqual(result["worst_class_recall"], 1.0)

    def test_probability_validation(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.array([0, 1]), np.ones((2, 2)), ["a", "b"])

    def test_ece_is_zero_for_certain_correct_predictions(self):
        labels = np.array([0, 1])
        self.assertAlmostEqual(expected_calibration_error(labels, np.eye(2)), 0.0)


if __name__ == "__main__":
    unittest.main()
