import unittest

import numpy as np

from longtail_medical.statistical_analysis import aggregate_by_lesion


class StatisticalAnalysisTest(unittest.TestCase):
    def test_lesion_probabilities_are_mean_aggregated(self):
        labels = np.array([0, 0, 1])
        probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
        result_labels, result_probabilities, lesion_ids = aggregate_by_lesion(
            labels, probabilities, ["L1", "L1", "L2"], ["a", "b", "c"]
        )
        np.testing.assert_array_equal(result_labels, [0, 1])
        np.testing.assert_allclose(result_probabilities[0], [0.7, 0.3])
        self.assertEqual(lesion_ids, ["L1", "L2"])

    def test_conflicting_lesion_labels_fail(self):
        with self.assertRaises(ValueError):
            aggregate_by_lesion(
                np.array([0, 1]), np.array([[0.8, 0.2], [0.2, 0.8]]),
                ["same", "same"], ["a", "b"],
            )


if __name__ == "__main__":
    unittest.main()
