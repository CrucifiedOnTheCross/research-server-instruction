import unittest

try:
    import torch
    import torch.nn.functional as F
    from longtail_medical.losses import LongTailObjective, NormedLinear
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch tests run in the server training environment")
class LongTailLossTest(unittest.TestCase):
    def setUp(self):
        self.counts = [100, 10, 1]
        self.logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
        self.targets = torch.tensor([0, 1])

    def test_balanced_softmax_matches_reference_formula(self):
        objective = LongTailObjective({"method": "balanced_softmax"}, self.counts)
        actual = objective(self.logits, self.targets, epoch=1)
        expected = F.cross_entropy(
            self.logits + torch.tensor(self.counts).float().log(), self.targets
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_focal_gamma_zero_equals_cross_entropy(self):
        objective = LongTailObjective({"method": "focal", "gamma": 0.0}, self.counts)
        self.assertTrue(torch.allclose(
            objective(self.logits, self.targets, epoch=1),
            F.cross_entropy(self.logits, self.targets),
        ))

    def test_ldam_drw_activates_after_preregistered_epoch(self):
        objective = LongTailObjective({
            "method": "ldam_drw", "beta": 0.9999, "max_margin": 0.5,
            "scale": 30.0, "drw_start_epoch": 40,
        }, self.counts)
        before = objective(self.logits, self.targets, epoch=40)
        after = objective(self.logits, self.targets, epoch=41)
        self.assertFalse(torch.allclose(before, after))
        self.assertAlmostEqual(float(objective.margins[-1]), 0.5, places=6)

    def test_normed_linear_output_is_cosine_bounded(self):
        layer = NormedLinear(4, 3)
        output = layer(torch.randn(5, 4))
        self.assertLessEqual(float(output.abs().max()), 1.000001)


if __name__ == "__main__":
    unittest.main()
