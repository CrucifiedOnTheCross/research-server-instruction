import unittest

from longtail_medical.provenance import run_signature


class ProvenanceTest(unittest.TestCase):
    def test_signature_is_stable_and_sensitive(self):
        fields = {
            "commit": "abc",
            "config_hash": "cfg",
            "train_hash": "train",
            "validation_hash": "val",
            "protocol_version": "v1",
            "checkpoint_policy": "last",
            "epochs": 50,
        }
        first, payload = run_signature(**fields)
        second, _ = run_signature(**fields)
        self.assertEqual(first, second)
        self.assertEqual(payload["checkpoint_policy"], "last")
        changed, _ = run_signature(**{**fields, "epochs": 49})
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
