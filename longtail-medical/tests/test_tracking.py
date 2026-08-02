import unittest

from longtail_medical.tracking import build_mlflow_tags


class TrackingTagsTest(unittest.TestCase):
    def test_stage2_grouping_and_provenance_tags(self):
        config = {
            "experiment": {
                "name": "stage2_monica_decontaminated_exposure_matched_resnet50_ce",
                "seed": 43,
            },
            "data": {"protocol_version": "stage2_retrospective_contamination_v2_exposure_matched"},
            "model": {"name": "resnet50"},
            "checkpoint": {"primary_policy": "last"},
            "tracking": {"tags": {"stage": "stage2_confirmatory"}},
        }
        tags = build_mlflow_tags(config, git_commit="abc", run_signature="signature")
        self.assertEqual(tags["project"], "longtail-medical")
        self.assertEqual(tags["stage"], "stage2_confirmatory")
        self.assertEqual(tags["contamination_policy"], "decontaminated_exposure_matched")
        self.assertEqual(tags["seed"], "43")
        self.assertEqual(tags["checkpoint_policy"], "last")
        self.assertEqual(tags["git_commit"], "abc")
        self.assertEqual(tags["run_signature"], "signature")


if __name__ == "__main__":
    unittest.main()
