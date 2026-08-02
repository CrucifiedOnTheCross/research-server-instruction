import unittest

from longtail_medical.tracking import build_mlflow_tags, namespace_epoch_metrics


class TrackingTagsTest(unittest.TestCase):
    def test_epoch_metrics_keep_legacy_and_add_namespaces(self):
        metrics = namespace_epoch_metrics({
            "epoch": 1,
            "train_loss": 0.5,
            "val_mcc": 0.4,
            "epoch_seconds": 12.0,
            "gpu_max_memory_gib": 11.0,
        })
        self.assertEqual(metrics["train_loss"], 0.5)
        self.assertEqual(metrics["train/loss"], 0.5)
        self.assertEqual(metrics["val/mcc"], 0.4)
        self.assertEqual(metrics["system/epoch_seconds"], 12.0)

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

    def test_stage3b_has_separate_split_and_model_seed_tags(self):
        config = {
            "experiment": {"name": "stage3b_ldam_drw_resnet50_split202", "seed": 44},
            "data": {"protocol_version": "stage3b", "split_seed": 202},
            "model": {"name": "resnet50"},
            "checkpoint": {"primary_policy": "last"},
            "tracking": {"tags": {"stage": "stage3b_confirmation"}},
        }
        tags = build_mlflow_tags(config)
        self.assertEqual(tags["experiment_arm"], "ldam_drw_resnet50_split202")
        self.assertEqual(tags["split_seed"], "202")
        self.assertEqual(tags["model_seed"], "44")


if __name__ == "__main__":
    unittest.main()
