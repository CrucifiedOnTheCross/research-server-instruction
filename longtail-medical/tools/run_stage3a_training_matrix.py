#!/usr/bin/env python3
"""Run the preregistered Stage 3A validation-only loss screening matrix."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from longtail_medical.provenance import code_commit, resolved_config_sha256, run_signature, sha256_file


SEEDS = (42, 43, 44)
ARMS = (
    ("stage3a_ce_resnet50_lesion_disjoint", {"method": "cross_entropy"}),
    ("stage3a_weighted_ce_resnet50_lesion_disjoint", {"method": "weighted_cross_entropy"}),
    ("stage3a_focal_resnet50_lesion_disjoint", {"method": "focal", "gamma": 2.0}),
    ("stage3a_cb_focal_resnet50_lesion_disjoint", {"method": "class_balanced_focal", "beta": 0.9999, "gamma": 2.0}),
    ("stage3a_balanced_softmax_resnet50_lesion_disjoint", {"method": "balanced_softmax"}),
    ("stage3a_ldam_drw_resnet50_lesion_disjoint", {
        "method": "ldam_drw", "beta": 0.9999, "max_margin": 0.5,
        "scale": 30.0, "drw_start_epoch": 40,
    }),
)


def expected_signature(project: Path, config: dict, commit: str) -> str:
    signature, _ = run_signature(
        commit=commit,
        config_hash=resolved_config_sha256(config),
        train_hash=sha256_file(project / config["data"]["train_csv"]),
        validation_hash=sha256_file(project / config["data"]["val_csv"]),
        protocol_version=config["data"]["protocol_version"],
        checkpoint_policy=config["checkpoint"]["primary_policy"],
        epochs=int(config["training"]["epochs"]),
    )
    return signature


def completed(project: Path, experiment: str, seed: int, signature: str) -> bool:
    root = project / "outputs" / "stage3a_screening" / experiment
    for summary_path in root.glob(f"*_{seed}/summary.json"):
        run_dir = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = (
            "last.pt", "best.pt", "config.resolved.yaml", "split_manifest.json",
            "run_signature.json", "loss_initialization.json", "model_initialization.json",
            "val_metrics_last.json", "val_metrics_best.json",
            "val_predictions_last.csv", "val_predictions_best.csv",
        )
        if (
            summary.get("status") == "completed"
            and summary.get("test_evaluated") is False
            and summary.get("epochs_completed") == 50
            and summary.get("checkpoint_policy") == "last"
            and summary.get("monitor") == "lesion_mcc"
            and summary.get("run_signature") == signature
            and all((run_dir / item).exists() for item in required)
        ):
            return True
    return False


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    commit = code_commit(project)
    template = yaml.safe_load(
        (project / "configs" / "stage3a_lesion_disjoint_screening.yaml").read_text(encoding="utf-8")
    )
    runtime = project / "outputs" / "stage3a_screening" / "runtime_configs"
    runtime.mkdir(parents=True, exist_ok=True)
    for experiment, loss_overrides in ARMS:
        for seed in SEEDS:
            config = copy.deepcopy(template)
            config["experiment"] = {"name": experiment, "seed": seed}
            config["loss"].update(loss_overrides)
            config["tracking"]["tags"]["loss_method"] = config["loss"]["method"]
            signature = expected_signature(project, config, commit)
            if completed(project, experiment, seed, signature):
                print(f"SKIP signature match: {experiment} seed={seed}", flush=True)
                continue
            path = runtime / f"{experiment}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"START {experiment} seed={seed}", flush=True)
            subprocess.run(
                [sys.executable, str(project / "train.py"), "--config", str(path)],
                cwd=project, check=True,
            )


if __name__ == "__main__":
    main()
