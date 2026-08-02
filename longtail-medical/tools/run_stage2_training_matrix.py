#!/usr/bin/env python3
"""Run the preregistered Stage 2 validation-only training matrix."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


SEEDS = (42, 43, 44)
ARMS = (
    (
        "stage2_monica_original_resnet50_ce",
        "data_splits/monica/ir100/train.csv",
        "data_splits/monica/ir100/val.csv",
        "data_splits/monica/ir100/test.csv",
        (43, 44),  # Seed 42 is the completed Stage 1 anchor.
    ),
    (
        "stage2_monica_decontaminated_unmatched_resnet50_ce",
        "data_splits/stage2/monica_causal/train_decontaminated_unmatched.csv",
        "data_splits/stage2/monica_causal/val_monica.csv",
        "data_splits/stage2/monica_causal/test_monica.csv",
        SEEDS,
    ),
    (
        "stage2_monica_decontaminated_exposure_matched_resnet50_ce",
        "data_splits/stage2/monica_causal/train_decontaminated_matched.csv",
        "data_splits/stage2/monica_causal/val_monica.csv",
        "data_splits/stage2/monica_causal/test_monica.csv",
        SEEDS,
    ),
    (
        "stage2_lesion_disjoint_ir100_resnet50_ce",
        "data_splits/stage2/lesion_disjoint_ir100/train.csv",
        "data_splits/stage2/lesion_disjoint_ir100/val.csv",
        "data_splits/stage2/lesion_disjoint_ir100/test.csv",
        SEEDS,
    ),
)


def completed(project: Path, experiment: str, seed: int) -> bool:
    root = project / "outputs" / "stage2" / experiment
    for summary_path in root.glob(f"*_{seed}/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") == "completed" and summary.get("test_evaluated") is False:
            return True
    return False


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    template = yaml.safe_load((project / "configs" / "stage1_monica_ir100_ce.yaml").read_text(encoding="utf-8"))
    runtime = project / "outputs" / "stage2" / "runtime_configs"
    runtime.mkdir(parents=True, exist_ok=True)
    for experiment, train_csv, val_csv, test_csv, seeds in ARMS:
        for seed in seeds:
            if completed(project, experiment, seed):
                print(f"SKIP completed: {experiment} seed={seed}", flush=True)
                continue
            config = copy.deepcopy(template)
            config["experiment"] = {"name": experiment, "seed": seed}
            config["data"].update({"train_csv": train_csv, "val_csv": val_csv, "test_csv": test_csv})
            config["tracking"]["output_root"] = "outputs/stage2"
            config["tracking"]["mlflow_experiment"] = "ISIC2019-LT Stage2 Leakage"
            config["evaluation"]["run_test"] = False
            path = runtime / f"{experiment}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"START {experiment} seed={seed}", flush=True)
            subprocess.run([sys.executable, str(project / "train.py"), "--config", str(path)], cwd=project, check=True)


if __name__ == "__main__":
    main()
