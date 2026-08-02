#!/usr/bin/env python3
"""Run the preregistered Stage 2 validation-only training matrix."""

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
    (
        "stage2_monica_original_resnet50_ce",
        "data_splits/monica/ir100/train.csv",
        "data_splits/monica/ir100/val.csv",
        "data_splits/monica/ir100/test.csv",
        SEEDS,
        "monica_isic2019_lt_ir100_pinned_3dd808d6",
    ),
    (
        "stage2_monica_decontaminated_unmatched_resnet50_ce",
        "data_splits/stage2/monica_causal/train_decontaminated_unmatched.csv",
        "data_splits/stage2/monica_causal/val_monica.csv",
        "data_splits/stage2/monica_causal/test_monica.csv",
        SEEDS,
        "stage2_retrospective_contamination_v2_unmatched",
    ),
    (
        "stage2_monica_decontaminated_exposure_matched_resnet50_ce",
        "data_splits/stage2/monica_causal/train_decontaminated_matched.csv",
        "data_splits/stage2/monica_causal/val_monica.csv",
        "data_splits/stage2/monica_causal/test_monica.csv",
        SEEDS,
        "stage2_retrospective_contamination_v2_exposure_matched",
    ),
    (
        "stage2_lesion_disjoint_ir100_resnet50_ce",
        "data_splits/stage2/lesion_disjoint_ir100/train.csv",
        "data_splits/stage2/lesion_disjoint_ir100/val.csv",
        "data_splits/stage2/lesion_disjoint_ir100/test.csv",
        SEEDS,
        "isic2019_lt_ir100_lesion_disjoint_v1",
    ),
)


def expected_signature(project: Path, config: dict, commit: str) -> str:
    train = project / config["data"]["train_csv"]
    validation = project / config["data"]["val_csv"]
    signature, _ = run_signature(
        commit=commit,
        config_hash=resolved_config_sha256(config),
        train_hash=sha256_file(train),
        validation_hash=sha256_file(validation),
        protocol_version=config["data"]["protocol_version"],
        checkpoint_policy=config["checkpoint"]["primary_policy"],
        epochs=int(config["training"]["epochs"]),
    )
    return signature


def completed(project: Path, experiment: str, seed: int, signature: str) -> bool:
    root = project / "outputs" / "stage2_confirmatory" / experiment
    for summary_path in root.glob(f"*_{seed}/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = summary_path.parent
        if (
            summary.get("status") == "completed"
            and summary.get("test_evaluated") is False
            and summary.get("run_signature") == signature
            and summary.get("checkpoint_policy") == "last"
            and summary.get("epochs_completed") == 50
            and (run_dir / "last.pt").exists()
            and (run_dir / "best.pt").exists()
        ):
            return True
    return False


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    commit = code_commit(project)
    template = yaml.safe_load((project / "configs" / "stage1_monica_ir100_ce.yaml").read_text(encoding="utf-8"))
    runtime = project / "outputs" / "stage2_confirmatory" / "runtime_configs"
    runtime.mkdir(parents=True, exist_ok=True)
    for experiment, train_csv, val_csv, test_csv, seeds, protocol_version in ARMS:
        for seed in seeds:
            config = copy.deepcopy(template)
            config["experiment"] = {"name": experiment, "seed": seed}
            config["data"].update({"train_csv": train_csv, "val_csv": val_csv, "test_csv": test_csv})
            config["data"]["protocol_version"] = protocol_version
            config["tracking"]["output_root"] = "outputs/stage2_confirmatory"
            config["tracking"]["mlflow_experiment"] = "ISIC2019-LT Stage2 Confirmatory"
            config["tracking"]["tags"]["stage"] = "stage2_confirmatory"
            config["evaluation"]["run_test"] = False
            signature = expected_signature(project, config, commit)
            if completed(project, experiment, seed, signature):
                print(f"SKIP signature match: {experiment} seed={seed} signature={signature}", flush=True)
                continue
            path = runtime / f"{experiment}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"START {experiment} seed={seed}", flush=True)
            subprocess.run([sys.executable, str(project / "train.py"), "--config", str(path)], cwd=project, check=True)


if __name__ == "__main__":
    main()
