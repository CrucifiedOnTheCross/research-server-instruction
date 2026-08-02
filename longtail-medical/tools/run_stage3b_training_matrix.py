#!/usr/bin/env python3
"""Run the preregistered 2 x 3 x 3 Stage 3B confirmation matrix."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from longtail_medical.provenance import code_commit, resolved_config_sha256, run_signature, sha256_file
from tools.build_stage3b_splits import SPLIT_SEEDS


MODEL_SEEDS = (42, 43, 44)
ARMS = (
    ("ce", {"method": "cross_entropy"}),
    ("ldam_drw", {
        "method": "ldam_drw", "beta": 0.9999, "max_margin": 0.5,
        "scale": 30.0, "drw_start_epoch": 40,
    }),
)


def experiment_name(arm: str, split_seed: int) -> str:
    return f"stage3b_{arm}_resnet50_split{split_seed}"


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


def completed(project: Path, name: str, model_seed: int, signature: str) -> bool:
    root = project / "outputs" / "stage3b_confirmation" / name
    for summary_path in root.glob(f"*_{model_seed}/summary.json"):
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
            and summary.get("run_signature") == signature
            and all((summary_path.parent / item).exists() for item in required)
        ):
            return True
    return False


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    commit = code_commit(project)
    registry = json.loads((project / "data_splits/stage3b/split_registry.json").read_text(encoding="utf-8"))
    if registry["split_seeds"] != list(SPLIT_SEEDS) or registry["test_evaluated"]:
        raise RuntimeError("Invalid Stage 3B split registry")
    template = yaml.safe_load((project / "configs/stage3b_split_confirmation.yaml").read_text(encoding="utf-8"))
    runtime = project / "outputs/stage3b_confirmation/runtime_configs"
    runtime.mkdir(parents=True, exist_ok=True)
    for split_seed in SPLIT_SEEDS:
        split = registry["splits"][str(split_seed)]
        for model_seed in MODEL_SEEDS:
            for arm, loss_overrides in ARMS:
                config = copy.deepcopy(template)
                name = experiment_name(arm, split_seed)
                config["experiment"] = {"name": name, "seed": model_seed}
                config["data"].update({
                    "split_seed": split_seed,
                    "protocol_version": f"isic2019_lt_ir100_lesion_disjoint_v2_stage3b_split{split_seed}",
                    "train_csv": split["train_manifest"],
                    "val_csv": split["validation_manifest"],
                })
                config["loss"].update(loss_overrides)
                config["tracking"]["tags"].update({
                    "loss_method": config["loss"]["method"],
                    "split_seed": str(split_seed),
                    "model_seed": str(model_seed),
                })
                signature = expected_signature(project, config, commit)
                if completed(project, name, model_seed, signature):
                    print(f"SKIP signature match: {name} model_seed={model_seed}", flush=True)
                    continue
                path = runtime / f"{name}_modelseed{model_seed}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                print(f"START {name} model_seed={model_seed}", flush=True)
                subprocess.run([sys.executable, str(project / "train.py"), "--config", str(path)], cwd=project, check=True)


if __name__ == "__main__":
    main()
