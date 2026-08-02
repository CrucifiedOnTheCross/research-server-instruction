#!/usr/bin/env python3
"""Open the Stage 3B test gate only for one homogeneous preregistered matrix."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from longtail_medical.provenance import code_commit, sha256_file
from tools.build_stage3b_splits import SPLIT_SEEDS
from tools.run_stage3b_training_matrix import ARMS, MODEL_SEEDS, experiment_name


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    commit = code_commit(project)
    registry_path = project / "data_splits/stage3b/split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    accepted, errors = [], []
    for split_seed in SPLIT_SEEDS:
        expected = registry["splits"][str(split_seed)]
        for arm, _ in ARMS:
            name = experiment_name(arm, split_seed)
            for model_seed in MODEL_SEEDS:
                candidates = sorted((project / "outputs/stage3b_confirmation" / name).glob(f"*_{model_seed}/summary.json"))
                valid = []
                for summary_path in candidates:
                    run = summary_path.parent
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        split = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))
                        signature = json.loads((run / "run_signature.json").read_text(encoding="utf-8"))
                        config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
                        last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
                        best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                    except Exception:
                        continue
                    checks = (
                        summary.get("status") == "completed",
                        summary.get("test_evaluated") is False,
                        summary.get("epochs_completed") == 50,
                        summary.get("git_commit") == commit,
                        summary.get("run_signature") == signature.get("run_signature"),
                        config["data"].get("split_seed") == split_seed,
                        config["experiment"]["seed"] == model_seed,
                        config["evaluation"]["run_test"] is False,
                        "test_csv" not in config["data"],
                        split.get("test_loaded") is False,
                        split.get("test_evaluated") is False,
                        split["train"]["sha256"] == expected["train_sha256"],
                        split["validation"]["sha256"] == expected["validation_sha256"],
                        last.get("epoch") == 50 and last.get("checkpoint_policy") == "last",
                        best.get("checkpoint_policy") == "best_validation",
                    )
                    if all(checks):
                        valid.append(run)
                if len(valid) != 1:
                    errors.append(f"{name} model_seed={model_seed}: valid={len(valid)}")
                else:
                    accepted.append(str(valid[0]))
    derivations = []
    for kind in ("logit_adjustment", "temperature_scaling"):
        for split_seed in SPLIT_SEEDS:
            for model_seed in MODEL_SEEDS:
                path = project / "outputs/stage3b_derivations" / kind / f"split_{split_seed}" / f"model_seed_{model_seed}"
                try:
                    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
                except Exception:
                    errors.append(f"Missing derivation: {kind}/{split_seed}/{model_seed}")
                    continue
                if summary.get("test_evaluated") is not False or summary.get("selection_data") != "validation_only":
                    errors.append(f"Invalid derivation: {kind}/{split_seed}/{model_seed}")
                else:
                    derivations.append(str(path))
    for split_seed in SPLIT_SEEDS:
        expected = registry["splits"][str(split_seed)]
        if sha256_file(project / expected["test_manifest"]) != expected["test_sha256"]:
            errors.append(f"Test manifest hash mismatch: split {split_seed}")
    result = {
        "ready": not errors and len(accepted) == 18 and len(derivations) == 18,
        "git_commit": commit, "split_registry_sha256": sha256_file(registry_path),
        "accepted_runs": accepted, "accepted_derivations": derivations,
        "errors": errors, "test_loaded": False, "test_evaluated": False,
    }
    output = project / "outputs/stage3b_confirmation/readiness.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
