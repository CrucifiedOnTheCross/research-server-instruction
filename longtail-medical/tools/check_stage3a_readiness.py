#!/usr/bin/env python3
"""Validate completeness and test isolation for Stage 3A screening."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from longtail_medical.provenance import code_commit, sha256_file
from tools.run_stage3a_training_matrix import ARMS, SEEDS


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    commit = code_commit(project)
    root = project / "outputs" / "stage3a_screening"
    accepted = []
    errors = []
    train_hash = sha256_file(project / "data_splits/stage2/lesion_disjoint_ir100/train.csv")
    validation_hash = sha256_file(project / "data_splits/stage2/lesion_disjoint_ir100/val.csv")
    for arm, _ in ARMS:
        for seed in SEEDS:
            candidates = sorted((root / arm).glob(f"*_{seed}/summary.json"))
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
                    summary.get("monitor") == "lesion_mcc",
                    summary.get("git_commit") == commit,
                    summary.get("run_signature") == signature.get("run_signature"),
                    split.get("test_loaded") is False,
                    split.get("test_evaluated") is False,
                    split["train"]["sha256"] == train_hash,
                    split["validation"]["sha256"] == validation_hash,
                    config["evaluation"]["run_test"] is False,
                    "test_csv" not in config["data"],
                    last.get("epoch") == 50,
                    last.get("checkpoint_policy") == "last",
                    best.get("checkpoint_policy") == "best_validation",
                    last.get("run_signature") == summary.get("run_signature"),
                    best.get("run_signature") == summary.get("run_signature"),
                    (run / "loss_initialization.json").exists(),
                    (run / "val_metrics_last.json").exists(),
                    (run / "val_predictions_last.csv").exists(),
                )
                if all(checks):
                    valid.append(run)
            if len(valid) != 1:
                errors.append(f"{arm} seed={seed}: expected one valid run, found {len(valid)}")
            else:
                accepted.append(str(valid[0]))
    derived = []
    for seed in SEEDS:
        summary_path = project / "outputs" / "stage3a_logit_adjustment_tau1" / f"seed_{seed}" / "summary.json"
        if not summary_path.exists():
            errors.append(f"Logit adjustment seed={seed}: summary missing")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test_evaluated") is not False or summary.get("tau") != 1.0:
            errors.append(f"Logit adjustment seed={seed}: invalid test lock or tau")
        else:
            derived.append(str(summary_path.parent))
    result = {
        "ready": not errors and len(accepted) == 18 and len(derived) == 3,
        "git_commit": commit,
        "accepted_training_runs": accepted,
        "accepted_derived_runs": derived,
        "errors": errors,
        "test_loaded": False,
        "test_evaluated": False,
    }
    destination = root / "readiness.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
