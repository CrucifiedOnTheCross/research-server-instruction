#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from longtail_medical.provenance import code_commit, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/stage2_test_registry.yaml"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/stage2_confirmatory"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage2_confirmatory/readiness.json"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((project / args.registry).read_text(encoding="utf-8"))
    commit = code_commit(project)
    accepted = []
    errors = []
    for arm, expected in registry["arms"].items():
        for seed in registry["seeds"]:
            candidates = sorted((project / args.runs_root / arm).glob(f"*_{seed}/summary.json"))
            valid = []
            for summary_path in candidates:
                run_dir = summary_path.parent
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
                signature = json.loads((run_dir / "run_signature.json").read_text(encoding="utf-8"))
                resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
                checks = [
                    summary.get("status") == "completed",
                    summary.get("test_evaluated") is False,
                    summary.get("epochs_completed") == registry["epochs"],
                    summary.get("checkpoint_policy") == "last",
                    summary.get("git_commit") == commit,
                    summary.get("protocol_version") == expected["protocol_version"],
                    summary.get("run_signature") == signature.get("run_signature"),
                    sha256_file(run_dir / "config.resolved.yaml") == signature.get("resolved_config_sha256"),
                    signature.get("train_manifest_sha256") == expected["train_sha256"],
                    signature.get("validation_manifest_sha256") == expected["validation_sha256"],
                    signature.get("git_commit") == commit,
                    signature.get("protocol_version") == expected["protocol_version"],
                    signature.get("checkpoint_policy") == "last",
                    signature.get("epochs") == registry["epochs"],
                    split["train"]["sha256"] == expected["train_sha256"],
                    split["validation"]["sha256"] == expected["validation_sha256"],
                    (run_dir / registry["checkpoint_policy"]["primary"]).exists(),
                    (run_dir / registry["checkpoint_policy"]["secondary"]).exists(),
                    resolved["evaluation"]["run_test"] is False,
                ]
                if all(checks):
                    valid.append(run_dir)
            if len(valid) != 1:
                errors.append(f"{arm} seed={seed}: expected exactly one valid signature, found {len(valid)}")
            else:
                accepted.append(str(valid[0]))
    for arm, expected in registry["arms"].items():
        test_path = project / expected["test_manifest"]
        if sha256_file(test_path) != expected["test_sha256"]:
            errors.append(f"Test manifest hash mismatch: {arm}")
    result = {
        "ready": not errors and len(accepted) == 12,
        "git_commit": commit,
        "registry_sha256": sha256_file(project / args.registry),
        "accepted_runs": accepted,
        "errors": errors,
        "test_evaluated": False,
    }
    output = project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
