#!/usr/bin/env python3
"""Build the three preregistered lesion-disjoint Stage 3B splits in parallel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from longtail_medical.provenance import sha256_file
from tools.build_stage2_protocols import build_lesion_disjoint


SPLIT_SEEDS = (101, 202, 303)


def build_one(job: tuple[Path, Path, int]) -> tuple[int, dict]:
    root, output_root, seed = job
    return seed, build_lesion_disjoint(root, output_root / f"split_{seed}", seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_splits/stage3b"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(args.root, output, seed) for seed in SPLIT_SEEDS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        reports = dict(executor.map(build_one, jobs))
    registry = {
        "protocol_version": "stage3b_split_generalization_v1",
        "split_seeds": list(SPLIT_SEEDS),
        "test_evaluated": False,
        "splits": {},
    }
    for seed in SPLIT_SEEDS:
        split = output / f"split_{seed}"
        report = reports[seed]
        if any(report["lesion_overlap"].values()):
            raise RuntimeError(f"Split {seed} contains lesion overlap")
        registry["splits"][str(seed)] = {
            "protocol": report["protocol"],
            "train_manifest": str((split / "train.csv").relative_to(project)),
            "validation_manifest": str((split / "val.csv").relative_to(project)),
            "test_manifest": str((split / "test.csv").relative_to(project)),
            "train_sha256": sha256_file(split / "train.csv"),
            "validation_sha256": sha256_file(split / "val.csv"),
            "test_sha256": sha256_file(split / "test.csv"),
            "counts": report["counts"],
            "class_counts": report["class_counts"],
            "lesion_overlap": report["lesion_overlap"],
        }
    destination = output / "split_registry.json"
    destination.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
