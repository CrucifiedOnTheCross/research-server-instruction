from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tools.make_source_matched_replay import build_source_replay
except ModuleNotFoundError:
    from make_source_matched_replay import build_source_replay


STRATA = ("strict_id", "aid_radial", "ood_far", "random_remaining")
REQUIRED_SCORE_COLUMNS = {
    "image_id",
    "image_path",
    "label",
    "is_synthetic",
    "source_image_id",
    "source_image_path",
    "source_group_id",
    "nearest_real_distance",
    "feature_margin",
    "inside_real_manifold",
    "geometry_score",
    "passes_geometry_filter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build equal-dose Stage 10 geometry strata and source-matched replay controls."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-csv", default="splits/stage8/train_real.csv")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--out-dir", default="splits/stage10")
    parser.add_argument("--target-classes", default="mel,akiec,bkl")
    parser.add_argument("--dose-per-class", type=int, default=30)
    parser.add_argument("--sample-weight", type=float, default=0.5)
    parser.add_argument("--max-source-reuse-per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(row: pd.Series) -> str:
    return str(row["source_image_id"])


def select_with_source_cap(
    candidates: pd.DataFrame,
    dose: int,
    max_source_reuse: int,
) -> pd.DataFrame:
    selected_indices: list[int] = []
    source_counts: Counter[str] = Counter()
    for index, row in candidates.iterrows():
        source = source_key(row)
        if source_counts[source] >= max_source_reuse:
            continue
        selected_indices.append(index)
        source_counts[source] += 1
        if len(selected_indices) == dose:
            break
    if len(selected_indices) != dose:
        raise ValueError(
            f"Could select only {len(selected_indices)} of {dose} rows with "
            f"max_source_reuse={max_source_reuse}"
        )
    return candidates.loc[selected_indices].copy()


def build_strata(
    scores: pd.DataFrame,
    target_classes: tuple[str, ...],
    dose_per_class: int,
    max_source_reuse: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    missing = REQUIRED_SCORE_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(f"Synthetic scores are missing columns: {sorted(missing)}")
    if dose_per_class <= 0 or max_source_reuse <= 0:
        raise ValueError("dose_per_class and max_source_reuse must be positive")
    if scores["image_id"].duplicated().any():
        raise ValueError("Synthetic scores contain duplicate image_id values")

    pool = scores[scores["label"].astype(str).isin(target_classes)].copy()
    pool["stage10_stratum"] = ""
    pool["stage10_selected"] = 0
    pool["stage10_rank"] = np.nan
    selections: dict[str, list[pd.DataFrame]] = {stratum: [] for stratum in STRATA}
    class_metadata: dict[str, Any] = {}
    used_ids: set[str] = set()

    for class_offset, label in enumerate(target_classes):
        class_pool = pool[pool["label"].astype(str) == label].copy()
        if class_pool.empty:
            raise ValueError(f"No synthetic candidates for class {label}")
        q50 = float(class_pool["nearest_real_distance"].quantile(0.50))
        q75 = float(class_pool["nearest_real_distance"].quantile(0.75))
        midpoint = (q50 + q75) / 2.0

        strict_candidates = class_pool[
            class_pool["passes_geometry_filter"].astype(int) == 1
        ].sort_values(
            ["geometry_score", "nearest_real_distance", "image_id"],
            ascending=[False, True, True],
        )
        aid_candidates = class_pool[
            (class_pool["passes_geometry_filter"].astype(int) == 0)
            & (class_pool["nearest_real_distance"] > q50)
            & (class_pool["nearest_real_distance"] <= q75)
        ].copy()
        aid_candidates["_radial_midpoint_distance"] = (
            aid_candidates["nearest_real_distance"] - midpoint
        ).abs()
        aid_candidates = aid_candidates.sort_values(
            ["_radial_midpoint_distance", "feature_margin", "image_id"],
            ascending=[True, False, True],
        )
        ood_candidates = class_pool[
            (class_pool["inside_real_manifold"].astype(int) == 0)
            & (class_pool["nearest_real_distance"] > q75)
        ].sort_values(
            ["nearest_real_distance", "feature_margin", "image_id"],
            ascending=[False, True, True],
        )

        candidate_sets = {
            "strict_id": strict_candidates,
            "aid_radial": aid_candidates,
            "ood_far": ood_candidates,
        }
        class_selected: dict[str, pd.DataFrame] = {}
        for stratum, candidates in candidate_sets.items():
            candidates = candidates[
                ~candidates["image_id"].astype(str).isin(used_ids)
            ]
            chosen = select_with_source_cap(
                candidates, dose_per_class, max_source_reuse
            )
            class_selected[stratum] = chosen
            used_ids.update(chosen["image_id"].astype(str))

        remaining = class_pool[
            ~class_pool["image_id"].astype(str).isin(used_ids)
        ].copy()
        rng = np.random.default_rng(seed + class_offset)
        remaining["_random_order"] = rng.permutation(len(remaining))
        remaining = remaining.sort_values(["_random_order", "image_id"])
        random_chosen = select_with_source_cap(
            remaining, dose_per_class, max_source_reuse
        )
        class_selected["random_remaining"] = random_chosen
        used_ids.update(random_chosen["image_id"].astype(str))

        class_metadata[label] = {
            "pool_rows": int(len(class_pool)),
            "nearest_real_distance_q50": q50,
            "nearest_real_distance_q75": q75,
            "candidate_rows": {
                stratum: int(len(candidates))
                for stratum, candidates in {
                    **candidate_sets,
                    "random_remaining": remaining,
                }.items()
            },
        }
        for stratum, selected in class_selected.items():
            selected = selected.copy()
            selected["stage10_stratum"] = stratum
            selected["stage10_selected"] = 1
            selected["stage10_rank"] = np.arange(1, len(selected) + 1)
            selections[stratum].append(selected)

    combined = {
        stratum: pd.concat(rows, ignore_index=True, sort=False)
        for stratum, rows in selections.items()
    }
    all_selected = pd.concat(combined.values(), ignore_index=True, sort=False)
    if all_selected["image_id"].duplicated().any():
        raise ValueError("Stage 10 strata overlap")
    expected_rows = len(target_classes) * dose_per_class
    for stratum, frame in combined.items():
        if len(frame) != expected_rows:
            raise ValueError(f"{stratum}: expected {expected_rows} rows, found {len(frame)}")
        counts = frame["label"].value_counts().to_dict()
        if any(int(counts.get(label, 0)) != dose_per_class for label in target_classes):
            raise ValueError(f"{stratum}: class dose mismatch {counts}")

    assignment = pool.copy()
    assignment_by_id = all_selected.set_index("image_id")[
        ["stage10_stratum", "stage10_selected", "stage10_rank"]
    ]
    assignment = assignment.drop(
        columns=["stage10_stratum", "stage10_selected", "stage10_rank"]
    ).join(assignment_by_id, on="image_id")
    assignment["stage10_stratum"] = assignment["stage10_stratum"].fillna("")
    assignment["stage10_selected"] = (
        assignment["stage10_selected"].fillna(0).astype(int)
    )

    metadata = {
        "protocol": "stage10_equal_dose_geometry_strata",
        "target_classes": list(target_classes),
        "strata": list(STRATA),
        "dose_per_class": int(dose_per_class),
        "rows_per_stratum": int(expected_rows),
        "max_source_reuse_per_stratum": int(max_source_reuse),
        "seed": int(seed),
        "definitions": {
            "strict_id": "passes_geometry_filter == 1",
            "aid_radial": "class q50 < nearest_real_distance <= class q75; strict rows excluded",
            "ood_far": "outside PRDC real manifold and nearest_real_distance > class q75",
            "random_remaining": "deterministic random sample after excluding all other selected rows",
        },
        "class_thresholds": class_metadata,
        "selection_counts": {
            stratum: {
                str(label): int(count)
                for label, count in frame["label"].value_counts().sort_index().items()
            }
            for stratum, frame in combined.items()
        },
        "strata_overlap_rows": 0,
    }
    return combined, assignment, metadata


def build_outputs(
    train: pd.DataFrame,
    scores: pd.DataFrame,
    target_classes: tuple[str, ...],
    dose_per_class: int,
    max_source_reuse: int,
    sample_weight: float,
    seed: int,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame, dict[str, Any]]:
    strata, assignment, metadata = build_strata(
        scores,
        target_classes=target_classes,
        dose_per_class=dose_per_class,
        max_source_reuse=max_source_reuse,
        seed=seed,
    )
    real_train = train.copy()
    if "is_synthetic" in real_train:
        real_train = real_train[real_train["is_synthetic"].astype(int) == 0].copy()
    if real_train["image_id"].duplicated().any():
        raise ValueError("Real train CSV contains duplicate image_id values")

    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    replay_metadata: dict[str, Any] = {}
    for stratum, selected in strata.items():
        synthetic_train = pd.concat(
            [real_train, selected], ignore_index=True, sort=False
        )
        selected_for_replay = selected.copy()
        selected_for_replay["selected_by_stage6"] = 1
        replay_train, replay_rows, replay_info = build_source_replay(
            real_train,
            selected_for_replay,
            sample_weight=sample_weight,
            expected_selected=len(selected),
        )
        outputs[stratum] = {
            "selected": selected,
            "synthetic_train": synthetic_train,
            "replay_rows": replay_rows,
            "replay_train": replay_train,
        }
        replay_metadata[stratum] = replay_info
    metadata["sample_weight"] = float(sample_weight)
    metadata["replay_controls"] = replay_metadata
    return outputs, assignment, metadata


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    train_path = resolve(root, args.train_csv)
    scores_path = resolve(root, args.scores_csv)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_classes = tuple(
        item.strip() for item in args.target_classes.split(",") if item.strip()
    )

    train = pd.read_csv(train_path)
    scores = pd.read_csv(scores_path)
    outputs, assignment, metadata = build_outputs(
        train,
        scores,
        target_classes=target_classes,
        dose_per_class=int(args.dose_per_class),
        max_source_reuse=int(args.max_source_reuse_per_stratum),
        sample_weight=float(args.sample_weight),
        seed=int(args.seed),
    )
    metadata["inputs"] = {
        "train_csv": str(train_path),
        "train_csv_sha256": sha256(train_path),
        "scores_csv": str(scores_path),
        "scores_csv_sha256": sha256(scores_path),
    }

    assignment.to_csv(out_dir / "stage10_candidate_assignments.csv", index=False)
    for stratum, frames in outputs.items():
        frames["selected"].to_csv(
            out_dir / f"selected_synthetic_{stratum}.csv", index=False
        )
        frames["synthetic_train"].to_csv(
            out_dir / f"train_synthetic_{stratum}.csv", index=False
        )
        frames["replay_rows"].to_csv(
            out_dir / f"source_replay_rows_{stratum}.csv", index=False
        )
        frames["replay_train"].to_csv(
            out_dir / f"train_source_replay_{stratum}.csv", index=False
        )
    (out_dir / "stage10_strata_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
