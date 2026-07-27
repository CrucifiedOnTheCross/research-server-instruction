from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fiftyone as fo
import pandas as pd


FLOAT_FIELDS = (
    "strength",
    "guidance_scale",
    "nearest_real_distance",
    "nearest_confusing_distance",
    "feature_margin",
    "nearest_synthetic_distance",
    "geometry_score",
)
INT_FIELDS = ("seed", "inference_steps", "stage10_rank")
BOOL_FIELDS = (
    "inside_real_manifold",
    "passes_geometry_filter",
    "selected_by_stage6",
    "stage10_selected",
)
TEXT_FIELDS = (
    "image_id",
    "source_image_id",
    "source_group_id",
    "source",
    "model_id",
    "prompt",
    "negative_prompt",
    "preprocess_mode",
    "stage10_stratum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a persistent paired synthetic-data dataset in FiftyOne.")
    parser.add_argument("--csv", default="/srv/research/projects/default/ham10000/splits/stage10/stage10_candidate_assignments.csv")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--dataset-name", default="ham10000-synthetic-audit")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def optional_value(row: pd.Series, field: str, converter: Any) -> Any | None:
    value = row.get(field)
    if pd.isna(value) or value == "":
        return None
    return converter(value)


def resolve_media(root: Path, value: Any) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return str(path)


def apply_fields(sample: fo.Sample, row: pd.Series, role: str) -> None:
    sample["role"] = role
    sample["ground_truth"] = fo.Classification(label=str(row["label"]))
    for field in TEXT_FIELDS:
        value = optional_value(row, field, str)
        if value is not None:
            sample[field] = value
    for field in FLOAT_FIELDS:
        value = optional_value(row, field, float)
        if value is not None:
            sample[field] = value
    for field in INT_FIELDS:
        value = optional_value(row, field, lambda item: int(float(item)))
        if value is not None:
            sample[field] = value
    for field in BOOL_FIELDS:
        value = optional_value(row, field, lambda item: bool(int(float(item))))
        if value is not None:
            sample[field] = value


def build_dataset(
    frame: pd.DataFrame,
    root: Path,
    source_csv: Path,
    name: str,
    replace: bool,
) -> fo.Dataset:
    if name in fo.list_datasets():
        if not replace:
            raise RuntimeError(f"FiftyOne dataset already exists: {name}; pass --replace")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name)
    dataset.persistent = True
    dataset.add_group_field("pair", default="synthetic")
    samples: list[fo.Sample] = []
    missing: dict[str, int] = {"synthetic": 0, "source": 0, "nearest_real": 0}
    for _, row in frame.iterrows():
        group = fo.Group()
        media = {
            "synthetic": resolve_media(root, row.get("image_path")),
            "source": resolve_media(root, row.get("source_image_path")),
            "nearest_real": resolve_media(root, row.get("nearest_real_image_path")),
        }
        for role, filepath in media.items():
            if filepath is None or not Path(filepath).is_file():
                missing[role] += 1
                continue
            sample = fo.Sample(filepath=filepath, pair=group.element(role))
            apply_fields(sample, row, role)
            samples.append(sample)
    dataset.add_samples(samples)
    dataset.info = {
        "source_csv": str(source_csv.resolve()),
        "rows": int(len(frame)),
        "samples": int(len(samples)),
        "missing_media": missing,
        "views": {
            "before_selection": "role == synthetic",
            "stage6_selected": "role == synthetic and selected_by_stage6 == true",
            "stage10_selected": "role == synthetic and stage10_selected == true",
        },
    }
    dataset.save()
    return dataset


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    dataset = build_dataset(
        frame,
        Path(args.data_root),
        Path(args.csv),
        args.dataset_name,
        args.replace,
    )
    print(json.dumps({"dataset": dataset.name, "samples": len(dataset), "info": dataset.info}, indent=2))


if __name__ == "__main__":
    main()
