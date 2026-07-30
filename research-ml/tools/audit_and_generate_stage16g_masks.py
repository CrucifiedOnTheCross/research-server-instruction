from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as transform

from tools.stage16g_mask_metrics import (
    deterministic_group_sample,
    mask_characteristics,
    overlap_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage 16G segmentation and create rare-class train masks."
    )
    parser.add_argument(
        "--protocol", default="configs/stage16g_generator_qualification.yaml"
    )
    parser.add_argument(
        "--segmenter", default="configs/stage16g_segmentation_qualification.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/stage16g_segmentation/"
            "stage16g_isic2018_deeplabv3_segmentation_384/best.pt"
        ),
    )
    parser.add_argument(
        "--report-root", default="outputs/reports/stage16g_segmentation_audit"
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def preprocess(image: Image.Image, size: int) -> tuple[torch.Tensor, dict[str, int]]:
    width, height = image.size
    scale = size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, (left, top))
    tensor = transform.normalize(
        transform.to_tensor(canvas),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return tensor, {
        "width": width,
        "height": height,
        "left": left,
        "top": top,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }


def restore_probability(
    probability: torch.Tensor, geometry: dict[str, int]
) -> np.ndarray:
    cropped = probability[
        geometry["top"] : geometry["top"] + geometry["resized_height"],
        geometry["left"] : geometry["left"] + geometry["resized_width"],
    ]
    restored = functional.interpolate(
        cropped[None, None],
        size=(geometry["height"], geometry["width"]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return restored.float().cpu().numpy()


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


def predict(
    model: torch.nn.Module, image: Image.Image, size: int, device: torch.device
) -> np.ndarray:
    tensor, geometry = preprocess(image, size)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        probability = model(tensor[None].to(device))["out"].sigmoid()[0, 0]
    return restore_probability(probability, geometry)


def main() -> None:
    args = parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    segmenter = yaml.safe_load(Path(args.segmenter).read_text(encoding="utf-8"))
    data_root = Path(protocol["data"]["root"])
    output = Path(args.report_root)
    output.mkdir(parents=True, exist_ok=True)
    threshold = float(segmenter["evaluation"]["threshold"])
    size = int(segmenter["data"]["image_size"])
    device = torch.device("cuda")
    model = load_model(Path(args.checkpoint), device)

    qualification_rows = read_csv(
        data_root / protocol["data"]["segmentation_qualification_manifest"]
    )
    qualification: list[dict[str, Any]] = []
    for row in qualification_rows:
        with Image.open(data_root / row["image_path"]) as handle:
            image = handle.convert("RGB")
        with Image.open(data_root / row["mask_path"]) as handle:
            truth = np.asarray(handle.convert("L")) >= 128
        probability = predict(model, image, size, device)
        metrics = {
            "image_id": row["image_id"],
            **mask_characteristics(probability, threshold),
            **overlap_metrics(probability >= threshold, truth),
        }
        qualification.append(metrics)
    qualification_frame = pd.DataFrame(qualification)
    qualification_frame.to_csv(output / "qualification_per_image.csv", index=False)
    qualification_frame["area_quartile"] = pd.qcut(
        qualification_frame["gt_area_fraction"],
        q=4,
        labels=["Q1_small", "Q2", "Q3", "Q4_large"],
        duplicates="drop",
    )
    subgroup = (
        qualification_frame.groupby("area_quartile", observed=True)[
            ["dice", "iou", "gt_area_fraction"]
        ]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    subgroup.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in subgroup.columns
    ]
    subgroup.to_csv(output / "qualification_area_subgroups.csv", index=False)

    train_rows = read_csv(data_root / protocol["data"]["train_split"])
    settings = protocol["segmentation_bootstrap"]
    candidates = deterministic_group_sample(
        train_rows,
        list(protocol["targeting"]["candidate_classes"]),
        int(settings["pseudo_mask_candidates_per_class"]),
    )
    mask_root = data_root / "auxiliary" / "stage16g_pseudo_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    pseudo_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidates, start=1):
        with Image.open(data_root / row["image_path"]) as handle:
            image = handle.convert("RGB")
        probability = predict(model, image, size, device)
        characteristics = mask_characteristics(probability, threshold)
        passed = (
            characteristics["foreground_confidence"]
            >= float(settings["pseudo_mask_minimum_confidence"])
            and characteristics["largest_component_fraction"]
            >= float(settings["pseudo_mask_minimum_largest_component_fraction"])
            and characteristics["component_count"]
            <= int(settings["pseudo_mask_maximum_components"])
            and 0.005 <= characteristics["area_fraction"] <= 0.95
        )
        mask_path = mask_root / f"{row['image_id']}_segmentation.png"
        Image.fromarray(
            ((probability >= threshold) * 255).astype(np.uint8), mode="L"
        ).save(mask_path)
        pseudo_rows.append(
            {
                **row,
                "pseudo_mask_path": mask_path.relative_to(data_root).as_posix(),
                "pseudo_mask_passed": int(passed),
                **characteristics,
            }
        )
        if index % 100 == 0:
            pd.DataFrame(pseudo_rows).to_csv(
                output / "pseudo_mask_manifest.partial.csv", index=False
            )
    pseudo_frame = pd.DataFrame(pseudo_rows)
    pseudo_frame.to_csv(output / "pseudo_mask_manifest.csv", index=False)
    class_summary = (
        pseudo_frame.groupby("label")
        .agg(
            candidates=("image_id", "size"),
            passed=("pseudo_mask_passed", "sum"),
            mean_confidence=("foreground_confidence", "mean"),
            mean_area_fraction=("area_fraction", "mean"),
        )
        .reset_index()
    )
    class_summary.to_csv(output / "pseudo_mask_class_summary.csv", index=False)
    summary = {
        "protocol": "stage16g_segmentation_audit_v1",
        "locked_test_evaluated": False,
        "qualification_images": len(qualification_frame),
        "qualification_mean_dice": float(qualification_frame["dice"].mean()),
        "qualification_mean_iou": float(qualification_frame["iou"].mean()),
        "qualification_worst_area_quartile_dice": float(
            subgroup["dice_mean"].min()
        ),
        "pseudo_mask_candidates": len(pseudo_frame),
        "pseudo_mask_passed": int(pseudo_frame["pseudo_mask_passed"].sum()),
        "checkpoint": str(Path(args.checkpoint)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
