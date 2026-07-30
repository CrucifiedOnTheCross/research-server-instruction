from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.datasets import build_transforms, make_dataloaders
from src.models import create_model, extract_features
from tools.audit_and_generate_stage16g_masks import load_model as load_segmenter
from tools.audit_and_generate_stage16g_masks import predict as predict_mask
from tools.generate_stage16g_generator_smoke import fit_image, prepare_mask
from tools.stage16g_generator_metrics import summarize_arm
from tools.stage16g_mask_metrics import overlap_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify Stage 16G generator smoke outputs.")
    parser.add_argument("--config", default="configs/stage16g_generator_smoke.yaml")
    parser.add_argument(
        "--classifier-run-dir",
        default="outputs/stage16_isic2019_real_ce_natural_384/20260729-211405_42",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        default="outputs/stage16g_segmentation/stage16g_isic2018_deeplabv3_segmentation_384/best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.rows[index]["path"]).convert("RGB")
        return self.transform(image), index


@torch.no_grad()
def classifier_outputs(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    transform: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ImageRows(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    probabilities: list[np.ndarray] = []
    features: list[np.ndarray] = []
    model.eval()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            embedding = extract_features(model, images)
        probabilities.append(functional.softmax(logits.float(), dim=1).cpu().numpy())
        features.append(embedding.float().cpu().numpy())
    feature_array = np.concatenate(features)
    feature_array /= np.linalg.norm(feature_array, axis=1, keepdims=True).clip(min=1e-12)
    return np.concatenate(probabilities), feature_array


def pixel_metrics(source: Image.Image, generated: Image.Image, mask: Image.Image) -> dict[str, float]:
    source_array = np.asarray(source, dtype=np.float32) / 255.0
    generated_array = np.asarray(generated, dtype=np.float32) / 255.0
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5
    difference = np.abs(source_array - generated_array).mean(axis=2)
    return {
        "outside_mask_mae": float(difference[~mask_array].mean()),
        "inside_mask_mae": float(difference[mask_array].mean()),
    }


def make_contact_sheet(frame: pd.DataFrame, output: Path, size: int = 192) -> None:
    rows = frame.sort_values(["label", "source_image_id", "generator_arm", "seed"])
    groups = list(rows.groupby(["label", "source_image_id"], sort=True))
    canvas = Image.new("RGB", (size * 5, (size + 28) * len(groups)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, ((label, source_id), group) in enumerate(groups):
        y = row_index * (size + 28)
        source = fit_image(Path(group.iloc[0]["source_image_path"]), size, "RGB")
        mask = prepare_mask(Path(group.iloc[0]["mask_path"]), size, 0, 0)
        overlay = source.copy()
        red = Image.new("RGB", overlay.size, (255, 0, 0))
        overlay = Image.blend(overlay, Image.composite(red, overlay, mask), 0.35)
        canvas.paste(source, (0, y))
        canvas.paste(overlay, (size, y))
        for column, (_, item) in enumerate(group.head(3).iterrows(), start=2):
            canvas.paste(fit_image(Path(item["image_path"]), size, "RGB"), (column * size, y))
        draw.text((4, y + size + 4), f"{label} {source_id}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(config["data"]["output_root"])
    manifests = []
    for arm_name in config["arms"]:
        frame = pd.read_csv(output_root / arm_name / "manifest.csv")
        frame["generator_arm"] = arm_name
        manifests.append(frame)
    manifest = pd.concat(manifests, ignore_index=True)
    manifest["path"] = manifest["image_path"].astype(str)

    device = torch.device("cuda")
    classifier_run = Path(args.classifier_run_dir)
    classifier_config = load_config(
        classifier_run / "config.resolved.yaml",
        ["runtime.num_workers=0", "runtime.persistent_workers=false"],
    )
    bundle = make_dataloaders(classifier_config)
    class_to_idx = bundle.class_to_idx
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    classifier = create_model(classifier_config, len(class_to_idx)).to(device)
    classifier.load_state_dict(
        torch.load(classifier_run / "best.pt", map_location=device, weights_only=True)["model"]
    )
    transform = build_transforms(classifier_config, train=False)

    generated_rows = [{"path": path} for path in manifest["image_path"]]
    source_paths = manifest["source_image_path"].drop_duplicates().tolist()
    source_rows = [{"path": path} for path in source_paths]
    generated_probabilities, generated_features = classifier_outputs(
        classifier, generated_rows, transform, args.batch_size, args.num_workers, device
    )
    _, source_features = classifier_outputs(
        classifier, source_rows, transform, args.batch_size, args.num_workers, device
    )
    source_feature_map = dict(zip(source_paths, source_features))

    segmenter = load_segmenter(Path(args.segmenter_checkpoint), device)
    results: list[dict[str, Any]] = []
    image_size = int(config["common"]["image_size"])
    for index, row in manifest.iterrows():
        generated_path = Path(row["image_path"])
        source_path = Path(row["source_image_path"])
        mask_path = Path(row["mask_path"])
        source = fit_image(source_path, image_size, "RGB")
        generated = fit_image(generated_path, image_size, "RGB")
        arm = config["arms"][str(row["generator_arm"])]
        conditioned_mask = prepare_mask(
            mask_path,
            image_size,
            int(arm.get("mask_dilation_pixels", 0)),
            float(arm.get("mask_blur_radius", 0)),
        )
        reference_mask = np.asarray(
            prepare_mask(mask_path, image_size, 0, 0), dtype=np.uint8
        ) >= 128
        regenerated_mask = predict_mask(segmenter, generated, 384, device) >= 0.5
        mask_scores = overlap_metrics(reference_mask, regenerated_mask)
        label_index = class_to_idx[str(row["label"])]
        predicted_index = int(generated_probabilities[index].argmax())
        result = dict(row)
        result.update(pixel_metrics(source, generated, conditioned_mask))
        result.update(
            {
                "output_hash_valid": sha256(generated_path) == str(row["output_sha256"]),
                "predicted_label": idx_to_class[predicted_index],
                "label_agreement": predicted_index == label_index,
                "true_label_probability": float(generated_probabilities[index, label_index]),
                "source_cosine_similarity": float(
                    generated_features[index] @ source_feature_map[str(source_path)]
                ),
                "regenerated_mask_iou": mask_scores["iou"],
                "regenerated_mask_dice": mask_scores["dice"],
                "test_evaluated": False,
            }
        )
        results.append(result)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_root / "qualification_per_image.csv", index=False)
    summaries = {
        arm: summarize_arm(group, config["promotion_gates"])
        for arm, group in result_frame.groupby("generator_arm")
    }
    report = {
        "protocol": config["protocol"],
        "test_evaluated": False,
        "classifier_run_dir": str(classifier_run),
        "segmenter_checkpoint": args.segmenter_checkpoint,
        "arms": summaries,
        "p1_allowed": bool(
            summaries.get("generic_sd15_mask_inpaint", {}).get("technical_gate_passed", False)
        ),
        "visual_audit_required": True,
    }
    (output_root / "qualification_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame.from_dict(summaries, orient="index").to_csv(
        output_root / "qualification_by_arm.csv"
    )
    for arm, group in result_frame.groupby("generator_arm"):
        make_contact_sheet(group, output_root / f"contact_sheet_{arm}.jpg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
