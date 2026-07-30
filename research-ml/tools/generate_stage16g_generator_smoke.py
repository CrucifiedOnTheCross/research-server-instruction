from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Stage 16G generator smoke arms.")
    parser.add_argument("--config", default="configs/stage16g_generator_smoke.yaml")
    parser.add_argument("--arm", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_image(path: Path, size: int, mode: str) -> Image.Image:
    image = Image.open(path).convert(mode)
    return ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS if mode == "RGB" else Image.Resampling.NEAREST,
    )


def prepare_mask(path: Path, size: int, dilation: int, blur_radius: float) -> Image.Image:
    mask = fit_image(path, size, "L").point(lambda value: 255 if value >= 128 else 0)
    if dilation > 0:
        kernel = dilation * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    return mask


def load_pipeline(arm: dict[str, Any]) -> tuple[Any, str]:
    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting
    from huggingface_hub import model_info

    pipeline_type = (
        AutoPipelineForInpainting if arm["kind"] == "inpaint" else AutoPipelineForImage2Image
    )
    revision = str(model_info(arm["model_id"]).sha)
    pipe = pipeline_type.from_pretrained(
        arm["model_id"],
        revision=revision,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, revision


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import torch

    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.arm not in config["arms"]:
        raise ValueError(f"Unknown arm: {args.arm}")
    arm = config["arms"][args.arm]
    data_root = Path(config["data"]["root"])
    output_root = Path(config["data"]["output_root"])
    anchors = pd.read_csv(output_root / "anchors.csv")
    arm_root = output_root / args.arm
    image_root = arm_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_path = arm_root / "manifest.csv"
    existing = pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
    existing_ids = {str(row["image_id"]) for row in existing}
    pipe, revision = load_pipeline(arm)

    environment = {
        "protocol": config["protocol"],
        "arm": args.arm,
        "test_evaluated": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_id": arm["model_id"],
        "model_revision": revision,
        "config_sha256": sha256(config_path),
    }
    (arm_root / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    common = config["common"]
    size = int(common["image_size"])
    candidates = int(config["selection"]["candidates_per_anchor"])
    rows = list(existing)
    tasks = [
        (anchor, candidate_index)
        for _, anchor in anchors.iterrows()
        for candidate_index in range(candidates)
    ]
    for anchor, candidate_index in tqdm(tasks, desc=args.arm):
        seed = int(config["selection"]["seed"]) + int(anchor["anchor_index"]) * 1000
        seed += list(config["selection"]["target_classes"]).index(anchor["label"]) * 10000
        seed += candidate_index
        image_id = f"{args.arm}_{anchor['image_id']}_{candidate_index:02d}_{seed}"
        if image_id in existing_ids:
            continue
        source_path = data_root / str(anchor["image_path"])
        mask_path = data_root / str(anchor["pseudo_mask_path"])
        source = fit_image(source_path, size, "RGB")
        prompt = common["prompt_templates"][str(anchor["label"])]
        kwargs = {
            "prompt": prompt,
            "negative_prompt": common["negative_prompt"],
            "image": source,
            "strength": float(arm["strength"]),
            "guidance_scale": float(common["guidance_scale"]),
            "num_inference_steps": int(common["inference_steps"]),
            "generator": torch.Generator(device="cuda").manual_seed(seed),
        }
        if arm["kind"] == "inpaint":
            mask = prepare_mask(
                mask_path,
                size,
                int(arm.get("mask_dilation_pixels", 0)),
                float(arm.get("mask_blur_radius", 0.0)),
            )
            kwargs["mask_image"] = mask
        generated = pipe(**kwargs).images[0]
        output_path = image_root / str(anchor["label"]) / f"{image_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated.save(output_path, format="PNG", optimize=True)
        rows.append(
            {
                "image_path": output_path.as_posix(),
                "label": anchor["label"],
                "image_id": image_id,
                "group_id": f"synthetic_{image_id}",
                "source_image_id": anchor["image_id"],
                "source_group_id": anchor["group_id"],
                "source_image_path": source_path.as_posix(),
                "source_sha256": anchor["source_sha256_verified"],
                "mask_path": mask_path.as_posix(),
                "mask_sha256": anchor["mask_sha256"],
                "generator_arm": args.arm,
                "generator_kind": arm["kind"],
                "model_id": arm["model_id"],
                "model_revision": revision,
                "seed": seed,
                "strength": arm["strength"],
                "guidance_scale": common["guidance_scale"],
                "inference_steps": common["inference_steps"],
                "prompt": prompt,
                "negative_prompt": common["negative_prompt"],
                "output_sha256": sha256(output_path),
                "test_evaluated": False,
            }
        )
        write_manifest(manifest_path, rows)
    summary = {
        **environment,
        "completed_images": len(rows),
        "expected_images": len(tasks),
        "complete": len(rows) == len(tasks),
    }
    (arm_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
