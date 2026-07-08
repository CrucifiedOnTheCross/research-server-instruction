from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image-conditioned synthetic HAM10000 samples with diffusion img2img.")
    parser.add_argument("--config", default="configs/stage2_generation.yaml")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--train-csv", default="splits/train.csv")
    parser.add_argument("--override-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["generation"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "is_synthetic",
        "image_id",
        "group_id",
        "source",
        "source_image_path",
        "source_image_id",
        "prompt",
        "negative_prompt",
        "model_id",
        "seed",
        "strength",
        "guidance_scale",
        "inference_steps",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pipeline(model_id: str):
    from diffusers import AutoPipelineForImage2Image

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def open_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def make_negative_prompt(label: str, config: dict[str, Any]) -> str:
    confusing = config.get("confusing_classes", {}).get(label, [])
    prompts = config.get("prompt_templates", {})
    confusing_terms = []
    for cls in confusing:
        text = prompts.get(cls, cls)
        confusing_terms.append(text.split(",")[0])
    parts = [config.get("negative_prompt_base", "")]
    if confusing_terms:
        parts.append("confusing class appearance: " + "; ".join(confusing_terms))
    return ", ".join(part for part in parts if part)


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    if args.override_name:
        cfg["name"] = args.override_name
        cfg["output_dir"] = f"synthetic/{args.override_name}"

    data_root = Path(args.data_root)
    train_rows = read_rows(data_root / args.train_csv)
    target_classes = set(cfg["target_classes"])
    rng = random.Random(int(cfg["seed"]))

    by_class: dict[str, list[dict[str, str]]] = {label: [] for label in target_classes}
    for row in train_rows:
        if int(row.get("is_synthetic", 0)) == 0 and row["label"] in target_classes:
            by_class[row["label"]].append(row)
    for rows in by_class.values():
        rng.shuffle(rows)

    selected: list[dict[str, str]] = []
    for label in sorted(by_class):
        rows = by_class[label]
        max_real = int(cfg.get("max_real_per_class", 0))
        if max_real > 0:
            rows = rows[:max_real]
        selected.extend(rows)

    out_dir = data_root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "synthetic_manifest.csv"
    metadata_path = out_dir / "generation_config.resolved.yaml"
    metadata_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    plan_count = len(selected) * int(cfg["num_images_per_real"])
    print(json.dumps({"selected_real": len(selected), "planned_synthetic": plan_count, "output": str(out_dir)}, indent=2))
    if args.dry_run:
        return

    pipe = load_pipeline(cfg["model_id"])
    synthetic_rows: list[dict[str, Any]] = []
    base_seed = int(cfg["seed"])
    image_size = int(cfg["image_size"])

    for row in tqdm(selected, desc="generate"):
        label = row["label"]
        source_path = data_root / row["image_path"]
        prompt = cfg["prompt_templates"][label]
        negative_prompt = make_negative_prompt(label, cfg)
        init_image = open_image(source_path, image_size)
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        source_id = row.get("image_id") or source_path.stem
        for index in range(int(cfg["num_images_per_real"])):
            seed = base_seed + len(synthetic_rows)
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=float(cfg["strength"]),
                guidance_scale=float(cfg["guidance_scale"]),
                num_inference_steps=int(cfg["inference_steps"]),
                generator=generator,
            )
            image = result.images[0]
            out_name = f"{label}_{source_id}_{index:02d}_{seed}.jpg"
            out_path = label_dir / out_name
            image.save(out_path, quality=95)
            synthetic_rows.append(
                {
                    "image_path": out_path.relative_to(data_root).as_posix(),
                    "label": label,
                    "is_synthetic": 1,
                    "image_id": Path(out_name).stem,
                    "group_id": f"synthetic_{Path(out_name).stem}",
                    "source": cfg["name"],
                    "source_image_path": row["image_path"],
                    "source_image_id": source_id,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model_id": cfg["model_id"],
                    "seed": seed,
                    "strength": cfg["strength"],
                    "guidance_scale": cfg["guidance_scale"],
                    "inference_steps": cfg["inference_steps"],
                }
            )
            if len(synthetic_rows) % 25 == 0:
                write_rows(manifest_path, synthetic_rows)

    write_rows(manifest_path, synthetic_rows)
    print(json.dumps({"synthetic": len(synthetic_rows), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

