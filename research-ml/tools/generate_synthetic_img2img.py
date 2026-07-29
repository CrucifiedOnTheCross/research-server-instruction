from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageOps
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
        "source_group_id",
        "prompt",
        "negative_prompt",
        "model_id",
        "seed",
        "strength",
        "guidance_scale",
        "inference_steps",
        "preprocess_mode",
        "image_format",
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


def open_image(path: Path, size: int, preprocess_mode: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if preprocess_mode == "center_crop":
        return ImageOps.fit(
            image,
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    if preprocess_mode == "stretch":
        return image.resize((size, size), Image.Resampling.LANCZOS)
    raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}")


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


def task_image_stem(task: dict[str, Any]) -> str:
    strength_tag = f"{float(task['strength']):.2f}".replace(".", "p")
    return (
        f"{task['label']}_{task['source_id']}_s{strength_tag}_"
        f"{int(task['index']):02d}_{int(task['seed'])}"
    )


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    if args.override_name:
        cfg["name"] = args.override_name
        cfg["output_dir"] = f"synthetic/{args.override_name}"

    data_root = Path(args.data_root)
    train_rows = read_rows(data_root / args.train_csv)
    target_classes = sorted(set(cfg["target_classes"]))
    rng = random.Random(int(cfg["seed"]))

    by_class: dict[str, list[dict[str, str]]] = {label: [] for label in target_classes}
    for row in train_rows:
        if int(row.get("is_synthetic", 0)) == 0 and row["label"] in target_classes:
            by_class[row["label"]].append(row)
    for label in target_classes:
        rng.shuffle(by_class[label])

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
    if manifest_path.exists() and metadata_path.exists():
        previous_cfg = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        if previous_cfg != cfg:
            raise ValueError(
                "Refusing to resume generation with a different resolved config"
            )
    metadata_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    strengths = [float(value) for value in cfg.get("strengths", [cfg["strength"]])]
    plan_count = len(selected) * int(cfg["num_images_per_real"]) * len(strengths)
    print(json.dumps({"selected_real": len(selected), "planned_synthetic": plan_count, "output": str(out_dir)}, indent=2))
    if args.dry_run:
        return

    pipe = load_pipeline(cfg["model_id"])
    tasks: list[dict[str, Any]] = []
    base_seed = int(cfg["seed"])
    image_size = int(cfg["image_size"])
    preprocess_mode = str(cfg.get("preprocess_mode", "center_crop"))
    image_format = str(cfg.get("image_format", "png")).lower()
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("generation.image_format must be png, jpg, or jpeg")
    extension = "jpg" if image_format in {"jpg", "jpeg"} else "png"
    for row in selected:
        label = row["label"]
        prompt = cfg["prompt_templates"][label]
        negative_prompt = make_negative_prompt(label, cfg)
        source_id = row.get("image_id") or Path(row["image_path"]).stem
        for strength in strengths:
            for index in range(int(cfg["num_images_per_real"])):
                seed = base_seed + len(tasks)
                tasks.append(
                    {
                        "row": row,
                        "label": label,
                        "source_id": source_id,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "strength": strength,
                        "index": index,
                        "seed": seed,
                    }
                )

    synthetic_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        synthetic_rows = read_rows(manifest_path)
    existing_by_id = {
        str(row["image_id"]): row
        for row in synthetic_rows
        if (data_root / str(row["image_path"])).is_file()
    }
    if len(existing_by_id) != len(synthetic_rows):
        synthetic_rows = list(existing_by_id.values())
    batch_size = max(1, int(cfg.get("batch_size", 1)))
    task_batches: list[list[dict[str, Any]]] = []
    for strength in strengths:
        strength_tasks = [
            task
            for task in tasks
            if float(task["strength"]) == strength
            and task_image_stem(task) not in existing_by_id
        ]
        strength_tasks.sort(key=lambda task: (str(task["label"]), str(task["source_id"])))
        task_batches.extend(
            strength_tasks[offset : offset + batch_size]
            for offset in range(0, len(strength_tasks), batch_size)
        )
    for batch in tqdm(task_batches, desc="generate"):
        init_images = [
            open_image(data_root / task["row"]["image_path"], image_size, preprocess_mode)
            for task in batch
        ]
        generators = [
            torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(task["seed"]))
            for task in batch
        ]
        result = pipe(
            prompt=[str(task["prompt"]) for task in batch],
            negative_prompt=[str(task["negative_prompt"]) for task in batch],
            image=init_images,
            strength=float(batch[0]["strength"]),
            guidance_scale=float(cfg["guidance_scale"]),
            num_inference_steps=int(cfg["inference_steps"]),
            generator=generators,
        )
        for task, image in zip(batch, result.images):
            row = task["row"]
            label = str(task["label"])
            source_id = str(task["source_id"])
            seed = int(task["seed"])
            label_dir = out_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{task_image_stem(task)}.{extension}"
            out_path = label_dir / out_name
            if extension == "png":
                image.save(out_path, format="PNG", optimize=True)
            else:
                image.save(out_path, format="JPEG", quality=95, subsampling=0)
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
                    "source_group_id": row.get("group_id", source_id),
                    "prompt": task["prompt"],
                    "negative_prompt": task["negative_prompt"],
                    "model_id": cfg["model_id"],
                    "seed": seed,
                    "strength": task["strength"],
                    "guidance_scale": cfg["guidance_scale"],
                    "inference_steps": cfg["inference_steps"],
                    "preprocess_mode": preprocess_mode,
                    "image_format": image_format,
                }
            )
            if len(synthetic_rows) % 25 == 0:
                write_rows(manifest_path, synthetic_rows)

    write_rows(manifest_path, synthetic_rows)
    print(json.dumps({"synthetic": len(synthetic_rows), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
