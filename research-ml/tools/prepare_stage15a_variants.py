from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


TARGET_CLASSES = ("mel", "akiec", "bkl")
EXPECTED_PER_CLASS = 30
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare lesion-matched Stage 15A crop, VAE, and img2img arms."
    )
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/ham10000"
    )
    parser.add_argument(
        "--selected-csv",
        default="splits/stage13_bkl_expansion/selected_synthetic_coverage_targeted.csv",
    )
    parser.add_argument(
        "--base-train-csv", default="splits/stage8/train_real.csv"
    )
    parser.add_argument(
        "--generation-manifest",
        action="append",
        default=[
            "synthetic/ham10000_stage13_low_strength_img2img/synthetic_manifest.csv",
            "synthetic/ham10000_stage13_bkl_expansion_img2img/synthetic_manifest.csv",
        ],
    )
    parser.add_argument("--out-split-dir", default="splits/stage15a")
    parser.add_argument("--out-image-dir", default="synthetic/stage15a")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selected(selected: pd.DataFrame) -> dict[str, Any]:
    required = {
        "image_id",
        "label",
        "source_image_id",
        "source_image_path",
        "source_group_id",
    }
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Selected CSV is missing columns: {sorted(missing)}")
    counts = selected["label"].astype(str).value_counts().to_dict()
    expected = {label: EXPECTED_PER_CLASS for label in TARGET_CLASSES}
    if counts != expected:
        raise ValueError(f"Expected balanced 30-per-class selection, found {counts}")
    if len(selected) != 90:
        raise ValueError(f"Expected 90 selected rows, found {len(selected)}")
    if selected["source_image_id"].astype(str).nunique() != 90:
        raise ValueError("Stage 15A requires 90 unique source image IDs")
    if selected["source_group_id"].astype(str).nunique() != 90:
        raise ValueError("Stage 15A requires 90 unique source lesion groups")
    return {
        "selected_rows": 90,
        "selected_by_class": expected,
        "unique_source_images": 90,
        "unique_source_groups": 90,
    }


def choose_strength05_rows(
    selected: pd.DataFrame, generation: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "image_path",
        "image_id",
        "label",
        "source_image_id",
        "source_image_path",
        "source_group_id",
        "strength",
    }
    missing = required - set(generation.columns)
    if missing:
        raise ValueError(f"Generation manifest is missing columns: {sorted(missing)}")
    wanted = set(selected["source_image_id"].astype(str))
    candidates = generation[
        generation["source_image_id"].astype(str).isin(wanted)
        & np.isclose(generation["strength"].astype(float), 0.05)
    ].copy()
    counts = candidates["source_image_id"].astype(str).value_counts()
    bad = counts[counts != 1].to_dict()
    missing_sources = sorted(wanted - set(counts.index))
    if bad or missing_sources:
        raise ValueError(
            "Cannot create exact strength=0.05 arm: "
            f"non_unique={bad}, missing={missing_sources}"
        )
    candidates = candidates.set_index(
        candidates["source_image_id"].astype(str), drop=False
    ).loc[selected["source_image_id"].astype(str)].reset_index(drop=True)
    if not (
        candidates["source_group_id"].astype(str).to_numpy()
        == selected["source_group_id"].astype(str).to_numpy()
    ).all():
        raise ValueError("Strength=0.05 source-group mapping differs from selection")
    return candidates


def center_crop(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def variant_row(
    source: pd.Series,
    image_path: str,
    image_id: str,
    variant: str,
) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "label": str(source["label"]),
        "is_synthetic": 1,
        "image_id": image_id,
        "group_id": f"stage15a_{variant}_{image_id}",
        "source": f"stage15a_{variant}",
        "source_image_path": str(source["source_image_path"]),
        "source_image_id": str(source["source_image_id"]),
        "source_group_id": str(source["source_group_id"]),
        "stage15a_variant": variant,
    }


def save_crop_variants(
    data_root: Path,
    selected: pd.DataFrame,
    output_root: Path,
    size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in selected.to_dict("records"):
        label = str(source["label"])
        source_id = str(source["source_image_id"])
        relative = (
            output_root / "offline_center_crop" / label / f"{source_id}.png"
        )
        absolute = data_root / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if not absolute.is_file():
            center_crop(data_root / str(source["source_image_path"]), size).save(
                absolute, format="PNG", optimize=True
            )
        rows.append(
            variant_row(
                pd.Series(source),
                relative.as_posix(),
                f"stage15a_crop_{source_id}",
                "offline_center_crop",
            )
        )
    return pd.DataFrame(rows)


def save_vae_variants(
    data_root: Path,
    crop_rows: pd.DataFrame,
    output_root: Path,
    batch_size: int,
) -> pd.DataFrame:
    import torch
    from diffusers import AutoencoderKL

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 15A VAE preparation requires CUDA")
    vae = AutoencoderKL.from_pretrained(
        SD15_MODEL_ID,
        subfolder="vae",
        revision=SD15_REVISION,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to("cuda")
    vae.eval()
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(crop_rows), batch_size):
        batch = crop_rows.iloc[offset : offset + batch_size]
        arrays = []
        for path in batch["image_path"].astype(str):
            with Image.open(data_root / path) as image:
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.float32))
        tensor = (
            torch.from_numpy(np.stack(arrays))
            .permute(0, 3, 1, 2)
            .to("cuda", dtype=torch.float32)
            / 127.5
            - 1.0
        )
        with torch.inference_mode():
            latents = vae.encode(tensor).latent_dist.mode()
            decoded = vae.decode(latents).sample.clamp(-1, 1)
        images = (
            ((decoded + 1.0) * 127.5)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for (_, crop), array in zip(batch.iterrows(), images):
            label = str(crop["label"])
            source_id = str(crop["source_image_id"])
            relative = output_root / "sd15_vae_roundtrip" / label / f"{source_id}.png"
            absolute = data_root / relative
            absolute.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array, mode="RGB").save(
                absolute, format="PNG", optimize=True
            )
            source = pd.Series(
                {
                    "label": label,
                    "source_image_path": crop["source_image_path"],
                    "source_image_id": source_id,
                    "source_group_id": crop["source_group_id"],
                }
            )
            rows.append(
                variant_row(
                    source,
                    relative.as_posix(),
                    f"stage15a_vae_{source_id}",
                    "sd15_vae_roundtrip",
                )
            )
    del vae
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def normalize_img2img_rows(rows: pd.DataFrame) -> pd.DataFrame:
    normalized = rows.copy()
    normalized["is_synthetic"] = 1
    normalized["stage15a_variant"] = "sd15_img2img_strength05"
    return normalized


def write_arm(
    base: pd.DataFrame,
    additions: pd.DataFrame,
    split_dir: Path,
    name: str,
) -> None:
    additions.to_csv(split_dir / f"{name}_rows.csv", index=False)
    augmented = pd.concat([base, additions], ignore_index=True, sort=False)
    if augmented["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} produced duplicate image IDs")
    augmented.to_csv(split_dir / f"train_{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    selected_path = resolve(data_root, args.selected_csv)
    base_path = resolve(data_root, args.base_train_csv)
    manifest_paths = [resolve(data_root, value) for value in args.generation_manifest]
    selected = pd.read_csv(selected_path)
    base = pd.read_csv(base_path)
    generation = pd.concat(
        [pd.read_csv(path) for path in manifest_paths], ignore_index=True
    )
    selection = validate_selected(selected)
    strength05 = choose_strength05_rows(selected, generation)
    inputs = {
        "selected_csv": str(selected_path),
        "selected_csv_sha256": sha256(selected_path),
        "base_train_csv": str(base_path),
        "base_train_csv_sha256": sha256(base_path),
        "generation_manifests": [
            {"path": str(path), "sha256": sha256(path)} for path in manifest_paths
        ],
    }
    plan = {
        "protocol": "stage15a_causal_generator_decomposition",
        "locked_test_used": False,
        "selection": selection,
        "strength05_rows": int(len(strength05)),
        "sd15_model_id": SD15_MODEL_ID,
        "sd15_revision": SD15_REVISION,
        "image_size": int(args.image_size),
        "inputs": inputs,
    }
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return

    split_dir = resolve(data_root, args.out_split_dir)
    output_root = Path(args.out_image_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    crop_rows = save_crop_variants(
        data_root, selected, output_root, int(args.image_size)
    )
    vae_rows = save_vae_variants(
        data_root, crop_rows, output_root, int(args.batch_size)
    )
    img2img_rows = normalize_img2img_rows(strength05)
    write_arm(base, crop_rows, split_dir, "offline_crop")
    write_arm(base, vae_rows, split_dir, "vae_roundtrip")
    write_arm(base, img2img_rows, split_dir, "img2img_strength05")
    pd.concat(
        [crop_rows, vae_rows, img2img_rows],
        ignore_index=True,
        sort=False,
    ).to_csv(split_dir / "visual_audit_rows.csv", index=False)

    plan["artifacts"] = {
        "offline_crop_rows": int(len(crop_rows)),
        "vae_roundtrip_rows": int(len(vae_rows)),
        "img2img_strength05_rows": int(len(img2img_rows)),
        "visual_audit_rows": int(len(crop_rows) + len(vae_rows) + len(img2img_rows)),
        "split_dir": str(split_dir),
        "image_root": str(data_root / output_root),
    }
    (split_dir / "preparation_manifest.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
