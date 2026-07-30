from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stratified Stage 16G pseudo-mask contact sheets."
    )
    parser.add_argument(
        "--csv",
        default=(
            "outputs/reports/stage16g_segmentation_audit/"
            "pseudo_mask_manifest.csv"
        ),
    )
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/isic2019"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports/stage16g_segmentation_audit/contact_sheets",
    )
    return parser.parse_args()


def choose_rows(frame: pd.DataFrame) -> pd.DataFrame:
    passed = frame[frame["pseudo_mask_passed"] == 1]
    failed = frame[frame["pseudo_mask_passed"] == 0]
    parts = [
        passed.nsmallest(2, "foreground_confidence").assign(audit_role="low_conf"),
        passed.nlargest(2, "border_foreground_fraction").assign(
            audit_role="border"
        ),
        failed.nlargest(2, "component_count").assign(audit_role="failed"),
    ]
    return pd.concat(parts, ignore_index=True).drop_duplicates("image_id")


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    binary = mask.point(lambda value: 255 if value >= 128 else 0)
    red = Image.new("RGB", image.size, (255, 20, 20))
    composite = Image.composite(red, image, binary.point(lambda value: 85 if value else 0))
    edge = binary.filter(ImageFilter.MaxFilter(7))
    inner = binary.filter(ImageFilter.MinFilter(7))
    boundary = Image.fromarray(
        (
            np.asarray(edge, dtype="int16")
            - np.asarray(inner, dtype="int16")
        )
        .clip(0, 255)
        .astype("uint8")
    )
    return Image.composite(red, composite, boundary)


def render_tile(
    row: pd.Series, root: Path, tile_size: int = 288
) -> Image.Image:
    with Image.open(root / str(row["image_path"])) as handle:
        image = handle.convert("RGB")
    with Image.open(root / str(row["pseudo_mask_path"])) as handle:
        mask = handle.convert("L")
    image.thumbnail((tile_size, tile_size - 42), Image.Resampling.LANCZOS)
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (tile_size, tile_size), "white")
    rendered = overlay_mask(image, mask)
    left = (tile_size - rendered.width) // 2
    top = (tile_size - 42 - rendered.height) // 2
    canvas.paste(rendered, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, tile_size - 42, tile_size, tile_size), fill="black")
    text = (
        f"{row['audit_role']} {row['image_id']} "
        f"c={row['foreground_confidence']:.2f} "
        f"k={int(row['component_count'])}"
    )
    draw.text((5, tile_size - 34), text, fill="white", font=ImageFont.load_default())
    return canvas


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    root = Path(args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for label, class_rows in frame.groupby("label"):
        selected = choose_rows(class_rows)
        tiles = [render_tile(row, root) for _, row in selected.iterrows()]
        sheet = Image.new("RGB", (len(tiles) * 288, 288), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (index * 288, 0))
        sheet.save(output / f"{label}_mask_audit.png")
    print(f"Saved contact sheets to {output}")


if __name__ == "__main__":
    main()
