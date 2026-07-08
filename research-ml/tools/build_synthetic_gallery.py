from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build browsable galleries for raw and selected synthetic images.")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--selection", action="append", default=[], help="name:path pairs, e.g. strict:splits/stage2/selected.csv")
    parser.add_argument("--out-dir", default="reports/synthetic_galleries")
    parser.add_argument("--gallery-name", default="")
    parser.add_argument("--max-raw-per-class", type=int, default=0, help="0 means include every raw synthetic image.")
    parser.add_argument("--thumb-size", type=int, default=288)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def safe_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in value)[:180]


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_pair_image(row: dict[str, str], data_root: Path, out_path: Path, size: int) -> bool:
    source_path = resolve_path(data_root, row.get("source_image_path", ""))
    synthetic_path = resolve_path(data_root, row.get("image_path", ""))
    if not source_path.exists() or not synthetic_path.exists():
        return False

    label_h = 64
    gutter = 10
    canvas = Image.new("RGB", (size * 2 + gutter, size + label_h), "white")
    canvas.paste(fit_image(source_path, size), (0, label_h))
    canvas.paste(fit_image(synthetic_path, size), (size + gutter, label_h))

    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    small = load_font(12)
    label = row.get("label", "")
    source_id = row.get("source_image_id", "")
    image_id = row.get("image_id", "")
    same = row.get("same_class_distance", "")
    margin = row.get("feature_margin", "")
    metrics = f"same={float(same):.3f} margin={float(margin):.3f}" if same and margin else "raw synthetic"
    draw.text((8, 6), f"{label} | real: {source_id}", fill=(20, 20, 20), font=font)
    draw.text((size + gutter + 8, 6), f"synthetic: {image_id}", fill=(20, 20, 20), font=font)
    draw.text((8, 34), metrics, fill=(85, 85, 85), font=small)
    draw.line((size + gutter // 2, label_h, size + gutter // 2, size + label_h), fill=(210, 210, 210), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return True


def sample_raw_rows(rows: list[dict[str, str]], max_per_class: int) -> list[dict[str, str]]:
    if max_per_class <= 0:
        return rows
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if len(by_class[row["label"]]) < max_per_class:
            by_class[row["label"]].append(row)
    sampled: list[dict[str, str]] = []
    for label in sorted(by_class):
        sampled.extend(by_class[label])
    return sampled


def build_gallery(name: str, rows: list[dict[str, str]], data_root: Path, out_dir: Path, thumb_size: int) -> dict[str, Any]:
    gallery_dir = out_dir / safe_name(name)
    pairs_dir = gallery_dir / "pairs"
    rendered_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for index, row in enumerate(rows, start=1):
        label = row.get("label", "unknown")
        stem = safe_name(f"{index:04d}_{label}_{row.get('source_image_id', '')}_{row.get('image_id', '')}.jpg")
        pair_path = pairs_dir / safe_name(label) / stem
        if not make_pair_image(row, data_root, pair_path, thumb_size):
            continue
        rel_pair = pair_path.relative_to(gallery_dir).as_posix()
        rendered = dict(row)
        rendered["pair_image"] = rel_pair
        rendered_rows.append(rendered)
        same = row.get("same_class_distance", "")
        margin = row.get("feature_margin", "")
        metric_text = ""
        if same and margin:
            metric_text = f"<div>same={float(same):.4f} margin={float(margin):.4f}</div>"
        cards.append(
            "<article>"
            f"<a href=\"{html.escape(rel_pair)}\"><img loading=\"lazy\" src=\"{html.escape(rel_pair)}\" alt=\"pair\"></a>"
            f"<h3>{html.escape(label)} · {html.escape(row.get('source_image_id', ''))}</h3>"
            f"<div>{html.escape(row.get('image_id', ''))}</div>"
            f"{metric_text}"
            "</article>"
        )

    write_rows(gallery_dir / "gallery_manifest.csv", rendered_rows)
    counts = Counter(row.get("label", "unknown") for row in rendered_rows)
    (gallery_dir / "summary.json").write_text(
        json.dumps({"name": name, "total": len(rendered_rows), "by_class": dict(counts)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (gallery_dir / "index.html").write_text(render_html(name, cards, counts), encoding="utf-8")
    return {"name": name, "total": len(rendered_rows), "by_class": dict(counts), "path": gallery_dir.name}


def render_html(title: str, cards: list[str], counts: Counter[str]) -> str:
    counts_text = ", ".join(f"{html.escape(k)}={v}" for k, v in sorted(counts.items()))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 20px; color: #17202a; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .sub {{ color: #586069; margin: 6px 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }}
    article {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 10px; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 6px; background: #f6f8fa; }}
    h1 {{ font-size: 24px; margin: 0; }}
    h3 {{ font-size: 15px; margin: 8px 0 4px; }}
    article div {{ color: #586069; font-size: 13px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="sub">Всего: {len(cards)}. По классам: {counts_text}</div>
  <p><a href="../index.html">Назад к сводке</a> · <a href="gallery_manifest.csv">gallery_manifest.csv</a> · <a href="summary.json">summary.json</a></p>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""


def parse_selection(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise ValueError(f"Selection must have name:path form: {value}")
    name, path = value.split(":", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    synthetic_csv = resolve_path(data_root, args.synthetic_csv)
    gallery_name = args.gallery_name or synthetic_csv.parent.name
    out_root = resolve_path(data_root, args.out_dir) / safe_name(gallery_name)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_rows = read_rows(synthetic_csv)
    summaries = [
        build_gallery("before_selection_raw", sample_raw_rows(raw_rows, args.max_raw_per_class), data_root, out_root, args.thumb_size)
    ]
    shutil.copy2(synthetic_csv, out_root / "synthetic_manifest.csv")

    for selection in args.selection:
        name, path = parse_selection(selection)
        selected_csv = resolve_path(data_root, str(path))
        rows = read_rows(selected_csv)
        summaries.append(build_gallery(f"after_selection_{name}", rows, data_root, out_root, args.thumb_size))
        shutil.copy2(selected_csv, out_root / f"selected_{safe_name(name)}.csv")

    links = "\n".join(
        f"<li><a href=\"{html.escape(item['path'])}/index.html\">{html.escape(item['name'])}</a>: "
        f"{item['total']} ({html.escape(json.dumps(item['by_class'], ensure_ascii=False))})</li>"
        for item in summaries
    )
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "index.html").write_text(
        f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(gallery_name)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #17202a; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    li {{ margin: 8px 0; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(gallery_name)}</h1>
  <p>Пары изображений: слева реальное исходное изображение HAM10000, справа синтетическое изображение.</p>
  <ul>{links}</ul>
  <p>CSV: <a href="synthetic_manifest.csv">synthetic_manifest.csv</a> · <a href="summary.json">summary.json</a></p>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(out_root)


if __name__ == "__main__":
    main()
