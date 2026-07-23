from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit obvious pixel artifacts before synthetic feature analysis.")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--real-csv", default="splits/train.csv")
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--border-fraction", type=float, default=0.125)
    parser.add_argument("--black-threshold", type=float, default=0.05)
    parser.add_argument("--max-median-black-border-share", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def image_features(path: Path, border_fraction: float, black_threshold: float) -> dict[str, float | int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        array = np.asarray(rgb.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    border_height = max(1, int(round(array.shape[0] * border_fraction)))
    border = np.concatenate((array[:border_height], array[-border_height:]), axis=0)
    center = array[border_height:-border_height] if border_height * 2 < array.shape[0] else array
    return {
        "width": width,
        "height": height,
        "border_mean": float(border.mean()),
        "border_std": float(border.std()),
        "center_mean": float(center.mean()),
        "center_std": float(center.std()),
        "black_border_share": float((border < black_threshold).mean()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = Counter(f"{row['width']}x{row['height']}" for row in rows)
    result: dict[str, Any] = {
        "count": len(rows),
        "dimensions": dict(dimensions.most_common()),
    }
    for key in ("border_mean", "border_std", "center_mean", "center_std", "black_border_share"):
        values = np.asarray([float(row[key]) for row in rows])
        result[key] = {
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return result


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_rows = [row for row in read_rows(resolve(root, args.real_csv)) if int(row.get("is_synthetic", 0)) == 0]
    synthetic_rows = read_rows(resolve(root, args.synthetic_csv))
    rng = np.random.default_rng(args.seed)
    if len(real_rows) > args.sample_size:
        real_rows = [real_rows[index] for index in rng.choice(len(real_rows), args.sample_size, replace=False)]
    if len(synthetic_rows) > args.sample_size:
        synthetic_rows = [
            synthetic_rows[index] for index in rng.choice(len(synthetic_rows), args.sample_size, replace=False)
        ]

    annotated: list[dict[str, Any]] = []
    for domain, rows in (("real", real_rows), ("synthetic", synthetic_rows)):
        for row in rows:
            path = resolve(root, row["image_path"])
            annotated.append(
                {
                    "domain": domain,
                    "image_path": row["image_path"],
                    **image_features(path, args.border_fraction, args.black_threshold),
                }
            )

    real_features = [row for row in annotated if row["domain"] == "real"]
    synthetic_features = [row for row in annotated if row["domain"] == "synthetic"]
    labels = np.asarray([0] * len(real_features) + [1] * len(synthetic_features))
    separability: dict[str, float] = {}
    for key in ("border_mean", "border_std", "center_mean", "center_std", "black_border_share"):
        values = np.asarray([float(row[key]) for row in real_features + synthetic_features])
        auc = float(roc_auc_score(labels, values))
        separability[key] = max(auc, 1.0 - auc)

    synthetic_black_median = float(np.median([row["black_border_share"] for row in synthetic_features]))
    passed = synthetic_black_median <= float(args.max_median_black_border_share)
    report = {
        "real_csv": str(resolve(root, args.real_csv)),
        "synthetic_csv": str(resolve(root, args.synthetic_csv)),
        "settings": {
            "border_fraction": args.border_fraction,
            "black_threshold": args.black_threshold,
            "max_median_black_border_share": args.max_median_black_border_share,
        },
        "real": summarize(real_features),
        "synthetic": summarize(synthetic_features),
        "single_feature_real_synthetic_auroc": separability,
        "black_border_check_passed": passed,
    }
    fieldnames = list(annotated[0]) if annotated else []
    with (out_dir / "artifact_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)
    (out_dir / "artifact_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(
            f"Synthetic black-border audit failed: median={synthetic_black_median:.4f}, "
            f"limit={args.max_median_black_border_share:.4f}"
        )


if __name__ == "__main__":
    main()
