from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train CSVs for class-specific synthetic ablations.")
    parser.add_argument("--real-train-csv", required=True)
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--classes", required=True, help="Comma-separated synthetic classes to keep.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--synthetic-col", default="is_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    real = pd.read_csv(args.real_train_csv)
    synthetic = pd.read_csv(args.synthetic_csv)
    if args.synthetic_col not in real.columns:
        real[args.synthetic_col] = 0
    if args.synthetic_col not in synthetic.columns:
        synthetic[args.synthetic_col] = 1

    real = real[real[args.synthetic_col].astype(int) == 0].copy()
    synthetic = synthetic[synthetic[args.label_col].astype(str).isin(keep_classes)].copy()
    result = pd.concat([real, synthetic], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    report = {
        "real_train_csv": args.real_train_csv,
        "synthetic_csv": args.synthetic_csv,
        "out_csv": str(out_csv),
        "classes": sorted(keep_classes),
        "real_count": int(len(real)),
        "synthetic_count": int(len(synthetic)),
        "synthetic_by_class": synthetic[args.label_col].value_counts().sort_index().to_dict(),
        "total_count": int(len(result)),
    }
    (out_csv.with_suffix(".summary.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
