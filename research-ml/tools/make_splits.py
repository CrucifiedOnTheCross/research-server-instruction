from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def stratify_or_none(frame: pd.DataFrame, label_col: str) -> pd.Series | None:
    counts = frame[label_col].value_counts()
    if counts.empty or counts.min() < 2:
        print("Warning: at least one class has fewer than 2 samples; falling back to non-stratified split.")
        return None
    return frame[label_col]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible stratified train/val/test CSV splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--synthetic-col", default="is_synthetic")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    if args.synthetic_col not in frame.columns:
        frame[args.synthetic_col] = 0
    real = frame[frame[args.synthetic_col].astype(int) == 0].copy()
    synthetic = frame[frame[args.synthetic_col].astype(int) == 1].copy()

    train_real, test = train_test_split(
        real,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_or_none(real, args.label_col),
    )
    relative_val = args.val_size / (1.0 - args.test_size)
    train_real, val = train_test_split(
        train_real,
        test_size=relative_val,
        random_state=args.seed,
        stratify=stratify_or_none(train_real, args.label_col),
    )
    train = pd.concat([train_real, synthetic], ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(
        {
            "train": len(train),
            "train_real": len(train_real),
            "train_synthetic": len(synthetic),
            "val": len(val),
            "test": len(test),
        }
    )


if __name__ == "__main__":
    main()
