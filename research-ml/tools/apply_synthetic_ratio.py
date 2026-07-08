from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a controlled train CSV with a fixed synthetic-to-real ratio.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--synthetic-col", default="is_synthetic")
    parser.add_argument("--ratio", type=float, default=0.0, help="Synthetic samples per real sample within each class.")
    parser.add_argument("--balance-to-head", action="store_true", help="Use synthetic samples to bring each class up to head real count.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.train_csv)
    if args.synthetic_col not in frame.columns:
        frame[args.synthetic_col] = 0

    real = frame[frame[args.synthetic_col].astype(int) == 0].copy()
    synthetic = frame[frame[args.synthetic_col].astype(int) == 1].copy()
    real_counts = real[args.label_col].value_counts().to_dict()
    head_count = max(real_counts.values()) if real_counts else 0

    pieces = [real]
    selected_rows = []
    for label, real_count in sorted(real_counts.items()):
        candidates = synthetic[synthetic[args.label_col] == label]
        if args.balance_to_head:
            need = max(0, head_count - int(real_count))
        else:
            need = max(0, round(int(real_count) * float(args.ratio)))
        if need <= 0 or candidates.empty:
            continue
        selected_rows.append(candidates.sample(n=min(need, len(candidates)), random_state=args.seed))

    if selected_rows:
        pieces.extend(selected_rows)
    result = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    print(
        {
            "real": int(len(real)),
            "synthetic_available": int(len(synthetic)),
            "synthetic_selected": int((result[args.synthetic_col].astype(int) == 1).sum()),
            "total": int(len(result)),
            "out_csv": str(out_csv),
        }
    )


if __name__ == "__main__":
    main()

