from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fresh group-aware Stage 8 split with a locked test set.")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--input-csvs", default="splits/train.csv,splits/val.csv,splits/test.csv")
    parser.add_argument("--synthetic-csv", default=None)
    parser.add_argument("--out-dir", default="splits/stage8")
    parser.add_argument("--locked-test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    return groups


def group_label(rows: list[dict[str, str]]) -> str:
    return Counter(str(row["label"]) for row in rows).most_common(1)[0][0]


def choose_groups(
    groups: dict[str, list[dict[str, str]]],
    fraction: float,
    seed: int,
    eligible: set[str] | None = None,
) -> set[str]:
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    totals: Counter[str] = Counter()
    for group_id, rows in groups.items():
        label = group_label(rows)
        totals[label] += len(rows)
        if eligible is None or group_id in eligible:
            by_label[label].append((group_id, rows))

    selected: set[str] = set()
    for label, candidates in sorted(by_label.items()):
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
        target = max(1, round(totals[label] * fraction))
        current = 0
        for group_id, rows in candidates:
            size = len(rows)
            before = abs(target - current)
            after = abs(target - (current + size))
            if current < target and (after <= before or current == 0):
                selected.add(group_id)
                current += size
    return selected


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "num_images": len(rows),
        "num_groups": len({row["group_id"] for row in rows}),
        "labels": dict(Counter(row["label"] for row in rows)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    out_dir = root / args.out_dir
    input_paths = [root / value.strip() for value in args.input_csvs.split(",") if value.strip()]
    rows_by_id: dict[str, dict[str, str]] = {}
    for path in input_paths:
        for row in read_rows(path):
            if int(row.get("is_synthetic", 0)) == 0:
                rows_by_id[row["image_id"]] = row
    real_rows = list(rows_by_id.values())
    groups = group_rows(real_rows)

    synthetic_path = Path(args.synthetic_csv) if args.synthetic_csv else None
    if synthetic_path is not None and not synthetic_path.is_absolute():
        synthetic_path = root / synthetic_path
    synthetic_rows = read_rows(synthetic_path) if synthetic_path is not None else []
    source_groups = {
        str(row.get("source_group_id", "")).strip()
        for row in synthetic_rows
        if str(row.get("source_group_id", "")).strip()
    }
    locked_groups = choose_groups(groups, args.locked_test_size, args.seed)
    remaining_groups = {group_id: rows for group_id, rows in groups.items() if group_id not in locked_groups}
    val_groups = choose_groups(remaining_groups, args.val_size, args.seed + 1)

    splits = {
        "train": [
            row
            for row in real_rows
            if row["group_id"] not in locked_groups and row["group_id"] not in val_groups
        ],
        "val": [row for row in real_rows if row["group_id"] in val_groups],
        "locked_test": [row for row in real_rows if row["group_id"] in locked_groups],
    }
    group_sets = {name: {row["group_id"] for row in split_rows} for name, split_rows in splits.items()}
    for left, right in (("train", "val"), ("train", "locked_test"), ("val", "locked_test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise RuntimeError(f"Group leakage between {left} and {right}: {len(overlap)} groups")
    source_locked_overlap = source_groups & group_sets["locked_test"]
    if source_locked_overlap:
        raise RuntimeError(f"Synthetic source leakage into locked test: {len(source_locked_overlap)} groups")
    source_val_overlap = source_groups & group_sets["val"]
    if source_val_overlap:
        raise RuntimeError(f"Synthetic source leakage into validation: {len(source_val_overlap)} groups")
    unknown_source_groups = source_groups - group_sets["train"]
    if unknown_source_groups:
        raise RuntimeError(f"Synthetic sources outside Stage 8 train: {len(unknown_source_groups)} groups")

    real_fields = ["image_path", "label", "is_synthetic", "image_id", "group_id", "source"]
    write_rows(out_dir / "train_real.csv", splits["train"], real_fields)
    write_rows(out_dir / "val_real.csv", splits["val"], real_fields)
    write_rows(out_dir / "locked_test_real.csv", splits["locked_test"], real_fields)

    train_groups = group_sets["train"]
    train_synthetic = [
        row for row in synthetic_rows if str(row.get("source_group_id", "")).strip() in train_groups
    ]
    synthetic_fields: list[str] = []
    for row in train_synthetic:
        for key in row:
            if key not in synthetic_fields:
                synthetic_fields.append(key)
    if train_synthetic:
        write_rows(out_dir / "synthetic_candidates_train.csv", train_synthetic, synthetic_fields)

    report = {
        "seed": args.seed,
        "locked_test_size": args.locked_test_size,
        "val_size_of_remaining": args.val_size,
        "source_synthetic_csv": str(synthetic_path) if synthetic_path is not None else None,
        "source_group_count": len(source_groups),
        "source_groups_in_locked_test": len(source_locked_overlap),
        "source_groups_in_validation": len(source_val_overlap),
        "splits": {name: summarize(split_rows) for name, split_rows in splits.items()},
        "synthetic_candidates": {
            "input": len(synthetic_rows),
            "train_source_only": len(train_synthetic),
            "labels": dict(Counter(row["label"] for row in train_synthetic)),
        },
        "notes": (
            "The locked test contains no lesion group used as an img2img source. "
            "Generation must use train_real.csv. Stage 8 screening must set evaluation.run_test=false."
        ),
    }
    (out_dir / "split_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
