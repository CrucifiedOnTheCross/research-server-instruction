from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HAM10000_COLLECTION_ID = 66
HAM10000_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DIAGNOSIS_TO_LABEL = {
    "actinic keratoses": "akiec",
    "actinic keratosis": "akiec",
    "akiec": "akiec",
    "basal cell carcinoma": "bcc",
    "bcc": "bcc",
    "benign keratosis-like lesions": "bkl",
    "benign keratosis": "bkl",
    "bkl": "bkl",
    "dermatofibroma": "df",
    "df": "df",
    "melanoma": "mel",
    "mel": "mel",
    "melanocytic nevi": "nv",
    "nevus": "nv",
    "nv": "nv",
    "vascular lesions": "vasc",
    "vascular lesion": "vasc",
    "vasc": "vasc",
    "melanoma, nos": "mel",
    "malignant melanocytic proliferations (melanoma)": "mel",
    "nevus": "nv",
    "benign melanocytic proliferations": "nv",
    "pigmented benign keratosis": "bkl",
    "benign epidermal proliferations": "bkl",
    "solar or actinic keratosis": "akiec",
    "squamous cell carcinoma, nos": "akiec",
    "indeterminate epidermal proliferations": "akiec",
    "malignant epidermal proliferations": "akiec",
    "malignant adnexal epithelial proliferations - follicular": "bcc",
    "benign soft tissue proliferations - fibro-histiocytic": "df",
    "benign soft tissue proliferations - vascular": "vasc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache HAM10000 from ISIC and create reproducible manifests, class folders, and splits."
    )
    parser.add_argument("--root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--source", choices=["isic-cli", "existing"], default="isic-cli")
    parser.add_argument("--collection-id", type=int, default=HAM10000_COLLECTION_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--link-mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="ISIC download limit. 0 means all images.")
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_isic_cli() -> None:
    if shutil.which("isic"):
        return
    run([sys.executable, "-m", "pip", "install", "--upgrade", "isic-cli"])
    if not shutil.which("isic"):
        raise RuntimeError("isic-cli was installed, but `isic` executable is still not on PATH.")


def download_with_isic_cli(root: Path, collection_id: int, limit: int, force: bool, skip: bool) -> Path:
    raw_dir = root / "raw" / f"isic_collection_{collection_id}"
    metadata_path = raw_dir / "metadata.csv"
    if skip:
        print("Skipping download by request.")
        return raw_dir
    if metadata_path.exists() and any(raw_dir.rglob("*.jpg")) and not force:
        print(f"Using cached ISIC download at {raw_dir}")
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_isic_cli()
    run(["isic", "image", "download", "-l", str(limit), "-c", str(collection_id), str(raw_dir)])
    return raw_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_metadata(root: Path) -> Path:
    preferred = [
        root / "raw" / f"isic_collection_{HAM10000_COLLECTION_ID}" / "metadata.csv",
        root / "raw" / "metadata.csv",
        root / "HAM10000_metadata.csv",
        root / "raw" / "HAM10000_metadata.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(root.rglob("*metadata*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No metadata CSV found under {root}")


def image_id_from_row(row: dict[str, str]) -> str:
    for key in ("image_id", "isic_id", "isic_id_unquoted", "name", "id"):
        value = row.get(key)
        if value:
            return Path(value).stem
    raise ValueError(f"Cannot infer image id from row keys: {sorted(row)}")


def label_from_row(row: dict[str, str]) -> str:
    diagnosis_keys = (
        "dx",
        "diagnosis",
        "diagnosis_5",
        "diagnosis_4",
        "diagnosis_3",
        "diagnosis_2",
        "diagnosis_1",
        "benign_malignant",
    )
    for key in diagnosis_keys:
        value = row.get(key)
        if not value:
            continue
        normalized = value.strip().lower().replace("_", " ")
        if normalized in DIAGNOSIS_TO_LABEL:
            return DIAGNOSIS_TO_LABEL[normalized]
        if normalized in HAM10000_LABELS:
            return normalized
    raise ValueError(f"Cannot map diagnosis to HAM10000 label for image {row}")


def group_id_from_row(row: dict[str, str], image_id: str) -> str:
    for key in ("lesion_id", "lesion", "patient_id", "patient", "patient_id_unquoted"):
        value = row.get(key)
        if value:
            return value
    return image_id


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        for path in root.rglob(suffix):
            parts = {part.lower() for part in path.parts}
            if "images_by_class" in parts or "splits" in parts or "synthetic" in parts:
                continue
            index[path.stem] = path
    return index


def build_manifest(root: Path) -> list[dict[str, Any]]:
    metadata_path = find_metadata(root)
    rows = read_csv(metadata_path)
    image_index = build_image_index(root)
    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        image_id = image_id_from_row(row)
        image_path = image_index.get(image_id)
        if image_path is None:
            missing.append(image_id)
            continue
        label = label_from_row(row)
        manifest.append(
            {
                "image_path": image_path.relative_to(root).as_posix(),
                "label": label,
                "is_synthetic": 0,
                "image_id": image_id,
                "group_id": group_id_from_row(row, image_id),
                "source": "ham10000",
            }
        )
    if missing:
        preview = ", ".join(missing[:10])
        print(f"Warning: {len(missing)} metadata rows have no image file. First missing: {preview}")
    if not manifest:
        raise RuntimeError(f"No usable images found from metadata {metadata_path}")
    return manifest


def make_class_folders(root: Path, manifest: list[dict[str, Any]], mode: str) -> None:
    if mode == "none":
        return
    class_root = root / "images_by_class"
    class_root.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        src = root / row["image_path"]
        dst = class_root / row["label"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)


def split_groups(
    manifest: list[dict[str, Any]],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        groups[str(row["group_id"])].append(row)

    train_size = 1.0 - val_size - test_size
    split_fracs = {"train": train_size, "val": val_size, "test": test_size}
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    groups_by_label: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for rows in groups.values():
        labels = [str(row["label"]) for row in rows]
        label = Counter(labels).most_common(1)[0][0]
        if len(set(labels)) > 1:
            print(f"Warning: group {rows[0]['group_id']} has multiple labels; using majority label {label}.")
        groups_by_label[label].append(rows)

    for label, label_groups in sorted(groups_by_label.items()):
        rng.shuffle(label_groups)
        label_groups.sort(key=len, reverse=True)
        label_total = sum(len(rows) for rows in label_groups)
        label_targets = {
            "val": round(label_total * val_size),
            "test": round(label_total * test_size),
        }
        label_targets["train"] = label_total - label_targets["val"] - label_targets["test"]
        current = {split: 0 for split in splits}

        for rows in label_groups:
            deficits = {split: label_targets[split] - current[split] for split in splits}
            positive = {split: gap for split, gap in deficits.items() if gap > 0}
            if positive:
                best_split = max(positive, key=positive.get)
            else:
                best_split = min(splits, key=lambda split: current[split] / max(1, label_targets[split]))
            splits[best_split].extend(rows)
            current[best_split] += len(rows)
    return splits


def validate_group_disjoint_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    group_sets = {split: {str(row["group_id"]) for row in rows} for split, rows in splits.items()}
    split_names = sorted(group_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:10])
                raise RuntimeError(
                    f"Group leakage detected between {left} and {right}: "
                    f"{len(overlap)} shared lesion/patient ids. First overlapping ids: {preview}"
                )


def write_summary(root: Path, manifest: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    validate_group_disjoint_splits(splits)
    summary = {
        "dataset": "HAM10000 / ISIC Challenge 2018 Task 3 Training",
        "isic_collection_id": args.collection_id,
        "num_images": len(manifest),
        "labels": dict(Counter(row["label"] for row in manifest)),
        "splits": {
            split: {
                "num_images": len(rows),
                "labels": dict(Counter(row["label"] for row in rows)),
                "num_groups": len({row["group_id"] for row in rows}),
            }
            for split, rows in splits.items()
        },
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "notes": "Splits are group-aware by lesion_id/patient_id when available; validation and test are real-only. The script fails if a group_id appears in more than one split.",
        "citations": [
            "Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset. Scientific Data 5, 180161 (2018).",
            "Codella et al. Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by ISIC.",
        ],
    }
    with (root / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "synthetic").mkdir(parents=True, exist_ok=True)

    if args.source == "isic-cli":
        download_with_isic_cli(root, args.collection_id, args.limit, args.force_download, args.skip_download)

    manifest = build_manifest(root)
    manifest_fields = ["image_path", "label", "is_synthetic", "image_id", "group_id", "source"]
    write_csv(root / "manifest.csv", manifest, manifest_fields)
    make_class_folders(root, manifest, args.link_mode)
    splits = split_groups(manifest, args.val_size, args.test_size, args.seed)
    for split, rows in splits.items():
        write_csv(root / "splits" / f"{split}.csv", rows, manifest_fields)
    write_summary(root, manifest, splits, args)
    print(json.dumps({"root": str(root), "num_images": len(manifest), "labels": dict(Counter(r["label"] for r in manifest))}, indent=2))


if __name__ == "__main__":
    main()
