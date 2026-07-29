from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


TRAIN_IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    "ISIC_2019_Training_Input.zip"
)
TRAIN_LABELS_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    "ISIC_2019_Training_GroundTruth.csv"
)
TRAIN_METADATA_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    "ISIC_2019_Training_Metadata.csv"
)
CLASS_NAMES = ("mel", "nv", "bcc", "ak", "bkl", "df", "vasc", "scc")
GROUND_TRUTH_COLUMNS = {name.upper(): name for name in CLASS_NAMES}
EXPECTED_IMAGES = 25_331


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, audit and split official ISIC 2019 training data."
    )
    parser.add_argument(
        "--root", default="/srv/research/projects/default/isic2019"
    )
    parser.add_argument(
        "--ham-manifest",
        default="/srv/research/projects/default/ham10000/manifest.csv",
    )
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-image-audit", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:0{hash_size * hash_size // 4}x}"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def download(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"Using cached {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for resumable official downloads")
    run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "10",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ]
    )


def extract_images(archive: Path, raw_root: Path) -> Path:
    image_root = raw_root / "ISIC_2019_Training_Input"
    existing = list(image_root.glob("*.jpg"))
    if len(existing) == EXPECTED_IMAGES:
        print(f"Using {len(existing)} extracted images at {image_root}", flush=True)
        return image_root
    if existing:
        raise RuntimeError(
            f"Partial extraction at {image_root}: {len(existing)} images"
        )
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
        handle.extractall(raw_root)
    images = list(image_root.glob("*.jpg"))
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} extracted images, found {len(images)}"
        )
    return image_root


def load_ham_ids(path: Path) -> set[str]:
    if not path.is_file():
        print(f"HAM manifest not found at {path}; source overlap remains unknown")
        return set()
    return {
        str(row.get("image_id", "")).strip()
        for row in read_csv(path)
        if str(row.get("image_id", "")).strip()
    }


def label_from_ground_truth(row: dict[str, str]) -> str:
    active = [
        label
        for column, label in GROUND_TRUTH_COLUMNS.items()
        if float(row.get(column, 0) or 0) == 1.0
    ]
    if len(active) != 1:
        raise ValueError(
            f"{row.get('image')}: expected exactly one known label, found {active}"
        )
    if float(row.get("UNK", 0) or 0) != 0.0:
        raise ValueError(f"{row.get('image')}: UNK is not allowed in training data")
    return active[0]


def audit_image(path: Path, skip: bool) -> dict[str, Any]:
    file_hash = sha256_file(path)
    if skip:
        return {
            "sha256": file_hash,
            "visual_hash": "",
            "width": "",
            "height": "",
            "aspect_ratio": "",
            "decode_ok": 1,
        }
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            visual_hash = difference_hash(image)
    except Exception as exc:
        return {
            "sha256": file_hash,
            "visual_hash": "",
            "width": "",
            "height": "",
            "aspect_ratio": "",
            "decode_ok": 0,
            "decode_error": repr(exc),
        }
    return {
        "sha256": file_hash,
        "visual_hash": visual_hash,
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "decode_ok": 1,
    }


def build_manifest(
    root: Path,
    labels_path: Path,
    metadata_path: Path,
    image_root: Path,
    ham_ids: set[str],
    skip_image_audit: bool,
) -> list[dict[str, Any]]:
    labels = {row["image"]: row for row in read_csv(labels_path)}
    metadata = {row["image"]: row for row in read_csv(metadata_path)}
    if len(labels) != EXPECTED_IMAGES or len(metadata) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Unexpected official CSV sizes labels={len(labels)} metadata={len(metadata)}"
        )
    if set(labels) != set(metadata):
        raise RuntimeError("Ground truth and metadata image IDs differ")

    manifest: list[dict[str, Any]] = []
    for index, image_id in enumerate(sorted(labels), start=1):
        image_path = image_root / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        audit = audit_image(image_path, skip=skip_image_audit)
        metadata_row = metadata[image_id]
        manifest.append(
            {
                "image_path": image_path.relative_to(root).as_posix(),
                "label": label_from_ground_truth(labels[image_id]),
                "is_synthetic": 0,
                "sample_weight": 1.0,
                "image_id": image_id,
                "patient_id": "",
                "lesion_id": str(metadata_row.get("lesion_id", "")).strip(),
                "group_id": "",
                "source": (
                    "ham10000"
                    if image_id in ham_ids
                    else "isic2019_non_ham"
                ),
                "dataset_release": "ISIC_2019_Training",
                "age_approx": metadata_row.get("age_approx", ""),
                "sex": metadata_row.get("sex", ""),
                "anatom_site_general": metadata_row.get(
                    "anatom_site_general", ""
                ),
                **audit,
            }
        )
        if index % 1000 == 0:
            print(f"Audited {index}/{EXPECTED_IMAGES} images", flush=True)

    bad = [row for row in manifest if int(row["decode_ok"]) != 1]
    if bad:
        raise RuntimeError(f"{len(bad)} images failed decoding")
    return manifest


def assign_connected_groups(
    manifest: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_ids = [str(row["image_id"]) for row in manifest]
    union_find = UnionFind(image_ids)
    key_owner: dict[str, str] = {}

    for row in manifest:
        image_id = str(row["image_id"])
        keys = [
            f"sha256:{row['sha256']}",
            f"visual_hash:{row['visual_hash']}",
        ]
        lesion_id = str(row.get("lesion_id", "")).strip()
        if lesion_id:
            keys.append(f"lesion:{lesion_id}")
        for key in keys:
            owner = key_owner.setdefault(key, image_id)
            union_find.union(image_id, owner)

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        components[union_find.find(str(row["image_id"]))].append(row)

    kept: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for rows in components.values():
        labels = {str(row["label"]) for row in rows}
        stable_group = "isicgrp_" + hashlib.sha256(
            "\n".join(sorted(str(row["image_id"]) for row in rows)).encode()
        ).hexdigest()[:20]
        for row in rows:
            row["group_id"] = stable_group
            row["group_size"] = len(rows)
            row["group_label_conflict"] = int(len(labels) > 1)
        if len(labels) > 1:
            quarantine.extend(rows)
        else:
            kept.extend(rows)
    return kept, quarantine


def split_groups(
    manifest: list[dict[str, Any]],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val/test sizes must be positive and sum to less than one")
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        grouped[str(row["group_id"])].append(row)

    by_label: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for rows in grouped.values():
        labels = {str(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels survived quarantine: {labels}")
        by_label[next(iter(labels))].append(rows)

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "locked_test": [],
    }
    fractions = {
        "train": 1.0 - val_size - test_size,
        "val": val_size,
        "locked_test": test_size,
    }
    for label, groups in sorted(by_label.items()):
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = sum(len(rows) for rows in groups)
        targets = {
            split: round(total * fraction)
            for split, fraction in fractions.items()
        }
        targets["train"] = total - targets["val"] - targets["locked_test"]
        current = {split: 0 for split in splits}
        for rows in groups:
            deficits = {
                split: targets[split] - current[split] for split in splits
            }
            positive = {split: gap for split, gap in deficits.items() if gap > 0}
            if positive:
                destination = max(positive, key=positive.get)
            else:
                destination = min(
                    splits,
                    key=lambda split: current[split] / max(1, targets[split]),
                )
            for row in rows:
                row["split"] = destination
                row["split_seed"] = seed
                row["split_policy_version"] = "isic2019_connected_group_v1"
            splits[destination].extend(rows)
            current[destination] += len(rows)
        if any(
            not any(str(row["label"]) == label for row in rows)
            for rows in splits.values()
        ):
            raise RuntimeError(f"Class {label} missing from one or more splits")
    return splits


def validate_disjoint(splits: dict[str, list[dict[str, Any]]]) -> None:
    for key in ("group_id", "lesion_id", "sha256", "visual_hash"):
        sets: dict[str, set[str]] = {}
        for split, rows in splits.items():
            sets[split] = {
                str(row.get(key, "")).strip()
                for row in rows
                if str(row.get(key, "")).strip()
            }
        names = list(sets)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise RuntimeError(
                        f"{key} leakage {left}/{right}: {len(overlap)} values"
                    )


def make_smoke_splits(
    splits: dict[str, list[dict[str, Any]]],
    per_class: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_label[str(row["label"])].append(row)
        selected: list[dict[str, Any]] = []
        for label in CLASS_NAMES:
            selected.extend(sorted(by_label[label], key=lambda row: row["image_id"])[:per_class])
        result[split] = selected
    return result


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        label: int(count)
        for label, count in sorted(Counter(row["label"] for row in rows).items())
    }


def manifest_fields() -> list[str]:
    return [
        "image_path",
        "label",
        "is_synthetic",
        "sample_weight",
        "image_id",
        "patient_id",
        "lesion_id",
        "group_id",
        "group_size",
        "group_label_conflict",
        "source",
        "dataset_release",
        "age_approx",
        "sex",
        "anatom_site_general",
        "sha256",
        "visual_hash",
        "width",
        "height",
        "aspect_ratio",
        "decode_ok",
        "split",
        "split_seed",
        "split_policy_version",
    ]


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    raw_root = root / "raw"
    archive = raw_root / "ISIC_2019_Training_Input.zip"
    labels_path = raw_root / "ISIC_2019_Training_GroundTruth.csv"
    metadata_path = raw_root / "ISIC_2019_Training_Metadata.csv"
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download(TRAIN_IMAGES_URL, archive, args.force_download)
        download(TRAIN_LABELS_URL, labels_path, args.force_download)
        download(TRAIN_METADATA_URL, metadata_path, args.force_download)
    required = [archive, labels_path, metadata_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official files: {missing}")

    image_root = extract_images(archive, raw_root)
    manifest = build_manifest(
        root,
        labels_path,
        metadata_path,
        image_root,
        load_ham_ids(Path(args.ham_manifest)),
        args.skip_image_audit,
    )
    manifest, quarantine = assign_connected_groups(manifest)
    splits = split_groups(manifest, args.val_size, args.test_size, args.seed)
    validate_disjoint(splits)
    smoke = make_smoke_splits(splits)

    fields = manifest_fields()
    write_csv(root / "manifest.csv", manifest, fields)
    write_csv(root / "quarantine_label_conflicts.csv", quarantine, fields)
    for split, rows in splits.items():
        write_csv(root / "splits" / f"{split}.csv", rows, fields)
    for split, rows in smoke.items():
        write_csv(root / "splits" / f"smoke_{split}.csv", rows, fields)

    archive_manifest = {
        "training_images": {
            "url": TRAIN_IMAGES_URL,
            "path": str(archive.relative_to(root)),
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "ground_truth": {
            "url": TRAIN_LABELS_URL,
            "path": str(labels_path.relative_to(root)),
            "sha256": sha256_file(labels_path),
            "bytes": labels_path.stat().st_size,
        },
        "metadata": {
            "url": TRAIN_METADATA_URL,
            "path": str(metadata_path.relative_to(root)),
            "sha256": sha256_file(metadata_path),
            "bytes": metadata_path.stat().st_size,
        },
    }
    summary = {
        "protocol": "stage16a_isic2019_curated_v1",
        "dataset": "ISIC 2019 Challenge Training",
        "license": "CC-BY-NC-4.0",
        "expected_images": EXPECTED_IMAGES,
        "usable_images": len(manifest),
        "quarantined_label_conflicts": len(quarantine),
        "class_names": list(CLASS_NAMES),
        "class_counts": class_counts(manifest),
        "unique_groups": len({row["group_id"] for row in manifest}),
        "ham10000_overlap_images": sum(
            str(row["source"]) == "ham10000" for row in manifest
        ),
        "split_seed": args.seed,
        "split_policy_version": "isic2019_connected_group_v1",
        "locked_test_evaluated": False,
        "splits": {
            split: {
                "images": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "class_counts": class_counts(rows),
                "source_counts": dict(Counter(row["source"] for row in rows)),
            }
            for split, rows in splits.items()
        },
        "archives": archive_manifest,
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
