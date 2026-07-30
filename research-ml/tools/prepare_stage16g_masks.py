from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


MASK_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task1_Training_GroundTruth.zip"
)
IMAGE_ID_PATTERN = re.compile(r"(ISIC_\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an ISIC segmentation-mask manifest for train only."
    )
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/isic2019"
    )
    parser.add_argument("--archive-url", default=MASK_URL)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty mask manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
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


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.is_file() and not force:
        return
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for resumable official downloads")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
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
        ],
        check=True,
    )


def image_id_from_mask_name(name: str) -> str | None:
    match = IMAGE_ID_PATTERN.search(PurePosixPath(name).name)
    return match.group(1).upper() if match else None


def safe_mask_members(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if info.is_dir() or member.suffix.lower() != ".png":
            continue
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
        image_id = image_id_from_mask_name(info.filename)
        if image_id:
            members[image_id].append(info.filename)
    return dict(members)


def choose_mask_member(image_id: str, members: list[str]) -> str:
    canonical = [
        name
        for name in members
        if PurePosixPath(name).stem.lower()
        == f"{image_id.lower()}_segmentation"
    ]
    if len(canonical) == 1:
        return canonical[0]
    if len(members) == 1:
        return members[0]
    raise RuntimeError(
        f"{image_id}: ambiguous masks ({len(members)}); consensus policy required"
    )


def extract_selected_masks(
    archive_path: Path, destination: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")
        grouped = safe_mask_members(archive)
        selected: dict[str, Path] = {}
        for image_id, members in sorted(grouped.items()):
            member = choose_mask_member(image_id, members)
            output = destination / f"{image_id}_segmentation.png"
            if not output.is_file():
                with archive.open(member) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
            selected[image_id] = output
    return selected, {
        "archive_png_members": sum(len(value) for value in grouped.values()),
        "archive_unique_image_ids": len(grouped),
        "selected_masks": len(selected),
    }


def build_train_mask_manifest(
    data_root: Path, selected: dict[str, Path], archive_sha256: str, source_url: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_root = data_root / "splits"
    train = read_csv(split_root / "train.csv")
    validation = read_csv(split_root / "val.csv")
    locked_test = read_csv(split_root / "locked_test.csv")
    train_ids = {row["image_id"].upper() for row in train}
    eval_ids = {
        row["image_id"].upper() for row in validation + locked_test
    }
    qualified: list[dict[str, Any]] = []
    for row in train:
        image_id = row["image_id"].upper()
        mask_path = selected.get(image_id)
        if mask_path is None:
            continue
        output = dict(row)
        output.update(
            {
                "mask_path": mask_path.relative_to(data_root).as_posix(),
                "mask_sha256": sha256_file(mask_path),
                "mask_source": "ISIC2018_Task1_Training_GroundTruth",
                "mask_archive_sha256": archive_sha256,
            }
        )
        qualified.append(output)

    output_ids = {row["image_id"].upper() for row in qualified}
    if output_ids & eval_ids:
        raise RuntimeError("Mask manifest contains validation or locked-test IDs")
    if not output_ids <= train_ids:
        raise RuntimeError("Mask manifest contains IDs outside the train split")
    if len(output_ids) != len(qualified):
        raise RuntimeError("Mask-qualified train image IDs are not unique")

    class_counts = Counter(str(row["label"]) for row in qualified)
    lesion_column = "group_id" if train and "group_id" in train[0] else "image_id"
    lesion_counts = {
        label: len(
            {
                str(row.get(lesion_column) or row["image_id"])
                for row in qualified
                if str(row["label"]) == label
            }
        )
        for label in sorted(class_counts)
    }
    inventory = {
        "protocol": "stage16g_train_only_mask_manifest_v1",
        "locked_test_evaluated": False,
        "source_url": source_url,
        "archive_sha256": archive_sha256,
        "train_images": len(train),
        "validation_images_not_used": len(validation),
        "locked_test_images_not_used": len(locked_test),
        "available_mask_ids": len(selected),
        "mask_ids_in_train": len(output_ids),
        "mask_ids_in_validation_or_test_not_used": len(set(selected) & eval_ids),
        "class_image_counts": dict(sorted(class_counts.items())),
        "class_unique_lesion_counts": lesion_counts,
    }
    return qualified, inventory


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    auxiliary = data_root / "auxiliary" / "isic2018_task1_masks"
    archive_path = auxiliary / "ISIC2018_Task1_Training_GroundTruth.zip"
    mask_root = auxiliary / "masks"
    download(args.archive_url, archive_path, args.force_download)
    archive_sha256 = sha256_file(archive_path)
    selected, archive_stats = extract_selected_masks(archive_path, mask_root)
    rows, inventory = build_train_mask_manifest(
        data_root, selected, archive_sha256, args.archive_url
    )
    inventory.update(archive_stats)
    output_root = data_root / "splits" / "stage16g"
    write_csv(output_root / "mask_qualified_train.csv", rows)
    (output_root / "mask_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
