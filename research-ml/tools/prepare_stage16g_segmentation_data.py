from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


IMAGE_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task1-2_Training_Input.zip"
)
ID_PATTERN = re.compile(r"(ISIC_\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-aware ISIC 2018 segmentation manifests."
    )
    parser.add_argument(
        "--data-root", default="/srv/research/projects/default/isic2019"
    )
    parser.add_argument("--image-url", default=IMAGE_URL)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        return
    if not shutil.which("curl"):
        raise RuntimeError("curl is required")
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


def image_id(name: str) -> str | None:
    match = ID_PATTERN.search(PurePosixPath(name).name)
    return match.group(1).upper() if match else None


def extract_images(
    archive_path: Path, destination: Path, required_ids: set[str]
) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    selected: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
            identifier = image_id(info.filename)
            if not identifier or identifier not in required_ids:
                continue
            if identifier in selected:
                raise RuntimeError(f"Duplicate image in archive: {identifier}")
            output = destination / f"{identifier}.jpg"
            if not output.is_file():
                with archive.open(info) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
            selected[identifier] = output
    missing = required_ids - set(selected)
    if missing:
        raise RuntimeError(f"{len(missing)} mask IDs have no matching input image")
    return selected


def split_segmentation_rows(
    mask_rows: list[dict[str, str]],
    stage16_splits: dict[str, list[dict[str, str]]],
    image_paths: dict[str, Path],
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_ids = {
        name: {row["image_id"].upper() for row in rows}
        for name, rows in stage16_splits.items()
    }
    all_stage16_ids = set().union(*split_ids.values())
    evaluation_hashes = {
        str(row.get("sha256", ""))
        for name in ("val", "locked_test")
        for row in stage16_splits[name]
        if str(row.get("sha256", ""))
    }
    train: list[dict[str, Any]] = []
    qualification: list[dict[str, Any]] = []
    excluded_eval_id = 0
    excluded_eval_hash = 0
    for row in mask_rows:
        identifier = row["image_id"].upper()
        image_path = image_paths[identifier]
        digest = sha256_file(image_path)
        output = {
            "image_id": identifier,
            "image_path": image_path.relative_to(data_root).as_posix(),
            "mask_path": row["mask_path"],
            "image_sha256": digest,
            "mask_sha256": row["mask_sha256"],
            "split_role": "",
        }
        if identifier in split_ids["train"]:
            output["split_role"] = "stage16_train_qualification"
            qualification.append(output)
        elif identifier in split_ids["val"] | split_ids["locked_test"]:
            excluded_eval_id += 1
        elif digest in evaluation_hashes:
            excluded_eval_hash += 1
        elif identifier not in all_stage16_ids:
            output["split_role"] = "external_isic2018_segmentation_train"
            train.append(output)

    if {row["image_id"] for row in train} & all_stage16_ids:
        raise RuntimeError("Segmentation train contains a Stage 16 image ID")
    if {row["image_id"] for row in qualification} - split_ids["train"]:
        raise RuntimeError("Qualification contains non-train Stage 16 IDs")
    report = {
        "protocol": "stage16g_segmentation_split_v1",
        "locked_test_evaluated": False,
        "segmentation_train_images": len(train),
        "train_only_qualification_images": len(qualification),
        "excluded_stage16_validation_or_test_ids": excluded_eval_id,
        "excluded_stage16_validation_or_test_hashes": excluded_eval_hash,
    }
    return train, qualification, report


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    stage16g_root = data_root / "splits" / "stage16g"
    inventory = json.loads(
        (stage16g_root / "mask_inventory.json").read_text(encoding="utf-8")
    )
    mask_root = data_root / "auxiliary" / "isic2018_task1_masks" / "masks"
    known_mask_ids = {
        path.stem.replace("_segmentation", "").upper(): path
        for path in mask_root.glob("*_segmentation.png")
    }
    if len(known_mask_ids) != int(inventory["available_mask_ids"]):
        raise RuntimeError("Extracted mask count differs from mask inventory")

    auxiliary = data_root / "auxiliary" / "isic2018_task1_images"
    archive_path = auxiliary / "ISIC2018_Task1-2_Training_Input.zip"
    download(args.image_url, archive_path, args.force_download)
    image_paths = extract_images(
        archive_path, auxiliary / "images", set(known_mask_ids)
    )

    stage16_splits = {
        name: read_csv(data_root / "splits" / f"{name}.csv")
        for name in ("train", "val", "locked_test")
    }
    full_mask_rows = [
        {
            "image_id": identifier,
            "mask_path": path.relative_to(data_root).as_posix(),
            "mask_sha256": sha256_file(path),
        }
        for identifier, path in sorted(known_mask_ids.items())
    ]
    train, qualification, report = split_segmentation_rows(
        full_mask_rows, stage16_splits, image_paths, data_root
    )
    report.update(
        {
            "source_url": args.image_url,
            "image_archive_sha256": sha256_file(archive_path),
            "available_image_mask_pairs": len(full_mask_rows),
        }
    )
    write_csv(stage16g_root / "segmentation_train.csv", train)
    write_csv(stage16g_root / "segmentation_qualification.csv", qualification)
    (stage16g_root / "segmentation_split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
