#!/usr/bin/env python3
"""Create audited ISIC-2019-LT manifests from a pinned MONICA revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


MONICA_COMMIT = "3dd808d6d578b9e0f9bf4ee1402727ff46d1c243"
MONICA_BASE = f"https://raw.githubusercontent.com/PyJulie/MONICA/{MONICA_COMMIT}/numpy/isic"
OFFICIAL_BASE = "https://isic-archive.s3.amazonaws.com/challenges/2019"
CLASS_NAMES = ["nv", "mel", "bcc", "bkl", "ak", "scc", "vasc", "df"]
OFFICIAL_COLUMNS = ["NV", "MEL", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]
EXPECTED_SHA256 = {
    "dic.npy": "6d553b9e4b8407757598ceaed4a3df50bc0dcfba3c255745b9c677c62c36e7b9",
    "train_100.npy": "755378c41ada8999360acb4d8bf3e96510b9561424e2438572ca8283197caa2f",
    "val_100.npy": "32871cb1cf93bc02ab8abce2b93c972e7a1c4c084d2a3c36fb0239d3414e744e",
    "test_100.npy": "738435843f60280feda059a47fa73119e56efff47c42c629db6f5a16d568f77c",
    "train_200.npy": "ba807299dd0fb52f4e540781f20d1e9205cf6038e00b2f0fa9eea78ff333d8e7",
    "val_200.npy": "5e21c06af1ca2f3a851da98a066c3c0b4de5ea62716a3d90af893c7f0590e92b",
    "test_200.npy": "a393a3b12b7d0b5178a1f4297674ba5d570e3149d13c44a8328d9ab38d7b8df2",
    "train_500.npy": "e71ca39b0a1e91b67d392e265eceed2be168fb2e80f19f7c6f4f0047a6c0fb1a",
    "val_500.npy": "190cb36977ba3d01e669d91092f01a30a72696010f9039735876e863d4e0ee1d",
    "test_500.npy": "e6c82e33c822cde6e802723f84235d0dd90fae0264af1653adbdac2fa2554daf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def ensure_file(path: Path, url: str, expected_hash: str | None = None) -> None:
    if not path.exists() or (expected_hash and sha256(path) != expected_hash):
        download(url, path)
    if expected_hash and sha256(path) != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch after download: {path}")


def read_csv_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image"]: row for row in rows}


def official_label(row: dict[str, str]) -> int:
    positives = [index for index, name in enumerate(OFFICIAL_COLUMNS) if float(row[name]) == 1.0]
    if len(positives) != 1:
        raise ValueError(f"Expected one positive class for {row.get('image')}: {positives}")
    return positives[0]


def lesion_overlap(rows_a: list[dict], rows_b: list[dict]) -> dict[str, int]:
    lesions_a = {row["lesion_id"] for row in rows_a if row["lesion_id"]}
    lesions_b = {row["lesion_id"] for row in rows_b if row["lesion_id"]}
    shared = lesions_a & lesions_b
    return {
        "shared_lesion_ids": len(shared),
        "images_in_a_with_shared_lesion": sum(row["lesion_id"] in shared for row in rows_a),
        "images_in_b_with_shared_lesion": sum(row["lesion_id"] in shared for row in rows_b),
    }


def make_manifests(root: Path, output: Path, ratios: list[int], download_images: bool) -> dict:
    raw = root / "raw"
    protocol = root / "protocol_sources" / f"monica_{MONICA_COMMIT}"
    labels_path = raw / "ISIC_2019_Training_GroundTruth.csv"
    metadata_path = raw / "ISIC_2019_Training_Metadata.csv"
    ensure_file(labels_path, f"{OFFICIAL_BASE}/ISIC_2019_Training_GroundTruth.csv")
    ensure_file(metadata_path, f"{OFFICIAL_BASE}/ISIC_2019_Training_Metadata.csv")

    image_dir = raw / "ISIC_2019_Training_Input"
    if download_images and not image_dir.exists():
        archive = raw / "ISIC_2019_Training_Input.zip"
        ensure_file(archive, f"{OFFICIAL_BASE}/ISIC_2019_Training_Input.zip")
        with zipfile.ZipFile(archive) as source:
            source.extractall(raw)

    required = ["dic.npy"] + [f"{split}_{ratio}.npy" for ratio in ratios for split in ("train", "val", "test")]
    for filename in required:
        ensure_file(protocol / filename, f"{MONICA_BASE}/{filename}", EXPECTED_SHA256[filename])

    label_rows = read_csv_index(labels_path)
    metadata_rows = read_csv_index(metadata_path)
    mapping = np.load(protocol / "dic.npy", allow_pickle=True).item()
    if set(mapping) != {f"{image}.jpg" for image in label_rows}:
        raise RuntimeError("Pinned MONICA dictionary does not match official image identifiers")
    for filename, numeric_label in mapping.items():
        image_id = Path(filename).stem
        if int(numeric_label) != official_label(label_rows[image_id]):
            raise RuntimeError(f"Class mapping mismatch for {image_id}")

    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "MONICA ISIC-2019-LT",
        "monica_commit": MONICA_COMMIT,
        "test_evaluated": False,
        "official_metadata_columns": list(next(iter(metadata_rows.values())).keys()),
        "patient_id_available": "patient_id" in next(iter(metadata_rows.values())),
        "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "source_sha256": {name: sha256(protocol / name) for name in required},
        "ratios": {},
    }
    eval_ids: dict[int, dict[str, set[str]]] = {}
    missing_images: set[str] = set()
    for ratio in ratios:
        ratio_dir = output / f"ir{ratio}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        split_rows: dict[str, list[dict]] = {}
        eval_ids[ratio] = {}
        for split in ("train", "val", "test"):
            filenames = np.load(protocol / f"{split}_{ratio}.npy", allow_pickle=True).tolist()
            rows: list[dict] = []
            for filename in filenames:
                image_id = Path(str(filename)).stem
                if image_id not in metadata_rows or image_id not in label_rows:
                    raise RuntimeError(f"Unknown image in MONICA {split}: {image_id}")
                label = int(mapping[f"{image_id}.jpg"])
                image_path = image_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    missing_images.add(image_id)
                meta = metadata_rows[image_id]
                rows.append({
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "label": label,
                    "class_name": CLASS_NAMES[label],
                    "lesion_id": meta.get("lesion_id", ""),
                    "source": "isic2019",
                    "split": split,
                })
            with (ratio_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            split_rows[split] = rows
            eval_ids[ratio][split] = {row["image_id"] for row in rows}

        image_overlaps = {
            f"{a}_{b}": len(eval_ids[ratio][a] & eval_ids[ratio][b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        report["ratios"][str(ratio)] = {
            "counts": {split: len(rows) for split, rows in split_rows.items()},
            "class_counts": {
                split: {CLASS_NAMES[i]: Counter(row["label"] for row in rows).get(i, 0) for i in range(8)}
                for split, rows in split_rows.items()
            },
            "image_id_overlap": image_overlaps,
            "lesion_overlap": {
                f"{a}_{b}": lesion_overlap(split_rows[a], split_rows[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
            },
            "manifest_sha256": {split: sha256(ratio_dir / f"{split}.csv") for split in split_rows},
        }

    report["cross_ratio_eval_overlap"] = {
        f"ir{a}_ir{b}": {
            split: len(eval_ids[a][split] & eval_ids[b][split]) for split in ("val", "test")
        }
        for index, a in enumerate(ratios) for b in ratios[index + 1 :]
    }
    report["missing_image_count"] = len(missing_images)
    report["images_ready"] = not missing_images
    (output / "protocol_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data_splits/monica"))
    parser.add_argument("--ratios", nargs="+", type=int, default=[100, 200, 500], choices=[100, 200, 500])
    parser.add_argument("--download-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = make_manifests(args.root, args.output, args.ratios, args.download_images)
    print(json.dumps({"audit": str(args.output / "protocol_audit.json"), "images_ready": result["images_ready"]}, indent=2))
