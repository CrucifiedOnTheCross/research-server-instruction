#!/usr/bin/env python3
"""Build preregistered contamination-control and lesion-disjoint protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from tools.prepare_isic2019_monica import CLASS_NAMES, official_label, read_csv_index
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from prepare_isic2019_monica import CLASS_NAMES, official_label, read_csv_index


TARGET_TRAIN = [5000, 2590, 1342, 695, 360, 187, 97, 51]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap_record(named_rows: dict[str, list[dict]], split_names: tuple[str, ...]) -> dict:
    lesion_sets = {
        name: {row["lesion_id"] for row in named_rows[name] if row["lesion_id"]}
        for name in split_names
    }
    shared = set.intersection(*(lesion_sets[name] for name in split_names))
    return {
        "splits": list(split_names),
        "shared_lesion_ids": len(shared),
        "affected_images": {
            name: sum(row["lesion_id"] in shared for row in named_rows[name])
            for name in split_names
        },
        "affected_images_by_class": {
            name: {
                CLASS_NAMES[label]: sum(
                    row["lesion_id"] in shared and int(row["label"]) == label
                    for row in named_rows[name]
                )
                for label in range(8)
            }
            for name in split_names
        },
    }


def full_overlap_audit(named_rows: dict[str, list[dict]]) -> dict:
    return {
        "train_validation": overlap_record(named_rows, ("train", "validation")),
        "train_test": overlap_record(named_rows, ("train", "test")),
        "validation_test": overlap_record(named_rows, ("validation", "test")),
        "train_validation_test": overlap_record(named_rows, ("train", "validation", "test")),
    }


def write_replacement_balance_report(
    path: Path, cohorts: list[tuple[str, list[dict]]], metadata: dict[str, dict[str, str]],
    image_dir: Path,
) -> list[dict]:
    from PIL import Image

    lesion_counts = Counter(row.get("lesion_id", "") for row in metadata.values() if row.get("lesion_id", ""))
    fields = [
        "cohort", "image_id", "class_name", "age_approx", "sex",
        "anatomical_site_general", "source_dataset", "source_basis", "width", "height",
        "lesion_id", "images_per_lesion", "is_repeated_exposure",
    ]
    records = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, rows in cohorts:
            for row in rows:
                image_id = row["image_id"]
                meta = metadata[image_id]
                image_path = image_dir / f"{image_id}.jpg"
                width = height = ""
                if image_path.exists():
                    with Image.open(image_path) as image:
                        width, height = image.size
                lesion_id = meta.get("lesion_id", "")
                prefix = lesion_id.split("_", 1)[0].upper() if "_" in lesion_id else ""
                source = {"HAM": "ham10000", "BCN": "bcn20000"}.get(prefix, "unknown")
                record = {
                    "cohort": cohort,
                    "image_id": image_id,
                    "class_name": CLASS_NAMES[int(row["label"])],
                    "age_approx": meta.get("age_approx", ""),
                    "sex": meta.get("sex", ""),
                    "anatomical_site_general": meta.get("anatom_site_general", ""),
                    "source_dataset": source,
                    "source_basis": "inferred_from_lesion_id_prefix" if source != "unknown" else "unavailable",
                    "width": width,
                    "height": height,
                    "lesion_id": lesion_id,
                    "images_per_lesion": lesion_counts.get(lesion_id, 1),
                    "is_repeated_exposure": cohort == "repeated_clean_exposure",
                }
                writer.writerow(record)
                records.append(record)
    return records


def balance_summary(records: list[dict]) -> dict:
    result = {}
    for cohort in sorted({row["cohort"] for row in records}):
        rows = [row for row in records if row["cohort"] == cohort]
        numeric = lambda field: [float(row[field]) for row in rows if row[field] not in ("", None)]
        result[cohort] = {
            "exposures": len(rows),
            "unique_images": len({row["image_id"] for row in rows}),
            "class_counts": dict(Counter(row["class_name"] for row in rows)),
            "sex_counts": dict(Counter(row["sex"] or "missing" for row in rows)),
            "anatomical_site_counts": dict(Counter(row["anatomical_site_general"] or "missing" for row in rows)),
            "source_counts": dict(Counter(row["source_dataset"] for row in rows)),
            "age_mean": float(np.mean(numeric("age_approx"))) if numeric("age_approx") else None,
            "width_mean": float(np.mean(numeric("width"))) if numeric("width") else None,
            "height_mean": float(np.mean(numeric("height"))) if numeric("height") else None,
            "images_per_lesion_mean": float(np.mean(numeric("images_per_lesion"))) if numeric("images_per_lesion") else None,
        }
    return result


def select_exact_groups(groups: list[tuple[str, list[dict]]], target: int) -> tuple[list[dict], list[tuple[str, list[dict]]]]:
    reachable = [False] * (target + 1)
    previous_sum = [-1] * (target + 1)
    previous_group = [-1] * (target + 1)
    reachable[0] = True
    for group_index, (_, rows) in enumerate(groups):
        size = len(rows)
        for total in range(target, size - 1, -1):
            if not reachable[total] and reachable[total - size]:
                reachable[total] = True
                previous_sum[total] = total - size
                previous_group[total] = group_index
    if not reachable[target]:
        raise RuntimeError(f"Cannot select lesion groups totalling exactly {target} images")
    selected_indices: set[int] = set()
    cursor = target
    while cursor:
        index = previous_group[cursor]
        selected_indices.add(index)
        cursor = previous_sum[cursor]
    selected = [row for index in selected_indices for row in groups[index][1]]
    remaining = [group for index, group in enumerate(groups) if index not in selected_indices]
    return selected, remaining


def make_row(image_id: str, label: int, metadata: dict[str, str], image_dir: Path, split: str, source: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "image_path": str(image_dir / f"{image_id}.jpg"),
        "label": label,
        "class_name": CLASS_NAMES[label],
        "lesion_id": metadata.get("lesion_id", ""),
        "source": source,
        "split": split,
    }


def build_causal_control(root: Path, monica_dir: Path, output: Path, seed: int) -> dict:
    train = read_rows(monica_dir / "train.csv")
    val = read_rows(monica_dir / "val.csv")
    test = read_rows(monica_dir / "test.csv")
    test_lesions = {row["lesion_id"] for row in test if row["lesion_id"]}
    val_lesions = {row["lesion_id"] for row in val if row["lesion_id"]}
    removed = [row for row in train if row["lesion_id"] in test_lesions]
    kept = [row for row in train if row["lesion_id"] not in test_lesions]
    required = Counter(int(row["label"]) for row in removed)

    labels = read_csv_index(root / "raw" / "ISIC_2019_Training_GroundTruth.csv")
    metadata = read_csv_index(root / "raw" / "ISIC_2019_Training_Metadata.csv")
    image_dir = root / "raw" / "ISIC_2019_Training_Input"
    benchmark_ids = {row["image_id"] for row in train + val + test}
    candidates: dict[int, list[str]] = defaultdict(list)
    for image_id, label_row in labels.items():
        lesion_id = metadata[image_id].get("lesion_id", "")
        if image_id in benchmark_ids or lesion_id in test_lesions or lesion_id in val_lesions:
            continue
        candidates[official_label(label_row)].append(image_id)
    rng = random.Random(seed)
    replacements: list[dict] = []
    repeated: list[dict] = []
    for label, count in sorted(required.items()):
        pool = sorted(candidates[label])
        rng.shuffle(pool)
        selected = pool[:count]
        replacements.extend(make_row(image_id, label, metadata[image_id], image_dir, "train", "isic2019_unique_replacement") for image_id in selected)
        deficit = count - len(selected)
        if deficit:
            repeat_pool = [
                row for row in kept
                if int(row["label"]) == label and row["lesion_id"] not in val_lesions and row["lesion_id"] not in test_lesions
            ]
            repeat_pool += [row for row in replacements if int(row["label"]) == label]
            if not repeat_pool:
                raise RuntimeError(f"No clean exposure-matching pool for {CLASS_NAMES[label]}")
            for _ in range(deficit):
                duplicate = dict(rng.choice(repeat_pool))
                duplicate["source"] = "isic2019_clean_repeated_exposure"
                repeated.append(duplicate)
    matched = kept + replacements + repeated
    rng.shuffle(matched)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "train_decontaminated_unmatched.csv", kept)
    write_rows(output / "train_decontaminated_matched.csv", matched)
    write_rows(output / "val_monica.csv", val)
    write_rows(output / "test_monica.csv", test)
    leaked_test = [row for row in test if row["lesion_id"] in {item["lesion_id"] for item in train if item["lesion_id"]}]
    clean_test = [row for row in test if row not in leaked_test]
    write_rows(output / "test_leaked.csv", leaked_test)
    write_rows(output / "test_clean.csv", clean_test)
    replacement_records = write_replacement_balance_report(
        output / "replacement_balance_report.csv",
        [("removed", removed), ("unique_replacement", replacements), ("repeated_clean_exposure", repeated)],
        metadata,
        image_dir,
    )
    (output / "replacement_balance_summary.json").write_text(
        json.dumps({
            "source_dataset_is_inferred": True,
            "source_inference_rule": "HAM_* -> ham10000; BCN_* -> bcn20000; otherwise unknown",
            "matching_guarantees": ["class", "number_of_training_exposures"],
            "not_guaranteed": ["source", "device", "age", "sex", "anatomical_site", "resolution", "clinical_difficulty"],
            "cohorts": balance_summary(replacement_records),
        }, indent=2), encoding="utf-8"
    )
    final_counts = Counter(int(row["label"]) for row in matched)
    if [final_counts[index] for index in range(8)] != TARGET_TRAIN:
        raise RuntimeError("Matched train class counts changed")
    if test_lesions & {row["lesion_id"] for row in matched if row["lesion_id"]}:
        raise RuntimeError("Decontaminated train still overlaps MONICA test lesions")
    report = {
        "seed": seed,
        "test_evaluated": False,
        "removed_images": len(removed),
        "removed_by_class": {CLASS_NAMES[i]: required[i] for i in range(8)},
        "replacement_images": len(replacements),
        "repeated_exposures": len(repeated),
        "unique_replacements_by_class": {CLASS_NAMES[i]: sum(int(row["label"]) == i for row in replacements) for i in range(8)},
        "repeated_exposures_by_class": {CLASS_NAMES[i]: sum(int(row["label"]) == i for row in repeated) for i in range(8)},
        "final_class_counts": {CLASS_NAMES[i]: final_counts[i] for i in range(8)},
        "test_leaked_images": len(leaked_test),
        "test_clean_images": len(clean_test),
        "test_leaked_by_class": {CLASS_NAMES[i]: sum(int(row["label"]) == i for row in leaked_test) for i in range(8)},
        "manifest_sha256": {path.name: sha256(path) for path in output.glob("*.csv")},
    }
    (output / "causal_control_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    audit = {
        "protocol_version": "stage2_retrospective_contamination_v2",
        "test_predictions_evaluated": False,
        "original_monica": full_overlap_audit({"train": train, "validation": val, "test": test}),
        "decontaminated_unmatched": full_overlap_audit({"train": kept, "validation": val, "test": test}),
        "decontaminated_exposure_matched": full_overlap_audit({"train": matched, "validation": val, "test": test}),
    }
    (output.parent / "overlap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return report


def build_lesion_disjoint(root: Path, output: Path, seed: int) -> dict:
    labels = read_csv_index(root / "raw" / "ISIC_2019_Training_GroundTruth.csv")
    metadata = read_csv_index(root / "raw" / "ISIC_2019_Training_Metadata.csv")
    image_dir = root / "raw" / "ISIC_2019_Training_Input"
    groups_by_class: dict[int, dict[str, list[dict]]] = {index: defaultdict(list) for index in range(8)}
    lesion_labels: dict[str, set[int]] = defaultdict(set)
    for image_id, label_row in labels.items():
        label = official_label(label_row)
        lesion_id = metadata[image_id].get("lesion_id", "") or f"image:{image_id}"
        lesion_labels[lesion_id].add(label)
        groups_by_class[label][lesion_id].append(make_row(image_id, label, metadata[image_id], image_dir, "", "isic2019_lesion_disjoint"))
    mixed = {lesion: values for lesion, values in lesion_labels.items() if len(values) > 1}
    if mixed:
        raise RuntimeError(f"Found {len(mixed)} lesion IDs with conflicting labels")

    splits = {"train": [], "val": [], "test": []}
    for label in range(8):
        groups = list(groups_by_class[label].items())
        random.Random(seed + label).shuffle(groups)
        test_rows, groups = select_exact_groups(groups, 100)
        val_rows, groups = select_exact_groups(groups, 50)
        train_rows, groups = select_exact_groups(groups, TARGET_TRAIN[label])
        for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            for row in rows:
                row["split"] = split
            splits[split].extend(rows)
    rng = random.Random(seed)
    for rows in splits.values():
        rng.shuffle(rows)
    lesion_sets = {split: {row["lesion_id"] or f"image:{row['image_id']}" for row in rows} for split, rows in splits.items()}
    overlaps = {
        "train_val": len(lesion_sets["train"] & lesion_sets["val"]),
        "train_test": len(lesion_sets["train"] & lesion_sets["test"]),
        "val_test": len(lesion_sets["val"] & lesion_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Lesion overlap in strict protocol: {overlaps}")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_rows(output / f"{split}.csv", rows)
    report = {
        "protocol": "isic2019_lt_ir100_lesion_disjoint_v1",
        "seed": seed,
        "test_evaluated": False,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "class_counts": {split: {CLASS_NAMES[i]: sum(int(row["label"]) == i for row in rows) for i in range(8)} for split, rows in splits.items()},
        "lesion_overlap": overlaps,
        "split_units": {
            split: {
                CLASS_NAMES[label]: {
                    "images": sum(int(row["label"]) == label for row in rows),
                    "unique_lesions": len({
                        row["lesion_id"] or f"image:{row['image_id']}"
                        for row in rows if int(row["label"]) == label
                    }),
                    "images_per_lesion": (
                        sum(int(row["label"]) == label for row in rows)
                        / len({row["lesion_id"] or f"image:{row['image_id']}" for row in rows if int(row["label"]) == label})
                    ),
                }
                for label in range(8)
            }
            for split, rows in splits.items()
        },
        "image_level_imbalance_ratio_train": max(TARGET_TRAIN) / min(TARGET_TRAIN),
        "lesion_level_imbalance_ratio_train": (
            max(len({row["lesion_id"] or f"image:{row['image_id']}" for row in splits["train"] if int(row["label"]) == label}) for label in range(8))
            / min(len({row["lesion_id"] or f"image:{row['image_id']}" for row in splits["train"] if int(row["label"]) == label}) for label in range(8))
        ),
        "manifest_sha256": {path.name: sha256(path) for path in output.glob("*.csv")},
    }
    (output / "protocol_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--monica-dir", type=Path, default=Path("data_splits/monica/ir100"))
    parser.add_argument("--output", type=Path, default=Path("data_splits/stage2"))
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    causal = build_causal_control(args.root, args.monica_dir, args.output / "monica_causal", args.seed)
    strict = build_lesion_disjoint(args.root, args.output / "lesion_disjoint_ir100", args.seed)
    print(json.dumps({"causal": causal, "lesion_disjoint": strict}, indent=2))


if __name__ == "__main__":
    main()
