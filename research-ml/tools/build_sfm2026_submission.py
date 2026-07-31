#!/usr/bin/env python3
"""Build reproducible figures and evidence files for the SFM 2026 submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "sfm2026"
DEFAULT_SUBMISSION = ROOT / "submissions" / "sfm2026"

COLORS = {
    "blue": "#2563A6",
    "green": "#15856F",
    "red": "#C9463D",
    "amber": "#D08A17",
    "gray": "#667085",
    "grid": "#D0D5DD",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def build_stage10_forest(source: Path, output: Path) -> list[dict[str, object]]:
    labels = {
        "strict_id": "Strict in-distribution",
        "aid_radial": "AID radial",
        "ood_far": "OOD far",
        "random_remaining": "Random remaining",
    }
    rows = [row for row in read_csv(source) if row["metric"] in {"mcc", "macro_f1"}]
    order = list(labels)
    figure_rows: list[dict[str, object]] = []
    for metric in ("mcc", "macro_f1"):
        indexed = {row["stratum"]: row for row in rows if row["metric"] == metric}
        for stratum in order:
            row = indexed[stratum]
            figure_rows.append(
                {
                    "stratum": stratum,
                    "metric": metric,
                    "mean_difference": float(row["mean_difference"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "bootstrap_replicates": int(row["replicates"]),
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metric, title in zip(axes, ("mcc", "macro_f1"), ("MCC", "Macro F1")):
        metric_rows = [row for row in figure_rows if row["metric"] == metric]
        y = np.arange(len(order))
        means = np.array([row["mean_difference"] for row in metric_rows])
        lows = np.array([row["ci95_low"] for row in metric_rows])
        highs = np.array([row["ci95_high"] for row in metric_rows])
        colors = [COLORS["green"] if low > 0 else COLORS["red"] if high < 0 else COLORS["blue"] for low, high in zip(lows, highs)]
        for yi, mean, low, high, color in zip(y, means, lows, highs, colors):
            ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=3, lw=1.5)
        ax.axvline(0, color="#111827", lw=0.9)
        ax.grid(axis="x", color=COLORS["grid"], lw=0.6, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Synthetic minus matched real replay")
        ax.set_yticks(y, [labels[item] for item in order])
        ax.invert_yaxis()
    fig.suptitle("Feature geometry moderates synthetic-data utility (validation)", y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return figure_rows


def arm_metric_map(path: Path) -> dict[str, float]:
    return {
        row["metric"]: float(row["mean_difference_synthetic_minus_replay"])
        for row in read_csv(path)
    }


def build_stage15a_decomposition(source_dir: Path, output: Path) -> list[dict[str, object]]:
    b_minus_a = arm_metric_map(source_dir / "offline_crop" / "paired_seed_comparisons.csv")
    c_minus_a = arm_metric_map(source_dir / "vae_roundtrip" / "paired_seed_comparisons.csv")
    d_minus_a = arm_metric_map(source_dir / "img2img_strength05" / "paired_seed_comparisons.csv")
    contrasts = {
        "Offline crop (B-A)": b_minus_a,
        "VAE only (C-B)": {key: c_minus_a[key] - b_minus_a[key] for key in b_minus_a},
        "One-step UNet (D-C)": {key: d_minus_a[key] - c_minus_a[key] for key in b_minus_a},
        "Complete pipeline (D-A)": d_minus_a,
    }
    metrics = {
        "auprc_ovr_macro": "Macro AUPRC",
        "macro_f1": "Macro F1",
        "mcc": "MCC",
    }
    rows = [
        {"contrast": contrast, "metric": metric, "difference": values[metric]}
        for contrast, values in contrasts.items()
        for metric in metrics
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(len(contrasts))
    width = 0.23
    palette = [COLORS["blue"], COLORS["amber"], COLORS["red"]]
    for index, (metric, label) in enumerate(metrics.items()):
        values = [contrasts[contrast][metric] for contrast in contrasts]
        ax.bar(x + (index - 1) * width, values, width, label=label, color=palette[index])
    ax.axhline(0, color="#111827", lw=0.9)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6, alpha=0.8)
    ax.set_xticks(x, list(contrasts), rotation=12, ha="right")
    ax.set_ylabel("Validation difference")
    ax.set_title("Causal decomposition of the SD1.5 img2img pipeline", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return rows


def build_generator_gate(p0_path: Path, p1_path: Path, output: Path) -> list[dict[str, object]]:
    arm_names = {
        "historical_sd15_img2img": "Historical img2img",
        "generic_sd15_mask_inpaint": "Generic inpaint",
        "domain_lora_img2img": "Domain LoRA img2img",
        "domain_lora_mask_inpaint": "Domain LoRA inpaint",
    }
    rows: list[dict[str, object]] = []
    for path in (p0_path, p1_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for arm, values in payload["arms"].items():
            rows.append(
                {
                    "arm": arm,
                    "label": arm_names[arm],
                    "label_agreement": float(values["label_agreement"]),
                    "mask_iou": float(values["mean_mask_iou"]),
                    "outside_mask_mae": float(values["mean_outside_mask_mae"]),
                    "near_duplicate_fraction": float(values["near_duplicate_fraction"]),
                    "technical_gate_passed": bool(values["technical_gate_passed"]),
                }
            )

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    for index, row in enumerate(rows):
        color = [COLORS["gray"], COLORS["amber"], COLORS["blue"], COLORS["green"]][index]
        ax.scatter(row["label_agreement"], row["mask_iou"], s=72, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(row["label"], (row["label_agreement"], row["mask_iou"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0.80, color=COLORS["red"], linestyle="--", lw=1.0, label="Preregistered gates")
    ax.axhline(0.70, color=COLORS["red"], linestyle="--", lw=1.0)
    ax.fill_between([0.80, 1.0], 0.70, 1.0, color=COLORS["green"], alpha=0.08)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.55, 0.9)
    ax.set_xlabel("Diagnosis-label agreement")
    ax.set_ylabel("Mean lesion-mask IoU")
    ax.set_title("Generator qualification: no candidate passed both gates", fontweight="bold")
    ax.grid(color=COLORS["grid"], lw=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return rows


def build_example_panel(images: list[tuple[str, Path]], output: Path) -> bool:
    if any(not path.exists() for _, path in images):
        return False
    font = ImageFont.load_default()
    panel_width = 580
    title_height = 28
    crop_height = 760
    prepared: list[Image.Image] = []
    for title, path in images:
        image = Image.open(path).convert("RGB")
        crop = image.crop((0, 0, image.width, min(crop_height, image.height)))
        scale = panel_width / crop.width
        crop = crop.resize((panel_width, int(crop.height * scale)), Image.Resampling.LANCZOS)
        titled = Image.new("RGB", (panel_width, crop.height + title_height), "white")
        draw = ImageDraw.Draw(titled)
        draw.text((8, 8), title, fill="black", font=font)
        titled.paste(crop, (0, title_height))
        prepared.append(titled)
    cell_height = max(image.height for image in prepared)
    canvas = Image.new("RGB", (panel_width * 2, cell_height * 2), "white")
    for index, image in enumerate(prepared):
        canvas.paste(image, ((index % 2) * panel_width, (index // 2) * cell_height))
    canvas.save(output, quality=92, optimize=True)
    return True


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    submission_dir = args.submission_dir.resolve()
    source_dir = report_dir / "source"
    figure_dir = report_dir / "figures"
    figure_data_dir = report_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    stage10_source = source_dir / "stage10_hierarchical_lesion_bootstrap.csv"
    stage15_source = source_dir / "stage15a"
    p0_source = source_dir / "stage16g_p0_qualification_summary.json"
    p1_source = source_dir / "stage16g_p1_qualification_summary.json"

    stage10_rows = build_stage10_forest(stage10_source, figure_dir / "figure_1_stage10_geometry_forest.png")
    stage15_rows = build_stage15a_decomposition(stage15_source, figure_dir / "figure_2_stage15a_causal_decomposition.png")
    generator_rows = build_generator_gate(p0_source, p1_source, figure_dir / "figure_3_generator_qualification.png")
    write_csv(figure_data_dir / "figure_1_stage10_geometry.csv", stage10_rows)
    write_csv(figure_data_dir / "figure_2_stage15a_decomposition.csv", stage15_rows)
    write_csv(figure_data_dir / "figure_3_generator_qualification.csv", generator_rows)

    examples_created = build_example_panel(
        [
            ("Historical SD1.5 img2img", ROOT / "artifacts" / "contact_sheet_historical_sd15_img2img.jpg"),
            ("Generic SD1.5 inpaint", ROOT / "artifacts" / "contact_sheet_generic_sd15_mask_inpaint.jpg"),
            ("Domain LoRA img2img", ROOT / "artifacts" / "stage16g_p1" / "contact_sheet_domain_lora_img2img.jpg"),
            ("Domain LoRA inpaint", ROOT / "artifacts" / "stage16g_p1" / "contact_sheet_domain_lora_mask_inpaint.jpg"),
        ],
        figure_dir / "figure_4_generator_examples_internal_audit.jpg",
    )

    abstract_path = submission_dir / "abstract_en.txt"
    abstract = abstract_path.read_text(encoding="utf-8").strip()
    word_count = count_words(abstract)
    if not 200 <= word_count <= 250:
        raise ValueError(f"SFM abstract must contain 200-250 words; got {word_count}")

    input_paths = [
        stage10_source,
        source_dir / "stage12" / "analysis_summary.json",
        stage15_source / "offline_crop" / "paired_seed_comparisons.csv",
        stage15_source / "vae_roundtrip" / "paired_seed_comparisons.csv",
        stage15_source / "img2img_strength05" / "paired_seed_comparisons.csv",
        stage15_source / "img2img_strength05" / "hierarchical_lesion_bootstrap.csv",
        p0_source,
        p1_source,
        abstract_path,
    ]
    manifest = {
        "conference": "Saratov Fall Meeting XXX (SFM 2026)",
        "abstract_word_count": word_count,
        "locked_test_evaluated": False,
        "new_training_started_for_submission": False,
        "examples_created": examples_created,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        ],
        "figures": [str(path.relative_to(ROOT)) for path in sorted(figure_dir.glob("*"))],
    }
    (report_dir / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
