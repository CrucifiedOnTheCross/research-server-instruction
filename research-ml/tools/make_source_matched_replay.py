from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a real-image replay control matched one-for-one to selected "
            "synthetic samples."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--synthetic-scores-csv", required=True)
    parser.add_argument("--out-train-csv", required=True)
    parser.add_argument("--out-replay-csv", required=True)
    parser.add_argument("--sample-weight", type=float, default=0.5)
    parser.add_argument("--expected-selected", type=int, default=240)
    return parser.parse_args()


def resolve(root: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def build_source_replay(
    train: pd.DataFrame,
    scores: pd.DataFrame,
    sample_weight: float,
    expected_selected: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required_train = {"image_id", "image_path", "label", "group_id"}
    required_scores = {
        "image_id",
        "label",
        "source_image_id",
        "source_image_path",
        "source_group_id",
        "selected_by_stage6",
    }
    missing_train = required_train - set(train.columns)
    missing_scores = required_scores - set(scores.columns)
    if missing_train:
        raise ValueError(f"Train CSV is missing columns: {sorted(missing_train)}")
    if missing_scores:
        raise ValueError(f"Synthetic scores CSV is missing columns: {sorted(missing_scores)}")
    if sample_weight <= 0:
        raise ValueError("sample_weight must be positive")

    real_train = train.copy()
    if "is_synthetic" in real_train:
        real_train = real_train[real_train["is_synthetic"].astype(int) == 0].copy()
    selected = scores[scores["selected_by_stage6"].astype(int) == 1].copy()
    if expected_selected is not None and len(selected) != expected_selected:
        raise ValueError(
            f"Expected {expected_selected} selected synthetic rows, found {len(selected)}"
        )
    if real_train["image_id"].duplicated().any():
        raise ValueError("Real train CSV contains duplicate image_id values")

    source_rows = real_train.set_index("image_id", drop=False)
    replay_rows: list[pd.Series] = []
    for synthetic in selected.itertuples(index=False):
        source_id = str(synthetic.source_image_id)
        if source_id not in source_rows.index:
            raise ValueError(f"Selected synthetic source is absent from train: {source_id}")
        source = source_rows.loc[source_id].copy()
        if str(source["label"]) != str(synthetic.label):
            raise ValueError(
                f"Label mismatch for source {source_id}: "
                f"{source['label']} != {synthetic.label}"
            )
        if str(source["group_id"]) != str(synthetic.source_group_id):
            raise ValueError(
                f"Group mismatch for source {source_id}: "
                f"{source['group_id']} != {synthetic.source_group_id}"
            )
        if str(source["image_path"]) != str(synthetic.source_image_path):
            raise ValueError(
                f"Path mismatch for source {source_id}: "
                f"{source['image_path']} != {synthetic.source_image_path}"
            )
        source["image_id"] = f"replay_{source_id}_{synthetic.image_id}"
        source["is_synthetic"] = 0
        source["is_replay"] = 1
        source["sample_weight"] = float(sample_weight)
        source["replay_source_image_id"] = source_id
        source["replay_for_synthetic_image_id"] = str(synthetic.image_id)
        source["source"] = "stage9_source_matched_replay"
        replay_rows.append(source)

    replay = pd.DataFrame(replay_rows)
    base = real_train.copy()
    base["is_synthetic"] = 0
    base["is_replay"] = 0
    base["sample_weight"] = 1.0
    base["replay_source_image_id"] = ""
    base["replay_for_synthetic_image_id"] = ""
    augmented = pd.concat([base, replay], ignore_index=True, sort=False)
    if augmented["image_id"].duplicated().any():
        raise ValueError("Replay construction produced duplicate image_id values")

    metadata: dict[str, object] = {
        "real_rows": int(len(base)),
        "replay_rows": int(len(replay)),
        "augmented_rows": int(len(augmented)),
        "sample_weight": float(sample_weight),
        "replay_by_class": {
            str(label): int(count)
            for label, count in replay["label"].value_counts().sort_index().items()
        },
        "unique_replay_sources": int(replay["replay_source_image_id"].nunique()),
        "source_reuse_max": int(
            replay["replay_source_image_id"].value_counts().max()
        ),
    }
    return augmented, replay, metadata


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    train = pd.read_csv(resolve(root, args.train_csv))
    scores = pd.read_csv(resolve(root, args.synthetic_scores_csv))
    augmented, replay, metadata = build_source_replay(
        train,
        scores,
        sample_weight=float(args.sample_weight),
        expected_selected=int(args.expected_selected),
    )

    out_train = resolve(root, args.out_train_csv)
    out_replay = resolve(root, args.out_replay_csv)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_replay.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(out_train, index=False)
    replay.to_csv(out_replay, index=False)
    metadata_path = out_replay.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
