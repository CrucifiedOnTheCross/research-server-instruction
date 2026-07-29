from __future__ import annotations

import unittest

import pandas as pd

from tools.prepare_stage15a_variants import (
    choose_strength05_rows,
    validate_selected,
)


def selected_frame() -> pd.DataFrame:
    rows = []
    for label in ("mel", "akiec", "bkl"):
        for index in range(30):
            source = f"{label}_{index}"
            rows.append(
                {
                    "image_id": f"selected_{source}",
                    "label": label,
                    "source_image_id": source,
                    "source_image_path": f"raw/{source}.jpg",
                    "source_group_id": f"lesion_{source}",
                }
            )
    return pd.DataFrame(rows)


class Stage15AProtocolTests(unittest.TestCase):
    def test_selection_requires_unique_lesions(self) -> None:
        selected = selected_frame()
        selected.loc[1, "source_group_id"] = selected.loc[0, "source_group_id"]
        with self.assertRaisesRegex(ValueError, "source lesion"):
            validate_selected(selected)

    def test_strength05_mapping_is_exact_and_ordered(self) -> None:
        selected = selected_frame()
        rows = []
        for source in selected.to_dict("records"):
            for strength in (0.05, 0.10):
                rows.append(
                    {
                        "image_path": f"synthetic/{source['source_image_id']}_{strength}.png",
                        "image_id": f"{source['source_image_id']}_{strength}",
                        "label": source["label"],
                        "source_image_id": source["source_image_id"],
                        "source_image_path": source["source_image_path"],
                        "source_group_id": source["source_group_id"],
                        "strength": strength,
                    }
                )
        generation = pd.DataFrame(rows).sample(frac=1, random_state=42)
        chosen = choose_strength05_rows(selected, generation)
        self.assertEqual(
            chosen["source_image_id"].tolist(),
            selected["source_image_id"].tolist(),
        )
        self.assertTrue((chosen["strength"].astype(float) == 0.05).all())

    def test_strength05_mapping_rejects_duplicates(self) -> None:
        selected = selected_frame()
        generation = selected.rename(columns={"image_id": "selected_image_id"}).copy()
        generation["image_path"] = "synthetic/example.png"
        generation["image_id"] = [
            f"generated_{index}" for index in range(len(generation))
        ]
        generation["strength"] = 0.05
        generation = pd.concat([generation, generation.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "non_unique"):
            choose_strength05_rows(selected, generation)


if __name__ == "__main__":
    unittest.main()
