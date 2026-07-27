from __future__ import annotations

import os

import fiftyone as fo


name = os.environ.get("FIFTYONE_DATASET", "ham10000-synthetic-audit")
if name in fo.list_datasets():
    dataset = fo.load_dataset(name)
else:
    dataset = fo.Dataset(name)
    dataset.persistent = True

session = fo.launch_app(
    dataset,
    address="0.0.0.0",
    port=int(os.environ.get("FIFTYONE_PORT", "5151")),
    remote=True,
)
session.wait()
