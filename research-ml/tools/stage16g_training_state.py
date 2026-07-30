from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def completed_summary(output_root: Path, max_steps: int) -> dict[str, Any] | None:
    path = output_root / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("complete") and int(summary.get("completed_steps", -1)) == max_steps:
        return summary
    return None
