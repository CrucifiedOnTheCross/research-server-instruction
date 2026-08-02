from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_config_sha256(config: dict) -> str:
    encoded = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_commit(project_root: Path) -> str:
    environment = os.environ.get("CODE_COMMIT", "").strip()
    if environment:
        return environment
    version_file = project_root / ".code-version"
    if version_file.exists() and version_file.read_text(encoding="utf-8").strip():
        return version_file.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as error:
        raise RuntimeError("CODE_COMMIT or a git checkout is required") from error


def signature_payload(
    *, commit: str, config_hash: str, train_hash: str, validation_hash: str,
    protocol_version: str, checkpoint_policy: str, epochs: int,
) -> dict[str, object]:
    return {
        "git_commit": commit,
        "resolved_config_sha256": config_hash,
        "train_manifest_sha256": train_hash,
        "validation_manifest_sha256": validation_hash,
        "protocol_version": protocol_version,
        "checkpoint_policy": checkpoint_policy,
        "epochs": epochs,
    }


def run_signature(**kwargs) -> tuple[str, dict[str, object]]:
    payload = signature_payload(**kwargs)
    return sha256_payload(payload), payload
