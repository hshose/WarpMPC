"""Result serialization and runtime provenance shared by the example CLIs."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def provenance() -> dict[str, Any]:
    import jax

    return {
        "host": platform.node(),
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "versions": {
            m: importlib.metadata.version(m)
            for m in ("jax", "warp-lang", "mujoco", "mink", "daqp", "casadi", "numpy")
        },
    }


def source_revision(root: Path) -> str | None:
    """Source archives retain file hashes even when Git metadata is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
