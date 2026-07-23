#!/usr/bin/env python
"""Validate external video2sim environments without importing their SDKs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def check_manifest(path: Path, *, check_imports: bool = False) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    required = {
        "interpreters": ("reconstruction", "nurec", "isaac"),
        "repositories": ("lingbot", "nurec"),
        "assets": ("input_video", "robot_usd"),
    }
    resolved: dict[str, dict[str, Path]] = {}
    for section, names in required.items():
        values: dict[str, Any] = manifest.get(section, {})
        resolved[section] = {}
        for name in names:
            value = values.get(name)
            if not value:
                errors.append(f"missing {section}.{name}")
                continue
            item = _resolve(path.parent, str(value))
            resolved[section][name] = item
            if not item.exists():
                errors.append(f"{section}.{name} does not exist: {item}")
    workspace = manifest.get("workspace")
    if not workspace:
        errors.append("missing workspace")
    elif not _resolve(path.parent, workspace).parent.exists():
        errors.append(f"workspace parent does not exist: {_resolve(path.parent, workspace).parent}")
    if check_imports and not errors:
        probes = {
            "reconstruction": "import torch, cv2",
            "nurec": "import torch",
            "isaac": "import isaacsim",
        }
        for name, statement in probes.items():
            result = subprocess.run(
                [str(resolved["interpreters"][name]), "-c", statement],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                errors.append(f"{name} import probe failed: {result.stderr.strip()}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    errors = check_manifest(args.manifest.resolve(), check_imports=args.check_imports)
    if errors:
        raise SystemExit("video2sim preflight failed:\n- " + "\n- ".join(errors))
    print("video2sim manifest is complete and all requested checks passed")


if __name__ == "__main__":
    main()
