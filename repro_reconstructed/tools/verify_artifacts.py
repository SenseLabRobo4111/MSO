#!/usr/bin/env python3
"""Verify hashes and strict architecture loading for committed artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_reconstructed.artifacts import sha256_file, unwrap_generator_state  # noqa: E402
from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv  # noqa: E402


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, filename = line.split(maxsplit=1)
        sums[filename.lstrip("* ")] = digest
    return sums


def verify(package_root: Path = PACKAGE_ROOT) -> dict[str, object]:
    sums_path = package_root / "SHA256SUMS"
    sums = read_sums(sums_path)
    checked = []
    for relative, expected in sorted(sums.items()):
        path = (package_root / relative).resolve()
        if package_root.resolve() not in path.parents:
            raise ValueError(f"Hash entry escapes package: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")
        checked.append({"path": relative, "sha256": actual})

    provenance = json.loads(
        (package_root / "checkpoints" / "provenance.json").read_text(encoding="utf-8")
    )
    loaded = []
    for artifact in provenance["artifacts"]:
        relative = f"checkpoints/{artifact['filename']}"
        path = package_root / relative
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = unwrap_generator_state(payload)
        model = DistillMapNetDeconv(image_size=256, dim=4)
        model.load_state_dict(state, strict=True)
        count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if count != 342_771:
            raise ValueError(f"Unexpected parameter count for {relative}: {count}")
        loaded.append(
            {
                "path": relative,
                "strict_load": True,
                "trainable_parameter_count": count,
            }
        )
    return {"status": "verified", "hashes": checked, "candidate_loads": loaded}


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
