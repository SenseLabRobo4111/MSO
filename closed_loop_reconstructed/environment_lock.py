"""Measured numerical-runtime lock for prospective closed-loop execution."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Callable

from .hashing import canonical_json_sha256


ENVIRONMENT_SCHEMA = "mso.closed_loop.environment/1"
DEVICE = re.compile(r"^(cpu|cuda:[0-9]+)$")


def normalize_device(value: str) -> str:
    """Return an explicit CPU or indexed CUDA device; reject aliases."""

    device = str(value).strip().lower()
    if not DEVICE.fullmatch(device):
        raise ValueError("execution device must be exactly 'cpu' or 'cuda:<index>'")
    return device


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _driver_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4 or not all(fields):
            raise RuntimeError("nvidia-smi returned a malformed device row")
        rows.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
            }
        )
    return rows


def capture_runtime_environment(execution_device: str) -> dict[str, Any]:
    """Measure the exact numerical backend and selected accelerator."""

    device = normalize_device(execution_device)
    import cv2
    import numpy as np
    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    selected_gpu: dict[str, Any] | None = None
    driver_rows = _driver_inventory()
    if device.startswith("cuda:"):
        index = int(device.split(":", maxsplit=1)[1])
        if not cuda_available or index >= cuda_count:
            raise RuntimeError(f"locked CUDA device is unavailable: {device}")
        properties = torch.cuda.get_device_properties(index)
        selected_gpu = {
            "index": index,
            "name": str(properties.name),
            "compute_capability": [int(value) for value in properties.major_minor]
            if hasattr(properties, "major_minor")
            else [int(value) for value in torch.cuda.get_device_capability(index)],
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        }
        matching = [row for row in driver_rows if row["index"] == str(index)]
        if len(matching) != 1:
            raise RuntimeError("selected CUDA device lacks a unique driver record")
        selected_gpu["uuid"] = matching[0]["uuid"]
        selected_gpu["driver_version"] = matching[0]["driver_version"]

    try:
        pillow_version = metadata.version("Pillow")
    except metadata.PackageNotFoundError:
        pillow_version = None
    snapshot = {
        "schema": ENVIRONMENT_SCHEMA,
        "execution_device": device,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": {
            "numpy": str(np.__version__),
            "opencv_python": str(cv2.__version__),
            "pillow": pillow_version,
            "torch": str(torch.__version__),
        },
        "torch_backend": {
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_count,
            "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.version() is not None
                else None
            ),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "selected_gpu": selected_gpu,
        "driver_inventory": driver_rows,
        "distributions": {
            name: _package_version(name)
            for name in ("scipy", "scikit-image", "scikit-learn")
        },
    }
    return snapshot


def make_environment_lock(execution_device: str) -> dict[str, Any]:
    snapshot = capture_runtime_environment(execution_device)
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "snapshot": snapshot,
        "snapshot_sha256": canonical_json_sha256(snapshot),
    }


def verify_environment_lock(
    path: Path,
    execution_device: str,
    *,
    snapshot_provider: Callable[[str], dict[str, Any]] = capture_runtime_environment,
) -> dict[str, Any]:
    """Require a self-consistent lock equal to a fresh runtime measurement."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != ENVIRONMENT_SCHEMA:
        raise ValueError("environment lock schema is incompatible")
    locked = value.get("snapshot")
    digest = value.get("snapshot_sha256")
    if not isinstance(locked, dict) or not locked:
        raise ValueError("environment lock snapshot is absent")
    if digest != canonical_json_sha256(locked):
        raise ValueError("environment lock snapshot digest is inconsistent")
    device = normalize_device(execution_device)
    if locked.get("execution_device") != device:
        raise ValueError("requested execution device differs from the environment lock")
    measured = snapshot_provider(device)
    if measured != locked:
        raise ValueError("current numerical environment differs from the lock")
    return {
        "execution_device": device,
        "snapshot_sha256": str(digest),
        "snapshot": locked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = make_environment_lock(args.device)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
