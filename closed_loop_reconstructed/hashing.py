"""Canonical hashing helpers used by manifests, states, and event logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    suffixes = {".py", ".json", ".md", ".txt"}
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            relative = path.relative_to(root).as_posix()
            if relative.startswith("results/") or "__pycache__" in path.parts:
                continue
            hashes[relative] = sha256_file(path)
    return hashes
