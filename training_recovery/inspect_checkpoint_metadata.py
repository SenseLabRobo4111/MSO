#!/usr/bin/env python3
"""Read metadata from trusted local Lightning checkpoints without tensor dumps."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalarize(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<depth-limit>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {
            str(key): scalarize(item, depth + 1)
            for key, item in value.items()
            if str(key) not in {"state", "params"}
        }
    if isinstance(value, (list, tuple)):
        return [scalarize(item, depth + 1) for item in value[:20]]
    return f"<{type(value).__module__}.{type(value).__name__}>"


def inspect(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
    prefixes = Counter(key.split(".", 1)[0] for key in state)
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "global_step": payload.get("global_step") if isinstance(payload, dict) else None,
        "lightning_version": payload.get("pytorch-lightning_version")
        if isinstance(payload, dict)
        else None,
        "state_tensor_count": len(state),
        "state_prefix_counts": dict(sorted(prefixes.items())),
        "hyper_parameters": scalarize(payload.get("hyper_parameters", {}))
        if isinstance(payload, dict)
        else {},
        "optimizer_states": scalarize(payload.get("optimizer_states", []))
        if isinstance(payload, dict)
        else [],
        "lr_schedulers": scalarize(payload.get("lr_schedulers", []))
        if isinstance(payload, dict)
        else [],
        "callbacks": scalarize(payload.get("callbacks", {}))
        if isinstance(payload, dict)
        else {},
        "loops": scalarize(payload.get("loops", {}))
        if isinstance(payload, dict)
        else {},
    }
    return result


if __name__ == "__main__":
    print(json.dumps([inspect(Path(name)) for name in sys.argv[1:]], indent=2))
