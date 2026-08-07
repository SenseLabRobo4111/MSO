"""Hashing and state-dictionary helpers used by reconstruction tools."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_prefixed_state(
    state: Mapping[str, torch.Tensor], prefix: str = "gen."
) -> OrderedDict[str, torch.Tensor]:
    result = OrderedDict(
        (key[len(prefix) :], value.detach().cpu())
        for key, value in state.items()
        if key.startswith(prefix)
    )
    if not result:
        raise ValueError(f"No tensors use the required prefix {prefix!r}")
    return result


def unwrap_generator_state(payload: object) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("state_dict must be a mapping")
    if any(str(key).startswith("gen.") for key in state):
        return extract_prefixed_state(state)
    tensors = OrderedDict(
        (str(key), value.detach().cpu())
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    )
    if len(tensors) != len(state):
        raise ValueError("Plain state dictionary contains non-tensor values")
    return tensors


def tensor_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a stable order."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()
