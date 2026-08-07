#!/usr/bin/env python3
"""Inspect trusted full checkpoints or weights-only candidate artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_reconstructed.artifacts import (  # noqa: E402
    sha256_file,
    tensor_fingerprint,
    unwrap_generator_state,
)
from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv  # noqa: E402


def optimizer_summary(payload: dict) -> list[dict[str, object]]:
    summaries = []
    for index, optimizer in enumerate(payload.get("optimizer_states", [])):
        groups = []
        for group in optimizer.get("param_groups", []):
            groups.append(
                {
                    key: group.get(key)
                    for key in ("lr", "betas", "eps", "weight_decay", "amsgrad")
                }
            )
        summaries.append({"optimizer_index": index, "param_groups": groups})
    return summaries


def prefix_summary(state: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for prefix in ("gen.", "teacher.", "critic.", "ResNetPL_criterion."):
        tensors = [value for key, value in state.items() if key.startswith(prefix)]
        if tensors:
            result[prefix] = {
                "tensor_count": len(tensors),
                "state_value_count": sum(value.numel() for value in tensors),
            }
    return result


def inspect(path: Path, trusted_full_checkpoint: bool) -> dict[str, object]:
    if trusted_full_checkpoint:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    state = unwrap_generator_state(payload)
    model = DistillMapNetDeconv(image_size=256, dim=4)
    model.load_state_dict(state, strict=True)
    result: dict[str, object] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "strict_architecture_load": True,
        "architecture": "DistillMapNetDeconv(image_size=256, dim=4)",
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "generator_tensor_count": len(state),
        "generator_state_value_count": sum(value.numel() for value in state.values()),
        "generator_tensor_fingerprint": tensor_fingerprint(state),
    }
    if isinstance(payload, dict) and "state_dict" in payload:
        full_state = payload["state_dict"]
        result.update(
            {
                "epoch": payload.get("epoch"),
                "global_step": payload.get("global_step"),
                "pytorch_lightning_version": payload.get("pytorch-lightning_version"),
                "state_prefixes": prefix_summary(full_state),
                "optimizer_summary": optimizer_summary(payload),
                "lr_schedulers": payload.get("lr_schedulers", []),
                "hyper_parameters": payload.get("hyper_parameters", {}),
                "callbacks": {str(key): value for key, value in payload.get("callbacks", {}).items()},
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--trusted-full-checkpoint",
        action="store_true",
        help="Allow pickle-based loading for a trusted full Lightning checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            inspect(args.checkpoint, args.trusted_full_checkpoint),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
