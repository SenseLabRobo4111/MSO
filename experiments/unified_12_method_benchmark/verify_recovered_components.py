#!/usr/bin/env python3
"""Verify recoverable MSO topology against the fixed historical candidate.

The output is a compact provenance contract.  No historical weights are
reused to initialise the prospective teacher, students, projections or
critics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from mso_full_components import (
    ProjectedTeacher,
    ReconstructedFFCPatchCritic,
    trainable_parameter_count,
)


STATUS = "recovered_topology_verified_weights_not_reused"
EXPECTED_CHECKPOINT_SHA256 = (
    "0e7f949925b9f6310f93e36b10152e9729a8cacef15c9061d9e361f1b7c649ad"
)
EXPECTED_TEACHER_KEYS = 1075
EXPECTED_TEACHER_PARAMETERS = 51_607_728
EXPECTED_CRITIC_KEYS = 440
EXPECTED_CRITIC_PARAMETERS = 21_936_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        descriptor = f"{name}\t{value.dtype}\t{tuple(value.shape)}\n".encode()
        digest.update(descriptor)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trusted-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    checkpoint_sha = sha256(args.trusted_checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("historical candidate SHA-256 differs from the frozen contract")

    sys.path.insert(0, str(args.source_root.resolve()))
    from sensemap.explore_model.SenseMapNet import FFCBlock, TeacherMapNet2

    payload = torch.load(args.trusted_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("historical candidate has no state_dict")
    teacher_state: dict[str, torch.Tensor] = {}
    prefixes = {
        "teacher.teacher.": "core.",
        "teacher.conv2.": "proj_conv2.",
        "teacher.conv3.": "proj_conv3.",
        "teacher.decode2.": "proj_decode2.",
        "teacher.decode3.": "proj_decode3.",
    }
    for source_name, value in state.items():
        for source_prefix, target_prefix in prefixes.items():
            if source_name.startswith(source_prefix):
                target_name = target_prefix + source_name[len(source_prefix):]
                teacher_state[target_name] = value.detach().cpu()
                break
    critic_state = {
        name[len("critic."):]: value.detach().cpu()
        for name, value in state.items()
        if name.startswith("critic.")
    }

    teacher = ProjectedTeacher(TeacherMapNet2)
    critic = ReconstructedFFCPatchCritic(FFCBlock)
    teacher.load_state_dict(teacher_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    if len(teacher_state) != EXPECTED_TEACHER_KEYS:
        raise ValueError("teacher state-key count differs from the frozen contract")
    if trainable_parameter_count(teacher) != EXPECTED_TEACHER_PARAMETERS:
        raise ValueError("teacher parameter count differs from the frozen contract")
    if len(critic_state) != EXPECTED_CRITIC_KEYS:
        raise ValueError("critic state-key count differs from the frozen contract")
    if trainable_parameter_count(critic) != EXPECTED_CRITIC_PARAMETERS:
        raise ValueError("critic parameter count differs from the frozen contract")

    with torch.inference_mode():
        teacher_output = teacher(torch.zeros(1, 3, 256, 256))
        critic_output = critic(
            torch.zeros(1, 1, 256, 256), torch.ones(1, 1, 256, 256)
        )
    expected_features = (
        (1, 8, 128, 128),
        (1, 16, 64, 64),
        (1, 16, 64, 64),
        (1, 8, 128, 128),
    )
    if tuple(tuple(value.shape) for value in teacher_output[1:]) != expected_features:
        raise ValueError("teacher feature geometry differs from the frozen contract")
    if tuple(critic_output.shape) != (1, 1, 16, 16):
        raise ValueError("critic patch geometry differs from the declared reconstruction")

    result = {
        "status": STATUS,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_epoch": int(payload.get("epoch", -1)),
        "source_checkpoint_global_step": int(payload.get("global_step", -1)),
        "teacher_state_key_count": len(teacher_state),
        "teacher_trainable_parameters": trainable_parameter_count(teacher),
        "teacher_state_fingerprint": state_fingerprint(teacher_state),
        "teacher_feature_shapes": [list(value) for value in expected_features],
        "critic_state_key_count": len(critic_state),
        "critic_trainable_parameters": trainable_parameter_count(critic),
        "critic_state_fingerprint": state_fingerprint(critic_state),
        "critic_output_shape": list(critic_output.shape),
        "historical_trainer_recovered": False,
        "historical_critic_forward_recovered": False,
        "historical_weights_reused_for_initialisation": False,
        "verifier_sha256": sha256(Path(__file__)),
        "network_source_sha256": sha256(
            args.source_root / "sensemap" / "explore_model" / "SenseMapNet.py"
        ),
        "ffc_source_sha256": sha256(
            args.source_root / "sensemap" / "explore_model" / "ffc.py"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
