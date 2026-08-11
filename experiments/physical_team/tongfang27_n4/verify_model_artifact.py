#!/usr/bin/env python3
"""Verify the locked 342,771-parameter MSO deployment artifact.

Only reviewed loaders are registered here.  A model name supplied by YAML is
never imported dynamically, and a mismatched state dictionary fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODEL_ID = "mso_deconv_342771_candidate_a"
LOADER_ID = "distill_map_net_deconv_raw_state_v1"
ARCHITECTURE = (
    "sensemap.explore_model.SenseMapNet."
    "DistillMapNetDeconv(image_size=256, dim=4)"
)
EXPECTED_PARAMETERS = 342771
EXPECTED_STATE_TENSORS = 594
EXPECTED_STATE_VALUES = 347690
EXPECTED_FFC_BLOCKS = 4
EXPECTED_INPUT_SHAPE = [1, 3, 256, 256]
EXPECTED_OUTPUT_SHAPE = [1, 1, 256, 256]
EXPECTED_ALL_OUTPUT_SHAPES = [
    [1, 1, 256, 256],
    [1, 8, 128, 128],
    [1, 16, 64, 64],
    [1, 16, 64, 64],
    [1, 8, 128, 128],
]
EXPECTED_ARTIFACT_SHA256 = (
    "da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8"
)
EXPECTED_FIXTURE_SHA256 = (
    "a653af98e3f3ab8e23e6a7d46c013668954fa22286ef64a8800d236a215071ef"
)
ARCHITECTURE_SOURCE = REPOSITORY_ROOT / "sensemap/explore_model/SenseMapNet.py"
FFC_SOURCE = REPOSITORY_ROOT / "sensemap/explore_model/ffc.py"


def load_deconv_raw_state(
    artifact: Path,
) -> tuple[Any, list[str], list[str], str]:
    """Strictly load the reviewed raw generator state into its exact class."""
    import torch

    from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("artifact must be a raw state-dictionary mapping")
    if "state_dict" in payload or "model_state_dict" in payload:
        raise ValueError("wrapped checkpoints are not accepted by this loader")
    model = DistillMapNetDeconv(image_size=256, dim=4)
    incompatible = model.load_state_dict(payload, strict=True)
    return (
        model,
        list(incompatible.missing_keys),
        list(incompatible.unexpected_keys),
        ARCHITECTURE,
    )


# A reviewed loader returns (model, missing_keys, unexpected_keys,
# architecture_class). There is no arbitrary-import fallback.
SUPPORTED_LOADERS: dict[
    str, Callable[[Path], tuple[Any, list[str], list[str], str]]
] = {LOADER_ID: load_deconv_raw_state}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_fingerprint(array: Any) -> str:
    """Hash a canonical float32 output including its shape."""
    import numpy as np

    value = np.asarray(array, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def verify(
    artifact: Path, loader_id: str, fixture: Path, output_path: Path
) -> tuple[dict[str, Any], int]:
    """Run strict identity and finite-forward checks or fail closed."""
    report: dict[str, Any] = {
        "schema_version": "mso_model_verification_v2",
        "verified": False,
        "model_id": MODEL_ID,
        "artifact_role": (
            "recovered_candidate_from_checkpoint_retained_in_deployment_copy"
        ),
        "historical_manuscript_checkpoint_claim": False,
        "artifact_filename": artifact.name,
        "artifact_sha256": None,
        "loader_id": loader_id,
        "architecture_class": None,
        "strict_state_load": False,
        "missing_keys": None,
        "unexpected_keys": None,
        "trainable_parameter_count": None,
        "state_tensor_count": None,
        "state_value_count": None,
        "finite_state": False,
        "ffc_block_count": None,
        "input_shape": EXPECTED_INPUT_SHAPE,
        "output_shape": None,
        "output_count": None,
        "all_output_shapes": None,
        "all_outputs_finite": False,
        "all_output_fingerprints": None,
        "output_min": None,
        "output_max": None,
        "finite_forward": False,
        "fixture_sha256": None,
        "output_fingerprint": None,
        "python_version": platform.python_version(),
        "device": "cpu",
        "verifier_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "architecture_source_sha256": file_sha256(ARCHITECTURE_SOURCE),
        "ffc_source_sha256": file_sha256(FFC_SOURCE),
        "errors": [],
    }
    artifact = artifact.expanduser().resolve()
    fixture = fixture.expanduser().resolve()
    if not artifact.is_file():
        report["errors"].append("required model artifact does not exist")
    else:
        report["artifact_sha256"] = file_sha256(artifact)
        if report["artifact_sha256"] != EXPECTED_ARTIFACT_SHA256:
            report["errors"].append(
                "artifact SHA-256 does not identify the locked candidate A")
    if not fixture.is_file():
        report["errors"].append("fixed 256x256 fixture does not exist")
    else:
        report["fixture_sha256"] = file_sha256(fixture)
        if report["fixture_sha256"] != EXPECTED_FIXTURE_SHA256:
            report["errors"].append(
                "fixture SHA-256 does not identify the locked test input")
    loader = SUPPORTED_LOADERS.get(loader_id)
    if loader is None:
        report["errors"].append(
            "loader_id is not explicitly supported by this verifier revision")

    if not report["errors"]:
        try:
            import numpy as np
            import torch

            from sensemap.explore_model.SenseMapNet import FFCBlock

            model, missing, unexpected, architecture = loader(artifact)
            report["architecture_class"] = architecture
            report["missing_keys"] = list(missing)
            report["unexpected_keys"] = list(unexpected)
            report["strict_state_load"] = not missing and not unexpected
            parameters = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
            report["trainable_parameter_count"] = int(parameters)
            state = model.state_dict()
            report["state_tensor_count"] = len(state)
            report["state_value_count"] = int(sum(
                value.numel() for value in state.values()))
            report["finite_state"] = all(
                not (torch.is_floating_point(value) or value.is_complex())
                or bool(torch.isfinite(value).all())
                for value in state.values())
            report["ffc_block_count"] = sum(
                isinstance(module, FFCBlock) for module in model.modules())
            fixture_array = np.load(fixture, allow_pickle=False)
            if list(fixture_array.shape) == [3, 256, 256]:
                fixture_array = fixture_array[None, ...]
            if list(fixture_array.shape) != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"fixture shape {list(fixture_array.shape)} is not "
                    f"{EXPECTED_INPUT_SHAPE}")
            if not np.isin(fixture_array, (0, 1)).all():
                raise ValueError("fixture must contain only one-hot 0/1 values")
            if not np.all(fixture_array.sum(axis=1) == 1):
                raise ValueError(
                    "fixture must be one-hot across occupied/unknown/free")
            model.eval()
            with torch.no_grad():
                model_input = torch.as_tensor(
                    fixture_array, dtype=torch.float32)
                prediction_group = model(model_input)
            if not isinstance(prediction_group, (tuple, list)):
                raise TypeError("model must return its prediction and four features")
            report["output_count"] = len(prediction_group)
            if not all(isinstance(value, torch.Tensor)
                       for value in prediction_group):
                raise TypeError("all five model outputs must be tensors")
            outputs = [
                value.detach().cpu().numpy() for value in prediction_group
            ]
            report["all_output_shapes"] = [
                list(value.shape) for value in outputs
            ]
            report["all_outputs_finite"] = all(
                bool(np.isfinite(value).all()) for value in outputs
            )
            report["all_output_fingerprints"] = [
                output_fingerprint(value) for value in outputs
            ]
            output = outputs[0]
            report["output_shape"] = list(output.shape)
            report["finite_forward"] = report["all_outputs_finite"]
            report["output_fingerprint"] = output_fingerprint(output)
            report["output_min"] = float(output.min())
            report["output_max"] = float(output.max())
            report["torch_version"] = torch.__version__
            if not report["strict_state_load"]:
                report["errors"].append("state load was not strict")
            if parameters != EXPECTED_PARAMETERS:
                report["errors"].append(
                    f"trainable parameter count is {parameters}, expected "
                    f"{EXPECTED_PARAMETERS}")
            if report["state_tensor_count"] != EXPECTED_STATE_TENSORS:
                report["errors"].append(
                    "state tensor count differs from the locked architecture")
            if report["state_value_count"] != EXPECTED_STATE_VALUES:
                report["errors"].append(
                    "state value count differs from the locked architecture")
            if not report["finite_state"]:
                report["errors"].append(
                    "model state contains non-finite values")
            if report["ffc_block_count"] != EXPECTED_FFC_BLOCKS:
                report["errors"].append(
                    "FFC block count differs from the locked architecture")
            if report["output_count"] != 5:
                report["errors"].append(
                    "model must return one prediction and four feature tensors")
            if report["all_output_shapes"] != EXPECTED_ALL_OUTPUT_SHAPES:
                report["errors"].append(
                    "prediction or feature shape differs from the locked contract")
            if not report["finite_forward"]:
                report["errors"].append(
                    "prediction or feature output contains non-finite values")
            if report["output_shape"] != EXPECTED_OUTPUT_SHAPE:
                report["errors"].append(
                    f"output shape is {report['output_shape']}, expected "
                    f"{EXPECTED_OUTPUT_SHAPE}")
            if report["finite_forward"] and (
                    float(output.min()) < 0.0 or float(output.max()) > 1.0):
                report["errors"].append(
                    "sigmoid occupancy output is outside [0, 1]")
        except Exception as error:  # fail closed and preserve diagnostic evidence
            report["errors"].append(f"verification exception: {error}")

    report["verified"] = not report["errors"]
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"))
    return report, 0 if report["verified"] else 2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--loader-id", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write a machine-readable report and return fail-closed status."""
    args = parse_args()
    report, status = verify(
        args.artifact, args.loader_id, args.fixture, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
