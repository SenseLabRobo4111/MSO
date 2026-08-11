#!/usr/bin/env python3
"""Verify an explicitly supported 304,000-parameter inference artifact.

No 304K architecture is present in this worktree, so the supported-loader
registry is intentionally empty. Adding support requires a reviewed source
change that constructs the exact architecture and performs a strict state load;
an arbitrary import string or hand-written YAML declaration is never trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable


EXPECTED_PARAMETERS = 304000
EXPECTED_INPUT_SHAPE = [1, 3, 256, 256]

# A reviewed loader must return (model, missing_keys, unexpected_keys,
# architecture_class). There is deliberately no fallback to the public
# 342,771-parameter model.
SUPPORTED_LOADERS: dict[str, Callable[[Path], tuple[Any, list[str], list[str], str]]] = {}


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
        "schema_version": "mso_model_verification_v1",
        "verified": False,
        "model_id": "mso_304k",
        "artifact_filename": artifact.name,
        "artifact_sha256": None,
        "loader_id": loader_id,
        "architecture_class": None,
        "strict_state_load": False,
        "missing_keys": None,
        "unexpected_keys": None,
        "trainable_parameter_count": None,
        "input_shape": EXPECTED_INPUT_SHAPE,
        "output_shape": None,
        "finite_forward": False,
        "fixture_sha256": None,
        "output_fingerprint": None,
        "python_version": platform.python_version(),
        "verifier_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "errors": [],
    }
    artifact = artifact.expanduser().resolve()
    fixture = fixture.expanduser().resolve()
    if not artifact.is_file():
        report["errors"].append("required model artifact does not exist")
    else:
        report["artifact_sha256"] = file_sha256(artifact)
    if not fixture.is_file():
        report["errors"].append("fixed 256x256 fixture does not exist")
    else:
        report["fixture_sha256"] = file_sha256(fixture)
    loader = SUPPORTED_LOADERS.get(loader_id)
    if loader is None:
        report["errors"].append(
            "loader_id is not explicitly supported by this verifier revision")

    if not report["errors"]:
        try:
            import numpy as np
            import torch

            model, missing, unexpected, architecture = loader(artifact)
            report["architecture_class"] = architecture
            report["missing_keys"] = list(missing)
            report["unexpected_keys"] = list(unexpected)
            report["strict_state_load"] = not missing and not unexpected
            parameters = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
            report["trainable_parameter_count"] = int(parameters)
            fixture_array = np.load(fixture, allow_pickle=False)
            if list(fixture_array.shape) == [3, 256, 256]:
                fixture_array = fixture_array[None, ...]
            if list(fixture_array.shape) != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"fixture shape {list(fixture_array.shape)} is not "
                    f"{EXPECTED_INPUT_SHAPE}")
            model.eval()
            with torch.no_grad():
                model_input = torch.as_tensor(
                    fixture_array, dtype=torch.float32)
                prediction = model(model_input)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            output = prediction.detach().cpu().numpy()
            report["output_shape"] = list(output.shape)
            report["finite_forward"] = bool(np.isfinite(output).all())
            report["output_fingerprint"] = output_fingerprint(output)
            report["torch_version"] = torch.__version__
            if not report["strict_state_load"]:
                report["errors"].append("state load was not strict")
            if parameters != EXPECTED_PARAMETERS:
                report["errors"].append(
                    f"trainable parameter count is {parameters}, expected "
                    f"{EXPECTED_PARAMETERS}")
            if not report["finite_forward"]:
                report["errors"].append("forward output contains non-finite values")
        except Exception as error:  # fail closed and preserve diagnostic evidence
            report["errors"].append(f"verification exception: {error}")

    report["verified"] = not report["errors"]
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
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
