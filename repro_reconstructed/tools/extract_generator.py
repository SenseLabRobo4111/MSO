#!/usr/bin/env python3
"""Extract a generator subtree from a trusted full checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_reconstructed.artifacts import (  # noqa: E402
    extract_prefixed_state,
    sha256_file,
    tensor_fingerprint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="gen.")
    parser.add_argument(
        "--trust-source",
        action="store_true",
        help="Required because full torch checkpoints may execute pickled objects.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trust_source:
        raise SystemExit("Refusing full-checkpoint load without --trust-source")
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Expected a full checkpoint with state_dict")
    state = extract_prefixed_state(payload["state_dict"], args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    result = {
        "source": args.source.name,
        "source_sha256": sha256_file(args.source),
        "output": args.output.name,
        "output_sha256": sha256_file(args.output),
        "tensor_count": len(state),
        "state_value_count": sum(value.numel() for value in state.values()),
        "tensor_fingerprint": tensor_fingerprint(state),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
