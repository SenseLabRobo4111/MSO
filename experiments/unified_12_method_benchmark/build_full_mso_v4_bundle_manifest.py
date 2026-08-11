#!/usr/bin/env python3
"""Build the deterministic SHA-256 manifest for the V4 remote bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


FILES = (
    "V4_PROTOCOL.md",
    "PREDICTION_OUTPUT_CONTRACT.md",
    "Dockerfile.full_mso",
    "full_mso_config_v4.json",
    "lama_dependency_sha256.tsv",
    "mso_full_components.py",
    "verify_recovered_components.py",
    "train_paper_mso_reconstructed.py",
    "train_paper_mso_reconstructed_v4.py",
    "train_unified_adapter.py",
    "unified_models.py",
    "evaluate_predictions.py",
    "aggregate_full_mso_reconstruction.py",
    "aggregate_full_mso_reconstruction_v4.py",
    "plot_full_mso_reconstruction_v4.py",
    "run_full_mso_campaign_v4_remote.sh",
    "full_mso_config.json",
    "FULL_MSO_BUNDLE_SHA256SUMS",
    "test_full_mso_v4_protocol.py",
    "build_full_mso_v4_bundle_manifest.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--output", default="FULL_MSO_V4_BUNDLE_SHA256SUMS", type=Path
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    if output.parent != root:
        raise ValueError("bundle manifest output must remain in the bundle root")
    lines = []
    for relative in FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or linked bundle input: {relative}")
        lines.append(f"{sha256(path)}  {relative}\n")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(sha256(output))


if __name__ == "__main__":
    main()
