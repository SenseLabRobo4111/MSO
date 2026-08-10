"""Freeze protocol and software digests before any held-out run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    PACKAGE_PARENT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PACKAGE_PARENT))
    from closed_loop_reconstructed.hashing import canonical_json_sha256, sha256_file
    from closed_loop_reconstructed.software_closure import (
        PROSPECTIVE_ENTRYPOINTS,
        hash_local_import_closure,
    )
    from closed_loop_reconstructed.freeze_contract import (
        validate_freeze_artifacts,
        validate_protocol_implementation_contract,
    )
    from closed_loop_reconstructed.run_campaign import verify_split_contract
    from closed_loop_reconstructed.world import split_floorplans
else:
    from .hashing import canonical_json_sha256, sha256_file
    from .software_closure import (
        PROSPECTIVE_ENTRYPOINTS,
        hash_local_import_closure,
    )
    from .freeze_contract import (
        validate_freeze_artifacts,
        validate_protocol_implementation_contract,
    )
    from .run_campaign import verify_split_contract
    from .world import split_floorplans


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
GENERATED_PREFIXES = ("FROZEN_", "SOFTWARE_")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_software_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".json", ".md", ".txt"
        }:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts or relative.startswith("results/"):
            continue
        if path.name.startswith(GENERATED_PREFIXES):
            continue
        hashes[f"closed_loop_reconstructed/{relative}"] = sha256_file(path)
    hashes.update(
        hash_local_import_closure(REPOSITORY_ROOT, PROSPECTIVE_ENTRYPOINTS)
    )
    return hashes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--selection-trace", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--v2-lock", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint_policy = protocol["checkpoint_policy"]
    if not bool(checkpoint_policy.get("geometry_frozen", False)):
        raise RuntimeError(
            "refusing to freeze while replacement-training geometry remains unresolved"
        )
    if not bool(protocol["freeze"].get("test_execution_enabled", False)):
        raise RuntimeError("refusing to freeze while held-out execution is disabled")
    validate_protocol_implementation_contract(
        protocol, require_freeze_ready=True
    )
    if (ROOT / "results" / "test").exists():
        raise RuntimeError(
            "refusing to freeze after a held-out output directory exists"
        )
    artifact_paths = {
        "checkpoint": args.checkpoint,
        "training_manifest": args.training_manifest,
        "selection_trace": args.selection_trace,
        "environment_lock": args.environment_lock,
        "v2_lock": args.v2_lock,
        "dataset_root": args.dataset_root,
        "dataset_manifest": args.dataset_manifest,
        "device": args.device,
    }
    missing_artifact_args = [
        name for name, path in artifact_paths.items() if path is None]
    if missing_artifact_args:
        raise RuntimeError(
            f"refusing to freeze without artifact inputs: {missing_artifact_args}")
    artifact_contract = validate_freeze_artifacts(
        checkpoint_path=args.checkpoint,
        training_manifest_path=args.training_manifest,
        selection_trace_path=args.selection_trace,
        environment_lock_path=args.environment_lock,
        v2_lock_path=args.v2_lock,
        dataset_root=args.dataset_root,
        dataset_manifest_path=args.dataset_manifest,
        execution_device=args.device,
        protocol=protocol,
    )
    splits = split_floorplans(
        args.source_root.resolve(),
        int(protocol["dataset"]["split_seed"]),
        readable_splits=(),
    )
    verify_split_contract(splits, protocol)
    held_out_source_hashes = {
        row.source_relpath: sha256_file(row.gt_path) for row in splits["test"]}
    protocol_sha = sha256_file(protocol_path)
    protocol_digest_path = ROOT / "FROZEN_PROTOCOL.sha256"
    protocol_digest_path.write_text(protocol_sha + "\n", encoding="ascii")
    software_path = ROOT / "SOFTWARE_SHA256SUMS.json"
    _write_json(software_path, collect_software_hashes())
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": protocol_sha,
        "software_manifest_sha256": sha256_file(software_path),
        **artifact_contract,
        "split_membership_checked_without_opening_held_out_bitmaps": True,
        "held_out_floorplans": len(splits["test"]),
        "held_out_buildings": len({row.building_id for row in splits["test"]}),
        "held_out_source_hashes": held_out_source_hashes,
        "held_out_source_manifest_sha256": canonical_json_sha256(
            held_out_source_hashes),
        "test_outputs_present_at_freeze": False,
    }
    _write_json(ROOT / "FROZEN_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
