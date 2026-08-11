"""Command-line runner for matched reconstructed closed-loop campaigns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

if __package__ in (None, ""):
    PACKAGE_PARENT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PACKAGE_PARENT))
    from closed_loop_reconstructed.analysis import analyze_paired_campaign
    from closed_loop_reconstructed.analysis_multiarms import analyze_multiarm_campaign
    from closed_loop_reconstructed.engine import ClosedLoopRun
    from closed_loop_reconstructed.environment_lock import (
        capture_runtime_environment,
        normalize_device,
    )
    from closed_loop_reconstructed.hashing import canonical_json_sha256, sha256_file
    from closed_loop_reconstructed.freeze_contract import (
        validate_freeze_artifacts,
        validate_protocol_implementation_contract,
    )
    from closed_loop_reconstructed.predictor import make_predictor
    from closed_loop_reconstructed.registrar import (
        IdentityFixtureRegistrar,
        ReferenceRegistrarAdapter,
    )
    from closed_loop_reconstructed.world import split_floorplans
else:
    from .analysis import analyze_paired_campaign
    from .analysis_multiarms import analyze_multiarm_campaign
    from .engine import ClosedLoopRun
    from .environment_lock import capture_runtime_environment, normalize_device
    from .hashing import canonical_json_sha256, sha256_file
    from .freeze_contract import (
        validate_freeze_artifacts,
        validate_protocol_implementation_contract,
    )
    from .predictor import make_predictor
    from .registrar import IdentityFixtureRegistrar, ReferenceRegistrarAdapter
    from .world import split_floorplans


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "protocol.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_split_contract(
    splits: dict[str, list[Any]], protocol: dict[str, Any]
) -> None:
    expected = protocol["dataset"]["expected_floorplans"]
    actual = {name: len(splits[name]) for name in ("train", "val", "test")}
    if actual != {name: int(expected[name]) for name in actual}:
        raise RuntimeError(
            f"floorplan split count mismatch: expected={expected}, actual={actual}"
        )
    test_ids = [record.floorplan_id for record in splits["test"]]
    if test_ids != sorted(protocol["dataset"]["test_floorplans"]):
        raise RuntimeError(
            "held-out floorplan membership does not match the frozen protocol"
        )
    test_buildings = len({record.building_id for record in splits["test"]})
    if test_buildings != int(protocol["dataset"]["expected_test_buildings"]):
        raise RuntimeError("held-out building count does not match the frozen protocol")


def verify_software_files(
    software_path: Path, repository_root: Path = ROOT.parent
) -> dict[str, str]:
    """Rehash every file named by the frozen software manifest."""
    manifest = _read_json(software_path)
    if not manifest:
        raise RuntimeError("software manifest is empty")
    checked: dict[str, str] = {}
    root = repository_root.resolve()
    for relative, expected in sorted(manifest.items()):
        if not isinstance(relative, str) or not relative.strip() or not isinstance(
                expected, str) or not SHA256.fullmatch(expected):
            raise RuntimeError("software manifest contains a malformed entry")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "software manifest path escapes the repository"
            ) from error
        if not path.is_file():
            raise RuntimeError(f"software file is missing after freeze: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"software changed after freezing: {relative}")
        checked[relative] = actual
    return checked


def verify_held_out_content_lock(
    records: list[Any], frozen_manifest: dict[str, Any]
) -> None:
    """Check held-out bytes only after an authorized unlock."""
    expected = frozen_manifest.get("held_out_source_hashes")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("frozen held-out content manifest is absent")
    actual = {record.source_relpath: sha256_file(record.gt_path) for record in records}
    if actual != expected:
        raise RuntimeError("held-out floorplan content changed after freezing")
    if canonical_json_sha256(actual) != frozen_manifest.get(
            "held_out_source_manifest_sha256"):
        raise RuntimeError("held-out source manifest digest is inconsistent")


def verify_test_unlock(
    *,
    protocol_path: Path,
    unlock_path: Path | None,
    output_root: Path,
    checkpoint_path: Path | None = None,
    training_manifest_path: Path | None = None,
    selection_trace_path: Path | None = None,
    environment_lock_path: Path | None = None,
    v2_lock_path: Path | None = None,
    dataset_root: Path | None = None,
    dataset_manifest_path: Path | None = None,
    execution_device: str | None = None,
) -> dict[str, Any]:
    if unlock_path is None:
        raise RuntimeError("test is sealed; an explicit --test-unlock file is required")
    frozen_protocol_path = ROOT / "FROZEN_PROTOCOL.sha256"
    software_path = ROOT / "SOFTWARE_SHA256SUMS.json"
    frozen_manifest_path = ROOT / "FROZEN_MANIFEST.json"
    for required in (frozen_protocol_path, software_path, frozen_manifest_path):
        if not required.is_file():
            raise RuntimeError(
                f"test is sealed; missing freeze artifact {required.name}"
            )
    expected_protocol = frozen_protocol_path.read_text(encoding="ascii").strip()
    actual_protocol = sha256_file(protocol_path)
    if actual_protocol != expected_protocol:
        raise RuntimeError("protocol changed after freezing")
    software_sha = sha256_file(software_path)
    frozen = _read_json(frozen_manifest_path)
    if frozen.get("protocol_sha256") != actual_protocol:
        raise RuntimeError("frozen manifest protocol digest mismatch")
    if frozen.get("software_manifest_sha256") != software_sha:
        raise RuntimeError("software manifest changed after freezing")
    verify_software_files(software_path)
    artifact_paths = {
        "checkpoint": checkpoint_path,
        "training_manifest": training_manifest_path,
        "selection_trace": selection_trace_path,
        "environment_lock": environment_lock_path,
        "v2_lock": v2_lock_path,
        "dataset_root": dataset_root,
        "dataset_manifest": dataset_manifest_path,
    }
    if execution_device is None:
        raise RuntimeError("test is sealed; an explicit execution device is required")
    missing = [name for name, path in artifact_paths.items() if path is None]
    if missing:
        raise RuntimeError(f"test is sealed; missing frozen artifact inputs: {missing}")
    artifact_contract = validate_freeze_artifacts(
        checkpoint_path=checkpoint_path,
        training_manifest_path=training_manifest_path,
        selection_trace_path=selection_trace_path,
        environment_lock_path=environment_lock_path,
        v2_lock_path=v2_lock_path,
        dataset_root=dataset_root,
        dataset_manifest_path=dataset_manifest_path,
        execution_device=execution_device,
        protocol=_read_json(protocol_path),
    )
    if any(frozen.get(key) != value for key, value in artifact_contract.items()):
        raise RuntimeError("runtime artifacts differ from the frozen manifest")
    unlock = _read_json(unlock_path)
    expected_unlock = {
        "protocol_sha256": actual_protocol,
        "software_manifest_sha256": software_sha,
        "checkpoint_sha256": artifact_contract["checkpoint_sha256"],
        "training_manifest_sha256": artifact_contract[
            "training_manifest_sha256"],
        "selection_trace_sha256": artifact_contract["selection_trace_sha256"],
        "environment_lock_sha256": artifact_contract[
            "environment_lock_sha256"],
        "v2_lock_sha256": artifact_contract["v2_lock_sha256"],
        "dataset_manifest_sha256": artifact_contract[
            "dataset_manifest_sha256"
        ],
        "dataset_file_inventory_sha256": artifact_contract[
            "dataset_file_inventory_sha256"
        ],
        "allow_test": True,
    }
    if any(unlock.get(key) != value for key, value in expected_unlock.items()):
        raise RuntimeError(
            "test unlock does not match the frozen protocol and software"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("test output must start in a new empty directory")
    freeze_time = datetime.fromisoformat(
        str(frozen["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if freeze_time > datetime.now(timezone.utc):
        raise RuntimeError("freeze timestamp is in the future")
    return frozen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--floorplan-id", action="append", default=[])
    parser.add_argument("--limit-floorplans", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=(
            "prediction_on",
            "observed_only",
            "prediction_registration_only",
            "prediction_frontier_only",
            "prediction_both",
            "oracle_completion",
            "adversarial_prediction",
        ),
    )
    parser.add_argument(
        "--predictor", choices=("fixture", "checkpoint"), default="fixture"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--selection-trace", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--v2-lock", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--registrar", choices=("reference", "identity"), default="reference"
    )
    parser.add_argument("--maximum-ticks", type=int)
    parser.add_argument("--test-unlock", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = _read_json(protocol_path)
    validate_protocol_implementation_contract(protocol)
    execution_device = normalize_device(args.device)
    output_root = args.output_root.resolve()
    frozen_manifest = None
    if args.split == "test":
        if not bool(protocol["freeze"].get("test_execution_enabled", False)):
            raise RuntimeError(
                "test is sealed while replacement-training geometry is unresolved"
            )
        if args.predictor != "checkpoint" or args.registrar != "reference":
            raise RuntimeError(
                "test requires the frozen checkpoint and reference registrar"
            )
        if args.maximum_ticks is not None or args.limit_floorplans is not None:
            raise RuntimeError("test forbids tick and floorplan-limit overrides")
        if args.floorplan_id:
            raise RuntimeError("test must run the complete frozen held-out split")
        frozen_manifest = verify_test_unlock(
            protocol_path=protocol_path,
            unlock_path=args.test_unlock,
            output_root=output_root,
            checkpoint_path=args.checkpoint,
            training_manifest_path=args.training_manifest,
            selection_trace_path=args.selection_trace,
            environment_lock_path=args.environment_lock,
            v2_lock_path=args.v2_lock,
            dataset_root=args.dataset_root,
            dataset_manifest_path=args.dataset_manifest,
            execution_device=execution_device,
        )
        readable_splits = ("test",)
    else:
        if args.test_unlock is not None:
            raise RuntimeError("--test-unlock is accepted only for the test split")
        readable_splits = (args.split,)
    if args.maximum_ticks is not None and args.maximum_ticks <= 0:
        raise RuntimeError("--maximum-ticks must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    splits = split_floorplans(
        args.source_root.resolve(),
        int(protocol["dataset"]["split_seed"]),
        readable_splits=readable_splits,
    )
    verify_split_contract(splits, protocol)
    if args.split == "test":
        verify_held_out_content_lock(splits["test"], frozen_manifest or {})
    records = splits[args.split]
    if args.floorplan_id:
        selected = set(args.floorplan_id)
        records = [record for record in records if record.floorplan_id in selected]
        missing = selected.difference(record.floorplan_id for record in records)
        if missing:
            raise RuntimeError(
                f"requested floorplans not in {args.split}: {sorted(missing)}"
            )
    if args.limit_floorplans is not None:
        if args.limit_floorplans <= 0:
            raise RuntimeError("--limit-floorplans must be positive")
        records = records[: args.limit_floorplans]
    seeds = args.seeds or [int(value) for value in protocol["randomization"]["seeds"]]
    arms = args.arms or list(protocol["arms"])
    if len(arms) != len(set(arms)):
        raise RuntimeError("arms must be unique")
    if args.split == "test":
        if seeds != [int(value) for value in protocol["randomization"]["seeds"]]:
            raise RuntimeError("test seeds must exactly match the frozen protocol")
        if arms != list(protocol["arms"]):
            raise RuntimeError("test arms must exactly match the frozen protocol")
    registrar = (
        ReferenceRegistrarAdapter()
        if args.registrar == "reference"
        else IdentityFixtureRegistrar()
    )
    runtime_snapshot = capture_runtime_environment(execution_device)
    runtime_environment_contract = {
        "execution_device": execution_device,
        "snapshot_sha256": canonical_json_sha256(runtime_snapshot),
        "snapshot": runtime_snapshot,
    }
    if args.split == "test" and frozen_manifest is not None and (
        runtime_environment_contract
        != frozen_manifest.get("runtime_environment_contract")
    ):
        raise RuntimeError("campaign runtime environment differs from the frozen lock")
    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    pair_streams: dict[tuple[str, str, int], set[str]] = {}
    pair_initial_states: dict[tuple[str, str, int], set[str]] = {}
    pair_arms: dict[tuple[str, str, int], set[str]] = {}
    for record in records:
        for seed in seeds:
            for arm in arms:
                predictor = make_predictor(
                    arm,
                    args.predictor,
                    checkpoint=args.checkpoint,
                    device=execution_device,
                    model_input_px=int(
                        protocol["simulation"]["prediction_model_input_px"]
                    ),
                )
                run_dir = output_root / f"{record.floorplan_id}__seed{seed}__{arm}"
                summary = ClosedLoopRun(
                    protocol=protocol,
                    protocol_path=protocol_path,
                    record=record,
                    split=args.split,
                    arm=arm,
                    seed=seed,
                    predictor=predictor,
                    registrar=registrar,
                    output_dir=run_dir,
                    maximum_ticks=args.maximum_ticks,
                    runtime_environment_contract=runtime_environment_contract,
                ).run()
                summaries.append(summary)
                pair_key = (record.building_id, record.floorplan_id, seed)
                pair_streams.setdefault(pair_key, set()).add(
                    str(summary["random_stream_manifest_sha256"])
                )
                pair_initial_states.setdefault(pair_key, set()).add(
                    str(summary["initial_state_sha256"])
                )
                pair_arms.setdefault(pair_key, set()).add(arm)
    complete_arm_selection = set(arms) == set(protocol["arms"])
    incomplete_pairs = [
        pair for pair, values in pair_arms.items()
        if complete_arm_selection and values != set(protocol["arms"])
    ]
    unmatched = [
        pair for pair, values in pair_streams.items()
        if len(values) != 1 or pair in incomplete_pairs
    ]
    if unmatched:
        raise AssertionError(f"arm random streams diverged for {unmatched}")
    unmatched_initial = [
        pair for pair, values in pair_initial_states.items()
        if len(values) != 1 or pair in incomplete_pairs
    ]
    if unmatched_initial:
        raise AssertionError(
            f"arm initial maps/starts/private frames diverged for {unmatched_initial}"
        )
    if complete_arm_selection:
        if set(protocol["arms"]) == {"prediction_on", "observed_only"}:
            paired_analysis = analyze_paired_campaign(summaries, protocol)
        else:
            paired_analysis = analyze_multiarm_campaign(summaries, protocol)
        analysis_status = str(paired_analysis["status"])
    else:
        paired_analysis = {
            "status": "not_estimable_incomplete_arm_selection",
            "selected_arms": arms,
            "required_arms": list(protocol["arms"]),
        }
        analysis_status = "not_estimable"
    analysis_sha256 = canonical_json_sha256(paired_analysis)
    _write_json(output_root / "building_cluster_analysis.json", paired_analysis)
    campaign = {
        "split": args.split,
        "floorplans": [record.floorplan_id for record in records],
        "seeds": seeds,
        "arms": arms,
        "protocol_sha256": sha256_file(protocol_path),
        "campaign_contract_sha256": canonical_json_sha256(
            {
                "split": args.split,
                "floorplans": [record.floorplan_id for record in records],
                "seeds": seeds,
                "arms": arms,
                "maximum_ticks": args.maximum_ticks,
            }
        ),
        "run_count": len(summaries),
        "matched_random_stream_pairs": len(pair_streams),
        "all_arm_random_streams_matched": not unmatched,
        "matched_initial_state_pairs": len(pair_initial_states),
        "all_arm_initial_states_matched": not unmatched_initial,
        "building_cluster_analysis_status": analysis_status,
        "building_cluster_analysis_sha256": analysis_sha256,
        "frozen_artifact_contract": (
            {
                key: frozen_manifest[key]
                for key in (
                    "checkpoint_sha256",
                    "training_manifest_sha256",
                    "selection_trace_sha256",
                    "environment_lock_sha256",
                    "v2_lock_sha256",
                    "dataset_manifest_sha256",
                    "dataset_file_inventory_sha256",
                    "software_manifest_sha256",
                    "held_out_source_manifest_sha256",
                )
            }
            if args.split == "test" and frozen_manifest is not None
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": summaries,
    }
    _write_json(output_root / "campaign_summary.json", campaign)
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return campaign


if __name__ == "__main__":
    main()
