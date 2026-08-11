#!/usr/bin/env python3
"""Create one N=4 run lock only after every collection gate passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from validate_campaign import load_yaml, resolve_artifact, sha256, validate_package


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one stable JSON object."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def prepare_run(
    *,
    run_id: str,
    operator: str,
    output_root: Path,
    campaign_path: Path,
    team_path: Path,
    arena_path: Path,
    start_poses_path: Path,
    model_path: Path,
    online_stack_path: Path,
    network_path: Path,
    pilot_path: Path,
    reference_path: Path,
) -> tuple[dict[str, Any], int]:
    """Validate all locks, then atomically establish a pre-capture skeleton."""
    validation = validate_package(
        campaign_path, team_path, arena_path, start_poses_path,
        model_path, online_stack_path, network_path, pilot_path,
        reference_path)
    if not validation.get("plan_valid") or not validation.get("collection_ready"):
        return {
            "prepared": False,
            "run_id": run_id,
            "claim_authorized": False,
            "reason": "campaign package is not collection-ready",
            "validation": validation,
        }, 2
    validation_paths = {
        "campaign": campaign_path.resolve(),
        "team": team_path.resolve(),
        "arena": arena_path.resolve(),
        "start_poses": start_poses_path.resolve(),
        "model": model_path.resolve(),
        "online_stack": online_stack_path.resolve(),
        "network": network_path.resolve(),
        "pilot": pilot_path.resolve(),
        "reference": reference_path.resolve(),
    }
    current_validation_hashes = {
        name: sha256(path) for name, path in validation_paths.items()
    }
    if validation.get("input_sha256") != current_validation_hashes:
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": "campaign lock bytes changed after validation",
        }, 2
    input_paths = {
        "campaign": campaign_path.resolve(),
        "team": team_path.resolve(),
        "arena": arena_path.resolve(),
        "start_poses": start_poses_path.resolve(),
        "model_lock": model_path.resolve(),
        "online_stack_lock": online_stack_path.resolve(),
        "network_lock": network_path.resolve(),
        "pilot_plan": pilot_path.resolve(),
        "reference": reference_path.resolve(),
    }
    validation_name_for_input = {
        "campaign": "campaign",
        "team": "team",
        "arena": "arena",
        "start_poses": "start_poses",
        "model_lock": "model",
        "online_stack_lock": "online_stack",
        "network_lock": "network",
        "pilot_plan": "pilot",
        "reference": "reference",
    }
    input_hashes = {
        name: current_validation_hashes[validation_name_for_input[name]]
        for name in input_paths
    }
    campaign = load_yaml(campaign_path)
    planned = {
        row["run_id"]: row for row in campaign.get("ordered_runs") or []
        if isinstance(row, dict) and row.get("run_id")
    }
    if run_id not in planned:
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": "run_id is not in the frozen 32-run order",
        }, 2
    if not operator.strip():
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": "operator is required",
        }, 2

    run_dir = output_root.expanduser().resolve() / run_id
    if run_dir.exists():
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": f"run directory already exists: {run_dir}",
        }, 2

    team = load_yaml(team_path)
    arena = load_yaml(arena_path)
    model = load_yaml(model_path)
    online_stack = load_yaml(online_stack_path)
    network = load_yaml(network_path)
    reference = load_yaml(reference_path)
    artifact = resolve_artifact(model.get("artifact_path"), model_path.resolve())
    if artifact is None or not artifact.is_file():
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": "verified model artifact disappeared after validation",
        }, 2
    artifact_hash = sha256(artifact)
    if artifact_hash != model.get("artifact_sha256"):
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": "model artifact bytes differ from the validated lock",
        }, 2
    planned_run = planned[run_id]
    run_lock = {
        "schema_version": "1.0",
        "campaign_id": campaign["campaign_id"],
        "run_id": run_id,
        "order": planned_run["order"],
        "block_id": planned_run["block_id"],
        "trial_id": planned_run["trial_id"],
        "predictor_mode": planned_run["predictor_mode"],
        "network_condition": planned_run["network_condition"],
        "start_pose_set_id": planned_run["start_pose_set_id"],
        "hardware_pose_assignment": planned_run["hardware_pose_assignment"],
        "isolated_robot_id": planned_run["isolated_robot_id"],
        "duration_s": planned_run["duration_s"],
        "site_id": campaign["site_id"],
        "arena_id": campaign["arena_id"],
        "team_size": 4,
        "operator": operator.strip(),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "software_commit": team["protocol_lock"]["software_commit"],
        "model_id": model["model_id"],
        "model_artifact_filename": artifact.name,
        "model_artifact_relpath": f"model/{artifact.name}",
        "model_artifact_sha256": artifact_hash,
        "model_artifact_role": model["artifact_role"],
        "model_loader_id": model["architecture_loader_id"],
        "model_input_contract": model["input_contract"],
        "model_output_contract": model["output_contract"],
        "model_selection_record_sha256": model["selection_record_sha256"],
        "model_verifier_report_sha256": model["verifier_report_sha256"],
        "model_architecture_source_sha256": model[
            "architecture_source_sha256"],
        "model_ffc_source_sha256": model["ffc_source_sha256"],
        "exact_trainable_parameter_count": model[
            "exact_trainable_parameter_count"],
        "online_stack_id": online_stack["stack_id"],
        "online_stack_repository_commit": online_stack[
            "external_repository_commit"],
        "online_stack_launch_entrypoint": online_stack["launch_entrypoint"],
        "network_lock_id": network["network_lock_id"],
        "gt_frame": arena["coordinate_frame"],
        "reference_id": reference["reference_id"],
        "input_lock_sha256": input_hashes,
        "evidence_status": "unverified_pre_capture",
        "results_status": "not_collected",
        "claim_authorized": False,
    }

    # Build beside the final directory and rename only after every write passes.
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_id}.preparing-", dir=output_root))
    try:
        locks_dir = temporary / "locks"
        locks_dir.mkdir()
        model_dir = temporary / "model"
        model_dir.mkdir()
        for directory in ("events", "evidence", "telemetry", "analysis"):
            (temporary / directory).mkdir()
        for name, path in input_paths.items():
            shutil.copy2(path, locks_dir / f"{name}{path.suffix}")
        model_copy = model_dir / artifact.name
        shutil.copy2(artifact, model_copy)
        if sha256(model_copy) != artifact_hash:
            raise RuntimeError("staged model artifact hash mismatch")
        write_json(temporary / "run_lock.json", run_lock)
        inventory = [
            f"{sha256(path)}  {path.relative_to(temporary).as_posix()}"
            for path in sorted(locks_dir.iterdir()) if path.is_file()
        ]
        inventory.append(
            f"{sha256(model_copy)}  {model_copy.relative_to(temporary).as_posix()}")
        inventory.append(
            f"{sha256(temporary / 'run_lock.json')}  run_lock.json")
        (temporary / "PRECAPTURE_SHA256SUMS").write_text(
            "\n".join(inventory) + "\n", encoding="utf-8")
        final_input_hashes = {
            name: sha256(path) for name, path in input_paths.items()
        }
        if final_input_hashes != input_hashes:
            raise RuntimeError("source lock bytes changed during staging")
        temporary.replace(run_dir)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        return {
            "prepared": False, "run_id": run_id,
            "claim_authorized": False,
            "reason": f"pre-capture staging failed: {error}",
        }, 2
    return {
        "prepared": True,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "claim_authorized": False,
        "next_required_step": "independent preflight and supervised capture",
    }, 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, default=base / "config/campaign.yaml")
    parser.add_argument("--team", type=Path, default=base / "config/team_4.yaml")
    parser.add_argument("--arena", type=Path, default=base / "config/arena.yaml")
    parser.add_argument("--start-poses", type=Path, default=base / "config/start_poses.yaml")
    parser.add_argument("--model-lock", type=Path, default=base / "config/model_lock.yaml")
    parser.add_argument(
        "--online-stack-lock", type=Path,
        default=base / "config/online_stack_lock.yaml")
    parser.add_argument(
        "--network-lock", type=Path,
        default=base / "config/network_lock.yaml")
    parser.add_argument(
        "--pilot-plan", type=Path,
        default=base / "config/pilot_plan.yaml")
    parser.add_argument("--reference", type=Path, default=base / "config/reference.yaml")
    return parser.parse_args()


def main() -> int:
    """Prepare one run or stop before filesystem mutation."""
    args = parse_args()
    report, status = prepare_run(
        run_id=args.run_id,
        operator=args.operator,
        output_root=args.output_root,
        campaign_path=args.campaign,
        team_path=args.team,
        arena_path=args.arena,
        start_poses_path=args.start_poses,
        model_path=args.model_lock,
        online_stack_path=args.online_stack_lock,
        network_path=args.network_lock,
        pilot_path=args.pilot_plan,
        reference_path=args.reference,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
