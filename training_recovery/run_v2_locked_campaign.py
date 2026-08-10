#!/usr/bin/env python3
"""The single non-degradable prospective-v2 campaign entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_v2_lock import environment_snapshot  # noqa: E402
from mso_recovery.v2_lock import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    load_lock,
    read_csv_artifact,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args()


def run_step(
    campaign_root: Path,
    source_root: Path,
    name: str,
    script: str,
    arguments: list[str],
    allow_no_go: bool = False,
) -> int:
    logs = campaign_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(source_root / script), *arguments]
    with (logs / f"{name}.stdout.log").open("wb") as stdout, (
        logs / f"{name}.stderr.log"
    ).open("wb") as stderr:
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
            env={
                **os.environ,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "0",
                "MSO_V2_TEST_DECODE_ALLOWED": "0",
            },
        )
    allowed = {0, 2} if allow_no_go else {0}
    if completed.returncode not in allowed:
        raise RuntimeError(f"Locked step failed: {name}, rc={completed.returncode}")
    atomic_write_json(
        campaign_root / "steps" / f"{name}.json",
        {
            "name": name,
            "script": script,
            "arguments": arguments,
            "returncode": completed.returncode,
            "stdout_sha256": sha256_file(logs / f"{name}.stdout.log"),
            "stderr_sha256": sha256_file(logs / f"{name}.stderr.log"),
            "test_decode_allowed": False,
        },
    )
    return completed.returncode


def verify_source_bytes_without_decode(lock_root: Path, lock: dict) -> None:
    inventory = read_csv_artifact(lock_root, lock["artifacts"]["source_inventory"])
    source_root = Path(lock["source_root"])
    for row in inventory:
        gt_path = source_root / row["gt_relpath"]
        if sha256_file(gt_path) != row["gt_sha256"]:
            raise ValueError(f"Source GT bytes changed: {row['floorplan_id']}")
        if row["metadata_relpath"]:
            metadata = source_root / row["metadata_relpath"]
            if sha256_file(metadata) != row["metadata_sha256"]:
                raise ValueError(f"Source metadata changed: {row['floorplan_id']}")


def validate_full_result(path: Path, lock_sha: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("v2_lock_sha256") != lock_sha:
        raise ValueError("Full validation is not lock-bound")
    exact = (
        result["floorplan_full_coverage_fraction"] == {"train": 1.0, "val": 1.0}
        and result["one_hot_fraction"] == 1.0
        and result["known_target_agreement"] == 1.0
        and result["cross_split_operational_group_leakage"] == 0
        and result["cross_split_source_hash_leakage"] == 0
        and result["cross_split_processed_pair_leakage"] == 0
        and result["test_images_decoded"] == 0
    )
    if not exact:
        raise RuntimeError("Full dataset failed its immutable integrity/coverage lock")
    return result


def main() -> None:
    args = parse_args()
    lock_root = args.lock_root.resolve()
    lock, lock_sha = load_lock(lock_root)
    source_root = lock_root / "source"
    if Path(__file__).resolve() != (source_root / "run_v2_locked_campaign.py").resolve():
        raise RuntimeError("Runner must execute from the immutable lock source directory")
    locked_environment = json.loads(
        (lock_root / lock["artifacts"]["environment"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    current_environment = environment_snapshot()
    if canonical_json_sha256(current_environment) != canonical_json_sha256(
        locked_environment
    ):
        raise RuntimeError("Current environment differs from the immutable v2 lock")
    verify_source_bytes_without_decode(lock_root, lock)
    campaign_root = args.campaign_root.resolve()
    if campaign_root.exists() and any(campaign_root.iterdir()):
        raise FileExistsError("Locked campaign root must be new and empty")
    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "mso.prospective_v2.campaign/1",
        "status": "RUNNING",
        "v2_lock_sha256": lock_sha,
        "runner_sha256": sha256_file(Path(__file__)),
        "environment_sha256": lock["artifacts"]["environment"]["sha256"],
        "source_inventory_sha256": lock["artifacts"]["source_inventory"]["sha256"],
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "positive_control_sha256": lock["positive_control"]["sha256"],
        "test_decode_allowed": False,
        "test_outcomes_accessed": False,
    }
    atomic_write_json(campaign_root / "campaign_manifest.json", manifest)
    lock_arg = ["--lock-root", str(lock_root)]
    smoke_root = campaign_root / "datasets" / "smoke"
    run_step(
        campaign_root,
        source_root,
        "01_generate_smoke",
        "generate_v2_locked_dataset.py",
        [*lock_arg, "--output-root", str(smoke_root), "--dataset-kind", "smoke"],
    )
    smoke_validation = campaign_root / "validation" / "smoke.json"
    run_step(
        campaign_root,
        source_root,
        "02_validate_smoke",
        "validate_v2_dataset.py",
        [
            *lock_arg,
            "--dataset-root",
            str(smoke_root),
            "--dataset-kind",
            "smoke",
            "--output",
            str(smoke_validation),
        ],
    )
    positive_output = campaign_root / "validation" / "positive_control_A.json"
    run_step(
        campaign_root,
        source_root,
        "03_positive_control_A",
        "evaluate_v2_positive_control.py",
        [
            *lock_arg,
            "--dataset-root",
            str(smoke_root),
            "--output",
            str(positive_output),
        ],
    )
    run_step(
        campaign_root,
        source_root,
        "04_smoke_teacher_101",
        "train_v2_locked.py",
        [
            *lock_arg,
            "--campaign-root",
            str(campaign_root),
            "--job-id",
            "smoke_teacher_101",
        ],
    )
    gate_output = campaign_root / "gate" / "atomic_five_gate_decision.json"
    run_step(
        campaign_root,
        source_root,
        "05_atomic_five_gate_audit",
        "audit_v2_gates.py",
        [
            *lock_arg,
            "--dataset-validation",
            str(smoke_validation),
            "--positive-control",
            str(positive_output),
            "--teacher-metrics",
            str(campaign_root / "runs" / "smoke_teacher_101" / "metrics.csv"),
            "--output",
            str(gate_output),
        ],
        allow_no_go=True,
    )
    gate = json.loads(gate_output.read_text(encoding="utf-8"))
    if gate.get("status") != "GO":
        manifest["status"] = "NO_GO_STOPPED_BEFORE_FULL_TRAINING"
        manifest["atomic_gate_sha256"] = sha256_file(gate_output)
        atomic_write_json(campaign_root / "campaign_manifest.json", manifest)
        raise SystemExit(2)
    full_root = campaign_root / "datasets" / "full"
    run_step(
        campaign_root,
        source_root,
        "06_generate_full",
        "generate_v2_locked_dataset.py",
        [*lock_arg, "--output-root", str(full_root), "--dataset-kind", "full"],
    )
    full_validation = campaign_root / "validation" / "full.json"
    run_step(
        campaign_root,
        source_root,
        "07_validate_full",
        "validate_v2_dataset.py",
        [
            *lock_arg,
            "--dataset-root",
            str(full_root),
            "--dataset-kind",
            "full",
            "--output",
            str(full_validation),
        ],
    )
    validate_full_result(full_validation, lock_sha)
    teacher_job = f"teacher_{int(lock['training']['teacher_seed'])}"
    run_step(
        campaign_root,
        source_root,
        "08_full_teacher",
        "train_v2_locked.py",
        [*lock_arg, "--campaign-root", str(campaign_root), "--job-id", teacher_job],
    )
    for index, seed in enumerate(lock["training"]["student_seeds"], start=9):
        run_step(
            campaign_root,
            source_root,
            f"{index:02d}_student_{int(seed)}",
            "train_v2_locked.py",
            [
                *lock_arg,
                "--campaign-root",
                str(campaign_root),
                "--job-id",
                f"student_{int(seed)}",
            ],
        )
    summary_output = campaign_root / "validation" / "five_seed_summary.json"
    run_step(
        campaign_root,
        source_root,
        "14_five_seed_summary",
        "summarize_v2_multiseed.py",
        [
            *lock_arg,
            "--campaign-root",
            str(campaign_root),
            "--output",
            str(summary_output),
        ],
    )
    manifest["status"] = "COMPLETE_VALIDATION_ONLY_TEST_DISABLED"
    manifest["atomic_gate_sha256"] = sha256_file(gate_output)
    manifest["full_dataset_validation_sha256"] = sha256_file(full_validation)
    manifest["five_seed_summary_sha256"] = sha256_file(summary_output)
    manifest["test_outcomes_accessed"] = False
    atomic_write_json(campaign_root / "campaign_manifest.json", manifest)


if __name__ == "__main__":
    main()
