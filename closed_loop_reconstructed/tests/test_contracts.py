from __future__ import annotations

import csv
from copy import deepcopy
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from closed_loop_reconstructed.analysis import analyze_paired_campaign
from closed_loop_reconstructed.artifact_chain import (
    ARTIFACT_STATUS,
    CHECKPOINT_SCHEMA,
    COMPLETION_SCHEMA,
    LOCK_SCHEMA,
    LOCK_STATUS,
    REQUIRED_LOCKED_TRAINING_SOURCES,
    RUN_SCHEMA,
    SELECTION_SCHEMA,
    TRAINING_SCHEMA,
    audit_train_validation_dataset,
    strict_model_validation,
)
from closed_loop_reconstructed.consensus import TemporalCycleGate
from closed_loop_reconstructed.engine import (
    ClosedLoopRun,
    assess_registration_transforms,
    dependency_cascade,
    fixed_budget_coverage_endpoints,
    matched_stream_seeds,
)
from closed_loop_reconstructed.events import EventWriter
from closed_loop_reconstructed.environment_lock import ENVIRONMENT_SCHEMA
from closed_loop_reconstructed.freeze_contract import (
    validate_freeze_artifacts,
    validate_protocol_implementation_contract,
)
from closed_loop_reconstructed.freeze_protocol import (
    collect_software_hashes,
    main as freeze_main,
)
from closed_loop_reconstructed.hashing import (
    array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from closed_loop_reconstructed.mapping import (
    PersistentMeasuredMap,
    make_private_frames,
    transform_point_xy,
)
from closed_loop_reconstructed.planner import plan_measured_safe
from closed_loop_reconstructed.predictor import (
    DeterministicFixturePredictor,
    ObservedControl,
    _resize_semantic,
    validate_probability_output,
)
from closed_loop_reconstructed.registrar import (
    IdentityFixtureRegistrar,
    ReferenceRegistrarAdapter,
    reference_registrar,
    validate_se2_transform,
)
from closed_loop_reconstructed.run_campaign import (
    main as campaign_main,
    verify_software_files,
    verify_test_unlock,
)
from closed_loop_reconstructed.verify_event_log import verify_event_log
from closed_loop_reconstructed.world import FloorplanRecord


ROOT = Path(__file__).resolve().parents[1]


def test_private_frames_support_ten_robot_scaling() -> None:
    frames, canvas_size, quadrants = make_private_frames(
        (80, 120), 10, np.random.default_rng(11), margin=16
    )
    assert len(frames) == len(quadrants) == 10
    assert canvas_size == 4 * 152
    origins = {
        tuple(round(value, 6) for value in transform_point_xy(frame, (0.0, 0.0)))
        for frame in frames
    }
    assert len(origins) == 10
    corners = [(0.0, 0.0), (119.0, 0.0), (0.0, 79.0), (119.0, 79.0)]
    for frame in frames:
        for point in corners:
            x, y = transform_point_xy(frame, point)
            assert 0 <= x < canvas_size
            assert 0 <= y < canvas_size


def _protocol() -> dict:
    return json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))


def _writer(path: Path) -> EventWriter:
    return EventWriter(
        path,
        run_id="r",
        arm="prediction_on",
        split="val",
        seed=11,
        building_id="b",
        floorplan_id="f",
        protocol_sha256="p",
    )


def _write_verified_fixture(
    root: Path,
    *,
    missing_scan_robot: bool = False,
    duplicate_scan_robot: bool = False,
    bad_stage_order: bool = False,
    bad_parent: bool = False,
    include_registration: bool = True,
    detached_commit: bool = False,
) -> Path:
    path = root / "events.jsonl"
    writer = _writer(path)
    initial_sha = "1" * 64
    map_sha = "2" * 64
    writer.emit(
        "world_init",
        0,
        outputs={"initial_state": initial_sha},
        initial_state_sha256=initial_sha,
        initial_state_manifest={"persistent_map_sha256": map_sha},
        team_size=3,
    )
    scan_events = []
    for robot_id in range(2 if missing_scan_robot else 3):
        scan_events.append(writer.emit("scan", 0, robot_id=robot_id))
    if duplicate_scan_robot:
        writer.emit("scan", 0, robot_id=0)
    for robot_id in range(3):
        writer.emit("predictor", 0, robot_id=robot_id)
    if include_registration:
        event_id = "e"
        encounter = writer.emit(
            "encounter",
            0,
            event_id=event_id,
            source_robot=1,
            target_robot=0,
            outcome="triggered",
        )
        registration = writer.emit(
            "registration",
            0,
            event_id=event_id,
            source_robot=1,
            target_robot=0,
            parent_sequence=(
                int(scan_events[0]["sequence"])
                if bad_parent else int(encounter["sequence"])
            ),
        )
        gate = writer.emit(
            "gate",
            0,
            event_id=event_id,
            source_robot=1,
            target_robot=0,
            parent_sequence=int(registration["sequence"]),
        )
        consistency = writer.emit(
            "temporal_cycle_consistency",
            0,
            event_id=event_id,
            source_robot=1,
            target_robot=0,
            parent_sequence=int(gate["sequence"]),
        )
        terminal = writer.emit(
            "commit",
            0,
            event_id=event_id,
            source_robot=1,
            target_robot=0,
            parent_sequence=int(consistency["sequence"]),
            map_revision_before=0,
            map_revision_after=0,
            persistent_hash_before=("3" * 64 if detached_commit else map_sha),
            persistent_hash_after=("3" * 64 if detached_commit else map_sha),
            outcome="safe_noncommit",
        )
        writer.emit(
            "post_decision_evaluation",
            0,
            inputs={"persistent_map": map_sha},
            event_id=event_id,
            source_robot=1,
            target_robot=0,
            parent_sequence=int(terminal["sequence"]),
            persistent_map_quality={
                "persistent_map_sha256": map_sha,
                "revision": 0,
            },
        )
    if bad_stage_order:
        writer.emit("metric", 0)
    for robot_id in range(3):
        writer.emit("planner", 0, robot_id=robot_id)
    if not bad_stage_order:
        writer.emit("metric", 0)
    summary = {
        "event_count": writer.event_count + 1,
        "status": "fixture",
        "run_id": "r",
        "arm": "prediction_on",
        "split": "val",
        "seed": 11,
        "building_id": "b",
        "floorplan_id": "f",
        "ticks_completed": 0,
        "persistent_map_sha256": map_sha,
        "map_revision": 0,
        "registration": {
            "unsafe_registration_commit": 0,
            "recovered_unsafe_registration_commit": 0,
        },
        "unsafe_commit_unrecovered_count": 0,
        "run_safety_failure": False,
    }
    summary_sha = canonical_json_sha256(summary)
    writer.emit(
        "run_complete",
        0,
        outputs={"summary": summary_sha},
        map_revision_before=0,
        map_revision_after=0,
        persistent_hash_before=map_sha,
        persistent_hash_after=map_sha,
        summary_sha256=summary_sha,
    )
    writer.close()
    manifest = {
        "run_id": "r",
        "arm": "prediction_on",
        "split": "val",
        "seed": 11,
        "building_id": "b",
        "floorplan_id": "f",
        "protocol_sha256": "p",
        "initial_state_sha256": initial_sha,
        "parameters": {
            "team_size": 3,
            "prediction_period_ticks": 1,
            "planner_period_ticks": 1,
        },
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_artifact_fixture(root: Path) -> dict[str, object]:
    repository = ROOT.parent
    root.mkdir(parents=True, exist_ok=True)
    lock_root = root / "v2_lock"
    lock_root.mkdir()
    locked_source_root = lock_root / "source"
    source_records = []
    for relative in sorted(REQUIRED_LOCKED_TRAINING_SOURCES):
        suffix = relative.removeprefix("source/")
        source = (
            repository / suffix
            if suffix.startswith("sensemap/")
            else repository / "training_recovery" / suffix
        )
        target = lock_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        source_records.append({"path": relative, "sha256": sha256_file(target)})
    source_inventory = lock_root / "source_inventory.csv"
    source_inventory.write_text("floorplan_id\ntrain_floor\nval_floor\n", encoding="utf-8")
    train_source_sha = "a" * 64
    val_source_sha = "b" * 64
    source_split = lock_root / "source_split.csv"
    _write_csv(
        source_split,
        [
            {
                "floorplan_id": "train_floor",
                "split": "train",
                "gt_sha256": train_source_sha,
            },
            {
                "floorplan_id": "val_floor",
                "split": "val",
                "gt_sha256": val_source_sha,
            },
            {
                "floorplan_id": "sealed_floor",
                "split": "test",
                "gt_sha256": "c" * 64,
            },
        ],
    )
    training_environment = lock_root / "environment.json"
    training_environment.write_text("{}\n", encoding="utf-8")
    training_policy = {
        "models": {
            "student": {
                "class": (
                    "sensemap.explore_model.SenseMapNet.DistillMapNetDeconv"
                ),
                "image_size": 256,
                "dim": 4,
                "parameter_count": 342771,
            }
        },
        "student_seeds": [11, 23, 37, 53, 71],
        "checkpoint_selection": {
            "metric": "validation_unknown_bce",
            "direction": "minimize",
            "tie_break": "earliest_epoch",
            "test_used": False,
        },
    }
    geometry: dict[str, object] = {}
    lock = {
        "schema": LOCK_SCHEMA,
        "status": LOCK_STATUS,
        "geometry": geometry,
        "training": training_policy,
        "configuration_sha256": canonical_json_sha256(
            {"geometry": geometry, "training": training_policy}
        ),
        "artifacts": {
            "source_inventory": {
                "path": source_inventory.name,
                "sha256": sha256_file(source_inventory),
            },
            "source_split": {
                "path": source_split.name,
                "sha256": sha256_file(source_split),
            },
            "environment": {
                "path": training_environment.name,
                "sha256": sha256_file(training_environment),
            },
            "source_files": source_records,
        },
    }
    lock_path = lock_root / "v2_lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    lock_sha = sha256_file(lock_path)
    (lock_root / "v2_lock.sha256").write_text(
        f"{lock_sha}  v2_lock.json\n", encoding="ascii"
    )

    dataset_root = root / "dataset"
    dataset_root.mkdir()
    observation = np.zeros((256, 256, 3), dtype=np.uint8)
    observation[:, :, 1] = 255
    target = np.zeros((256, 256), dtype=np.uint8)
    dataset_rows = []
    for split, floorplan, source_sha in (
        ("train", "train_floor", train_source_sha),
        ("val", "val_floor", val_source_sha),
    ):
        paths = {
            "obs": dataset_root / f"{split}_obs.png",
            "target": dataset_root / f"{split}_target.png",
            "raw_obs": dataset_root / f"{split}_raw_obs.png",
            "raw_target": dataset_root / f"{split}_raw_target.png",
        }
        assert cv2.imwrite(str(paths["obs"]), observation)
        assert cv2.imwrite(str(paths["target"]), target)
        assert cv2.imwrite(str(paths["raw_obs"]), observation)
        assert cv2.imwrite(str(paths["raw_target"]), target)
        dataset_rows.append(
            {
                "sample_id": f"{split}_sample",
                "split": split,
                "group_key": f"{split}_group",
                "group_kind": "operational_prefix_group",
                "floorplan_id": floorplan,
                "source_sha256": source_sha,
                "obs_relpath": paths["obs"].name,
                "target_relpath": paths["target"].name,
                "obs_sha256": sha256_file(paths["obs"]),
                "target_sha256": sha256_file(paths["target"]),
                "raw_obs_relpath": paths["raw_obs"].name,
                "raw_target_relpath": paths["raw_target"].name,
                "raw_obs_sha256": sha256_file(paths["raw_obs"]),
                "raw_target_sha256": sha256_file(paths["raw_target"]),
            }
        )
    dataset_manifest = dataset_root / "manifest.csv"
    _write_csv(dataset_manifest, dataset_rows)
    dataset = audit_train_validation_dataset(
        dataset_root,
        dataset_manifest,
        source_split_path=source_split,
        source_split_sha256=sha256_file(source_split),
    )

    metrics = {11: 0.50, 23: 0.30, 37: 0.40, 53: 0.60, 71: 0.70}
    candidates = []
    canonical_candidates = []
    for seed, metric in metrics.items():
        run_dir = root / "runs" / f"student_{seed}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        checkpoint_sha = sha256_file(checkpoint)
        bindings = {
            "v2_lock_sha256": lock_sha,
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "source_split_sha256": dataset["source_split_sha256"],
            "training_configuration_sha256": lock["configuration_sha256"],
        }
        run = {
            "schema": RUN_SCHEMA,
            "status": ARTIFACT_STATUS,
            **bindings,
            "job": {"stage": "student", "dataset_kind": "full", "seed": seed},
            "train_samples": 1,
            "validation_samples": 1,
            "test_outcomes_accessed": False,
        }
        completion = {
            "schema": COMPLETION_SCHEMA,
            "status": ARTIFACT_STATUS,
            **bindings,
            "stage": "student",
            "seed": seed,
            "best_checkpoint_sha256": checkpoint_sha,
            "best_validation_unknown_bce": metric,
            "test_outcomes_accessed": False,
        }
        run_path = run_dir / "run_manifest.json"
        completion_path = run_dir / "completion.json"
        run_path.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
        completion_path.write_text(
            json.dumps(completion, sort_keys=True), encoding="utf-8"
        )
        relative_checkpoint = checkpoint.relative_to(root).as_posix()
        relative_run = run_path.relative_to(root).as_posix()
        relative_completion = completion_path.relative_to(root).as_posix()
        candidate = {
            "seed": seed,
            "stage": "student",
            "eligible": True,
            "validation_unknown_bce": metric,
            "checkpoint_path": relative_checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "run_manifest_path": relative_run,
            "run_manifest_sha256": sha256_file(run_path),
            "completion_path": relative_completion,
            "completion_sha256": sha256_file(completion_path),
        }
        candidates.append(candidate)
        canonical_candidates.append(
            {
                key: candidate[key]
                for key in (
                    "seed",
                    "stage",
                    "validation_unknown_bce",
                    "checkpoint_sha256",
                    "run_manifest_sha256",
                    "completion_sha256",
                    "checkpoint_path",
                    "run_manifest_path",
                    "completion_path",
                )
            }
        )
    universe_sha = canonical_json_sha256(
        sorted(canonical_candidates, key=lambda item: item["seed"])
    )
    training_manifest = {
        "schema": TRAINING_SCHEMA,
        "status": ARTIFACT_STATUS,
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "source_split_sha256": dataset["source_split_sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "dataset_file_inventory_sha256": dataset["file_inventory_sha256"],
        "dataset_file_count": dataset["file_count"],
        "dataset_sample_counts": dataset["sample_counts"],
        "dataset_floorplan_counts": dataset["floorplan_counts"],
        "test_outcomes_accessed": False,
        "candidate_universe_complete": True,
        "candidate_universe_sha256": universe_sha,
        "candidates": candidates,
    }
    training_manifest_path = root / "training_manifest.json"
    training_manifest_path.write_text(
        json.dumps(training_manifest, sort_keys=True), encoding="utf-8"
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["validation_unknown_bce"],
            item["seed"],
            item["checkpoint_sha256"],
        ),
    )
    winner = ranked[0]
    selection = {
        "schema": SELECTION_SCHEMA,
        "status": ARTIFACT_STATUS,
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "candidate_universe_sha256": universe_sha,
        "candidate_universe_complete": True,
        "selection_split": "val",
        "selection_metric": "validation_unknown_bce",
        "selection_direction": "minimize",
        "tie_break": "seed_then_checkpoint_sha256",
        "test_outcomes_accessed": False,
        "ranking": [item["checkpoint_sha256"] for item in ranked],
        "selected_checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_checkpoint_seed": winner["seed"],
        "selected_validation_metric_value": winner["validation_unknown_bce"],
        "selected_run_manifest_sha256": winner["run_manifest_sha256"],
        "selected_completion_sha256": winner["completion_sha256"],
    }
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    snapshot = {
        "schema": ENVIRONMENT_SCHEMA,
        "execution_device": "cpu",
        "fixture_runtime": "validation_only",
    }
    environment = {
        "schema": ENVIRONMENT_SCHEMA,
        "snapshot": snapshot,
        "snapshot_sha256": canonical_json_sha256(snapshot),
    }
    environment_path = root / "environment.lock.json"
    environment_path.write_text(
        json.dumps(environment, sort_keys=True), encoding="utf-8"
    )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "status": ARTIFACT_STATUS,
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "source_split_sha256": dataset["source_split_sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "stage": "student",
        "seed": winner["seed"],
        "epoch": 20,
        "selected_validation_metric": winner["validation_unknown_bce"],
        "selected_epoch": 17,
        "parent_teacher_checkpoint_sha256": "d" * 64,
        "model_state_dict": {"fake": np.asarray([1.0])},
        "optimizer_state_dict": {"state": {1: {}}},
    }
    return {
        "checkpoint": root / winner["checkpoint_path"],
        "payload": payload,
        "training_manifest": training_manifest_path,
        "selection": selection_path,
        "environment": environment_path,
        "snapshot": snapshot,
        "v2_lock": lock_path,
        "dataset_root": dataset_root,
        "dataset_manifest": dataset_manifest,
        "validation_observation": dataset_root / "val_obs.png",
    }


def _summary(
    building: str,
    floorplan: str,
    seed: int,
    arm: str,
    *,
    coverage: float,
    distance: float,
    censored: bool = False,
    attempt_rate: float | None = 0.8,
) -> dict:
    return {
        "building_id": building,
        "floorplan_id": floorplan,
        "seed": seed,
        "arm": arm,
        "coverage_auc_normalized": coverage,
        "coverage_at_budget": coverage,
        "restricted_distance_to_80_m": distance,
        "distance_to_80_percent_m": None if censored else distance,
        "distance_to_80_percent_censored": censored,
        "collision_count": 1,
        "collision_rate_per_motion_attempt": 0.02,
        "motion": {"attempted": 50, "no_route": 2, "stuck_episode_count": 0},
        "planning": {"replans": 5},
        "motion_attempt_rate_per_robot_tick": attempt_rate,
        "cumulative_team_distance_m": 100.0,
        "gate_wrong_accept_rate": None,
        "final_decision_false_accept_rate": None,
        "final_decision_false_reject_count": 0,
        "unsafe_commit_unrecovered_count": 0,
    }


def test_array_hash_covers_shape_dtype_and_payload() -> None:
    base = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert array_sha256(base) == array_sha256(base.copy())
    assert array_sha256(base) != array_sha256(base.reshape(4, 3))
    assert array_sha256(base) != array_sha256(base.astype(np.uint16))
    changed = base.copy()
    changed[0, 0] += 1
    assert array_sha256(base) != array_sha256(changed)


def test_event_chain_and_reject_hash_invariant(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = _writer(path)
    first = writer.emit("scan", 0, inputs={"a": "1"}, outputs={"b": "2"})
    second = writer.emit(
        "commit",
        0,
        map_revision_before=0,
        map_revision_after=0,
        persistent_hash_before="same",
        persistent_hash_after="same",
        outcome="rejected",
    )
    writer.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["previous_event_sha256"] == first["event_sha256"]
    for row in rows:
        digest = row.pop("event_sha256")
        assert canonical_json_sha256(row) == digest
    assert second["persistent_hash_before"] == second["persistent_hash_after"]


def test_event_writer_rejects_mutation_and_invalid_parent(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "bad.jsonl")
    with pytest.raises(AssertionError):
        writer.emit(
            "commit",
            0,
            map_revision_before=0,
            map_revision_after=1,
            persistent_hash_before="before",
            persistent_hash_after="after",
            outcome="rejected",
        )
    with pytest.raises(ValueError, match="earlier"):
        writer.emit("scan", 0, parent_sequence=0)
    writer.close()


def test_event_verifier_checks_complete_robot_and_transaction_coverage(
    tmp_path: Path,
) -> None:
    path = _write_verified_fixture(tmp_path)
    result = verify_event_log(path)
    assert result["robot_coverage_verified"]
    assert result["registration_transactions_verified"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"missing_scan_robot": True}, "scan robot coverage"),
        ({"duplicate_scan_robot": True}, "scan robot coverage"),
        ({"bad_stage_order": True}, "stage order"),
        ({"bad_parent": True}, "parent stage"),
        ({"detached_commit": True}, "global chain"),
    ],
)
def test_event_verifier_rejects_adversarial_logs(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    path = _write_verified_fixture(tmp_path, **kwargs)
    with pytest.raises(AssertionError, match=message):
        verify_event_log(path)


def test_event_verifier_rejects_summary_hash_drift(tmp_path: Path) -> None:
    path = _write_verified_fixture(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["changed_after_log"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(AssertionError, match="exact summary"):
        verify_event_log(path)


def test_atomic_commit_rebuild_and_reject_are_authoritative() -> None:
    target = np.full((8, 8), 100, dtype=np.uint8)
    source = target.copy()
    source[2, 3] = 0
    source[4, 5] = 255
    state = PersistentMeasuredMap(target)
    rejected = state.apply(source, accepted=False)
    assert rejected["changed_cells"] == 0
    assert rejected["hash_before"] == rejected["hash_after"]
    committed = state.apply(source, accepted=True)
    assert committed["committed"] and committed["changed_cells"] == 2
    rebuilt = target.copy()
    recovery = state.replace_with_rebuild(rebuilt)
    assert recovery["changed_cells"] == 2
    assert np.array_equal(state.grid, rebuilt)


def test_random_streams_are_matched_and_only_declared_when_used() -> None:
    streams = ["team_starts", "private_frames"]
    left = matched_stream_seeds("p", "floor", 23, streams)
    right = matched_stream_seeds("p", "floor", 23, streams)
    assert left == right
    assert len(set(left.values())) == len(streams)
    assert _protocol()["randomization"]["streams"] == streams


def test_temporal_consensus_requires_fresh_three_sample_cycle_recheck() -> None:
    gate = TemporalCycleGate(resolution_m=0.04, consensus_count=3)
    transform = np.eye(3)
    assert not gate.observe((0, 1), transform).ready
    assert not gate.observe((0, 1), transform).ready
    assert gate.observe((0, 1), transform).ready
    bad = transform.copy()
    bad[0, 2] = 20
    first = gate.observe((0, 1), bad, existing_transform=transform)
    second = gate.observe((0, 1), bad, existing_transform=transform)
    third = gate.observe((0, 1), bad, existing_transform=transform)
    assert first.reason == "temporal_inconsistent_reset"
    assert second.reason == "awaiting_cycle_recheck_consensus"
    assert not third.accepted and third.ready
    assert third.reason == "cycle_inconsistent_consensus"


def test_n3_cascade_uses_actual_root_transform_not_pairwise_only() -> None:
    identity = np.eye(3)
    wrong_target_root = identity.copy()
    wrong_target_root[0, 2] = 10.0
    result = assess_registration_transforms(
        relative_estimate=identity,
        relative_truth=identity,
        target_root_estimate=wrong_target_root,
        target_root_truth=identity,
        source_root_actual=wrong_target_root,
        source_root_truth=identity,
        resolution_m=0.04,
        translation_limit_m=0.25,
        yaw_limit_deg=5.0,
    )
    assert result["relative"]["within_limits"] is True
    assert result["actual_source_root"]["within_limits"] is False
    assert result["cascade_from_incorrect_target_root"] is True


def test_dependency_cascade_invalidates_every_descendant() -> None:
    assert dependency_cascade({0: None, 1: 0, 2: 1}, 1) == [1, 2]
    with pytest.raises(ValueError, match="cycle"):
        dependency_cascade({0: None, 1: 2, 2: 1}, 1)


def test_fixed_budget_interpolates_without_clamping_post_budget_value() -> None:
    result = fixed_budget_coverage_endpoints(
        [(0.0, 0.1), (179.0, 0.4), (181.0, 1.0)], budget_m=180.0
    )
    assert result["coverage_at_budget"] == pytest.approx(0.7)
    assert result["distance_to_80_percent_m"] is None
    assert result["distance_to_80_percent_censored"] is True
    assert result["restricted_distance_to_80_m"] == 180.0
    expected_auc = (179.0 * 0.25 + 1.0 * 0.55) / 180.0
    assert result["coverage_auc_normalized"] == pytest.approx(expected_auc)


def test_fixed_budget_interpolates_threshold_and_keeps_latest_duplicate() -> None:
    result = fixed_budget_coverage_endpoints(
        [(0.0, 0.2), (0.0, 0.1), (100.0, 0.7), (200.0, 0.9)],
        budget_m=180.0,
    )
    assert result["coverage_at_budget"] == pytest.approx(0.86)
    assert result["distance_to_80_percent_m"] == pytest.approx(150.0)
    assert result["distance_to_80_percent_censored"] is False


def test_building_cluster_analysis_is_paired_seeded_and_multiplicity_locked() -> None:
    protocol = _protocol()
    protocol["statistics"]["bootstrap_replicates"] = 64
    protocol["statistics"]["randomization_replicates"] = 64
    rows = []
    for index, building in enumerate(("b1", "b2")):
        rate = None if index else 0.8
        rows.extend(
            [
                _summary(
                    building,
                    f"f{index}",
                    11,
                    "prediction_on",
                    coverage=0.6 + index * 0.1,
                    distance=170.0 if index == 1 else 100.0,
                    attempt_rate=rate,
                ),
                _summary(
                    building,
                    f"f{index}",
                    11,
                    "observed_only",
                    coverage=0.5 + index * 0.1,
                    distance=180.0 if index == 1 else 110.0,
                    censored=index == 1,
                    attempt_rate=rate,
                ),
            ]
        )
    first = analyze_paired_campaign(rows, protocol)
    second = analyze_paired_campaign(rows, protocol)
    assert first == second
    assert first["independent_building_count"] == 2
    assert first["audit_dataset_eligible"] is False
    assert first["benefit_claim_authorized"] is False
    assert first["audit_eligibility_checks"]["frozen_protocol"] is False
    coverage = first["primary_endpoints"]["coverage_auc_normalized"]
    distance = first["primary_endpoints"]["restricted_distance_to_80_m"]
    assert coverage["estimate"] == pytest.approx(0.1)
    assert distance["estimate"] == pytest.approx(10.0)
    assert "holm_adjusted_p_value" in coverage
    attempt = first["joint_safety_descriptive"][
        "motion_attempt_rate_per_robot_tick"
    ]
    assert attempt["defined_pair_count"] == 1
    assert attempt["undefined_pair_count"] == 1
    unsafe_rows = deepcopy(rows)
    unsafe_rows[0]["run_safety_failure"] = True
    unsafe = analyze_paired_campaign(unsafe_rows, protocol)
    assert unsafe["audit_dataset_eligible"] is False
    assert unsafe["benefit_claim_authorized"] is False
    assert unsafe["unrecovered_safety_failure_runs"]


def test_building_cluster_analysis_rejects_incomplete_pairs() -> None:
    row = _summary(
        "b", "f", 11, "prediction_on", coverage=0.5, distance=180.0
    )
    with pytest.raises(ValueError, match="incomplete paired keys"):
        analyze_paired_campaign([row], _protocol())


def test_building_cluster_analysis_rejects_inconsistent_censoring() -> None:
    prediction = _summary(
        "b", "f", 11, "prediction_on", coverage=0.5, distance=100.0,
        censored=True,
    )
    observed = _summary(
        "b", "f", 11, "observed_only", coverage=0.5, distance=110.0
    )
    with pytest.raises(ValueError, match="restriction point"):
        analyze_paired_campaign([prediction, observed], _protocol())


def test_audit_never_authorizes_benefit_when_both_primary_effects_worsen() -> None:
    protocol = _protocol()
    protocol["statistics"]["bootstrap_replicates"] = 32
    protocol["statistics"]["randomization_replicates"] = 32
    protocol["freeze"]["current_protocol_state"] = "ready_for_freeze"
    protocol["freeze"]["test_execution_enabled"] = True
    rows = []
    for building in ("b1", "b2"):
        prediction = _summary(
            building,
            f"{building}_floor",
            11,
            "prediction_on",
            coverage=0.30,
            distance=170.0,
        )
        observed = _summary(
            building,
            f"{building}_floor",
            11,
            "observed_only",
            coverage=0.70,
            distance=90.0,
        )
        prediction["split"] = "test"
        observed["split"] = "test"
        rows.extend((prediction, observed))
    result = analyze_paired_campaign(rows, protocol)
    assert result["audit_dataset_eligible"] is True
    assert result["benefit_claim_authorized"] is False
    assert all(
        endpoint["building_weighted_mean_difference"] < 0.0
        for endpoint in result["primary_endpoints"].values()
    )


def test_gate_and_planner_apis_have_no_truth_input() -> None:
    registration_parameters = set(
        inspect.signature(ReferenceRegistrarAdapter.register).parameters
    )
    assert "world" not in registration_parameters
    assert "truth" not in registration_parameters
    measured = np.full((21, 21), 100, dtype=np.uint8)
    measured[9:12, 4:17] = 0
    rank = np.full(measured.shape, 0.5, dtype=np.float32)
    first = plan_measured_safe(
        measured,
        (10, 10),
        resolution_m=0.04,
        inflation_cells=0,
        rank_probability=rank,
        rank_radius_cells=1,
        prediction_rank_weight_cells=1.0,
    )
    second = plan_measured_safe(
        measured,
        (10, 10),
        resolution_m=0.04,
        inflation_cells=0,
        rank_probability=rank,
        rank_radius_cells=1,
        prediction_rank_weight_cells=1.0,
    )
    assert first == second


def test_predictor_output_contract_rejects_shape_nan_and_range() -> None:
    assert validate_probability_output(np.full((2, 3), 0.5), (2, 3)).shape == (
        2,
        3,
    )
    with pytest.raises(ValueError, match="shape"):
        validate_probability_output(np.full((3, 2), 0.5), (2, 3))
    with pytest.raises(ValueError, match="non-finite"):
        validate_probability_output(np.asarray([[np.nan]]), (1, 1))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probability_output(np.asarray([[1.1]]), (1, 1))


def test_prediction_geometry_uses_nearest_and_preserves_measured_cells() -> None:
    semantic = np.array([[0, 100], [255, 0]], dtype=np.uint8)
    resized = _resize_semantic(semantic, 7)
    assert set(np.unique(resized)).issubset({0, 100, 255})
    world_crop = np.full((31, 47), 100, dtype=np.uint8)
    world_crop[0, 0] = 0
    world_crop[-1, -1] = 255
    prediction = ObservedControl().predict(world_crop)
    assert prediction.shape == world_crop.shape
    assert prediction[0, 0] == 0.0 and prediction[-1, -1] == 1.0


def _registration_call(adapter: ReferenceRegistrarAdapter):
    raster = np.zeros((8, 8), dtype=np.uint8)
    return adapter.register(
        event_id="e",
        target_input=raster,
        source_input=raster,
        target_observed=raster,
        source_observed=raster,
        target_origin_xy=(0, 0),
        source_origin_xy=(0, 0),
    )


def test_registrar_fails_closed_for_none_invalid_and_nonrigid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ReferenceRegistrarAdapter()
    monkeypatch.setattr(reference_registrar, "register", lambda **_kwargs: None)
    assert _registration_call(adapter).gate_accepted is False
    bad = {
        "candidate_returned": True,
        "accepted": True,
        "rigid_valid": True,
        "H_i_from_j_pixel": [[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]],
    }
    monkeypatch.setattr(reference_registrar, "register", lambda **_kwargs: bad)
    assert _registration_call(adapter).reject_reason == "invalid_transform_contract"
    nonrigid = dict(bad)
    nonrigid["H_i_from_j_pixel"] = np.eye(3).tolist()
    nonrigid["rigid_valid"] = False
    monkeypatch.setattr(reference_registrar, "register", lambda **_kwargs: nonrigid)
    assert _registration_call(adapter).reject_reason == "rigid_valid_not_true"
    valid = dict(nonrigid)
    valid["rigid_valid"] = True
    monkeypatch.setattr(reference_registrar, "register", lambda **_kwargs: valid)
    accepted = _registration_call(adapter)
    assert accepted.candidate_returned is True
    assert accepted.gate_accepted is True


def test_se2_contract_rejects_reflection_and_singular_transform() -> None:
    assert np.array_equal(validate_se2_transform(np.eye(3)), np.eye(3))
    reflection = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="rigid"):
        validate_se2_transform(reflection)
    singular = np.zeros((3, 3), dtype=float)
    with pytest.raises(ValueError):
        validate_se2_transform(singular)


def test_software_manifest_rehash_detects_post_manifest_mutation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    artifact = repository / "module.py"
    artifact.write_text("value = 1\n", encoding="utf-8")
    manifest = tmp_path / "software.json"
    manifest.write_text(
        json.dumps({"module.py": sha256_file(artifact)}), encoding="utf-8"
    )
    assert verify_software_files(manifest, repository)["module.py"] == sha256_file(
        artifact
    )
    artifact.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        verify_software_files(manifest, repository)


def _validate_artifact_fixture(fixture: dict[str, object]) -> dict:
    return validate_freeze_artifacts(
        checkpoint_path=fixture["checkpoint"],
        training_manifest_path=fixture["training_manifest"],
        selection_trace_path=fixture["selection"],
        environment_lock_path=fixture["environment"],
        v2_lock_path=fixture["v2_lock"],
        dataset_root=fixture["dataset_root"],
        dataset_manifest_path=fixture["dataset_manifest"],
        execution_device="cpu",
        protocol=_protocol(),
        checkpoint_loader=lambda _path: fixture["payload"],
        environment_snapshot_provider=lambda _device: fixture["snapshot"],
        model_validator=lambda *_args: {
            "strict_state_dict_load": True,
            "validation_fixture_only": True,
        },
    )


def test_software_closure_includes_transitive_model_and_v2_stack() -> None:
    hashes = collect_software_hashes()
    required = {
        "sensemap/explore_model/ffc.py",
        "sensemap/explore_model/SenseMapNet.py",
        "training_recovery/train_v2_locked.py",
        "training_recovery/run_v2_locked_campaign.py",
        "training_recovery/generate_v2_locked_dataset.py",
        "training_recovery/mso_recovery/v2_lock.py",
        "training_recovery/mso_recovery/v2_data.py",
        "training_recovery/mso_recovery/objective.py",
    }
    assert required.issubset(hashes)


def test_checkpoint_training_selection_environment_and_dataset_are_closed(
    tmp_path: Path,
) -> None:
    fixture = _make_artifact_fixture(tmp_path)
    result = _validate_artifact_fixture(fixture)
    assert result["checkpoint_sha256"] == sha256_file(fixture["checkpoint"])
    assert result["runtime_environment_contract"]["execution_device"] == "cpu"
    assert result["selection_contract"]["candidate_count"] == 5
    assert result["checkpoint_contract"]["seed"] == 23


def test_selection_rejects_wrong_ranking_and_incomplete_universe(
    tmp_path: Path,
) -> None:
    wrong_rank = _make_artifact_fixture(tmp_path / "wrong_rank")
    selection_path = wrong_rank["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["ranking"] = list(reversed(selection["ranking"]))
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="ranking"):
        _validate_artifact_fixture(wrong_rank)

    incomplete = _make_artifact_fixture(tmp_path / "incomplete")
    training_path = incomplete["training_manifest"]
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["candidates"].pop()
    training_path.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="universe is incomplete"):
        _validate_artifact_fixture(incomplete)


def test_dataset_and_live_environment_are_remeasured(tmp_path: Path) -> None:
    changed_data = _make_artifact_fixture(tmp_path / "data")
    observation = changed_data["dataset_root"] / "train_obs.png"
    observation.write_bytes(observation.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="dataset file changed"):
        _validate_artifact_fixture(changed_data)

    changed_environment = _make_artifact_fixture(tmp_path / "environment")
    with pytest.raises(ValueError, match="numerical environment"):
        validate_freeze_artifacts(
            checkpoint_path=changed_environment["checkpoint"],
            training_manifest_path=changed_environment["training_manifest"],
            selection_trace_path=changed_environment["selection"],
            environment_lock_path=changed_environment["environment"],
            v2_lock_path=changed_environment["v2_lock"],
            dataset_root=changed_environment["dataset_root"],
            dataset_manifest_path=changed_environment["dataset_manifest"],
            execution_device="cpu",
            protocol=_protocol(),
            checkpoint_loader=lambda _path: changed_environment["payload"],
            environment_snapshot_provider=lambda _device: {
                **changed_environment["snapshot"],
                "fixture_runtime": "different",
            },
            model_validator=lambda *_args: {},
        )


def test_fake_state_dict_cannot_pass_runtime_model_validation(tmp_path: Path) -> None:
    fixture = _make_artifact_fixture(tmp_path)
    with pytest.raises(ValueError, match="strictly"):
        strict_model_validation(
            fixture["payload"], fixture["validation_observation"], "cpu"
        )


def test_protocol_implementation_contract_refuses_unimplemented_fields() -> None:
    protocol = _protocol()
    validate_protocol_implementation_contract(protocol)
    with pytest.raises(ValueError, match="ready for freeze"):
        validate_protocol_implementation_contract(protocol, require_freeze_ready=True)
    changed = deepcopy(protocol)
    changed["randomization"]["streams"].append("unused_stream")
    with pytest.raises(ValueError, match="random streams"):
        validate_protocol_implementation_contract(changed)


def test_minimal_validation_engine_run_closes_current_contract(tmp_path: Path) -> None:
    image = np.full((40, 40), 255, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    image[15:25, 20] = 0
    floorplan = tmp_path / "fixture_GT.bmp"
    assert cv2.imwrite(str(floorplan), image)
    record = FloorplanRecord(
        floorplan_id="validation_fixture",
        building_id="validation_building",
        split="val",
        gt_path=floorplan,
        source_relpath=floorplan.name,
        source_sha256=sha256_file(floorplan),
        source_resolution_m=0.04,
    )
    run_dir = tmp_path / "run"
    summary = ClosedLoopRun(
        protocol=_protocol(),
        protocol_path=ROOT / "protocol.json",
        record=record,
        split="val",
        arm="prediction_on",
        seed=11,
        predictor=DeterministicFixturePredictor(),
        registrar=IdentityFixtureRegistrar(),
        output_dir=run_dir,
        maximum_ticks=0,
    ).run()
    verification = verify_event_log(run_dir / "events.jsonl")
    assert verification["event_count"] == summary["event_count"]
    assert "persistent_map_quality_final" in summary
    assert summary["unsafe_commit_unrecovered_count"] == 0


def test_test_split_is_sealed_without_unlock(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sealed"):
        verify_test_unlock(
            protocol_path=ROOT / "protocol.json",
            unlock_path=None,
            output_root=tmp_path / "sealed_output",
        )


def test_draft_protocol_blocks_held_out_campaign_before_source_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="geometry is unresolved"):
        campaign_main(
            [
                "--source-root",
                str(tmp_path / "must_not_be_opened"),
                "--output-root",
                str(tmp_path / "sealed_output"),
                "--split",
                "test",
                "--predictor",
                "checkpoint",
                "--registrar",
                "reference",
            ]
        )


def test_draft_protocol_blocks_freeze_before_source_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="geometry remains unresolved"):
        freeze_main(["--source-root", str(tmp_path / "must_not_be_opened")])


def test_protocol_and_package_claims_remain_nonphysical_and_disabled() -> None:
    protocol = _protocol()
    assert protocol["freeze"]["current_protocol_state"] == "draft_validation_only"
    assert protocol["freeze"]["test_execution_enabled"] is False
    assert protocol["checkpoint_policy"]["geometry_frozen"] is False
    assert protocol["experiment"]["physical_experiment"] is False
