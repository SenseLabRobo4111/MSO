"""Closed provenance checks for the prospective-v2 predictor artifact chain."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

from .hashing import array_sha256, canonical_json_sha256, sha256_file


LOCK_SCHEMA = "mso.prospective_v2.lock/1"
LOCK_STATUS = "prospective_v2_unrun_not_historical_recovery"
CHECKPOINT_SCHEMA = "mso.prospective_v2.checkpoint/1"
RUN_SCHEMA = "mso.prospective_v2.run/1"
COMPLETION_SCHEMA = "mso.prospective_v2.run_completion/1"
ARTIFACT_STATUS = "prospective_v2_not_historical_recovery"
TRAINING_SCHEMA = "mso.closed_loop.prospective_training/1"
SELECTION_SCHEMA = "mso.closed_loop.validation_selection/1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_DATASET_COLUMNS = {
    "sample_id",
    "split",
    "group_key",
    "group_kind",
    "floorplan_id",
    "source_sha256",
    "obs_relpath",
    "target_relpath",
    "obs_sha256",
    "target_sha256",
    "raw_obs_relpath",
    "raw_target_relpath",
    "raw_obs_sha256",
    "raw_target_sha256",
}
REQUIRED_LOCKED_TRAINING_SOURCES = {
    "source/build_v2_lock.py",
    "source/generate_v2_locked_dataset.py",
    "source/run_v2_locked_campaign.py",
    "source/validate_v2_dataset.py",
    "source/audit_v2_gates.py",
    "source/evaluate_v2_positive_control.py",
    "source/train_v2_locked.py",
    "source/summarize_v2_multiseed.py",
    "source/mso_recovery/common.py",
    "source/mso_recovery/objective.py",
    "source/mso_recovery/v2_data.py",
    "source/mso_recovery/v2_geometry.py",
    "source/mso_recovery/v2_lock.py",
    "source/sensemap/explore_model/ffc.py",
    "source/sensemap/explore_model/SenseMapNet.py",
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return value


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{label} path is empty")
    item = Path(relative)
    if item.is_absolute():
        raise ValueError(f"{label} path must be relative")
    resolved_root = root.resolve()
    candidate = (resolved_root / item).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its artifact root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} file is missing: {relative}")
    return candidate


def _read_digest_file(path: Path) -> str:
    fields = path.read_text(encoding="ascii").strip().split()
    if not fields:
        raise ValueError("v2 lock digest file is empty")
    return _require_sha(fields[0], "v2 lock digest")


def validate_v2_lock(lock_path: Path) -> dict[str, Any]:
    """Validate v2 protocol and immutable training software without source images."""

    lock_path = lock_path.resolve()
    lock = _read_object(lock_path)
    lock_sha = sha256_file(lock_path)
    digest_path = lock_path.with_name("v2_lock.sha256")
    if not digest_path.is_file() or _read_digest_file(digest_path) != lock_sha:
        raise ValueError("v2 lock digest file does not bind the lock")
    if lock.get("schema") != LOCK_SCHEMA or lock.get("status") != LOCK_STATUS:
        raise ValueError("v2 training lock schema or status is incompatible")
    configuration_sha = _require_sha(
        lock.get("configuration_sha256"), "v2 training configuration"
    )
    if configuration_sha != canonical_json_sha256(
        {"geometry": lock.get("geometry"), "training": lock.get("training")}
    ):
        raise ValueError("v2 training configuration digest is inconsistent")

    root = lock_path.parent
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("v2 lock artifacts are absent")
    text_artifacts: dict[str, dict[str, str]] = {}
    for name in ("source_inventory", "source_split", "environment"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"v2 lock {name} record is absent")
        path = _safe_child(root, record.get("path"), f"v2 {name}")
        expected = _require_sha(record.get("sha256"), f"v2 {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"v2 lock {name} artifact changed")
        text_artifacts[name] = {"path": str(path), "sha256": expected}

    source_records = artifacts.get("source_files")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("v2 locked training-source inventory is absent")
    locked_sources: dict[str, str] = {}
    for record in source_records:
        if not isinstance(record, dict):
            raise ValueError("v2 locked training-source record is malformed")
        relative = record.get("path")
        path = _safe_child(root, relative, "v2 training source")
        expected = _require_sha(record.get("sha256"), "v2 training source")
        if sha256_file(path) != expected:
            raise ValueError(f"v2 training source changed: {relative}")
        if relative in locked_sources:
            raise ValueError("v2 locked training-source path is duplicated")
        locked_sources[str(relative)] = expected
    missing_sources = REQUIRED_LOCKED_TRAINING_SOURCES.difference(locked_sources)
    if missing_sources:
        raise ValueError(
            f"v2 training-source closure is incomplete: {sorted(missing_sources)}"
        )

    training = lock.get("training")
    if not isinstance(training, dict):
        raise ValueError("v2 training policy is absent")
    students = training.get("models", {}).get("student", {})
    if students != {
        "class": "sensemap.explore_model.SenseMapNet.DistillMapNetDeconv",
        "image_size": 256,
        "dim": 4,
        "parameter_count": 342771,
    }:
        raise ValueError("v2 student architecture differs from the runtime model")
    seeds = training.get("student_seeds")
    if not isinstance(seeds, list) or len(seeds) != 5 or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("v2 student candidate seeds are not a complete five-seed set")
    selection = training.get("checkpoint_selection")
    if not isinstance(selection, dict) or any(
        (
            selection.get("metric") != "validation_unknown_bce",
            selection.get("direction") != "minimize",
            selection.get("tie_break") != "earliest_epoch",
            selection.get("test_used") is not False,
        )
    ):
        raise ValueError("v2 checkpoint-selection contract is incompatible")
    return {
        "lock": lock,
        "lock_sha256": lock_sha,
        "configuration_sha256": configuration_sha,
        "source_split_path": text_artifacts["source_split"]["path"],
        "source_split_sha256": text_artifacts["source_split"]["sha256"],
        "training_environment_sha256": text_artifacts["environment"]["sha256"],
        "student_seeds": [int(seed) for seed in seeds],
        "locked_training_sources": locked_sources,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV artifact is empty: {path}")
    return rows


def audit_train_validation_dataset(
    dataset_root: Path,
    manifest_path: Path,
    *,
    source_split_path: Path,
    source_split_sha256: str,
) -> dict[str, Any]:
    """Rehash every train/validation datum and bind it to the locked split text."""

    root = dataset_root.resolve()
    manifest_path = manifest_path.resolve()
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("train/validation dataset root or manifest is missing")
    if sha256_file(source_split_path) != source_split_sha256:
        raise ValueError("source split changed after the v2 lock")
    split_rows = _read_csv(source_split_path)
    split_by_floorplan: dict[str, tuple[str, str]] = {}
    for row in split_rows:
        floorplan = row.get("floorplan_id", "")
        split = row.get("split", "")
        source_sha = row.get("gt_sha256", "")
        if not floorplan or split not in {"train", "val", "test"}:
            raise ValueError("source split row is malformed")
        _require_sha(source_sha, "source split floorplan")
        if floorplan in split_by_floorplan:
            raise ValueError("source split floorplan is duplicated")
        split_by_floorplan[floorplan] = (split, source_sha)

    rows = _read_csv(manifest_path)
    missing = REQUIRED_DATASET_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"dataset manifest columns are missing: {sorted(missing)}")
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    inventory: list[dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0}
    floorplans = {"train": set(), "val": set()}
    for row in rows:
        sample_id = row.get("sample_id", "")
        split = row.get("split", "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError("dataset sample identifier is empty or duplicated")
        sample_ids.add(sample_id)
        if split not in {"train", "val"}:
            raise ValueError("dataset manifest may contain train/val rows only")
        if row.get("group_kind") != "operational_prefix_group" or not row.get(
            "group_key"
        ):
            raise ValueError("dataset grouping contract is malformed")
        group_splits.setdefault(row["group_key"], set()).add(split)
        floorplan = row.get("floorplan_id", "")
        locked_source = split_by_floorplan.get(floorplan)
        if locked_source != (split, row.get("source_sha256")):
            raise ValueError("dataset row differs from the locked source split")
        file_records = []
        for path_field, hash_field in (
            ("obs_relpath", "obs_sha256"),
            ("target_relpath", "target_sha256"),
            ("raw_obs_relpath", "raw_obs_sha256"),
            ("raw_target_relpath", "raw_target_sha256"),
        ):
            path = _safe_child(root, row.get(path_field), f"dataset {path_field}")
            expected = _require_sha(row.get(hash_field), f"dataset {hash_field}")
            if sha256_file(path) != expected:
                raise ValueError(f"dataset file changed for sample {sample_id}")
            file_records.append(
                {"role": path_field, "path": row[path_field], "sha256": expected}
            )
        inventory.append(
            {
                "sample_id": sample_id,
                "split": split,
                "group_key": row["group_key"],
                "floorplan_id": floorplan,
                "source_sha256": row["source_sha256"],
                "files": file_records,
            }
        )
        split_counts[split] += 1
        floorplans[split].add(floorplan)
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("dataset operational group leaks across train and validation")
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError("dataset requires non-empty train and validation splits")
    inventory = sorted(inventory, key=lambda row: str(row["sample_id"]))
    return {
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "source_split_sha256": source_split_sha256,
        "file_inventory_sha256": canonical_json_sha256(inventory),
        "sample_counts": split_counts,
        "floorplan_counts": {
            split: len(values) for split, values in floorplans.items()
        },
        "file_count": 4 * len(inventory),
        "manifest_rows": inventory,
        "validation_observation_path": str(
            _safe_child(
                root,
                next(row for row in rows if row["split"] == "val")["obs_relpath"],
                "validation observation",
            )
        ),
    }


def _artifact_record(
    root: Path, record: dict[str, Any], path_key: str, hash_key: str, label: str
) -> Path:
    path = _safe_child(root, record.get(path_key), label)
    expected = _require_sha(record.get(hash_key), label)
    if sha256_file(path) != expected:
        raise ValueError(f"{label} hash differs from its manifest")
    return path


def validate_training_and_selection(
    *,
    training_manifest_path: Path,
    selection_trace_path: Path,
    checkpoint_path: Path,
    v2: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify candidate completeness, per-run records, ranking, and winner."""

    training_manifest_path = training_manifest_path.resolve()
    selection_trace_path = selection_trace_path.resolve()
    training = _read_object(training_manifest_path)
    selection = _read_object(selection_trace_path)
    if training.get("schema") != TRAINING_SCHEMA or training.get(
        "status"
    ) != ARTIFACT_STATUS:
        raise ValueError("prospective training manifest schema or status is incompatible")
    expected_bindings = {
        "v2_lock_sha256": v2["lock_sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "source_split_sha256": dataset["source_split_sha256"],
        "training_configuration_sha256": v2["configuration_sha256"],
        "dataset_file_inventory_sha256": dataset["file_inventory_sha256"],
        "dataset_file_count": dataset["file_count"],
        "dataset_sample_counts": dataset["sample_counts"],
        "dataset_floorplan_counts": dataset["floorplan_counts"],
        "test_outcomes_accessed": False,
        "candidate_universe_complete": True,
    }
    mismatch = {
        key: {"expected": value, "observed": training.get(key)}
        for key, value in expected_bindings.items()
        if training.get(key) != value
    }
    if mismatch:
        raise ValueError(f"prospective training manifest binding failed: {mismatch}")
    candidates = training.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("prospective training candidate universe is absent")
    expected_seeds = sorted(v2["student_seeds"])
    observed_seeds = sorted(
        int(candidate.get("seed", -1))
        for candidate in candidates
        if isinstance(candidate, dict)
    )
    if observed_seeds != expected_seeds or len(candidates) != len(expected_seeds):
        raise ValueError("prospective training candidate universe is incomplete")

    root = training_manifest_path.parent
    validated: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate record is malformed")
        seed = candidate.get("seed")
        if candidate.get("stage") != "student" or candidate.get(
            "eligible"
        ) is not True:
            raise ValueError("candidate exclusion or non-student stage is forbidden")
        metric = candidate.get("validation_unknown_bce")
        if not isinstance(metric, (int, float)) or isinstance(
            metric, bool
        ) or not math.isfinite(float(metric)):
            raise ValueError("candidate validation metric is non-finite")
        run_path = _artifact_record(
            root,
            candidate,
            "run_manifest_path",
            "run_manifest_sha256",
            "candidate run manifest",
        )
        completion_path = _artifact_record(
            root,
            candidate,
            "completion_path",
            "completion_sha256",
            "candidate completion",
        )
        candidate_checkpoint = _artifact_record(
            root,
            candidate,
            "checkpoint_path",
            "checkpoint_sha256",
            "candidate checkpoint",
        )
        run = _read_object(run_path)
        completion = _read_object(completion_path)
        bindings = {
            "v2_lock_sha256": v2["lock_sha256"],
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "source_split_sha256": dataset["source_split_sha256"],
            "training_configuration_sha256": v2["configuration_sha256"],
        }
        if run.get("schema") != RUN_SCHEMA or run.get("status") != ARTIFACT_STATUS:
            raise ValueError("candidate run manifest schema or status is incompatible")
        if completion.get("schema") != COMPLETION_SCHEMA or completion.get(
            "status"
        ) != ARTIFACT_STATUS:
            raise ValueError("candidate completion schema or status is incompatible")
        if any(run.get(key) != value for key, value in bindings.items()) or any(
            completion.get(key) != value for key, value in bindings.items()
        ):
            raise ValueError("candidate run/checkpoint provenance binding failed")
        job = run.get("job")
        if not isinstance(job, dict) or any(
            (
                job.get("stage") != "student",
                job.get("dataset_kind") != "full",
                job.get("seed") != seed,
                run.get("test_outcomes_accessed") is not False,
                completion.get("test_outcomes_accessed") is not False,
                completion.get("stage") != "student",
                completion.get("seed") != seed,
                completion.get("best_checkpoint_sha256")
                != candidate.get("checkpoint_sha256"),
                not math.isclose(
                    float(completion.get("best_validation_unknown_bce", math.nan)),
                    float(metric),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                run.get("train_samples") != dataset["sample_counts"]["train"],
                run.get("validation_samples")
                != dataset["sample_counts"]["val"],
            )
        ):
            raise ValueError("candidate run/completion contract is inconsistent")
        validated.append(
            {
                "seed": int(seed),
                "stage": "student",
                "validation_unknown_bce": float(metric),
                "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                "run_manifest_sha256": str(candidate["run_manifest_sha256"]),
                "completion_sha256": str(candidate["completion_sha256"]),
                "checkpoint_path": str(candidate["checkpoint_path"]),
                "run_manifest_path": str(candidate["run_manifest_path"]),
                "completion_path": str(candidate["completion_path"]),
            }
        )
    canonical_candidates = sorted(validated, key=lambda item: item["seed"])
    universe_sha = canonical_json_sha256(canonical_candidates)
    if training.get("candidate_universe_sha256") != universe_sha:
        raise ValueError("training candidate-universe digest is inconsistent")

    if selection.get("schema") != SELECTION_SCHEMA or selection.get(
        "status"
    ) != ARTIFACT_STATUS:
        raise ValueError("selection trace schema or status is incompatible")
    selection_bindings = {
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "candidate_universe_sha256": universe_sha,
        "candidate_universe_complete": True,
        "selection_split": "val",
        "selection_metric": "validation_unknown_bce",
        "selection_direction": "minimize",
        "tie_break": "seed_then_checkpoint_sha256",
        "test_outcomes_accessed": False,
    }
    if any(selection.get(key) != value for key, value in selection_bindings.items()):
        raise ValueError("selection trace provenance or universe binding failed")
    ranked = sorted(
        canonical_candidates,
        key=lambda item: (
            item["validation_unknown_bce"],
            item["seed"],
            item["checkpoint_sha256"],
        ),
    )
    expected_ranking = [item["checkpoint_sha256"] for item in ranked]
    if selection.get("ranking") != expected_ranking:
        raise ValueError("selection trace ranking is incomplete or incorrectly sorted")
    winner = ranked[0]
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate["checkpoint_sha256"] == winner["checkpoint_sha256"]
    )
    if any(
        (
            selection.get("selected_checkpoint_sha256")
            != winner["checkpoint_sha256"],
            selection.get("selected_checkpoint_seed") != winner["seed"],
            not math.isclose(
                float(selection.get("selected_validation_metric_value", math.nan)),
                winner["validation_unknown_bce"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            sha256_file(checkpoint_path.resolve()) != winner["checkpoint_sha256"],
            _safe_child(
                root, winner["checkpoint_path"], "selected candidate checkpoint"
            ).resolve()
            != checkpoint_path.resolve(),
            selection.get("selected_run_manifest_sha256")
            != winner["run_manifest_sha256"],
            selection.get("selected_completion_sha256")
            != winner["completion_sha256"],
        )
    ):
        raise ValueError("selection trace does not select the validation optimum")
    selected_run_path = _safe_child(
        root, selected_candidate["run_manifest_path"], "selected run manifest"
    )
    return training, selection, {
        "candidate_universe_sha256": universe_sha,
        "candidate_count": len(ranked),
        "selected_seed": winner["seed"],
        "selected_validation_metric_value": winner["validation_unknown_bce"],
        "selected_checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_run_manifest_sha256": sha256_file(selected_run_path),
    }


def validate_checkpoint_payload(
    payload: Any,
    *,
    checkpoint_sha256: str,
    v2: dict[str, Any],
    dataset: dict[str, Any],
    selection_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate every metadata link in the selected v2 student checkpoint."""

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    required = {
        "schema",
        "status",
        "v2_lock_sha256",
        "dataset_manifest_sha256",
        "source_split_sha256",
        "training_configuration_sha256",
        "stage",
        "seed",
        "epoch",
        "selected_validation_metric",
        "selected_epoch",
        "parent_teacher_checkpoint_sha256",
        "model_state_dict",
        "optimizer_state_dict",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint contract fields are missing: {missing}")
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "status": ARTIFACT_STATUS,
        "v2_lock_sha256": v2["lock_sha256"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "source_split_sha256": dataset["source_split_sha256"],
        "training_configuration_sha256": v2["configuration_sha256"],
        "stage": "student",
        "seed": selection_metadata["selected_seed"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("selected checkpoint provenance binding failed")
    for name in ("epoch", "selected_epoch"):
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"checkpoint {name} must be a positive integer")
    if payload["selected_epoch"] > payload["epoch"]:
        raise ValueError("selected checkpoint epoch occurs after saved epoch")
    metric = payload.get("selected_validation_metric")
    if not isinstance(metric, (int, float)) or isinstance(
        metric, bool
    ) or not math.isfinite(float(metric)) or not math.isclose(
        float(metric),
        selection_metadata["selected_validation_metric_value"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("checkpoint validation metric differs from selection")
    _require_sha(
        payload.get("parent_teacher_checkpoint_sha256"),
        "parent teacher checkpoint",
    )
    if not isinstance(payload.get("model_state_dict"), dict) or not payload[
        "model_state_dict"
    ]:
        raise ValueError("checkpoint model state is absent or empty")
    if not isinstance(payload.get("optimizer_state_dict"), dict) or not payload[
        "optimizer_state_dict"
    ]:
        raise ValueError("checkpoint optimizer state is absent or empty")
    if checkpoint_sha256 != selection_metadata["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint bytes differ from selection")
    return {
        **expected,
        "epoch": int(payload["epoch"]),
        "selected_epoch": int(payload["selected_epoch"]),
        "selected_validation_metric": float(metric),
        "parent_teacher_checkpoint_sha256": str(
            payload["parent_teacher_checkpoint_sha256"]
        ),
    }


def strict_model_validation(
    payload: dict[str, Any], validation_observation_path: Path, execution_device: str
) -> dict[str, Any]:
    """Instantiate the runtime network, load strictly, and run one locked val input."""

    import torch
    from PIL import Image
    from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv

    device = torch.device(execution_device)
    model = DistillMapNetDeconv(image_size=256, dim=4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 342771:
        raise ValueError("runtime predictor parameter count is incompatible")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError("checkpoint cannot be loaded strictly into the runtime model") from error
    observation = np.asarray(
        Image.open(validation_observation_path).convert("RGB"), dtype=np.uint8
    ).copy()
    if observation.shape != (256, 256, 3):
        raise ValueError("validation inference observation has an unexpected shape")
    one_hot = observation == 255
    if not np.all(one_hot.sum(axis=2) == 1):
        raise ValueError("validation inference observation is not one-hot")
    tensor = (
        torch.from_numpy(one_hot.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    model = model.to(device).eval()
    with torch.inference_mode():
        outputs = model(tensor)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("runtime predictor output structure is incompatible")
    prediction = outputs[0]
    if tuple(prediction.shape) != (1, 1, 256, 256):
        raise ValueError("runtime predictor output shape is incompatible")
    values = prediction.detach().float().cpu().numpy()
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("runtime predictor output is not a finite probability map")
    return {
        "model_class": (
            "sensemap.explore_model.SenseMapNet.DistillMapNetDeconv"
        ),
        "model_input_shape": [1, 3, 256, 256],
        "model_output_shape": [1, 1, 256, 256],
        "parameter_count": parameter_count,
        "strict_state_dict_load": True,
        "validation_inference_output_sha256": array_sha256(values),
        "validation_observation_sha256": sha256_file(
            validation_observation_path
        ),
    }
