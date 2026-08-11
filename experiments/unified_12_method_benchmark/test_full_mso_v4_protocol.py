from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import train_paper_mso_reconstructed_v4 as v4
import aggregate_full_mso_reconstruction_v4 as aggregate_v4


ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FixedProbability(nn.Module):
    def __init__(self, probability: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("probability", probability)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor]:
        if features.shape[0] != self.probability.shape[0]:
            raise ValueError("fixture batch size differs")
        return (self.probability,)


def test_v3_bundle_remains_byte_identical() -> None:
    rows = []
    for line in (ROOT / "FULL_MSO_BUNDLE_SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        rows.append((expected, relative))
    assert len(rows) == 14
    for expected, relative in rows:
        assert file_sha256(ROOT / relative) == expected


def test_v4_config_keeps_training_invariants_and_validates() -> None:
    v3_config = json.loads((ROOT / "full_mso_config.json").read_text())
    v4_config = json.loads((ROOT / "full_mso_config_v4.json").read_text())
    v4.validate_config(v4_config)
    invariant_keys = (
        "teacher_seed",
        "projection_seed",
        "image_size",
        "batch_size",
        "num_workers",
        "maximum_epochs",
        "learning_rate",
        "adam_betas",
        "weight_decay",
        "mixed_precision",
        "student_seeds",
        "model_manifest_sha256",
        "common_manifest_sha256",
        "split_counts",
        "loss_weights",
        "teacher_training",
        "projection_policy",
        "recovered_checkpoint_sha256",
        "teacher_base_width",
        "student_base_width",
        "network_source_sha256",
        "ffc_source_sha256",
        "augmentation",
        "feature_reconstruction",
        "discriminator",
        "drop_last",
        "augmentation_probabilities",
        "known_cell_target_policy",
        "decision_threshold",
        "latency_warmup",
        "latency_repetitions",
    )
    assert {key: v4_config[key] for key in invariant_keys} == {
        key: v3_config[key] for key in invariant_keys
    }
    assert v4_config["maximum_epochs"] == 500


def test_validation_fixture_records_integer_confusion_and_calibration() -> None:
    probability = torch.tensor(
        [[[[0.9, 0.4, 0.8], [0.2, 0.6, 0.1]]]], dtype=torch.float32
    )
    target = torch.tensor(
        [[[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32
    )
    unknown = torch.ones_like(target)
    features = torch.zeros((1, 3, 2, 3), dtype=torch.float32)
    loader = [(features, target, unknown, ["fixture"])]
    metrics = v4.validation_metrics(
        FixedProbability(probability), loader, "cpu", threshold=0.5, ece_bins=15
    )
    assert metrics["unknown_count"] == 6
    assert metrics["unknown_tp"] == 2
    assert metrics["unknown_fp"] == 1
    assert metrics["unknown_fn"] == 1
    assert metrics["unknown_tn"] == 2
    assert all(
        isinstance(metrics[key], int)
        for key in ("unknown_tp", "unknown_fp", "unknown_fn", "unknown_tn")
    )
    assert metrics["unknown_f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["unknown_iou"] == pytest.approx(0.5)
    assert metrics["unknown_brier"] == pytest.approx(1.22 / 6.0, abs=1e-7)
    assert metrics["unknown_ece15"] == pytest.approx(2.2 / 6.0, abs=1e-7)
    clipped = np.clip(probability.numpy().reshape(-1), 1e-6, 1 - 1e-6)
    truth = target.numpy().reshape(-1)
    expected_bce = float(
        (-truth * np.log(clipped) - (1 - truth) * np.log1p(-clipped)).mean()
    )
    assert metrics["unknown_bce"] == pytest.approx(expected_bce, abs=1e-7)


def record(
    epoch: int,
    tp: int,
    fp: int,
    fn: int,
    brier: float,
    bce: float = 0.5,
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "unknown_tp": tp,
        "unknown_fp": fp,
        "unknown_fn": fn,
        "unknown_brier": brier,
        "unknown_bce": bce,
    }


def test_primary_order_is_exact_f1_then_brier_then_earlier() -> None:
    incumbent = record(epoch=3, tp=1, fp=1, fn=0, brier=0.20)
    same_f1_better_brier = record(epoch=4, tp=2, fp=2, fn=0, brier=0.19)
    same_f1_same_brier_later = record(epoch=5, tp=2, fp=2, fn=0, brier=0.20)
    higher_f1 = record(epoch=6, tp=3, fp=1, fn=0, brier=0.99)
    assert v4.primary_is_better(incumbent, None)
    assert v4.primary_is_better(same_f1_better_brier, incumbent)
    assert not v4.primary_is_better(same_f1_same_brier_later, incumbent)
    assert v4.primary_is_better(higher_f1, incumbent)


def test_calibration_order_is_brier_then_bce_then_earlier() -> None:
    incumbent = record(epoch=3, tp=1, fp=1, fn=0, brier=0.2, bce=0.4)
    assert v4.calibration_is_better(
        record(epoch=4, tp=1, fp=1, fn=0, brier=0.19, bce=9.0), incumbent
    )
    assert v4.calibration_is_better(
        record(epoch=4, tp=1, fp=1, fn=0, brier=0.2, bce=0.39), incumbent
    )
    assert not v4.calibration_is_better(
        record(epoch=4, tp=1, fp=1, fn=0, brier=0.2, bce=0.4), incumbent
    )


def test_student_parent_accepts_only_v4_teacher_primary() -> None:
    payload = {
        "checkpoint_kind": v4.PRIMARY_KIND,
        "stage": "teacher",
        "seed": 101,
        "protocol_id": "mso_paper_equation_reconstruction_v4",
    }
    v4.validate_parent_checkpoint_payload(
        payload, "mso_paper_equation_reconstruction_v4"
    )
    for kind in (v4.CALIBRATION_KIND, v4.SNAPSHOT_KIND, v4.RESUME_KIND):
        rejected = {**payload, "checkpoint_kind": kind}
        with pytest.raises(ValueError, match="best_primary"):
            v4.validate_parent_checkpoint_payload(
                rejected, "mso_paper_equation_reconstruction_v4"
            )
    with pytest.raises(ValueError, match="protocol"):
        v4.validate_parent_checkpoint_payload(payload, "forged_v4_protocol")


def test_student_parent_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_primary.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    actual = file_sha256(checkpoint)
    assert v4.validate_parent_checkpoint_sha(checkpoint, actual) == actual
    with pytest.raises(ValueError, match="differs"):
        v4.validate_parent_checkpoint_sha(checkpoint, "0" * 64)
    with pytest.raises(ValueError, match="explicit lowercase"):
        v4.validate_parent_checkpoint_sha(checkpoint, actual.upper())


def test_selection_manifest_recomputes_tsv_and_checks_both_checkpoints(
    tmp_path: Path,
) -> None:
    records = [
        {
            "epoch": 0,
            "unknown_count": 10,
            "unknown_tp": 1,
            "unknown_fp": 0,
            "unknown_fn": 3,
            "unknown_tn": 6,
            "unknown_precision": 1.0,
            "unknown_recall": 0.25,
            "unknown_f1": 0.4,
            "unknown_iou": 0.25,
            "unknown_bce": 0.3,
            "unknown_brier": 0.1,
            "unknown_ece15": 0.2,
        },
        {
            "epoch": 1,
            "unknown_count": 10,
            "unknown_tp": 3,
            "unknown_fp": 1,
            "unknown_fn": 1,
            "unknown_tn": 5,
            "unknown_precision": 0.75,
            "unknown_recall": 0.75,
            "unknown_f1": 0.75,
            "unknown_iou": 0.6,
            "unknown_bce": 0.5,
            "unknown_brier": 0.2,
            "unknown_ece15": 0.3,
        },
        {
            "epoch": 2,
            "unknown_count": 10,
            "unknown_tp": 2,
            "unknown_fp": 1,
            "unknown_fn": 2,
            "unknown_tn": 5,
            "unknown_precision": 2 / 3,
            "unknown_recall": 0.5,
            "unknown_f1": 4 / 7,
            "unknown_iou": 0.4,
            "unknown_bce": 0.2,
            "unknown_brier": 0.08,
            "unknown_ece15": 0.1,
        },
    ]
    rows = []
    for item in records:
        rows.append(
            {
                "epoch": item["epoch"],
                **{
                    f"validation_{key}": item[key]
                    for key in v4.VALIDATION_RECORD_KEYS
                },
            }
        )
    metrics = tmp_path / "metrics_by_epoch.tsv"
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    primary = tmp_path / "best_primary.pt"
    calibration = tmp_path / "best_calibration.pt"
    metadata = {
        "status": v4.STATUS,
        "protocol_id": "mso_paper_equation_reconstruction_v4",
        "configuration_sha256": "a" * 64,
        "model_manifest_sha256": "b" * 64,
        "source_hashes": {"trainer_v4": "c" * 64},
        "topology_contract_sha256": "d" * 64,
        "parent_teacher_checkpoint_kind": None,
        "parent_teacher_checkpoint_sha256": None,
        "container_image_id": "sha256:" + "e" * 64,
        "bundle_sha256": "f" * 64,
    }
    torch.save(
        {
            **metadata,
            "checkpoint_kind": v4.PRIMARY_KIND,
            "stage": "teacher",
            "seed": 101,
            "epoch": 1,
            "selection_record": records[1],
            "projection_state_fingerprint": None,
        },
        primary,
    )
    torch.save(
        {
            **metadata,
            "checkpoint_kind": v4.CALIBRATION_KIND,
            "stage": "teacher",
            "seed": 101,
            "epoch": 2,
            "selection_record": records[2],
            "projection_state_fingerprint": None,
        },
        calibration,
    )
    output = tmp_path / "selection_manifest.json"
    manifest = v4.write_selection_manifest(
        metrics,
        primary,
        calibration,
        output,
        "mso_paper_equation_reconstruction_v4",
        3,
        "teacher",
        101,
        metadata,
        None,
    )
    assert manifest["primary"]["epoch"] == 1
    assert manifest["calibration"]["epoch"] == 2
    assert manifest["primary"]["checkpoint_sha256"] == file_sha256(primary)
    assert json.loads(output.read_text()) == manifest
    corrupted = torch.load(primary, map_location="cpu", weights_only=False)
    corrupted["selection_record"] = records[0]
    torch.save(corrupted, primary)
    with pytest.raises(ValueError, match="selection_record"):
        v4.write_selection_manifest(
            metrics,
            primary,
            calibration,
            output,
            "mso_paper_equation_reconstruction_v4",
            3,
            "teacher",
            101,
            metadata,
            None,
        )
    corrupted["selection_record"] = records[1]
    corrupted["configuration_sha256"] = "0" * 64
    torch.save(corrupted, primary)
    with pytest.raises(ValueError, match="configuration_sha256"):
        v4.write_selection_manifest(
            metrics,
            primary,
            calibration,
            output,
            "mso_paper_equation_reconstruction_v4",
            3,
            "teacher",
            101,
            metadata,
            None,
        )


def write_fixture_sha_manifest(root: Path, names: tuple[str, ...]) -> None:
    (root / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def test_aggregate_rejects_forged_protocol_and_parent_sha(tmp_path: Path) -> None:
    parent_sha = "1" * 64
    primary = tmp_path / "best_primary.pt"
    calibration = tmp_path / "best_calibration.pt"
    primary.write_bytes(b"primary")
    calibration.write_bytes(b"calibration")
    provenance = {
        "protocol_id": aggregate_v4.PROTOCOL_ID,
        "parent_teacher_checkpoint_kind": aggregate_v4.PRIMARY_CHECKPOINT_KIND,
        "parent_teacher_checkpoint_sha256": parent_sha,
        "configuration_sha256": "2" * 64,
    }
    run_manifest = {
        **provenance,
        "stage": "student",
        "seed": 11,
        "configuration": {"protocol_id": aggregate_v4.PROTOCOL_ID},
    }
    selection = {
        "protocol_id": aggregate_v4.PROTOCOL_ID,
        "stage": "student",
        "seed": 11,
        "source": aggregate_v4.SELECTION_SOURCE,
        "run_provenance": provenance,
        "primary": {
            "checkpoint_kind": aggregate_v4.PRIMARY_CHECKPOINT_KIND,
            "checkpoint_sha256": file_sha256(primary),
            "epoch": 7,
        },
        "calibration": {
            "checkpoint_kind": aggregate_v4.CALIBRATION_CHECKPOINT_KIND,
            "checkpoint_sha256": file_sha256(calibration),
            "epoch": 3,
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(run_manifest))
    (tmp_path / "selection_manifest.json").write_text(json.dumps(selection))
    completion = {
        "completed": True,
        "stage": "student",
        "seed": 11,
        "protocol_id": aggregate_v4.PROTOCOL_ID,
        "selection_metric": "validation_unknown_occupied_micro_f1_at_0.5",
        "selection_manifest_sha256": file_sha256(
            tmp_path / "selection_manifest.json"
        ),
        "best_primary_checkpoint_sha256": file_sha256(primary),
        "best_calibration_checkpoint_sha256": file_sha256(calibration),
        "best_primary_epoch": 7,
        "best_calibration_epoch": 3,
        "parent_teacher_checkpoint_kind": aggregate_v4.PRIMARY_CHECKPOINT_KIND,
        "parent_teacher_checkpoint_sha256": parent_sha,
    }
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    names = (
        "run_manifest.json",
        "selection_manifest.json",
        "completion.json",
        "best_primary.pt",
        "best_calibration.pt",
    )
    write_fixture_sha_manifest(tmp_path, names)
    aggregate_v4.read_completion(tmp_path, "student", 11, parent_sha)

    completion["protocol_id"] = "forged_v4_protocol"
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    write_fixture_sha_manifest(tmp_path, names)
    with pytest.raises(ValueError, match="completion protocol"):
        aggregate_v4.read_completion(tmp_path, "student", 11, parent_sha)

    completion["protocol_id"] = aggregate_v4.PROTOCOL_ID
    completion["parent_teacher_checkpoint_sha256"] = "9" * 64
    (tmp_path / "completion.json").write_text(json.dumps(completion))
    write_fixture_sha_manifest(tmp_path, names)
    with pytest.raises(ValueError, match="parent teacher SHA"):
        aggregate_v4.read_completion(tmp_path, "student", 11, parent_sha)


def test_snapshot_schedule_has_fifty_generator_snapshots() -> None:
    epochs = list(range(9, 500, 10))
    paths = [v4.snapshot_relative_path(epoch).as_posix() for epoch in epochs]
    assert len(paths) == 50
    assert paths[0] == "snapshots/epoch_009.pt"
    assert paths[-1] == "snapshots/epoch_499.pt"


def archive_ranking_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    uniform_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(aggregate_v4.BASELINE_METHODS):
        complete = method != aggregate_v4.INCOMPLETE_BASELINE
        status_rows.append(
            {
                "method": method,
                "registered_seeds": 5,
                "completed_seeds": 5 if complete else 0,
                "status": "complete" if complete else "incomplete",
                "exclusion_from_complete_case_contrasts": 0 if complete else 1,
            }
        )
        if not complete:
            continue
        for seed_index, seed in enumerate(aggregate_v4.SEEDS):
            uniform_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_unknown_occupied_f1": (
                        0.20 + method_index * 0.02 + seed_index * 0.001
                    ),
                    "baseline_only_field": method_index,
                }
            )
    v4_rows = [
        {
            "method": aggregate_v4.V4_METHOD,
            "seed": seed,
            "test_unknown_occupied_f1": 0.80 + seed_index * 0.001,
            "v4_only_field": seed_index,
        }
        for seed_index, seed in enumerate(aggregate_v4.SEEDS)
    ]
    return v4_rows, uniform_rows, status_rows


def test_archive_ranking_includes_every_complete_seed_and_baseline() -> None:
    v4_rows, uniform_rows, status_rows = archive_ranking_fixture()
    tables = aggregate_v4.build_archive_benchmark_tables(
        v4_rows, uniform_rows, status_rows
    )
    combined = tables["combined_per_seed_results.tsv"]
    assert len(combined) == 60
    assert len({(row["method"], row["seed"]) for row in combined}) == 60
    assert all(row["comparison_scope"] == aggregate_v4.CLAIM_SCOPE for row in combined)
    assert any(row["baseline_only_field"] != "" for row in combined)
    assert any(row["v4_only_field"] != "" for row in combined)

    status = {row["method"]: row for row in tables["combined_method_status.tsv"]}
    assert len(status) == 13
    assert status["RePaint"]["completed_seeds"] == 0
    assert status["RePaint"]["ranking_eligible"] == 0
    assert status["RePaint"]["ranking_exclusion_reason"] == (
        "zero_of_five_seed_runs_complete"
    )
    assert all(row["seed_exclusions"] == 0 for row in status.values())

    ranking = tables["primary_metric_ranking.tsv"]
    eligible = [row for row in ranking if row["ranking_eligible"]]
    assert len(eligible) == 12
    assert [row["rank"] for row in eligible] == list(range(1, 13))
    assert eligible[0]["method"] == aggregate_v4.V4_METHOD
    assert ranking[-1]["method"] == "RePaint"
    assert ranking[-1]["rank"] == ""

    raw = tables["v4_vs_baseline_matched_seed_differences.tsv"]
    contrasts = tables["v4_vs_baseline_matched_seed_summary.tsv"]
    assert len(raw) == 55
    assert len(contrasts) == 11
    assert {row["baseline_method"] for row in contrasts} == (
        set(aggregate_v4.BASELINE_METHODS) - {"RePaint"}
    )
    assert all(row["seed_excluded"] == 0 for row in raw)
    mso = next(row for row in contrasts if row["baseline_method"] == "MSO")
    assert mso["v4_minus_baseline_mean"] == pytest.approx(0.60)
    assert mso["v4_minus_baseline_ci95_low"] == pytest.approx(0.60)
    assert mso["v4_minus_baseline_ci95_high"] == pytest.approx(0.60)


def test_archive_ranking_rejects_missing_seed_and_repaint_result() -> None:
    v4_rows, uniform_rows, status_rows = archive_ranking_fixture()
    missing_seed = [
        row
        for row in uniform_rows
        if not (row["method"] == "HINT" and int(row["seed"]) == 71)
    ]
    with pytest.raises(ValueError, match="seed grid"):
        aggregate_v4.build_archive_benchmark_tables(
            v4_rows, missing_seed, status_rows
        )
    repaint_present = [
        *uniform_rows,
        {
            "method": "RePaint",
            "seed": 11,
            "test_unknown_occupied_f1": 0.5,
        },
    ]
    with pytest.raises(ValueError, match="explicitly incomplete"):
        aggregate_v4.build_archive_benchmark_tables(
            v4_rows, repaint_present, status_rows
        )
