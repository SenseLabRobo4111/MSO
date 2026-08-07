from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = PACKAGE_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_archive(root: Path) -> None:
    for partition, samples in {"train": ("0_0", "1_0"), "test": ("2_0",)}.items():
        for sample in samples:
            folder = root / partition / sample
            folder.mkdir(parents=True)
            sample_index = int(sample.split("_", 1)[0])
            obs = np.zeros((16, 16, 3), dtype=np.uint8)
            obs[:, : 5 + sample_index, 1] = 255
            target = np.zeros((16, 16), dtype=np.uint8)
            target[3 + sample_index : 11 + sample_index, 4:12] = 255
            Image.fromarray(obs, mode="RGB").save(folder / "local_obs_0.png")
            Image.fromarray(target, mode="L").save(folder / "local_map_0.png")


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "split", "group_key", "group_kind")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_inventory_and_semantic_split(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_summary = tmp_path / "inventory.json"
    payload = inventory_tool.write_inventory(archive, inventory, inventory_summary)
    assert payload["partition_counts"] == {"test": 1, "train": 2}
    with inventory.open("r", encoding="utf-8", newline="") as stream:
        first_row = next(csv.DictReader(stream))
    assert first_row["split"] == "train"
    assert first_row["split_status"] == "preserved_archive_label_not_manuscript_split"

    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "B", "group_kind": "building"},
        ],
    )
    output = tmp_path / "split.csv"
    summary = tmp_path / "split.json"
    result = split_tool.write_split(inventory, metadata, output, summary)
    assert result["split_counts"] == {"train": 2, "val": 0, "test": 1}


def test_worker_prefix_group_is_rejected(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "0", "group_kind": "worker"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "1", "group_kind": "worker"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "2", "group_kind": "worker"},
        ],
    )
    with pytest.raises(ValueError, match="worker/thread/process"):
        split_tool.build(inventory, metadata)


def test_cross_split_semantic_group_is_rejected(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "B", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "A", "group_kind": "building"},
        ],
    )
    with pytest.raises(ValueError, match="cross splits"):
        split_tool.build(inventory, metadata)


def test_cross_split_exact_processed_pair_is_rejected(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    shutil.copy2(
        archive / "train" / "0_0" / "local_obs_0.png",
        archive / "test" / "2_0" / "local_obs_0.png",
    )
    shutil.copy2(
        archive / "train" / "0_0" / "local_map_0.png",
        archive / "test" / "2_0" / "local_map_0.png",
    )
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "B", "group_kind": "building"},
        ],
    )
    with pytest.raises(ValueError, match="Exact processed pairs cross splits"):
        split_tool.build(inventory, metadata)


def test_training_entry_dry_run_on_fixture(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "B", "group_kind": "building"},
        ],
    )
    manifest = tmp_path / "split.csv"
    split_tool.write_split(inventory, metadata, manifest, tmp_path / "split.json")
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "train.py"),
            "--stage",
            "teacher",
            "--dataset-root",
            str(archive),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "dry_run_only"' in completed.stdout
    assert '"dataset_integrity": "verified_manifest_raw_sha256"' in completed.stdout


def test_training_default_hash_verification_rejects_changed_image(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "B", "group_kind": "building"},
        ],
    )
    manifest = tmp_path / "split.csv"
    split_tool.write_split(inventory, metadata, manifest, tmp_path / "split.json")
    changed = np.full((16, 16, 3), 127, dtype=np.uint8)
    Image.fromarray(changed, mode="RGB").save(
        archive / "train" / "1_0" / "local_obs_0.png"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "train.py"),
            "--stage",
            "teacher",
            "--dataset-root",
            str(archive),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Observation hash mismatch: train/1_0" in completed.stderr


def test_evaluation_entry_on_fixture(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "B", "group_kind": "building"},
        ],
    )
    manifest = tmp_path / "split.csv"
    split_tool.write_split(inventory, metadata, manifest, tmp_path / "split.json")
    output = tmp_path / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "evaluate.py"),
            "--dataset-root",
            str(archive),
            "--manifest",
            str(manifest),
            "--weights",
            str(PACKAGE_ROOT / "checkpoints" / "recovered_candidate_deconv_a.pt"),
            "--output",
            str(output),
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--max-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["n"] == 1
    assert result["dataset_integrity"] == "verified_manifest_raw_sha256"
    assert result["weights_sha256"] == (
        "da4458514656d41fba0e0ce6d4f4967997ff0a97f2e905a458757609edf3a3a8"
    )


def test_evaluation_default_hash_verification_rejects_changed_image(tmp_path: Path):
    inventory_tool = load_tool("inventory_dataset")
    split_tool = load_tool("build_split_manifest")
    archive = tmp_path / "archive"
    make_archive(archive)
    inventory = tmp_path / "inventory.csv"
    inventory_tool.write_inventory(archive, inventory, tmp_path / "inventory.json")
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {"sample_id": "train/0_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "train/1_0", "split": "train", "group_key": "A", "group_kind": "building"},
            {"sample_id": "test/2_0", "split": "test", "group_key": "B", "group_kind": "building"},
        ],
    )
    manifest = tmp_path / "split.csv"
    split_tool.write_split(inventory, metadata, manifest, tmp_path / "split.json")
    changed = np.full((16, 16), 127, dtype=np.uint8)
    Image.fromarray(changed, mode="L").save(
        archive / "test" / "2_0" / "local_map_0.png"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "evaluate.py"),
            "--dataset-root",
            str(archive),
            "--manifest",
            str(manifest),
            "--weights",
            str(PACKAGE_ROOT / "checkpoints" / "recovered_candidate_deconv_a.pt"),
            "--output",
            str(tmp_path / "evaluation.json"),
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--max-samples",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Target hash mismatch: test/2_0" in completed.stderr
