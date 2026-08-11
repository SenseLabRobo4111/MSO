#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/mnt/data1/MSO_12method_benchmark_20260808")
DATASET = Path("/mnt/data1/SenseMapData/SenseMapDatasets")
OUTPUT = ROOT / "common_data"
def split_train_sample(sample_id: str) -> str:
    value = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    return "validation" if value % 10 == 0 else "train"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inspect_sample(source_split: str, sample: Path) -> dict[str, object]:
    obs_path = sample / "local_obs_0.png"
    target_path = sample / "local_map_0.png"
    with Image.open(obs_path) as image:
        obs = np.asarray(image.convert("RGB"))
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("L"))
    if obs.shape != (640, 640, 3) or target.shape != (640, 640):
        raise ValueError(f"unexpected geometry for {sample}")
    occupied = np.all(obs == np.asarray([255, 0, 0], dtype=np.uint8), axis=2)
    unknown = np.all(obs == np.asarray([0, 255, 0], dtype=np.uint8), axis=2)
    free = np.all(obs == np.asarray([0, 0, 255], dtype=np.uint8), axis=2)
    if not np.all(occupied | unknown | free):
        invalid_count = int((~(occupied | unknown | free)).sum())
        raise ValueError(
            f"unexpected categorical colours for {sample}: "
            f"{invalid_count} invalid pixels"
        )
    target_values = set(np.unique(target).tolist())
    if not target_values.issubset({0, 255}):
        raise ValueError(f"non-binary target for {sample}: {sorted(target_values)}")
    if not np.all((unknown.astype(np.uint8) + occupied + free) == 1):
        raise ValueError(f"one-hot contract failed for {sample}")
    assigned_split = "test" if source_split == "test" else split_train_sample(sample.name)
    return {
        "split": assigned_split,
        "source_split": source_split,
        "sample_id": sample.name,
        "observation_path": str(obs_path),
        "target_path": str(target_path),
        "observation_sha256": sha256(obs_path),
        "target_sha256": sha256(target_path),
        "unknown_fraction": float(unknown.mean()),
        "measured_occupied_fraction": float(occupied.mean()),
        "measured_free_fraction": float(free.mean()),
        "target_occupied_fraction": float((target == 255).mean()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source_split in ("train", "test"):
        samples = sorted(path for path in (DATASET / source_split).iterdir() if path.is_dir())
        rows.extend(inspect_sample(source_split, sample) for sample in samples)
    counts = Counter(str(row["split"]) for row in rows)
    if counts["train"] + counts["validation"] != 6841 or counts["test"] != 1463:
        raise ValueError(f"unexpected split counts: {counts}")
    fields = list(rows[0])
    manifest = OUTPUT / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "prospective_uniform_benchmark",
        "historical_split_recovered": False,
        "source_geometry": [640, 640],
        "model_geometry": [256, 256],
        "categorical_resize": "nearest",
        "rgb_semantics": {"red": "occupied", "green": "unknown", "blue": "free"},
        "selection_unit": "sample_level_hash_split_not_building_disjoint",
        "counts": dict(sorted(counts.items())),
        "manifest_sha256": sha256(manifest),
    }
    summary_path = OUTPUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "SHA256SUMS").write_text(
        f"{sha256(manifest)}  {manifest.name}\n{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
