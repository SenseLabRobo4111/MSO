#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_SIZE = (256, 256)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def materialize(task: tuple[dict[str, str], str]) -> dict[str, object]:
    row, cache_root_text = task
    cache_root = Path(cache_root_text)
    source_observation = Path(row["observation_path"])
    source_target = Path(row["target_path"])
    if sha256(source_observation) != row["observation_sha256"]:
        raise ValueError(f"source observation hash mismatch: {row['sample_id']}")
    if sha256(source_target) != row["target_sha256"]:
        raise ValueError(f"source target hash mismatch: {row['sample_id']}")
    with Image.open(source_observation) as image:
        observation_image = image.convert("RGB").resize(
            MODEL_SIZE, Image.Resampling.NEAREST
        )
    with Image.open(source_target) as image:
        target_image = image.convert("L").resize(MODEL_SIZE, Image.Resampling.NEAREST)
    observation = np.asarray(observation_image, dtype=np.uint8)
    target = np.asarray(target_image, dtype=np.uint8)
    occupied = np.all(
        observation == np.asarray([255, 0, 0], dtype=np.uint8), axis=2
    )
    unknown = np.all(
        observation == np.asarray([0, 255, 0], dtype=np.uint8), axis=2
    )
    free = np.all(
        observation == np.asarray([0, 0, 255], dtype=np.uint8), axis=2
    )
    if not np.all(occupied | unknown | free):
        raise ValueError(f"invalid resized categories: {row['sample_id']}")
    target_occupied = target >= 128
    known = ~unknown
    known_disagreement = np.logical_xor(occupied, target_occupied) & known
    relative_dir = Path("cache256") / row["split"] / row["sample_id"]
    output_dir = cache_root / relative_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    observation_path = output_dir / "observation.png"
    target_path = output_dir / "target.png"
    observation_image.save(observation_path, format="PNG", compress_level=6)
    target_image.save(target_path, format="PNG", compress_level=6)
    return {
        "split": row["split"],
        "source_split": row["source_split"],
        "sample_id": row["sample_id"],
        "observation_relpath": relative_dir.joinpath("observation.png").as_posix(),
        "target_relpath": relative_dir.joinpath("target.png").as_posix(),
        "observation_sha256": sha256(observation_path),
        "target_sha256": sha256(target_path),
        "source_observation_sha256": row["observation_sha256"],
        "source_target_sha256": row["target_sha256"],
        "unknown_fraction": float(unknown.mean()),
        "target_occupied_fraction": float(target_occupied.mean()),
        "known_cells": int(known.sum()),
        "known_disagreement_cells": int(known_disagreement.sum()),
        "known_disagreement_fraction": float(
            known_disagreement.sum() / max(1, int(known.sum()))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.common_manifest.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 8304:
        raise ValueError(f"unexpected source row count: {len(source_rows)}")
    tasks = [(row, str(args.output_root.resolve())) for row in source_rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(materialize, tasks, chunksize=8))
    rows.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    manifest = args.output_root / "model_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    summary = {
        "status": "prospective_uniform_256_cache",
        "counts": counts,
        "model_shape": list(MODEL_SIZE),
        "resampling": "nearest",
        "known_cell_agreement_checked": True,
        "known_cell_target_disagreement": {
            "samples_with_disagreement": sum(
                float(row["known_disagreement_fraction"]) > 0.0 for row in rows
            ),
            "mean_fraction": float(
                np.mean([float(row["known_disagreement_fraction"]) for row in rows])
            ),
            "maximum_fraction": float(
                max(float(row["known_disagreement_fraction"]) for row in rows)
            ),
            "policy": "report_conflict_and_keep_measured_cells_authoritative"
        },
        "common_manifest_sha256": sha256(args.common_manifest),
        "model_manifest_sha256": sha256(manifest),
    }
    summary_path = args.output_root / "cache_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "SHA256SUMS").write_text(
        f"{sha256(manifest)}  {manifest.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
