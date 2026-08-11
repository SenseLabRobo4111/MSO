#!/usr/bin/env python3
"""Generate a new floorplan-disjoint occupancy-prediction dataset.

This is an explicitly reconstructed experiment.  It does not infer provenance
for the preserved 8,304-sample archive and does not recreate a historical
split.  Every generated sample retains its real source floorplan identifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_recovery import RECONSTRUCTION_STATUS  # noqa: E402
from mso_recovery.common import sha256_file, stable_seed, write_json  # noqa: E402


@dataclass(frozen=True)
class GenerationConfig:
    image_size: int = 256
    working_resolution_m_per_cell: float = 0.04
    samples_per_floorplan: int = 48
    split_seed: int = 20260807
    sample_seed: int = 4111
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    laser_range_cells: int = 125
    laser_rays: int = 360
    minimum_scans: int = 4
    maximum_scans: int = 14
    movement_cells_per_scan: int = 10
    minimum_unknown_fraction: float = 0.20
    maximum_unknown_fraction: float = 0.92
    content_padding_cells: int = 8
    maximum_attempts_per_sample: int = 40


MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "group_key",
    "group_kind",
    "building_id",
    "floorplan_id",
    "source_relpath",
    "source_sha256",
    "resolution_m_per_cell",
    "working_resolution_m_per_cell",
    "generation_seed",
    "scan_count",
    "rotation_quadrants",
    "obs_relpath",
    "target_relpath",
    "obs_sha256",
    "target_sha256",
    "unknown_fraction",
    "occupied_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-floorplan", type=int, default=48)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--sample-seed", type=int, default=4111)
    parser.add_argument(
        "--limit-floorplans",
        type=int,
        help="Smoke-test only: deterministically retain this many floorplans.",
    )
    return parser.parse_args()


def safe_floorplan_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"Unsafe empty floorplan ID derived from {value!r}")
    return safe


def building_id_from_floorplan(floorplan_id: str) -> str:
    """Return the source collection's building-level grouping key.

    Numeric source IDs may have multiple layouts after an underscore.  Those
    layouts are kept together.  Non-numeric named maps are singleton groups.
    """
    match = re.match(r"^(\d+)(?:_|$)", floorplan_id)
    return match.group(1) if match else f"named:{floorplan_id}"


def discover_floorplans(source_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for gt_path in sorted(source_root.glob("*/GT.bmp")):
        floorplan_id = gt_path.parent.name
        metadata_path = gt_path.with_name("GT.json")
        resolution = 0.1
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution = float(metadata.get("resolution", resolution))
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"Invalid resolution for {floorplan_id}: {resolution}")
        records.append(
            {
                "floorplan_id": floorplan_id,
                "building_id": building_id_from_floorplan(floorplan_id),
                "safe_id": safe_floorplan_id(floorplan_id),
                "gt_path": gt_path,
                "source_relpath": gt_path.relative_to(source_root).as_posix(),
                "source_sha256": sha256_file(gt_path),
                "resolution_m_per_cell": resolution,
            }
        )
    if not records:
        raise ValueError(f"No */GT.bmp floorplans found below {source_root}")
    ids = [str(record["floorplan_id"]) for record in records]
    safe_ids = [str(record["safe_id"]) for record in records]
    if len(ids) != len(set(ids)) or len(safe_ids) != len(set(safe_ids)):
        raise ValueError("Floorplan identifiers are not unique")
    return records


def assign_splits(
    records: list[dict[str, object]], config: GenerationConfig
) -> dict[str, str]:
    if len(records) < 3:
        raise ValueError("At least three floorplans are required for train/val/test")
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(str(record["building_id"]), []).append(
            str(record["floorplan_id"])
        )
    building_ids = np.array(sorted(groups), dtype=object)
    rng = np.random.default_rng(config.split_seed)
    rng.shuffle(building_ids)
    n_val = max(1, int(round(len(records) * config.validation_fraction)))
    n_test = max(1, int(round(len(records) * config.test_fraction)))
    if n_val + n_test >= len(records):
        raise ValueError("Split fractions leave no training floorplans")

    def exact_group_subset(candidates: list[str], target: int) -> set[str]:
        paths: dict[int, tuple[str, ...]] = {0: ()}
        for building_id in candidates:
            size = len(groups[building_id])
            for total, path in sorted(paths.items(), reverse=True):
                new_total = total + size
                if new_total <= target and new_total not in paths:
                    paths[new_total] = path + (building_id,)
        if target not in paths:
            raise ValueError(
                f"Cannot form an atomic building split with {target} floorplans"
            )
        return set(paths[target])

    ordered = [str(value) for value in building_ids]
    validation_groups = exact_group_subset(ordered, n_val)
    remaining = [value for value in ordered if value not in validation_groups]
    test_groups = exact_group_subset(remaining, n_test)
    assignments: dict[str, str] = {}
    for building_id, floorplan_ids in groups.items():
        split = (
            "val"
            if building_id in validation_groups
            else "test"
            if building_id in test_groups
            else "train"
        )
        for floorplan_id in floorplan_ids:
            assignments[floorplan_id] = split

    building_splits: dict[str, set[str]] = {}
    for record in records:
        building_splits.setdefault(str(record["building_id"]), set()).add(
            assignments[str(record["floorplan_id"])]
        )
    if any(len(splits) != 1 for splits in building_splits.values()):
        raise AssertionError("A building was assigned to multiple splits")
    return assignments


def load_occupied_map(
    path: Path, source_resolution: float, config: GenerationConfig
) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    occupied = gray < 128

    # The source collection contains both white-background line maps and
    # black-background filled maps.  Occupancy is consistently dark; the
    # minority polarity is used only to locate the content bounding box.
    content = occupied if float(occupied.mean()) < 0.5 else ~occupied
    ys, xs = np.where(content)
    if not len(ys):
        raise ValueError(f"No floorplan content in {path}")
    pad = config.content_padding_cells
    y0, y1 = max(0, int(ys.min()) - pad), min(gray.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(gray.shape[1], int(xs.max()) + pad + 1)
    occupied = occupied[y0:y1, x0:x1].copy()
    if min(occupied.shape) < 8:
        raise ValueError(f"Degenerate content box in {path}: {occupied.shape}")

    # The archived floorplans are 0.10 m/cell, while the predictor's 256-cell
    # window was generated on the simulator's 0.04 m/cell lattice.  Nearest
    # resampling preserves binary occupancy and gives a 10.24 m model window.
    scale = float(source_resolution) / config.working_resolution_m_per_cell
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid source-to-working resolution scale: {scale}")
    if not math.isclose(scale, 1.0):
        occupied = cv2.resize(
            occupied.astype(np.uint8),
            (
                max(8, int(round(occupied.shape[1] * scale))),
                max(8, int(round(occupied.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Close the synthetic world at its crop boundary.  A large occupied pad
    # then makes every fixed-size local crop bounds-safe without inventing
    # traversable area beyond the source map.
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    outer_pad = config.image_size // 2 + config.laser_range_cells + 4
    return np.pad(occupied, outer_pad, mode="constant", constant_values=True)


def select_navigable_component(occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    free = (~occupied).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(free, 8)
    if count <= 1:
        raise ValueError("Floorplan has no free-space component")
    h, w = occupied.shape
    center = np.array([w / 2.0, h / 2.0])
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 64:
            continue
        distance = float(np.linalg.norm(centroids[label] - center))
        # Prefer substantial central components.  The content crop has only an
        # eight-cell exterior margin, so this rejects the common outside ring.
        score = area / (1.0 + 0.02 * distance)
        candidates.append((score, label))
    if not candidates:
        raise ValueError("No navigable free-space component with at least 64 cells")
    selected = max(candidates)[1]
    component = labels == selected
    clearance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    starts = np.argwhere(clearance >= 1.5)
    if not len(starts):
        starts = np.argwhere(component)
    return component, starts


def ray_offsets(ray_count: int, radius: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rays: list[tuple[np.ndarray, np.ndarray]] = []
    radii = np.arange(1, radius + 1, dtype=np.float64)
    for angle in np.linspace(0.0, 2.0 * np.pi, ray_count, endpoint=False):
        dy = np.rint(np.sin(angle) * radii).astype(np.int32)
        dx = np.rint(np.cos(angle) * radii).astype(np.int32)
        keep = np.ones(len(dy), dtype=bool)
        keep[1:] = (dy[1:] != dy[:-1]) | (dx[1:] != dx[:-1])
        rays.append((dy[keep], dx[keep]))
    return rays


MOVES = np.asarray(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int32,
)


def scan_from_pose(
    occupied: np.ndarray,
    known: np.ndarray,
    pose: np.ndarray,
    rays: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    y, x = int(pose[0]), int(pose[1])
    known[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True
    h, w = occupied.shape
    for dy, dx in rays:
        yy, xx = y + dy, x + dx
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yy, xx = yy[valid], xx[valid]
        if not len(yy):
            continue
        hits = np.flatnonzero(occupied[yy, xx])
        end = int(hits[0]) + 1 if len(hits) else len(yy)
        known[yy[:end], xx[:end]] = True


def move_random_walk(
    pose: np.ndarray, component: np.ndarray, rng: np.random.Generator, steps: int
) -> np.ndarray:
    current = pose.copy()
    heading = int(rng.integers(0, len(MOVES)))
    for _ in range(steps):
        preference = [heading]
        preference.extend(int(value) for value in rng.permutation(len(MOVES)))
        for direction in preference:
            candidate = current + MOVES[direction]
            if component[int(candidate[0]), int(candidate[1])]:
                current = candidate
                heading = direction
                break
    return current


def centered_crop(array: np.ndarray, pose: np.ndarray, size: int) -> np.ndarray:
    y, x = int(pose[0]), int(pose[1])
    half = size // 2
    crop = array[y - half : y - half + size, x - half : x - half + size]
    if crop.shape != (size, size):
        raise ValueError(f"Unexpected crop shape {crop.shape}; source padding failed")
    return crop


def generate_sample(
    occupied: np.ndarray,
    component: np.ndarray,
    starts: np.ndarray,
    rng: np.random.Generator,
    rays: list[tuple[np.ndarray, np.ndarray]],
    config: GenerationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    pose = starts[int(rng.integers(0, len(starts)))].astype(np.int32)
    known = np.zeros(occupied.shape, dtype=bool)
    scans = int(rng.integers(config.minimum_scans, config.maximum_scans + 1))
    for _ in range(scans):
        scan_from_pose(occupied, known, pose, rays)
        pose = move_random_walk(
            pose, component, rng, config.movement_cells_per_scan
        )
    scan_from_pose(occupied, known, pose, rays)

    target = centered_crop(occupied, pose, config.image_size)
    observed = centered_crop(known, pose, config.image_size)
    unknown_fraction = float((~observed).mean())
    occupied_fraction = float(target.mean())
    if not (
        config.minimum_unknown_fraction
        <= unknown_fraction
        <= config.maximum_unknown_fraction
    ):
        return None
    if not (0.002 <= occupied_fraction <= 0.95):
        return None
    if int((observed & ~target).sum()) < 32 or int((observed & target).sum()) < 4:
        return None

    observation = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
    observation[:, :, 0] = (observed & target).astype(np.uint8) * 255
    observation[:, :, 1] = (~observed).astype(np.uint8) * 255
    observation[:, :, 2] = (observed & ~target).astype(np.uint8) * 255
    target_image = target.astype(np.uint8) * 255
    rotation = int(rng.integers(0, 4))
    if rotation:
        observation = np.rot90(observation, rotation).copy()
        target_image = np.rot90(target_image, rotation).copy()
    return observation, target_image, {
        "scan_count": scans + 1,
        "rotation_quadrants": rotation,
        "unknown_fraction": unknown_fraction,
        "occupied_fraction": occupied_fraction,
    }


def save_png(path: Path, array: np.ndarray, mode: str) -> None:
    Image.fromarray(array, mode=mode).save(path, format="PNG", compress_level=6)


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        samples_per_floorplan=args.samples_per_floorplan,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
    )
    if config.samples_per_floorplan <= 0:
        raise SystemExit("--samples-per-floorplan must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    incomplete = output_root / "GENERATION_INCOMPLETE"
    incomplete.write_text(RECONSTRUCTION_STATUS + "\n", encoding="utf-8")

    records = discover_floorplans(args.source_root.resolve())
    if args.limit_floorplans is not None:
        if args.limit_floorplans < 3:
            raise SystemExit("--limit-floorplans must be at least 3")
        grouped: dict[str, list[dict[str, object]]] = {}
        for record in records:
            grouped.setdefault(str(record["building_id"]), []).append(record)
        selection_rng = np.random.default_rng(config.split_seed)
        building_ids = np.array(sorted(grouped), dtype=object)
        selection_rng.shuffle(building_ids)
        selected_records: list[dict[str, object]] = []
        for building_id in building_ids:
            selected_records.extend(grouped[str(building_id)])
            if len(selected_records) >= min(args.limit_floorplans, len(records)):
                break
        records = sorted(selected_records, key=lambda row: str(row["floorplan_id"]))
    assignments = assign_splits(records, config)

    source_hash_splits: dict[str, set[str]] = {}
    for record in records:
        source_hash_splits.setdefault(str(record["source_sha256"]), set()).add(
            assignments[str(record["floorplan_id"])]
        )
    source_hash_leakage = {
        digest: splits
        for digest, splits in source_hash_splits.items()
        if len(splits) != 1
    }
    if source_hash_leakage:
        raise ValueError("An exact source GT duplicate crosses splits")
    rays = ray_offsets(config.laser_rays, config.laser_range_cells)

    manifest_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for floorplan_index, record in enumerate(records, start=1):
        floorplan_id = str(record["floorplan_id"])
        building_id = str(record["building_id"])
        split = assignments[floorplan_id]
        occupied = load_occupied_map(
            Path(record["gt_path"]),
            float(record["resolution_m_per_cell"]),
            config,
        )
        component, starts = select_navigable_component(occupied)
        floorplan_seed = stable_seed(config.sample_seed, floorplan_id)
        rng = np.random.default_rng(floorplan_seed)
        split_rows.append(
            {
                "floorplan_id": floorplan_id,
                "building_id": building_id,
                "split": split,
                "source_relpath": record["source_relpath"],
                "source_sha256": record["source_sha256"],
                "resolution_m_per_cell": record["resolution_m_per_cell"],
            }
        )
        accepted = 0
        attempts = 0
        while accepted < config.samples_per_floorplan:
            attempts += 1
            if attempts > config.samples_per_floorplan * config.maximum_attempts_per_sample:
                raise RuntimeError(
                    f"Could generate only {accepted}/{config.samples_per_floorplan} "
                    f"valid samples for {floorplan_id}"
                )
            sample_seed = stable_seed(floorplan_seed, attempts)
            sample_rng = np.random.default_rng(sample_seed)
            generated = generate_sample(
                occupied, component, starts, sample_rng, rays, config
            )
            if generated is None:
                continue
            observation, target, details = generated
            sample_id = f"{record['safe_id']}__{accepted:03d}"
            sample_dir = output_root / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            obs_path = sample_dir / "local_obs_0.png"
            target_path = sample_dir / "local_map_0.png"
            save_png(obs_path, observation, "RGB")
            save_png(target_path, target, "L")
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "group_key": building_id,
                    "group_kind": "building",
                    "building_id": building_id,
                    "floorplan_id": floorplan_id,
                    "source_relpath": record["source_relpath"],
                    "source_sha256": record["source_sha256"],
                    "resolution_m_per_cell": record["resolution_m_per_cell"],
                    "working_resolution_m_per_cell": config.working_resolution_m_per_cell,
                    "generation_seed": sample_seed,
                    "scan_count": details["scan_count"],
                    "rotation_quadrants": details["rotation_quadrants"],
                    "obs_relpath": obs_path.relative_to(output_root).as_posix(),
                    "target_relpath": target_path.relative_to(output_root).as_posix(),
                    "obs_sha256": sha256_file(obs_path),
                    "target_sha256": sha256_file(target_path),
                    "unknown_fraction": f"{details['unknown_fraction']:.9f}",
                    "occupied_fraction": f"{details['occupied_fraction']:.9f}",
                }
            )
            accepted += 1
        print(
            f"[{floorplan_index:03d}/{len(records):03d}] {floorplan_id}: "
            f"split={split}, samples={accepted}, attempts={attempts}",
            flush=True,
        )

    manifest_path = output_root / "manifest.csv"

    pair_hash_splits: dict[tuple[str, str], set[str]] = {}
    for row in manifest_rows:
        pair = (str(row["obs_sha256"]), str(row["target_sha256"]))
        pair_hash_splits.setdefault(pair, set()).add(str(row["split"]))
    processed_pair_leakage = {
        pair: splits for pair, splits in pair_hash_splits.items() if len(splits) != 1
    }
    if processed_pair_leakage:
        raise ValueError("An exact processed observation-target pair crosses splits")

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    split_path = output_root / "floorplan_split.csv"
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        fields = tuple(split_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(split_rows, key=lambda row: str(row["floorplan_id"])))

    floorplan_counts = {
        split: sum(row["split"] == split for row in split_rows)
        for split in ("train", "val", "test")
    }
    building_counts = {
        split: len(
            {
                str(row["building_id"])
                for row in split_rows
                if row["split"] == split
            }
        )
        for split in ("train", "val", "test")
    }
    sample_counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in ("train", "val", "test")
    }
    summary = {
        "status": RECONSTRUCTION_STATUS,
        "source_root_recorded_at_generation": str(args.source_root.resolve()),
        "source_floorplan_count": len(records),
        "source_building_count": len({str(row["building_id"]) for row in split_rows}),
        "building_counts": building_counts,
        "floorplan_counts": floorplan_counts,
        "sample_counts": sample_counts,
        "configuration": asdict(config),
        "manifest_sha256": sha256_file(manifest_path),
        "floorplan_split_sha256": sha256_file(split_path),
        "cross_split_exact_source_gt_duplicates": 0,
        "cross_split_exact_processed_pair_duplicates": 0,
    }
    write_json(output_root / "dataset_summary.json", summary)
    incomplete.unlink()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
