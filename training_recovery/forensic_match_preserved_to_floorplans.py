#!/usr/bin/env python3
"""Conservative image-geometric assignment of preserved targets to floorplans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_recovery.common import sha256_file, write_json  # noqa: E402


ACCEPTANCE = {
    "minimum_sift_inliers": 40,
    "minimum_sift_inlier_ratio": 0.35,
    "maximum_median_reprojection_error": 3.0,
    "minimum_top1_top2_inlier_margin": 25,
    "minimum_top1_top2_inlier_ratio": 1.5,
    "minimum_edge_f1_at_3px": 0.55,
    "minimum_binary_iou": 0.45,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preserved-root", type=Path, required=True)
    parser.add_argument("--floorplan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=20260807)
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Explicit train sample directory name; repeat for a targeted smoke audit.",
    )
    parser.add_argument("--orb-shortlist", type=int, default=20)
    parser.add_argument("--orb-features", type=int, default=3000)
    parser.add_argument("--sift-features", type=int, default=4000)
    return parser.parse_args()


def building_id(floorplan_id: str) -> str:
    prefix = floorplan_id.split("_", 1)[0]
    return f"numeric:{prefix}" if prefix.isdigit() else f"named:{floorplan_id}"


def direct_obstacle(gray: np.ndarray) -> np.ndarray:
    return (gray < 128).astype(np.uint8) * 255


def visible_kth_candidate(gray: np.ndarray) -> np.ndarray:
    """Reproduce the visible KTH reader/simulator raster transforms."""
    coordinates = np.column_stack(np.where(gray != 0))
    if not len(coordinates):
        raise ValueError("Source contains no nonzero pixels")
    row_min, col_min = coordinates.min(axis=0)
    row_max, col_max = coordinates.max(axis=0)
    image = gray[
        max(int(row_min) - 10, 0) : min(int(row_max) + 10, gray.shape[0]),
        max(int(col_min) - 10, 0) : min(int(col_max) + 10, gray.shape[1]),
    ]
    image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=5)
    image = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=4)
    image = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5))
    image = cv2.erode(image, np.ones((3, 3), np.uint8), iterations=3)
    scale = max(image.shape) / 1000.0
    image = cv2.resize(
        image,
        (0, 0),
        fx=1.0 / scale,
        fy=1.0 / scale,
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)
    # The visible simulator maps exact input zero to obstacle; intermediate
    # antialias values remain initialized as free in its map conversion.
    return (image == 0).astype(np.uint8) * 255


def enhance(binary: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(binary, (3, 3), 0)


@dataclass
class SourceVariant:
    floorplan_id: str
    building_id: str
    variant: str
    image: np.ndarray
    orb_keypoints: object
    orb_descriptors: np.ndarray | None
    sift_keypoints: object
    sift_descriptors: np.ndarray | None


@dataclass
class Verification:
    floorplan_id: str
    building_id: str
    variant: str
    orb_matches: int = 0
    orb_inliers: int = 0
    sift_matches: int = 0
    sift_inliers: int = 0
    sift_inlier_ratio: float = 0.0
    median_reprojection_error: float = math.inf
    scale: float = math.nan
    rotation_degrees: float = math.nan
    valid_overlap_fraction: float = 0.0
    binary_iou: float = 0.0
    binary_agreement: float = 0.0
    edge_precision_at_3px: float = 0.0
    edge_recall_at_3px: float = 0.0
    edge_f1_at_3px: float = 0.0


def ratio_matches(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    norm: int,
    ratio: float,
) -> list[cv2.DMatch]:
    if descriptors_a is None or descriptors_b is None:
        return []
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    return [first for first, second in pairs if first.distance < ratio * second.distance]


def estimate(
    keypoints_target: object,
    keypoints_source: object,
    matches: list[cv2.DMatch],
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    if len(matches) < 4:
        return None, None, np.empty(0, dtype=np.float64)
    target_points = np.float32(
        [keypoints_target[match.queryIdx].pt for match in matches]
    )
    source_points = np.float32(
        [keypoints_source[match.trainIdx].pt for match in matches]
    )
    matrix, mask = cv2.estimateAffinePartial2D(
        target_points,
        source_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=5000,
        confidence=0.999,
        refineIters=25,
    )
    if matrix is None or mask is None:
        return None, None, np.empty(0, dtype=np.float64)
    selected = mask.ravel().astype(bool)
    if int(np.sum(selected)) < 4:
        return None, None, np.empty(0, dtype=np.float64)
    predicted = cv2.transform(target_points[:, None, :], matrix)[:, 0, :]
    errors = np.linalg.norm(predicted - source_points, axis=1)
    scale = float(math.hypot(matrix[0, 0], matrix[0, 1]))
    target_inliers = target_points[selected]
    source_inliers = source_points[selected]
    target_span = float(np.linalg.norm(np.ptp(target_inliers, axis=0)))
    source_span = float(np.linalg.norm(np.ptp(source_inliers, axis=0)))
    if (
        not math.isfinite(scale)
        or not 0.1 <= scale <= 10.0
        or target_span < 64.0
        or source_span < 20.0
    ):
        return None, None, np.empty(0, dtype=np.float64)
    return matrix, selected, errors[selected]


def edge_metrics(
    target: np.ndarray, source: np.ndarray, matrix_target_to_source: np.ndarray
) -> dict[str, float]:
    size = (target.shape[1], target.shape[0])
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(source, matrix_target_to_source, size, flags=flags, borderValue=0)
    valid = cv2.warpAffine(
        np.full(source.shape, 255, dtype=np.uint8),
        matrix_target_to_source,
        size,
        flags=flags,
        borderValue=0,
    ) > 0
    target_positive = (target > 127) & valid
    warped_positive = (warped > 127) & valid
    union = int(np.sum(target_positive | warped_positive))
    intersection = int(np.sum(target_positive & warped_positive))
    binary_iou = intersection / max(1, union)
    agreement = float(np.mean(target_positive[valid] == warped_positive[valid])) if np.any(valid) else 0.0

    target_edge = cv2.Canny(target, 50, 150) > 0
    source_edge = cv2.Canny(warped, 50, 150) > 0
    target_distance = cv2.distanceTransform((~target_edge).astype(np.uint8), cv2.DIST_L2, 3)
    source_distance = cv2.distanceTransform((~source_edge).astype(np.uint8), cv2.DIST_L2, 3)
    source_edge_valid = source_edge & valid
    target_edge_valid = target_edge & valid
    precision = float(np.mean(target_distance[source_edge_valid] <= 3.0)) if np.any(source_edge_valid) else 0.0
    recall = float(np.mean(source_distance[target_edge_valid] <= 3.0)) if np.any(target_edge_valid) else 0.0
    edge_f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "valid_overlap_fraction": float(np.mean(valid)),
        "binary_iou": binary_iou,
        "binary_agreement": agreement,
        "edge_precision_at_3px": precision,
        "edge_recall_at_3px": recall,
        "edge_f1_at_3px": edge_f1,
    }


def source_variants(
    root: Path, orb: cv2.ORB, sift: cv2.SIFT
) -> list[SourceVariant]:
    variants: list[SourceVariant] = []
    for path in sorted(root.glob("*/GT.bmp")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(path)
        for name, image in (
            ("direct_dark_obstacle", direct_obstacle(gray)),
            ("visible_kth_candidate", visible_kth_candidate(gray)),
        ):
            processed = enhance(image)
            orb_keypoints, orb_descriptors = orb.detectAndCompute(processed, None)
            sift_keypoints, sift_descriptors = sift.detectAndCompute(processed, None)
            variants.append(
                SourceVariant(
                    path.parent.name,
                    building_id(path.parent.name),
                    name,
                    image,
                    orb_keypoints,
                    orb_descriptors,
                    sift_keypoints,
                    sift_descriptors,
                )
            )
    return variants


def verify_sample(
    target: np.ndarray,
    variants: list[SourceVariant],
    orb: cv2.ORB,
    sift: cv2.SIFT,
    shortlist: int,
) -> list[Verification]:
    target_processed = enhance(target)
    target_orb_keypoints, target_orb_descriptors = orb.detectAndCompute(
        target_processed, None
    )
    orb_ranked: list[tuple[int, int, SourceVariant]] = []
    for source in variants:
        matches = ratio_matches(
            target_orb_descriptors, source.orb_descriptors, cv2.NORM_HAMMING, 0.78
        )
        _, inlier_mask, _ = estimate(
            target_orb_keypoints, source.orb_keypoints, matches, 5.0
        )
        inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
        orb_ranked.append((inliers, len(matches), source))
    orb_ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)

    # Keep the best variant per floorplan before expensive SIFT verification.
    selected_floorplans: list[str] = []
    seen_floorplans: set[str] = set()
    for candidate in orb_ranked:
        if candidate[2].floorplan_id in seen_floorplans:
            continue
        selected_floorplans.append(candidate[2].floorplan_id)
        seen_floorplans.add(candidate[2].floorplan_id)
        if len(selected_floorplans) >= shortlist:
            break
    selected_set = set(selected_floorplans)
    selected = [
        candidate
        for candidate in orb_ranked
        if candidate[2].floorplan_id in selected_set
    ]

    target_sift_keypoints, target_sift_descriptors = sift.detectAndCompute(
        target_processed, None
    )
    verified: list[Verification] = []
    for orb_inliers, orb_match_count, source in selected:
        matches = ratio_matches(
            target_sift_descriptors, source.sift_descriptors, cv2.NORM_L2, 0.75
        )
        matrix, inlier_mask, errors = estimate(
            target_sift_keypoints, source.sift_keypoints, matches, 4.0
        )
        inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
        result = Verification(
            floorplan_id=source.floorplan_id,
            building_id=source.building_id,
            variant=source.variant,
            orb_matches=orb_match_count,
            orb_inliers=orb_inliers,
            sift_matches=len(matches),
            sift_inliers=inliers,
            sift_inlier_ratio=inliers / max(1, len(matches)),
        )
        if matrix is not None and len(errors):
            result.median_reprojection_error = float(np.median(errors))
            result.scale = float(math.hypot(matrix[0, 0], matrix[0, 1]))
            result.rotation_degrees = float(
                math.degrees(math.atan2(matrix[0, 1], matrix[0, 0]))
            )
            for key, value in edge_metrics(target, source.image, matrix).items():
                setattr(result, key, value)
        verified.append(result)
    verified.sort(
        key=lambda value: (
            value.sift_inliers,
            value.edge_f1_at_3px,
            value.binary_iou,
            value.orb_inliers,
        ),
        reverse=True,
    )
    best_per_floorplan: list[Verification] = []
    seen_floorplans.clear()
    for result in verified:
        if result.floorplan_id in seen_floorplans:
            continue
        best_per_floorplan.append(result)
        seen_floorplans.add(result.floorplan_id)
    return best_per_floorplan


def accepted(top1: Verification, top2: Verification) -> tuple[bool, dict[str, float]]:
    margin = top1.sift_inliers - top2.sift_inliers
    ratio = top1.sift_inliers / max(1, top2.sift_inliers)
    decision = (
        top1.sift_inliers >= ACCEPTANCE["minimum_sift_inliers"]
        and top1.sift_inlier_ratio >= ACCEPTANCE["minimum_sift_inlier_ratio"]
        and top1.median_reprojection_error
        <= ACCEPTANCE["maximum_median_reprojection_error"]
        and margin >= ACCEPTANCE["minimum_top1_top2_inlier_margin"]
        and ratio >= ACCEPTANCE["minimum_top1_top2_inlier_ratio"]
        and top1.edge_f1_at_3px >= ACCEPTANCE["minimum_edge_f1_at_3px"]
        and top1.binary_iou >= ACCEPTANCE["minimum_binary_iou"]
    )
    return decision, {
        "top1_top2_sift_inlier_margin": margin,
        "top1_top2_sift_inlier_ratio": ratio,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_root = args.preserved_root / "train"
    samples = sorted(path for path in train_root.iterdir() if path.is_dir())
    if args.sample_id:
        by_name = {path.name: path for path in samples}
        missing = sorted(set(args.sample_id).difference(by_name))
        if missing:
            raise ValueError(f"Unknown explicit train sample IDs: {missing}")
        selected_samples = [by_name[value] for value in args.sample_id]
    else:
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(
            rng.choice(len(samples), size=min(args.sample_count, len(samples)), replace=False)
        )
        selected_samples = [samples[int(index)] for index in indices]
    orb = cv2.ORB_create(nfeatures=args.orb_features, fastThreshold=5)
    sift = cv2.SIFT_create(nfeatures=args.sift_features)
    variants = source_variants(args.floorplan_root, orb, sift)

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for sample_index, sample in enumerate(selected_samples):
        target_path = sample / "local_map_0.png"
        target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise IOError(target_path)
        ranked = verify_sample(target, variants, orb, sift, args.orb_shortlist)
        if len(ranked) < 2:
            raise RuntimeError(f"Fewer than two candidates for {sample}")
        top1, top2 = ranked[:2]
        is_accepted, comparison = accepted(top1, top2)
        row: dict[str, object] = {
            "sample_id": f"train/{sample.name}",
            "target_sha256": sha256_file(target_path),
            "accepted": int(is_accepted),
            "assigned_floorplan_id": top1.floorplan_id if is_accepted else "",
            "assigned_building_id": top1.building_id if is_accepted else "",
        }
        row.update({f"top1_{key}": value for key, value in asdict(top1).items()})
        row.update({f"top2_{key}": value for key, value in asdict(top2).items()})
        row.update(comparison)
        rows.append(row)
        details.append(
            {
                "sample_id": row["sample_id"],
                "accepted": is_accepted,
                "top_candidates": [asdict(value) for value in ranked[:5]],
                **comparison,
            }
        )
        print(
            json.dumps(
                {
                    "completed": sample_index + 1,
                    "sample_id": row["sample_id"],
                    "top1": top1.floorplan_id,
                    "top1_inliers": top1.sift_inliers,
                    "top1_edge_f1": top1.edge_f1_at_3px,
                    "accepted": is_accepted,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    csv_path = args.output_dir / "assignment_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details_path = args.output_dir / "top5_details.json"
    write_json(details_path, details)
    sample_digest = hashlib.sha256()
    for sample in selected_samples:
        sample_digest.update(f"train/{sample.name}\n".encode("utf-8"))
    summary = {
        "status": "preserved_train_floorplan_assignment_forensic_audit",
        "preserved_partition_accessed": "train_only",
        "test_partition_accessed": False,
        "source_floorplans": len({value.floorplan_id for value in variants}),
        "source_variants_per_floorplan": 2,
        "samples_audited": len(rows),
        "sample_seed": args.sample_seed,
        "sample_ids_sha256": sample_digest.hexdigest(),
        "acceptance_gates": ACCEPTANCE,
        "accepted_samples": sum(int(row["accepted"]) for row in rows),
        "accepted_fraction": sum(int(row["accepted"]) for row in rows)
        / max(1, len(rows)),
        "assignment_rows_sha256": sha256_file(csv_path),
        "top5_details_sha256": sha256_file(details_path),
    }
    write_json(args.output_dir / "assignment_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
