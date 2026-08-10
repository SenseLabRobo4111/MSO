"""Building-cluster paired analysis for the six-arm validation campaign."""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import (
    _building_bootstrap_interval,
    _coverage_auc,
    _holm,
    _randomization_p_value,
    _restricted_distance,
)


BASELINE_ARM = "observed_only"


def analyze_multiarm_campaign(
    summaries: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Compare every arm with the paired observed-only baseline.

    The independent unit is the operational building group. Seeds and
    floorplans inside a group are averaged before resampling or testing.
    """

    arms = [str(value) for value in protocol["arms"]]
    if BASELINE_ARM not in arms or len(arms) < 3:
        raise ValueError("multi-arm analysis requires observed_only and >=3 arms")
    rows: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        arm = str(summary.get("arm", ""))
        key = (
            str(summary.get("building_id", "")),
            str(summary.get("floorplan_id", "")),
            int(summary.get("seed", -1)),
        )
        if arm not in arms or not key[0] or not key[1] or key[2] < 0:
            raise ValueError("invalid multi-arm summary key")
        if arm in rows.setdefault(key, {}):
            raise ValueError(f"duplicate multi-arm summary: {key}, {arm}")
        rows[key][arm] = summary
    incomplete = [key for key, value in rows.items() if set(value) != set(arms)]
    if incomplete:
        raise ValueError(f"incomplete multi-arm pairs: {incomplete[:5]}")

    stats = protocol["statistics"]
    alpha = float(stats["alpha"])
    bootstrap_reps = int(stats["bootstrap_replicates"])
    bootstrap_seed = int(stats["bootstrap_seed"])
    randomization_reps = int(stats["randomization_replicates"])
    randomization_seed = int(stats["randomization_seed"])
    budget_m = float(protocol["planning"]["distance_budget_m"])
    endpoints = {
        "coverage_auc_normalized": (_coverage_auc, True),
        "restricted_distance_to_80_m": (
            lambda value: _restricted_distance(value, budget_m),
            False,
        ),
    }

    contrasts: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    contrast_index = 0
    for arm in arms:
        if arm == BASELINE_ARM:
            continue
        contrasts[arm] = {}
        for endpoint, (extractor, higher_is_better) in endpoints.items():
            by_building: dict[str, list[float]] = {}
            for (building, _floorplan, _seed), values in rows.items():
                effect = float(extractor(values[arm])) - float(
                    extractor(values[BASELINE_ARM])
                )
                if not higher_is_better:
                    effect = -effect
                by_building.setdefault(building, []).append(effect)
            cluster_effects = np.asarray(
                [float(np.mean(value)) for value in by_building.values()],
                dtype=float,
            )
            p_value, method, used = _randomization_p_value(
                cluster_effects,
                seed=randomization_seed + contrast_index,
                replicates=randomization_reps,
            )
            key = f"{arm}:{endpoint}"
            p_values[key] = p_value
            contrasts[arm][endpoint] = {
                "effect_direction": f"benefit_of_{arm}_versus_{BASELINE_ARM}",
                "building_level_effects": {
                    building: float(np.mean(values))
                    for building, values in sorted(by_building.items())
                },
                "building_weighted_mean_difference": float(
                    np.mean(cluster_effects)
                ),
                "building_cluster_bootstrap_interval": (
                    _building_bootstrap_interval(
                        cluster_effects,
                        seed=bootstrap_seed + contrast_index,
                        replicates=bootstrap_reps,
                        alpha=alpha,
                    )
                ),
                "raw_p_value": p_value,
                "randomization_method": method,
                "randomization_replicates_used": used,
            }
            contrast_index += 1

    adjusted = _holm(p_values, alpha)
    for key, inference in adjusted.items():
        arm, endpoint = key.split(":", 1)
        contrasts[arm][endpoint].update(inference)

    return {
        "status": "complete_multiarm_building_cluster_analysis",
        "baseline_arm": BASELINE_ARM,
        "arms": arms,
        "complete_pair_count": len(rows),
        "independent_building_count": len({key[0] for key in rows}),
        "multiplicity": (
            "Holm correction across every non-baseline arm by primary endpoint"
        ),
        "contrasts": contrasts,
        "benefit_claim_authorized": False,
        "interpretation": (
            "prospective validation-only contrasts; physical and historical "
            "claims are not authorized"
        ),
    }
