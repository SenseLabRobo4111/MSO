"""Prospectively specified building-cluster paired campaign analysis."""

from __future__ import annotations

from itertools import product
import math
from typing import Any, Callable

import numpy as np


PAIR_ARMS = ("prediction_on", "observed_only")


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (
        math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _nested(summary: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _coverage_auc(summary: dict[str, Any]) -> float:
    value = _finite_number(summary.get("coverage_auc_normalized"), "coverage AUC")
    if not 0.0 <= value <= 1.0:
        raise ValueError("coverage AUC is outside [0, 1]")
    return value


def _restricted_distance(summary: dict[str, Any], budget_m: float) -> float:
    restricted = _finite_number(
        summary.get("restricted_distance_to_80_m"), "restricted distance"
    )
    if not 0.0 <= restricted <= budget_m:
        raise ValueError("restricted distance is outside the fixed budget")
    censored = summary.get("distance_to_80_percent_censored")
    raw = summary.get("distance_to_80_percent_m")
    if censored is True:
        if raw is not None or not math.isclose(
            restricted, budget_m, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("censored distance does not equal the restriction point")
    elif censored is False:
        raw_value = _finite_number(raw, "uncensored distance")
        if not math.isclose(
            raw_value, restricted, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("uncensored and restricted distances differ")
    else:
        raise ValueError("distance censoring indicator is not Boolean")
    return restricted


def _holm(p_values: dict[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    running = 0.0
    output: dict[str, dict[str, Any]] = {}
    total = len(ordered)
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * p_values[name])
        running = max(running, adjusted)
        output[name] = {
            "raw_p_value": p_values[name],
            "holm_adjusted_p_value": running,
            "reject_at_familywise_alpha": running <= alpha,
        }
    return output


def _randomization_p_value(
    building_effects: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, str, int]:
    observed = abs(float(np.mean(building_effects)))
    count = int(building_effects.size)
    if count <= 20 and (1 << count) <= replicates:
        statistics = [
            abs(float(np.mean(building_effects * np.asarray(signs, dtype=float))))
            for signs in product((-1.0, 1.0), repeat=count)
        ]
        extreme = sum(value >= observed - 1e-15 for value in statistics)
        return float(extreme / len(statistics)), "exact_sign_randomization", len(
            statistics
        )
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(replicates):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=count, replace=True)
        statistic = abs(float(np.mean(building_effects * signs)))
        extreme += int(statistic >= observed - 1e-15)
    return (
        float((extreme + 1) / (replicates + 1)),
        "monte_carlo_sign_randomization",
        replicates,
    )


def _building_bootstrap_interval(
    building_effects: np.ndarray,
    *,
    seed: int,
    replicates: int,
    alpha: float,
) -> list[float]:
    rng = np.random.default_rng(seed)
    count = int(building_effects.size)
    indices = rng.integers(0, count, size=(replicates, count))
    estimates = np.mean(building_effects[indices], axis=1)
    return [
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    ]


def _paired_rows(
    summaries: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        key = (
            str(summary.get("building_id", "")),
            str(summary.get("floorplan_id", "")),
            int(summary.get("seed", -1)),
        )
        if not key[0] or not key[1] or key[2] < 0:
            raise ValueError("summary has an invalid building/floorplan/seed key")
        arm = summary.get("arm")
        if arm not in PAIR_ARMS:
            raise ValueError(f"summary has an unknown arm: {arm!r}")
        if arm in pairs.setdefault(key, {}):
            raise ValueError(f"duplicate paired run: {key}, {arm}")
        pairs[key][str(arm)] = summary
    incomplete = [key for key, arms in pairs.items() if set(arms) != set(PAIR_ARMS)]
    if incomplete:
        raise ValueError(f"campaign contains incomplete paired keys: {incomplete}")
    return pairs


def _cluster_effects(
    pairs: dict[tuple[str, str, int], dict[str, dict[str, Any]]],
    extractor: Callable[[dict[str, Any]], float],
    *,
    prediction_minus_observed: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    by_building: dict[str, list[float]] = {}
    for (building, _floorplan, _seed), arms in pairs.items():
        prediction = extractor(arms["prediction_on"])
        observed = extractor(arms["observed_only"])
        difference = prediction - observed
        if not prediction_minus_observed:
            difference = -difference
        by_building.setdefault(building, []).append(float(difference))
    averaged = {
        building: float(np.mean(values))
        for building, values in sorted(by_building.items())
    }
    return np.asarray(list(averaged.values()), dtype=float), averaged


def analyze_paired_campaign(
    summaries: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Analyze complete arm pairs while treating building as independent."""

    pairs = _paired_rows(summaries)
    statistics = protocol["statistics"]
    alpha = float(statistics["alpha"])
    randomization_seed = int(statistics["randomization_seed"])
    randomization_replicates = int(statistics["randomization_replicates"])
    bootstrap_seed = int(statistics["bootstrap_seed"])
    bootstrap_replicates = int(statistics["bootstrap_replicates"])
    budget_m = float(protocol["planning"]["distance_budget_m"])

    endpoint_specs = {
        "coverage_auc_normalized": (
            _coverage_auc,
            True,
        ),
        "restricted_distance_to_80_m": (
            lambda row: _restricted_distance(row, budget_m),
            False,
        ),
    }
    endpoints: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for index, (name, (extractor, direction)) in enumerate(endpoint_specs.items()):
        effects, building_values = _cluster_effects(
            pairs, extractor, prediction_minus_observed=direction
        )
        if effects.size < 1:
            raise ValueError("paired analysis has no independent buildings")
        p_value, method, actual_replicates = _randomization_p_value(
            effects,
            seed=randomization_seed + index,
            replicates=randomization_replicates,
        )
        raw_p_values[name] = p_value
        endpoints[name] = {
            "effect_direction": statistics["effect_directions"][name],
            "building_level_effects": building_values,
            "estimate": float(np.mean(effects)),
            "building_weighted_mean_difference": float(np.mean(effects)),
            "building_cluster_bootstrap_interval": _building_bootstrap_interval(
                effects,
                seed=bootstrap_seed + index,
                replicates=bootstrap_replicates,
                alpha=alpha,
            ),
            "bootstrap_seed_used": bootstrap_seed + index,
            "randomization_method": method,
            "randomization_seed_used": randomization_seed + index,
            "randomization_replicates_used": actual_replicates,
            "raw_p_value": p_value,
        }
    adjusted = _holm(raw_p_values, alpha)
    for name, inference in adjusted.items():
        endpoints[name].update(inference)

    safety_paths = {
        "collision_count": ("collision_count",),
        "collision_rate_per_motion_attempt": (
            "collision_rate_per_motion_attempt",
        ),
        "motion_attempt_count": ("motion", "attempted"),
        "motion_attempt_rate_per_robot_tick": (
            "motion_attempt_rate_per_robot_tick",
        ),
        "no_route_count": ("motion", "no_route"),
        "replan_count": ("planning", "replans"),
        "stuck_episode_count": ("motion", "stuck_episode_count"),
        "cumulative_team_distance_m": ("cumulative_team_distance_m",),
        "persistent_map_coverage_auc_normalized": (
            "coverage_auc_normalized",
        ),
        "persistent_map_coverage_at_budget": ("coverage_at_budget",),
        "gate_wrong_accept_rate": ("gate_wrong_accept_rate",),
        "final_decision_false_accept_rate": (
            "final_decision_false_accept_rate",
        ),
        "final_decision_false_reject_count": (
            "final_decision_false_reject_count",
        ),
        "unsafe_commit_unrecovered_count": (
            "unsafe_commit_unrecovered_count",
        ),
    }
    safety: dict[str, Any] = {}
    for name, path in safety_paths.items():
        values: list[float] = []
        values_by_building: dict[str, list[float]] = {}
        undefined = 0
        for (building, _floorplan, _seed), arms in pairs.items():
            prediction = _nested(arms["prediction_on"], path)
            observed = _nested(arms["observed_only"], path)
            if prediction is None or observed is None:
                undefined += 1
                continue
            difference = (
                _finite_number(prediction, name)
                - _finite_number(observed, name)
            )
            values.append(difference)
            values_by_building.setdefault(building, []).append(difference)
        building_values = [
            float(np.mean(building_pairs))
            for building_pairs in values_by_building.values()
        ]
        safety[name] = {
            "effect_direction": "prediction_on_minus_observed_only",
            "defined_pair_count": len(values),
            "undefined_pair_count": undefined,
            "defined_building_count": len(building_values),
            "building_weighted_mean_difference": (
                float(np.mean(building_values)) if building_values else None
            ),
            "inferential_test": None,
        }

    censored = {
        arm: sum(
            bool(rows[arm].get("distance_to_80_percent_censored"))
            for rows in pairs.values()
        )
        for arm in PAIR_ARMS
    }
    unsafe_runs = sorted(
        str(summary.get("run_id", (
            f"{summary['floorplan_id']}__seed{summary['seed']}__{summary['arm']}"
        )))
        for summary in summaries
        if summary.get("run_safety_failure") is True
    )
    held_out_campaign = bool(summaries) and all(
        summary.get("split") == "test" for summary in summaries
    )
    frozen_protocol = (
        protocol["freeze"]["current_protocol_state"] == "ready_for_freeze"
        and protocol["freeze"]["test_execution_enabled"] is True
    )
    audit_dataset_eligible = not unsafe_runs and held_out_campaign and frozen_protocol
    return {
        "status": (
            "complete_with_unrecovered_safety_failures"
            if unsafe_runs
            else "complete_building_cluster_paired_analysis"
        ),
        "audit_dataset_eligible": audit_dataset_eligible,
        "benefit_claim_authorized": False,
        "benefit_claim_rule": (
            "this audit reports prespecified effects and uncertainty but never "
            "authorizes a directional benefit claim"
        ),
        "audit_eligibility_checks": {
            "held_out_campaign": held_out_campaign,
            "frozen_protocol": frozen_protocol,
            "no_unrecovered_safety_failure": not unsafe_runs,
        },
        "unrecovered_safety_failure_runs": unsafe_runs,
        "prospective_protocol_state": protocol["freeze"][
            "current_protocol_state"
        ],
        "independent_unit": "building",
        "pair_keys": ["building_id", "floorplan_id", "seed"],
        "complete_pair_count": len(pairs),
        "independent_building_count": len({key[0] for key in pairs}),
        "alpha": alpha,
        "alternative": "two_sided",
        "multiplicity": statistics["multiple_primary_endpoints"],
        "censoring_policy": statistics["censoring_policy"],
        "censored_runs_by_arm": censored,
        "zero_denominator_policy": statistics["zero_denominator_policy"],
        "randomization_seed": randomization_seed,
        "randomization_replicates_planned": randomization_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "primary_endpoints": endpoints,
        "joint_safety_descriptive": safety,
    }
