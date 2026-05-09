from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElasticityEstimate:
    incentive_value: float
    treatment_avg_occupancy: float
    control_avg_occupancy: float
    lift: float
    n_treatment: int
    n_control: int
    is_significant: bool


def estimate_elasticity(
    historical_aggregates: List[Dict],
) -> List[ElasticityEstimate]:
    if not historical_aggregates:
        return []

    treatment_buckets: Dict[float, List[float]] = {}
    control_occupancies: List[float] = []

    for agg in historical_aggregates:
        outcome = agg.get("outcomeSnapshot")
        if not outcome or not isinstance(outcome, dict):
            continue

        occupancy = agg["finalOccupancyPct"]
        is_control = outcome.get("isControlGroup", False)

        if is_control:
            control_occupancies.append(occupancy)
        else:
            incentive_val = outcome.get("incentiveValue", 0)
            if incentive_val and incentive_val > 0:
                bucket = round(incentive_val / 25) * 25
                treatment_buckets.setdefault(bucket, []).append(occupancy)

    if not control_occupancies:
        logger.warning("No control group data found for elasticity estimation")
        return []

    control_avg = sum(control_occupancies) / len(control_occupancies)

    estimates = []
    for value, occupancies in sorted(treatment_buckets.items()):
        treatment_avg = sum(occupancies) / len(occupancies)
        lift = treatment_avg - control_avg
        estimates.append(ElasticityEstimate(
            incentive_value=value,
            treatment_avg_occupancy=treatment_avg,
            control_avg_occupancy=control_avg,
            lift=lift,
            n_treatment=len(occupancies),
            n_control=len(control_occupancies),
            is_significant=len(occupancies) >= 10 and len(control_occupancies) >= 5,
        ))

    logger.info(
        "Elasticity estimates from %d treatment and %d control observations: %s",
        sum(len(v) for v in treatment_buckets.values()),
        len(control_occupancies),
        [(e.incentive_value, f"{e.lift:+.3f}") for e in estimates],
    )

    return estimates


def build_lift_curve(
    estimates: List[ElasticityEstimate],
) -> Optional[Dict[str, Any]]:
    significant = [e for e in estimates if e.is_significant]
    if len(significant) < 3:
        return None

    from sklearn.isotonic import IsotonicRegression

    doses = np.array([e.incentive_value for e in significant])
    responses = np.array([e.lift for e in significant])

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(doses, responses)
    fitted = iso.predict(doses)

    return {
        "dose_values": doses.tolist(),
        "response_values": fitted.tolist(),
        "n_points": len(significant),
    }


def lookup_learned_lift(
    lift_curve: Optional[Dict],
    incentive_value: float,
) -> Optional[float]:
    if lift_curve is None or not lift_curve.get("dose_values"):
        return None

    doses = lift_curve["dose_values"]
    responses = lift_curve["response_values"]

    if incentive_value <= doses[0]:
        return responses[0]
    if incentive_value >= doses[-1]:
        return responses[-1]

    for i in range(len(doses) - 1):
        if doses[i] <= incentive_value <= doses[i + 1]:
            span = doses[i + 1] - doses[i]
            if span == 0:
                return responses[i]
            t = (incentive_value - doses[i]) / span
            return responses[i] + t * (responses[i + 1] - responses[i])

    return responses[-1]
