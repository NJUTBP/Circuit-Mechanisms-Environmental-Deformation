"""Data-independent synthetic checks for the standalone TE implementation.

These checks are software diagnostics, not evidence about the manuscript data.
They exercise three properties that should hold in simple controlled systems:

1. a known source-to-target coupling is directionally recovered;
2. the generating prediction delay is preferred over longer candidate delays;
3. conditioning on an observed common driver strongly reduces spurious TE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .estimator import estimate_cross_validated_predictive_te, estimate_gcmi_te
from .lagged import build_lagged_samples
from .metadata import atomic_write_json
from .types import Segment


def _directional_segments(
    seed: int,
    *,
    n_trajectories: int = 3,
    n_time: int = 1400,
) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    forward: list[Segment] = []
    reverse: list[Segment] = []
    master = np.random.SeedSequence(seed)
    for trajectory_index, child in enumerate(master.spawn(n_trajectories)):
        rng = np.random.default_rng(child)
        source = np.zeros((n_time, 2), dtype=np.float64)
        target = np.zeros((n_time, 2), dtype=np.float64)
        for time_index in range(2, n_time):
            source[time_index] = (
                0.72 * source[time_index - 1]
                + rng.normal(scale=0.70, size=2)
            )
            target[time_index] = (
                0.55 * target[time_index - 1]
                + 0.65 * source[time_index - 1]
                + rng.normal(scale=0.55, size=2)
            )
        trajectory_id = f"trajectory_{trajectory_index:02d}"
        forward.append(
            Segment(
                source=source,
                target=target,
                segment_id=f"forward_{trajectory_id}",
                trajectory_id=trajectory_id,
            )
        )
        reverse.append(
            Segment(
                source=target,
                target=source,
                segment_id=f"reverse_{trajectory_id}",
                trajectory_id=trajectory_id,
            )
        )
    return tuple(forward), tuple(reverse)


def _common_driver_values(seed: int, *, n_time: int = 3200) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    driver = np.zeros((n_time, 1), dtype=np.float64)
    source = np.zeros((n_time, 1), dtype=np.float64)
    target = np.zeros((n_time, 1), dtype=np.float64)
    for time_index in range(3, n_time):
        driver[time_index] = (
            0.80 * driver[time_index - 1] + rng.normal(scale=0.70)
        )
        source[time_index] = driver[time_index - 1] + rng.normal(scale=0.25)
        target[time_index] = driver[time_index - 2] + rng.normal(scale=0.25)

    unconditioned = build_lagged_samples(
        [Segment(source=source, target=target)],
        source_history=1,
        target_history=1,
        prediction_delay=1,
    )
    conditioned = build_lagged_samples(
        [Segment(source=source, target=target, conditioning=driver)],
        source_history=1,
        target_history=1,
        prediction_delay=1,
        conditioning_lags=(2,),
    )
    raw = estimate_gcmi_te(unconditioned, shrinkage=0.05).value_bits
    adjusted = estimate_gcmi_te(conditioned, shrinkage=0.05).value_bits
    return float(raw), float(adjusted)


def synthetic_check(seed: int = 20260909) -> dict[str, Any]:
    """Run deterministic software checks and return a JSON-compatible report."""

    forward_segments, reverse_segments = _directional_segments(seed)
    forward_by_delay: dict[int, float] = {}
    forward_samples = None
    for delay in (1, 2, 4):
        samples = build_lagged_samples(
            forward_segments,
            source_history=1,
            target_history=1,
            prediction_delay=delay,
        )
        forward_by_delay[delay] = float(
            estimate_gcmi_te(samples, shrinkage=0.05).value_bits
        )
        if delay == 1:
            forward_samples = samples
    assert forward_samples is not None

    reverse_samples = build_lagged_samples(
        reverse_segments,
        source_history=1,
        target_history=1,
        prediction_delay=1,
    )
    reverse_value = float(
        estimate_gcmi_te(reverse_samples, shrinkage=0.05).value_bits
    )
    predictive = estimate_cross_validated_predictive_te(
        forward_samples,
        ridge=1e-3,
        shrinkage=0.05,
    )
    raw_common_driver, adjusted_common_driver = _common_driver_values(seed + 1)

    peak_delay = max(forward_by_delay, key=forward_by_delay.get)
    checks = {
        "known_direction_recovered": (
            forward_by_delay[1] > reverse_value + 0.02
        ),
        "held_out_predictive_improvement_positive": (
            predictive.value_bits_per_sample > 0.0
        ),
        "generating_delay_recovered": peak_delay == 1,
        "common_driver_suppressed": (
            adjusted_common_driver < 0.25 * raw_common_driver
        ),
    }
    return {
        "software_only": True,
        "seed": int(seed),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "values_bits_per_sample": {
            "forward_delay_1": forward_by_delay[1],
            "forward_delay_2": forward_by_delay[2],
            "forward_delay_4": forward_by_delay[4],
            "reverse_delay_1": reverse_value,
            "forward_cross_validated_delay_1": float(
                predictive.value_bits_per_sample
            ),
            "common_driver_unconditioned": raw_common_driver,
            "common_driver_conditioned": adjusted_common_driver,
        },
        "recovered_peak_delay_frames": int(peak_delay),
        "n_cross_validation_folds": len(predictive.folds),
        "interpretation": (
            "Synthetic software verification only; no manuscript data were loaded."
        ),
    }


def run_synthetic_check(
    output: str | Path | None = None,
    *,
    seed: int = 20260909,
) -> tuple[dict[str, Any], Path | None]:
    """Run the checks and optionally write ``synthetic_check.json`` atomically."""

    report = synthetic_check(seed=seed)
    destination: Path | None = None
    if output is not None:
        candidate = Path(output).resolve()
        destination = (
            candidate
            if candidate.suffix.lower() == ".json"
            else candidate / "synthetic_check.json"
        )
        atomic_write_json(destination, report)
    return report, destination


__all__ = ["run_synthetic_check", "synthetic_check"]
