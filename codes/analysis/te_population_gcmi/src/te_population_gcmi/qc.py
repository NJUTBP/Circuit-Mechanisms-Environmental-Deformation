"""Quality-control metrics for pooled, autocorrelated TE samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .types import LaggedSamples, Segment


@dataclass(frozen=True, slots=True)
class TEQuality:
    """Pre-estimation sufficiency decision for one direction/configuration."""

    status: str
    reason: str
    n_segments: int
    n_trajectories: int
    n_samples: int
    kish_effective_n: float
    temporal_effective_n: float
    joint_dimension: int
    minimum_required_n: float
    maximum_autocorrelation_time: float

    @property
    def estimable(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _pooled_autocorrelation(
    arrays: Sequence[np.ndarray], feature: int, lag: int
) -> float:
    numerator = 0.0
    left_energy = 0.0
    right_energy = 0.0
    n_pairs = 0
    for array in arrays:
        if array.shape[0] <= lag:
            continue
        column = np.asarray(array[:, feature], dtype=float)
        centered = column - np.mean(column)
        left = centered[:-lag]
        right = centered[lag:]
        numerator += float(np.dot(left, right))
        left_energy += float(np.dot(left, left))
        right_energy += float(np.dot(right, right))
        n_pairs += left.size
    if n_pairs < 2 or left_energy <= 0.0 or right_energy <= 0.0:
        return 0.0
    return float(numerator / np.sqrt(left_energy * right_energy))


def estimate_temporal_effective_sample_size(
    segments: Sequence[Segment], *, max_lag: int = 100
) -> tuple[float, float]:
    """Estimate a conservative effective sample size from positive ACF sums.

    Source and target dimensions are assessed separately after centering within
    each segment.  For every feature, positive autocorrelations are summed until
    the first non-positive value.  The largest integrated autocorrelation time
    is used so that a slowly varying latent dimension cannot be hidden by faster
    dimensions.

    Returns
    -------
    effective_n, maximum_tau
        ``effective_n`` is the total number of time points divided by the most
        conservative integrated autocorrelation time.
    """

    segment_list = tuple(segments)
    if not segment_list:
        return 0.0, float("inf")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    total = int(sum(segment.n_time for segment in segment_list))
    source_arrays = [np.asarray(segment.source, dtype=float) for segment in segment_list]
    target_arrays = [np.asarray(segment.target, dtype=float) for segment in segment_list]
    maximum_tau = 1.0
    for arrays in (source_arrays, target_arrays):
        n_features = arrays[0].shape[1]
        for feature in range(n_features):
            positive_sum = 0.0
            usable_max = min(max_lag, max(array.shape[0] - 1 for array in arrays))
            for lag in range(1, usable_max + 1):
                rho = _pooled_autocorrelation(arrays, feature, lag)
                if not np.isfinite(rho) or rho <= 0.0:
                    break
                positive_sum += min(rho, 0.999)
            maximum_tau = max(maximum_tau, 1.0 + 2.0 * positive_sum)
    return float(total / maximum_tau), float(maximum_tau)


def assess_te_quality(
    samples: LaggedSamples,
    segments: Sequence[Segment],
    *,
    minimum_pool_samples: int,
    minimum_effective_samples_per_dimension: float,
    autocorrelation_max_lag: int = 100,
    require_multiple_trajectories: bool = True,
) -> TEQuality:
    """Apply explicit sample-size and temporal-dependence gates."""

    if minimum_pool_samples < 1:
        raise ValueError("minimum_pool_samples must be positive")
    if minimum_effective_samples_per_dimension <= 0:
        raise ValueError("minimum_effective_samples_per_dimension must be positive")
    segment_list = tuple(segments)
    _, tau = estimate_temporal_effective_sample_size(
        segment_list, max_lag=autocorrelation_max_lag
    )
    temporal_n = float(samples.n_samples / tau)
    joint_dimension = int(
        samples.target_future.shape[1]
        + samples.source_past.shape[1]
        + samples.conditioning_state.shape[1]
    )
    required = max(
        float(minimum_pool_samples),
        float(minimum_effective_samples_per_dimension) * joint_dimension,
    )
    conservative_effective_n = min(
        float(samples.effective_sample_size), temporal_n
    )
    reasons: list[str] = []
    if samples.n_samples < minimum_pool_samples:
        reasons.append(
            f"raw pooled samples {samples.n_samples} < {minimum_pool_samples}"
        )
    if conservative_effective_n < required:
        reasons.append(
            "conservative effective n "
            f"{conservative_effective_n:.3g} < required {required:.3g}"
        )
    trajectories = {str(value) for value in samples.trajectory_id}
    if require_multiple_trajectories and len(trajectories) < 2:
        reasons.append("fewer than two trajectory folds")
    return TEQuality(
        status="insufficient_data" if reasons else "ok",
        reason="; ".join(reasons) if reasons else "",
        n_segments=len(segment_list),
        n_trajectories=len(trajectories),
        n_samples=samples.n_samples,
        kish_effective_n=float(samples.effective_sample_size),
        temporal_effective_n=temporal_n,
        joint_dimension=joint_dimension,
        minimum_required_n=required,
        maximum_autocorrelation_time=tau,
    )
