"""Segment-preserving circular-shift nulls and finite permutation p-values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import Segment


Alternative = Literal["greater", "less", "two-sided"]
SegmentEstimator = Callable[[Sequence[Segment]], float]


@dataclass(frozen=True, slots=True)
class CircularShiftNullResult:
    """Observed estimate and its within-segment circular-shift null."""

    observed: float
    null_distribution: NDArray[np.float64]
    p_value: float
    alternative: Alternative

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null_distribution))

    @property
    def null_standard_deviation(self) -> float:
        return float(np.std(self.null_distribution, ddof=1))

    @property
    def bias_corrected(self) -> float:
        return self.observed - self.null_mean


def circular_shift_segments(
    segments: Sequence[Segment],
    *,
    rng: np.random.Generator,
    min_circular_distance: int = 1,
) -> tuple[Segment, ...]:
    """Circularly shift source activity independently inside each segment.

    A shift must be at least ``min_circular_distance`` from both zero and a
    complete wrap.  Target and conditioning data are left unchanged.  This
    preserves source marginal distribution and within-source autocorrelation,
    but intentionally breaks source alignment with target *and* behaviour; it
    should therefore be treated as a finite-sample diagnostic rather than a
    substitute for a behaviour-matched conditional null.
    """

    if min_circular_distance < 1:
        raise ValueError("min_circular_distance must be at least 1")
    shifted: list[Segment] = []
    for segment in segments:
        lower = min_circular_distance
        upper = segment.n_time - min_circular_distance
        if upper < lower:
            raise ValueError(
                f"segment {segment.segment_id!r} with length {segment.n_time} is "
                f"too short for min_circular_distance={min_circular_distance}"
            )
        shift = int(rng.integers(lower, upper + 1))
        shifted.append(
            Segment(
                source=np.roll(segment.source, shift=shift, axis=0),
                target=segment.target,
                conditioning=segment.conditioning,
                segment_id=segment.segment_id,
                trajectory_id=segment.trajectory_id,
                weight=segment.weight,
                metadata=segment.metadata,
            )
        )
    return tuple(shifted)


def plus_one_pvalue(
    observed: float,
    null_distribution: ArrayLike,
    *,
    alternative: Alternative = "greater",
) -> float:
    """Return a valid Monte Carlo p-value using ``(extreme + 1)/(M + 1)``."""

    null = np.asarray(null_distribution, dtype=np.float64).reshape(-1)
    if null.size == 0:
        raise ValueError("null_distribution cannot be empty")
    if not np.isfinite(observed) or not np.all(np.isfinite(null)):
        raise ValueError("observed and null estimates must all be finite")

    if alternative == "greater":
        extreme = int(np.sum(null >= observed))
    elif alternative == "less":
        extreme = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        center = float(np.median(null))
        extreme = int(np.sum(np.abs(null - center) >= abs(observed - center)))
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")
    return (extreme + 1.0) / (null.size + 1.0)


def circular_shift_test(
    segments: Sequence[Segment],
    estimator: SegmentEstimator,
    *,
    n_surrogates: int = 1000,
    seed: int | np.random.SeedSequence | None = None,
    min_circular_distance: int = 1,
    alternative: Alternative = "greater",
) -> CircularShiftNullResult:
    """Evaluate an estimator against independently shifted segment sources.

    ``estimator`` receives the complete segment sequence and must return one
    finite scalar.  A fresh, deterministic child random stream is used for
    every surrogate, so results are reproducible for a fixed ``seed``.
    """

    segment_tuple = tuple(segments)
    if not segment_tuple:
        raise ValueError("at least one Segment is required")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be at least 1")

    observed = float(estimator(segment_tuple))
    if not np.isfinite(observed):
        raise ValueError("estimator returned a non-finite observed value")

    seed_sequence = (
        seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    )
    child_sequences = seed_sequence.spawn(n_surrogates)
    null = np.empty(n_surrogates, dtype=np.float64)
    for index, child_seed in enumerate(child_sequences):
        surrogate_segments = circular_shift_segments(
            segment_tuple,
            rng=np.random.default_rng(child_seed),
            min_circular_distance=min_circular_distance,
        )
        null[index] = float(estimator(surrogate_segments))
        if not np.isfinite(null[index]):
            raise ValueError(f"estimator returned a non-finite value at surrogate {index}")

    return CircularShiftNullResult(
        observed=observed,
        null_distribution=null,
        p_value=plus_one_pvalue(observed, null, alternative=alternative),
        alternative=alternative,
    )
