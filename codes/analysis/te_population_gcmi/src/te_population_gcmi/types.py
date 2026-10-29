"""Core data containers for population transfer-entropy analysis.

All neural and behavioural arrays are time-major: ``(time, features)``.  The
containers deliberately do not know anything about files, experimental
conditions, or plotting so that the mathematical core remains reusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ObjectArray = NDArray[np.object_]


def _as_float_matrix(
    value: ArrayLike | None,
    name: str,
    *,
    n_rows: int | None = None,
) -> FloatArray:
    """Convert an input to a two-dimensional, time-major float array."""

    if value is None:
        if n_rows is None:
            raise ValueError(f"{name} cannot be None when n_rows is unknown")
        return np.empty((n_rows, 0), dtype=np.float64)

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(
            f"{name} must have shape (time, features); got {array.shape}"
        )
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(
            f"{name} has {array.shape[0]} rows; expected {n_rows} time points"
        )
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class Segment:
    """One contiguous trajectory segment.

    Lagged samples are constructed separately for every ``Segment``.  Thus,
    neither histories nor prediction pairs can cross a segment boundary.

    Parameters
    ----------
    source, target
        Time-major population activity arrays.  One-dimensional inputs are
        interpreted as a single feature.
    conditioning
        Optional time-aligned behavioural/common-driver covariates.  ``None``
        is represented as an array with zero columns.
    segment_id, trajectory_id
        Hashable labels propagated to every resulting lagged sample.
    weight
        Positive segment-level analysis weight.  With equal-segment weighting,
        this is the total weight assigned to all valid rows in the segment.
    metadata
        Descriptive information that is never used by the estimator.

    Notes
    -----
    The dataclass is frozen, but NumPy arrays are treated as read-only by
    convention rather than copied.  Avoid mutating them after construction.
    """

    source: ArrayLike
    target: ArrayLike
    conditioning: ArrayLike | None = None
    segment_id: Hashable | None = None
    trajectory_id: Hashable = 0
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        source = _as_float_matrix(self.source, "source")
        target = _as_float_matrix(self.target, "target", n_rows=source.shape[0])
        conditioning = _as_float_matrix(
            self.conditioning,
            "conditioning",
            n_rows=source.shape[0],
        )
        if source.shape[0] == 0:
            raise ValueError("a segment must contain at least one time point")
        if source.shape[1] == 0 or target.shape[1] == 0:
            raise ValueError("source and target must each contain at least one feature")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("segment weight must be finite and strictly positive")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "conditioning", conditioning)

    @property
    def n_time(self) -> int:
        return int(self.source.shape[0])

    @property
    def source_dimension(self) -> int:
        return int(self.source.shape[1])

    @property
    def target_dimension(self) -> int:
        return int(self.target.shape[1])

    @property
    def conditioning_dimension(self) -> int:
        return int(self.conditioning.shape[1])


@dataclass(frozen=True, slots=True)
class LaggedSamples:
    """Aligned samples used to estimate conditional transfer entropy.

    The represented estimand is

    ``I(target_future; source_past | target_past, conditioning)``.

    ``source_past`` and ``target_past`` already contain their requested history
    embeddings.  Row labels retain segment and trajectory membership for
    block-aware nulls and trajectory-fold cross-validation.
    """

    target_future: ArrayLike
    source_past: ArrayLike
    target_past: ArrayLike
    conditioning: ArrayLike
    segment_index: ArrayLike
    segment_id: ArrayLike
    trajectory_id: ArrayLike
    time_index: ArrayLike
    sample_weight: ArrayLike

    def __post_init__(self) -> None:
        target_future = _as_float_matrix(self.target_future, "target_future")
        n_rows = target_future.shape[0]
        if n_rows == 0:
            raise ValueError("LaggedSamples cannot be empty")

        source_past = _as_float_matrix(
            self.source_past, "source_past", n_rows=n_rows
        )
        target_past = _as_float_matrix(
            self.target_past, "target_past", n_rows=n_rows
        )
        conditioning = _as_float_matrix(
            self.conditioning, "conditioning", n_rows=n_rows
        )
        if source_past.shape[1] == 0:
            raise ValueError("source_past must contain at least one feature")

        segment_index = np.asarray(self.segment_index, dtype=np.int64).reshape(-1)
        time_index = np.asarray(self.time_index, dtype=np.int64).reshape(-1)
        segment_id = np.asarray(self.segment_id, dtype=object).reshape(-1)
        trajectory_id = np.asarray(self.trajectory_id, dtype=object).reshape(-1)
        sample_weight = np.asarray(self.sample_weight, dtype=np.float64).reshape(-1)
        for name, array in (
            ("segment_index", segment_index),
            ("segment_id", segment_id),
            ("trajectory_id", trajectory_id),
            ("time_index", time_index),
            ("sample_weight", sample_weight),
        ):
            if array.shape[0] != n_rows:
                raise ValueError(f"{name} has {array.shape[0]} rows; expected {n_rows}")
        if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0.0):
            raise ValueError("sample_weight values must be finite and strictly positive")

        object.__setattr__(self, "target_future", target_future)
        object.__setattr__(self, "source_past", source_past)
        object.__setattr__(self, "target_past", target_past)
        object.__setattr__(self, "conditioning", conditioning)
        object.__setattr__(self, "segment_index", segment_index)
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "time_index", time_index)
        object.__setattr__(self, "sample_weight", sample_weight)

    def __len__(self) -> int:
        return int(self.target_future.shape[0])

    @property
    def n_samples(self) -> int:
        return len(self)

    @property
    def conditioning_state(self) -> FloatArray:
        """Return ``[target_past, conditioning]`` as the TE condition."""

        if self.conditioning.shape[1] == 0:
            return self.target_past
        if self.target_past.shape[1] == 0:
            return self.conditioning
        return np.concatenate((self.target_past, self.conditioning), axis=1)

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size implied by ``sample_weight``."""

        total = float(np.sum(self.sample_weight))
        squared = float(np.dot(self.sample_weight, self.sample_weight))
        return total * total / squared

    def subset(self, selector: ArrayLike) -> "LaggedSamples":
        """Return rows selected by a Boolean mask or integer indices."""

        index = np.asarray(selector)
        return LaggedSamples(
            target_future=self.target_future[index],
            source_past=self.source_past[index],
            target_past=self.target_past[index],
            conditioning=self.conditioning[index],
            segment_index=self.segment_index[index],
            segment_id=self.segment_id[index],
            trajectory_id=self.trajectory_id[index],
            time_index=self.time_index[index],
            sample_weight=self.sample_weight[index],
        )
