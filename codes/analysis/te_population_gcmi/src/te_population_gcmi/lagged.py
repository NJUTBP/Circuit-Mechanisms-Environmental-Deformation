"""Boundary-safe construction of lagged transfer-entropy samples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import LaggedSamples, Segment


def build_lagged_samples(
    segments: Sequence[Segment],
    *,
    source_history: int = 1,
    target_history: int = 1,
    prediction_delay: int = 1,
    conditioning_lags: Sequence[int] = (0,),
    drop_nonfinite: bool = True,
    equal_segment_weight: bool = True,
    min_valid_samples_per_segment: int = 1,
) -> LaggedSamples:
    """Construct samples for ``I(Y_t; X_past | Y_past, B)``.

    For a prediction time ``t`` the source embedding is
    ``[X[t-delay], ..., X[t-delay-source_history+1]]`` and the target embedding
    is ``[Y[t-1], ..., Y[t-target_history]]``.  Thus prediction delay and both
    history lengths are independent parameters.  Conditioning covariates are
    included at each offset in ``conditioning_lags`` (for example ``(0, 1)``
    includes ``B[t]`` and ``B[t-1]``).

    Segments are processed independently, so no row can use a value from a
    neighbouring segment.  History blocks are ordered from most recent to
    oldest.
    """

    segment_list = tuple(segments)
    if not segment_list:
        raise ValueError("at least one Segment is required")
    if source_history < 1:
        raise ValueError("source_history must be at least 1")
    if target_history < 0:
        raise ValueError("target_history cannot be negative")
    if prediction_delay < 1:
        raise ValueError("prediction_delay must be at least 1")
    if min_valid_samples_per_segment < 1:
        raise ValueError("min_valid_samples_per_segment must be at least 1")

    condition_offsets = tuple(int(offset) for offset in conditioning_lags)
    if any(offset < 0 for offset in condition_offsets):
        raise ValueError("conditioning_lags cannot contain negative offsets")
    if len(set(condition_offsets)) != len(condition_offsets):
        raise ValueError("conditioning_lags must not contain duplicates")

    source_dimension = segment_list[0].source_dimension
    target_dimension = segment_list[0].target_dimension
    condition_dimension = segment_list[0].conditioning_dimension
    for segment in segment_list[1:]:
        dimensions = (
            segment.source_dimension,
            segment.target_dimension,
            segment.conditioning_dimension,
        )
        if dimensions != (source_dimension, target_dimension, condition_dimension):
            raise ValueError("all segments must have identical feature dimensions")

    if condition_dimension > 0 and not condition_offsets:
        raise ValueError(
            "conditioning_lags cannot be empty when segments contain conditioning data"
        )

    earliest_time = max(
        target_history,
        prediction_delay + source_history - 1,
        max(condition_offsets, default=0),
    )

    future_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    condition_blocks: list[np.ndarray] = []
    segment_index_blocks: list[np.ndarray] = []
    segment_id_blocks: list[np.ndarray] = []
    trajectory_id_blocks: list[np.ndarray] = []
    time_index_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []

    for segment_index, segment in enumerate(segment_list):
        if segment.n_time <= earliest_time:
            continue
        time_index = np.arange(earliest_time, segment.n_time, dtype=np.int64)
        target_future = segment.target[time_index]
        source_past = np.concatenate(
            [
                segment.source[time_index - prediction_delay - history_offset]
                for history_offset in range(source_history)
            ],
            axis=1,
        )
        if target_history:
            target_past = np.concatenate(
                [
                    segment.target[time_index - 1 - history_offset]
                    for history_offset in range(target_history)
                ],
                axis=1,
            )
        else:
            target_past = np.empty((time_index.size, 0), dtype=np.float64)

        if condition_dimension:
            conditioning = np.concatenate(
                [segment.conditioning[time_index - offset] for offset in condition_offsets],
                axis=1,
            )
        else:
            conditioning = np.empty((time_index.size, 0), dtype=np.float64)

        if drop_nonfinite:
            finite = (
                np.all(np.isfinite(target_future), axis=1)
                & np.all(np.isfinite(source_past), axis=1)
                & np.all(np.isfinite(target_past), axis=1)
                & np.all(np.isfinite(conditioning), axis=1)
            )
            target_future = target_future[finite]
            source_past = source_past[finite]
            target_past = target_past[finite]
            conditioning = conditioning[finite]
            time_index = time_index[finite]

        n_valid = time_index.size
        if n_valid < min_valid_samples_per_segment:
            continue
        row_weight = (
            float(segment.weight) / n_valid
            if equal_segment_weight
            else float(segment.weight)
        )

        future_blocks.append(target_future)
        source_blocks.append(source_past)
        target_blocks.append(target_past)
        condition_blocks.append(conditioning)
        segment_index_blocks.append(np.full(n_valid, segment_index, dtype=np.int64))
        segment_id_block = np.empty(n_valid, dtype=object)
        segment_id_block.fill(segment.segment_id)
        segment_id_blocks.append(segment_id_block)
        trajectory_id_block = np.empty(n_valid, dtype=object)
        trajectory_id_block.fill(segment.trajectory_id)
        trajectory_id_blocks.append(trajectory_id_block)
        time_index_blocks.append(time_index)
        weight_blocks.append(np.full(n_valid, row_weight, dtype=np.float64))

    if not future_blocks:
        raise ValueError(
            "no segment contains enough finite observations for the requested "
            "delay and history embeddings"
        )

    return LaggedSamples(
        target_future=np.concatenate(future_blocks, axis=0),
        source_past=np.concatenate(source_blocks, axis=0),
        target_past=np.concatenate(target_blocks, axis=0),
        conditioning=np.concatenate(condition_blocks, axis=0),
        segment_index=np.concatenate(segment_index_blocks),
        segment_id=np.concatenate(segment_id_blocks),
        trajectory_id=np.concatenate(trajectory_id_blocks),
        time_index=np.concatenate(time_index_blocks),
        sample_weight=np.concatenate(weight_blocks),
    )
