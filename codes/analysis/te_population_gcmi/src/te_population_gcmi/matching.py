"""Behaviour-matched cross-trajectory/session negative controls.

The matcher pairs source and target segments only inside the same behavioural
state and wall-axis stratum, while forbidding pairs from the same trajectory.
Continuous segment summaries are standardized within each stratum and matched
one-to-one without replacement.  The matched source activity is then linearly
resampled over normalized segment progress to the target segment's length.

This module deliberately depends only on NumPy and the standard library.  Its
audit objects expose ordinary records so callers may serialize them to JSON or
construct a pandas table outside the mathematical core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import Segment


FloatArray = NDArray[np.float64]

BASE_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "speed",
    "heading_sin",
    "heading_cos",
    "wall_left",
    "wall_right",
    "wall_bottom",
    "wall_top",
)
DEFAULT_FEATURE_NAMES = BASE_FEATURE_NAMES + ("log_length",)


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """Audit information for one accepted source-target segment pair."""

    source_index: int
    target_index: int
    source_segment_id: Hashable | None
    target_segment_id: Hashable | None
    source_trajectory_id: Hashable
    target_trajectory_id: Hashable
    state: Hashable
    wall_axis: Hashable
    source_length: int
    target_length: int
    relative_length_difference: float
    standardized_distance: float
    standardized_feature_differences: tuple[float, ...]
    source_summary: tuple[float, ...]
    target_summary: tuple[float, ...]

    def as_dict(self, feature_names: Sequence[str]) -> dict[str, Any]:
        """Return a flat, JSON-friendly audit record."""

        names = tuple(feature_names)
        if len(names) != len(self.source_summary):
            raise ValueError("feature_names length does not match this record")
        record: dict[str, Any] = {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_segment_id": self.source_segment_id,
            "target_segment_id": self.target_segment_id,
            "source_trajectory_id": self.source_trajectory_id,
            "target_trajectory_id": self.target_trajectory_id,
            "state": self.state,
            "wall_axis": self.wall_axis,
            "source_length": self.source_length,
            "target_length": self.target_length,
            "relative_length_difference": self.relative_length_difference,
            "standardized_distance": self.standardized_distance,
        }
        for index, name in enumerate(names):
            record[f"source_{name}"] = self.source_summary[index]
            record[f"target_{name}"] = self.target_summary[index]
            record[f"standardized_difference_{name}"] = (
                self.standardized_feature_differences[index]
            )
        return record


@dataclass(frozen=True, slots=True)
class MatchAudit:
    """Complete accounting for a cross-trajectory matching run.

    Exclusion counts refer to candidate *pairs*, whereas unmatched indices
    refer to source or target segments after greedy assignment.
    """

    feature_names: tuple[str, ...]
    matches: tuple[MatchRecord, ...]
    unmatched_source_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]
    n_source_segments: int
    n_target_segments: int
    n_pairs_considered: int
    n_pairs_excluded_same_trajectory: int
    n_pairs_excluded_length: int
    n_pairs_excluded_caliper: int
    length_tolerance: float | None
    caliper: float | None
    seed: int | None
    randomized_within_caliper: bool

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    def records(self) -> list[dict[str, Any]]:
        """Return accepted matches as a list of flat dictionaries."""

        return [match.as_dict(self.feature_names) for match in self.matches]


@dataclass(frozen=True, slots=True)
class MatchedNullSegments:
    """Matched cross-trajectory null segments and their audit trail."""

    segments: tuple[Segment, ...]
    audit: MatchAudit


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_index: int
    target_index: int
    distance: float
    differences: tuple[float, ...]
    tie_breaker: float


def match_cross_trajectory_segments(
    source_segments: Sequence[Segment],
    target_segments: Sequence[Segment],
    *,
    source_summaries: ArrayLike | None = None,
    target_summaries: ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    source_states: Sequence[Hashable] | None = None,
    target_states: Sequence[Hashable] | None = None,
    source_wall_axes: Sequence[Hashable] | None = None,
    target_wall_axes: Sequence[Hashable] | None = None,
    state_key: str = "state",
    wall_axis_key: str = "wall_axis",
    length_tolerance: float | None = 0.5,
    caliper: float | None = 2.5,
    include_log_length: bool = True,
    standardization_floor: float = 1e-12,
    randomize_within_caliper: bool = False,
    seed: int | None = None,
) -> MatchedNullSegments:
    """Build a behaviour-matched, cross-trajectory/session source null.

    Parameters
    ----------
    source_segments, target_segments
        Candidate segment groups.  An accepted output combines the ``source``
        activity from a source candidate with ``target`` and ``conditioning``
        from a target candidate.
    source_summaries, target_summaries
        Optional explicit ``(segments, features)`` arrays.  Supply both or
        neither.  Without them, the standard position, speed, heading and four
        wall-distance summaries are extracted from each segment's metadata.
        Array values stored in metadata are reduced by their finite mean.
    feature_names
        Names for explicit summary columns.  If omitted, 9 columns are assumed
        to follow ``BASE_FEATURE_NAMES`` and 10 columns ``DEFAULT_FEATURE_NAMES``.
    source_states, target_states, source_wall_axes, target_wall_axes
        Optional explicit stratum labels.  If omitted, labels are read from
        metadata using ``state_key`` and ``wall_axis_key``.
    length_tolerance
        Maximum ``abs(n_source-n_target)/max(n_source,n_target)``.  ``None``
        disables the hard length filter.
    caliper
        Maximum root-mean-square standardized summary difference.  ``None``
        disables the caliper.
    include_log_length
        Append natural-log segment length unless already supplied by name.
    randomize_within_caliper
        If false, use deterministic nearest-distance greedy matching (with the
        seed used only to break exact distance ties).  If true, the caliper,
        length rule, state, and wall axis define the admissible exchangeability
        set, and admissible pairs are shuffled before greedy matching.  This is
        the mode required for repeated null draws; it samples different
        cross-trajectory pairings without accepting out-of-caliper pairs.
    seed
        Reproducible random seed.  It is required when
        ``randomize_within_caliper`` is true.

    Returns
    -------
    MatchedNullSegments
        Output segments are ordered by target index.  Matching is one-to-one
        without replacement, uses pooled within-stratum standardization, and
        never pairs equal ``trajectory_id`` values.
    """

    sources = _validate_segments(source_segments, "source_segments")
    targets = _validate_segments(target_segments, "target_segments")
    if length_tolerance is not None and not 0.0 <= length_tolerance < 1.0:
        raise ValueError("length_tolerance must lie in [0, 1) or be None")
    if caliper is not None and (not np.isfinite(caliper) or caliper < 0.0):
        raise ValueError("caliper must be finite and non-negative or None")
    if not np.isfinite(standardization_floor) or standardization_floor <= 0.0:
        raise ValueError("standardization_floor must be finite and positive")
    if randomize_within_caliper and seed is None:
        raise ValueError(
            "A seed is required when randomize_within_caliper=True"
        )

    source_matrix, target_matrix, names = _prepare_summaries(
        sources,
        targets,
        source_summaries,
        target_summaries,
        feature_names,
        include_log_length=include_log_length,
    )
    source_state = _labels(
        sources, source_states, metadata_key=state_key, label="source_states"
    )
    target_state = _labels(
        targets, target_states, metadata_key=state_key, label="target_states"
    )
    source_axis = _labels(
        sources,
        source_wall_axes,
        metadata_key=wall_axis_key,
        label="source_wall_axes",
    )
    target_axis = _labels(
        targets,
        target_wall_axes,
        metadata_key=wall_axis_key,
        label="target_wall_axes",
    )

    source_strata = _group_indices(source_state, source_axis)
    target_strata = _group_indices(target_state, target_axis)
    random = np.random.default_rng(seed) if seed is not None else None
    candidates: list[_Candidate] = []
    considered = 0
    excluded_same_trajectory = 0
    excluded_length = 0
    excluded_caliper = 0

    for stratum, source_indices in source_strata.items():
        target_indices = target_strata.get(stratum, ())
        if not target_indices:
            continue
        source_index_array = np.asarray(source_indices, dtype=np.int64)
        target_index_array = np.asarray(target_indices, dtype=np.int64)
        pooled = np.concatenate(
            (source_matrix[source_index_array], target_matrix[target_index_array]),
            axis=0,
        )
        center = np.mean(pooled, axis=0)
        scale = np.std(pooled, axis=0, ddof=0)
        scale = np.where(scale >= standardization_floor, scale, 1.0)
        standardized_source = (source_matrix[source_index_array] - center) / scale
        standardized_target = (target_matrix[target_index_array] - center) / scale

        for local_source, source_index in enumerate(source_indices):
            for local_target, target_index in enumerate(target_indices):
                considered += 1
                if sources[source_index].trajectory_id == targets[target_index].trajectory_id:
                    excluded_same_trajectory += 1
                    continue
                relative_length = _relative_length_difference(
                    sources[source_index].n_time,
                    targets[target_index].n_time,
                )
                if length_tolerance is not None and relative_length > length_tolerance:
                    excluded_length += 1
                    continue
                difference = standardized_source[local_source] - standardized_target[local_target]
                distance = float(np.sqrt(np.mean(difference * difference)))
                if caliper is not None and distance > caliper:
                    excluded_caliper += 1
                    continue
                tie_breaker = float(random.random()) if random is not None else 0.0
                candidates.append(
                    _Candidate(
                        source_index=source_index,
                        target_index=target_index,
                        distance=distance,
                        differences=tuple(float(value) for value in difference),
                        tie_breaker=tie_breaker,
                    )
                )

    if randomize_within_caliper:
        assert random is not None
        order = random.permutation(len(candidates))
        candidates = [candidates[int(index)] for index in order]
    else:
        candidates.sort(
            key=lambda candidate: (
                candidate.distance,
                candidate.tie_breaker,
                candidate.source_index,
                candidate.target_index,
            )
        )
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    accepted: list[_Candidate] = []
    for candidate in candidates:
        if candidate.source_index in used_sources or candidate.target_index in used_targets:
            continue
        used_sources.add(candidate.source_index)
        used_targets.add(candidate.target_index)
        accepted.append(candidate)
    accepted.sort(key=lambda candidate: candidate.target_index)

    records: list[MatchRecord] = []
    output_segments: list[Segment] = []
    for candidate in accepted:
        source = sources[candidate.source_index]
        target = targets[candidate.target_index]
        state = target_state[candidate.target_index]
        wall_axis = target_axis[candidate.target_index]
        relative_length = _relative_length_difference(source.n_time, target.n_time)
        records.append(
            MatchRecord(
                source_index=candidate.source_index,
                target_index=candidate.target_index,
                source_segment_id=source.segment_id,
                target_segment_id=target.segment_id,
                source_trajectory_id=source.trajectory_id,
                target_trajectory_id=target.trajectory_id,
                state=state,
                wall_axis=wall_axis,
                source_length=source.n_time,
                target_length=target.n_time,
                relative_length_difference=relative_length,
                standardized_distance=candidate.distance,
                standardized_feature_differences=candidate.differences,
                source_summary=tuple(float(value) for value in source_matrix[candidate.source_index]),
                target_summary=tuple(float(value) for value in target_matrix[candidate.target_index]),
            )
        )
        metadata = dict(target.metadata)
        metadata.update(
            {
                "null_type": "behaviour_matched_cross_trajectory",
                "matched_source_segment_id": source.segment_id,
                "matched_source_trajectory_id": source.trajectory_id,
                "matched_target_segment_id": target.segment_id,
                "matched_target_trajectory_id": target.trajectory_id,
                "match_standardized_distance": candidate.distance,
                "match_relative_length_difference": relative_length,
            }
        )
        output_segments.append(
            Segment(
                source=resample_normalized_progress(source.source, target.n_time),
                target=target.target,
                conditioning=target.conditioning,
                segment_id=("cross_trajectory_null", source.segment_id, target.segment_id),
                trajectory_id=target.trajectory_id,
                weight=target.weight,
                metadata=metadata,
            )
        )

    audit = MatchAudit(
        feature_names=names,
        matches=tuple(records),
        unmatched_source_indices=tuple(
            index for index in range(len(sources)) if index not in used_sources
        ),
        unmatched_target_indices=tuple(
            index for index in range(len(targets)) if index not in used_targets
        ),
        n_source_segments=len(sources),
        n_target_segments=len(targets),
        n_pairs_considered=considered,
        n_pairs_excluded_same_trajectory=excluded_same_trajectory,
        n_pairs_excluded_length=excluded_length,
        n_pairs_excluded_caliper=excluded_caliper,
        length_tolerance=length_tolerance,
        caliper=caliper,
        seed=seed,
        randomized_within_caliper=bool(randomize_within_caliper),
    )
    return MatchedNullSegments(tuple(output_segments), audit)


def resample_normalized_progress(value: ArrayLike, output_length: int) -> FloatArray:
    """Linearly resample a time-major array on the closed interval [0, 1]."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("value must have shape (positive time, features)")
    if not np.all(np.isfinite(array)):
        raise ValueError("value contains NaN or infinite values")
    if output_length < 1:
        raise ValueError("output_length must be at least 1")
    if array.shape[0] == output_length:
        return array.copy()
    if array.shape[0] == 1:
        return np.repeat(array, output_length, axis=0)
    old_progress = np.linspace(0.0, 1.0, array.shape[0])
    new_progress = np.linspace(0.0, 1.0, output_length)
    output = np.empty((output_length, array.shape[1]), dtype=np.float64)
    for feature in range(array.shape[1]):
        output[:, feature] = np.interp(new_progress, old_progress, array[:, feature])
    return output


def _validate_segments(segments: Sequence[Segment], label: str) -> tuple[Segment, ...]:
    result = tuple(segments)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if not all(isinstance(segment, Segment) for segment in result):
        raise TypeError(f"every item in {label} must be a Segment")
    return result


def _prepare_summaries(
    sources: tuple[Segment, ...],
    targets: tuple[Segment, ...],
    source_summaries: ArrayLike | None,
    target_summaries: ArrayLike | None,
    feature_names: Sequence[str] | None,
    *,
    include_log_length: bool,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    if (source_summaries is None) != (target_summaries is None):
        raise ValueError("source_summaries and target_summaries must be supplied together")

    if source_summaries is None:
        if feature_names is not None and tuple(feature_names) not in (
            BASE_FEATURE_NAMES,
            DEFAULT_FEATURE_NAMES,
        ):
            raise ValueError(
                "metadata extraction uses the standard feature names; provide "
                "explicit summary arrays for a custom feature set"
            )
        source_matrix = np.vstack([_metadata_summary(segment) for segment in sources])
        target_matrix = np.vstack([_metadata_summary(segment) for segment in targets])
        names = BASE_FEATURE_NAMES
    else:
        source_matrix = _summary_matrix(source_summaries, len(sources), "source_summaries")
        target_matrix = _summary_matrix(target_summaries, len(targets), "target_summaries")
        if source_matrix.shape[1] != target_matrix.shape[1]:
            raise ValueError("source and target summary arrays need the same columns")
        if feature_names is None:
            if source_matrix.shape[1] == len(BASE_FEATURE_NAMES):
                names = BASE_FEATURE_NAMES
            elif source_matrix.shape[1] == len(DEFAULT_FEATURE_NAMES):
                names = DEFAULT_FEATURE_NAMES
            else:
                names = tuple(f"feature_{index}" for index in range(source_matrix.shape[1]))
        else:
            names = tuple(str(name) for name in feature_names)
            if len(names) != source_matrix.shape[1]:
                raise ValueError("feature_names length does not match summary columns")
            if len(set(names)) != len(names):
                raise ValueError("feature_names must be unique")

    if include_log_length and "log_length" not in names:
        source_matrix = np.column_stack(
            (source_matrix, np.log([segment.n_time for segment in sources]))
        )
        target_matrix = np.column_stack(
            (target_matrix, np.log([segment.n_time for segment in targets]))
        )
        names = names + ("log_length",)
    return source_matrix, target_matrix, names


def _summary_matrix(value: ArrayLike, n_rows: int, label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] < 1:
        raise ValueError(
            f"{label} must have shape ({n_rows}, positive features); got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinite values")
    return array


def _metadata_summary(segment: Segment) -> FloatArray:
    metadata = segment.metadata
    position = _optional_vector(metadata, ("position", "position_mean", "mean_position"), 2)
    position_x = (
        float(position[0])
        if position is not None
        else _metadata_mean(metadata, ("position_x", "mean_x", "x"))
    )
    position_y = (
        float(position[1])
        if position is not None
        else _metadata_mean(metadata, ("position_y", "mean_y", "y"))
    )
    speed = _metadata_mean(metadata, ("speed", "mean_speed"))

    heading_sin = _optional_mean(metadata, ("heading_sin", "mean_heading_sin"))
    heading_cos = _optional_mean(metadata, ("heading_cos", "mean_heading_cos"))
    if heading_sin is None or heading_cos is None:
        heading = _metadata_array(metadata, ("heading", "heading_radians"))
        if heading_sin is None:
            heading_sin = float(np.mean(np.sin(heading)))
        if heading_cos is None:
            heading_cos = float(np.mean(np.cos(heading)))

    wall = _optional_vector(
        metadata,
        ("wall_distances", "mean_wall_distances"),
        4,
    )
    if wall is None:
        wall = np.array(
            [
                _metadata_mean(
                    metadata,
                    ("wall_left", "left_wall_distance", "distance_left"),
                ),
                _metadata_mean(
                    metadata,
                    ("wall_right", "right_wall_distance", "distance_right"),
                ),
                _metadata_mean(
                    metadata,
                    ("wall_bottom", "bottom_wall_distance", "distance_bottom"),
                ),
                _metadata_mean(
                    metadata,
                    ("wall_top", "top_wall_distance", "distance_top"),
                ),
            ],
            dtype=np.float64,
        )

    summary = np.array(
        [
            position_x,
            position_y,
            speed,
            heading_sin,
            heading_cos,
            wall[0],
            wall[1],
            wall[2],
            wall[3],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(summary)):
        raise ValueError(f"segment {segment.segment_id!r} has non-finite metadata summaries")
    return summary


def _metadata_array(metadata: Mapping[str, Any], keys: Sequence[str]) -> FloatArray:
    for key in keys:
        if key in metadata:
            value = np.asarray(metadata[key], dtype=np.float64)
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"metadata field {key!r} is empty or non-finite")
            return value
    raise ValueError(f"metadata is missing one of the required fields: {tuple(keys)!r}")


def _metadata_mean(metadata: Mapping[str, Any], keys: Sequence[str]) -> float:
    return float(np.mean(_metadata_array(metadata, keys)))


def _optional_mean(metadata: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    if not any(key in metadata for key in keys):
        return None
    return _metadata_mean(metadata, keys)


def _optional_vector(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
    dimension: int,
) -> FloatArray | None:
    if not any(key in metadata for key in keys):
        return None
    value = _metadata_array(metadata, keys)
    if value.ndim == 1 and value.shape[0] == dimension:
        return value
    if value.ndim >= 2 and value.shape[-1] == dimension:
        return np.mean(value.reshape(-1, dimension), axis=0)
    raise ValueError(
        f"metadata field from {tuple(keys)!r} must end in dimension {dimension}"
    )


def _labels(
    segments: tuple[Segment, ...],
    explicit: Sequence[Hashable] | None,
    *,
    metadata_key: str,
    label: str,
) -> tuple[Hashable, ...]:
    if explicit is None:
        values: tuple[Hashable, ...] = tuple(
            _metadata_label(segment, metadata_key) for segment in segments
        )
    else:
        values = tuple(explicit)
        if len(values) != len(segments):
            raise ValueError(f"{label} must contain one value per segment")
    for value in values:
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(f"values in {label} must be hashable") from error
    return values


def _metadata_label(segment: Segment, key: str) -> Hashable:
    if key not in segment.metadata:
        raise ValueError(
            f"segment {segment.segment_id!r} metadata is missing required label {key!r}"
        )
    value = segment.metadata[key]
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"metadata label {key!r} must be hashable") from error
    return value


def _group_indices(
    states: Sequence[Hashable], wall_axes: Sequence[Hashable]
) -> dict[tuple[Hashable, Hashable], tuple[int, ...]]:
    mutable: dict[tuple[Hashable, Hashable], list[int]] = {}
    for index, (state, wall_axis) in enumerate(zip(states, wall_axes)):
        mutable.setdefault((state, wall_axis), []).append(index)
    return {key: tuple(indices) for key, indices in mutable.items()}


def _relative_length_difference(source_length: int, target_length: int) -> float:
    return abs(source_length - target_length) / max(source_length, target_length)
