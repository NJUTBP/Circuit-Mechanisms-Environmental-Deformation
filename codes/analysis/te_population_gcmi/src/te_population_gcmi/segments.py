"""Center-to-boundary movement segmentation with auditable geometry labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .io import TrajectoryData


WALL_NAMES = ("left", "right", "bottom", "top")
WALL_AXES = {"left": "x", "right": "x", "bottom": "y", "top": "y"}


@dataclass(frozen=True)
class MovementSegment:
    """A half-open time interval connecting center and boundary anchor zones."""

    segment_id: str
    trajectory_id: str
    start: int
    stop: int
    direction: str
    wall: str
    wall_axis: str
    axis_relation: str
    n_frames: int
    duration_s: float
    path_length_m: float
    net_displacement_m: float
    wall_distance_change_m: float
    monotonic_fraction: float
    mean_speed_m_s: float
    passed_qc: bool
    qc_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"approach", "departure"}:
            raise ValueError("direction must be approach or departure")
        if self.wall not in WALL_NAMES or self.wall_axis != WALL_AXES[self.wall]:
            raise ValueError("Invalid wall/wall_axis combination")
        if self.axis_relation not in {"deformed", "orthogonal", "baseline"}:
            raise ValueError("Invalid axis_relation")
        if self.start < 0 or self.stop <= self.start or self.n_frames != self.stop - self.start:
            raise ValueError("Invalid half-open segment bounds")

    def valid_lagged_samples(
        self, lag_frames: int, source_history: int, target_history: int
    ) -> int:
        """Return samples available without crossing this segment's boundaries."""

        if min(lag_frames, source_history, target_history) < 1:
            raise ValueError("Lag and history values must be positive")
        required_past = max(
            target_history,
            lag_frames + source_history - 1,
        )
        return max(0, self.n_frames - required_past)

    @property
    def end(self) -> int:
        """Exclusive end index (alias retained for tabular/data-contract wording)."""

        return self.stop

    def relative_progress(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_frames, dtype=np.float64)


@dataclass(frozen=True)
class SegmentationResult:
    segments: tuple[MovementSegment, ...]
    zone_labels: np.ndarray
    wall_distances_m: np.ndarray
    nearest_wall: np.ndarray
    n_candidates: int

    @property
    def accepted(self) -> tuple[MovementSegment, ...]:
        return tuple(segment for segment in self.segments if segment.passed_qc)

    @property
    def rejected(self) -> tuple[MovementSegment, ...]:
        return tuple(segment for segment in self.segments if not segment.passed_qc)


def arena_bounds(trajectory: TrajectoryData) -> tuple[float, float, float, float]:
    metadata = trajectory.metadata
    xmin = float(metadata.get("x_min_m", 0.0))
    ymin = float(metadata.get("y_min_m", 0.0))
    xmax = float(metadata.get("x_max_m", xmin + trajectory.width_m))
    ymax = float(metadata.get("y_max_m", ymin + trajectory.height_m))
    if not np.isclose(xmax - xmin, trajectory.width_m, rtol=1e-6, atol=1e-9):
        raise ValueError("Explicit x arena bounds disagree with width_m")
    if not np.isclose(ymax - ymin, trajectory.height_m, rtol=1e-6, atol=1e-9):
        raise ValueError("Explicit y arena bounds disagree with height_m")
    return xmin, xmax, ymin, ymax


def wall_distances(
    positions: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    matrix = np.asarray(positions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("positions must have shape (n_time, 2)")
    xmin, xmax, ymin, ymax = bounds
    distances = np.column_stack(
        (
            matrix[:, 0] - xmin,
            xmax - matrix[:, 0],
            matrix[:, 1] - ymin,
            ymax - matrix[:, 1],
        )
    )
    tolerance = 1e-6 * max(xmax - xmin, ymax - ymin, 1.0)
    if np.any(distances < -tolerance):
        raise ValueError("Positions lie outside the declared arena bounds")
    return np.maximum(distances, 0.0)


def classify_zones(
    trajectory: TrajectoryData,
    *,
    boundary_zone_m: float,
    centre_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify frames as transition=0, center=1, or boundary=2."""

    if boundary_zone_m <= 0 or not np.isfinite(boundary_zone_m):
        raise ValueError("boundary_zone_m must be finite and positive")
    if not (0 < centre_fraction < 1):
        raise ValueError("centre_fraction must lie strictly between zero and one")
    bounds = arena_bounds(trajectory)
    xmin, xmax, ymin, ymax = bounds
    margin_x = (1.0 - centre_fraction) * (xmax - xmin) / 2.0
    margin_y = (1.0 - centre_fraction) * (ymax - ymin) / 2.0
    if boundary_zone_m >= min(margin_x, margin_y):
        raise ValueError(
            "boundary_zone_m overlaps the central rectangle; reduce the boundary "
            "threshold or centre_fraction"
        )
    distances = wall_distances(trajectory.positions, bounds)
    nearest = np.argmin(distances, axis=1).astype(np.int8)
    x = trajectory.positions[:, 0]
    y = trajectory.positions[:, 1]
    center = (
        (x >= xmin + margin_x)
        & (x <= xmax - margin_x)
        & (y >= ymin + margin_y)
        & (y <= ymax - margin_y)
    )
    boundary = np.min(distances, axis=1) <= boundary_zone_m
    if np.any(center & boundary):  # guarded analytically, retained for audit safety
        raise RuntimeError("Center and boundary classifications overlap")
    labels = np.zeros(trajectory.n_time, dtype=np.int8)
    labels[center] = 1
    labels[boundary] = 2
    return labels, distances, nearest


def _candidate_intervals(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Find successive opposite-zone transitions using the last origin anchor."""

    candidates: list[tuple[int, int, str]] = []
    state: int | None = None
    last_anchor: int | None = None
    for index, label_raw in enumerate(labels):
        label = int(label_raw)
        if label == 0:
            continue
        if state is None:
            state = label
            last_anchor = index
            continue
        if label == state:
            last_anchor = index
            continue
        assert last_anchor is not None
        direction = "approach" if state == 1 and label == 2 else "departure"
        candidates.append((last_anchor, index + 1, direction))
        state = label
        last_anchor = index
    return candidates


def _wall_distance_for_name(distances: np.ndarray, wall: str) -> np.ndarray:
    return distances[:, WALL_NAMES.index(wall)]


def segment_trajectory(
    trajectory: TrajectoryData,
    segmentation: Mapping[str, Any],
) -> SegmentationResult:
    """Classify approach/departure episodes and retain explicit QC outcomes."""

    labels, distances, nearest = classify_zones(
        trajectory,
        boundary_zone_m=float(segmentation["boundary_zone_m"]),
        centre_fraction=float(segmentation["centre_fraction"]),
    )
    candidates = _candidate_intervals(labels)
    # Opt-in centre-side extension. Existing configurations keep exact old bounds.
    from .center_anchor import endpoint_extensions
    xmin, _, ymin, _ = arena_bounds(trajectory)
    extensions = endpoint_extensions(
        trajectory.positions - np.asarray([xmin, ymin]),
        trajectory.width_m, trajectory.height_m, segmentation,
    )
    if extensions is not None and len(extensions) != len(candidates):
        raise RuntimeError("Centre-extension candidate count differs from classifier")
    minimum_frames = int(segmentation["minimum_segment_frames"])
    maximum_raw = segmentation.get("maximum_segment_frames")
    maximum_frames = None if maximum_raw is None else int(maximum_raw)
    minimum_monotonic = float(segmentation.get("minimum_monotonic_fraction", 0.0))
    minimum_path = float(segmentation.get("minimum_path_length_m", 0.0))
    wall_tie_tolerance = float(segmentation.get("wall_tie_tolerance_m", 1e-9))
    velocity = np.diff(trajectory.positions, axis=0) / trajectory.dt_s
    frame_speed = np.empty(trajectory.n_time, dtype=np.float64)
    frame_speed[0] = np.linalg.norm(velocity[0])
    frame_speed[1:] = np.linalg.norm(velocity, axis=1)
    frame_heading = np.empty(trajectory.n_time, dtype=np.float64)
    frame_heading[0] = np.arctan2(velocity[0, 1], velocity[0, 0])
    frame_heading[1:] = np.arctan2(velocity[:, 1], velocity[:, 0])
    heading_sin = np.sin(frame_heading)
    heading_cos = np.cos(frame_heading)
    stationary = frame_speed <= np.finfo(float).eps
    heading_sin[stationary] = 0.0
    heading_cos[stationary] = 0.0
    segments: list[MovementSegment] = []
    for counter, (start, stop, direction) in enumerate(candidates):
        boundary_index = stop - 1 if direction == "approach" else start
        extension_metadata = {}
        if extensions is not None:
            start, stop, extension_metadata = extensions[counter]
        wall = WALL_NAMES[int(nearest[boundary_index])]
        axis = WALL_AXES[wall]
        if trajectory.deformed_axis in {"x", "y"}:
            relation = "deformed" if axis == trajectory.deformed_axis else "orthogonal"
        else:
            relation = "baseline"
        positions = trajectory.positions[start:stop]
        increments = np.diff(positions, axis=0)
        path_length = float(np.linalg.norm(increments, axis=1).sum())
        net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        wall_trace = _wall_distance_for_name(distances[start:stop], wall)
        wall_differences = np.diff(wall_trace)
        if wall_differences.size:
            correct = wall_differences <= 1e-12 if direction == "approach" else wall_differences >= -1e-12
            monotonic = float(np.mean(correct))
        else:
            monotonic = 0.0
        wall_change = float(wall_trace[-1] - wall_trace[0])
        reasons: list[str] = []
        ordered_wall_distances = np.sort(distances[boundary_index])
        if ordered_wall_distances[1] - ordered_wall_distances[0] <= wall_tie_tolerance:
            reasons.append("ambiguous_nearest_wall")
        n_frames = stop - start
        if n_frames < minimum_frames:
            reasons.append("too_short")
        if maximum_frames is not None and n_frames > maximum_frames:
            reasons.append("too_long")
        if path_length < minimum_path:
            reasons.append("insufficient_path_length")
        if monotonic < minimum_monotonic:
            reasons.append("low_wall_distance_monotonicity")
        if direction == "approach" and wall_change >= 0:
            reasons.append("wrong_net_wall_direction")
        if direction == "departure" and wall_change <= 0:
            reasons.append("wrong_net_wall_direction")
        if not np.all(np.isfinite(positions)):
            reasons.append("nonfinite_position")
        identifier = f"{trajectory.trajectory_id}:{counter:06d}:{direction}:{wall}"
        segments.append(
            MovementSegment(
                segment_id=identifier,
                trajectory_id=str(trajectory.trajectory_id),
                start=start,
                stop=stop,
                direction=direction,
                wall=wall,
                wall_axis=axis,
                axis_relation=relation,
                n_frames=n_frames,
                duration_s=float((n_frames - 1) * trajectory.dt_s),
                path_length_m=path_length,
                net_displacement_m=net_displacement,
                wall_distance_change_m=wall_change,
                monotonic_fraction=monotonic,
                mean_speed_m_s=float(np.mean(frame_speed[start:stop])),
                passed_qc=not reasons,
                qc_reasons=tuple(reasons),
                metadata={
                    **extension_metadata,
                    "network_id": trajectory.network_id,
                    "stage": trajectory.stage,
                    "environment": trajectory.environment,
                    "session": trajectory.session,
                    "deformed_axis": trajectory.deformed_axis,
                    "state": direction,
                    "wall_axis": axis,
                    "position_mean": np.mean(positions, axis=0).tolist(),
                    "mean_speed": float(np.mean(frame_speed[start:stop])),
                    "mean_heading_sin": float(np.mean(heading_sin[start:stop])),
                    "mean_heading_cos": float(np.mean(heading_cos[start:stop])),
                    "mean_wall_distances": np.mean(
                        distances[start:stop], axis=0
                    ).tolist(),
                    "mean_nearest_wall_distance": float(
                        np.mean(np.min(distances[start:stop], axis=1))
                    ),
                    "origin_zone": "center" if direction == "approach" else "boundary",
                    "destination_zone": "boundary" if direction == "approach" else "center",
                },
            )
        )
    return SegmentationResult(
        segments=tuple(segments),
        zone_labels=labels,
        wall_distances_m=distances,
        nearest_wall=nearest,
        n_candidates=len(candidates),
    )


def extract_segment(values: np.ndarray, segment: MovementSegment) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or segment.stop > matrix.shape[0]:
        raise ValueError("Signal cannot be sliced by the requested segment")
    return matrix[segment.start : segment.stop]


def to_te_segment(
    movement: MovementSegment,
    *,
    source: np.ndarray,
    target: np.ndarray,
    conditioning: np.ndarray | None = None,
    weight: float = 1.0,
) -> Any:
    """Convert a movement interval to the package's core ``types.Segment``.

    The import is intentionally local so segmentation remains usable while the
    estimator types module is optional during data-audit workflows.
    """

    from .types import Segment

    source_slice = extract_segment(source, movement)
    target_slice = extract_segment(target, movement)
    conditioning_slice = (
        None if conditioning is None else extract_segment(conditioning, movement)
    )
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Segment weight must be finite and positive")
    return Segment(
        source=source_slice,
        target=target_slice,
        conditioning=conditioning_slice,
        segment_id=movement.segment_id,
        trajectory_id=movement.trajectory_id,
        weight=float(weight),
        metadata={
            **movement.metadata,
            "direction": movement.direction,
            "wall": movement.wall,
            "wall_axis": movement.wall_axis,
            "axis_relation": movement.axis_relation,
            "passed_segmentation_qc": movement.passed_qc,
        },
    )


def segmentation_qc_table(
    results: Iterable[SegmentationResult],
) -> pd.DataFrame:
    """Return one audit row per candidate segment, including rejected ones."""

    rows: list[dict[str, Any]] = []
    for result in results:
        for segment in result.segments:
            rows.append(
                {
                    **segment.metadata,
                    "segment_id": segment.segment_id,
                    "trajectory_id": segment.trajectory_id,
                    "start_frame": segment.start,
                    "stop_frame_exclusive": segment.stop,
                    "direction": segment.direction,
                    "wall": segment.wall,
                    "wall_axis": segment.wall_axis,
                    "axis_relation": segment.axis_relation,
                    "n_frames": segment.n_frames,
                    "duration_s": segment.duration_s,
                    "path_length_m": segment.path_length_m,
                    "net_displacement_m": segment.net_displacement_m,
                    "wall_distance_change_m": segment.wall_distance_change_m,
                    "monotonic_fraction": segment.monotonic_fraction,
                    "mean_speed_m_s": segment.mean_speed_m_s,
                    "passed_qc": segment.passed_qc,
                    "qc_reasons": ";".join(segment.qc_reasons),
                }
            )
    return pd.DataFrame(rows)


def filter_segments(
    segments: Sequence[MovementSegment],
    *,
    direction: str | None = None,
    axis_relation: str | None = None,
    wall: str | None = None,
    passed_qc: bool = True,
) -> tuple[MovementSegment, ...]:
    """Select a predeclared state/wall pool without silently changing QC."""

    if direction is not None and direction not in {"approach", "departure"}:
        raise ValueError("Invalid direction filter")
    if axis_relation is not None and axis_relation not in {
        "deformed",
        "orthogonal",
        "baseline",
    }:
        raise ValueError("Invalid axis_relation filter")
    if wall is not None and wall not in WALL_NAMES:
        raise ValueError("Invalid wall filter")
    return tuple(
        segment
        for segment in segments
        if (not passed_qc or segment.passed_qc)
        and (direction is None or segment.direction == direction)
        and (axis_relation is None or segment.axis_relation == axis_relation)
        and (wall is None or segment.wall == wall)
    )
