"""Manifest and canonical time-major NPZ input contract.

No array orientation is inferred from its shape.  A string in ``array_keys``
declares that the corresponding NPZ array is already time-major.  Non-canonical
inputs must use ``{"key": ..., "time_axis": 1}`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from .config import Config


MANIFEST_REQUIRED_COLUMNS = (
    "network_id",
    "stage",
    "environment",
    "session",
    "trajectory_id",
    "width_m",
    "height_m",
    "deformed_axis",
    "data_path",
)


@dataclass(frozen=True)
class TrajectoryData:
    """One trajectory with every signal stored as ``(time, features)``."""

    network_id: str
    stage: str
    environment: str
    session: str | int
    trajectory_id: str
    width_m: float
    height_m: float
    deformed_axis: str | None
    dt_s: float
    positions: np.ndarray
    place_activity: np.ndarray
    grid_activity: dict[str, np.ndarray]
    bvc_activity: np.ndarray | None = None
    speed: np.ndarray | None = None
    heading: np.ndarray | None = None
    time_s: np.ndarray | None = None
    data_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_time = int(self.positions.shape[0])
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have canonical shape (n_time, 2)")
        if self.place_activity.ndim != 2 or self.place_activity.shape[0] != n_time:
            raise ValueError("place_activity must have shape (n_time, n_place_cells)")
        if not self.grid_activity:
            raise ValueError("grid_activity must contain at least one module")
        for module, activity in self.grid_activity.items():
            if activity.ndim != 2 or activity.shape[0] != n_time:
                raise ValueError(
                    f"grid_activity[{module!r}] must have shape (n_time, n_cells)"
                )
        for name, array in (
            ("bvc_activity", self.bvc_activity),
            ("speed", self.speed),
            ("heading", self.heading),
        ):
            if array is not None and (array.ndim != 2 or array.shape[0] != n_time):
                raise ValueError(f"{name} must have shape (n_time, n_features)")
        if self.time_s is not None:
            if self.time_s.ndim != 1 or self.time_s.shape[0] != n_time:
                raise ValueError("time_s must have shape (n_time,)")
        if n_time < 2:
            raise ValueError("A trajectory must contain at least two time points")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if not np.isfinite(self.width_m) or self.width_m <= 0:
            raise ValueError("width_m must be finite and positive")
        if not np.isfinite(self.height_m) or self.height_m <= 0:
            raise ValueError("height_m must be finite and positive")

    @property
    def n_time(self) -> int:
        return int(self.positions.shape[0])

    @property
    def key(self) -> tuple[str, str, str, str | int, str]:
        return (
            self.network_id,
            self.stage,
            self.environment,
            self.session,
            self.trajectory_id,
        )


def _clean_identifier_column(frame: pd.DataFrame, column: str) -> None:
    if frame[column].isna().any():
        rows = frame.index[frame[column].isna()].tolist()[:10]
        raise ValueError(f"Manifest column {column!r} has missing values at rows {rows}")
    frame[column] = frame[column].astype(str).str.strip()
    empty = frame[column].eq("")
    if empty.any():
        raise ValueError(
            f"Manifest column {column!r} has empty values at rows "
            f"{frame.index[empty].tolist()[:10]}"
        )


def load_manifest(
    source: str | Path | Config,
    *,
    config: Config | None = None,
    check_files: bool = False,
) -> pd.DataFrame:
    """Load and validate a manifest CSV.

    Relative ``data_path`` entries are interpreted relative to the manifest,
    never relative to the process working directory.  The returned frame keeps
    the user-provided path in ``data_path`` and stores the manifest location in
    ``frame.attrs``.
    """

    if isinstance(source, Config):
        if config is not None and config is not source:
            raise ValueError("Pass the Config either as source or config, not both")
        config_object = source
        manifest_path = source.resolve_path(source.data["manifest_path"])
    else:
        config_object = config
        manifest_path = Path(source).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_path}")
    frame = pd.read_csv(
        manifest_path,
        dtype={
            "network_id": "string",
            "stage": "string",
            "environment": "string",
            "trajectory_id": "string",
            "deformed_axis": "string",
            "data_path": "string",
        },
    )
    missing = sorted(set(MANIFEST_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Manifest contains no data rows")
    for column in (
        "network_id",
        "stage",
        "environment",
        "trajectory_id",
        "deformed_axis",
        "data_path",
    ):
        _clean_identifier_column(frame, column)
    if frame["session"].isna().any():
        raise ValueError("Manifest column 'session' contains missing values")
    for column in ("width_m", "height_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(frame[column].to_numpy()) | (frame[column] <= 0)
        if invalid.any():
            raise ValueError(
                f"Manifest column {column!r} must be positive and finite; bad rows "
                f"{frame.index[invalid].tolist()[:10]}"
            )
    allowed_axes = {"x", "y", "none", "na", "baseline"}
    bad_axes = ~frame["deformed_axis"].str.lower().isin(allowed_axes)
    if bad_axes.any():
        raise ValueError(
            "deformed_axis must be x, y, none, na, or baseline; bad rows "
            f"{frame.index[bad_axes].tolist()[:10]}"
        )
    identity = ["network_id", "stage", "environment", "session", "trajectory_id"]
    duplicate = frame.duplicated(identity, keep=False)
    if duplicate.any():
        raise ValueError(
            "Manifest trajectory identities must be unique; duplicate rows "
            f"{frame.index[duplicate].tolist()[:10]}"
        )
    root = manifest_path.parent
    if check_files:
        missing_files = [
            str((root / Path(value)).resolve())
            for value in frame["data_path"]
            if not (root / Path(value)).resolve().is_file()
        ]
        if missing_files:
            preview = missing_files[:10]
            raise FileNotFoundError(f"Manifest references missing NPZ files: {preview}")
    frame.attrs["manifest_path"] = str(manifest_path)
    frame.attrs["manifest_dir"] = str(root)
    if config_object is not None:
        validate_manifest_design(frame, config_object)
    return frame


def validate_manifest_design(manifest: pd.DataFrame, config: Config) -> None:
    """Validate replication and complete baseline/primary-target coverage."""

    minimum_trajectories = int(config.data.get("minimum_trajectories_per_group", 2))
    if minimum_trajectories < 2:
        raise ValueError(
            "data.minimum_trajectories_per_group cannot be below two because "
            "leave-one-trajectory-out conditioning is mandatory"
        )
    group_columns = ["network_id", "stage", "environment", "session"]
    counts = (
        manifest.groupby(group_columns, dropna=False)["trajectory_id"]
        .nunique()
        .rename("n_trajectories")
    )
    incomplete = counts[counts < minimum_trajectories]
    if not incomplete.empty:
        preview = [tuple(index) + (int(value),) for index, value in incomplete.head(10).items()]
        raise ValueError(
            "Every manifest condition requires at least "
            f"{minimum_trajectories} distinct trajectories; incomplete groups "
            f"(network, stage, environment, session, n): {preview}"
        )
    networks = set(manifest["network_id"].astype(str))
    baseline = select_baseline_manifest(manifest, config.baseline)
    baseline_networks = set(baseline["network_id"].astype(str))
    missing_baseline = sorted(networks.difference(baseline_networks))
    if missing_baseline:
        raise ValueError(
            f"Networks without the configured familiar baseline: {missing_baseline}"
        )
    for target in config.statistics["primary_targets"]:
        target_mask = (
            manifest["stage"].astype(str).eq(str(target["stage"]))
            & manifest["environment"].astype(str).eq(str(target["environment"]))
            & manifest["session"].astype(str).eq(str(target["session"]))
        )
        present = set(manifest.loc[target_mask, "network_id"].astype(str))
        missing = sorted(networks.difference(present))
        if missing:
            identity = (target["stage"], target["environment"], target["session"])
            raise ValueError(
                f"Primary target {identity} is absent for networks: {missing}"
            )


def select_baseline_manifest(
    manifest: pd.DataFrame, baseline: Mapping[str, Any]
) -> pd.DataFrame:
    """Return only the explicitly declared familiar-baseline rows."""

    mask = (
        manifest["stage"].astype(str).eq(str(baseline["stage"]))
        & manifest["environment"].astype(str).eq(str(baseline["environment"]))
        & manifest["session"].astype(str).eq(str(baseline["session"]))
    )
    selected = manifest.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            "The manifest contains no rows matching the configured familiar baseline"
        )
    selected.attrs.update(manifest.attrs)
    return selected


def _array_spec(specification: Any, logical_name: str) -> tuple[str, int]:
    if isinstance(specification, str):
        # A string is an explicit declaration of the canonical time-major form.
        return specification, 0
    if isinstance(specification, Mapping):
        key = specification.get("key")
        axis = specification.get("time_axis")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid key for {logical_name}")
        if axis not in (0, 1):
            raise ValueError(f"{logical_name}.time_axis must be 0 or 1")
        return key, int(axis)
    raise ValueError(f"Invalid array specification for {logical_name}")


def _read_time_major(
    archive: Mapping[str, np.ndarray],
    specification: Any,
    logical_name: str,
    *,
    allow_vector: bool = True,
) -> np.ndarray:
    key, time_axis = _array_spec(specification, logical_name)
    if key not in archive:
        raise KeyError(f"NPZ file is missing {logical_name} array {key!r}")
    raw = np.asarray(archive[key])
    if raw.dtype.kind in {"O", "U", "S", "V"}:
        raise TypeError(f"{logical_name} must be a numeric array, got {raw.dtype}")
    if raw.ndim == 0:
        raise ValueError(f"{logical_name} must have a time dimension")
    if raw.ndim == 1:
        if not allow_vector:
            raise ValueError(f"{logical_name} must be a two-dimensional array")
        if time_axis != 0:
            raise ValueError(f"A one-dimensional {logical_name} must use time_axis=0")
        return np.asarray(raw, dtype=np.float64).reshape(-1, 1)
    if time_axis >= raw.ndim:
        raise ValueError(f"time_axis for {logical_name} exceeds its rank")
    canonical = np.moveaxis(raw, time_axis, 0)
    canonical = canonical.reshape(canonical.shape[0], -1)
    return np.ascontiguousarray(canonical, dtype=np.float64)


def _resolve_data_path(row: Mapping[str, Any], manifest_dir: str | Path | None) -> Path:
    supplied = Path(str(row["data_path"]))
    if supplied.is_absolute():
        result = supplied.resolve()
    else:
        if manifest_dir is None:
            raise ValueError(
                "manifest_dir is required for a relative data_path. Pass a row from "
                "load_manifest() or provide manifest_dir explicitly."
            )
        result = (Path(manifest_dir) / supplied).resolve()
    if result.suffix.lower() != ".npz":
        raise ValueError(f"Trajectory input must be an NPZ file: {result}")
    if not result.is_file():
        raise FileNotFoundError(f"Trajectory NPZ does not exist: {result}")
    return result


def _resolve_dt_s(
    archive: Mapping[str, np.ndarray],
    row: Mapping[str, Any],
    data_config: Mapping[str, Any],
    time_s: np.ndarray | None,
) -> float:
    expected = float(data_config["expected_dt_s"])
    candidates: list[tuple[str, float]] = [("data.expected_dt_s", expected)]
    if "dt_s" in row and not pd.isna(row["dt_s"]):
        candidates.append(("manifest.dt_s", float(row["dt_s"])))
    key = data_config["array_keys"].get("dt_s")
    if key is not None:
        if key not in archive:
            raise KeyError(f"NPZ file is missing declared dt_s key {key!r}")
        scalar = np.asarray(archive[key])
        if scalar.size != 1:
            raise ValueError(f"NPZ {key!r} must be scalar")
        candidates.append((f"NPZ.{key}", float(scalar.reshape(-1)[0])))
    tolerance = float(data_config.get("dt_relative_tolerance", 1e-6))
    if time_s is not None:
        increments = np.diff(time_s)
        if not np.all(np.isfinite(increments)) or np.any(increments <= 0):
            raise ValueError("time array must be finite and strictly increasing")
        observed = float(np.median(increments))
        if not np.allclose(increments, observed, rtol=tolerance, atol=1e-12):
            raise ValueError("time array is not regularly sampled within tolerance")
        candidates.append(("NPZ.time increments", observed))
    for name, value in candidates:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        if not np.isclose(value, expected, rtol=tolerance, atol=1e-12):
            raise ValueError(
                f"Sampling interval mismatch: {name}={value:g}, expected {expected:g}"
            )
    return expected


def load_trajectory_npz(
    row: Mapping[str, Any] | pd.Series,
    config: Config | Mapping[str, Any],
    *,
    manifest_dir: str | Path | None = None,
) -> TrajectoryData:
    """Load one manifest row under the explicit canonical NPZ contract."""

    data_config = config.data if isinstance(config, Config) else config
    if manifest_dir is None and isinstance(row, pd.Series):
        # pandas does not propagate DataFrame attrs to a row reliably, so callers
        # should normally pass manifest_dir.  This supports explicitly attached rows.
        manifest_dir = row.attrs.get("manifest_dir")
    if manifest_dir is None and isinstance(config, Config):
        manifest_path = config.resolve_path(config.data["manifest_path"])
        manifest_dir = manifest_path.parent
    data_path = _resolve_data_path(row, manifest_dir)
    keys = data_config["array_keys"]
    with np.load(data_path, allow_pickle=False) as archive:
        positions = _read_time_major(
            archive, keys["positions"], "positions", allow_vector=False
        )
        if positions.shape[1] != 2:
            raise ValueError(
                "positions must be explicitly stored/configured as (n_time, 2); "
                f"received canonicalized shape {positions.shape}"
            )
        place = _read_time_major(archive, keys["place"], "place")
        grids = {
            str(module): _read_time_major(
                archive, specification, f"grid module {module}"
            )
            for module, specification in keys["grid_modules"].items()
        }
        bvc = None
        if "bvc" in keys:
            bvc_key, _ = _array_spec(keys["bvc"], "bvc")
            if bvc_key in archive:
                bvc = _read_time_major(archive, keys["bvc"], "bvc")
            elif data_config.get("require_bvc", False):
                raise KeyError(f"NPZ file is missing required bvc array {bvc_key!r}")
        speed = (
            _read_time_major(archive, keys["speed"], "speed")
            if "speed" in keys
            else None
        )
        heading = (
            _read_time_major(archive, keys["heading"], "heading")
            if "heading" in keys
            else None
        )
        time_s = None
        if "time" in keys:
            time_matrix = _read_time_major(archive, keys["time"], "time")
            if time_matrix.shape[1] != 1:
                raise ValueError("time must contain exactly one value per frame")
            time_s = time_matrix[:, 0]
        dt_s = _resolve_dt_s(archive, row, data_config, time_s)

    arrays = {
        "positions": positions,
        "place": place,
        **{f"grid:{name}": array for name, array in grids.items()},
    }
    if bvc is not None:
        arrays["bvc"] = bvc
    if speed is not None:
        arrays["speed"] = speed
    if heading is not None:
        arrays["heading"] = heading
    n_time = positions.shape[0]
    mismatched = {name: value.shape[0] for name, value in arrays.items() if value.shape[0] != n_time}
    if mismatched:
        raise ValueError(
            f"All arrays must share n_time={n_time}; mismatched arrays: {mismatched}"
        )
    if not data_config.get("allow_nonfinite", False):
        bad = [name for name, value in arrays.items() if not np.all(np.isfinite(value))]
        if bad:
            raise ValueError(f"Non-finite values found in trajectory arrays: {bad}")

    axis_value = str(row["deformed_axis"]).strip().lower()
    deformed_axis = axis_value if axis_value in {"x", "y"} else None
    known_manifest = set(MANIFEST_REQUIRED_COLUMNS) | {"dt_s"}
    metadata = {str(k): row[k] for k in row.keys() if k not in known_manifest}
    return TrajectoryData(
        network_id=str(row["network_id"]),
        stage=str(row["stage"]),
        environment=str(row["environment"]),
        session=row["session"],
        trajectory_id=str(row["trajectory_id"]),
        width_m=float(row["width_m"]),
        height_m=float(row["height_m"]),
        deformed_axis=deformed_axis,
        dt_s=dt_s,
        positions=positions,
        place_activity=place,
        grid_activity=grids,
        bvc_activity=bvc,
        speed=speed,
        heading=heading,
        time_s=time_s,
        data_path=data_path,
        metadata=metadata,
    )


def iter_trajectories(
    manifest: pd.DataFrame,
    config: Config | Mapping[str, Any],
) -> Iterator[TrajectoryData]:
    """Yield validated trajectories in manifest order."""

    manifest_dir = manifest.attrs.get("manifest_dir")
    if manifest_dir is None:
        raise ValueError("Manifest is missing manifest_dir metadata; use load_manifest()")
    for _, row in manifest.iterrows():
        yield load_trajectory_npz(row, config, manifest_dir=manifest_dir)
