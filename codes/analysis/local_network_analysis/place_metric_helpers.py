"""Reanalyse locally available transients with main_run_prev/analysis defaults.

This is a sensitivity analysis.  It never edits the Methods-version results.
Only environments with two transient trajectories for every selected network
are included in network-level figures (currently 80, 100, and 120 cm).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "run" / "outputs" / "independent_networks"
DEFAULT_METHODS_OUTPUT = (
    PROJECT_ROOT / "local_network_analysis" / "outputs" / "initial_exposure_multinetwork_n7"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "local_network_analysis" / "outputs"
    / "initial_exposure_multinetwork_mainrun_defaults_n7"
)
OLD_ANALYSIS_ROOT = PROJECT_ROOT / "main_run_prev" / "analysis"

TRAJECTORIES = (0, 1)
MODULES = (1, 2, 3)
NOMINAL_GRID_SCALES_CM = {1: 36.9, 2: 54.1, 3: 76.8}
MODULE_COLORS = {1: "#2878B5", 2: "#D76A5D", 3: "#2A9D78"}
ENVIRONMENTS = (
    ("compressed_80", 80, "initial_exposure"),
    ("fam_100", 100, "baseline"),
    ("elongated_120", 120, "initial_exposure"),
)

# Defaults in main_run_prev/analysis.
BIN_SIZE_CM = 2.0
GRID_PHASE_BOUNDARY_CM = 12.0
GRID_PHASE_BOOTSTRAP = 50
GRID_PHASE_SMOOTH_SIGMA_BINS = 1.5
GRID_PHASE_MIN_OVERLAP_BINS = 20
GRID_RESCALE_RANGE_CM = 10.0
GRID_RESCALE_STEP_CM = 2.0
GRID_RESCALE_MODE = "valid"
GRID_RESCALE_MIN_OVERLAP = 0.5
GRID_RATEMAP_SMOOTH_SIGMA_BINS = 1.5
GRID_RATEMAP_PEAK_FRACTION = 0.2
GRID_RATEMAP_MIN_RATE = 0.1
PLACE_FIELD_SMOOTH_CM = 2.0
PLACE_FIELD_PEAK_FRACTION = 0.1
PLACE_PHASE_ANGLE_DEG = 45.0
PLACE_PHASE_MIN_SPEED_CM_S = 5.0
PLACE_PHASE_BOOTSTRAP = 100
PLACE_PHASE_DT_S = 0.02


@dataclass(frozen=True)
class Environment:
    identifier: str
    length_cm: int
    stage: str

    @property
    def ratio(self) -> float:
        return self.length_cm / 100.0

    @property
    def relative_dir(self) -> Path:
        return Path(self.stage) / self.identifier


ENVS = tuple(Environment(*item) for item in ENVIRONMENTS)


def _load_old_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, OLD_ANALYSIS_ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIELD_LENGTH = _load_old_module("legacy_field_length", "Field_length.py")
GRID_RESCALE = _load_old_module("legacy_grid_rescale", "grid_rescale_cal.py")


def _network_id(path: Path) -> int:
    match = re.fullmatch(r"network_(\d+)", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def transient_path(network_root: Path, env: Environment, trajectory: int) -> Path:
    return network_root / env.relative_dir / f"trajectory_{trajectory:02d}" / "transient_activity.npz"


def inventory(data_root: Path) -> pd.DataFrame:
    all_envs = (
        ("compressed_70", 70, "initial_exposure"),
        ("compressed_80", 80, "initial_exposure"),
        ("compressed_90", 90, "initial_exposure"),
        ("fam_100", 100, "baseline"),
        ("elongated_110", 110, "initial_exposure"),
        ("elongated_120", 120, "initial_exposure"),
        ("elongated_130", 130, "initial_exposure"),
    )
    rows = []
    for network_root in sorted(data_root.glob("network_*")):
        if not network_root.is_dir():
            continue
        network_id = _network_id(network_root)
        for identifier, length_cm, stage in all_envs:
            env = Environment(identifier, length_cm, stage)
            for trajectory in TRAJECTORIES:
                path = transient_path(network_root, env, trajectory)
                rows.append(
                    {
                        "network_id": network_id,
                        "environment": identifier,
                        "length_cm": length_cm,
                        "trajectory_id": trajectory,
                        "transient_available": path.is_file(),
                        "path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def select_complete_networks(data_root: Path, requested: list[int] | None) -> list[tuple[int, Path]]:
    roots = sorted((p for p in data_root.glob("network_*") if p.is_dir()), key=_network_id)
    if requested is not None:
        wanted = set(requested)
        roots = [p for p in roots if _network_id(p) in wanted]
        absent = wanted.difference({_network_id(p) for p in roots})
        if absent:
            raise FileNotFoundError(f"Missing requested network directories: {sorted(absent)}")
    complete = []
    for root in roots:
        if all(transient_path(root, env, trajectory).is_file() for env in ENVS for trajectory in TRAJECTORIES):
            complete.append((_network_id(root), root))
    if not complete:
        raise FileNotFoundError("No network has complete 80/100/120-cm transient data for both trajectories")
    return complete


def load_transient(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    with np.load(path) as data:
        required = {"positions", "place", "grid_m1", "grid_m2", "grid_m3", "record_dt_s"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        positions_cm = np.asarray(data["positions"], dtype=float) * 100.0
        activity = {
            "place": np.asarray(data["place"], dtype=float),
            "grid_m1": np.asarray(data["grid_m1"], dtype=float),
            "grid_m2": np.asarray(data["grid_m2"], dtype=float),
            "grid_m3": np.asarray(data["grid_m3"], dtype=float),
        }
        record_dt_s = float(data["record_dt_s"])
    if positions_cm.shape != (100000, 2):
        raise ValueError(f"Unexpected positions shape in {path}: {positions_cm.shape}")
    for name, values in activity.items():
        if values.shape != (100, 100000):
            raise ValueError(f"Unexpected {name} shape in {path}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {name}: {path}")
    return positions_cm, activity, record_dt_s


def spatial_bins(positions_cm: np.ndarray, length_cm: int) -> tuple[np.ndarray, int, int]:
    nx = int(math.ceil(length_cm / BIN_SIZE_CM))
    ny = int(math.ceil(100.0 / BIN_SIZE_CM))
    bx = np.clip((positions_cm[:, 0] / BIN_SIZE_CM).astype(int), 0, nx - 1)
    by = np.clip((positions_cm[:, 1] / BIN_SIZE_CM).astype(int), 0, ny - 1)
    return by * nx + bx, nx, ny


def binned_means(activity: np.ndarray, flat_bins: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    n_bins = nx * ny
    occupancy = np.bincount(flat_bins, minlength=n_bins).reshape(ny, nx).astype(float)
    sums = np.empty((activity.shape[0], ny, nx), dtype=float)
    for cell in range(activity.shape[0]):
        sums[cell] = np.bincount(flat_bins, weights=activity[cell], minlength=n_bins).reshape(ny, nx)
    means = np.zeros_like(sums)
    np.divide(sums, occupancy[None, :, :], out=means, where=occupancy[None, :, :] > 0)
    return means, occupancy


def legacy_grid_rate_maps(activity: np.ndarray, flat_bins: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Vectorized equivalent of ratemap_cal.cal_ratemap for the old analysis."""
    maps, _ = binned_means(activity, flat_bins, nx, ny)
    maps = ndimage.gaussian_filter(maps, sigma=(0.0, GRID_RATEMAP_SMOOTH_SIGMA_BINS, GRID_RATEMAP_SMOOTH_SIGMA_BINS))
    peaks = np.max(maps, axis=(1, 2))
    maps[maps < (GRID_RATEMAP_PEAK_FRACTION * peaks)[:, None, None]] = 0.0
    maps[maps < GRID_RATEMAP_MIN_RATE] = 0.0
    return maps


def legacy_place_rate_maps(activity: np.ndarray, flat_bins: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Vectorized equivalent of Field_length.compute_rate_map defaults."""
    n_bins = nx * ny
    occupancy = np.bincount(flat_bins, minlength=n_bins).reshape(ny, nx).astype(float)
    sums = np.empty((activity.shape[0], ny, nx), dtype=float)
    for cell in range(activity.shape[0]):
        sums[cell] = np.bincount(flat_bins, weights=activity[cell], minlength=n_bins).reshape(ny, nx)
    sigma_bins = PLACE_FIELD_SMOOTH_CM / BIN_SIZE_CM
    sums = ndimage.gaussian_filter(sums, sigma=(0.0, sigma_bins, sigma_bins))
    occ_smooth = ndimage.gaussian_filter(occupancy, sigma=sigma_bins)
    occ_smooth[occ_smooth < 0.1] = np.nan
    return sums / occ_smooth[None, :, :]


def boundary_labels(positions_cm: np.ndarray, length_cm: int) -> np.ndarray:
    """Exact stateful North/South/East/West precedence used by phase_shift_grid.py."""
    x, y = positions_cm[:, 0], positions_cm[:, 1]
    event = np.zeros(len(x), dtype=np.int8)
    event[y >= 100.0 - GRID_PHASE_BOUNDARY_CM] = 1
    south = (event == 0) & (y <= GRID_PHASE_BOUNDARY_CM)
    event[south] = 2
    east = (event == 0) & (x >= length_cm - GRID_PHASE_BOUNDARY_CM)
    event[east] = 3
    west = (event == 0) & (x <= GRID_PHASE_BOUNDARY_CM)
    event[west] = 4
    last = 0
    labels = np.empty_like(event)
    for i, value in enumerate(event):
        if value:
            last = int(value)
        labels[i] = last
    return labels


def velocity_labels(positions_cm: np.ndarray, axis: str) -> np.ndarray:
    """Exact central-difference and direction rules from phase_shift_place.py defaults."""
    velocity = np.zeros_like(positions_cm, dtype=float)
    velocity[1:-1] = (positions_cm[2:] - positions_cm[:-2]) / (2.0 * PLACE_PHASE_DT_S)
    velocity[0] = (positions_cm[1] - positions_cm[0]) / PLACE_PHASE_DT_S
    velocity[-1] = (positions_cm[-1] - positions_cm[-2]) / PLACE_PHASE_DT_S
    speed = np.linalg.norm(velocity, axis=1)
    angle = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    labels = np.zeros(len(angle), dtype=np.int8)
    fast = speed >= PLACE_PHASE_MIN_SPEED_CM_S
    labels[fast & (np.abs(angle) <= PLACE_PHASE_ANGLE_DEG)] = 1
    left = fast & ((np.abs(angle - 180.0) <= PLACE_PHASE_ANGLE_DEG) | (np.abs(angle + 180.0) <= PLACE_PHASE_ANGLE_DEG))
    labels[(labels == 0) & left] = 2
    labels[(labels == 0) & fast & (np.abs(angle - 90.0) <= PLACE_PHASE_ANGLE_DEG)] = 3
    labels[(labels == 0) & fast & (np.abs(angle + 90.0) <= PLACE_PHASE_ANGLE_DEG)] = 4
    return labels


def grouped_indices(flat_bins: np.ndarray, labels: np.ndarray, label_value: int, n_bins: int) -> list[np.ndarray]:
    indices = np.flatnonzero(labels == label_value)
    selected_bins = flat_bins[indices]
    order = np.argsort(selected_bins, kind="stable")
    indices = indices[order]
    selected_bins = selected_bins[order]
    cuts = np.searchsorted(selected_bins, np.arange(n_bins + 1))
    return [indices[cuts[i]:cuts[i + 1]] for i in range(n_bins)]


def matched_maps_all_cells(
    activity: np.ndarray,
    groups1: list[np.ndarray],
    groups2: list[np.ndarray],
    nx: int,
    ny: int,
    seed: int,
    nan_fill: bool,
    smooth_sigma: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch-equivalent bootstrap: old code resets the same seed for every cell."""
    rng = np.random.RandomState(seed)
    fill = np.nan if nan_fill else 0.0
    map1 = np.full((activity.shape[0], nx * ny), fill, dtype=float)
    map2 = np.full((activity.shape[0], nx * ny), fill, dtype=float)
    for bin_id, (idx1, idx2) in enumerate(zip(groups1, groups2)):
        n1, n2 = len(idx1), len(idx2)
        if n1 == 0 or n2 == 0:
            continue
        n_match = min(n1, n2)
        chosen1 = idx1[rng.choice(n1, n_match, replace=False)]
        chosen2 = idx2[rng.choice(n2, n_match, replace=False)]
        map1[:, bin_id] = np.mean(activity[:, chosen1], axis=1)
        map2[:, bin_id] = np.mean(activity[:, chosen2], axis=1)
    map1 = map1.reshape(activity.shape[0], ny, nx)
    map2 = map2.reshape(activity.shape[0], ny, nx)
    if smooth_sigma is not None:
        valid1 = np.isfinite(map1)
        valid2 = np.isfinite(map2)
        values1 = np.where(valid1, map1, 0.0)
        values2 = np.where(valid2, map2, 0.0)
        num1 = ndimage.gaussian_filter(values1, sigma=(0.0, smooth_sigma, smooth_sigma), mode="constant")
        num2 = ndimage.gaussian_filter(values2, sigma=(0.0, smooth_sigma, smooth_sigma), mode="constant")
        den1 = ndimage.gaussian_filter(valid1.astype(float), sigma=(0.0, smooth_sigma, smooth_sigma), mode="constant")
        den2 = ndimage.gaussian_filter(valid2.astype(float), sigma=(0.0, smooth_sigma, smooth_sigma), mode="constant")
        map1 = np.full_like(num1, np.nan)
        map2 = np.full_like(num2, np.nan)
        np.divide(num1, den1, out=map1, where=den1 > 1e-12)
        np.divide(num2, den2, out=map2, where=den2 > 1e-12)
        map1[~valid1] = np.nan
        map2[~valid2] = np.nan
    return map1, map2


def best_lags_all_cells(
    map1: np.ndarray, map2: np.ndarray, axis: str, max_lag: int,
    min_overlap_bins: int = GRID_PHASE_MIN_OVERLAP_BINS,
) -> np.ndarray:
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if axis == "x":
            if lag > 0:
                a, b = map1[:, :, :-lag], map2[:, :, lag:]
            elif lag < 0:
                a, b = map1[:, :, -lag:], map2[:, :, :lag]
            else:
                a, b = map1, map2
        else:
            if lag > 0:
                a, b = map1[:, :-lag, :], map2[:, lag:, :]
            elif lag < 0:
                a, b = map1[:, -lag:, :], map2[:, :lag, :]
            else:
                a, b = map1, map2
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
        valid = np.isfinite(a) & np.isfinite(b)
        count = valid.sum(axis=1)
        av = np.where(valid, a, 0.0)
        bv = np.where(valid, b, 0.0)
        ma = np.divide(av.sum(axis=1), count, out=np.zeros(a.shape[0]), where=count > 0)
        mb = np.divide(bv.sum(axis=1), count, out=np.zeros(a.shape[0]), where=count > 0)
        da = np.where(valid, a - ma[:, None], 0.0)
        db = np.where(valid, b - mb[:, None], 0.0)
        denom = np.sqrt(np.sum(da * da, axis=1) * np.sum(db * db, axis=1))
        corr = np.full(a.shape[0], np.nan)
        np.divide(np.sum(da * db, axis=1), denom, out=corr,
                  where=(count >= min_overlap_bins) & (denom > 0))
        correlations.append(corr)
    matrix = np.stack(correlations, axis=1)
    lags = np.arange(-max_lag, max_lag + 1)
    peaks = np.zeros(map1.shape[0], dtype=float)
    center_idx = len(lags) // 2
    for cell in range(map1.shape[0]):
        score = np.where(np.isfinite(matrix[cell]), matrix[cell], -1.0)
        positive = score > 0.0
        if not np.any(positive):
            continue
        labeled, n_blobs = ndimage.label(positive)
        if n_blobs == 0:
            continue
        best_blob = min(range(1, n_blobs + 1), key=lambda blob_id: abs(
            np.mean(np.where(labeled == blob_id)[0]) - center_idx))
        blob_score = score.copy()
        blob_score[labeled != best_blob] = -np.inf
        peaks[cell] = float(lags[np.argmax(blob_score)])
    return peaks


def legacy_grid_phase(
    positions_cm: np.ndarray,
    activity: np.ndarray,
    length_cm: int,
    module: int,
    axis: str,
    flat_bins: np.ndarray,
    nx: int,
    ny: int,
    n_bootstrap: int = GRID_PHASE_BOOTSTRAP,
    smooth_sigma: float | None = GRID_PHASE_SMOOTH_SIGMA_BINS,
    min_overlap_bins: int = GRID_PHASE_MIN_OVERLAP_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    labels = boundary_labels(positions_cm, length_cm)
    pair = (3, 4) if axis == "x" else (1, 2)
    n_bins = nx * ny
    groups1 = grouped_indices(flat_bins, labels, pair[0], n_bins)
    groups2 = grouped_indices(flat_bins, labels, pair[1], n_bins)
    if sum(map(len, groups1)) == 0 or sum(map(len, groups2)) == 0:
        nan = np.full(activity.shape[0], np.nan)
        return nan, nan
    scale = NOMINAL_GRID_SCALES_CM[module]
    max_lag = int(scale / (2.0 * BIN_SIZE_CM))
    lags = []
    for seed in range(n_bootstrap):
        maps = matched_maps_all_cells(
            activity, groups1, groups2, nx, ny, seed,
            nan_fill=True, smooth_sigma=smooth_sigma,
        )
        lags.append(best_lags_all_cells(
            *maps, axis=axis, max_lag=max_lag,
            min_overlap_bins=min_overlap_bins,
        ))
    shift_cm = np.abs(np.mean(np.stack(lags), axis=0)) * BIN_SIZE_CM
    ratio_half_period = 2.0 * shift_cm / scale
    return shift_cm, ratio_half_period


def legacy_place_phase(
    positions_cm: np.ndarray,
    activity: np.ndarray,
    length_cm: int,
    axis: str,
    flat_bins: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    labels = velocity_labels(positions_cm, axis)
    pair = (1, 2) if axis == "x" else (3, 4)
    if np.sum(labels == pair[0]) < 100 or np.sum(labels == pair[1]) < 100:
        return np.full(activity.shape[0], np.nan)
    n_bins = nx * ny
    groups1 = grouped_indices(flat_bins, labels, pair[0], n_bins)
    groups2 = grouped_indices(flat_bins, labels, pair[1], n_bins)
    max_lag = min(nx, ny) // 2
    lags = []
    for seed in range(PLACE_PHASE_BOOTSTRAP):
        maps = matched_maps_all_cells(
            activity, groups1, groups2, nx, ny, seed,
            nan_fill=False, smooth_sigma=None,
        )
        lags.append(best_lags_all_cells(*maps, axis=axis, max_lag=max_lag))
    return np.mean(np.stack(lags), axis=0) * BIN_SIZE_CM


def session_metrics(path: Path, env: Environment) -> dict[str, object]:
    positions, activity, stored_dt = load_transient(path)
    flat_bins, nx, ny = spatial_bins(positions, env.length_cm)
    grid_maps = {}
    grid_phase = {}
    for module in MODULES:
        values = activity[f"grid_m{module}"]
        grid_maps[module] = legacy_grid_rate_maps(values, flat_bins, nx, ny)
        for axis in ("x", "y"):
            grid_phase[(module, axis)] = legacy_grid_phase(
                positions, values, env.length_cm, module, axis, flat_bins, nx, ny
            )
    place_maps = legacy_place_rate_maps(activity["place"], flat_bins, nx, ny)
    place_length_x, place_length_y = FIELD_LENGTH.compute_place_field_length_batch(
        list(place_maps), bin_size=BIN_SIZE_CM, threshold_ratio=PLACE_FIELD_PEAK_FRACTION
    )
    place_phase_x = legacy_place_phase(
        positions, activity["place"], env.length_cm, "x", flat_bins, nx, ny
    )
    place_phase_y = legacy_place_phase(
        positions, activity["place"], env.length_cm, "y", flat_bins, nx, ny
    )
    return {
        "grid_maps": grid_maps,
        "grid_phase": grid_phase,
        "place_length_x": place_length_x,
        "place_length_y": place_length_y,
        "place_phase_x": place_phase_x,
        "place_phase_y": place_phase_y,
        "stored_record_dt_s": stored_dt,
    }


def analyse_network(network_id: int, network_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_env = next(env for env in ENVS if env.length_cm == 100)
    for trajectory in TRAJECTORIES:
        print(f"network={network_id:03d} trajectory={trajectory:02d} baseline", flush=True)
        baseline = session_metrics(transient_path(network_root, baseline_env, trajectory), baseline_env)
        for env in ENVS:
            if env.length_cm == 100:
                current = baseline
            else:
                print(f"network={network_id:03d} trajectory={trajectory:02d} env={env.identifier}", flush=True)
                current = session_metrics(transient_path(network_root, env, trajectory), env)
            for module in MODULES:
                if env.length_cm == 100:
                    factors = np.ones(100)
                    correlations = np.ones(100)
                else:
                    factors, correlations = GRID_RESCALE.compute_grid_rescale_factor_batch(
                        list(baseline["grid_maps"][module]),
                        list(current["grid_maps"][module]),
                        100.0,
                        float(env.length_cm),
                        bin_size=BIN_SIZE_CM,
                        scale_range=GRID_RESCALE_RANGE_CM,
                        scale_step=GRID_RESCALE_STEP_CM,
                        mode=GRID_RESCALE_MODE,
                        min_overlap_ratio=GRID_RESCALE_MIN_OVERLAP,
                    )
                phase_x, ratio_x = current["grid_phase"][(module, "x")]
                phase_y, ratio_y = current["grid_phase"][(module, "y")]
                base_x, base_ratio_x = baseline["grid_phase"][(module, "x")]
                base_y, base_ratio_y = baseline["grid_phase"][(module, "y")]
                for cell in range(100):
                    rows.append(
                        {
                            "population": "grid", "module": module, "cell_id": cell,
                            "network_id": network_id, "trajectory_id": trajectory,
                            "environment": env.identifier, "length_cm": env.length_cm,
                            "deformation_ratio": env.ratio,
                            "rescale_factor": factors[cell],
                            "rescale_best_correlation": correlations[cell],
                            "phase_shift_x_cm": phase_x[cell], "phase_shift_y_cm": phase_y[cell],
                            "phase_ratio_x_half_period": ratio_x[cell],
                            "phase_ratio_y_half_period": ratio_y[cell],
                            "delta_phase_shift_x_cm": phase_x[cell] - base_x[cell],
                            "delta_phase_shift_y_cm": phase_y[cell] - base_y[cell],
                            "delta_phase_ratio_x_half_period": ratio_x[cell] - base_ratio_x[cell],
                            "delta_phase_ratio_y_half_period": ratio_y[cell] - base_ratio_y[cell],
                        }
                    )
            for cell in range(100):
                rows.append(
                    {
                        "population": "place", "module": 0, "cell_id": cell,
                        "network_id": network_id, "trajectory_id": trajectory,
                        "environment": env.identifier, "length_cm": env.length_cm,
                        "deformation_ratio": env.ratio,
                        "field_length_x_cm": current["place_length_x"][cell],
                        "field_length_y_cm": current["place_length_y"][cell],
                        "place_phase_x_cm": current["place_phase_x"][cell],
                        "place_phase_y_cm": current["place_phase_y"][cell],
                        "delta_field_length_x_cm": current["place_length_x"][cell] - baseline["place_length_x"][cell],
                        "delta_field_length_y_cm": current["place_length_y"][cell] - baseline["place_length_y"][cell],
                        "delta_place_phase_x_cm": current["place_phase_x"][cell] - baseline["place_phase_x"][cell],
                        "delta_place_phase_y_cm": current["place_phase_y"][cell] - baseline["place_phase_y"][cell],
                    }
                )
    return rows


GRID_METRICS = (
    "rescale_factor", "rescale_best_correlation",
    "phase_shift_x_cm", "phase_shift_y_cm",
    "phase_ratio_x_half_period", "phase_ratio_y_half_period",
    "delta_phase_shift_x_cm", "delta_phase_shift_y_cm",
    "delta_phase_ratio_x_half_period", "delta_phase_ratio_y_half_period",
)
PLACE_METRICS = (
    "field_length_x_cm", "field_length_y_cm", "place_phase_x_cm", "place_phase_y_cm",
    "delta_field_length_x_cm", "delta_field_length_y_cm",
    "delta_place_phase_x_cm", "delta_place_phase_y_cm",
)


def aggregate(cell_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for population, metrics in (("grid", GRID_METRICS), ("place", PLACE_METRICS)):
        subset = cell_frame[cell_frame["population"] == population]
        keys = ["network_id", "trajectory_id", "environment", "length_cm", "deformation_ratio", "module"]
        for key, group in subset.groupby(keys, sort=True):
            for metric in metrics:
                values = group[metric].to_numpy(float)
                finite = values[np.isfinite(values)]
                rows.append(
                    dict(zip(keys, key))
                    | {"population": population, "metric": metric,
                       "trajectory_mean": float(np.mean(finite)) if finite.size else np.nan,
                       "n_cells_finite": int(finite.size)}
                )
    trajectory = pd.DataFrame(rows)
    network_rows = []
    keys = ["population", "network_id", "environment", "length_cm", "deformation_ratio", "module", "metric"]
    for key, group in trajectory.groupby(keys, sort=True):
        values = group["trajectory_mean"].to_numpy(float)
        finite = values[np.isfinite(values)]
        network_rows.append(
            dict(zip(keys, key))
            | {"network_mean": float(np.mean(finite)) if finite.size else np.nan,
               "sd_across_trajectories": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
               "n_trajectories_finite": int(finite.size)}
        )
    network = pd.DataFrame(network_rows)
    group_rows = []
    keys = ["population", "environment", "length_cm", "deformation_ratio", "module", "metric"]
    for key, group in network.groupby(keys, sort=True):
        values = group["network_mean"].to_numpy(float)
        finite = values[np.isfinite(values)]
        n = finite.size
        mean = float(np.mean(finite)) if n else np.nan
        sd = float(np.std(finite, ddof=1)) if n > 1 else np.nan
        sem = sd / math.sqrt(n) if n > 1 else np.nan
        if n > 1:
            half = float(stats.t.ppf(0.975, n - 1) * sem)
            ci_low, ci_high = mean - half, mean + half
        else:
            ci_low = ci_high = np.nan
        group_rows.append(
            dict(zip(keys, key))
            | {"group_mean": mean, "sd_across_networks": sd, "sem_across_networks": sem,
               "ci95_low": ci_low, "ci95_high": ci_high, "n_networks": int(n)}
        )
    return trajectory, network, pd.DataFrame(group_rows)


def parameter_table() -> pd.DataFrame:
    rows = [
        ("Rate maps", "bin_size", "2.5 cm", "2.0 cm", "Fig2 sometimes 1.25 cm; Fig3 2.0 cm"),
        ("Grid phase", "boundary_threshold", "10 cm", "12 cm", "Fig3 explicitly 10 cm"),
        ("Grid phase", "bootstrap", "100 per axis", "50 per axis (batch default)", "Fig3 x=10, y=50"),
        ("Grid phase", "grid spacing", "measured per cell from baseline", "fixed nominal 36.9/54.1/76.8 cm", "Fig3 fixed nominal"),
        ("Grid phase", "matched-map smoothing", "1.5 bins", "1.5 bins", "same"),
        ("Grid rescale", "search step", "2.5 cm", "2.0 cm", "old function default 2.0 cm"),
        ("Grid rescale", "search range/mode/overlap", "10 cm / valid / 0.5", "10 cm / valid / 0.5", "same"),
        ("Grid rescale map", "smoothing and threshold", "1.5 bins; no peak clipping in saved production maps", "1.5 bins; retain >=20% peak and >=0.1", "ratemap_cal defaults"),
        ("Place field", "smoothing", "2.0 bins (=5 cm)", "2.0 cm (=1 bin)", "old Field_length default"),
        ("Place field", "field threshold", "max(0.1 Hz, 20% peak)", "10% peak; no absolute cutoff", "Fig3 grid-field call used 50%"),
        ("Place field", "field geometry", "longest suprathreshold x/y run", "largest 2-D connected component span", "algorithmic difference"),
        ("Place phase", "minimum speed", "10 cm/s", "5 cm/s", "old phase_shift_place default"),
        ("Place phase", "velocity dt", "stored dt=0.01 s; first difference", "fixed dt=0.02 s; central difference", "algorithmic difference"),
        ("Place phase", "bootstrap", "none in production movement-displacement metric", "100", "old phase_shift_place default"),
    ]
    return pd.DataFrame(rows, columns=["analysis", "parameter", "methods_version", "main_run_default_version", "old_notebook_note"])


def compare_methods(network: pd.DataFrame, methods_output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = methods_output / "metrics" / "network_level_values.csv"
    if not path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    methods = pd.read_csv(path)
    keys = ["population", "network_id", "environment", "length_cm", "deformation_ratio", "module", "metric"]
    keep_env = {env.identifier for env in ENVS}
    keep_metrics = set(GRID_METRICS) | set(PLACE_METRICS)
    methods = methods[methods["environment"].isin(keep_env) & methods["metric"].isin(keep_metrics)]
    methods = methods[keys + ["network_mean"]].rename(columns={"network_mean": "network_mean_methods"})
    legacy = network[keys + ["network_mean"]].rename(columns={"network_mean": "network_mean_mainrun"})
    merged = legacy.merge(methods, on=keys, how="left", validate="one_to_one")
    merged["mainrun_minus_methods"] = merged["network_mean_mainrun"] - merged["network_mean_methods"]
    summary_rows = []
    group_keys = ["population", "environment", "length_cm", "deformation_ratio", "module", "metric"]
    for key, group in merged.groupby(group_keys, sort=True):
        values = group["mainrun_minus_methods"].to_numpy(float)
        values = values[np.isfinite(values)]
        summary_rows.append(
            dict(zip(group_keys, key))
            | {"mean_parameter_change": float(np.mean(values)) if values.size else np.nan,
               "sd_across_networks": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
               "n_networks": int(values.size)}
        )
    return merged, pd.DataFrame(summary_rows)


def _panel(ax, network: pd.DataFrame, metric: str, module: int, ylabel: str, color: str, baseline: float | None = None):
    subset = network[(network["metric"] == metric) & (network["module"] == module)].copy()
    for _, group in subset.groupby("network_id"):
        group = group.sort_values("deformation_ratio")
        ax.plot(group["deformation_ratio"], group["network_mean"], color="#B8B8B8", lw=0.8, alpha=0.65)
        ax.scatter(group["deformation_ratio"], group["network_mean"], color="#777777", s=12, alpha=0.75, zorder=2)
    grouped = subset.groupby("deformation_ratio")["network_mean"]
    xs, means, lows, highs = [], [], [], []
    for x, values in grouped:
        values = values.to_numpy(float)
        values = values[np.isfinite(values)]
        mean = np.mean(values)
        sem = stats.sem(values) if len(values) > 1 else np.nan
        half = stats.t.ppf(0.975, len(values) - 1) * sem if len(values) > 1 else 0.0
        xs.append(x); means.append(mean); lows.append(mean - half); highs.append(mean + half)
    ax.errorbar(xs, means, yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
                color=color, marker="o", ms=5, lw=1.8, capsize=3, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, color="#444444", lw=0.8, ls="--")
    ax.set_xticks([0.8, 1.0, 1.2])
    ax.set_xlim(0.76, 1.24)
    ax.set_xlabel("Environment length / familiar length")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Module {module}")
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, figures: Path, stem: str):
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(network: pd.DataFrame, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    for module, ax in zip(MODULES, axes):
        _panel(ax, network, "rescale_factor", module, "Grid rescaling factor", MODULE_COLORS[module], baseline=1.0)
        ax.plot([0.8, 1.2], [0.8, 1.2], color="#222222", ls=":", lw=1.0)
    save_figure(fig, figures, "figure_mainrun_grid_rescaling_network_level")

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), sharex=True)
    for col, module in enumerate(MODULES):
        _panel(axes[0, col], network, "phase_ratio_x_half_period", module,
               "X boundary phase / half-period", MODULE_COLORS[module])
        _panel(axes[1, col], network, "phase_ratio_y_half_period", module,
               "Y boundary phase / half-period", MODULE_COLORS[module])
    save_figure(fig, figures, "figure_mainrun_grid_boundary_phase_network_level")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    _panel(axes[0], network, "delta_field_length_x_cm", 0,
           "Change in place-field X length (cm)", "#2878B5", baseline=0.0)
    axes[0].set_title("Place-field length")
    _panel(axes[1], network, "delta_place_phase_x_cm", 0,
           "Change in place-cell X phase (cm)", "#E59A2F", baseline=0.0)
    axes[1].set_title("Place-cell phase")
    save_figure(fig, figures, "figure_mainrun_place_length_and_phase_change_network_level")


def write_readme(output: Path, networks: list[int], availability: pd.DataFrame):
    complete_counts = (
        availability.groupby(["environment", "length_cm"])["transient_available"].sum().reset_index()
    )
    lines = [
        "# Main-run-default parameter sensitivity analysis",
        "",
        "The original Methods-version output was left unchanged. This directory was generated",
        "from raw transient activity using defaults in `main_run_prev/analysis`.",
        "",
        f"Networks: {', '.join(f'{n:03d}' for n in networks)} (n={len(networks)}).",
        "Main network-level figures use 80, 100, and 120 cm because these are the only",
        "environment lengths with both trajectories available for every included network.",
        "Cells were averaged within trajectory, trajectories within network, and uncertainty",
        "was calculated across networks.",
        "",
        "Important: this is not a pure one-parameter-at-a-time test. The old functions also",
        "differ in rate-map thresholding, field geometry, fixed versus measured grid spacing,",
        "and the place-phase estimator. See `metrics/parameter_comparison.csv`.",
        "",
        "## Local transient availability",
        "",
        complete_counts.to_markdown(index=False),
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--methods-output", type=Path, default=DEFAULT_METHODS_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--networks", type=int, nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    metrics = output / "metrics"
    figures = output / "figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    availability = inventory(args.data_root.resolve())
    availability.to_csv(metrics / "transient_data_availability.csv", index=False)
    selected = select_complete_networks(args.data_root.resolve(), args.networks)
    print("selected_networks", [network_id for network_id, _ in selected], flush=True)

    rows: list[dict[str, object]] = []
    for network_id, network_root in selected:
        rows.extend(analyse_network(network_id, network_root))
    cells = pd.DataFrame(rows)
    trajectory, network, group = aggregate(cells)

    cells.to_csv(metrics / "cell_level_values.csv.gz", index=False, compression="gzip")
    trajectory.to_csv(metrics / "trajectory_level_values.csv", index=False)
    network.to_csv(metrics / "network_level_values.csv", index=False)
    group.to_csv(metrics / "group_summary_network_unit.csv", index=False)
    parameter_table().to_csv(metrics / "parameter_comparison.csv", index=False)

    comparison, comparison_group = compare_methods(network, args.methods_output.resolve())
    if not comparison.empty:
        comparison.to_csv(metrics / "network_level_methods_vs_mainrun.csv", index=False)
        comparison_group.to_csv(metrics / "group_change_methods_vs_mainrun.csv", index=False)

    make_figures(network, figures)
    write_readme(output, [network_id for network_id, _ in selected], availability)
    provenance = {
        "data_root": str(args.data_root.resolve()),
        "methods_output_preserved": str(args.methods_output.resolve()),
        "output": str(output),
        "networks": [network_id for network_id, _ in selected],
        "environments": [env.identifier for env in ENVS],
        "statistical_unit": "independent initialized network",
        "main_run_defaults": {
            "bin_size_cm": BIN_SIZE_CM,
            "grid_phase_boundary_cm": GRID_PHASE_BOUNDARY_CM,
            "grid_phase_bootstrap": GRID_PHASE_BOOTSTRAP,
            "grid_rescale_step_cm": GRID_RESCALE_STEP_CM,
            "place_field_peak_fraction": PLACE_FIELD_PEAK_FRACTION,
            "place_phase_min_speed_cm_s": PLACE_PHASE_MIN_SPEED_CM_S,
            "place_phase_bootstrap": PLACE_PHASE_BOOTSTRAP,
            "place_phase_dt_s": PLACE_PHASE_DT_S,
        },
    }
    (metrics / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"done: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
