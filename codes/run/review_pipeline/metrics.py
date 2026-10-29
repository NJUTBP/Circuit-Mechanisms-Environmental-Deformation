from __future__ import annotations

import gzip
import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig
from .seeds import derive_seed
from .simulation import SimulationResult
from .storage import atomic_csv_gz, atomic_npz


CELL_FIELDS = [
    "network_id", "stage", "environment", "trajectory_id", "branch", "session",
    "population", "module", "cell_id", "mean_rate_hz", "peak_rate_hz",
    "spatial_information_bits_per_spike", "field_length_x_cm", "field_length_y_cm",
    "grid_spacing_cm", "grid_score", "phase_shift_x_cm", "phase_shift_y_cm",
    "phase_ratio_x", "phase_ratio_y", "movement_displacement_x_cm",
    "movement_displacement_y_cm",
]

COMPARISON_FIELDS = [
    "network_id", "stage", "environment", "trajectory_id", "branch", "session",
    "population", "module", "cell_id", "deformation_axis", "deformation_ratio",
    "rescale_factor", "rescale_best_correlation", "delta_field_length_cm",
    "delta_phase_shift_cm", "r_phase", "delta_movement_displacement_cm",
    "delta_peak_rate_hz", "peak_rate_ratio", "delta_mean_rate_hz", "mean_rate_ratio",
    "delta_spatial_information_bits_per_spike",
]

SUMMARY_FIELDS = [
    "network_id", "stage", "environment", "variant", "session", "population",
    "module", "metric", "mean", "sd_across_trajectories", "n_trajectories",
    "n_cells",
]


def _load_analysis_module(config: PipelineConfig, filename: str):
    directory = config.resolve_path(
        config.data["analysis"]["legacy_function_dir"], base=config.project_root
    )
    path = directory / filename
    module_name = f"review_pipeline_legacy_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import analysis module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and np.isfinite(denominator) else np.nan


@contextmanager
def _seed_legacy_bootstrap(module: Any, base_seed: int):
    """Map legacy draw indices (0..n-1) onto a task-specific seed stream."""
    original = module._compute_matched_rate_maps

    def seeded(*args: Any, **kwargs: Any):
        draw = kwargs.get("seed")
        kwargs["seed"] = derive_seed(base_seed, "draw", 0 if draw is None else int(draw))
        return original(*args, **kwargs)

    random_state = np.random.get_state()
    module._compute_matched_rate_maps = seeded
    try:
        yield
    finally:
        module._compute_matched_rate_maps = original
        np.random.set_state(random_state)


def build_rate_maps(
    positions_m: np.ndarray,
    activity: np.ndarray,
    width_m: float,
    height_m: float,
    bin_m: float,
    sigma_bins: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    n_x = int(round(width_m / bin_m))
    n_y = int(round(height_m / bin_m))
    x_bin = np.clip((positions_m[:, 0] / bin_m).astype(int), 0, n_x - 1)
    y_bin = np.clip((positions_m[:, 1] / bin_m).astype(int), 0, n_y - 1)
    flat_bin = y_bin * n_x + x_bin
    occupancy = np.bincount(flat_bin, minlength=n_x * n_y).reshape(n_y, n_x)
    maps = np.empty((activity.shape[0], n_y, n_x), dtype=np.float32)
    visited = occupancy > 0
    for cell in range(activity.shape[0]):
        weighted = np.bincount(
            flat_bin,
            weights=np.asarray(activity[cell], dtype=float),
            minlength=n_x * n_y,
        ).reshape(n_y, n_x)
        raw_map = np.full((n_y, n_x), np.nan, dtype=float)
        np.divide(
            weighted,
            occupancy,
            out=raw_map,
            where=visited,
        )
        # Supplementary Note 1: first calculate the mean within each spatial
        # bin, then smooth that rate map. This normalized-mask convolution
        # prevents unvisited bins from being treated as zero firing.
        numerator = gaussian_filter(
            np.where(visited, raw_map, 0.0), sigma_bins, mode="constant"
        )
        denominator = gaussian_filter(
            visited.astype(float), sigma_bins, mode="constant"
        )
        rate_map = np.full((n_y, n_x), np.nan, dtype=float)
        np.divide(numerator, denominator, out=rate_map, where=denominator > 1e-12)
        rate_map[~visited] = np.nan
        maps[cell] = rate_map
    return maps, occupancy.astype(np.int32)


def spatial_information(
    rate_map: np.ndarray,
    occupancy: np.ndarray,
    min_occupancy_fraction: float = 1e-4,
) -> float:
    total_occupancy = float(np.sum(occupancy))
    if total_occupancy <= 0:
        return np.nan
    occupancy_fraction = occupancy.astype(float) / total_occupancy
    visited = (
        (occupancy_fraction >= float(min_occupancy_fraction))
        & np.isfinite(rate_map)
    )
    if not np.any(visited):
        return np.nan
    probability = occupancy[visited].astype(float)
    probability /= probability.sum()
    rates = rate_map[visited].astype(float)
    mean_rate = float(np.sum(probability * rates))
    if mean_rate <= 0:
        return np.nan
    positive = rates > 0
    ratio = rates[positive] / mean_rate
    return float(np.sum(probability[positive] * ratio * np.log2(ratio)))


def _cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = tuple(np.asarray(a.shape) + np.asarray(b.shape) - 1)
    axes = tuple(range(a.ndim))
    first = np.fft.rfftn(a, s=shape, axes=axes)
    second = np.fft.rfftn(b[::-1, ::-1], s=shape, axes=axes)
    return np.fft.irfftn(first * second, s=shape, axes=axes)


def _central_component(mask: np.ndarray, centre: tuple[int, int]) -> np.ndarray:
    component = np.zeros(mask.shape, dtype=bool)
    if not mask[centre]:
        return component
    stack = [centre]
    component[centre] = True
    height, width = mask.shape
    while stack:
        y, x = stack.pop()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and mask[next_y, next_x]
                and not component[next_y, next_x]
            ):
                component[next_y, next_x] = True
                stack.append((next_y, next_x))
    return component


def _pearson_autocorrelation(
    rate_map: np.ndarray, min_overlap_bins: int
) -> np.ndarray:
    valid = np.isfinite(rate_map)
    mask = valid.astype(float)
    values = np.where(valid, rate_map, 0.0)
    count = np.rint(_cross_correlation(mask, mask))
    sum_a = _cross_correlation(values, mask)
    sum_b = _cross_correlation(mask, values)
    sum_a2 = _cross_correlation(values * values, mask)
    sum_b2 = _cross_correlation(mask, values * values)
    sum_ab = _cross_correlation(values, values)
    safe_count = np.maximum(count, 1.0)
    covariance = sum_ab - (sum_a * sum_b / safe_count)
    variance_a = sum_a2 - (sum_a * sum_a / safe_count)
    variance_b = sum_b2 - (sum_b * sum_b / safe_count)
    denominator = np.sqrt(np.maximum(variance_a, 0.0) * np.maximum(variance_b, 0.0))
    correlation = np.full(count.shape, np.nan, dtype=float)
    usable = (count >= int(min_overlap_bins)) & (denominator > 0.0)
    np.divide(covariance, denominator, out=correlation, where=usable)
    return np.clip(correlation, -1.0, 1.0)


def grid_field_lengths(
    rate_maps: np.ndarray,
    bin_cm: float,
    threshold: float,
    min_overlap_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    lengths_x = np.full(len(rate_maps), np.nan)
    lengths_y = np.full(len(rate_maps), np.nan)
    for cell, rate_map in enumerate(rate_maps):
        autocorrelation = _pearson_autocorrelation(rate_map, min_overlap_bins)
        centre_y, centre_x = np.array(autocorrelation.shape) // 2
        component = _central_component(
            np.isfinite(autocorrelation) & (autocorrelation > threshold),
            (int(centre_y), int(centre_x)),
        )
        if not np.any(component):
            continue
        yy, xx = np.where(component)
        lengths_x[cell] = (int(xx.max()) - int(xx.min()) + 1) * bin_cm
        lengths_y[cell] = (int(yy.max()) - int(yy.min()) + 1) * bin_cm
    return lengths_x, lengths_y


def _longest_true_run(values: np.ndarray) -> int:
    padded = np.concatenate(([False], np.asarray(values, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(edges[1::2] - edges[::2])) if edges.size else 0


def place_field_lengths(
    rate_maps: np.ndarray,
    bin_cm: float,
    min_rate_hz: float,
    peak_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    lengths_x = np.full(len(rate_maps), np.nan)
    lengths_y = np.full(len(rate_maps), np.nan)
    for cell, rate_map in enumerate(rate_maps):
        if not np.any(np.isfinite(rate_map)):
            continue
        peak = float(np.nanmax(rate_map))
        if peak <= min_rate_hz:
            continue
        retained = np.isfinite(rate_map) & (
            rate_map >= max(min_rate_hz, peak_fraction * peak)
        )
        run_x = max((_longest_true_run(row) for row in retained), default=0)
        run_y = max((_longest_true_run(column) for column in retained.T), default=0)
        if run_x:
            lengths_x[cell] = run_x * bin_cm
        if run_y:
            lengths_y[cell] = run_y * bin_cm
    return lengths_x, lengths_y


def _axis_peak_lag_cm(
    map_a: np.ndarray,
    map_b: np.ndarray,
    axis: int,
    max_lag: int,
    bin_cm: float,
    min_overlap_bins: int,
) -> float:
    correlations = np.full(2 * max_lag + 1, np.nan)
    lags = np.arange(-max_lag, max_lag + 1)
    for index, lag in enumerate(lags):
        if axis == 1:
            a = map_a[:, : map_a.shape[1] - lag] if lag > 0 else (
                map_a[:, -lag:] if lag < 0 else map_a
            )
            b = map_b[:, lag:] if lag > 0 else (
                map_b[:, : map_b.shape[1] + lag] if lag < 0 else map_b
            )
        else:
            a = map_a[: map_a.shape[0] - lag, :] if lag > 0 else (
                map_a[-lag:, :] if lag < 0 else map_a
            )
            b = map_b[lag:, :] if lag > 0 else (
                map_b[: map_b.shape[0] + lag, :] if lag < 0 else map_b
            )
        valid = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(valid)) < min_overlap_bins:
            continue
        first = a[valid]
        second = b[valid]
        if np.std(first) > 0.0 and np.std(second) > 0.0:
            correlations[index] = np.corrcoef(first, second)[0, 1]
    if not np.any(np.isfinite(correlations)):
        return np.nan
    return abs(float(lags[int(np.nanargmax(correlations))])) * bin_cm


def movement_displacement_batch(
    positions_m: np.ndarray,
    activity: np.ndarray,
    width_m: float,
    height_m: float,
    axis: str,
    bin_m: float,
    sigma_bins: float,
    min_speed_cm_s: float,
    record_dt_s: float,
    min_overlap_bins: int,
) -> np.ndarray:
    velocity = np.zeros_like(positions_m, dtype=float)
    velocity[1:] = np.diff(positions_m.astype(float), axis=0) / record_dt_s
    speed = np.linalg.norm(velocity, axis=1)
    sufficiently_fast = speed >= (min_speed_cm_s / 100.0)
    if axis == "x":
        cardinal = np.abs(velocity[:, 0]) >= np.abs(velocity[:, 1])
        first_mask = sufficiently_fast & cardinal & (velocity[:, 0] > 0.0)
        second_mask = sufficiently_fast & cardinal & (velocity[:, 0] < 0.0)
        correlation_axis = 1
    else:
        cardinal = np.abs(velocity[:, 1]) > np.abs(velocity[:, 0])
        first_mask = sufficiently_fast & cardinal & (velocity[:, 1] > 0.0)
        second_mask = sufficiently_fast & cardinal & (velocity[:, 1] < 0.0)
        correlation_axis = 0
    if not np.any(first_mask) or not np.any(second_mask):
        return np.full(activity.shape[0], np.nan)
    first_maps, _ = build_rate_maps(
        positions_m[first_mask], activity[:, first_mask], width_m, height_m,
        bin_m, sigma_bins,
    )
    second_maps, _ = build_rate_maps(
        positions_m[second_mask], activity[:, second_mask], width_m, height_m,
        bin_m, sigma_bins,
    )
    max_lag = min(first_maps.shape[1:]) // 2
    bin_cm = bin_m * 100.0
    return np.asarray([
        _axis_peak_lag_cm(
            first_maps[cell], second_maps[cell], correlation_axis, max_lag,
            bin_cm, min_overlap_bins,
        )
        for cell in range(activity.shape[0])
    ])


def grid_spacing_reference(cells_path: Path) -> dict[int, np.ndarray]:
    rows = read_csv_gz(cells_path)
    reference: dict[int, np.ndarray] = {}
    for module in (1, 2, 3):
        selected = [
            row for row in rows
            if row["population"] == "grid" and int(row["module"]) == module
        ]
        selected.sort(key=lambda row: int(row["cell_id"]))
        reference[module] = np.asarray([
            _float(row, "grid_spacing_cm") for row in selected
        ])
    return reference


def _grid_scale_and_score(rate_maps: np.ndarray, bin_cm: float) -> tuple[np.ndarray, np.ndarray]:
    import opexebo

    scales = np.full(len(rate_maps), np.nan)
    scores = np.full(len(rate_maps), np.nan)
    for index, rate_map in enumerate(rate_maps):
        try:
            autocorrelation = opexebo.analysis.autocorrelation(rate_map)
            score, statistics = opexebo.analysis.grid_score(autocorrelation)
            scores[index] = float(score)
            spacing = float(statistics["grid_spacing"])
            scales[index] = spacing * bin_cm
        except Exception:
            continue
    return scales, scores


def analyze_result(
    config: PipelineConfig,
    result: SimulationResult,
    env: dict[str, Any],
    output_dir: Path,
    identity: dict[str, Any],
    analysis_seed: int,
    phase_reference_cells_path: Path | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    analysis = config.data["analysis"]
    bin_m = float(analysis["rate_map_bin_m"])
    bin_cm = bin_m * 100.0
    sigma_bins = float(analysis["rate_map_sigma_bins"])
    width_m = float(env["width_m"])
    height_m = float(env["height_m"])
    populations = {
        "place": result.place_activity,
        "grid_m1": result.grid_activity[0],
        "grid_m2": result.grid_activity[1],
        "grid_m3": result.grid_activity[2],
    }
    maps: dict[str, np.ndarray] = {}
    occupancy: np.ndarray | None = None
    for name, activity in populations.items():
        maps[name], current_occupancy = build_rate_maps(
            result.positions,
            activity,
            width_m,
            height_m,
            bin_m,
            sigma_bins,
        )
        if occupancy is None:
            occupancy = current_occupancy
    assert occupancy is not None
    # Supplementary Note 3 uses a separate sigma=2-bin PC map specifically
    # for place-field length, whereas the general maps in Note 1 use 1.5 bins.
    place_field_maps, _ = build_rate_maps(
        result.positions,
        result.place_activity,
        width_m,
        height_m,
        bin_m,
        float(analysis["place_field_rate_map_sigma_bins"]),
    )

    phase_grid_module = _load_analysis_module(config, "phase_shift_grid.py")
    positions_cm = result.positions.astype(float) * 100.0
    env_size_cm = (width_m * 100.0, height_m * 100.0)
    grid_nominal_scales = [float(value) for value in analysis["grid_nominal_scales_cm"]]
    phase_reference = (
        grid_spacing_reference(phase_reference_cells_path)
        if phase_reference_cells_path is not None
        else None
    )
    min_overlap_bins = int(analysis.get("correlation_min_overlap_bins", 20))
    rows: list[dict[str, Any]] = []

    for population, activity in populations.items():
        rate_maps = maps[population]
        if population == "place":
            length_x, length_y = place_field_lengths(
                place_field_maps,
                bin_cm,
                float(analysis["place_min_rate_hz"]),
                float(analysis["place_field_peak_fraction"]),
            )
            grid_spacing = np.full(len(rate_maps), np.nan)
            grid_score = np.full(len(rate_maps), np.nan)
            phase_x = phase_y = np.full(len(rate_maps), np.nan)
            movement_x = movement_displacement_batch(
                result.positions, activity, width_m, height_m, "x", bin_m,
                sigma_bins, float(analysis["movement_min_speed_cm_s"]),
                result.record_dt_s, min_overlap_bins,
            )
            movement_y = movement_displacement_batch(
                result.positions, activity, width_m, height_m, "y", bin_m,
                sigma_bins, float(analysis["movement_min_speed_cm_s"]),
                result.record_dt_s, min_overlap_bins,
            )
            module_index: int | str = ""
        else:
            module_index = int(population[-1])
            length_x, length_y = grid_field_lengths(
                rate_maps,
                bin_cm,
                float(analysis["grid_field_autocorr_threshold"]),
                min_overlap_bins,
            )
            if bool(analysis.get("measure_grid_spacing", True)):
                grid_spacing, grid_score = _grid_scale_and_score(rate_maps, bin_cm)
            else:
                grid_spacing = np.full(len(rate_maps), np.nan)
                grid_score = np.full(len(rate_maps), np.nan)
            nominal = grid_nominal_scales[module_index - 1]
            phase_scales = (
                phase_reference[module_index].copy()
                if phase_reference is not None
                else grid_spacing.copy()
            )
            if bool(analysis.get("phase_spacing_fallback_to_nominal", False)):
                phase_scales[~np.isfinite(phase_scales)] = nominal
            with _seed_legacy_bootstrap(
                phase_grid_module,
                derive_seed(analysis_seed, "grid", module_index, "x"),
            ):
                phase_x_ratio = phase_grid_module.compute_phase_shift_batch(
                    positions_cm,
                    activity,
                    env_size_cm,
                    phase_scales,
                    axis="x",
                    bin_size=bin_cm,
                    boundary_threshold=float(analysis["boundary_threshold_cm"]),
                    n_bootstrap=int(analysis["phase_bootstrap"]["grid_x"]),
                )
            with _seed_legacy_bootstrap(
                phase_grid_module,
                derive_seed(analysis_seed, "grid", module_index, "y"),
            ):
                phase_y_ratio = phase_grid_module.compute_phase_shift_batch(
                    positions_cm,
                    activity,
                    env_size_cm,
                    phase_scales,
                    axis="y",
                    bin_size=bin_cm,
                    boundary_threshold=float(analysis["boundary_threshold_cm"]),
                    n_bootstrap=int(analysis["phase_bootstrap"]["grid_y"]),
                )
            phase_x = phase_x_ratio * phase_scales
            phase_y = phase_y_ratio * phase_scales
            movement_x = movement_y = np.full(len(rate_maps), np.nan)

        for cell in range(len(rate_maps)):
            peak = float(np.nanmax(rate_maps[cell])) if np.any(np.isfinite(rate_maps[cell])) else np.nan
            row = {
                **identity,
                "population": "place" if population == "place" else "grid",
                "module": module_index,
                "cell_id": cell,
                "mean_rate_hz": float(np.mean(activity[cell])),
                "peak_rate_hz": peak,
                "spatial_information_bits_per_spike": spatial_information(
                    rate_maps[cell],
                    occupancy,
                    float(analysis.get(
                        "spatial_information_min_occupancy_fraction", 1e-4
                    )),
                ),
                "field_length_x_cm": float(length_x[cell]),
                "field_length_y_cm": float(length_y[cell]),
                "grid_spacing_cm": float(grid_spacing[cell]),
                "grid_score": float(grid_score[cell]),
                "phase_shift_x_cm": float(phase_x[cell]),
                "phase_shift_y_cm": float(phase_y[cell]),
                "phase_ratio_x": _safe_ratio(float(phase_x[cell]), float(phase_scales[cell]))
                if population != "place" else np.nan,
                "phase_ratio_y": _safe_ratio(float(phase_y[cell]), float(phase_scales[cell]))
                if population != "place" else np.nan,
                "movement_displacement_x_cm": float(movement_x[cell]),
                "movement_displacement_y_cm": float(movement_y[cell]),
            }
            rows.append(row)

    maps_path = output_dir / "rate_maps.npz"
    cells_path = output_dir / "cell_metrics.csv.gz"
    atomic_npz(
        maps_path,
        compressed=bool(config.data["storage"].get("compress_rate_maps", True)),
        place=maps["place"],
        grid_m1=maps["grid_m1"],
        grid_m2=maps["grid_m2"],
        grid_m3=maps["grid_m3"],
        place_field=place_field_maps,
        occupancy=occupancy,
        bin_size_cm=np.array(bin_cm),
        width_cm=np.array(width_m * 100.0),
        height_cm=np.array(height_m * 100.0),
    )
    atomic_csv_gz(cells_path, rows, CELL_FIELDS)
    return maps_path, cells_path, rows


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    import csv

    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def compare_to_baseline(
    config: PipelineConfig,
    env: dict[str, Any],
    output_dir: Path,
    identity: dict[str, Any],
    current_rows: list[dict[str, Any]],
    baseline_cells_path: Path,
    baseline_maps_path: Path,
    current_maps_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    baseline_rows = read_csv_gz(baseline_cells_path)
    baseline_by_key = {
        (row["population"], row["module"], row["cell_id"]): row for row in baseline_rows
    }
    with np.load(baseline_maps_path, allow_pickle=False) as baseline_data:
        baseline_maps = {key: baseline_data[key] for key in ("grid_m1", "grid_m2", "grid_m3")}
        familiar_width = float(baseline_data["width_cm"])
        familiar_height = float(baseline_data["height_cm"])
    with np.load(current_maps_path, allow_pickle=False) as current_data:
        current_maps = {key: current_data[key] for key in ("grid_m1", "grid_m2", "grid_m3")}
    rescale_module = _load_analysis_module(config, "grid_rescale_cal.py")
    axis = str(env.get("deformation_axis", "x"))
    deformed_size = float(env["width_m"] if axis == "x" else env["height_m"]) * 100.0
    familiar_size = familiar_width if axis == "x" else familiar_height
    bin_cm = float(config.data["analysis"]["rate_map_bin_m"]) * 100.0
    rescale_by_module: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for module in (1, 2, 3):
        familiar = baseline_maps[f"grid_m{module}"]
        deformed = current_maps[f"grid_m{module}"]
        if axis == "y":
            familiar = familiar.transpose(0, 2, 1)
            deformed = deformed.transpose(0, 2, 1)
        rescale_by_module[module] = rescale_module.compute_grid_rescale_factor_batch(
            familiar,
            deformed,
            familiar_size,
            deformed_size,
            bin_size=bin_cm,
            scale_range=float(config.data["analysis"]["rescale_scale_range_cm"]),
            scale_step=float(config.data["analysis"]["rescale_scale_step_cm"]),
            mode=str(config.data["analysis"]["rescale_mode"]),
            min_overlap_ratio=float(config.data["analysis"]["rescale_min_overlap_ratio"]),
        )

    comparison: list[dict[str, Any]] = []
    field_key = f"field_length_{axis}_cm"
    phase_key = f"phase_shift_{axis}_cm"
    movement_key = f"movement_displacement_{axis}_cm"
    for current in current_rows:
        key = (str(current["population"]), str(current["module"]), str(current["cell_id"]))
        baseline = baseline_by_key[key]
        module = int(current["module"]) if current["population"] == "grid" else None
        cell = int(current["cell_id"])
        current_peak = float(current["peak_rate_hz"])
        current_mean = float(current["mean_rate_hz"])
        baseline_peak = _float(baseline, "peak_rate_hz")
        baseline_mean = _float(baseline, "mean_rate_hz")
        delta_phase = float(current[phase_key]) - _float(baseline, phase_key)
        baseline_spacing = _float(baseline, "grid_spacing_cm") if module else np.nan
        row = {
            **identity,
            "population": current["population"],
            "module": "" if module is None else module,
            "cell_id": cell,
            "deformation_axis": axis,
            "deformation_ratio": deformed_size / familiar_size,
            "rescale_factor": rescale_by_module[module][0][cell] if module else np.nan,
            "rescale_best_correlation": rescale_by_module[module][1][cell] if module else np.nan,
            "delta_field_length_cm": float(current[field_key]) - _float(baseline, field_key),
            "delta_phase_shift_cm": delta_phase if module else np.nan,
            "r_phase": (
                _safe_ratio(2.0 * delta_phase, baseline_spacing)
                if module else np.nan
            ),
            "delta_movement_displacement_cm": (
                float(current[movement_key]) - _float(baseline, movement_key)
                if module is None else np.nan
            ),
            "delta_peak_rate_hz": current_peak - baseline_peak,
            "peak_rate_ratio": _safe_ratio(current_peak, baseline_peak),
            "delta_mean_rate_hz": current_mean - baseline_mean,
            "mean_rate_ratio": _safe_ratio(current_mean, baseline_mean),
            "delta_spatial_information_bits_per_spike": (
                float(current["spatial_information_bits_per_spike"])
                - _float(baseline, "spatial_information_bits_per_spike")
            ),
        }
        comparison.append(row)
    path = output_dir / "comparison_metrics.csv.gz"
    atomic_csv_gz(path, comparison, COMPARISON_FIELDS)
    return path, comparison


def summarize_comparisons(
    rows: Iterable[dict[str, Any]],
    *,
    network_id: int,
    stage: str,
    environment: str,
    variant: str = "",
    session: int | str = "",
) -> list[dict[str, Any]]:
    numeric_metrics = [
        "rescale_factor", "rescale_best_correlation", "delta_field_length_cm",
        "delta_phase_shift_cm", "r_phase", "delta_movement_displacement_cm",
        "delta_peak_rate_hz", "peak_rate_ratio", "delta_mean_rate_hz",
        "mean_rate_ratio",
        "delta_spatial_information_bits_per_spike",
    ]
    rows = list(rows)
    summaries: list[dict[str, Any]] = []
    groups = sorted({(str(row["population"]), str(row.get("module", ""))) for row in rows})
    for population, module in groups:
        group = [row for row in rows if str(row["population"]) == population and str(row.get("module", "")) == module]
        trajectory_ids = sorted({str(row["trajectory_id"]) for row in group})
        n_cells = len({str(row["cell_id"]) for row in group})
        for metric in numeric_metrics:
            trajectory_means = []
            for trajectory_id in trajectory_ids:
                values = [
                    float(row[metric])
                    for row in group
                    if str(row["trajectory_id"]) == trajectory_id
                    and row.get(metric) not in (None, "")
                    and np.isfinite(float(row[metric]))
                ]
                if values:
                    trajectory_means.append(float(np.mean(values)))
            if not trajectory_means:
                continue
            summaries.append({
                "network_id": network_id,
                "stage": stage,
                "environment": environment,
                "variant": variant,
                "session": session,
                "population": population,
                "module": module,
                "metric": metric,
                "mean": float(np.mean(trajectory_means)),
                "sd_across_trajectories": float(np.std(trajectory_means, ddof=1))
                if len(trajectory_means) > 1 else np.nan,
                "n_trajectories": len(trajectory_means),
                "n_cells": n_cells,
            })
    return summaries


def summarize_cells(
    rows: Iterable[dict[str, Any]],
    *,
    network_id: int,
    stage: str,
    environment: str,
    variant: str = "",
    session: int | str = "",
) -> list[dict[str, Any]]:
    numeric_metrics = [
        "mean_rate_hz", "peak_rate_hz", "spatial_information_bits_per_spike",
        "field_length_x_cm", "field_length_y_cm", "grid_spacing_cm", "grid_score",
        "phase_shift_x_cm", "phase_shift_y_cm", "phase_ratio_x", "phase_ratio_y",
        "movement_displacement_x_cm", "movement_displacement_y_cm",
    ]
    rows = list(rows)
    summaries: list[dict[str, Any]] = []
    groups = sorted({(str(row["population"]), str(row.get("module", ""))) for row in rows})
    for population, module in groups:
        group = [row for row in rows if str(row["population"]) == population and str(row.get("module", "")) == module]
        trajectory_ids = sorted({str(row["trajectory_id"]) for row in group})
        n_cells = len({str(row["cell_id"]) for row in group})
        for metric in numeric_metrics:
            trajectory_means = []
            for trajectory_id in trajectory_ids:
                values = []
                for row in group:
                    if str(row["trajectory_id"]) != trajectory_id:
                        continue
                    try:
                        value = float(row[metric])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
                if values:
                    trajectory_means.append(float(np.mean(values)))
            if not trajectory_means:
                continue
            summaries.append({
                "network_id": network_id,
                "stage": stage,
                "environment": environment,
                "variant": variant,
                "session": session,
                "population": population,
                "module": module,
                "metric": metric,
                "mean": float(np.mean(trajectory_means)),
                "sd_across_trajectories": float(np.std(trajectory_means, ddof=1))
                if len(trajectory_means) > 1 else np.nan,
                "n_trajectories": len(trajectory_means),
                "n_cells": n_cells,
            })
    return summaries
