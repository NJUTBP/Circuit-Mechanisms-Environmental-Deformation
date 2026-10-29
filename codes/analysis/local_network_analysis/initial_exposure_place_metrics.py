"""Compute initial-exposure place and phase metrics for the final analyses.

This analysis is deliberately separate from the production Methods profile.
It reads the retained transient activity, reconstructs the old 2-cm rate-map
workflow, and writes network-level source data/statistics for all seven chamber
lengths and all configured independently initialized networks.

Original-profile contract
-------------------------
* grid rescaling: ``Field_len_scale.cal_rescale_factor`` with 2-cm raw
  occupancy-normalized maps and ``shift_bin=4``;
* grid phase: 2-cm bins, 10-cm boundary zone, 50 bootstrap draws per axis,
  fixed nominal periods 36.9/54.1/76.8 cm;
* place field: 2-cm bins, 2-cm smoothing, 10% peak threshold, largest connected
  component span;
* place phase: 2-cm bins, central-difference velocity with dt=0.02 s,
  5 cm/s minimum speed, 45-degree direction ranges, 100 bootstrap draws.

Cells are averaged within trajectory, the two trajectories are averaged within
network, and all confidence intervals/tests use network means only.  Session
results are cached so interrupted runs can resume without recomputation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "local_network_analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import place_metric_helpers as legacy  # noqa: E402
from network_contrasts import (  # noqa: E402
    build_network_contrasts,
    summarize_contrasts,
)


DEFAULT_DATA_ROOT = (
    PROJECT_ROOT.parent
    / "run"
    / "outputs"
    / "independent_networks"
    / "networks"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT.parent
    / "run"
    / "outputs"
    / "independent_networks"
    / "config.resolved.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "initial_exposure_place_metrics"
)

TRAJECTORIES = (0, 1)
MODULES = (1, 2, 3)
ENVIRONMENTS = (
    legacy.Environment("compressed_70", 70, "initial_exposure"),
    legacy.Environment("compressed_80", 80, "initial_exposure"),
    legacy.Environment("compressed_90", 90, "initial_exposure"),
    legacy.Environment("fam_100", 100, "baseline"),
    legacy.Environment("elongated_110", 110, "initial_exposure"),
    legacy.Environment("elongated_120", 120, "initial_exposure"),
    legacy.Environment("elongated_130", 130, "initial_exposure"),
)

BIN_SIZE_CM = 2.0
BOUNDARY_THRESHOLD_CM = 10.0
GRID_BOOTSTRAP_X = 50
GRID_BOOTSTRAP_Y = 50
GRID_SMOOTH_SIGMA_BINS = 1.5
GRID_MIN_OVERLAP_BINS = 20
NOMINAL_GRID_SCALES_CM = {1: 36.9, 2: 54.1, 3: 76.8}
RESCALE_SHIFT_BIN = 4
PLACE_FIELD_SMOOTH_CM = 2.0
PLACE_FIELD_THRESHOLD = 0.1
PLACE_PHASE_MIN_SPEED_CM_S = 5.0
PLACE_PHASE_BOOTSTRAP = 100
PLACE_PHASE_DT_S = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transient_path(network_root: Path, env: legacy.Environment, trajectory: int) -> Path:
    return (
        network_root
        / env.relative_dir
        / f"trajectory_{trajectory:02d}"
        / "transient_activity.npz"
    )


def cache_path(
    cache_root: Path,
    network_id: int,
    env: legacy.Environment,
    trajectory: int,
) -> Path:
    return (
        cache_root
        / f"network_{network_id:03d}"
        / env.identifier
        / f"trajectory_{trajectory:02d}.npz"
    )


def discover_networks(data_root: Path, requested: list[int]) -> list[tuple[int, Path]]:
    selected: list[tuple[int, Path]] = []
    for network_id in requested:
        root = data_root / f"network_{network_id:03d}"
        if not root.is_dir():
            raise FileNotFoundError(root)
        missing = [
            transient_path(root, env, trajectory)
            for env in ENVIRONMENTS
            for trajectory in TRAJECTORIES
            if not transient_path(root, env, trajectory).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Network {network_id:03d} lacks {len(missing)} transient files; first: {missing[0]}"
            )
        selected.append((network_id, root))
    return selected


def _configure_legacy_globals() -> None:
    legacy.BIN_SIZE_CM = BIN_SIZE_CM
    legacy.GRID_PHASE_BOUNDARY_CM = BOUNDARY_THRESHOLD_CM
    legacy.GRID_PHASE_BOOTSTRAP = GRID_BOOTSTRAP_X
    legacy.GRID_PHASE_SMOOTH_SIGMA_BINS = GRID_SMOOTH_SIGMA_BINS
    legacy.GRID_PHASE_MIN_OVERLAP_BINS = GRID_MIN_OVERLAP_BINS
    legacy.NOMINAL_GRID_SCALES_CM = NOMINAL_GRID_SCALES_CM
    legacy.PLACE_FIELD_SMOOTH_CM = PLACE_FIELD_SMOOTH_CM
    legacy.PLACE_FIELD_PEAK_FRACTION = PLACE_FIELD_THRESHOLD
    legacy.PLACE_PHASE_MIN_SPEED_CM_S = PLACE_PHASE_MIN_SPEED_CM_S
    legacy.PLACE_PHASE_BOOTSTRAP = PLACE_PHASE_BOOTSTRAP
    legacy.PLACE_PHASE_DT_S = PLACE_PHASE_DT_S


def _session_task(payload: dict[str, object]) -> str:
    """Compute/cache all original-profile metrics for one run."""
    _configure_legacy_globals()
    network_id = int(payload["network_id"])
    environment = legacy.Environment(
        str(payload["environment"]),
        int(payload["length_cm"]),
        str(payload["stage"]),
    )
    trajectory = int(payload["trajectory"])
    source = Path(str(payload["source"]))
    destination = Path(str(payload["destination"]))

    positions_cm, activity, stored_dt_s = legacy.load_transient(source)
    flat_bins, nx, ny = legacy.spatial_bins(positions_cm, environment.length_cm)
    arrays: dict[str, np.ndarray | float | int] = {
        "network_id": network_id,
        "trajectory_id": trajectory,
        "length_cm": environment.length_cm,
        "stored_record_dt_s": stored_dt_s,
    }

    for module in MODULES:
        values = activity[f"grid_m{module}"]
        raw_maps, _ = legacy.binned_means(values, flat_bins, nx, ny)
        arrays[f"grid_raw_maps_m{module}"] = raw_maps.astype(np.float32)
        for axis, draws in (("x", GRID_BOOTSTRAP_X), ("y", GRID_BOOTSTRAP_Y)):
            shift, ratio = legacy.legacy_grid_phase(
                positions_cm,
                values,
                environment.length_cm,
                module,
                axis,
                flat_bins,
                nx,
                ny,
                n_bootstrap=draws,
                smooth_sigma=GRID_SMOOTH_SIGMA_BINS,
                min_overlap_bins=GRID_MIN_OVERLAP_BINS,
            )
            arrays[f"grid_phase_shift_{axis}_m{module}"] = shift
            arrays[f"grid_phase_ratio_{axis}_m{module}"] = ratio

    place_maps = legacy.legacy_place_rate_maps(activity["place"], flat_bins, nx, ny)
    length_x, length_y = legacy.FIELD_LENGTH.compute_place_field_length_batch(
        list(place_maps), bin_size=BIN_SIZE_CM, threshold_ratio=PLACE_FIELD_THRESHOLD
    )
    arrays["place_field_length_x_cm"] = length_x
    arrays["place_field_length_y_cm"] = length_y
    arrays["place_phase_x_cm"] = legacy.legacy_place_phase(
        positions_cm, activity["place"], environment.length_cm, "x", flat_bins, nx, ny
    )
    arrays["place_phase_y_cm"] = legacy.legacy_place_phase(
        positions_cm, activity["place"], environment.length_cm, "y", flat_bins, nx, ny
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    return str(destination)


def build_session_tasks(
    networks: list[tuple[int, Path]], cache_root: Path
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for network_id, network_root in networks:
        for env in ENVIRONMENTS:
            for trajectory in TRAJECTORIES:
                destination = cache_path(cache_root, network_id, env, trajectory)
                if destination.is_file():
                    continue
                tasks.append(
                    {
                        "network_id": network_id,
                        "environment": env.identifier,
                        "length_cm": env.length_cm,
                        "stage": env.stage,
                        "trajectory": trajectory,
                        "source": str(transient_path(network_root, env, trajectory)),
                        "destination": str(destination),
                    }
                )
    return tasks


def run_session_tasks(tasks: list[dict[str, object]], workers: int) -> None:
    if not tasks:
        print("All original-profile session caches are already present", flush=True)
        return
    print(f"Computing {len(tasks)} session caches with {workers} workers", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_session_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"session {completed}/{len(tasks)}: {result}", flush=True)


def _load_cache(path: Path) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _exact_rescale(
    base_maps: np.ndarray,
    current_maps: np.ndarray,
    rescale_module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(
        [
            rescale_module.cal_rescale_factor(base_maps[cell], current_maps[cell], RESCALE_SHIFT_BIN)
            for cell in range(base_maps.shape[0])
        ],
        dtype=float,
    )
    return rows[:, 0], rows[:, 1], rows[:, 2]


def assemble_cells(
    networks: list[tuple[int, Path]], cache_root: Path
) -> pd.DataFrame:
    rescale_module = _load_module(
        "legacy_fig2_rescale_assembly",
        PROJECT_ROOT / "main_run_prev" / "analysis" / "Field_len_scale.py",
    )
    rows: list[dict[str, object]] = []
    baseline_env = next(env for env in ENVIRONMENTS if env.length_cm == 100)
    for network_id, _ in networks:
        for trajectory in TRAJECTORIES:
            baseline = _load_cache(
                cache_path(cache_root, network_id, baseline_env, trajectory)
            )
            for env in ENVIRONMENTS:
                current = _load_cache(cache_path(cache_root, network_id, env, trajectory))
                for module in MODULES:
                    if env.length_cm == 100:
                        factor = np.ones(100, dtype=float)
                        correlation = np.ones(100, dtype=float)
                        alignment = np.zeros(100, dtype=float)
                    else:
                        factor, correlation, alignment = _exact_rescale(
                            np.asarray(baseline[f"grid_raw_maps_m{module}"], dtype=float),
                            np.asarray(current[f"grid_raw_maps_m{module}"], dtype=float),
                            rescale_module,
                        )
                    phase_x = np.asarray(current[f"grid_phase_shift_x_m{module}"], float)
                    phase_y = np.asarray(current[f"grid_phase_shift_y_m{module}"], float)
                    ratio_x = np.asarray(current[f"grid_phase_ratio_x_m{module}"], float)
                    ratio_y = np.asarray(current[f"grid_phase_ratio_y_m{module}"], float)
                    base_phase_x = np.asarray(baseline[f"grid_phase_shift_x_m{module}"], float)
                    base_phase_y = np.asarray(baseline[f"grid_phase_shift_y_m{module}"], float)
                    base_ratio_x = np.asarray(baseline[f"grid_phase_ratio_x_m{module}"], float)
                    base_ratio_y = np.asarray(baseline[f"grid_phase_ratio_y_m{module}"], float)
                    for cell in range(100):
                        rows.append(
                            {
                                "population": "grid",
                                "module": module,
                                "cell_id": cell,
                                "network_id": network_id,
                                "trajectory_id": trajectory,
                                "environment": env.identifier,
                                "length_cm": env.length_cm,
                                "deformation_ratio": env.ratio,
                                "rescale_factor": factor[cell],
                                "rescale_best_correlation": correlation[cell],
                                "rescale_alignment": alignment[cell],
                                "phase_shift_x_cm": phase_x[cell],
                                "phase_shift_y_cm": phase_y[cell],
                                "phase_ratio_x_half_period": ratio_x[cell],
                                "phase_ratio_y_half_period": ratio_y[cell],
                                "delta_phase_shift_x_cm": phase_x[cell] - base_phase_x[cell],
                                "delta_phase_shift_y_cm": phase_y[cell] - base_phase_y[cell],
                                "delta_phase_ratio_x_half_period": ratio_x[cell] - base_ratio_x[cell],
                                "delta_phase_ratio_y_half_period": ratio_y[cell] - base_ratio_y[cell],
                            }
                        )

                current_length_x = np.asarray(current["place_field_length_x_cm"], float)
                current_length_y = np.asarray(current["place_field_length_y_cm"], float)
                current_phase_x = np.asarray(current["place_phase_x_cm"], float)
                current_phase_y = np.asarray(current["place_phase_y_cm"], float)
                base_length_x = np.asarray(baseline["place_field_length_x_cm"], float)
                base_length_y = np.asarray(baseline["place_field_length_y_cm"], float)
                base_place_x = np.asarray(baseline["place_phase_x_cm"], float)
                base_place_y = np.asarray(baseline["place_phase_y_cm"], float)
                for cell in range(100):
                    rows.append(
                        {
                            "population": "place",
                            "module": 0,
                            "cell_id": cell,
                            "network_id": network_id,
                            "trajectory_id": trajectory,
                            "environment": env.identifier,
                            "length_cm": env.length_cm,
                            "deformation_ratio": env.ratio,
                            "field_length_x_cm": current_length_x[cell],
                            "field_length_y_cm": current_length_y[cell],
                            "place_phase_x_cm": current_phase_x[cell],
                            "place_phase_y_cm": current_phase_y[cell],
                            "delta_field_length_x_cm": current_length_x[cell] - base_length_x[cell],
                            "delta_field_length_y_cm": current_length_y[cell] - base_length_y[cell],
                            "delta_place_phase_x_cm": current_phase_x[cell] - base_place_x[cell],
                            "delta_place_phase_y_cm": current_phase_y[cell] - base_place_y[cell],
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["network_id", "length_cm", "trajectory_id", "population", "module", "cell_id"]
    ).reset_index(drop=True)


GRID_METRICS = (
    "rescale_factor",
    "rescale_best_correlation",
    "phase_shift_x_cm",
    "phase_shift_y_cm",
    "phase_ratio_x_half_period",
    "phase_ratio_y_half_period",
    "delta_phase_shift_x_cm",
    "delta_phase_shift_y_cm",
    "delta_phase_ratio_x_half_period",
    "delta_phase_ratio_y_half_period",
)
PLACE_METRICS = (
    "field_length_x_cm",
    "field_length_y_cm",
    "place_phase_x_cm",
    "place_phase_y_cm",
    "delta_field_length_x_cm",
    "delta_field_length_y_cm",
    "delta_place_phase_x_cm",
    "delta_place_phase_y_cm",
)


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate cells -> trajectories -> networks -> group summaries."""
    trajectory_rows: list[dict[str, object]] = []
    keys = [
        "population",
        "network_id",
        "trajectory_id",
        "environment",
        "length_cm",
        "deformation_ratio",
        "module",
    ]
    for population, metrics in (("grid", GRID_METRICS), ("place", PLACE_METRICS)):
        subset = cells.loc[cells["population"].eq(population)]
        for key, group in subset.groupby(keys, sort=True):
            for metric in metrics:
                values = group[metric].to_numpy(float)
                finite = values[np.isfinite(values)]
                trajectory_rows.append(
                    dict(zip(keys, key))
                    | {
                        "metric": metric,
                        "trajectory_mean": float(np.mean(finite)) if finite.size else np.nan,
                        "trajectory_median": float(np.median(finite)) if finite.size else np.nan,
                        "sd_across_cells": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
                        "n_cells_finite": int(finite.size),
                    }
                )
    trajectory = pd.DataFrame(trajectory_rows)

    network_keys = [
        "population",
        "network_id",
        "environment",
        "length_cm",
        "deformation_ratio",
        "module",
        "metric",
    ]
    network_rows: list[dict[str, object]] = []
    for key, group in trajectory.groupby(network_keys, sort=True):
        values = group["trajectory_mean"].to_numpy(float)
        finite = values[np.isfinite(values)]
        network_rows.append(
            dict(zip(network_keys, key))
            | {
                "network_mean": float(np.mean(finite)) if finite.size else np.nan,
                "sd_across_trajectories": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
                "n_trajectories_finite": int(finite.size),
            }
        )
    network = pd.DataFrame(network_rows)

    group_keys = [
        "population",
        "environment",
        "length_cm",
        "deformation_ratio",
        "module",
        "metric",
    ]
    group_rows: list[dict[str, object]] = []
    from scipy import stats

    for key, subset in network.groupby(group_keys, sort=True):
        values = subset["network_mean"].to_numpy(float)
        finite = values[np.isfinite(values)]
        n = int(finite.size)
        mean = float(np.mean(finite)) if n else np.nan
        sd = float(np.std(finite, ddof=1)) if n > 1 else np.nan
        sem = sd / math.sqrt(n) if n > 1 else np.nan
        half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
        group_rows.append(
            dict(zip(group_keys, key))
            | {
                "group_mean": mean,
                "sd_across_networks": sd,
                "sem_across_networks": sem,
                "ci95_low": mean - half if np.isfinite(half) else np.nan,
                "ci95_high": mean + half if np.isfinite(half) else np.nan,
                "median_across_networks": float(np.median(finite)) if n else np.nan,
                "q25_across_networks": float(np.quantile(finite, 0.25)) if n else np.nan,
                "q75_across_networks": float(np.quantile(finite, 0.75)) if n else np.nan,
                "n_networks": n,
            }
        )
    return trajectory, network, pd.DataFrame(group_rows)


def write_outputs(
    output: Path,
    config_path: Path,
    networks: list[tuple[int, Path]],
    cells: pd.DataFrame,
    trajectory: pd.DataFrame,
    network: pd.DataFrame,
    group: pd.DataFrame,
    contrasts: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    metrics = output / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    grid = cells.loc[cells["population"].eq("grid")]
    place = cells.loc[cells["population"].eq("place")]
    grid.to_csv(metrics / "grid_source_data_per_cell_trajectory.csv.gz", index=False, compression="gzip")
    place.to_csv(metrics / "place_source_data_per_cell_trajectory.csv.gz", index=False, compression="gzip")
    trajectory.to_csv(metrics / "trajectory_level_values.csv.gz", index=False, compression="gzip")
    network.to_csv(metrics / "network_level_values.csv", index=False)
    group.to_csv(metrics / "group_summary_network_unit.csv", index=False)
    contrasts.to_csv(metrics / "network_level_contrasts.csv", index=False)
    statistics.to_csv(metrics / "network_level_statistics.csv", index=False)
    statistics.loc[statistics["family"].str.startswith("fig3_")].to_csv(
        metrics / "fig3_network_level_statistics.csv", index=False
    )
    trajectory[
        [
            "population",
            "network_id",
            "environment",
            "length_cm",
            "trajectory_id",
            "module",
            "metric",
            "n_cells_finite",
        ]
    ].to_csv(metrics / "finite_cell_counts.csv", index=False)
    group[
        ["population", "environment", "length_cm", "module", "metric", "n_networks"]
    ].to_csv(metrics / "finite_network_counts.csv", index=False)

    parameters = pd.DataFrame(
        [
            ("profile", "original Fig. 2/3 code profile"),
            ("bin_size_cm", BIN_SIZE_CM),
            ("rescale_function", "Field_len_scale.cal_rescale_factor"),
            ("rescale_shift_bin", RESCALE_SHIFT_BIN),
            ("grid_boundary_threshold_cm", BOUNDARY_THRESHOLD_CM),
            ("grid_phase_bootstrap_x", GRID_BOOTSTRAP_X),
            ("grid_phase_bootstrap_y", GRID_BOOTSTRAP_Y),
            ("grid_phase_periods_cm", "36.9/54.1/76.8"),
            ("place_field_smoothing_cm", PLACE_FIELD_SMOOTH_CM),
            ("place_field_peak_fraction", PLACE_FIELD_THRESHOLD),
            ("place_phase_min_speed_cm_s", PLACE_PHASE_MIN_SPEED_CM_S),
            ("place_phase_bootstrap", PLACE_PHASE_BOOTSTRAP),
            ("place_phase_velocity_dt_s", PLACE_PHASE_DT_S),
            ("independent_unit", "network"),
        ],
        columns=["parameter", "value"],
    )
    parameters.to_csv(metrics / "analysis_parameters.csv", index=False)

    network_ids = [network_id for network_id, _ in networks]
    code_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "local_network_analysis" / "place_metric_helpers.py",
        PROJECT_ROOT / "main_run_prev" / "analysis" / "Field_len_scale.py",
        PROJECT_ROOT / "main_run_prev" / "analysis" / "Field_length.py",
        PROJECT_ROOT / "main_run_prev" / "analysis" / "phase_shift_grid.py",
        PROJECT_ROOT / "main_run_prev" / "analysis" / "phase_shift_place.py",
        config_path,
    ]
    provenance = {
        "analysis_profile": "original Fig. 2/3 code profile",
        "data_root": str(DEFAULT_DATA_ROOT.resolve()),
        "config": str(config_path),
        "network_ids": network_ids,
        "n_networks": len(network_ids),
        "n_trajectories_per_condition": 2,
        "independent_unit": "initialized network",
        "aggregation_order": "finite cells within trajectory; trajectories within network; networks for inference",
        "phase_raw_definition": "2 * absolute mean signed-bootstrap lag in cm / fixed nominal module period",
        "phase_change_definition": "current raw normalized phase minus matched familiar raw normalized phase",
        "statistics": "two-sided one-sample t-tests of network differences; Wilcoxon sensitivity; BH within family",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in code_paths
            if path.is_file()
        },
    }
    (output / "analysis_qa_and_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# v4 initial-exposure original-code profile\n\n"
        f"Networks: {network_ids} (`n={len(network_ids)}`). All inferential statistics use networks, not cells.\n\n"
        "This directory is a parameter-sensitivity counterpart to the production Methods profile. "
        "The two profiles must remain separately labelled because their rate-map, phase-normalization, "
        "place-field, and place-phase definitions differ.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--network-ids", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    cache_root = output / "session_cache"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("n_networks", -1)) != 8:
        raise ValueError(f"Expected n_networks=8 in {config_path}")

    networks = discover_networks(data_root, args.network_ids)
    tasks = build_session_tasks(networks, cache_root)
    run_session_tasks(tasks, max(1, int(args.workers)))
    cells = assemble_cells(networks, cache_root)
    trajectory, network, group = aggregate(cells)
    contrasts = build_network_contrasts(network)
    statistics = summarize_contrasts(contrasts)
    write_outputs(
        output,
        config_path,
        networks,
        cells,
        trajectory,
        network,
        group,
        contrasts,
        statistics,
    )
    print(f"Original-profile analysis complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
