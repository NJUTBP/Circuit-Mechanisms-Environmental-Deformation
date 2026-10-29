"""Compute initial-exposure Figure 2 source tables with the adopted estimator.

Analysis only; manuscript figure layout is separate.
The place-field estimates come from the audited original-profile per-cell table.
For place phase, the new nondirectional change is |current signed phase| minus
|matched familiar signed phase|, evaluated per cell and trajectory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy import stats

import grid_rescale_primitives as core
from grid_rescaling import compute_recommended_rescale_batch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "run/outputs/independent_networks/networks"
PLACE_SOURCE = ROOT / "outputs/initial_exposure_place_metrics/metrics/place_source_data_per_cell_trajectory.csv.gz"
OUTPUT = ROOT / "outputs/initial_exposure_rescaling"
ENVIRONMENTS = {
    "compressed_70": 70,
    "compressed_80": 80,
    "compressed_90": 90,
    "fam_100": 100,
    "elongated_110": 110,
    "elongated_120": 120,
    "elongated_130": 130,
}
MODULES = (1, 2, 3)
TRAJECTORIES = (0, 1)
CACHE_VERSION = "initial_exposure_grid_five_edge_v1"
PLACE_METRICS = (
    "delta_field_length_x_cm",
    "delta_field_length_y_cm",
    "delta_abs_place_phase_x_cm",
    "delta_abs_place_phase_y_cm",
)


def session_path(data: Path, network_id: int, environment: str, trajectory_id: int) -> Path:
    parent = data / f"network_{network_id:03d}"
    stage = "baseline" if environment == "fam_100" else "initial_exposure"
    return parent / stage / environment / f"trajectory_{trajectory_id:02d}" / "transient_activity.npz"


def source_signature(paths: list[Path]) -> str:
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def load_selected_maps(path: Path, width_cm: float) -> dict[int, np.ndarray]:
    """Same 2.5-cm, sigma=1.5 normalized-mask maps as the Methods variant."""
    with np.load(path, allow_pickle=False) as data:
        positions_cm = np.asarray(data["positions"], np.float64) * 100.0
        flat, occupancy, _, _ = core._spatial_index(positions_cm, width_cm, 2.5)
        return {
            module: core._smooth_rate_maps(
                core._raw_rate_maps(np.asarray(data[f"grid_m{module}"], np.float64), flat, occupancy),
                1.5,
            )
            for module in MODULES
        }


def compute_network(data: Path, output: Path, network_id: int, edge_positions_per_side: int = 5) -> str:
    paths = [session_path(data, network_id, env, traj) for traj in TRAJECTORIES for env in ENVIRONMENTS]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    signature = source_signature(paths)
    cache = output / "cache" / f"network_{network_id:03d}.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["cache_version"]) == f"{CACHE_VERSION}_edge{edge_positions_per_side}" and str(saved["source_signature"]) == signature:
                    return str(cache)
        except (OSError, ValueError, KeyError):
            pass

    arrays: dict[str, np.ndarray] = {
        "cache_version": np.array(f"{CACHE_VERSION}_edge{edge_positions_per_side}"),
        "source_signature": np.array(signature),
    }
    for trajectory in TRAJECTORIES:
        familiar = load_selected_maps(session_path(data, network_id, "fam_100", trajectory), 100)
        for module in MODULES:
            count = familiar[module].shape[0]
            prefix = f"fam_100_t{trajectory:02d}_m{module}"
            arrays[f"factor_{prefix}"] = np.ones(count, float)
            arrays[f"corr_{prefix}"] = np.full(count, np.nan)
        for environment, width_cm in ENVIRONMENTS.items():
            if environment == "fam_100":
                continue
            current = load_selected_maps(session_path(data, network_id, environment, trajectory), width_cm)
            for module in MODULES:
                factor, correlation = compute_recommended_rescale_batch(
                    familiar[module], current[module], width_cm,
                    edge_positions_per_side=edge_positions_per_side, bin_cm=2.5,
                )
                prefix = f"{environment}_t{trajectory:02d}_m{module}"
                arrays[f"factor_{prefix}"] = factor
                arrays[f"corr_{prefix}"] = correlation
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(cache)


def grid_cells(output: Path, network_ids: list[int]) -> pd.DataFrame:
    rows = []
    for network_id in network_ids:
        with np.load(output / "cache" / f"network_{network_id:03d}.npz", allow_pickle=False) as saved:
            for trajectory_id in TRAJECTORIES:
                for environment, width_cm in ENVIRONMENTS.items():
                    for module in MODULES:
                        prefix = f"{environment}_t{trajectory_id:02d}_m{module}"
                        factors = np.asarray(saved[f"factor_{prefix}"], float)
                        correlations = np.asarray(saved[f"corr_{prefix}"], float)
                        for cell_id, (factor, correlation) in enumerate(zip(factors, correlations)):
                            rows.append({
                                "population": "grid", "module": module, "cell_id": cell_id,
                                "network_id": network_id, "trajectory_id": trajectory_id,
                                "environment": environment, "length_cm": width_cm,
                                "deformation_ratio": width_cm / 100.0,
                                "rescale_factor": factor, "rescale_best_correlation": correlation,
                            })
    return pd.DataFrame(rows)


def place_cells(source: Path, network_ids: list[int]) -> pd.DataFrame:
    data = pd.read_csv(source)
    data = data.loc[data.network_id.isin(network_ids) & data.population.eq("place")].copy()
    keys = ["network_id", "trajectory_id", "cell_id"]
    baseline = data.loc[data.environment.eq("fam_100"), keys + ["place_phase_x_cm", "place_phase_y_cm"]]
    if baseline.duplicated(keys).any():
        raise ValueError("Matched place baseline is not unique")
    baseline = baseline.rename(columns={"place_phase_x_cm": "baseline_phase_x_cm", "place_phase_y_cm": "baseline_phase_y_cm"})
    data = data.merge(baseline, on=keys, how="left", validate="many_to_one")
    if data[["baseline_phase_x_cm", "baseline_phase_y_cm"]].isna().any().any():
        raise ValueError("A place cell has no matched baseline phase")
    for axis in ("x", "y"):
        current = data[f"place_phase_{axis}_cm"].to_numpy(float)
        base = data[f"baseline_phase_{axis}_cm"].to_numpy(float)
        signed_delta = data[f"delta_place_phase_{axis}_cm"].to_numpy(float)
        if not np.allclose(current - base, signed_delta, equal_nan=True, atol=1e-10):
            raise ValueError(f"Original signed phase baseline mismatch on {axis}")
        data[f"delta_abs_place_phase_{axis}_cm"] = np.abs(current) - np.abs(base)
    return data


def long_values(grid: pd.DataFrame, place: pd.DataFrame) -> pd.DataFrame:
    ids = ["population", "module", "cell_id", "network_id", "trajectory_id", "environment", "length_cm", "deformation_ratio"]
    g = grid.melt(id_vars=ids, value_vars=["rescale_factor"], var_name="metric", value_name="value")
    p = place.melt(id_vars=ids, value_vars=list(PLACE_METRICS), var_name="metric", value_name="value")
    return pd.concat((g, p), ignore_index=True)


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj_keys = ["population", "module", "network_id", "trajectory_id", "environment", "length_cm", "deformation_ratio", "metric"]
    trajectory = cells.groupby(traj_keys, dropna=False, sort=True).agg(
        trajectory_mean=("value", "mean"), n_cells_finite=("value", "count")
    ).reset_index()
    net_keys = [k for k in traj_keys if k != "trajectory_id"]
    network = trajectory.groupby(net_keys, dropna=False, sort=True).agg(
        network_mean=("trajectory_mean", "mean"), n_trajectories_finite=("trajectory_mean", "count")
    ).reset_index()
    group_keys = [k for k in net_keys if k != "network_id"]
    rows = []
    for key, part in network.groupby(group_keys, dropna=False, sort=True):
        finite = part.network_mean.to_numpy(float)
        finite = finite[np.isfinite(finite)]
        n = len(finite)
        mean = float(np.mean(finite)) if n else np.nan
        sd = float(np.std(finite, ddof=1)) if n > 1 else np.nan
        sem = sd / math.sqrt(n) if n > 1 else np.nan
        half = float(stats.t.ppf(.975, n - 1) * sem) if n > 1 else np.nan
        rows.append(dict(zip(group_keys, key)) | {
            "group_mean": mean, "sd_across_networks": sd, "sem_across_networks": sem,
            "ci95_low": mean - half if np.isfinite(half) else np.nan,
            "ci95_high": mean + half if np.isfinite(half) else np.nan,
            "n_networks": n,
        })
    return trajectory, network, pd.DataFrame(rows)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    result = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid):
        return result
    ordered = valid[np.argsort(pvalues[valid])]
    adjusted = pvalues[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
    result[ordered] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def network_statistics(network: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (population, module, metric), part in network.loc[network.environment.ne("fam_100")].groupby(["population", "module", "metric"]):
        family = f"{population}_m{module}_{metric}_six_environments"
        family_rows = []
        for environment, condition in part.groupby("environment"):
            null = 1.0 if metric == "rescale_factor" else 0.0
            values = condition.network_mean.to_numpy(float)
            values = values[np.isfinite(values)]
            n = len(values)
            test = stats.ttest_1samp(values, null) if n > 1 else None
            family_rows.append({
                "family": family, "population": population, "module": module,
                "metric": metric, "environment": environment,
                "length_cm": int(condition.length_cm.iloc[0]), "n_networks": n,
                "null_value": null, "mean_minus_null": float(np.mean(values) - null) if n else np.nan,
                "t_statistic": float(test.statistic) if test else np.nan,
                "df": n - 1, "p_two_sided": float(test.pvalue) if test else np.nan,
            })
        pvalues = np.asarray([row["p_two_sided"] for row in family_rows], float)
        for row, q in zip(family_rows, bh_adjust(pvalues)):
            row["q_bh_within_six_environment_family"] = q
        rows.extend(family_rows)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--place-source", type=Path, default=PLACE_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--network-ids", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def main(edge_positions_per_side: int = 5) -> int:
    args = parse_args()
    network_ids = sorted(set(args.network_ids))
    if edge_positions_per_side != 5:
        raise ValueError("This release includes only the adopted five-edge estimator")
    alias = f"Edge{edge_positions_per_side}-M"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(compute_network, args.data_root.resolve(), output, network_id, edge_positions_per_side) for network_id in network_ids]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"{alias} cache {index}/{len(futures)}: {future.result()}", flush=True)
    grid = grid_cells(output, network_ids)
    place = place_cells(args.place_source.resolve(), network_ids)
    cells = long_values(grid, place)
    trajectory, network, group = aggregate(cells)
    statistics = network_statistics(network)
    metrics = output / "metrics"
    metrics.mkdir(exist_ok=True)
    grid.to_csv(metrics / "grid_source_data_per_cell_trajectory.csv.gz", index=False, compression="gzip")
    place[["population", "module", "cell_id", "network_id", "trajectory_id", "environment", "length_cm", "deformation_ratio", "place_phase_x_cm", "place_phase_y_cm", "baseline_phase_x_cm", "baseline_phase_y_cm", *PLACE_METRICS]].to_csv(metrics / "place_source_data_per_cell_trajectory.csv.gz", index=False, compression="gzip")
    trajectory.to_csv(metrics / "trajectory_level_values.csv.gz", index=False, compression="gzip")
    network.to_csv(metrics / "network_level_values.csv", index=False)
    group.to_csv(metrics / "group_summary_network_unit.csv", index=False)
    statistics.to_csv(metrics / "network_level_statistics.csv", index=False)
    provenance = {
        "alias": alias,
        "description": f"Methods 2.5-cm normalized-mask maps, sigma 1.5 bins, Methods physical scale candidates, {edge_positions_per_side} complete-overlap translations at each x edge",
        "grid": {"bin_cm": 2.5, "gaussian_sigma_bins": 1.5, "unvisited": "NaN/pairwise finite", "edge_positions_per_side": edge_positions_per_side, "scale_candidates": "Methods physical candidate grid", "baseline": "matched network and trajectory, factor=1 by definition"},
        "place_field_length": "Reused audited original-profile per-cell values, current minus matched familiar",
        "place_phase": "abs(current signed phase) - abs(matched familiar signed phase), per cell and axis; not abs(current - familiar)",
        "data_root": str(args.data_root.resolve()), "place_source": str(args.place_source.resolve()),
        "network_ids": network_ids, "trajectories": list(TRAJECTORIES),
        "independent_unit": "network", "aggregation": "cells within trajectory, equal-weight trajectories within network, then networks",
        "statistics": "two-sided one-sample t-test of network means against 1 (grid) or 0 (place), BH across six deformed environments separately per metric/module",
        "python": sys.executable, "python_version": platform.python_version(), "scipy_version": __import__("scipy").__version__,
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), Path(core.__file__), ROOT / "local_network_analysis/grid_rescaling.py")},
    }
    (output / "analysis_qa_and_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{alias} Fig. 2 analysis complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
