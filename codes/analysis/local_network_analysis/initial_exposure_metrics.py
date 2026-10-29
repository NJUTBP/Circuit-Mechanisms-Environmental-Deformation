"""Network-level analysis of initial exposure across all locally available networks.

The simulation directory is treated as read-only.  The independent unit is an
initialized network: cells are averaged within trajectory, the two trajectory
means are averaged within network, and uncertainty/tests are computed from
network means only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = (0, 1)
MODULES = (1, 2, 3)
MODULE_COLORS = {1: "#2878B5", 2: "#D76A5D", 3: "#2A9D78"}
MODULE_LIGHT = {1: "#8EC2DC", 2: "#F0A17F", 3: "#A5D86A"}
PLACE_COLORS = {
    "delta_field_length_x_cm": "#2878B5",
    "delta_place_phase_x_cm": "#E59A2F",
}


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
        if self.stage == "baseline":
            return Path("baseline") / self.identifier
        return Path("initial_exposure") / self.identifier


ENVIRONMENTS = (
    Environment("compressed_70", 70, "initial_exposure"),
    Environment("compressed_80", 80, "initial_exposure"),
    Environment("compressed_90", 90, "initial_exposure"),
    Environment("fam_100", 100, "baseline"),
    Environment("elongated_110", 110, "initial_exposure"),
    Environment("elongated_120", 120, "initial_exposure"),
    Environment("elongated_130", 130, "initial_exposure"),
)

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


def _network_id(path: Path) -> int:
    match = re.fullmatch(r"network_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid network directory name: {path.name}")
    return int(match.group(1))


def discover_networks(data_root: Path, requested: Sequence[int] | None) -> list[tuple[int, Path]]:
    discovered = sorted(
        ((_network_id(path), path) for path in data_root.glob("network_*") if path.is_dir()),
        key=lambda item: item[0],
    )
    if requested is None:
        selected = discovered
    else:
        available = dict(discovered)
        missing = [network_id for network_id in requested if network_id not in available]
        if missing:
            raise FileNotFoundError(f"Requested network directories are missing: {missing}")
        selected = [(network_id, available[network_id]) for network_id in requested]
    if not selected:
        raise FileNotFoundError(f"No network_* directories found below {data_root}")
    return selected


def _read_cells(path: Path, network_id: int, env: Environment, trajectory: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "population", "module", "cell_id", "field_length_x_cm", "field_length_y_cm",
        "grid_spacing_cm", "phase_shift_x_cm", "phase_shift_y_cm",
        "movement_displacement_x_cm", "movement_displacement_y_cm",
    }
    absent = sorted(required.difference(frame.columns))
    if absent:
        raise ValueError(f"{path} lacks columns: {absent}")
    counts = frame["population"].value_counts().to_dict()
    if counts.get("grid") != 300 or counts.get("place") != 100:
        raise ValueError(f"Unexpected population counts in {path}: {counts}")
    stored_networks = set(frame["network_id"].astype(int).unique())
    if stored_networks != {network_id}:
        raise ValueError(f"Network id mismatch in {path}: {stored_networks} versus {network_id}")
    stored_envs = set(frame["environment"].astype(str).unique())
    if stored_envs != {env.identifier}:
        raise ValueError(f"Environment mismatch in {path}: {stored_envs}")
    stored_trajectories = set(frame["trajectory_id"].astype(int).unique())
    if stored_trajectories != {trajectory}:
        raise ValueError(f"Trajectory mismatch in {path}: {stored_trajectories}")
    return frame


def _read_comparison(path: Path, network_id: int, env: Environment, trajectory: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "population", "module", "cell_id", "rescale_factor",
        "rescale_best_correlation", "delta_field_length_cm",
        "delta_movement_displacement_cm",
    }
    absent = sorted(required.difference(frame.columns))
    if absent:
        raise ValueError(f"{path} lacks columns: {absent}")
    counts = frame["population"].value_counts().to_dict()
    if counts.get("grid") != 300 or counts.get("place") != 100:
        raise ValueError(f"Unexpected comparison counts in {path}: {counts}")
    if set(frame["network_id"].astype(int).unique()) != {network_id}:
        raise ValueError(f"Network id mismatch in {path}")
    if set(frame["environment"].astype(str).unique()) != {env.identifier}:
        raise ValueError(f"Environment mismatch in {path}")
    if set(frame["trajectory_id"].astype(int).unique()) != {trajectory}:
        raise ValueError(f"Trajectory mismatch in {path}")
    return frame


def _max_finite_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def load_network_metrics(
    network_id: int,
    network_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    grid_rows: list[dict[str, object]] = []
    place_rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for trajectory in TRAJECTORIES:
        baseline_env = next(env for env in ENVIRONMENTS if env.length_cm == 100)
        baseline_path = (
            network_root / baseline_env.relative_dir / f"trajectory_{trajectory:02d}"
            / "cell_metrics.csv.gz"
        )
        baseline_cells = _read_cells(baseline_path, network_id, baseline_env, trajectory)
        baseline_grid = baseline_cells.loc[baseline_cells["population"].eq("grid")].copy()
        baseline_grid["module"] = baseline_grid["module"].astype(int)
        baseline_grid["cell_id"] = baseline_grid["cell_id"].astype(int)
        baseline_place = baseline_cells.loc[baseline_cells["population"].eq("place")].copy()
        baseline_place["cell_id"] = baseline_place["cell_id"].astype(int)

        for env in ENVIRONMENTS:
            run_dir = network_root / env.relative_dir / f"trajectory_{trajectory:02d}"
            cell_path = run_dir / "cell_metrics.csv.gz"
            current_cells = _read_cells(cell_path, network_id, env, trajectory)
            comparison = None
            if env.length_cm != 100:
                comparison = _read_comparison(
                    run_dir / "comparison_metrics.csv.gz", network_id, env, trajectory
                )

            current_grid = current_cells.loc[current_cells["population"].eq("grid")].copy()
            current_grid["module"] = current_grid["module"].astype(int)
            current_grid["cell_id"] = current_grid["cell_id"].astype(int)
            grid = current_grid.merge(
                baseline_grid[
                    ["module", "cell_id", "grid_spacing_cm", "phase_shift_x_cm", "phase_shift_y_cm"]
                ],
                on=["module", "cell_id"], how="inner", validate="one_to_one",
                suffixes=("", "_baseline"),
            )
            if len(grid) != 300:
                raise ValueError(f"Grid matching failed for network {network_id}, {env.identifier}, trajectory {trajectory}")
            if comparison is None:
                grid["rescale_factor"] = 1.0
                grid["rescale_best_correlation"] = 1.0
                rescale_source = "definition at familiar baseline"
            else:
                comp_grid = comparison.loc[comparison["population"].eq("grid")].copy()
                comp_grid["module"] = comp_grid["module"].astype(int)
                comp_grid["cell_id"] = comp_grid["cell_id"].astype(int)
                grid = grid.merge(
                    comp_grid[["module", "cell_id", "rescale_factor", "rescale_best_correlation"]],
                    on=["module", "cell_id"], how="inner", validate="one_to_one",
                )
                rescale_source = "saved production comparison_metrics"

            spacing = grid["grid_spacing_cm_baseline"].to_numpy(float)
            phase_x = grid["phase_shift_x_cm"].to_numpy(float)
            phase_y = grid["phase_shift_y_cm"].to_numpy(float)
            base_x = grid["phase_shift_x_cm_baseline"].to_numpy(float)
            base_y = grid["phase_shift_y_cm_baseline"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_x = 2.0 * phase_x / spacing
                ratio_y = 2.0 * phase_y / spacing
                delta_ratio_x = 2.0 * (phase_x - base_x) / spacing
                delta_ratio_y = 2.0 * (phase_y - base_y) / spacing
            for row_index, row in grid.reset_index(drop=True).iterrows():
                grid_rows.append({
                    "network_id": network_id,
                    "stage": env.stage,
                    "environment": env.identifier,
                    "length_cm": env.length_cm,
                    "deformation_ratio": env.ratio,
                    "trajectory_id": trajectory,
                    "module": int(row["module"]),
                    "cell_id": int(row["cell_id"]),
                    "baseline_grid_spacing_cm": spacing[row_index],
                    "rescale_factor": float(row["rescale_factor"]),
                    "rescale_best_correlation": float(row["rescale_best_correlation"]),
                    "phase_shift_x_cm": phase_x[row_index],
                    "phase_shift_y_cm": phase_y[row_index],
                    "phase_ratio_x_half_period": ratio_x[row_index],
                    "phase_ratio_y_half_period": ratio_y[row_index],
                    "delta_phase_shift_x_cm": phase_x[row_index] - base_x[row_index],
                    "delta_phase_shift_y_cm": phase_y[row_index] - base_y[row_index],
                    "delta_phase_ratio_x_half_period": delta_ratio_x[row_index],
                    "delta_phase_ratio_y_half_period": delta_ratio_y[row_index],
                })

            current_place = current_cells.loc[current_cells["population"].eq("place")].copy()
            current_place["cell_id"] = current_place["cell_id"].astype(int)
            place = current_place.merge(
                baseline_place[
                    ["cell_id", "field_length_x_cm", "field_length_y_cm",
                     "movement_displacement_x_cm", "movement_displacement_y_cm"]
                ],
                on="cell_id", how="inner", validate="one_to_one",
                suffixes=("", "_baseline"),
            )
            if len(place) != 100:
                raise ValueError(f"Place-cell matching failed for network {network_id}, {env.identifier}, trajectory {trajectory}")
            direct_delta_length_x = (
                place["field_length_x_cm"].to_numpy(float)
                - place["field_length_x_cm_baseline"].to_numpy(float)
            )
            direct_delta_phase_x = (
                place["movement_displacement_x_cm"].to_numpy(float)
                - place["movement_displacement_x_cm_baseline"].to_numpy(float)
            )
            if comparison is None:
                delta_length_x = direct_delta_length_x
                delta_phase_x = direct_delta_phase_x
                delta_length_check = 0.0
                delta_phase_check = 0.0
                place_source = "matched familiar values; zero by definition where finite"
            else:
                comp_place = comparison.loc[comparison["population"].eq("place")].copy()
                comp_place["cell_id"] = comp_place["cell_id"].astype(int)
                place = place.merge(
                    comp_place[["cell_id", "delta_field_length_cm", "delta_movement_displacement_cm"]],
                    on="cell_id", how="inner", validate="one_to_one",
                )
                delta_length_x = place["delta_field_length_cm"].to_numpy(float)
                delta_phase_x = place["delta_movement_displacement_cm"].to_numpy(float)
                delta_length_check = _max_finite_abs(delta_length_x - direct_delta_length_x)
                delta_phase_check = _max_finite_abs(delta_phase_x - direct_delta_phase_x)
                if delta_length_check > 1e-9 or delta_phase_check > 1e-9:
                    raise ValueError(
                        "Saved and directly matched place changes disagree for "
                        f"network {network_id}, {env.identifier}, trajectory {trajectory}: "
                        f"length={delta_length_check}, displacement={delta_phase_check}"
                    )
                place_source = "saved production comparison_metrics"
            delta_length_y = (
                place["field_length_y_cm"].to_numpy(float)
                - place["field_length_y_cm_baseline"].to_numpy(float)
            )
            delta_phase_y = (
                place["movement_displacement_y_cm"].to_numpy(float)
                - place["movement_displacement_y_cm_baseline"].to_numpy(float)
            )
            for row_index, row in place.reset_index(drop=True).iterrows():
                place_rows.append({
                    "network_id": network_id,
                    "stage": env.stage,
                    "environment": env.identifier,
                    "length_cm": env.length_cm,
                    "deformation_ratio": env.ratio,
                    "trajectory_id": trajectory,
                    "cell_id": int(row["cell_id"]),
                    "field_length_x_cm": float(row["field_length_x_cm"]),
                    "field_length_y_cm": float(row["field_length_y_cm"]),
                    "place_phase_x_cm": float(row["movement_displacement_x_cm"]),
                    "place_phase_y_cm": float(row["movement_displacement_y_cm"]),
                    "baseline_field_length_x_cm": float(row["field_length_x_cm_baseline"]),
                    "baseline_field_length_y_cm": float(row["field_length_y_cm_baseline"]),
                    "baseline_place_phase_x_cm": float(row["movement_displacement_x_cm_baseline"]),
                    "baseline_place_phase_y_cm": float(row["movement_displacement_y_cm_baseline"]),
                    "delta_field_length_x_cm": delta_length_x[row_index],
                    "delta_field_length_y_cm": delta_length_y[row_index],
                    "delta_place_phase_x_cm": delta_phase_x[row_index],
                    "delta_place_phase_y_cm": delta_phase_y[row_index],
                })

            audit.append({
                "network_id": network_id,
                "environment": env.identifier,
                "length_cm": env.length_cm,
                "trajectory_id": trajectory,
                "cell_metrics": str(cell_path.relative_to(PROJECT_ROOT)),
                "comparison_metrics": (
                    None if comparison is None
                    else str((run_dir / "comparison_metrics.csv.gz").relative_to(PROJECT_ROOT))
                ),
                "rescale_source": rescale_source,
                "place_change_source": place_source,
                "max_abs_place_length_crosscheck_cm": delta_length_check,
                "max_abs_place_displacement_crosscheck_cm": delta_phase_check,
            })

    grid_frame = pd.DataFrame(grid_rows).sort_values(
        ["network_id", "length_cm", "trajectory_id", "module", "cell_id"]
    )
    place_frame = pd.DataFrame(place_rows).sort_values(
        ["network_id", "length_cm", "trajectory_id", "cell_id"]
    )
    return grid_frame, place_frame, audit


def _summarize_population(
    frame: pd.DataFrame,
    population: str,
    metrics: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_vars = [
        "network_id", "environment", "length_cm", "deformation_ratio",
        "trajectory_id",
    ]
    working = frame.copy()
    if population == "grid":
        id_vars.append("module")
    else:
        working["module"] = 0
        id_vars.append("module")
    long = working.melt(
        id_vars=id_vars,
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    trajectory_keys = id_vars + ["metric"]
    trajectory = (
        long.groupby(trajectory_keys, sort=True, dropna=False)["value"]
        .agg(
            trajectory_mean="mean",
            trajectory_median="median",
            sd_across_cells="std",
            n_cells_finite="count",
        )
        .reset_index()
    )
    trajectory["population"] = population
    network_keys = [
        "network_id", "environment", "length_cm", "deformation_ratio", "module", "metric",
    ]
    network = (
        trajectory.groupby(network_keys, sort=True, dropna=False)["trajectory_mean"]
        .agg(
            network_mean="mean",
            sd_across_trajectories="std",
            n_trajectories_finite="count",
        )
        .reset_index()
    )
    network["population"] = population

    group_rows: list[dict[str, object]] = []
    group_keys = ["environment", "length_cm", "deformation_ratio", "module", "metric"]
    for key, subset in network.groupby(group_keys, sort=True, dropna=False):
        values = subset["network_mean"].to_numpy(float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        sem = sd / math.sqrt(n) if n > 1 else np.nan
        half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
        group_rows.append({
            "population": population,
            "environment": key[0],
            "length_cm": key[1],
            "deformation_ratio": key[2],
            "module": key[3],
            "metric": key[4],
            "group_mean": mean,
            "sd_across_networks": sd,
            "sem_across_networks": sem,
            "ci95_low": mean - half if np.isfinite(half) else np.nan,
            "ci95_high": mean + half if np.isfinite(half) else np.nan,
            "median_across_networks": float(np.median(values)) if n else np.nan,
            "q25_across_networks": float(np.quantile(values, 0.25)) if n else np.nan,
            "q75_across_networks": float(np.quantile(values, 0.75)) if n else np.nan,
            "n_networks": n,
        })
    return trajectory, network, pd.DataFrame(group_rows)


def _fdr_bh(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    result = np.full(len(array), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(array))
    if finite_indices.size == 0:
        return pd.Series(result, index=values.index)
    order = finite_indices[np.argsort(array[finite_indices])]
    ranked = array[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def _difference_test(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = sd / math.sqrt(n) if n > 1 else np.nan
    half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
    if n > 1 and sd > 0:
        t_stat, p_t = stats.ttest_1samp(values, 0.0)
        dz = mean / sd
    elif n and np.allclose(values, 0.0):
        t_stat, p_t, dz = 0.0, 1.0, 0.0
    else:
        t_stat = p_t = dz = np.nan
    try:
        if n == 0:
            w_stat = p_w = np.nan
        elif np.allclose(values, 0.0):
            w_stat, p_w = 0.0, 1.0
        else:
            w_stat, p_w = stats.wilcoxon(values)
    except ValueError:
        w_stat = p_w = np.nan
    return {
        "n_networks": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": mean - half if np.isfinite(half) else np.nan,
        "ci95_high": mean + half if np.isfinite(half) else np.nan,
        "cohens_dz": dz,
        "t_statistic": float(t_stat),
        "p_ttest": float(p_t),
        "wilcoxon_statistic": float(w_stat),
        "p_wilcoxon": float(p_w),
    }


def compute_network_statistics(network_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grid_rescale = network_values.loc[
        network_values["population"].eq("grid")
        & network_values["metric"].eq("rescale_factor")
        & network_values["length_cm"].ne(100)
    ]
    for (environment, length_cm, ratio, module), subset in grid_rescale.groupby(
        ["environment", "length_cm", "deformation_ratio", "module"], sort=True
    ):
        values = subset["network_mean"].to_numpy(float)
        for family, reference in (
            ("grid_rescaling_vs_no_rescaling", 1.0),
            ("grid_rescaling_vs_geometric_rescaling", float(ratio)),
        ):
            rows.append({
                "family": family,
                "population": "grid",
                "metric": "rescale_factor",
                "contrast": f"network mean minus {reference:.3f}",
                "environment": environment,
                "length_cm": length_cm,
                "deformation_ratio": ratio,
                "module": module,
                "reference": reference,
                **_difference_test(values - reference),
            })

    phase = network_values.loc[
        network_values["population"].eq("grid")
        & network_values["metric"].isin(
            ["phase_ratio_x_half_period", "phase_ratio_y_half_period"]
        )
    ].pivot_table(
        index=["network_id", "environment", "length_cm", "deformation_ratio", "module"],
        columns="metric", values="network_mean",
    ).reset_index()
    for (environment, length_cm, ratio, module), subset in phase.groupby(
        ["environment", "length_cm", "deformation_ratio", "module"], sort=True
    ):
        difference = (
            subset["phase_ratio_x_half_period"].to_numpy(float)
            - subset["phase_ratio_y_half_period"].to_numpy(float)
        )
        rows.append({
            "family": "grid_phase_deformed_vs_undeformed_axis",
            "population": "grid",
            "metric": "phase_ratio_x_minus_y",
            "contrast": "deformed x axis minus undeformed y axis",
            "environment": environment,
            "length_cm": length_cm,
            "deformation_ratio": ratio,
            "module": module,
            "reference": 0.0,
            **_difference_test(difference),
        })

    place = network_values.loc[
        network_values["population"].eq("place")
        & network_values["metric"].isin(
            ["delta_field_length_x_cm", "delta_place_phase_x_cm"]
        )
        & network_values["length_cm"].ne(100)
    ]
    for (metric, environment, length_cm, ratio), subset in place.groupby(
        ["metric", "environment", "length_cm", "deformation_ratio"], sort=True
    ):
        rows.append({
            "family": "place_change_vs_familiar",
            "population": "place",
            "metric": metric,
            "contrast": "deformed minus matched familiar value",
            "environment": environment,
            "length_cm": length_cm,
            "deformation_ratio": ratio,
            "module": 0,
            "reference": 0.0,
            **_difference_test(subset["network_mean"].to_numpy(float)),
        })

    result = pd.DataFrame(rows)
    for column in ("p_ttest", "p_wilcoxon"):
        result[f"{column}_fdr_bh_within_family"] = (
            result.groupby("family", sort=False, group_keys=False)[column].apply(_fdr_bh)
        )
    return result.sort_values(["family", "metric", "module", "length_cm"]).reset_index(drop=True)


def _configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.2,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.55, length=3)


def _save_all(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg", ".tiff")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    with Image.open(outputs[0]) as preview:
        tifffile.imwrite(
            outputs[3], np.asarray(preview), compression="zlib", resolution=(600, 600)
        )
    return outputs


def _network_metric(network_values: pd.DataFrame, population: str, metric: str) -> pd.DataFrame:
    return network_values.loc[
        network_values["population"].eq(population)
        & network_values["metric"].eq(metric)
    ].copy()


def plot_grid_rescaling(
    network_values: pd.DataFrame,
    group_summary: pd.DataFrame,
    figure_dir: Path,
    n_networks: int,
) -> list[Path]:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True, sharey=True)
    x_values = np.asarray([env.ratio for env in ENVIRONMENTS])
    network_data = _network_metric(network_values, "grid", "rescale_factor")
    group_data = group_summary.loc[
        group_summary["population"].eq("grid")
        & group_summary["metric"].eq("rescale_factor")
    ]
    for index, module in enumerate(MODULES):
        ax = axes[index]
        module_networks = network_data.loc[network_data["module"].eq(module)]
        for _, subset in module_networks.groupby("network_id", sort=True):
            subset = subset.sort_values("length_cm")
            ax.plot(
                subset["deformation_ratio"], subset["network_mean"],
                color=MODULE_COLORS[module], alpha=0.20, linewidth=0.75,
                marker="o", markersize=2.3, markerfacecolor="white", markeredgewidth=0.45,
            )
        curve = group_data.loc[group_data["module"].eq(module)].sort_values("length_cm")
        ax.errorbar(
            curve["deformation_ratio"], curve["group_mean"],
            yerr=curve["sd_across_networks"],
            color=MODULE_COLORS[module], linewidth=1.65, elinewidth=0.7,
            capsize=1.8, marker="o", markersize=4.0, zorder=4,
            label="Network mean +/- SD",
        )
        ax.plot(x_values, x_values, color="#666666", linestyle="--", linewidth=0.9,
                label="Geometric rescaling")
        ax.axhline(1.0, color="#999999", linestyle=":", linewidth=0.9,
                   label="No rescaling")
        ax.set_title(f"Grid module {module}", color=MODULE_COLORS[module], fontweight="bold")
        ax.set_xlabel("Environmental deformation ratio")
        ax.set_xlim(0.675, 1.325)
        ax.set_xticks(x_values)
        ax.set_ylim(0.66, 1.42)
        _despine(ax)
        ax.text(-0.15, 1.07, chr(ord("A") + index), transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", va="top")
        if index == 0:
            ax.set_ylabel("Grid-rescaling factor")
            ax.legend(loc="upper left", handlelength=2.0)
    fig.text(
        0.5, 0.005,
        f"n = {n_networks} independent networks; thin curves are network means and bold curves show mean +/- SD across networks.",
        ha="center", va="bottom", fontsize=5.9, color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.23, wspace=0.22)
    outputs = _save_all(fig, figure_dir / "figure_grid_rescaling_network_level")
    plt.close(fig)
    return outputs


def plot_grid_phase(
    network_values: pd.DataFrame,
    group_summary: pd.DataFrame,
    figure_dir: Path,
    n_networks: int,
) -> list[Path]:
    _configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 4.25), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.2, 0.82), hspace=0.48, wspace=0.28)
    metric_specs = (
        ("phase_ratio_x_half_period", "Deformed x axis", "o"),
        ("phase_ratio_y_half_period", "Undeformed y axis", "s"),
    )
    x_values = np.asarray([env.ratio for env in ENVIRONMENTS])
    for index, module in enumerate(MODULES):
        ax = fig.add_subplot(grid[0, index])
        for metric_index, (metric, label, marker) in enumerate(metric_specs):
            network_data = _network_metric(network_values, "grid", metric)
            network_data = network_data.loc[network_data["module"].eq(module)]
            color = MODULE_COLORS[module] if metric_index == 0 else MODULE_LIGHT[module]
            for _, subset in network_data.groupby("network_id", sort=True):
                subset = subset.sort_values("length_cm")
                ax.plot(
                    subset["deformation_ratio"], subset["network_mean"],
                    color=color, alpha=0.16, linewidth=0.65,
                    marker=marker, markersize=1.9, markerfacecolor="white", markeredgewidth=0.4,
                )
            curve = group_summary.loc[
                group_summary["population"].eq("grid")
                & group_summary["metric"].eq(metric)
                & group_summary["module"].eq(module)
            ].sort_values("length_cm")
            ax.errorbar(
                curve["deformation_ratio"], curve["group_mean"],
                yerr=curve["sd_across_networks"], color=color,
                linewidth=1.45, elinewidth=0.6, capsize=1.5,
                marker=marker, markersize=3.7, label=label, zorder=4,
            )
        ax.axvline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.set_title(f"Grid module {module}", color=MODULE_COLORS[module], fontweight="bold")
        ax.set_xlabel("Environmental deformation ratio")
        ax.set_xlim(0.675, 1.325)
        ax.set_xticks(x_values)
        ax.set_ylim(bottom=0)
        _despine(ax)
        ax.text(-0.16, 1.08, chr(ord("A") + index), transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", va="top")
        if index == 0:
            ax.set_ylabel("Boundary phase ratio, $R_{phase}$")
            ax.legend(loc="upper left")

    ax = fig.add_subplot(grid[1, :])
    phase = network_values.loc[
        network_values["population"].eq("grid")
        & network_values["metric"].isin([spec[0] for spec in metric_specs])
        & network_values["length_cm"].ne(100)
    ]
    collapsed = (
        phase.groupby(["network_id", "module", "metric"], sort=True)["network_mean"]
        .mean().reset_index()
    )
    centers = np.arange(3, dtype=float)
    offset = 0.11
    for module in MODULES:
        pivot = collapsed.loc[collapsed["module"].eq(module)].pivot(
            index="network_id", columns="metric", values="network_mean"
        ).dropna()
        xpos = centers[module - 1]
        for _, row in pivot.iterrows():
            ax.plot(
                [xpos - offset, xpos + offset],
                [row["phase_ratio_x_half_period"], row["phase_ratio_y_half_period"]],
                color="#A7A7A7", alpha=0.55, linewidth=0.65, zorder=1,
            )
            ax.scatter(
                [xpos - offset, xpos + offset],
                [row["phase_ratio_x_half_period"], row["phase_ratio_y_half_period"]],
                s=10, facecolor="white",
                edgecolor=[MODULE_COLORS[module], MODULE_LIGHT[module]],
                linewidth=0.55, zorder=2,
            )
        for metric_index, (metric, _, marker) in enumerate(metric_specs):
            values = pivot[metric].to_numpy(float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            position = xpos + (-offset if metric_index == 0 else offset)
            color = MODULE_COLORS[module] if metric_index == 0 else MODULE_LIGHT[module]
            ax.errorbar(
                position, mean, yerr=sd, color=color, marker=marker,
                markersize=5.0, linewidth=0, elinewidth=1.0, capsize=2.2, zorder=4,
            )
    ax.set_xticks(centers, [f"Module {module}" for module in MODULES])
    ax.set_ylabel("Mean $R_{phase}$ across six deformed environments")
    ax.set_ylim(bottom=0)
    _despine(ax)
    ax.legend(
        handles=[
            Line2D([], [], color="#444444", marker="o", linestyle="none", label="Deformed x axis"),
            Line2D([], [], color="#AAAAAA", marker="s", linestyle="none", label="Undeformed y axis"),
        ],
        loc="upper left", ncol=2,
    )
    ax.text(-0.05, 1.07, "D", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="top")
    fig.text(
        0.5, 0.008,
        f"n = {n_networks} independent networks; curves and paired points use network means. Error bars are SD across networks.",
        ha="center", va="bottom", fontsize=5.9, color="#444444",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.105)
    outputs = _save_all(fig, figure_dir / "figure_grid_boundary_phase_network_level")
    plt.close(fig)
    return outputs


def plot_place_changes(
    network_values: pd.DataFrame,
    group_summary: pd.DataFrame,
    figure_dir: Path,
    n_networks: int,
) -> list[Path]:
    _configure_matplotlib()
    specs = (
        ("delta_field_length_x_cm", r"Change in place-field length, $\Delta L_{PF}$ (cm)"),
        ("delta_place_phase_x_cm", r"Change in opposing-direction displacement, $\Delta D_{PF}$ (cm)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)
    x_values = np.asarray([env.ratio for env in ENVIRONMENTS])
    for index, (metric, ylabel) in enumerate(specs):
        ax = axes[index]
        color = PLACE_COLORS[metric]
        network_data = _network_metric(network_values, "place", metric)
        for _, subset in network_data.groupby("network_id", sort=True):
            subset = subset.sort_values("length_cm")
            ax.plot(
                subset["deformation_ratio"], subset["network_mean"],
                color=color, alpha=0.22, linewidth=0.75,
                marker="o", markersize=2.4, markerfacecolor="white", markeredgewidth=0.45,
            )
        curve = group_summary.loc[
            group_summary["population"].eq("place")
            & group_summary["metric"].eq(metric)
        ].sort_values("length_cm")
        ax.errorbar(
            curve["deformation_ratio"], curve["group_mean"],
            yerr=curve["sd_across_networks"], color=color,
            linewidth=1.7, elinewidth=0.75, capsize=1.9,
            marker="o", markersize=4.2, zorder=4,
        )
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.85)
        ax.axvline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Environmental deformation ratio")
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis="y", colors=color)
        ax.spines["left"].set_color(color)
        ax.set_xlim(0.675, 1.325)
        ax.set_xticks(x_values)
        _despine(ax)
        ax.text(-0.12, 1.06, chr(ord("A") + index), transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", va="top")
    fig.text(
        0.5, 0.008,
        f"Deformed minus matched 100-cm familiar value; n = {n_networks} networks. Thin curves are network means; bold curves show mean +/- SD.",
        ha="center", va="bottom", fontsize=5.9, color="#444444",
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.91, bottom=0.22, wspace=0.30)
    outputs = _save_all(fig, figure_dir / "figure_place_length_and_phase_change_network_level")
    plt.close(fig)
    return outputs


def write_outputs(
    output_root: Path,
    networks: list[tuple[int, Path]],
    grid_cells: pd.DataFrame,
    place_cells: pd.DataFrame,
    trajectory_values: pd.DataFrame,
    network_values: pd.DataFrame,
    group_summary: pd.DataFrame,
    statistics: pd.DataFrame,
    source_audit: list[dict[str, object]],
    figure_paths: list[Path],
) -> None:
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    grid_cells.to_csv(metrics_dir / "grid_source_data_per_cell_trajectory.csv.gz", index=False)
    place_cells.to_csv(metrics_dir / "place_source_data_per_cell_trajectory.csv.gz", index=False)
    trajectory_values.to_csv(metrics_dir / "trajectory_level_values.csv.gz", index=False)
    network_values.to_csv(metrics_dir / "network_level_values.csv", index=False)
    group_summary.to_csv(metrics_dir / "group_summary_network_unit.csv", index=False)
    statistics.to_csv(metrics_dir / "network_level_statistics.csv", index=False)
    trajectory_values[
        [
            "population", "network_id", "environment", "length_cm", "trajectory_id",
            "module", "metric", "n_cells_finite",
        ]
    ].to_csv(metrics_dir / "finite_cell_counts.csv", index=False)

    expected_ids = list(range(min(network_id for network_id, _ in networks),
                              max(network_id for network_id, _ in networks) + 1))
    present_ids = [network_id for network_id, _ in networks]
    qa = {
        "figure_contract": "local_network_analysis/figure_contract_multinetwork.md",
        "core_conclusion": (
            "Environment deformation is summarized through module-specific grid rescaling, "
            "axis-specific grid phase, and matched place-field length/displacement changes "
            "using independent networks as the statistical unit."
        ),
        "available_network_ids": present_ids,
        "missing_network_ids_within_observed_range": sorted(set(expected_ids).difference(present_ids)),
        "n_networks": len(networks),
        "n_trajectories_per_condition": 2,
        "n_cells_per_grid_module": 100,
        "n_place_cells": 100,
        "n_expected_sessions": len(networks) * len(ENVIRONMENTS) * len(TRAJECTORIES),
        "n_loaded_sessions": len(source_audit),
        "independent_unit": "network",
        "aggregation_order": (
            "finite cells within trajectory, then two trajectory means within network, "
            "then network means across the available networks"
        ),
        "uncertainty": "SD across network means",
        "grid_phase_definition": (
            "2 * absolute opposing-boundary phase separation / matched familiar grid spacing"
        ),
        "place_length_definition": (
            "production field_length_x_cm; delta is deformed minus matched familiar value"
        ),
        "place_phase_definition": (
            "production movement_displacement_x_cm: absolute x lag maximizing correlation "
            "between rate maps formed during movement in opposite x directions; delta is "
            "deformed minus matched familiar value"
        ),
        "missing_value_policy": (
            "No imputation. Metrics are averaged over finite matched cell values, and finite "
            "cell counts are exported for every trajectory and condition."
        ),
        "statistics": (
            "Two-sided one-sample/paired tests on network means only; Benjamini-Hochberg "
            "correction is applied separately within each declared family."
        ),
        "source_audit": source_audit,
        "figure_files": [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths],
    }
    (output_root / "analysis_qa_and_provenance.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data_simu")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "local_network_analysis" / "outputs" / "initial_exposure_multinetwork_n7",
    )
    parser.add_argument(
        "--network-ids", type=int, nargs="*", default=None,
        help="Optional explicit network ids. By default all local network_* directories are used.",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    networks = discover_networks(data_root, args.network_ids)
    grid_frames: list[pd.DataFrame] = []
    place_frames: list[pd.DataFrame] = []
    source_audit: list[dict[str, object]] = []
    for network_id, network_root in networks:
        print(f"Loading network {network_id:03d} from {network_root.name}...")
        grid, place, audit = load_network_metrics(network_id, network_root)
        grid_frames.append(grid)
        place_frames.append(place)
        source_audit.extend(audit)

    grid_cells = pd.concat(grid_frames, ignore_index=True)
    place_cells = pd.concat(place_frames, ignore_index=True)
    grid_trajectory, grid_network, grid_group = _summarize_population(
        grid_cells, "grid", GRID_METRICS
    )
    place_trajectory, place_network, place_group = _summarize_population(
        place_cells, "place", PLACE_METRICS
    )
    trajectory_values = pd.concat([grid_trajectory, place_trajectory], ignore_index=True)
    network_values = pd.concat([grid_network, place_network], ignore_index=True)
    group_summary = pd.concat([grid_group, place_group], ignore_index=True)
    statistics = compute_network_statistics(network_values)

    figure_dir = output_root / "figures"
    figure_paths: list[Path] = []
    figure_paths.extend(plot_grid_rescaling(network_values, group_summary, figure_dir, len(networks)))
    figure_paths.extend(plot_grid_phase(network_values, group_summary, figure_dir, len(networks)))
    figure_paths.extend(plot_place_changes(network_values, group_summary, figure_dir, len(networks)))
    write_outputs(
        output_root, networks, grid_cells, place_cells, trajectory_values,
        network_values, group_summary, statistics, source_audit, figure_paths,
    )
    print(f"Analysis complete: {output_root}")
    print(f"Networks: {[network_id for network_id, _ in networks]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
