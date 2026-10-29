"""Recalculate Figure 3F from one firing field in each grid-cell rate map.

This deliberately leaves the older autocorrelation-based Figure 3 output
untouched.  The primary field is the 4-connected, >=50%-of-peak component
containing the cell's global maximum in the saved 2.5-cm Methods rate map.
Its x/y extents are numbers of occupied columns/rows times 2.5 cm.  A 30%
threshold is exported as a sensitivity check, not substituted into the plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

import grid_phase_network as previous


ROOT = previous.ROOT
OUTPUT = ROOT / "outputs" / "grid_field_extent"
THRESHOLDS = (0.5, 0.3)
STRUCTURE_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
GROUP_KEYS = ["network_id", "environment", "length_cm", "deformation_ratio", "module"]


def single_field_extents(
    rate_maps: np.ndarray, bin_cm: float, peak_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounding-bin widths of the single field containing each map's peak."""
    x_lengths = np.full(len(rate_maps), np.nan, dtype=float)
    y_lengths = np.full(len(rate_maps), np.nan, dtype=float)
    for cell_id, rate_map in enumerate(rate_maps):
        finite = np.isfinite(rate_map)
        if not finite.any():
            continue
        peak_index = np.nanargmax(rate_map)
        peak = float(rate_map.flat[peak_index])
        if peak <= 0:
            continue
        peak_y, peak_x = np.unravel_index(peak_index, rate_map.shape)
        mask = finite & (rate_map >= peak_fraction * peak)
        components, _ = label(mask, structure=STRUCTURE_4)
        field = components == components[peak_y, peak_x]
        yy, xx = np.nonzero(field)
        if len(xx) == 0:
            continue
        x_lengths[cell_id] = (int(xx.max()) - int(xx.min()) + 1) * bin_cm
        y_lengths[cell_id] = (int(yy.max()) - int(yy.min()) + 1) * bin_cm
    return x_lengths, y_lengths


def replace_grid_field_lengths(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    cells = cells.copy()
    cells["old_autocorr_field_length_x_cm"] = cells["field_length_x_cm"]
    cells["old_autocorr_field_length_y_cm"] = cells["field_length_y_cm"]
    sensitivity_parts: list[pd.DataFrame] = []
    map_records: list[dict] = []
    expected_groups = len(previous.NETWORK_IDS) * len(previous.ENVIRONMENTS) * len(previous.TRAJECTORIES)
    grouped = cells.groupby(["network_id", "environment", "trajectory_id"], sort=True)
    if grouped.ngroups != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} map groups; found {grouped.ngroups}")
    for (network_id, environment, trajectory_id), condition_cells in grouped:
        stage = "baseline" if environment == "fam_100" else "initial_exposure"
        path = (
            previous.DATA_ROOT / f"network_{int(network_id):03d}" / stage
            / environment / f"trajectory_{int(trajectory_id):02d}" / "rate_maps.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as maps:
            bin_cm = float(maps["bin_size_cm"])
            width_cm = float(maps["width_cm"])
            if not np.isclose(bin_cm, 2.5):
                raise RuntimeError(f"Unexpected spatial bin in {path}: {bin_cm}")
            expected_width = float(condition_cells["length_cm"].iloc[0])
            if not np.isclose(width_cm, expected_width):
                raise RuntimeError(f"Width mismatch in {path}: {width_cm} != {expected_width}")
            for module, module_cells in condition_cells.groupby("module", sort=True):
                cell_ids = module_cells["cell_id"].to_numpy(int)
                if len(cell_ids) != 100 or not np.array_equal(np.sort(cell_ids), np.arange(100)):
                    raise RuntimeError(f"Unexpected grid-cell indexing in {path}, module {module}")
                rate_maps = np.asarray(maps[f"grid_m{int(module)}"], dtype=float)
                if rate_maps.shape[0] != 100:
                    raise RuntimeError(f"Unexpected map count in {path}, module {module}")
                for threshold in THRESHOLDS:
                    x, y = single_field_extents(rate_maps, bin_cm, threshold)
                    selected = module_cells[["network_id", "environment", "length_cm",
                                             "deformation_ratio", "trajectory_id", "module",
                                             "cell_id"]].copy()
                    selected["threshold_fraction"] = threshold
                    selected["field_length_x_cm"] = x[cell_ids]
                    selected["field_length_y_cm"] = y[cell_ids]
                    sensitivity_parts.append(selected)
                    if threshold == THRESHOLDS[0]:
                        cells.loc[module_cells.index, "field_length_x_cm"] = x[cell_ids]
                        cells.loc[module_cells.index, "field_length_y_cm"] = y[cell_ids]
        map_records.append(
            {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
             "sha256": previous._sha256(path)}
        )
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    return cells, sensitivity, map_records


def summarize_properties(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = previous.PROPERTY_COLUMNS
    trajectory = (
        cells.groupby(GROUP_KEYS + ["trajectory_id"], as_index=False, sort=True)[metrics]
        .mean()
    )
    counts = (
        cells.groupby(GROUP_KEYS + ["trajectory_id"], as_index=False, sort=True)[metrics]
        .count()
        .rename(columns={m: f"n_finite_{m}" for m in metrics})
    )
    trajectory = trajectory.merge(counts, on=GROUP_KEYS + ["trajectory_id"], validate="one_to_one")
    grid_props = trajectory.groupby(GROUP_KEYS, as_index=False, sort=True)[metrics].mean()
    n_trajectories = (
        trajectory.groupby(GROUP_KEYS, as_index=False, sort=True)["trajectory_id"]
        .nunique().rename(columns={"trajectory_id": "n_trajectories"})
    )
    grid_props = grid_props.merge(n_trajectories, on=GROUP_KEYS, validate="one_to_one")
    if not (grid_props["n_trajectories"] == 2).all():
        raise RuntimeError("Each network-condition-module must have two trajectories")
    return trajectory, grid_props


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase = previous.load_phase_network_values()
    cells, _, _, metric_records = previous.load_grid_properties()
    cells, sensitivity, map_records = replace_grid_field_lengths(cells)
    trajectory, grid_props = summarize_properties(cells)
    abc, panel_values, _ = previous.build_panel_values(phase, grid_props)
    tests, paired = previous.calculate_statistics(panel_values)
    abc_summary = previous._summary(
        abc, ["module", "axis", "environment", "length_cm", "deformation_ratio"]
    )
    panel_summary = previous._summary(panel_values, ["panel", "measure", "module", "axis", "group"])
    if not (tests["n_networks"] == len(previous.NETWORK_IDS)).all():
        raise RuntimeError("All Figure 3 comparisons must include paired networks")
    if not (abc_summary["n_networks"] == len(previous.NETWORK_IDS)).all():
        raise RuntimeError("All A-C points must include every network")
    if cells[["field_length_x_cm", "field_length_y_cm"]].isna().any().any():
        raise RuntimeError("One or more primary grid-field extents are nonfinite")

    cell_sensitivity = (
        sensitivity.groupby(GROUP_KEYS + ["trajectory_id", "threshold_fraction"],
                            as_index=False, sort=True)[["field_length_x_cm", "field_length_y_cm"]].mean()
    )
    network_sensitivity = (
        cell_sensitivity.groupby(GROUP_KEYS + ["threshold_fraction"], as_index=False, sort=True)
        [["field_length_x_cm", "field_length_y_cm"]].mean()
    )
    cells.to_csv(OUTPUT / "grid_property_cell_trajectory.csv.gz", index=False, compression="gzip")
    trajectory.to_csv(OUTPUT / "grid_property_trajectory_summary.csv", index=False)
    grid_props.to_csv(OUTPUT / "grid_property_network_condition_module.csv", index=False)
    network_sensitivity.to_csv(OUTPUT / "single_field_threshold_sensitivity_network_values.csv", index=False)
    abc.to_csv(OUTPUT / "panels_ABC_network_values.csv", index=False)
    abc_summary.to_csv(OUTPUT / "panels_ABC_group_summary.csv", index=False)
    panel_values.to_csv(OUTPUT / "panels_DEFG_network_values.csv", index=False)
    panel_summary.to_csv(OUTPUT / "panels_DEFG_group_summary.csv", index=False)
    tests.to_csv(OUTPUT / "network_level_statistics.csv", index=False)
    paired.to_csv(OUTPUT / "network_level_paired_values.csv", index=False)
    provenance = {
        "analysis": "V4 Figure 3 with corrected single-field grid-cell length in F",
        "grid_field_length": {
            "primary_threshold_fraction_of_cell_peak": THRESHOLDS[0],
            "sensitivity_threshold_fraction_of_cell_peak": THRESHOLDS[1],
            "field_selection": "4-connected suprathreshold component containing global map maximum",
            "length_definition": "bounding extent of that one component in x/y bins times 2.5 cm",
            "input": "saved 2.5-cm occupancy-normalized rate maps smoothed with sigma=1.5 bins",
            "not_used": "central autocorrelation component or inter-field spacing",
        },
        "aggregation": "cell -> trajectory -> network-condition-module; F familiar x averages modules and deformation x/y averages six conditions and modules within each network",
        "statistics": "n=8 paired networks, two-sided paired t; BH correction across the two F comparisons",
        "source_cell_metrics": metric_records,
        "source_rate_maps": map_records,
        "source_phase": str(previous.PHASE_SOURCE.relative_to(ROOT)),
    }
    (OUTPUT / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote corrected Figure 3 source tables to {OUTPUT}")
    print(panel_summary.loc[panel_summary["panel"] == "F",
                            ["group", "mean", "sem"]].to_string(index=False))
    print(tests.loc[tests["panel"] == "F",
                    ["contrast", "mean_difference_a_minus_b", "t", "p_raw",
                     "q_bh_within_panel_family"]].to_string(index=False))


if __name__ == "__main__":
    main()
