"""Compute initial-exposure source data and network-level statistics.

This entry point is intentionally analysis-only.  It reads the saved
production metrics, writes tidy source-data tables, and performs inference with
the independently initialized network as ``n``.  Figure rendering lives in
separate manuscript plotting code.

Aggregation order
-----------------
1. finite cells within each trajectory;
2. the two trajectory means within each network;
3. summaries and paired/one-sample tests across network means.

The script exports both phase estimands required by the v4 handoff:

* raw absolute boundary phase ratio:
  ``2 * current absolute phase shift / matched baseline grid spacing``;
* baseline-subtracted phase change:
  ``2 * (current phase shift - baseline phase shift) / baseline spacing``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from initial_exposure_metrics import (
    ENVIRONMENTS,
    GRID_METRICS,
    PLACE_METRICS,
    _summarize_population,
    discover_networks,
    load_network_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    / "network_contrasts"
)

PHASE_RAW = ("phase_ratio_x_half_period", "phase_ratio_y_half_period")
PHASE_DELTA = (
    "delta_phase_ratio_x_half_period",
    "delta_phase_ratio_y_half_period",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fdr_bh(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment while preserving missing values/index."""
    array = values.to_numpy(dtype=float)
    adjusted = np.full(array.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    if finite.size == 0:
        return pd.Series(adjusted, index=values.index)
    order = finite[np.argsort(array[finite])]
    ranked = array[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    adjusted[order] = np.minimum(corrected, 1.0)
    return pd.Series(adjusted, index=values.index)


def _difference_test(values: Iterable[float]) -> dict[str, float | int]:
    """Two-sided tests of network-level paired differences against zero."""
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = sd / math.sqrt(n) if n > 1 else np.nan
    ci_half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan

    if n > 1 and sd > 0:
        t_statistic, p_ttest = stats.ttest_1samp(values, popmean=0.0)
        cohens_dz = mean / sd
    elif n and np.allclose(values, 0.0):
        t_statistic, p_ttest, cohens_dz = 0.0, 1.0, 0.0
    else:
        t_statistic = p_ttest = cohens_dz = np.nan

    try:
        if n == 0:
            wilcoxon_statistic = p_wilcoxon = np.nan
        elif np.allclose(values, 0.0):
            wilcoxon_statistic, p_wilcoxon = 0.0, 1.0
        else:
            wilcoxon_statistic, p_wilcoxon = stats.wilcoxon(
                values, alternative="two-sided"
            )
    except ValueError:
        wilcoxon_statistic = p_wilcoxon = np.nan

    return {
        "n_networks": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "sem_difference": sem,
        "ci95_low": mean - ci_half if np.isfinite(ci_half) else np.nan,
        "ci95_high": mean + ci_half if np.isfinite(ci_half) else np.nan,
        "cohens_dz": float(cohens_dz),
        "t_statistic": float(t_statistic),
        "t_degrees_of_freedom": n - 1 if n > 1 else np.nan,
        "p_ttest_two_sided": float(p_ttest),
        "wilcoxon_statistic": float(wilcoxon_statistic),
        "p_wilcoxon_two_sided": float(p_wilcoxon),
    }


def _append_contrast(
    rows: list[dict[str, object]],
    *,
    family: str,
    contrast_id: str,
    population: str,
    metric: str,
    comparison: str,
    module: int,
    environment: str,
    length_cm: float,
    deformation_ratio: float,
    values: pd.DataFrame,
) -> None:
    """Append one row per network for an already aligned paired contrast."""
    required = {"network_id", "value_a", "value_b"}
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"Contrast {contrast_id} lacks columns {sorted(missing)}")
    for row in values.itertuples(index=False):
        value_a = float(row.value_a)
        value_b = float(row.value_b)
        difference = value_a - value_b
        rows.append(
            {
                "family": family,
                "contrast_id": contrast_id,
                "population": population,
                "metric": metric,
                "comparison": comparison,
                "network_id": int(row.network_id),
                "environment": environment,
                "length_cm": length_cm,
                "deformation_ratio": deformation_ratio,
                "module": int(module),
                "value_a": value_a,
                "value_b": value_b,
                "difference": difference,
            }
        )


def build_network_contrasts(network: pd.DataFrame) -> pd.DataFrame:
    """Build all Fig. 2/3 contrasts from network-level values only."""
    rows: list[dict[str, object]] = []

    # Fig. 2: module-specific rescaling against two prespecified references.
    rescale = network.loc[
        network["population"].eq("grid")
        & network["metric"].eq("rescale_factor")
        & network["length_cm"].ne(100)
    ]
    for (environment, length_cm, ratio, module), subset in rescale.groupby(
        ["environment", "length_cm", "deformation_ratio", "module"], sort=True
    ):
        base = subset[["network_id", "network_mean"]].rename(
            columns={"network_mean": "value_a"}
        )
        for family, label, reference in (
            (
                "fig2_grid_rescale_vs_no_rescaling_by_condition",
                "network mean rescale factor minus 1",
                1.0,
            ),
            (
                "fig2_grid_rescale_vs_geometric_by_condition",
                "network mean rescale factor minus deformation ratio",
                float(ratio),
            ),
        ):
            values = base.assign(value_b=reference)
            _append_contrast(
                rows,
                family=family,
                contrast_id=f"{environment}_module_{int(module)}",
                population="grid",
                metric="rescale_factor",
                comparison=label,
                module=int(module),
                environment=str(environment),
                length_cm=float(length_cm),
                deformation_ratio=float(ratio),
                values=values,
            )

    # Fig. 2: place-field length and opposing-direction displacement changes.
    place_metrics = (
        "delta_field_length_x_cm",
        "delta_field_length_y_cm",
        "delta_place_phase_x_cm",
        "delta_place_phase_y_cm",
    )
    place = network.loc[
        network["population"].eq("place")
        & network["metric"].isin(place_metrics)
        & network["length_cm"].ne(100)
    ]
    for (metric, environment, length_cm, ratio), subset in place.groupby(
        ["metric", "environment", "length_cm", "deformation_ratio"], sort=True
    ):
        values = subset[["network_id", "network_mean"]].rename(
            columns={"network_mean": "value_a"}
        ).assign(value_b=0.0)
        _append_contrast(
            rows,
            family="fig2_place_change_vs_zero_by_condition",
            contrast_id=f"{metric}_{environment}",
            population="place",
            metric=str(metric),
            comparison="deformed minus matched familiar value",
            module=0,
            environment=str(environment),
            length_cm=float(length_cm),
            deformation_ratio=float(ratio),
            values=values,
        )

    phase_keys = [
        "network_id",
        "environment",
        "length_cm",
        "deformation_ratio",
        "module",
    ]

    # Fig. 3: raw absolute x-y phase contrast at every displayed condition.
    raw = network.loc[
        network["population"].eq("grid") & network["metric"].isin(PHASE_RAW)
    ].pivot(index=phase_keys, columns="metric", values="network_mean").reset_index()
    for (environment, length_cm, ratio, module), subset in raw.groupby(
        ["environment", "length_cm", "deformation_ratio", "module"], sort=True
    ):
        values = subset[
            ["network_id", PHASE_RAW[0], PHASE_RAW[1]]
        ].rename(columns={PHASE_RAW[0]: "value_a", PHASE_RAW[1]: "value_b"})
        _append_contrast(
            rows,
            family="fig3_raw_phase_x_vs_y_by_condition",
            contrast_id=f"{environment}_module_{int(module)}",
            population="grid",
            metric="raw_phase_ratio_x_minus_y",
            comparison="absolute x-axis minus absolute y-axis boundary phase ratio",
            module=int(module),
            environment=str(environment),
            length_cm=float(length_cm),
            deformation_ratio=float(ratio),
            values=values,
        )

    # Fig. 3: baseline-subtracted phase changes, both vs zero and x vs y.
    delta = network.loc[
        network["population"].eq("grid")
        & network["metric"].isin(PHASE_DELTA)
        & network["length_cm"].ne(100)
    ].pivot(index=phase_keys, columns="metric", values="network_mean").reset_index()
    for (environment, length_cm, ratio, module), subset in delta.groupby(
        ["environment", "length_cm", "deformation_ratio", "module"], sort=True
    ):
        paired = subset[
            ["network_id", PHASE_DELTA[0], PHASE_DELTA[1]]
        ].rename(columns={PHASE_DELTA[0]: "value_a", PHASE_DELTA[1]: "value_b"})
        _append_contrast(
            rows,
            family="fig3_delta_phase_x_vs_y_by_condition",
            contrast_id=f"{environment}_module_{int(module)}",
            population="grid",
            metric="delta_phase_ratio_x_minus_y",
            comparison="baseline-subtracted x-axis minus y-axis phase change",
            module=int(module),
            environment=str(environment),
            length_cm=float(length_cm),
            deformation_ratio=float(ratio),
            values=paired,
        )
        for metric in PHASE_DELTA:
            values = subset[["network_id", metric]].rename(
                columns={metric: "value_a"}
            ).assign(value_b=0.0)
            _append_contrast(
                rows,
                family="fig3_delta_phase_vs_zero_by_condition",
                contrast_id=f"{metric}_{environment}_module_{int(module)}",
                population="grid",
                metric=metric,
                comparison="baseline-subtracted phase change minus zero",
                module=int(module),
                environment=str(environment),
                length_cm=float(length_cm),
                deformation_ratio=float(ratio),
                values=values,
            )

    # Fig. 3 collapsed tests reproduce the intended old panel comparisons but
    # average environments within network before inference (n=8, not cells).
    deformed_raw = raw.loc[raw["length_cm"].ne(100)].copy()
    collapsed = (
        deformed_raw.groupby(["network_id", "module"], sort=True)[list(PHASE_RAW)]
        .mean()
        .reset_index()
    )
    familiar = raw.loc[raw["length_cm"].eq(100), ["network_id", "module", PHASE_RAW[0]]]
    familiar = familiar.rename(columns={PHASE_RAW[0]: "familiar_x"})
    collapsed = collapsed.merge(
        familiar, on=["network_id", "module"], how="inner", validate="one_to_one"
    )
    for module, subset in collapsed.groupby("module", sort=True):
        familiar_values = subset[
            ["network_id", PHASE_RAW[0], "familiar_x"]
        ].rename(columns={PHASE_RAW[0]: "value_a", "familiar_x": "value_b"})
        _append_contrast(
            rows,
            family="fig3_raw_phase_deformed_x_vs_familiar_x_collapsed",
            contrast_id=f"module_{int(module)}",
            population="grid",
            metric="mean_deformed_raw_phase_x_minus_familiar_x",
            comparison="mean across six deformed environments minus familiar x-axis",
            module=int(module),
            environment="six_deformed_environments_collapsed_within_network",
            length_cm=np.nan,
            deformation_ratio=np.nan,
            values=familiar_values,
        )
        axis_values = subset[
            ["network_id", PHASE_RAW[0], PHASE_RAW[1]]
        ].rename(columns={PHASE_RAW[0]: "value_a", PHASE_RAW[1]: "value_b"})
        _append_contrast(
            rows,
            family="fig3_raw_phase_deformed_x_vs_y_collapsed",
            contrast_id=f"module_{int(module)}",
            population="grid",
            metric="mean_deformed_raw_phase_x_minus_y",
            comparison="mean x-axis minus y-axis across six deformed environments",
            module=int(module),
            environment="six_deformed_environments_collapsed_within_network",
            length_cm=np.nan,
            deformation_ratio=np.nan,
            values=axis_values,
        )

    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        raise ValueError("No network-level contrasts were generated")
    return contrasts.sort_values(
        ["family", "module", "length_cm", "network_id"], na_position="last"
    ).reset_index(drop=True)


def summarize_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize contrast rows and correct tests within declared families."""
    keys = [
        "family",
        "contrast_id",
        "population",
        "metric",
        "comparison",
        "environment",
        "length_cm",
        "deformation_ratio",
        "module",
    ]
    rows: list[dict[str, object]] = []
    for key, subset in contrasts.groupby(keys, sort=True, dropna=False):
        rows.append(
            dict(zip(keys, key))
            | _difference_test(subset["difference"].to_numpy(float))
        )
    result = pd.DataFrame(rows)
    for column in ("p_ttest_two_sided", "p_wilcoxon_two_sided"):
        result[f"{column}_fdr_bh_within_family"] = (
            result.groupby("family", sort=False, group_keys=False)[column]
            .apply(_fdr_bh)
        )
    return result.sort_values(
        ["family", "module", "length_cm"], na_position="last"
    ).reset_index(drop=True)


def _parameter_table(config: dict[str, object]) -> pd.DataFrame:
    analysis = dict(config.get("analysis", {}))
    rows = [
        (
            "profile",
            "analysis_profile",
            "production Methods metrics saved with the v4 runs",
            "Do not label as the original-code profile",
        ),
        (
            "aggregation",
            "independent_unit",
            "network",
            "Cells then trajectories are averaged before group inference",
        ),
        (
            "grid rescaling",
            "search",
            f"range={analysis.get('rescale_scale_range_cm')} cm; step={analysis.get('rescale_scale_step_cm')} cm; mode={analysis.get('rescale_mode')}",
            "Saved comparison_metrics values",
        ),
        (
            "grid phase",
            "raw absolute estimand",
            "2 * current absolute boundary phase shift / matched baseline cell spacing",
            "Not baseline-subtracted",
        ),
        (
            "grid phase",
            "change estimand",
            "2 * (current phase shift - baseline phase shift) / matched baseline cell spacing",
            "May be negative",
        ),
        (
            "grid phase",
            "boundary/bootstrap",
            f"threshold={analysis.get('boundary_threshold_cm')} cm; x={dict(analysis.get('phase_bootstrap', {})).get('grid_x')}; y={dict(analysis.get('phase_bootstrap', {})).get('grid_y')}",
            "Both axes retained",
        ),
        (
            "place field",
            "definition",
            f"smooth={analysis.get('place_field_rate_map_sigma_bins')} bins; peak fraction={analysis.get('place_field_peak_fraction')}; min rate={analysis.get('place_min_rate_hz')} Hz",
            "Longest suprathreshold run in x/y",
        ),
        (
            "place phase",
            "definition",
            f"opposing-movement displacement; min speed={analysis.get('movement_min_speed_cm_s')} cm/s",
            "Current minus matched baseline is also exported",
        ),
        (
            "statistics",
            "primary test",
            "two-sided one-sample t-test of network-level paired differences",
            "Wilcoxon signed-rank is exported as a sensitivity analysis",
        ),
        (
            "statistics",
            "multiplicity",
            "Benjamini-Hochberg within each named comparison family",
            "Exact raw and adjusted p values are exported",
        ),
    ]
    return pd.DataFrame(rows, columns=["analysis", "parameter", "value", "note"])


def _write_readme(output: Path, network_ids: list[int]) -> None:
    text = f"""# v4 initial-exposure Methods-profile analysis

This directory contains the Fig. 2/3 source data and statistics for networks
{', '.join(f'{value:03d}' for value in network_ids)} (`n={len(network_ids)}`).

The independent unit for all confidence intervals and hypothesis tests is the
initialized network.  Finite cells are averaged within each trajectory, the two
trajectory means are averaged within network, and only then are networks
summarized or tested.  Missing values are not imputed.

`metrics/network_level_contrasts.csv` contains every independent value entering
the tests.  `metrics/network_level_statistics.csv` reports effect sizes, 95%
t-based confidence intervals, two-sided one-sample t-tests, Wilcoxon sensitivity
tests, and Benjamini-Hochberg-adjusted p values within each named family.

Raw absolute and baseline-subtracted grid phase are separate estimands throughout.
This analysis uses the production Methods profile saved with the v4 simulations;
it must not be described as the original-code profile.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_outputs(
    *,
    output: Path,
    config_path: Path,
    data_root: Path,
    networks: list[tuple[int, Path]],
    grid_cells: pd.DataFrame,
    place_cells: pd.DataFrame,
    trajectory: pd.DataFrame,
    network: pd.DataFrame,
    group: pd.DataFrame,
    contrasts: pd.DataFrame,
    statistics: pd.DataFrame,
    source_audit: list[dict[str, object]],
) -> None:
    metrics = output / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)

    grid_cells.to_csv(
        metrics / "grid_source_data_per_cell_trajectory.csv.gz",
        index=False,
        compression="gzip",
    )
    place_cells.to_csv(
        metrics / "place_source_data_per_cell_trajectory.csv.gz",
        index=False,
        compression="gzip",
    )
    trajectory.to_csv(
        metrics / "trajectory_level_values.csv.gz", index=False, compression="gzip"
    )
    network.to_csv(metrics / "network_level_values.csv", index=False)
    group.to_csv(metrics / "group_summary_network_unit.csv", index=False)
    contrasts.to_csv(metrics / "network_level_contrasts.csv", index=False)
    statistics.to_csv(metrics / "network_level_statistics.csv", index=False)
    statistics.loc[statistics["family"].str.startswith("fig3_")].to_csv(
        metrics / "fig3_network_level_statistics.csv", index=False
    )
    contrasts.loc[
        contrasts["family"].isin(
            {
                "fig3_raw_phase_deformed_x_vs_familiar_x_collapsed",
                "fig3_raw_phase_deformed_x_vs_y_collapsed",
            }
        )
    ].to_csv(metrics / "fig3_collapsed_network_contrasts.csv", index=False)

    finite_cells = trajectory[
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
    ].copy()
    finite_cells.to_csv(metrics / "finite_cell_counts.csv", index=False)
    finite_networks = group[
        [
            "population",
            "environment",
            "length_cm",
            "module",
            "metric",
            "n_networks",
        ]
    ].copy()
    finite_networks.to_csv(metrics / "finite_network_counts.csv", index=False)

    audit = pd.DataFrame(source_audit)
    audit.to_csv(metrics / "input_file_audit.csv", index=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _parameter_table(config).to_csv(metrics / "analysis_parameters.csv", index=False)

    network_ids = [network_id for network_id, _ in networks]
    expected_sessions = len(networks) * len(ENVIRONMENTS) * 2
    if len(source_audit) != expected_sessions:
        raise ValueError(
            f"Expected {expected_sessions} run records, observed {len(source_audit)}"
        )
    if int(finite_networks["n_networks"].min()) < len(networks):
        # Missing metric values are allowed, but the audit must make the loss visible.
        finite_warning = "Some metric/condition summaries contain fewer finite networks; see finite_network_counts.csv."
    else:
        finite_warning = "All metric/condition summaries contain all selected networks."

    code_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "local_network_analysis" / "initial_exposure_metrics.py",
        PROJECT_ROOT / "cluster_pipeline" / "review_pipeline" / "metrics.py",
        PROJECT_ROOT / "main_run_prev" / "analysis" / "phase_shift_grid.py",
        config_path,
    ]
    provenance = {
        "analysis_profile": "v4 production Methods metrics",
        "data_root": str(data_root),
        "config": str(config_path),
        "network_ids": network_ids,
        "n_networks": len(network_ids),
        "n_trajectories_per_condition": 2,
        "n_run_records_loaded": len(source_audit),
        "independent_unit": "initialized network",
        "aggregation_order": (
            "finite cells within trajectory; two trajectory means within network; "
            "group summaries/tests across network means"
        ),
        "uncertainty": "95% t confidence interval across network means",
        "primary_test": "two-sided one-sample t-test of network-level paired differences",
        "sensitivity_test": "two-sided Wilcoxon signed-rank test of network-level paired differences",
        "multiple_comparisons": "Benjamini-Hochberg within each named family",
        "phase_raw_definition": (
            "2 * current absolute opposing-boundary phase shift / matched baseline cell spacing"
        ),
        "phase_change_definition": (
            "2 * (current phase shift - baseline phase shift) / matched baseline cell spacing"
        ),
        "place_phase_definition": (
            "opposing-movement displacement in cm; change is current minus matched baseline"
        ),
        "missing_value_policy": "No imputation; finite cell and network counts are exported.",
        "finite_value_audit": finite_warning,
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in code_paths
            if path.is_file()
        },
    }
    (output / "analysis_qa_and_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    _write_readme(output, network_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--network-ids",
        type=int,
        nargs="*",
        default=list(range(8)),
        help="Independent initialized networks to include (default: 0..7).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("n_networks", -1)) != 8:
        raise ValueError(f"Expected n_networks=8 in {config_path}")

    networks = discover_networks(data_root, args.network_ids)
    grid_frames: list[pd.DataFrame] = []
    place_frames: list[pd.DataFrame] = []
    source_audit: list[dict[str, object]] = []
    for network_id, network_root in networks:
        print(f"Loading network {network_id:03d} from {network_root}", flush=True)
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
    trajectory = pd.concat([grid_trajectory, place_trajectory], ignore_index=True)
    network = pd.concat([grid_network, place_network], ignore_index=True)
    group = pd.concat([grid_group, place_group], ignore_index=True)

    contrasts = build_network_contrasts(network)
    statistics = summarize_contrasts(contrasts)
    write_outputs(
        output=output,
        config_path=config_path,
        data_root=data_root,
        networks=networks,
        grid_cells=grid_cells,
        place_cells=place_cells,
        trajectory=trajectory,
        network=network,
        group=group,
        contrasts=contrasts,
        statistics=statistics,
        source_audit=source_audit,
    )
    print(f"Analysis complete: {output}", flush=True)
    print(f"Networks: {[network_id for network_id, _ in networks]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
