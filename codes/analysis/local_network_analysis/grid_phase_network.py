"""Prepare network-level source data and statistics for Figure 3.

The script deliberately separates analysis from plotting.  It reuses the
Methods-version, baseline-matched phase table and reads the saved per-cell
metrics only for grid-field length and firing-rate panels.

Inference hierarchy
-------------------
cell -> trajectory -> condition/module -> within-network collapse -> networks

Run from the repository root with::

    python codes/analysis/local_network_analysis/grid_phase_network.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "run" / "outputs" / "independent_networks" / "networks"
PHASE_SOURCE = (
    ROOT
    / "outputs"
    / "network_contrasts"
    / "metrics"
    / "network_level_values.csv"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "grid_phase_network"
)

NETWORK_IDS = list(range(8))
TRAJECTORIES = [0, 1]
MODULES = [1, 2, 3]
ENVIRONMENTS = [
    ("compressed_70", 70, 0.70),
    ("compressed_80", 80, 0.80),
    ("compressed_90", 90, 0.90),
    ("fam_100", 100, 1.00),
    ("elongated_110", 110, 1.10),
    ("elongated_120", 120, 1.20),
    ("elongated_130", 130, 1.30),
]
DEFORMED = [name for name, length, _ in ENVIRONMENTS if length != 100]
PROPERTY_COLUMNS = [
    "field_length_x_cm",
    "field_length_y_cm",
    "peak_rate_hz",
    "mean_rate_hz",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sem(values: pd.Series) -> float:
    x = values.dropna().to_numpy(float)
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan


def _summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in frame.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x = group["value"].dropna().to_numpy(float)
        n = x.size
        mean = float(np.mean(x)) if n else np.nan
        sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        row = dict(zip(group_cols, keys))
        row.update(
            n_networks=n,
            mean=mean,
            sd=sd,
            sem=sem,
            ci95_low=mean - tcrit * sem if n > 1 else np.nan,
            ci95_high=mean + tcrit * sem if n > 1 else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _bh_adjust(pvalues: pd.Series) -> np.ndarray:
    p = pvalues.to_numpy(float)
    result = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result


def _paired_test(
    panel: str,
    family: str,
    contrast: str,
    a_label: str,
    b_label: str,
    a: pd.Series,
    b: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    paired = pd.concat([a.rename("value_a"), b.rename("value_b")], axis=1).dropna()
    paired.index.name = "network_id"
    paired = paired.reset_index()
    paired["difference_a_minus_b"] = paired["value_a"] - paired["value_b"]
    paired.insert(0, "comparison_b", b_label)
    paired.insert(0, "comparison_a", a_label)
    paired.insert(0, "contrast", contrast)
    paired.insert(0, "family", family)
    paired.insert(0, "panel", panel)

    diff = paired["difference_a_minus_b"].to_numpy(float)
    n = diff.size
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    sem_diff = sd_diff / np.sqrt(n) if n > 1 else np.nan
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    t_result = stats.ttest_1samp(diff, 0.0, nan_policy="omit") if n > 1 else None
    try:
        w_result = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        w_stat = float(w_result.statistic)
        w_p = float(w_result.pvalue)
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    row = {
        "panel": panel,
        "family": family,
        "contrast": contrast,
        "comparison_a": a_label,
        "comparison_b": b_label,
        "n_networks": n,
        "mean_difference_a_minus_b": mean_diff,
        "sd_difference": sd_diff,
        "sem_difference": sem_diff,
        "ci95_low": mean_diff - tcrit * sem_diff if n > 1 else np.nan,
        "ci95_high": mean_diff + tcrit * sem_diff if n > 1 else np.nan,
        "cohen_dz": mean_diff / sd_diff if n > 1 and sd_diff > 0 else np.nan,
        "t": float(t_result.statistic) if t_result is not None else np.nan,
        "df": n - 1,
        "p_raw": float(t_result.pvalue) if t_result is not None else np.nan,
        "wilcoxon_w": w_stat,
        "wilcoxon_p": w_p,
    }
    return row, paired


def load_phase_network_values() -> pd.DataFrame:
    phase = pd.read_csv(PHASE_SOURCE)
    metrics = {
        "delta_phase_ratio_x_half_period": "x",
        "delta_phase_ratio_y_half_period": "y",
    }
    phase = phase.loc[
        (phase["population"] == "grid") & phase["metric"].isin(metrics),
        [
            "network_id",
            "environment",
            "length_cm",
            "deformation_ratio",
            "module",
            "metric",
            "network_mean",
            "n_trajectories_finite",
        ],
    ].copy()
    phase["axis"] = phase["metric"].map(metrics)
    phase = phase.rename(columns={"network_mean": "value"})
    phase["network_id"] = phase["network_id"].astype(int)
    phase["module"] = phase["module"].astype(int)
    expected = len(NETWORK_IDS) * len(ENVIRONMENTS) * len(MODULES) * 2
    if len(phase) != expected:
        raise RuntimeError(f"Expected {expected} phase rows, found {len(phase)}")
    if set(phase["network_id"]) != set(NETWORK_IDS):
        raise RuntimeError("Phase source does not contain exactly networks 0-7")
    if not (phase["n_trajectories_finite"] == 2).all():
        raise RuntimeError("Every phase network mean must contain both trajectories")
    baseline = phase.loc[phase["length_cm"] == 100, "value"].to_numpy(float)
    if not np.allclose(baseline, 0.0, atol=1e-12, equal_nan=False):
        raise RuntimeError("Baseline-subtracted familiar phase values are not zero")
    return phase.sort_values(["module", "axis", "length_cm", "network_id"]).reset_index(drop=True)


def load_grid_properties() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    cell_frames: list[pd.DataFrame] = []
    trajectory_rows: list[dict] = []
    input_records: list[dict] = []
    for network_id in NETWORK_IDS:
        for environment, length_cm, ratio in ENVIRONMENTS:
            for trajectory_id in TRAJECTORIES:
                network_root = DATA_ROOT / f"network_{network_id:03d}"
                # Familiar runs are stored in the network-level baseline branch;
                # deformed runs are stored below initial_exposure.
                condition_root = network_root / ("baseline" if environment == "fam_100" else "initial_exposure")
                path = (
                    condition_root
                    / environment
                    / f"trajectory_{trajectory_id:02d}"
                    / "cell_metrics.csv.gz"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                usecols = ["population", "module", "cell_id", *PROPERTY_COLUMNS]
                raw = pd.read_csv(path, usecols=usecols)
                grid = raw.loc[raw["population"] == "grid"].copy()
                if len(grid) != 300 or set(grid["module"].dropna().astype(int)) != set(MODULES):
                    raise RuntimeError(f"Unexpected grid population in {path}")
                grid["network_id"] = network_id
                grid["environment"] = environment
                grid["length_cm"] = length_cm
                grid["deformation_ratio"] = ratio
                grid["trajectory_id"] = trajectory_id
                grid["module"] = grid["module"].astype(int)
                cell_frames.append(
                    grid[
                        [
                            "network_id",
                            "environment",
                            "length_cm",
                            "deformation_ratio",
                            "trajectory_id",
                            "module",
                            "cell_id",
                            *PROPERTY_COLUMNS,
                        ]
                    ]
                )
                input_records.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
                for module, module_frame in grid.groupby("module", sort=True):
                    row = {
                        "network_id": network_id,
                        "environment": environment,
                        "length_cm": length_cm,
                        "deformation_ratio": ratio,
                        "trajectory_id": trajectory_id,
                        "module": int(module),
                    }
                    for metric in PROPERTY_COLUMNS:
                        x = pd.to_numeric(module_frame[metric], errors="coerce")
                        row[metric] = float(x.mean())
                        row[f"n_finite_{metric}"] = int(x.notna().sum())
                    trajectory_rows.append(row)

    cells = pd.concat(cell_frames, ignore_index=True)
    trajectory = pd.DataFrame(trajectory_rows)
    network_condition_module = (
        trajectory.groupby(
            ["network_id", "environment", "length_cm", "deformation_ratio", "module"],
            as_index=False,
            sort=True,
        )[PROPERTY_COLUMNS]
        .mean()
    )
    counts = (
        trajectory.groupby(
            ["network_id", "environment", "length_cm", "deformation_ratio", "module"],
            as_index=False,
            sort=True,
        )["trajectory_id"]
        .nunique()
        .rename(columns={"trajectory_id": "n_trajectories"})
    )
    network_condition_module = network_condition_module.merge(
        counts,
        on=["network_id", "environment", "length_cm", "deformation_ratio", "module"],
        validate="one_to_one",
    )
    if not (network_condition_module["n_trajectories"] == 2).all():
        raise RuntimeError("Every grid-property network mean must contain both trajectories")
    return cells, trajectory, network_condition_module, input_records


def build_panel_values(
    phase: pd.DataFrame, grid_props: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # A-C: each module and axis across all seven environments, n=8 networks.
    abc = phase[
        [
            "network_id",
            "environment",
            "length_cm",
            "deformation_ratio",
            "module",
            "axis",
            "value",
        ]
    ].copy()

    # D: each network first averages all six deformed environments.
    d = (
        phase.loc[phase["length_cm"] != 100]
        .groupby(["network_id", "module", "axis"], as_index=False, sort=True)["value"]
        .mean()
    )
    d["panel"] = "D"
    d["measure"] = "delta_R_phase"
    d["group"] = d["axis"].map({"x": "Deformed axis (x)", "y": "Undeformed axis (y)"})

    # E: equal-weight the three module-level means within each network.
    e = d.groupby(["network_id", "axis"], as_index=False, sort=True)["value"].mean()
    e["module"] = np.nan
    e["panel"] = "E"
    e["measure"] = "delta_R_phase"
    e["group"] = e["axis"].map({"x": "Deformed axis (x)", "y": "Undeformed axis (y)"})

    familiar = grid_props.loc[grid_props["length_cm"] == 100]
    deformed = grid_props.loc[grid_props["length_cm"] != 100]

    # F: the user explicitly requested familiar x as the third comparison group.
    f_parts = []
    for label, source, metric in [
        ("Familiar x", familiar, "field_length_x_cm"),
        ("Deformed x", deformed, "field_length_x_cm"),
        ("Deformed y", deformed, "field_length_y_cm"),
    ]:
        part = source.groupby("network_id", as_index=False, sort=True)[metric].mean()
        part = part.rename(columns={metric: "value"})
        part["panel"] = "F"
        part["measure"] = "grid_field_length_cm"
        part["group"] = label
        part["module"] = np.nan
        part["axis"] = {"Familiar x": "x", "Deformed x": "x", "Deformed y": "y"}[label]
        f_parts.append(part)
    f = pd.concat(f_parts, ignore_index=True)

    # G: one familiar value and one six-environment deformation value per network.
    g_parts = []
    for measure, metric in [("peak_rate_hz", "peak_rate_hz"), ("mean_rate_hz", "mean_rate_hz")]:
        for label, source in [("Familiar", familiar), ("Deformation", deformed)]:
            part = source.groupby("network_id", as_index=False, sort=True)[metric].mean()
            part = part.rename(columns={metric: "value"})
            part["panel"] = "G"
            part["measure"] = measure
            part["group"] = label
            part["module"] = np.nan
            part["axis"] = np.nan
            g_parts.append(part)
    g = pd.concat(g_parts, ignore_index=True)

    panel_values = pd.concat(
        [
            d[["panel", "measure", "group", "module", "axis", "network_id", "value"]],
            e[["panel", "measure", "group", "module", "axis", "network_id", "value"]],
            f[["panel", "measure", "group", "module", "axis", "network_id", "value"]],
            g[["panel", "measure", "group", "module", "axis", "network_id", "value"]],
        ],
        ignore_index=True,
    )
    return abc, panel_values, d


def calculate_statistics(panel_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests: list[dict] = []
    paired_rows: list[pd.DataFrame] = []

    # D: separate x-y contrast in each module; one BH family of three tests.
    for module in MODULES:
        sub = panel_values.loc[(panel_values["panel"] == "D") & (panel_values["module"] == module)]
        wide = sub.pivot(index="network_id", columns="group", values="value")
        row, pairs = _paired_test(
            "D",
            "D_modules_x_vs_y",
            f"module_{module}_x_minus_y",
            "Deformed axis (x)",
            "Undeformed axis (y)",
            wide["Deformed axis (x)"],
            wide["Undeformed axis (y)"],
        )
        row["module"] = module
        pairs["module"] = module
        tests.append(row)
        paired_rows.append(pairs)

    # E: one combined-module x-y contrast.
    sub = panel_values.loc[panel_values["panel"] == "E"]
    wide = sub.pivot(index="network_id", columns="group", values="value")
    row, pairs = _paired_test(
        "E",
        "E_combined_x_vs_y",
        "combined_modules_x_minus_y",
        "Deformed axis (x)",
        "Undeformed axis (y)",
        wide["Deformed axis (x)"],
        wide["Undeformed axis (y)"],
    )
    row["module"] = np.nan
    pairs["module"] = np.nan
    tests.append(row)
    paired_rows.append(pairs)

    # F: retain the two prespecified Xu-style comparisons and correct them together.
    sub = panel_values.loc[panel_values["panel"] == "F"]
    wide = sub.pivot(index="network_id", columns="group", values="value")
    for contrast, a_label, b_label in [
        ("deformed_x_minus_familiar_x", "Deformed x", "Familiar x"),
        ("deformed_x_minus_deformed_y", "Deformed x", "Deformed y"),
    ]:
        row, pairs = _paired_test(
            "F", "F_grid_field_length", contrast, a_label, b_label, wide[a_label], wide[b_label]
        )
        row["module"] = np.nan
        pairs["module"] = np.nan
        tests.append(row)
        paired_rows.append(pairs)

    # G: peak and mean rate are the two tests in the G-panel family.
    for measure in ["peak_rate_hz", "mean_rate_hz"]:
        sub = panel_values.loc[
            (panel_values["panel"] == "G") & (panel_values["measure"] == measure)
        ]
        wide = sub.pivot(index="network_id", columns="group", values="value")
        row, pairs = _paired_test(
            "G",
            "G_firing_rates",
            f"{measure}_deformation_minus_familiar",
            "Deformation",
            "Familiar",
            wide["Deformation"],
            wide["Familiar"],
        )
        row["module"] = np.nan
        pairs["module"] = np.nan
        tests.append(row)
        paired_rows.append(pairs)

    result = pd.DataFrame(tests)
    result["q_bh_within_panel_family"] = np.nan
    for _, idx in result.groupby("family", sort=False).groups.items():
        result.loc[idx, "q_bh_within_panel_family"] = _bh_adjust(result.loc[idx, "p_raw"])
    result["significance"] = pd.cut(
        result["q_bh_within_panel_family"],
        bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
        labels=["***", "**", "*", "ns"],
        right=False,
    ).astype(str)
    paired = pd.concat(paired_rows, ignore_index=True)
    return result, paired


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase = load_phase_network_values()
    cells, trajectory, grid_props, input_records = load_grid_properties()
    abc, panel_values, _ = build_panel_values(phase, grid_props)
    tests, paired = calculate_statistics(panel_values)

    abc_summary = _summary(
        abc, ["module", "axis", "environment", "length_cm", "deformation_ratio"]
    )
    panel_summary = _summary(panel_values, ["panel", "measure", "module", "axis", "group"])

    # Strong invariants: all inferential comparisons include every selected network.
    if not (tests["n_networks"] == len(NETWORK_IDS)).all():
        raise RuntimeError("At least one inferential comparison lacks networks")
    if not (abc_summary["n_networks"] == len(NETWORK_IDS)).all():
        raise RuntimeError("At least one A-C point lacks networks")

    cells.to_csv(OUTPUT / "grid_property_cell_trajectory.csv.gz", index=False, compression="gzip")
    trajectory.to_csv(OUTPUT / "grid_property_trajectory_summary.csv", index=False)
    grid_props.to_csv(OUTPUT / "grid_property_network_condition_module.csv", index=False)
    abc.to_csv(OUTPUT / "panels_ABC_network_values.csv", index=False)
    abc_summary.to_csv(OUTPUT / "panels_ABC_group_summary.csv", index=False)
    panel_values.to_csv(OUTPUT / "panels_DEFG_network_values.csv", index=False)
    panel_summary.to_csv(OUTPUT / "panels_DEFG_group_summary.csv", index=False)
    tests.to_csv(OUTPUT / "network_level_statistics.csv", index=False)
    paired.to_csv(OUTPUT / "network_level_paired_values.csv", index=False)

    phase_record = {
        "path": str(PHASE_SOURCE.relative_to(ROOT)),
        "size_bytes": PHASE_SOURCE.stat().st_size,
        "sha256": _sha256(PHASE_SOURCE),
    }
    provenance = {
        "analysis": "Revised V4 Figure 3, Methods phase, network-unit inference",
        "phase_definition": (
            "delta R_phase = 2 * (current absolute boundary phase - matched baseline "
            "absolute boundary phase) / matched baseline grid spacing"
        ),
        "aggregation": {
            "ABC": "cell -> trajectory -> network; plotted across n=8 networks",
            "D": "within each network/module/axis, equal mean over six deformed environments",
            "E": "within each network/axis, equal mean over six environments and three modules",
            "F": (
                "cell -> trajectory -> network-condition-module; familiar x averages modules; "
                "deformed x/y average six environments and three modules within network"
            ),
            "G": (
                "cell -> trajectory -> network-condition-module; familiar averages modules; "
                "deformation averages six environments and three modules within network"
            ),
        },
        "statistics": {
            "primary": "two-sided paired t-test across independent network means",
            "sensitivity": "two-sided Wilcoxon signed-rank test",
            "multiplicity": "Benjamini-Hochberg correction separately within D, E, F, and G families",
            "outlier_exclusion": "none",
            "significance_thresholds": {"***": "q < 0.001", "**": "q < 0.01", "*": "q < 0.05", "ns": "q >= 0.05"},
        },
        "phase_input": phase_record,
        "raw_property_inputs": input_records,
    }
    (OUTPUT / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote revised Figure 3 analysis to: {OUTPUT}")
    print(tests[["panel", "contrast", "n_networks", "p_raw", "q_bh_within_panel_family", "significance"]].to_string(index=False))


if __name__ == "__main__":
    main()
