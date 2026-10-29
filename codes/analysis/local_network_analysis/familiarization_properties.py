"""Independent-network familiarization source data for Fig. 5 and SI Fig. 3.

Only saved production/Methods metrics are read. Session 0 is initial exposure
trajectory 00; sessions 1-9 are intact-plasticity familiarization runs. Every
change/ratio is matched to the same network/cell's familiar 100-cm trajectory
00. Cells are nested within network; only network means enter inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DEFAULT = ROOT.parent / "run" / "outputs" / "independent_networks"
OUTPUT_DEFAULT = ROOT / "outputs" / "familiarization_properties"
ENVS = {"compressed_80": 80, "elongated_120": 120}
SESSIONS = range(10)
NETWORKS = range(8)
KEYS = ["population", "module", "cell_id"]
BASE_METRICS = [
    "mean_rate_hz", "peak_rate_hz", "field_length_x_cm", "field_length_y_cm",
    "grid_spacing_cm", "phase_shift_x_cm", "phase_shift_y_cm",
    "movement_displacement_x_cm", "movement_displacement_y_cm",
]
COMPARISON_METRICS = [
    "rescale_factor", "rescale_best_correlation", "delta_field_length_cm",
    "delta_phase_shift_cm", "r_phase", "delta_movement_displacement_cm",
    "delta_peak_rate_hz", "peak_rate_ratio", "delta_mean_rate_hz", "mean_rate_ratio",
]
PRIMARY_FIG5 = [
    ("grid", 1, "rescale_factor"), ("grid", 2, "rescale_factor"),
    ("grid", 3, "rescale_factor"),
    ("grid", 1, "delta_phase_shift_x_cm"),
    ("grid", 2, "delta_phase_shift_x_cm"),
    ("grid", 3, "delta_phase_shift_x_cm"),
    ("place", 0, "delta_field_length_x_cm"),
    ("place", 0, "delta_place_phase_x_cm"),
]
PRIMARY_SI3 = [
    ("grid", 1, "peak_rate_ratio"), ("grid", 2, "peak_rate_ratio"),
    ("grid", 3, "peak_rate_ratio"),
    ("grid", 1, "mean_rate_ratio"), ("grid", 2, "mean_rate_ratio"),
    ("grid", 3, "mean_rate_ratio"),
    ("place", 0, "delta_peak_rate_hz"),
    ("place", 0, "delta_mean_rate_hz"),
]
METRICS = [
    "rescale_factor", "rescale_best_correlation",
    "phase_shift_x_cm", "phase_shift_y_cm",
    "delta_phase_shift_x_cm", "delta_phase_shift_y_cm",
    "phase_ratio_x_half_period", "phase_ratio_y_half_period",
    "delta_phase_ratio_x_half_period", "delta_phase_ratio_y_half_period",
    "field_length_x_cm", "field_length_y_cm",
    "delta_field_length_x_cm", "delta_field_length_y_cm",
    "place_phase_x_cm", "place_phase_y_cm",
    "delta_place_phase_x_cm", "delta_place_phase_y_cm",
    "peak_rate_hz", "mean_rate_hz", "delta_peak_rate_hz",
    "delta_mean_rate_hz", "peak_rate_ratio", "mean_rate_ratio",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def session_dir(network_root: Path, env: str, session: int) -> Path:
    if session == 0:
        return network_root / "initial_exposure" / env / "trajectory_00"
    return network_root / "familiarization" / env / "intact" / "trajectory_00" / f"session_{session:02d}"


def _normalize_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["population"] = frame["population"].astype(str)
    frame["module"] = pd.to_numeric(frame["module"], errors="coerce").fillna(0).astype(int)
    frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="raise").astype(int)
    if frame.duplicated(KEYS).any() or len(frame) != 400:
        raise ValueError(f"{label}: expected 400 unique population/module/cell rows")
    if set(frame["population"]) != {"grid", "place"}:
        raise ValueError(f"{label}: unexpected populations")
    return frame


def _read_metrics(directory: Path, compare: bool) -> tuple[pd.DataFrame, Path]:
    name = "comparison_metrics.csv.gz" if compare else "cell_metrics.csv.gz"
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _normalize_keys(pd.read_csv(path), str(path))
    expected = COMPARISON_METRICS if compare else BASE_METRICS
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return frame, path


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def load_all(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    input_audit: list[dict[str, object]] = []
    for network_id in NETWORKS:
        network_root = data_root / "networks" / f"network_{network_id:03d}"
        baseline, baseline_path = _read_metrics(
            network_root / "baseline" / "fam_100" / "trajectory_00", False
        )
        baseline = baseline[KEYS + BASE_METRICS].rename(
            columns={name: f"baseline_{name}" for name in BASE_METRICS}
        )
        input_audit.append({"path": str(baseline_path), "sha256": sha256(baseline_path), "rows": 400})
        for environment, length_cm in ENVS.items():
            for session in SESSIONS:
                directory = session_dir(network_root, environment, session)
                current, cell_path = _read_metrics(directory, False)
                comparison, comp_path = _read_metrics(directory, True)
                required_stage = "initial_exposure" if session == 0 else "familiarization"
                for source, path in ((current, cell_path), (comparison, comp_path)):
                    if not source["stage"].astype(str).eq(required_stage).all():
                        raise ValueError(f"{path}: stage mismatch")
                    if not source["environment"].astype(str).eq(environment).all():
                        raise ValueError(f"{path}: environment mismatch")
                    if not pd.to_numeric(source["session"]).eq(session).all():
                        raise ValueError(f"{path}: session mismatch")
                    if not pd.to_numeric(source["network_id"]).eq(network_id).all():
                        raise ValueError(f"{path}: network mismatch")
                    if not pd.to_numeric(source["trajectory_id"]).eq(0).all():
                        raise ValueError(f"{path}: trajectory mismatch")
                    input_audit.append({"path": str(path), "sha256": sha256(path), "rows": 400})
                merged = current[KEYS + BASE_METRICS].merge(
                    comparison[KEYS + COMPARISON_METRICS], on=KEYS,
                    validate="one_to_one", suffixes=("", "_saved"),
                ).merge(baseline, on=KEYS, validate="one_to_one")
                if len(merged) != 400:
                    raise ValueError(f"{directory}: incomplete cell matching")
                result = merged[KEYS].copy()
                result["network_id"] = network_id
                result["environment"] = environment
                result["length_cm"] = length_cm
                result["deformation_ratio"] = length_cm / 100.0
                result["trajectory_id"] = 0
                result["session"] = session
                result["rescale_factor"] = _numeric(merged, "rescale_factor")
                result["rescale_best_correlation"] = _numeric(merged, "rescale_best_correlation")
                for axis in ("x", "y"):
                    phase = _numeric(merged, f"phase_shift_{axis}_cm")
                    baseline_phase = _numeric(merged, f"baseline_phase_shift_{axis}_cm")
                    spacing = _numeric(merged, "baseline_grid_spacing_cm")
                    movement = _numeric(merged, f"movement_displacement_{axis}_cm")
                    baseline_movement = _numeric(merged, f"baseline_movement_displacement_{axis}_cm")
                    length = _numeric(merged, f"field_length_{axis}_cm")
                    baseline_length = _numeric(merged, f"baseline_field_length_{axis}_cm")
                    result[f"phase_shift_{axis}_cm"] = phase
                    result[f"delta_phase_shift_{axis}_cm"] = phase - baseline_phase
                    result[f"phase_ratio_{axis}_half_period"] = _ratio(2 * phase, spacing)
                    result[f"delta_phase_ratio_{axis}_half_period"] = _ratio(2 * (phase - baseline_phase), spacing)
                    result[f"field_length_{axis}_cm"] = length
                    result[f"delta_field_length_{axis}_cm"] = length - baseline_length
                    result[f"place_phase_{axis}_cm"] = movement
                    result[f"delta_place_phase_{axis}_cm"] = movement - baseline_movement
                for measure in ("peak_rate_hz", "mean_rate_hz"):
                    current_values = _numeric(merged, measure)
                    familiar_values = _numeric(merged, f"baseline_{measure}")
                    result[measure] = current_values
                    result[f"delta_{measure}"] = current_values - familiar_values
                    result[measure.replace("_hz", "_ratio")] = _ratio(current_values, familiar_values)
                # Saved comparison values are a cross-check of the x-axis and
                # firing-rate transformations, not an alternate estimator.
                compare = {
                    "delta_field_length_cm": "delta_field_length_x_cm",
                    "delta_phase_shift_cm": "delta_phase_shift_x_cm",
                    "r_phase": "delta_phase_ratio_x_half_period",
                    "delta_movement_displacement_cm": "delta_place_phase_x_cm",
                    "delta_peak_rate_hz": "delta_peak_rate_hz",
                    "peak_rate_ratio": "peak_rate_ratio",
                    "delta_mean_rate_hz": "delta_mean_rate_hz",
                    "mean_rate_ratio": "mean_rate_ratio",
                }
                for saved, calculated in compare.items():
                    a = _numeric(merged, saved)
                    b = pd.to_numeric(result[calculated], errors="coerce").to_numpy(float)
                    applicable = merged["population"].eq("grid" if saved in {"delta_phase_shift_cm", "r_phase"} else "place" if saved in {"delta_field_length_cm", "delta_movement_displacement_cm"} else "grid").to_numpy() if saved in {"delta_phase_shift_cm", "r_phase", "delta_field_length_cm", "delta_movement_displacement_cm"} else np.ones(len(merged), bool)
                    if not np.allclose(a[applicable], b[applicable], equal_nan=True, atol=1e-8, rtol=1e-6):
                        raise ValueError(f"{directory}: {saved} differs from matched-baseline recomputation")
                rows.append(result)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(input_audit)


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = ["network_id", "environment", "length_cm", "session", "population", "module"]
    long = cells.melt(id_vars=ids + ["cell_id"], value_vars=METRICS,
                      var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    counts = long.groupby(ids + ["metric"], sort=True)["value"].agg(
        n_cells_total="size", n_cells_finite="count"
    ).reset_index()
    finite = long[np.isfinite(long["value"].to_numpy(float))]
    network = finite.groupby(ids + ["metric"], sort=True)["value"].agg(
        network_mean="mean", network_cell_sd="std", n_cells="size"
    ).reset_index()
    group_ids = ["environment", "length_cm", "session", "population", "module", "metric"]
    group = network.groupby(group_ids, sort=True)["network_mean"].agg(
        group_mean="mean", sd_across_networks="std", n_networks="size"
    ).reset_index()
    group["sem_across_networks"] = group["sd_across_networks"] / np.sqrt(group["n_networks"])
    tcrit = stats.t.ppf(0.975, group["n_networks"] - 1)
    group["ci95_low"] = group["group_mean"] - tcrit * group["sem_across_networks"]
    group["ci95_high"] = group["group_mean"] + tcrit * group["sem_across_networks"]
    return network, group, counts


def _bh(values: pd.Series) -> pd.Series:
    ordered = values.sort_values()
    count = len(ordered)
    adjusted = (ordered.to_numpy(float) * count / np.arange(1, count + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    return pd.Series(np.minimum(adjusted, 1.0), index=ordered.index).reindex(values.index)


def endpoint_statistics(network: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_rows: list[dict[str, object]] = []
    for family, metrics in (("fig5", PRIMARY_FIG5), ("supplementary_fig3", PRIMARY_SI3)):
        for environment in ENVS:
            for population, module, metric in metrics:
                subset = network[
                    network["environment"].eq(environment)
                    & network["population"].eq(population)
                    & network["module"].eq(module)
                    & network["metric"].eq(metric)
                    & network["session"].isin([0, 9])
                ]
                wide = subset.pivot(index="network_id", columns="session", values="network_mean")
                if set(wide.columns) != {0, 9} or len(wide) != len(NETWORKS) or wide.isna().any().any():
                    raise ValueError(f"Incomplete endpoint pairing: {family}/{environment}/{module}/{metric}")
                for network_id, record in wide.iterrows():
                    contrast_rows.append({
                        "family": family, "environment": environment,
                        "population": population, "module": module,
                        "metric": metric, "network_id": int(network_id),
                        "session_0": float(record[0]), "session_9": float(record[9]),
                        "difference_9_minus_0": float(record[9] - record[0]),
                    })
    differences = pd.DataFrame(contrast_rows)
    summary: list[dict[str, object]] = []
    group_ids = ["family", "environment", "population", "module", "metric"]
    for key, frame in differences.groupby(group_ids, sort=True):
        values = frame["difference_9_minus_0"].to_numpy(float)
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        sem = sd / np.sqrt(n)
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        t = stats.ttest_1samp(values, 0.0)
        wilcoxon = stats.wilcoxon(values, alternative="two-sided") if np.any(values != 0) else None
        summary.append(dict(zip(group_ids, key)) | {
            "n_networks": n, "mean_difference": mean, "sd_difference": sd,
            "ci95_low": ci[0], "ci95_high": ci[1],
            "cohens_dz": mean / sd if sd else np.nan,
            "t_statistic": float(t.statistic), "t_degrees_of_freedom": n - 1,
            "p_ttest_two_sided": float(t.pvalue),
            "p_wilcoxon_two_sided": float(wilcoxon.pvalue) if wilcoxon is not None else np.nan,
        })
    table = pd.DataFrame(summary)
    table["p_ttest_fdr_bh_within_figure"] = table.groupby("family")["p_ttest_two_sided"].transform(_bh)
    return differences, table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config_path = data_root / "config.resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("n_networks") != len(NETWORKS):
        raise ValueError(f"Expected {len(NETWORKS)} independent networks")
    cells, input_audit = load_all(data_root)
    network, group, counts = aggregate(cells)
    differences, tests = endpoint_statistics(network)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(metrics_dir / "cell_level_values.csv.gz", index=False)
    network.to_csv(metrics_dir / "network_level_values.csv", index=False)
    group.to_csv(metrics_dir / "group_summary_network_unit.csv", index=False)
    counts.to_csv(metrics_dir / "finite_cell_counts.csv", index=False)
    differences.to_csv(metrics_dir / "endpoint_network_differences.csv", index=False)
    tests.to_csv(metrics_dir / "endpoint_statistics.csv", index=False)
    input_audit.to_csv(output / "input_file_audit.csv", index=False)
    provenance = {
        "profile": "production Methods", "data_root": str(data_root),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "network_ids": list(NETWORKS), "n_independent_networks": 8,
        "environments": ENVS, "sessions": list(SESSIONS),
        "session_0": "initial exposure trajectory_00",
        "sessions_1_to_9": "intact-plasticity familiarization trajectory_00",
        "matched_baseline": "same network/cell, familiar 100 cm trajectory_00",
        "ratio_zero_denominator": "NaN; no imputation",
        "independent_unit": "network; cells nested within one trajectory per session",
        "summary": "network mean and cross-network 95% t confidence interval",
        "endpoint_test": "paired network-level session_9 minus session_0, two-sided one-sample t; BH within each figure (16 tests)",
        "cell_rows": len(cells), "network_rows": len(network),
        "n_input_files": len(input_audit),
        "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
        "data_definitions": {
            "rescale_factor": "saved grid comparison rescaling along deformed x axis",
            "delta_phase_shift_axis_cm": "current absolute grid boundary phase minus matched familiar phase",
            "delta_field_length_axis_cm": "current place-field length minus matched familiar length",
            "delta_place_phase_axis_cm": "current opposing-movement displacement minus matched familiar displacement",
            "grid_rate_ratio": "current cell peak/mean rate divided by matched familiar rate; only positive denominator",
            "place_rate_delta_hz": "current cell peak/mean rate minus matched familiar rate",
        },
    }
    (output / "analysis_qa_and_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Methods familiarization analysis complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
