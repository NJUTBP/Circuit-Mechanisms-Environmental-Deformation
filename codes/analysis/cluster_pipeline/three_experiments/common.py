"""Shared audit and metric code for the three independent cluster analyses.

Simulation files are read only. Each worker writes one condition/network shard
under that experiment's analysis directory. Edge5-M is imported from the
frozen local-network implementation, so its 2.5-cm binning, 1.5-bin
normalized-mask smoothing, candidate scales, and five edge translations are
identical to Figures 2 and 5.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ANALYSIS = PROJECT_ROOT / "local_network_analysis"
FROZEN_EDGE5_SHA256 = {
    "initial_exposure_rescaling.py": "5f4816d85012e1031f7def42dde9662c3ec7e28e1139e77e12643013ad6a4b14",
    "grid_rescaling.py": "cc335ea89850214175ce06097b1c032216fbc72bf3d9851b94a48eb5ae1fb815",
    "grid_rescale_primitives.py": "95b29f789259a0cd64a4b372dd8661462ee34e5c82708f272d4242426ac4b8be",
}
if str(LOCAL_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ANALYSIS))
from initial_exposure_rescaling import load_selected_maps  # noqa: E402
from grid_rescaling import compute_recommended_rescale_batch  # noqa: E402

ENVS = {
    "compressed_70": 70, "compressed_80": 80, "compressed_90": 90,
    "fam_100": 100, "elongated_110": 110,
    "elongated_120": 120, "elongated_130": 130,
}
DEFORMED = tuple(env for env in ENVS if env != "fam_100")
FAM_ENVS = ("compressed_80", "elongated_120")
GRID_MODULES = (1, 2, 3)
CELL_KEYS = ["population", "module", "cell_id"]
MEASURES = (
    "rescale_factor", "rescale_best_correlation",
    "grid_spacing_cm", "grid_score",
    "phase_shift_x_cm", "phase_shift_y_cm",
    "delta_grid_phase_x_cm", "delta_grid_phase_y_cm",
    "delta_R_phase_x", "delta_R_phase_y",
    "field_length_x_cm", "field_length_y_cm",
    "delta_place_field_length_x_cm", "delta_place_field_length_y_cm",
    "movement_displacement_x_cm", "movement_displacement_y_cm",
    "delta_place_phase_distance_x_cm", "delta_place_phase_distance_y_cm",
    "mean_rate_hz", "peak_rate_hz", "spatial_information_bits_per_spike",
    "delta_mean_rate_hz", "delta_peak_rate_hz",
    "delta_spatial_information_bits_per_spike",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_edge5_sources() -> None:
    for name, wanted in FROZEN_EDGE5_SHA256.items():
        path = LOCAL_ANALYSIS / name
        if not path.is_file() or sha256(path) != wanted:
            raise RuntimeError(
                f"Frozen Edge5-M implementation missing or changed: {path}. "
                "Upload the three audited local_network_analysis source files."
            )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_config(path: Path, n_place: int, *, alpha: float | None = None,
                 fixed_delta: bool = False) -> dict[str, Any]:
    config = read_json(path)
    analysis = config["analysis"]
    checks = {
        "rate_map_bin_m": 0.025,
        "rate_map_sigma_bins": 1.5,
        "place_field_rate_map_sigma_bins": 2.0,
        "place_field_peak_fraction": 0.2,
    }
    for key, wanted in checks.items():
        if not np.isclose(float(analysis[key]), wanted):
            raise RuntimeError(f"{path}: {key}={analysis[key]} differs from the configured setting {wanted}")
    if int(config["model"]["n_place"]) != n_place:
        raise RuntimeError(f"{path}: n_place differs from {n_place}")
    if fixed_delta and not np.isclose(float(config["model"]["place_global_inhibition"]), 1.5):
        raise RuntimeError(f"{path}: place-cell inhibition is not fixed at 1.5")
    if 0 not in {int(value) for value in config["evaluation"]["trajectory_ids"]}:
        raise RuntimeError(f"{path}: trajectory_00 is not configured")
    if alpha is not None:
        gains = np.asarray(config["model"]["grid_velocity_gains"], dtype=float)
        wanted = np.asarray([alpha, 0.7 * alpha, 0.5 * alpha])
        if not np.allclose(gains, wanted, rtol=0, atol=1e-12):
            raise RuntimeError(f"{path}: alpha gain triplet {gains} differs from {wanted}")
    return config


@dataclass(frozen=True)
class Run:
    condition: str
    parameter: float
    network_id: int
    stage: str
    environment: str
    session: int
    root: Path
    n_place: int
    variant: str = "intact"
    trajectory_id: int = 0

    @property
    def network_root(self) -> Path:
        return self.root / "networks" / f"network_{self.network_id:03d}"

    @property
    def run_dir(self) -> Path:
        if self.stage == "baseline":
            return self.network_root / "baseline" / "fam_100" / f"trajectory_{self.trajectory_id:02d}"
        if self.stage == "initial_exposure":
            return self.network_root / "initial_exposure" / self.environment / f"trajectory_{self.trajectory_id:02d}"
        if self.stage == "familiarization":
            return (
                self.network_root / "familiarization" / self.environment / self.variant
                / f"trajectory_{self.trajectory_id:02d}" / f"session_{self.session:02d}"
            )
        raise ValueError(f"Run has no cell data for stage {self.stage}")

    @property
    def group_dir(self) -> Path:
        if self.stage == "familiarization":
            return (
                self.network_root / "familiarization" / self.environment / self.variant
                / "session_outputs" / f"session_{self.session:02d}"
            )
        return self.run_dir.parent

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "condition": self.condition, "parameter": self.parameter,
            "network_id": self.network_id, "stage": self.stage,
            "environment": self.environment, "length_cm": ENVS[self.environment],
            "session": self.session, "trajectory_id": self.trajectory_id,
            "variant": self.variant, "source_root": str(self.root),
        }


def _manifest_record(manifest: dict[str, Any], group: Path, target: Path) -> dict[str, Any] | None:
    wanted = target.resolve()
    for record in manifest.get("outputs", []):
        if (group / str(record.get("path", ""))).resolve() == wanted:
            return record
    return None


def _check_manifest(run: Run, verify_hash: bool, check_inputs: bool) -> tuple[bool, str, dict[str, Any]]:
    path = run.group_dir / "manifest.json"
    if not path.is_file():
        return False, "missing_manifest", {}
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False, "corrupt_manifest", {}
    if manifest.get("status") != "complete" or manifest.get("stage") != run.stage:
        return False, "manifest_not_complete_or_wrong_stage", manifest
    resolved = run.root / "config.resolved.json"
    if not resolved.is_file():
        return False, "missing_resolved_config", manifest
    try:
        expected_hash = read_json(resolved)["_resolved"]["run_sha256"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False, "invalid_resolved_config", manifest
    if manifest.get("run_sha256") != expected_hash:
        return False, "manifest_run_hash_mismatch", manifest
    ident = manifest.get("identity", {})
    for key, wanted in (("network_id", run.network_id), ("environment", run.environment)):
        if ident.get(key) != wanted:
            return False, f"manifest_{key}_mismatch", manifest
    if run.stage == "familiarization" and (
        ident.get("variant") != run.variant or int(ident.get("session", -1)) != run.session
    ):
        return False, "manifest_variant_or_session_mismatch", manifest
    required = [
        run.run_dir / "cell_metrics.csv.gz",
        run.run_dir / "rate_maps.npz",
        run.run_dir / "transient_activity.npz",
        run.run_dir / "checkpoint.npz",
        run.group_dir / "network_summary.csv.gz",
    ]
    if run.stage != "baseline":
        required.append(run.run_dir / "comparison_metrics.csv.gz")
    output_names = [str(record.get("path", "")) for record in manifest.get("outputs", [])]
    if len(output_names) != len(set(output_names)):
        return False, "duplicate_manifest_output_path", manifest
    for record in manifest.get("outputs", []):
        target = (run.group_dir / str(record.get("path", ""))).resolve()
        if not target.is_file():
            return False, "missing_recorded_output", manifest
        if target.stat().st_size != int(record.get("bytes", -1)):
            return False, "recorded_output_size_mismatch", manifest
        if verify_hash and sha256(target) != record.get("sha256"):
            return False, "recorded_output_sha256_mismatch", manifest
    for target in required:
        record = _manifest_record(manifest, run.group_dir, target)
        if record is None or not target.is_file():
            return False, f"missing_or_unrecorded:{target.name}", manifest
        if target.stat().st_size != int(record.get("bytes", -1)):
            return False, f"size_mismatch:{target.name}", manifest
    if check_inputs:
        for record in manifest.get("inputs", []):
            target = Path(str(record.get("path", "")))
            if not target.is_absolute():
                target = run.group_dir / target
            if not target.is_file():
                return False, "missing_manifest_input", manifest
            if target.stat().st_size != int(record.get("bytes", -1)):
                return False, "manifest_input_size_mismatch", manifest
            if verify_hash and sha256(target) != record.get("sha256"):
                return False, "manifest_input_sha256_mismatch", manifest
    return True, "complete", manifest


def audit_run(
    run: Run, *, verify_hash: bool = True, check_inputs: bool = True
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    qc = {**run.identity, "group_path": str(run.group_dir), "run_path": str(run.run_dir),
          "included": False, "reason": "", "manifest_status": "",
          "manifest_run_sha256": "", "n_cells": 0, "n_nonfinite_rates": 0,
          "place_shape": "", "grid_shape": "", "positions_shape": "",
          "trajectory_sha256": "", "bvc_sha256": ""}
    valid, reason, manifest = _check_manifest(run, verify_hash, check_inputs)
    qc["manifest_status"] = str(manifest.get("status", "missing"))
    qc["manifest_run_sha256"] = str(manifest.get("run_sha256", ""))
    if not valid:
        qc["reason"] = reason
        return qc, None
    try:
        with np.load(run.run_dir / "transient_activity.npz", allow_pickle=False) as data:
            positions = data["positions"]
            n_frames = len(positions)
            place = data["place"]
            grid = [data[f"grid_m{module}"] for module in GRID_MODULES]
            qc["positions_shape"] = str(positions.shape)
            qc["place_shape"] = str(place.shape)
            qc["grid_shape"] = str([item.shape for item in grid])
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError("positions_shape")
            if place.shape != (run.n_place, n_frames):
                raise ValueError("place_activity_shape")
            if any(item.shape != (100, n_frames) for item in grid):
                raise ValueError("grid_activity_shape")
        with np.load(run.run_dir / "rate_maps.npz", allow_pickle=False) as maps:
            if not np.isclose(float(maps["bin_size_cm"]), 2.5):
                raise ValueError("rate_map_bin_size")
            expected = (40, round(ENVS[run.environment] / 2.5))
            if maps["place"].shape != (run.n_place, *expected):
                raise ValueError("place_rate_map_shape")
            if any(maps[f"grid_m{module}"].shape != (100, *expected) for module in GRID_MODULES):
                raise ValueError("grid_rate_map_shape")
        cells = pd.read_csv(run.run_dir / "cell_metrics.csv.gz")
        cells["module"] = pd.to_numeric(cells["module"], errors="coerce").fillna(0).astype(int)
        cells["cell_id"] = pd.to_numeric(cells["cell_id"], errors="raise").astype(int)
        qc["n_cells"] = len(cells)
        if len(cells) != run.n_place + 300 or cells.duplicated(CELL_KEYS).any():
            raise ValueError("cell_metric_count_or_duplicate")
        counts = cells.groupby(["population", "module"]).size().to_dict()
        if counts != {("place", 0): run.n_place, **{("grid", i): 100 for i in GRID_MODULES}}:
            raise ValueError("cell_population_or_module_count")
        for field, wanted in (
            ("network_id", run.network_id), ("environment", run.environment),
            ("trajectory_id", run.trajectory_id), ("session", run.session),
            ("stage", run.stage),
        ):
            if not cells[field].eq(wanted).all():
                raise ValueError(f"cell_{field}_mismatch")
        required_columns = [
            "mean_rate_hz", "peak_rate_hz", "spatial_information_bits_per_spike",
            "grid_spacing_cm", "grid_score", "phase_shift_x_cm", "phase_shift_y_cm",
            "field_length_x_cm", "field_length_y_cm",
            "movement_displacement_x_cm", "movement_displacement_y_cm",
        ]
        if any(field not in cells for field in required_columns):
            raise ValueError("missing_cell_metric_columns")
        mean = pd.to_numeric(cells["mean_rate_hz"], errors="coerce").to_numpy(float)
        peak = pd.to_numeric(cells["peak_rate_hz"], errors="coerce").to_numpy(float)
        qc["n_nonfinite_rates"] = int((~np.isfinite(mean) | ~np.isfinite(peak) | (mean < 0) | (peak < 0)).sum())
        if qc["n_nonfinite_rates"]:
            raise ValueError("nonfinite_or_negative_firing_rate")
        work = run.run_dir / "work_manifest.json"
        if work.is_file():
            trajectory = Path(str(read_json(work).get("trajectory_path", "")))
            if trajectory.is_file():
                qc["trajectory_sha256"] = sha256(trajectory)
        bvc = run.network_root / "init" / "bvc_tuning.npz"
        if bvc.is_file():
            qc["bvc_sha256"] = sha256(bvc)
        qc["included"] = True
        qc["reason"] = "complete"
        return qc, cells
    except (OSError, ValueError, KeyError, TypeError) as exc:
        qc["reason"] = f"corrupt_data:{exc}"
        return qc, None


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=float)
    ok = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    np.divide(numerator, denominator, out=out, where=ok)
    return out


def derive_cells(
    run: Run, cells: pd.DataFrame, baseline: pd.DataFrame,
    baseline_maps: dict[int, np.ndarray], rescale_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> pd.DataFrame:
    current = cells.merge(
        baseline[CELL_KEYS + [
            "phase_shift_x_cm", "phase_shift_y_cm", "grid_spacing_cm",
            "field_length_x_cm", "field_length_y_cm",
            "movement_displacement_x_cm", "movement_displacement_y_cm",
            "mean_rate_hz", "peak_rate_hz", "spatial_information_bits_per_spike",
        ]],
        on=CELL_KEYS, how="left", validate="one_to_one", suffixes=("", "_baseline"),
    )
    if len(current) != len(cells):
        raise RuntimeError("Baseline cell matching changed the number of cells")
    current["condition"] = run.condition
    current["parameter"] = run.parameter
    current["length_cm"] = ENVS[run.environment]
    current["source_root"] = str(run.root)
    current["rescale_factor"] = np.nan
    current["rescale_best_correlation"] = np.nan
    for axis in ("x", "y"):
        phase = pd.to_numeric(current[f"phase_shift_{axis}_cm"], errors="coerce").to_numpy(float)
        base_phase = pd.to_numeric(current[f"phase_shift_{axis}_cm_baseline"], errors="coerce").to_numpy(float)
        spacing = pd.to_numeric(current["grid_spacing_cm_baseline"], errors="coerce").to_numpy(float)
        current[f"delta_grid_phase_{axis}_cm"] = np.abs(phase) - np.abs(base_phase)
        current[f"delta_R_phase_{axis}"] = _divide(
            2.0 * current[f"delta_grid_phase_{axis}_cm"].to_numpy(float), spacing
        )
        current[f"delta_place_field_length_{axis}_cm"] = (
            pd.to_numeric(current[f"field_length_{axis}_cm"], errors="coerce")
            - pd.to_numeric(current[f"field_length_{axis}_cm_baseline"], errors="coerce")
        )
        current[f"delta_place_phase_distance_{axis}_cm"] = (
            pd.to_numeric(current[f"movement_displacement_{axis}_cm"], errors="coerce").abs()
            - pd.to_numeric(current[f"movement_displacement_{axis}_cm_baseline"], errors="coerce").abs()
        )
    for metric in ("mean_rate_hz", "peak_rate_hz", "spatial_information_bits_per_spike"):
        current[f"delta_{metric}"] = (
            pd.to_numeric(current[metric], errors="coerce")
            - pd.to_numeric(current[f"{metric}_baseline"], errors="coerce")
        )
    if run.stage == "baseline":
        current.loc[current["population"] == "grid", "rescale_factor"] = 1.0
    else:
        cache_key = str(run.run_dir / "transient_activity.npz")
        if cache_key not in rescale_cache:
            maps = load_selected_maps(Path(cache_key), ENVS[run.environment])
            rescale_cache[cache_key] = {
                module: compute_recommended_rescale_batch(
                    baseline_maps[module], maps[module], ENVS[run.environment],
                    edge_positions_per_side=5, bin_cm=2.5,
                )
                for module in GRID_MODULES
            }
        for module in GRID_MODULES:
            mask = (current["population"] == "grid") & (current["module"] == module)
            ids = current.loc[mask, "cell_id"].to_numpy(int)
            factors, correlations = rescale_cache[cache_key][module]
            current.loc[mask, "rescale_factor"] = factors[ids]
            current.loc[mask, "rescale_best_correlation"] = correlations[ids]
    place = current["population"] == "place"
    grid = ~place
    for axis in ("x", "y"):
        current.loc[place, [f"delta_grid_phase_{axis}_cm", f"delta_R_phase_{axis}"]] = np.nan
        current.loc[grid, [f"delta_place_field_length_{axis}_cm",
                           f"delta_place_phase_distance_{axis}_cm"]] = np.nan
    return current


def network_long(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "parameter", "network_id", "environment", "length_cm",
            "session", "population", "module"]
    grouped = cells.groupby(keys, dropna=False, sort=True)
    means = grouped[list(MEASURES)].mean(numeric_only=True)
    counts = grouped[list(MEASURES)].count().add_prefix("count_")
    wide = means.join(counts).reset_index()
    values = wide.melt(id_vars=keys, value_vars=list(MEASURES), var_name="metric", value_name="network_mean")
    count_values = wide.melt(id_vars=keys, value_vars=[f"count_{m}" for m in MEASURES],
                             var_name="count_metric", value_name="n_cells_finite")
    values["n_cells_finite"] = count_values["n_cells_finite"].to_numpy(int)
    values["n_cells_total"] = values["module"].map({0: int(cells.loc[cells.population.eq("place"), "cell_id"].nunique()),
                                                      1: 100, 2: 100, 3: 100}).astype(int)
    return values


def write_shard(output: Path, condition: str, network_id: int,
                cells: pd.DataFrame, qc: pd.DataFrame) -> None:
    shard = output / "shards" / f"{condition}__network_{network_id:03d}"
    shard.mkdir(parents=True, exist_ok=True)
    qc.to_csv(shard / "task_qc.csv", index=False)
    if len(cells):
        cells.to_csv(shard / "cell_values.csv.gz", index=False, compression="gzip")
        network_long(cells).to_csv(shard / "network_values.csv", index=False)
    else:
        pd.DataFrame().to_csv(shard / "network_values.csv", index=False)


def group_summary(network: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "parameter", "environment", "length_cm", "session",
            "population", "module", "metric"]
    rows = []
    for key, part in network.groupby(keys, dropna=False, sort=True):
        values = part["network_mean"].to_numpy(float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
        rows.append(dict(zip(keys, key)) | {
            "mean": mean, "sd_across_networks": sd, "sem_across_networks": sem,
            "ci95_low": mean - half if n > 1 else np.nan,
            "ci95_high": mean + half if n > 1 else np.nan,
            "n_networks": n,
        })
    return pd.DataFrame(rows)


def paired_contrasts(network: pd.DataFrame, reference: str,
                     metrics: tuple[str, ...]) -> pd.DataFrame:
    """Prespecified network-paired contrasts; callers select stages/sessions."""
    selected = network.loc[network["metric"].isin(metrics)]
    keys = ["environment", "session", "population", "module", "metric"]
    rows = []
    for key, part in selected.groupby(keys, dropna=False, sort=True):
        ref = part.loc[part.condition == reference].set_index("network_id")["network_mean"]
        if ref.empty:
            continue
        for condition, treatment in part.loc[part.condition != reference].groupby("condition"):
            test_values = treatment["network_mean"].to_numpy(float)
            reference_values = ref.to_numpy(float)
            test_values = test_values[np.isfinite(test_values)]
            reference_values = reference_values[np.isfinite(reference_values)]
            if len(test_values) < 2 or len(reference_values) < 2:
                continue
            welch = stats.ttest_ind(test_values, reference_values, equal_var=False)
            test_var = float(np.var(test_values, ddof=1) / len(test_values))
            reference_var = float(np.var(reference_values, ddof=1) / len(reference_values))
            welch_se = np.sqrt(test_var + reference_var)
            welch_df = (test_var + reference_var) ** 2 / (
                test_var ** 2 / (len(test_values) - 1)
                + reference_var ** 2 / (len(reference_values) - 1)
            ) if test_var + reference_var > 0 else np.nan
            unpaired_mean = float(np.mean(test_values) - np.mean(reference_values))
            unpaired_half = float(stats.t.ppf(0.975, welch_df) * welch_se) if np.isfinite(welch_df) else 0.0
            paired = treatment.set_index("network_id")["network_mean"].to_frame("test").join(
                ref.rename("reference"), how="inner"
            ).dropna()
            n = len(paired)
            mean, half, statistic, p_value = np.nan, np.nan, np.nan, np.nan
            if n >= 2:
                diff = paired["test"].to_numpy(float) - paired["reference"].to_numpy(float)
                mean = float(np.mean(diff))
                sem = float(np.std(diff, ddof=1) / np.sqrt(n))
                if np.allclose(diff, 0.0, rtol=0, atol=1e-12):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = stats.ttest_1samp(diff, 0.0)
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                half = float(stats.t.ppf(0.975, n - 1) * sem)
            rows.append(dict(zip(keys, key)) | {
                "condition": condition, "reference_condition": reference,
                "n_paired_networks": n, "network_ids": ",".join(map(str, paired.index.tolist())),
                "n_test_networks": len(test_values), "n_reference_networks": len(reference_values),
                "mean_paired_difference": mean, "ci95_low": mean - half,
                "ci95_high": mean + half, "t": statistic,
                "df": n - 1, "p_two_sided": p_value,
                "unpaired_mean_difference": unpaired_mean,
                "unpaired_ci95_low": unpaired_mean - unpaired_half,
                "unpaired_ci95_high": unpaired_mean + unpaired_half,
                "unpaired_t": float(welch.statistic),
                "unpaired_df": welch_df, "unpaired_p_two_sided": float(welch.pvalue),
            })
    frame = pd.DataFrame(rows)
    return frame


def add_bh_q(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["q_bh_within_metric"] = np.nan
    for _, index in frame.groupby(["metric", "environment"], dropna=False).groups.items():
        valid_index = frame.loc[index].index[
            np.isfinite(frame.loc[index, "p_two_sided"].to_numpy(float))
        ]
        p = frame.loc[valid_index, "p_two_sided"].to_numpy(float)
        if not len(p):
            continue
        order = np.argsort(p)
        q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        restored = np.empty_like(q)
        restored[order] = np.minimum(q, 1)
        frame.loc[valid_index, "q_bh_within_metric"] = restored
    return frame


def provenance(output: Path, experiment: str, configs: list[Path], inputs: pd.DataFrame,
               pairing: pd.DataFrame) -> None:
    sources = [Path(__file__), Path(__file__).with_name("runner.py"),
               LOCAL_ANALYSIS / "initial_exposure_rescaling.py",
               LOCAL_ANALYSIS / "grid_rescaling.py",
               LOCAL_ANALYSIS / "grid_rescale_primitives.py"]
    payload = {
        "experiment": experiment,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "grid_rescale": "Edge5-M: 2.5-cm occupancy-normalized rate maps, normalized-mask Gaussian sigma=1.5 bins, five complete-overlap translations per edge, Methods scale candidates",
            "place_field": "Methods: 2.5-cm bins, sigma=2 bins, max(0.1 Hz, 20% peak), longest continuous suprathreshold run",
            "phase": "Methods grid phase; delta_R_phase = 2*(abs(current phase)-abs(matched baseline phase))/baseline measured grid spacing",
            "place_phase_distance": "abs(current opposing-movement distance)-abs(matched baseline distance)",
            "spatial_information": "Saved production bits/spike on 2.5-cm sigma=1.5-bin maps",
            "aggregation": "finite cells within network/condition/stage/session/module, then independent networks",
            "missing_data": "no imputation; only manifest- and shape-valid runs are included",
            "baseline": "same-network fam_100 trajectory_00, matched by population/module/cell_id",
            "group_uncertainty": "95% Student-t interval across independent network means; individual networks shown as fine lines",
            "planned_tests": "two-sided paired network t test only when BVC and trajectory SHA-256 agree; otherwise independent-network Welch t test; BH FDR within metric and environment",
            "reference": {
                "learning_rate": "1x source run, first four networks; baseline and initial activity from that run",
                "place_cells": "n_place=100 source run, first four networks; fixed delta=1.5 for other groups",
                "alpha": "alpha1=0.10 source run, first four networks",
            }[experiment],
        },
        "input_metadata_files": [{"path": str(path), "sha256": sha256(path)} for path in configs if path.is_file()],
        "code_sha256": {str(path): sha256(path) for path in sources if path.is_file()},
        "included_task_count": int(inputs["included"].sum()),
        "expected_task_count": len(inputs),
        "pairing_audit": pairing.to_dict(orient="records"),
        "source_file_manifest": "source_file_manifest.csv",
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output / "provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
