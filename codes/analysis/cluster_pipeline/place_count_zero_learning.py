"""Adopted place-cell-count reanalysis with incomplete zero-rate controls.

Read-only inputs: source network, fixed-inhibition place-cell-count, and zero-rate simulations.
Workers write isolated shards; finalize validates every shard before publishing
network-level figure-source tables. Run `plan`, `worker --task-index N`, `finalize`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from three_experiments import common, runner
import grid_rescale_primitives as core


HERE = Path(__file__).resolve().parent
DEFAULT_LR = HERE.parent.parent / "run/outputs/learning_rates"
DEFAULT_ZERO = HERE.parent.parent / "run/outputs/place_cell_zero_rates"
COUNTS = (100, 200, 300, 400)
NETWORKS = range(4)
PATHWAYS = ("W_bp", "W_pg", "W_gp")
ENVS = common.FAM_ENVS
METRICS = ("rescale_factor", "delta_R_phase_x", "delta_place_field_length_x_cm",
           "delta_place_phase_distance_x_cm")
def method_profile(edge_positions_per_side: int) -> dict:
    if edge_positions_per_side != 5:
        raise ValueError("Only the adopted five-edge profile is included")
    return {"name": f"Edge{edge_positions_per_side}-M", "bin_cm": 2.5,
            "sigma_bins": 1.5, "scale_step_cm": 2.5, "scale_margin_cm": 10.0,
            "edge_positions_per_side": edge_positions_per_side, "min_valid_pixels": 20}


METHOD = method_profile(5)


def configure_method(edge_positions_per_side: int) -> None:
    global METHOD
    METHOD = method_profile(edge_positions_per_side)
MAP_BUILDER_COMPATIBLE_SHA256 = {
    "5f4816d85012e1031f7def42dde9662c3ec7e28e1139e77e12643013ad6a4b14",
}


def verify_rescale_sources() -> None:
    for name in ("grid_rescaling.py", "grid_rescale_primitives.py"):
        path = common.LOCAL_ANALYSIS / name
        if not path.is_file() or common.sha256(path) != common.FROZEN_EDGE5_SHA256[name]:
            raise RuntimeError(f"Frozen grid-rescaling estimator source missing or changed: {path}")
    builder = common.LOCAL_ANALYSIS / "initial_exposure_rescaling.py"
    if not builder.is_file() or common.sha256(builder).lower() not in MAP_BUILDER_COMPATIBLE_SHA256:
        raise RuntimeError(f"Unexpected 2.5-cm/sigma=1.5 map-builder source: {builder}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temp, index=False, compression="gzip" if path.suffix == ".gz" else None)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def count_condition(count: int) -> str:
    return "npc100_reference" if count == 100 else f"npc{count}_delta_fixed"


def units() -> list[dict]:
    return ([{"kind": "intact", "count": count, "network_id": net, "pathway": "intact"}
             for count in COUNTS for net in NETWORKS]
            + [{"kind": "zero", "count": count, "network_id": net, "pathway": pathway}
               for count in COUNTS[1:] for net in NETWORKS for pathway in PATHWAYS])


def unit_slug(unit: dict) -> str:
    return f"npc{unit['count']}__{unit['pathway']}__network_{unit['network_id']:03d}"


def source_run(unit: dict, source: Path, root: Path, zero: Path,
               env: str, session: int) -> common.Run:
    count, net = unit["count"], unit["network_id"]
    base = source if count == 100 else root / count_condition(count)
    stage = "initial_exposure" if session == 0 else "familiarization"
    run_root = base if unit["kind"] == "intact" or session == 0 else zero / count_condition(count)
    variant = "intact" if unit["kind"] == "intact" or session == 0 else f"{unit['pathway']}_0p0x"
    return common.Run(condition=count_condition(count), parameter=float(count),
                      network_id=net, stage=stage, environment=env, session=session,
                      root=run_root, n_place=count, variant=variant)


def zero_environment_availability(unit: dict, source: Path, root: Path,
                                  zero: Path, env: str) -> tuple[bool, str]:
    """Require all nine familiarization sessions for one network/pathway/env.

    This lightweight preflight checks complete manifests and recorded output
    sizes. The worker still runs hash and content QC on every included run.
    A partly produced environment is excluded as a whole so that every
    plotted curve uses the same networks for sessions 0-9.
    """
    if unit["kind"] != "zero":
        raise ValueError("Availability preflight is only for zero-control units")
    problems = []
    for session in range(1, 10):
        run = source_run(unit, source, root, zero, env, session)
        try:
            valid, reason, _ = common._check_manifest(run, verify_hash=False, check_inputs=True)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            valid, reason = False, f"invalid_manifest:{type(exc).__name__}"
        if not valid:
            problems.append(f"session_{session:02d}:{reason}")
    return (not problems, "complete" if not problems else ";".join(problems))


def zero_availability_report(source: Path, root: Path, zero: Path) -> list[dict]:
    return [{**unit, "environment": env, "available": available, "reason": reason}
            for unit in units() if unit["kind"] == "zero" for env in ENVS
            for available, reason in [zero_environment_availability(unit, source, root, zero, env)]]


def baseline_run(unit: dict, source: Path, root: Path) -> common.Run:
    count = unit["count"]
    base = source if count == 100 else root / count_condition(count)
    return common.Run(condition=count_condition(count), parameter=float(count),
                      network_id=unit["network_id"], stage="baseline", environment="fam_100",
                      session=0, root=base, n_place=count)


def checked_maps(run: common.Run) -> dict[int, np.ndarray]:
    valid, reason, _ = common._check_manifest(run, verify_hash=False, check_inputs=False)
    if not valid:
        raise RuntimeError(f"Input manifest/size QC failed ({reason}): {run.run_dir}")
    maps = common.load_selected_maps(run.run_dir / "transient_activity.npz",
                                      common.ENVS[run.environment])
    wanted = (100, 40, round(common.ENVS[run.environment] / 2.5))
    if any(maps[module].shape != wanted for module in common.GRID_MODULES):
        raise RuntimeError(f"Wrong grid map shape: {run.run_dir}")
    return maps


def prior_intact_qc(root: Path) -> pd.DataFrame:
    directory = root / "analysis"
    audit = common.read_json(directory / "experiment_plan_audit.json")
    if audit.get("status") != "pass":
        raise RuntimeError("Existing place-cell-count analysis plan audit did not pass")
    qc = pd.read_csv(directory / "task_integrity_qc.csv")
    qc["included_comparison"] = qc.included_comparison.astype(str).str.lower().eq("true")
    return qc


def checked_prior(run: common.Run, qc: pd.DataFrame) -> None:
    matching = qc.loc[qc.condition.eq(run.condition) & qc.network_id.eq(run.network_id)
                      & qc.stage.eq(run.stage) & qc.environment.eq(run.environment)
                      & qc.session.eq(run.session)]
    if len(matching) != 1 or not matching.iloc[0].included_comparison:
        raise RuntimeError(f"Prior full-analysis QC missing or failed: {run.identity}")
    if Path(str(matching.iloc[0].run_path)).resolve() != run.run_dir.resolve():
        raise RuntimeError(f"Prior analysis points to different raw run: {run.run_dir}")


def validate_unit_config(unit: dict, source: Path, root: Path, zero: Path) -> None:
    count = unit["count"]
    base = source if count == 100 else root / count_condition(count)
    common.check_config(base / "config.resolved.json", count, fixed_delta=count != 100)
    if unit["kind"] != "zero":
        return
    condition_root = zero / count_condition(count)
    config = common.check_config(condition_root / "config.resolved.json", count, fixed_delta=True)
    variants = config["familiarization"]["plasticity_variants"]
    wanted = f"{unit['pathway']}_0p0x"
    matched = [variant for variant in variants if variant["id"] == wanted]
    if len(matched) != 1:
        raise RuntimeError(f"Zero-control variant missing/duplicate in config: {wanted}")
    pathway_key = {"W_bp": "bvc_to_place", "W_pg": "place_to_grid", "W_gp": "grid_to_place"}
    values = matched[0]["multipliers"]
    for short, key in pathway_key.items():
        expected = 0.0 if short == unit["pathway"] else 1.0
        if not np.isclose(float(values[key]), expected):
            raise RuntimeError(f"Wrong learning-rate multiplier for {wanted}: {key}")


def compute_rescale(baseline: np.ndarray, current: np.ndarray, length_cm: float):
    if (core.SEARCH_STEP_CM, core.SEARCH_RANGE_CM, core.MIN_VALID_PIXELS) != (2.5, 10, 20):
        raise RuntimeError("Frozen Methods estimator parameters differ")
    factor, correlation = common.compute_recommended_rescale_batch(
        baseline, current, length_cm,
        edge_positions_per_side=METHOD["edge_positions_per_side"], bin_cm=2.5)
    if factor.shape != (100,) or correlation.shape != (100,) or not np.isfinite(correlation).all():
        raise RuntimeError("Estimator returned invalid correlation; reject factor=1 fallback")
    return factor, correlation


def work(unit: dict, source: Path, root: Path, zero: Path, output: Path) -> None:
    verify_rescale_sources()
    slug = unit_slug(unit)
    shard = output / "shards" / slug
    code = common.sha256(Path(__file__))
    status = {"status": "running", "unit": unit, "method": METHOD, "code_sha256": code,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(status, shard / "status.json")
    qc_rows, cells, network, si = [], [], [], []
    active = None
    try:
        validate_unit_config(unit, source, root, zero)
        skipped_environments = {}
        included_environments = list(ENVS)
        if unit["kind"] == "zero":
            availability = {env: zero_environment_availability(unit, source, root, zero, env)
                            for env in ENVS}
            included_environments = [env for env, (available, _) in availability.items() if available]
            skipped_environments = {env: reason for env, (available, reason) in availability.items()
                                    if not available}
            if not included_environments:
                atomic_json({**status, "status": "skipped",
                             "included_environments": [],
                             "skipped_environments": skipped_environments,
                             "finished_utc": datetime.now(timezone.utc).isoformat()},
                            shard / "status.json")
                print(f"{slug}: no complete zero-control environment; skipped", flush=True)
                return
        prior = prior_intact_qc(root)
        base_run = baseline_run(unit, source, root)
        checked_prior(base_run, prior)
        baseline = checked_maps(base_run)
        planned = ([(env, 0) for env in common.DEFORMED] +
                   [(env, session) for env in ENVS for session in range(1, 10)]) if unit["kind"] == "intact" else [
                   (env, session) for env in included_environments for session in range(1, 10)]
        for env, session in planned:
            run = source_run(unit, source, root, zero, env, session)
            active = run
            if unit["kind"] == "intact":
                checked_prior(run, prior)
            else:
                audit, metrics = common.audit_run(run, verify_hash=True, check_inputs=True)
                if not audit["included"] or metrics is None:
                    raise RuntimeError(f"Zero-control run QC failed ({audit['reason']}): {run.run_dir}")
                values = pd.to_numeric(metrics["spatial_information_bits_per_spike"], errors="coerce")
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    raise RuntimeError(f"No valid spatial-information cells: {run.run_dir}")
                si.append({**unit, "environment": env, "session": session,
                           "metric": "spatial_information_bits_per_spike",
                           "network_mean": float(finite.mean()), "n_cells_finite": len(finite),
                           "n_cells_total": len(metrics)})
            maps = checked_maps(run)
            for module in common.GRID_MODULES:
                factor, corr = compute_rescale(baseline[module], maps[module], common.ENVS[env])
                identity = {**unit, "environment": env, "session": session, "module": module}
                cells.append(pd.DataFrame({**identity, "cell_id": np.arange(100),
                                           "rescale_factor": factor, "best_correlation": corr}))
                network.append({**identity, "network_mean": float(factor.mean()),
                                "n_cells_finite": 100, "mean_best_correlation": float(corr.mean())})
            qc_rows.append({**unit, "environment": env, "session": session,
                            "run_path": str(run.run_dir), "included": True, "reason": "complete"})
            print(f"{slug}: {env} session {session:02d} complete", flush=True)
        expected = 24 if unit["kind"] == "intact" else 9 * len(included_environments)
        if len(qc_rows) != expected or len(network) != expected * 3 or len(si) != (0 if unit["kind"] == "intact" else expected):
            raise RuntimeError("Incomplete analysis shard")
        for name, frame in (("cell_rescale.csv.gz", pd.concat(cells, ignore_index=True)),
                            ("network_rescale.csv", pd.DataFrame(network)),
                            ("network_spatial_information.csv", pd.DataFrame(si)),
                            ("task_qc.csv", pd.DataFrame(qc_rows))):
            atomic_csv(frame, shard / name)
        status.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      included_environments=included_environments,
                      skipped_environments=skipped_environments,
                      outputs={name: {"bytes": (shard / name).stat().st_size,
                                      "sha256": common.sha256(shard / name)} for name in (
                          "cell_rescale.csv.gz", "network_rescale.csv",
                          "network_spatial_information.csv", "task_qc.csv")})
        atomic_json(status, shard / "status.json")
    except Exception as exc:
        atomic_csv(pd.DataFrame(qc_rows + [{**unit, "environment": active.environment if active else "",
                                            "session": active.session if active else -1,
                                            "included": False, "reason": str(exc)}]), shard / "task_qc.csv")
        atomic_json({**status, "status": "failed", "error": str(exc)}, shard / "status.json")
        raise


def pool_si(data: pd.DataFrame) -> pd.DataFrame:
    part = data.loc[data.metric.eq("spatial_information_bits_per_spike")].copy()
    if part.empty:
        raise RuntimeError("No spatial information in source analysis")
    part["weighted"] = part.network_mean * part.n_cells_finite
    keys = ["condition", "network_id", "environment", "session"]
    pooled = part.groupby(keys, as_index=False).agg(
        weighted=("weighted", "sum"), n_cells_finite=("n_cells_finite", "sum"),
        n_cells_total=("n_cells_total", "sum"), n_modules=("module", "size"))
    if not pooled.n_modules.eq(4).all() or pooled.n_cells_finite.le(0).any():
        raise RuntimeError("Spatial-information pooling requires place plus three grid modules")
    pooled["network_mean"] = pooled.weighted / pooled.n_cells_finite
    return pooled.drop(columns="weighted")


def group_curve(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    group = frame.groupby(keys, as_index=False).agg(
        mean=("network_mean", "mean"), sd_networks=("network_mean", "std"),
        n_networks=("network_id", "nunique"))
    if group.empty or not group.n_networks.between(1, 4).all():
        raise RuntimeError("Each plotted point must include one to four independent networks")
    half = (stats.t.ppf(.975, group.n_networks - 1) *
            group.sd_networks / np.sqrt(group.n_networks))
    group["ci95_low"], group["ci95_high"] = group["mean"] - half, group["mean"] + half
    return group


def exact_coverage(frame: pd.DataFrame, keys: list[str], expected: int, label: str) -> None:
    if len(frame) != expected or frame.duplicated(keys).any() or not np.isfinite(frame.network_mean).all():
        raise RuntimeError(f"Incomplete/duplicate/nonfinite {label}: {len(frame)} rows, expected {expected}")


def finish(source: Path, root: Path, zero: Path, lr: Path, output: Path) -> None:
    verify_rescale_sources()
    atomic_json({"status": "validating", "method": METHOD}, output / "finalize_status.json")
    network_parts, si_parts, qc_parts, cell_manifest, issues, availability_rows = [], [], [], [], [], []
    for unit in units():
        slug = unit_slug(unit)
        shard = output / "shards" / slug
        try:
            status = common.read_json(shard / "status.json")
            if status["unit"] != unit or status["method"] != METHOD or status["code_sha256"] != common.sha256(Path(__file__)):
                raise RuntimeError("Incomplete or stale shard")
            if status["status"] not in ("complete", "skipped"):
                raise RuntimeError(f"Worker status is {status['status']}")
            included_environments = status.get("included_environments", list(ENVS) if unit["kind"] == "intact" else [])
            skipped_environments = status.get("skipped_environments", {})
            if unit["kind"] == "zero":
                if (set(included_environments) | set(skipped_environments)) != set(ENVS) or (
                        set(included_environments) & set(skipped_environments)):
                    raise RuntimeError("Zero-control environment inclusion is incomplete or overlapping")
                for env in ENVS:
                    availability_rows.append({**unit, "environment": env,
                                              "available": env in included_environments,
                                              "reason": "complete" if env in included_environments else skipped_environments[env]})
            if status["status"] == "skipped":
                if unit["kind"] != "zero" or included_environments:
                    raise RuntimeError("Only an entirely absent zero-control unit may be skipped")
                continue
            for name, record in status["outputs"].items():
                path = shard / name
                if path.stat().st_size != record["bytes"] or common.sha256(path) != record["sha256"]:
                    raise RuntimeError(f"Changed shard file: {name}")
            network = pd.read_csv(shard / "network_rescale.csv")
            si = pd.read_csv(shard / "network_spatial_information.csv") if unit["kind"] == "zero" else pd.DataFrame()
            qc = pd.read_csv(shard / "task_qc.csv")
            n = 24 if unit["kind"] == "intact" else 9 * len(included_environments)
            exact_coverage(network, ["environment", "session", "module"], n * 3, slug)
            if (len(qc) != n or not qc.included.astype(str).str.lower().eq("true").all()
                    or not network.n_cells_finite.eq(100).all()):
                raise RuntimeError("Task or cell-count QC failed")
            expected_runs = ({(env, 0) for env in common.DEFORMED} |
                             {(env, s) for env in ENVS for s in range(1, 10)}) if unit["kind"] == "intact" else {
                             (env, s) for env in included_environments for s in range(1, 10)}
            if set(qc[["environment", "session"]].itertuples(index=False, name=None)) != expected_runs:
                raise RuntimeError("Missing environment/session run")
            if set(network[["environment", "session"]].itertuples(index=False, name=None)) != expected_runs:
                raise RuntimeError("Rescale network rows do not cover the planned runs")
            if not network.groupby(["environment", "session"]).module.agg(
                    lambda value: set(value) == {1, 2, 3}).all():
                raise RuntimeError("A run lacks one or more grid modules")
            if unit["kind"] == "zero":
                exact_coverage(si, ["environment", "session"], n, slug + " SI")
                if not si.n_cells_total.eq(unit["count"] + 300).all():
                    raise RuntimeError("Wrong SI total cell count")
                if set(si[["environment", "session"]].itertuples(index=False, name=None)) != expected_runs:
                    raise RuntimeError("Zero-control SI lacks an environment/session run")
            network_parts.append(network)
            if len(si):
                si_parts.append(si)
            qc_parts.append(qc)
            cell_manifest.append({**unit, "path": str((shard / "cell_rescale.csv.gz").relative_to(output)),
                                  **status["outputs"]["cell_rescale.csv.gz"]})
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            issues.append({"shard": slug, "reason": str(exc)})
    if issues:
        atomic_json({"status": "failed", "issues": issues}, output / "finalize_status.json")
        raise RuntimeError(f"{len(issues)} shards failed; see finalize_status.json")
    availability = pd.DataFrame(availability_rows)
    if len(availability) != 36 * len(ENVS) or availability.duplicated(
            ["count", "network_id", "pathway", "environment"]).any():
        raise RuntimeError("Zero-control availability audit is incomplete")
    if not availability.available.any():
        raise RuntimeError("No complete zero-control network/environment outputs; check the zero-root path")
    atomic_csv(availability, output / "zero_control_availability.csv")
    edge = pd.concat(network_parts, ignore_index=True)
    atomic_csv(edge, output / f"network_rescale_{METHOD['name'].lower().replace('-', '')}.csv")
    atomic_csv(pd.concat(si_parts, ignore_index=True) if si_parts else pd.DataFrame(columns=[
        "kind", "count", "network_id", "pathway", "environment", "session", "metric",
        "network_mean", "n_cells_finite", "n_cells_total"]),
        output / "network_spatial_information_zero.csv")
    atomic_csv(pd.concat(qc_parts, ignore_index=True), output / "task_qc.csv")
    atomic_csv(pd.DataFrame(cell_manifest), output / "cell_files_manifest.csv")
    old = pd.read_csv(root / "analysis/network_values_long.csv")
    if set(old.condition.unique()) != {count_condition(c) for c in COUNTS}:
        raise RuntimeError("Existing intact analysis conditions differ from expected fixed-delta set")
    # Grid rescale: first average the three equally sized modules within each network.
    grid = edge.loc[edge.kind.eq("intact") & edge.environment.isin(ENVS)].groupby(
        ["count", "network_id", "environment", "session"], as_index=False
    ).agg(network_mean=("network_mean", "mean"), n_modules=("module", "size"))
    if not grid.n_modules.eq(3).all():
        raise RuntimeError("Grid rescale requires all three modules")
    grid["metric"] = "rescale_factor"
    intact_parts = [grid[["count", "network_id", "environment", "session", "metric", "network_mean"]]]
    for metric, pop in (("delta_R_phase_x", "grid"),
                        ("delta_place_field_length_x_cm", "place"),
                        ("delta_place_phase_distance_x_cm", "place")):
        part = old.loc[old.metric.eq(metric) & old.population.eq(pop) &
                       old.environment.isin(ENVS) & old.session.between(0, 9)].copy()
        if pop == "grid":
            part = part.groupby(["parameter", "network_id", "environment", "session"], as_index=False).agg(
                network_mean=("network_mean", "mean"), n_modules=("module", "size"))
            if not part.n_modules.eq(3).all():
                raise RuntimeError("Grid phase requires three module means")
        part["count"] = part.parameter.astype(int)
        part["metric"] = metric
        intact_parts.append(part[["count", "network_id", "environment", "session", "metric", "network_mean"]])
    intact = pd.concat(intact_parts, ignore_index=True)
    keys = ["count", "network_id", "environment", "session", "metric"]
    exact_coverage(intact, keys, 4 * 4 * 2 * 10 * 4, "intact Fig5 curves")
    atomic_csv(intact, output / "intact_network_curves.csv")
    atomic_csv(group_curve(intact, ["count", "environment", "session", "metric"]),
               output / "intact_group_curves.csv")
    # Initial-exposure summary includes the 100-cm baseline reference point.
    initial_grid = edge.loc[edge.kind.eq("intact") & edge.session.eq(0)].groupby(
        ["count", "network_id", "environment"], as_index=False
    ).agg(network_mean=("network_mean", "mean"), n_modules=("module", "size"))
    initial_grid["metric"] = "rescale_factor"
    baseline_grid = initial_grid.loc[initial_grid.environment.eq("compressed_80")].copy()
    baseline_grid["environment"], baseline_grid["network_mean"] = "fam_100", 1.0
    initial_parts = [pd.concat([initial_grid, baseline_grid], ignore_index=True)[
        ["count", "network_id", "environment", "metric", "network_mean"]]]
    for metric, pop in (("delta_R_phase_x", "grid"),
                        ("delta_place_field_length_x_cm", "place"),
                        ("delta_place_phase_distance_x_cm", "place")):
        part = old.loc[old.metric.eq(metric) & old.population.eq(pop) & old.session.eq(0)].copy()
        if pop == "grid":
            part = part.groupby(["parameter", "network_id", "environment"], as_index=False).agg(
                network_mean=("network_mean", "mean"), n_modules=("module", "size"))
            if not part.n_modules.eq(3).all():
                raise RuntimeError("Initial grid phase lacks three modules")
        part["count"] = part.parameter.astype(int)
        part["metric"] = metric
        initial_parts.append(part[["count", "network_id", "environment", "metric", "network_mean"]])
    initial = pd.concat(initial_parts, ignore_index=True)
    exact_coverage(initial, ["count", "network_id", "environment", "metric"], 4 * 4 * 7 * 4,
                   "initial exposure")
    atomic_csv(initial, output / "initial_network_curves.csv")
    atomic_csv(group_curve(initial, ["count", "environment", "metric"]),
               output / "initial_group_curves.csv")
    # Fig6: intact session 0 is shared initial exposure; 0x controls begin at session 1.
    lr_audit = common.read_json(lr / "analysis/experiment_plan_audit.json")
    if lr_audit.get("status") != "pass":
        raise RuntimeError("nPC=100 learning-rate full-analysis audit did not pass")
    lr_old_path = lr / "analysis/network_values_long.csv"
    lr_old = pd.read_csv(lr_old_path)
    lr_provenance = common.read_json(lr / "analysis/provenance.json")
    if not str(lr_provenance.get("parameters", {}).get("grid_rescale", "")).startswith("Edge5-M:"):
        raise RuntimeError("nPC=100 full learning-rate analysis is not Edge5-M")
    lr_rescale_path = lr_old_path
    lr_edge = lr_old.loc[lr_old.population.eq("grid") & lr_old.metric.eq("rescale_factor")].copy()
    intact_si = pool_si(old)
    npc100_si = pool_si(lr_old)
    zero_si = pd.concat(si_parts, ignore_index=True) if si_parts else pd.DataFrame(columns=[
        "kind", "count", "network_id", "pathway", "environment", "session", "metric",
        "network_mean", "n_cells_finite", "n_cells_total"])
    available_zero = set(availability.loc[availability.available,
                                          ["count", "pathway", "network_id", "environment"]]
                         .itertuples(index=False, name=None))
    rows = []
    for count in COUNTS:
        for pathway in ("intact", *PATHWAYS):
            for env in ENVS:
                for net in NETWORKS:
                    if count > 100 and pathway != "intact" and (
                            count, pathway, net, env) not in available_zero:
                        continue
                    for session in range(10):
                        if pathway == "intact" or session == 0:
                            r = edge.loc[edge.kind.eq("intact") & edge["count"].eq(count) &
                                         edge.network_id.eq(net) & edge.environment.eq(env) &
                                         edge.session.eq(session)]
                            s = intact_si.loc[intact_si.condition.eq(count_condition(count)) &
                                              intact_si.network_id.eq(net) & intact_si.environment.eq(env) &
                                              intact_si.session.eq(session)]
                        elif count == 100:
                            condition = f"{pathway}_0p0x"
                            r = lr_edge.loc[lr_edge.condition.eq(condition) & lr_edge.network_id.eq(net) &
                                            lr_edge.environment.eq(env) & lr_edge.session.eq(session)]
                            s = npc100_si.loc[npc100_si.condition.eq(condition) & npc100_si.network_id.eq(net) &
                                              npc100_si.environment.eq(env) & npc100_si.session.eq(session)]
                        else:
                            r = edge.loc[edge.kind.eq("zero") & edge["count"].eq(count) &
                                         edge.pathway.eq(pathway) & edge.network_id.eq(net) &
                                         edge.environment.eq(env) & edge.session.eq(session)]
                            s = zero_si.loc[zero_si["count"].eq(count) & zero_si.pathway.eq(pathway) &
                                            zero_si.network_id.eq(net) & zero_si.environment.eq(env) &
                                            zero_si.session.eq(session)]
                        if len(r) != 3 or len(s) != 1:
                            raise RuntimeError(f"Missing Fig6 sources: {count}, {pathway}, {env}, {session}, {net}")
                        if not set(r.module) == {1, 2, 3}:
                            raise RuntimeError("Fig6 rescale lacks three modules")
                        if int(s.iloc[0].n_cells_total) != count + 300:
                            raise RuntimeError("Fig6 pooled SI cell count mismatch")
                        for metric, value, n_cells in (
                            ("rescale_factor", float(r.network_mean.mean()), 300),
                            ("spatial_information_bits_per_spike", float(s.iloc[0].network_mean),
                             int(s.iloc[0].n_cells_finite))):
                            rows.append({"count": count, "pathway": pathway, "network_id": net,
                                         "environment": env, "session": session, "metric": metric,
                                         "network_mean": value, "n_cells_finite": n_cells})
    fig6 = pd.DataFrame(rows)
    expected_fig6_rows = (4 * 4 * len(ENVS) * len(NETWORKS) -
                          int((~availability.available).sum())) * 10 * 2
    exact_coverage(fig6, ["count", "pathway", "network_id", "environment", "session", "metric"],
                   expected_fig6_rows, "Fig6")
    curve_coverage = fig6.groupby(["count", "pathway", "network_id", "environment", "metric"]).session.nunique()
    if not curve_coverage.eq(10).all():
        raise RuntimeError("A retained control curve lacks one or more sessions")
    atomic_csv(fig6, output / "fig6_network_curves.csv")
    fig6_group = group_curve(fig6, ["count", "pathway", "environment", "session", "metric"])
    if not fig6_group.groupby(["count", "pathway", "environment", "metric"]).n_networks.nunique().eq(1).all():
        raise RuntimeError("A plotted arm changes network sample size across sessions")
    atomic_csv(fig6_group, output / "fig6_group_curves.csv")
    atomic_csv(fig6_group.loc[fig6_group.session.eq(9) &
                               fig6_group.metric.eq("rescale_factor"),
                               ["count", "pathway", "environment", "n_networks"]],
               output / "fig6_network_counts.csv")
    provenance = {"method": METHOD, "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "source_roots": {"reference": str(source), "place_fixed": str(root),
                                   "place_lr_zero": str(zero), "npc100_lr": str(lr)},
                  "input_analysis": [str(root / "analysis"), str(lr / "analysis")],
                  "input_table_sha256": {
                      str(path): common.sha256(path) for path in (
                          root / "analysis/network_values_long.csv",
                          root / "analysis/task_integrity_qc.csv",
                          lr / "analysis/network_values_long.csv", lr_rescale_path)},
                  "grid_rescale": f"2.5-cm occupancy-normalized maps, sigma 1.5 bins; Methods scales at 2.5-cm steps; first/last {METHOD['edge_positions_per_side']} complete-overlap translations; Pearson all valid pixels, minimum 20",
                  "phase_and_place": "Existing matched-baseline Methods metrics; grid delta_R_phase_x averaged equally across three modules; place distance is abs(current movement displacement)-abs(baseline displacement)",
                  "spatial_information": "Production Skaggs bits/spike; all finite place and grid cells pooled within network; no imputation",
                  "unit": "Independent networks; zero-rate arms use only complete network/pathway/environment curves, so n may be 1-4; cells and modules are not independent replicates",
                  "uncertainty": "two-sided 95% Student-t interval across available network means when n>=2; n=1 is descriptive only with undefined CI",
                  "missing_controls": "A network/pathway/environment is omitted for all sessions 0-9 if any of its nine familiarization outputs is absent/incomplete; no imputation or replacement",
                  "session_zero": "shared initial exposure in all intact and 0x curves; not independently repeated",
                  "code_sha256": common.sha256(Path(__file__)),
                  "frozen_sha256": {name: common.sha256(common.LOCAL_ANALYSIS / name)
                                    for name in common.FROZEN_EDGE5_SHA256}}
    atomic_json(provenance, output / "provenance.json")
    atomic_json({"status": "complete", "method": METHOD,
                 "intact_network_rows": len(intact), "fig6_network_rows": len(fig6),
                 "zero_environment_networks_available": int(availability.available.sum()),
                 "zero_environment_networks_skipped": int((~availability.available).sum()),
                 "fig6_n_networks_min": int(fig6_group.n_networks.min()),
                 "fig6_n_networks_max": int(fig6_group.n_networks.max())},
                output / "finalize_status.json")
    print(f"Complete: {output}; {len(intact)} intact and {len(fig6)} Fig6 network rows", flush=True)


def main(edge_positions_per_side: int = 5) -> None:
    configure_method(edge_positions_per_side)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "worker", "finalize"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--source-v4", type=Path, default=runner.DEFAULT_SOURCE)
    parser.add_argument("--place-root", type=Path, default=runner.ROOTS["place_cells"])
    parser.add_argument("--zero-root", type=Path, default=DEFAULT_ZERO)
    parser.add_argument("--lr-root", type=Path, default=DEFAULT_LR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source, root, zero, lr = (path.resolve() for path in
                              (args.source_v4, args.place_root, args.zero_root, args.lr_root))
    output = (args.output or root / f"analysis_{METHOD['name'].lower().replace('-', '')}_with_lr_zero").resolve()
    if output in (source, root, zero, lr, root / "analysis") or any(
            output.is_relative_to(path / "networks") for path in (source, root, zero, lr)):
        parser.error("Output must be a new analysis directory, not simulation or prior analysis data")
    planned = units()
    if args.action == "plan":
        availability = zero_availability_report(source, root, zero)
        summaries = [{"count": count, "pathway": pathway, "environment": env,
                      "network_ids_available": [row["network_id"] for row in availability
                                                if row["count"] == count and row["pathway"] == pathway
                                                and row["environment"] == env and row["available"]],
                      "n_networks_available": sum(row["count"] == count and row["pathway"] == pathway
                                                   and row["environment"] == env and row["available"]
                                                   for row in availability)}
                     for count in COUNTS[1:] for pathway in PATHWAYS for env in ENVS]
        print(json.dumps({"task_count": len(planned), "intact_units": 16,
                          "zero_control_units": 36, "output": str(output), "method": METHOD,
                          "zero_control_availability": availability,
                          "zero_control_summary": summaries,
                          "inclusion_rule": "One network/pathway/environment retained only if sessions 01-09 all have complete manifests and recorded outputs; full worker QC follows",
                          "units": [{"task_index": i, **u} for i, u in enumerate(planned)]}, indent=2))
    elif args.action == "worker":
        if args.task_index is None or not 0 <= args.task_index < len(planned):
            parser.error(f"--task-index must be 0..{len(planned) - 1}")
        work(planned[args.task_index], source, root, zero, output)
    else:
        finish(source, root, zero, lr, output)


if __name__ == "__main__":
    main()
