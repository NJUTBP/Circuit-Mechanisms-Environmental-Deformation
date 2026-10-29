"""Build experiment-specific plans, run isolated shards, and finalize tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import common

CLUSTER_PIPELINE = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = CLUSTER_PIPELINE.parent.parent / "run" / "outputs" / "independent_networks"
ROOTS = {
    "learning_rate": CLUSTER_PIPELINE.parent.parent / "run" / "outputs" / "learning_rates",
    "place_cells": CLUSTER_PIPELINE.parent.parent / "run" / "outputs" / "place_cell_count",
    "alpha": CLUSTER_PIPELINE.parent.parent / "run" / "outputs" / "grid_spacing",
}
LR_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
LR_PATHWAYS = ("W_bp", "W_pg", "W_gp")
PC_COUNTS = (100, 200, 300, 400)
ALPHAS = (0.08, 0.09, 0.10, 0.11, 0.12)
BASELINE_METRICS = (
    "grid_spacing_cm", "grid_score", "field_length_x_cm", "field_length_y_cm",
    "spatial_information_bits_per_spike", "peak_rate_hz", "mean_rate_hz",
)
RESPONSE_METRICS = (
    "rescale_factor", "delta_R_phase_x", "delta_R_phase_y",
    "delta_place_field_length_x_cm", "delta_place_field_length_y_cm",
    "delta_place_phase_distance_x_cm", "delta_place_phase_distance_y_cm",
    "delta_spatial_information_bits_per_spike", "delta_peak_rate_hz",
    "delta_mean_rate_hz",
)


def _label(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _unit(
    slug: str, network_id: int, conditions: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"slug": slug, "network_id": network_id, "conditions": conditions}


def plan(experiment: str, source: Path, root: Path) -> list[dict[str, Any]]:
    units = []
    if experiment == "learning_rate":
        for pathway in LR_PATHWAYS:
            for multiplier in LR_MULTIPLIERS:
                for network_id in range(4):
                    condition = f"{pathway}_{_label(multiplier)}x"
                    entry = {
                        "id": condition, "parameter": multiplier, "network_id": network_id,
                        "root": root if multiplier != 1.0 else source, "n_place": 100,
                        "baseline_root": source,
                        "variant": condition if multiplier != 1.0 else "intact",
                        "config": (root / "config.learning_rate_control.generated.json"
                                   if multiplier != 1.0 else source / "config.resolved.json"),
                    }
                    units.append(_unit(condition, network_id, [entry]))
    elif experiment == "place_cells":
        for n_place in PC_COUNTS:
            condition = (
                "npc100_reference" if n_place == 100 else f"npc{n_place}_delta_fixed"
            )
            condition_root = source if n_place == 100 else root / condition
            config = (
                source / "config.resolved.json" if n_place == 100
                else root / "configs" / f"{condition}.json"
            )
            for network_id in range(4):
                units.append(_unit(condition, network_id, [{
                    "id": condition, "parameter": float(n_place),
                    "network_id": network_id, "root": condition_root,
                    "n_place": n_place, "variant": "intact", "config": config,
                }]))
    elif experiment == "alpha":
        for alpha in ALPHAS:
            condition = f"alpha1_{alpha:.2f}".replace(".", "p")
            condition_root = source if np.isclose(alpha, 0.1) else root / condition
            config = (
                source / "config.resolved.json" if np.isclose(alpha, 0.1)
                else root / "configs" / f"{condition}.json"
            )
            for network_id in range(4):
                units.append(_unit(condition, network_id, [{
                    "id": condition, "parameter": alpha, "network_id": network_id,
                    "root": condition_root, "n_place": 100,
                    "variant": "intact", "config": config,
                }]))
    else:
        raise ValueError(experiment)
    return units


def runs_for(experiment: str, condition: dict[str, Any]) -> list[common.Run]:
    base = dict(
        condition=condition["id"], parameter=float(condition["parameter"]),
        network_id=int(condition["network_id"]), root=Path(condition["root"]),
        n_place=int(condition["n_place"]), variant=str(condition["variant"]),
    )
    reference_base = {
        **base, "root": Path(condition.get("baseline_root", condition["root"])),
        "variant": "intact",
    }
    runs = [common.Run(stage="baseline", environment="fam_100", session=0, **reference_base)]
    environments = common.FAM_ENVS if experiment == "learning_rate" else common.DEFORMED
    runs += [
        common.Run(stage="initial_exposure", environment=env, session=0, **reference_base)
        for env in environments
    ]
    if experiment in ("learning_rate", "place_cells"):
        runs += [
            common.Run(stage="familiarization", environment=env, session=session, **base)
            for env in common.FAM_ENVS for session in range(1, 10)
        ]
    return runs


def _expected_manifest_audit(experiment: str, condition: dict[str, Any],
                             verify_hash: bool) -> dict[str, Any]:
    init_root = Path(condition.get("baseline_root", condition["root"]))
    network = init_root / "networks" / f"network_{condition['network_id']:03d}"
    config_path = Path(condition["config"])
    result = {
        "condition": condition["id"], "network_id": condition["network_id"],
        "stage": "init", "environment": "", "session": 0, "trajectory_id": -1,
        "group_path": "", "run_path": "", "included": False,
        "reason": "", "manifest_status": "", "manifest_run_sha256": "",
        "n_cells": 0, "n_nonfinite_rates": 0, "place_shape": "",
        "grid_shape": "", "positions_shape": "",
        "trajectory_sha256": "", "bvc_sha256": "",
    }
    if experiment == "learning_rate":
        config_path = init_root / "config.resolved.json"
    if not config_path.is_file():
        result["reason"] = "missing_generated_config"
        return result
    payload = common.read_json(config_path)
    session = int(payload["initialization"]["sessions"])
    directory = network / "init" / f"session_{session:02d}"
    result["group_path"] = str(directory)
    result["run_path"] = str(directory)
    manifest = directory / "manifest.json"
    checkpoint = directory / "checkpoint.npz"
    if not manifest.is_file() or not checkpoint.is_file():
        result["reason"] = "missing_init_manifest_or_checkpoint"
        return result
    try:
        body = common.read_json(manifest)
        result["manifest_status"] = str(body.get("status", ""))
        result["manifest_run_sha256"] = str(body.get("run_sha256", ""))
        record = common._manifest_record(body, directory, checkpoint)
        if body.get("status") != "complete" or record is None:
            result["reason"] = "incomplete_init_manifest"
            return result
        if checkpoint.stat().st_size != int(record.get("bytes", -1)):
            result["reason"] = "init_checkpoint_size_mismatch"
            return result
        if verify_hash and common.sha256(checkpoint) != record.get("sha256"):
            result["reason"] = "init_checkpoint_sha256_mismatch"
            return result
        expected_hash = common.read_json(init_root / "config.resolved.json")["_resolved"]["run_sha256"]
        if body.get("run_sha256") != expected_hash:
            result["reason"] = "init_run_hash_mismatch"
            return result
        if experiment == "learning_rate" and Path(condition["root"]) != init_root:
            copied = Path(condition["root"]) / "networks" / f"network_{condition['network_id']:03d}" / "init" / "bvc_tuning.npz"
            original = network / "init" / "bvc_tuning.npz"
            if not copied.is_file() or (verify_hash and common.sha256(copied) != common.sha256(original)):
                result["reason"] = "learning_rate_copied_bvc_missing_or_changed"
                return result
        result["included"] = True
        result["reason"] = "complete"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result["reason"] = f"corrupt_init:{exc}"
    return result


def _task_map_match_count(run: common.Run) -> int:
    path = run.root / "task_maps" / f"{run.stage}.json"
    if not path.is_file():
        return 0
    rows = common.read_json(path)
    return sum(
        int(row.get("network_id", -1)) == run.network_id
        and row.get("environment") == run.environment
        and (run.stage != "familiarization" or row.get("variant") == run.variant)
        for row in rows
    )


def run_unit(
    experiment: str, unit: dict[str, Any], output: Path,
    *, verify_hash: bool, check_inputs: bool,
) -> None:
    audit_cache: dict[str, tuple[dict[str, Any], pd.DataFrame | None]] = {}
    map_cache: dict[str, dict[int, np.ndarray]] = {}
    rescale_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    qc_rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    for condition in unit["conditions"]:
        config_path = Path(condition["config"])
        if not config_path.is_file():
            for run in runs_for(experiment, condition):
                qc_rows.append({**run.identity, "included": False, "included_comparison": False,
                                "reason": "missing_condition_config", "group_path": str(run.group_dir),
                                "run_path": str(run.run_dir)})
            continue
        common.check_config(
            config_path, int(condition["n_place"]),
            alpha=float(condition["parameter"]) if experiment == "alpha" else None,
            fixed_delta=experiment == "place_cells",
        )
        init_qc = _expected_manifest_audit(experiment, condition, verify_hash)
        qc_rows.append(init_qc)
        runs = runs_for(experiment, condition)
        baseline_run = runs[0]
        baseline_frame: pd.DataFrame | None = None
        baseline_maps: dict[int, np.ndarray] | None = None
        for run in runs:
            key = str(run.run_dir)
            if key in audit_cache:
                original_qc, original_cells = audit_cache[key]
                qc = {**original_qc, **run.identity}
                cells = original_cells
            else:
                qc, cells = common.audit_run(
                    run, verify_hash=verify_hash, check_inputs=check_inputs
                )
                audit_cache[key] = qc, cells
            qc = dict(qc)
            qc["included_comparison"] = False
            try:
                qc["task_map_matches"] = _task_map_match_count(run)
            except (OSError, ValueError, json.JSONDecodeError):
                qc["task_map_matches"] = -1
            if qc["task_map_matches"] != 1:
                qc["included"] = False
                qc["reason"] = "missing_or_duplicate_task_map_entry"
                cells = None
            if not init_qc["included"]:
                qc["included"] = False
                qc["reason"] = f"init_unavailable:{init_qc['reason']}"
                cells = None
            if run.stage == "baseline" and cells is not None:
                baseline_frame = cells
                if key not in map_cache:
                    try:
                        map_cache[key] = common.load_selected_maps(
                            run.run_dir / "transient_activity.npz", 100.0
                        )
                    except (OSError, ValueError, KeyError) as exc:
                        qc["included"] = False
                        qc["reason"] = f"baseline_edge5m_map_error:{exc}"
                baseline_maps = map_cache.get(key)
            if cells is not None and baseline_frame is not None and baseline_maps is not None:
                try:
                    derived = common.derive_cells(
                        run, cells, baseline_frame, baseline_maps, rescale_cache
                    )
                    cell_frames.append(derived)
                    qc["included_comparison"] = True
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    qc["reason"] = f"metric_error:{exc}"
            elif cells is not None:
                qc["reason"] = "complete_raw_but_baseline_unavailable"
            qc_rows.append(qc)
    combined = pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame()
    common.write_shard(
        output, str(unit["slug"]), int(unit["network_id"]),
        combined, pd.DataFrame(qc_rows),
    )
    print(
        f"{experiment}: {unit['slug']} network {unit['network_id']:03d}: "
        f"{len(qc_rows)} QC rows, {len(combined)} analyzed cells", flush=True,
    )


def pairing_audit(network: pd.DataFrame, qc: pd.DataFrame, experiment: str) -> pd.DataFrame:
    reference = {
        "place_cells": "npc100_reference",
        "alpha": "alpha1_0p10",
    }.get(experiment)
    conditions = sorted(network["condition"].dropna().unique())
    rows = []
    for condition in conditions:
        ref = (
            f"{condition.rsplit('_', 1)[0]}_1p0x"
            if experiment == "learning_rate" else reference
        )
        if condition == ref:
            continue
        left = qc.loc[
            qc.condition.eq(condition) & qc.included_comparison.eq(True)
            & qc.stage.ne("init")
        ].copy()
        right = qc.loc[
            qc.condition.eq(ref) & qc.included_comparison.eq(True)
            & qc.stage.ne("init")
        ].copy()
        keys = ["network_id", "stage", "environment", "session"]
        matches = left.merge(
            right[keys + ["trajectory_sha256", "bvc_sha256"]],
            on=keys, suffixes=("_condition", "_reference"), validate="one_to_one",
        )
        valid = matches.loc[
            matches["trajectory_sha256_condition"].ne("")
            & matches["trajectory_sha256_reference"].ne("")
            & matches["bvc_sha256_condition"].ne("")
            & matches["bvc_sha256_reference"].ne("")
        ]
        equal = (
            valid["trajectory_sha256_condition"].eq(valid["trajectory_sha256_reference"])
            & valid["bvc_sha256_condition"].eq(valid["bvc_sha256_reference"])
        )
        rows.append({
            "condition": condition, "reference_condition": ref,
            "n_matched_runs": len(matches), "n_verified_runs": len(valid),
            "n_matching_seed_and_trajectory": int(equal.sum()),
            "paired_allowed": bool(len(valid) > 0 and equal.all()
                                   and len(valid) == len(matches)),
            "basis": "same network index, BVC file SHA-256 and trajectory SHA-256",
        })
    return pd.DataFrame(rows)


def _source_file_manifest(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in sorted(set(qc.get("group_path", pd.Series(dtype=str)).dropna().astype(str))):
        if not group:
            continue
        manifest_path = Path(group) / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = common.read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({"kind": "manifest", "path": str(manifest_path),
                     "bytes": manifest_path.stat().st_size,
                     "sha256": common.sha256(manifest_path), "group_path": group})
        for kind in ("inputs", "outputs"):
            for record in payload.get(kind, []):
                path = Path(str(record.get("path", "")))
                if not path.is_absolute():
                    path = Path(group) / path
                rows.append({"kind": kind[:-1], "path": str(path),
                             "bytes": record.get("bytes", ""),
                             "sha256": record.get("sha256", ""),
                             "group_path": group})
    return pd.DataFrame(rows).drop_duplicates()


def _experiment_plan_audit(experiment: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    roots = {Path(condition["root"]) for unit in units for condition in unit["conditions"]}
    source = next(
        Path(condition.get("baseline_root", condition["root"]))
        for unit in units for condition in unit["conditions"]
        if condition["id"] in ("npc100_reference", "alpha1_0p10", "W_bp_1p0x")
    )
    condition_root = next(path for path in roots if path != source)
    root = condition_root if experiment == "learning_rate" else condition_root.parent
    filename = {
        "learning_rate": "learning_rate_control_prepared.json",
        "place_cells": "fixed_only_downstream_manifest.json",
        "alpha": "sensitivity_manifest.json",
    }[experiment]
    path = root / filename
    issues = []
    body: dict[str, Any] = {}
    if not path.is_file():
        issues.append(f"missing {path}")
    else:
        try:
            body = common.read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"invalid {path}: {exc}")
    if body and body.get("status") != "ready":
        issues.append("preparation manifest is not ready")
    if body and experiment == "learning_rate":
        if int(body.get("task_count", -1)) != 120:
            issues.append("learning-rate preparation task_count is not 120")
        expected = {f"{pathway}_{_label(value)}x"
                    for pathway in LR_PATHWAYS for value in LR_MULTIPLIERS if value != 1.0}
        observed = {str(item.get("id")) for item in body.get("variants", [])}
        if expected != observed:
            issues.append("learning-rate preparation variants differ from 15 expected arms")
    if body and experiment == "place_cells":
        expected_counts = {"audit_init": 3, "baseline": 12,
                           "initial_exposure": 72, "familiarization": 24}
        if body.get("task_counts") != expected_counts:
            issues.append("fixed-only task counts differ from expected")
        observed = {str(item.get("condition_id")) for item in body.get("conditions", [])}
        if observed != {"npc200_delta_fixed", "npc300_delta_fixed", "npc400_delta_fixed"}:
            issues.append("fixed-only manifest contains missing or non-fixed conditions")
    if body and experiment == "alpha":
        expected_counts = {"init": 16, "baseline": 16, "initial_exposure": 96}
        for stage, wanted in expected_counts.items():
            if int(body.get("task_counts", {}).get(stage, -1)) != wanted:
                issues.append(f"alpha {stage} task count differs from {wanted}")
        observed = {str(item.get("condition_id")) for item in body.get("conditions", [])}
        if observed != {"alpha1_0p08", "alpha1_0p09", "alpha1_0p11", "alpha1_0p12"}:
            issues.append("alpha manifest condition IDs differ from expected")
    for condition_root in sorted(roots):
        if not (condition_root / "seed_ledger.csv").is_file():
            issues.append(f"missing seed ledger: {condition_root}")
    return {"experiment": experiment, "path": str(path), "status": "pass" if not issues else "issues",
            "issues": issues, "manifest_sha256": common.sha256(path) if path.is_file() else ""}


def finalize(experiment: str, units: list[dict[str, Any]], output: Path) -> None:
    qc_parts = []
    network_parts = []
    cell_index = []
    for unit in units:
        shard = output / "shards" / f"{unit['slug']}__network_{unit['network_id']:03d}"
        qc_path = shard / "task_qc.csv"
        network_path = shard / "network_values.csv"
        cell_path = shard / "cell_values.csv.gz"
        if qc_path.is_file():
            qc_parts.append(pd.read_csv(qc_path, keep_default_na=False))
        else:
            for condition in unit["conditions"]:
                for run in runs_for(experiment, condition):
                    qc_parts.append(pd.DataFrame([{**run.identity, "included": False,
                                                    "included_comparison": False,
                                                    "reason": "analysis_shard_missing",
                                                    "group_path": str(run.group_dir)}]))
        if network_path.is_file() and network_path.stat().st_size:
            try:
                part = pd.read_csv(network_path)
                if len(part):
                    network_parts.append(part)
            except pd.errors.EmptyDataError:
                pass
        if cell_path.is_file():
            cell_index.append({
                "condition_or_shard": unit["slug"], "network_id": unit["network_id"],
                "path": str(cell_path.relative_to(output)),
                "bytes": cell_path.stat().st_size, "sha256": common.sha256(cell_path),
            })
    qc = pd.concat(qc_parts, ignore_index=True) if qc_parts else pd.DataFrame()
    network = pd.concat(network_parts, ignore_index=True) if network_parts else pd.DataFrame()
    for field in ("included", "included_comparison"):
        if field in qc:
            qc[field] = qc[field].astype(str).str.lower().eq("true")
    output.mkdir(parents=True, exist_ok=True)
    plan_audit = _experiment_plan_audit(experiment, units)
    (output / "experiment_plan_audit.json").write_text(
        json.dumps(plan_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    qc.to_csv(output / "task_integrity_qc.csv", index=False)
    pd.DataFrame(cell_index).to_csv(output / "cell_files_manifest.csv", index=False)
    _source_file_manifest(qc).to_csv(output / "source_file_manifest.csv", index=False)
    completeness = (
        qc.loc[qc.stage.ne("init")]
        .groupby(["condition", "stage", "environment", "session"], dropna=False)
        .agg(n_expected_runs=("network_id", "size"),
             n_manifest_complete=("included", "sum"),
             n_analyzed_runs=("included_comparison", "sum"),
             reasons=("reason", lambda values: " | ".join(sorted(set(map(str, values)))))
             )
        .reset_index()
    )
    completeness.to_csv(output / "condition_completeness.csv", index=False)
    if network.empty:
        raise RuntimeError("No complete analysis shards; task_integrity_qc.csv was still written")
    if network.duplicated(
        ["condition", "network_id", "environment", "session", "population", "module", "metric"]
    ).any():
        raise RuntimeError("Duplicate logical network-level analysis rows")
    network.to_csv(output / "network_values_long.csv", index=False)
    summary = common.group_summary(network)
    summary.to_csv(output / "group_summary_long.csv", index=False)
    pairing = pairing_audit(network, qc, experiment)
    pairing.to_csv(output / "pairing_audit.csv", index=False)
    if experiment == "learning_rate":
        contrast_parts = []
        for pathway in LR_PATHWAYS:
            sub = network.loc[
                network.condition.str.startswith(pathway + "_")
                & network.environment.isin(common.FAM_ENVS)
                & network.session.eq(9)
            ]
            contrasts = common.paired_contrasts(
                sub, f"{pathway}_1p0x", RESPONSE_METRICS
            )
            if len(contrasts):
                contrasts["pathway"] = pathway
                contrast_parts.append(contrasts)
        contrasts = pd.concat(contrast_parts, ignore_index=True) if contrast_parts else pd.DataFrame()
    else:
        reference = "npc100_reference" if experiment == "place_cells" else "alpha1_0p10"
        parts = [
            common.paired_contrasts(
                network.loc[network.environment.eq("fam_100") & network.session.eq(0)],
                reference, BASELINE_METRICS,
            ),
            common.paired_contrasts(
                network.loc[network.environment.isin(common.DEFORMED) & network.session.eq(0)],
                reference, RESPONSE_METRICS,
            ),
        ]
        if experiment == "place_cells":
            parts.append(common.paired_contrasts(
                network.loc[network.environment.isin(common.FAM_ENVS) & network.session.eq(9)],
                reference, RESPONSE_METRICS,
            ))
        contrasts = pd.concat([part for part in parts if len(part)], ignore_index=True) if any(
            len(part) for part in parts
        ) else pd.DataFrame()
    if len(contrasts):
        allowed = pairing.set_index("condition")["paired_allowed"].to_dict() if len(pairing) else {}
        contrasts["pairing_verified"] = contrasts["condition"].map(allowed).fillna(False)
        contrasts["test"] = np.where(
            contrasts["pairing_verified"], "paired_network_t", "independent_network_welch_t"
        )
        unpaired = ~contrasts["pairing_verified"]
        for target, source in (
            ("mean_paired_difference", "unpaired_mean_difference"),
            ("ci95_low", "unpaired_ci95_low"), ("ci95_high", "unpaired_ci95_high"),
            ("t", "unpaired_t"), ("df", "unpaired_df"),
            ("p_two_sided", "unpaired_p_two_sided"),
        ):
            contrasts.loc[unpaired, target] = contrasts.loc[unpaired, source].to_numpy()
        contrasts = common.add_bh_q(contrasts)
    contrasts.to_csv(output / "planned_network_contrasts.csv", index=False)
    config_paths = sorted(
        {Path(condition["config"]) for unit in units for condition in unit["conditions"]}
        | {Path(condition["root"]) / "config.resolved.json"
           for unit in units for condition in unit["conditions"]}
        | {Path(condition["root"]) / "seed_ledger.csv"
           for unit in units for condition in unit["conditions"]}
        | {Path(condition["root"]) / "task_counts.json"
           for unit in units for condition in unit["conditions"]}
    )
    common.provenance(output, experiment, config_paths, qc, pairing)
    print(
        f"{experiment}: {int(qc.get('included_comparison', pd.Series(dtype=bool)).eq(True).sum())} "
        f"complete paired-data runs, {len(network)} network values, "
        f"{len(summary)} group rows; output={output}", flush=True,
    )


def cli(experiment: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "worker", "finalize"])
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--source-v4", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root", type=Path, default=ROOTS[experiment])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-hash", action="store_true",
                        help="Local pilot only; cluster submissions must verify hashes")
    parser.add_argument("--skip-input-check", action="store_true",
                        help="Local mirror pilot only; cluster submissions must check manifest inputs")
    args = parser.parse_args()
    common.verify_edge5_sources()
    source, root = args.source_v4.resolve(), args.root.resolve()
    output = (args.output or root / "analysis").resolve()
    units = plan(experiment, source, root)
    if args.action == "plan":
        print(json.dumps({
            "experiment": experiment, "task_count": len(units),
            "output": str(output), "source_v4": str(source), "root": str(root),
            "units": [{"task_index": index, "slug": unit["slug"],
                       "network_id": unit["network_id"],
                       "conditions": [c["id"] for c in unit["conditions"]]}
                      for index, unit in enumerate(units)],
        }, ensure_ascii=False, indent=2))
    elif args.action == "worker":
        if args.task_index is None or not 0 <= args.task_index < len(units):
            parser.error(f"--task-index must be in 0..{len(units)-1}")
        run_unit(
            experiment, units[args.task_index], output,
            verify_hash=not args.no_hash, check_inputs=not args.skip_input_check,
        )
    else:
        finalize(experiment, units, output)
