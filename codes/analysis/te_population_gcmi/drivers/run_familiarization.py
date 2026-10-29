"""Run the frozen population-PCA TE estimator across familiarization sessions.

The historical single-trajectory/unconditioned configuration is retained.
Configurations with two trajectories use the production cross-fitted
behavioral correction; optional initial-exposure session 0 is included in the
same run as a time-course reference. PCA is always fitted on the two baseline
trajectories and frozen for every target session.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from te_population_gcmi.config import Config
from te_population_gcmi.data_simu_interface import build_id_outputs
from te_population_gcmi.io import load_manifest, load_trajectory_npz
from te_population_gcmi.pipeline import (
    _fit_pcas_from_manifest,
    _input_qc_rows,
    _load_group,
    _manifest_group_rows,
    _pca_variance_table,
    estimate_prepared_group,
    prepare_group,
)
from te_population_gcmi.segments import segmentation_qc_table


BASELINE = {"stage": "baseline", "environment": "fam_100", "session": 0}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve(value: str | Path, parent: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def environment_width(environment: str) -> float:
    prefix, _, value = str(environment).rpartition("_")
    if prefix not in {"compressed", "elongated"} or not value.isdigit():
        raise ValueError(f"Invalid deformation environment: {environment}")
    return int(value) / 100.0


def targets(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    familiarization = [
        {
            "stage": "familiarization",
            "environment": str(environment),
            "session": int(session),
        }
        for environment in settings["environments"]
        for session in settings["sessions"]
    ]
    if settings.get("include_initial_exposure_session0", False):
        initial = [
            {"stage": "initial_exposure", "environment": str(environment), "session": 0}
            for environment in settings["environments"]
        ]
        return initial + familiarization
    return familiarization


def build_manifest(
    network: str,
    settings: Mapping[str, Any],
    config_path: Path,
    destination: Path,
) -> Path:
    root = resolve(settings["data_root"], config_path.parent)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for trajectory in ("trajectory_00", "trajectory_01"):
        source = (
            root
            / network
            / "baseline"
            / "fam_100"
            / trajectory
            / "transient_activity.npz"
        )
        missing.extend([] if source.is_file() else [str(source)])
        rows.append(
            {
                "network_id": network,
                "stage": "baseline",
                "environment": "fam_100",
                "session": 0,
                "trajectory_id": trajectory,
                "width_m": 1.0,
                "height_m": 1.0,
                "deformed_axis": "x",
                "data_path": str(source.resolve()),
            }
        )
    condition = str(settings.get("conditioning", "unconditioned"))
    target_trajectories = (
        ("trajectory_00", "trajectory_01")
        if condition == "behavior_conditioned"
        else ("trajectory_00",)
    )
    for target in targets(settings):
        environment = str(target["environment"])
        session = int(target["session"])
        for trajectory in target_trajectories:
            if target["stage"] == "initial_exposure":
                source = (
                    root / network / "initial_exposure" / environment
                    / trajectory / "transient_activity.npz"
                )
            else:
                source = (
                    root / network / "familiarization" / environment / "intact"
                    / trajectory / f"session_{session:02d}" / "transient_activity.npz"
                )
                if not source.is_file():
                    alternative = (
                        root / network / "familiarization" / environment
                        / trajectory / f"session_{session:02d}"
                        / "transient_activity.npz"
                    )
                    if alternative.is_file():
                        source = alternative
            missing.extend([] if source.is_file() else [str(source)])
            rows.append(
                {
                    "network_id": network,
                    "stage": str(target["stage"]),
                    "environment": environment,
                    "session": session,
                    "trajectory_id": trajectory,
                    "width_m": environment_width(environment),
                    "height_m": 1.0,
                    "deformed_axis": "x",
                    "data_path": str(source.resolve()),
                }
            )
    if missing:
        raise FileNotFoundError(
            "Missing familiarization input files:\n" + "\n".join(missing)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def build_base_config(
    settings: Mapping[str, Any],
    config_path: Path,
    manifest: Path,
    destination: Path,
) -> Path:
    payload = deepcopy(read_json(resolve(settings["base_config"], config_path.parent)))
    reference = settings["reference"]
    payload["data"]["manifest_path"] = str(manifest)
    payload["data"]["array_keys"] = {
        "positions": "positions",
        "place": {"key": "place", "time_axis": 1},
        "grid_modules": {
            "module_1": {"key": "grid_m1", "time_axis": 1},
            "module_2": {"key": "grid_m2", "time_axis": 1},
            "module_3": {"key": "grid_m3", "time_axis": 1},
        },
        "dt_s": "record_dt_s",
    }
    payload["data"]["require_bvc"] = False
    condition = str(settings.get("conditioning", "unconditioned"))
    if condition not in {"unconditioned", "behavior_conditioned"}:
        raise ValueError(f"Unsupported familiarization conditioning: {condition}")
    payload["data"]["allow_single_trajectory_unconditioned"] = (
        condition == "unconditioned"
    )
    payload["data"]["minimum_trajectories_per_group"] = (
        2 if condition == "behavior_conditioned" else 1
    )
    payload["representation"]["primary_components"] = int(
        reference["pca_components"]
    )
    payload["representation"]["sensitivity_components"] = []
    payload["conditioning"]["primary"] = condition
    payload["conditioning"]["branches"] = [condition]
    payload["te"]["effect_value"] = "te_raw_bits"
    payload["te"]["compute_cv_score"] = False
    payload["te"]["lag_frames"] = [int(reference["lag_frames"])]
    payload["te"]["source_history_frames"] = [int(reference["source_history"])]
    payload["te"]["target_history_frames"] = [int(reference["target_history"])]
    payload["te"]["bias_shuffles"] = int(settings["nulls"]["bias_shuffles"])
    payload["te"]["minimum_pool_samples"] = int(
        settings["qc"]["minimum_pool_samples"]
    )
    payload["te"]["minimum_effective_samples_per_dimension"] = float(
        settings["qc"]["minimum_effective_samples_per_dimension"]
    )
    payload["te"]["select_parameters_on_baseline"] = False
    payload["cross_trajectory_null"]["enabled"] = False
    payload["cross_trajectory_null"]["repetitions"] = int(
        settings["nulls"]["cross_trajectory_repetitions"]
    )
    payload["statistics"]["primary_targets"] = targets(settings)
    # Config validation requires one execution branch even though this runner
    # invokes the shared low-level estimator directly.
    payload["execution"]["run_primary"] = True
    payload["execution"]["run_sensitivity"] = False
    payload["execution"]["fail_on_qc"] = False
    write_json(destination, payload)
    return destination


def prepare_network(
    network: str,
    settings: Mapping[str, Any],
    config_path: Path,
) -> tuple[Path, Path]:
    root = resolve(settings["output_root"], config_path.parent)
    shared = root / "shared"
    manifest = build_manifest(
        network,
        settings,
        config_path,
        shared / "manifests" / f"{network}.csv",
    )
    runtime = build_base_config(
        settings,
        config_path,
        manifest,
        shared / "configs" / f"{network}.json",
    )
    output = root / "default" / network
    return runtime, output


def _load_analysis_group(
    rows: pd.DataFrame,
    config: Config,
    manifest_dir: str | Path,
) -> Sequence[Any]:
    is_baseline = (
        rows.iloc[0]["stage"] == "baseline"
        and rows.iloc[0]["environment"] == "fam_100"
    )
    if is_baseline:
        return _load_group(rows, config, manifest_dir)
    return tuple(
        load_trajectory_npz(row, config, manifest_dir=manifest_dir)
        for _, row in rows.sort_values("trajectory_id", kind="mergesort").iterrows()
    )


def _target_mask(frame: pd.DataFrame, target: Mapping[str, Any]) -> pd.Series:
    return (
        frame["stage"].astype(str).eq(str(target["stage"]))
        & frame["environment"].astype(str).eq(str(target["environment"]))
        & frame["session"].astype(str).eq(str(target["session"]))
    )


def _baseline_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["stage"].astype(str).eq("baseline")
        & frame["environment"].astype(str).eq("fam_100")
        & frame["session"].astype(str).eq("0")
    )


def run_network(config_path: Path, output: Path, *, overwrite: bool) -> Path:
    started = time.perf_counter()
    payload = read_json(config_path)
    config = Config.from_dict(payload, source_path=config_path)
    manifest_path = Path(config.data["manifest_path"]).resolve()
    manifest = load_manifest(manifest_path, check_files=True)
    output.mkdir(parents=True, exist_ok=True)
    pcas, baseline_input_qc = _fit_pcas_from_manifest(manifest, config, output)
    _pca_variance_table(pcas).to_csv(
        output / "pca_explained_variance.csv", index=False
    )

    estimates_list: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = list(baseline_input_qc)
    segment_frames: list[pd.DataFrame] = []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    groups = list(_manifest_group_rows(manifest))
    reference = (
        int(config.representation["primary_components"]),
        int(config.te["lag_frames"][0]),
        int(config.te["source_history_frames"][0]),
        int(config.te["target_history_frames"][0]),
    )
    condition = str(config.conditioning["primary"])
    for index, (key, rows) in enumerate(groups, start=1):
        trajectories = _load_analysis_group(
            rows, config, manifest.attrs["manifest_dir"]
        )
        input_rows.extend(_input_qc_rows(trajectories))
        prepared = prepare_group(
            trajectories,
            pcas[str(trajectories[0].network_id)],
            config,
        )
        segment_frames.append(segmentation_qc_table(prepared.segmentations.values()))
        group_name = "__".join(str(value) for value in key)
        checkpoint = checkpoint_dir / f"{group_name}__reference.csv"
        if checkpoint.is_file() and not overwrite:
            estimates = pd.read_csv(checkpoint)
            action = "reuse"
        else:
            estimates, _, _ = estimate_prepared_group(
                prepared,
                config,
                analysis_set="familiarization",
                parameters=[reference],
                branches=[condition],
                n_bias_shuffles=0,
                cross_null_for_primary=False,
            )
            estimates.insert(0, "candidate_id", "reference")
            estimates.to_csv(checkpoint, index=False)
            action = "run"
        estimates_list.append(estimates)
        print(
            f"[familiarization {index}/{len(groups)}] {action} {group_name}",
            flush=True,
        )

    estimates = pd.concat(estimates_list, ignore_index=True)
    estimates.to_csv(
        output / "te_estimates.csv.gz", index=False, compression="gzip"
    )
    input_qc = pd.DataFrame(input_rows).drop_duplicates(
        ["network_id", "stage", "environment", "session", "trajectory_id"]
    )
    input_qc.to_csv(output / "input_qc.csv", index=False)
    pd.concat(segment_frames, ignore_index=True).drop_duplicates(
        ["network_id", "stage", "environment", "session", "segment_id"]
    ).to_csv(output / "segment_qc.csv.gz", index=False, compression="gzip")

    all_cells: list[pd.DataFrame] = []
    all_network_cells: list[pd.DataFrame] = []
    all_modules: list[pd.DataFrame] = []
    all_effects: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    for target in config.statistics["primary_targets"]:
        cells, network_cells, modules, effects = build_id_outputs(
            estimates,
            baseline=BASELINE,
            target=target,
            target_axis="x",
            analysis_set="familiarization",
            conditioning=condition,
        )
        for frame in (cells, network_cells, modules, effects):
            frame.insert(0, "candidate_id", "reference")
            frame.insert(1, "pca_components", reference[0])
            frame.insert(2, "lag_frames", reference[1])
            frame.insert(3, "source_history", reference[2])
            frame.insert(4, "target_history", reference[3])
        all_cells.append(cells)
        all_network_cells.append(network_cells)
        all_modules.append(modules)
        all_effects.append(effects)

        scoped = estimates.loc[_baseline_mask(estimates) | _target_mask(estimates, target)]
        temporal = pd.to_numeric(scoped["temporal_effective_n"], errors="coerce")
        required = pd.to_numeric(scoped["minimum_required_n"], errors="coerce")
        ratio = temporal / required
        effect = effects.iloc[0] if not effects.empty else None
        qc_rows.append(
            {
                "candidate_id": "reference",
                "pca_components": reference[0],
                "lag_frames": reference[1],
                "source_history": reference[2],
                "target_history": reference[3],
                "environment": str(target["environment"]),
                "session": int(target["session"]),
                "n_expected_te_rows": 48,
                "n_te_rows": int(len(scoped)),
                "n_te_rows_ok": int(scoped["status"].astype(str).eq("ok").sum()),
                "minimum_temporal_effective_n": float(temporal.min()),
                "minimum_required_n": float(required.min()),
                "minimum_effective_n_ratio": float(ratio.min()),
                "n_target_trajectories": int(
                    manifest.loc[_target_mask(manifest, target), "trajectory_id"].nunique()
                ),
                "n_valid_modules": (
                    int(effect["n_valid_modules"]) if effect is not None else 0
                ),
                "n_expected_modules": (
                    int(effect["n_expected_modules"]) if effect is not None else 3
                ),
                "candidate_complete": (
                    bool(effect["network_contrast_valid"])
                    if effect is not None
                    else False
                ),
            }
        )

    pd.concat(all_cells, ignore_index=True).to_csv(
        output / "id_state_axis_cells.csv.gz", index=False, compression="gzip"
    )
    pd.concat(all_network_cells, ignore_index=True).to_csv(
        output / "id_network_state_axis.csv", index=False
    )
    pd.concat(all_modules, ignore_index=True).to_csv(
        output / "id_module_effects.csv", index=False
    )
    effects = pd.concat(all_effects, ignore_index=True)
    effects.to_csv(output / "id_network_effects.csv", index=False)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(output / "candidate_qc.csv", index=False)
    write_json(
        output / "run_summary.json",
        {
            "status": "complete",
            "network_id": str(manifest["network_id"].iloc[0]),
            "stage": "familiarization_with_initial_exposure_reference"
            if any(int(target["session"]) == 0 for target in config.statistics["primary_targets"])
            else "familiarization",
            "environments": sorted(set(manifest.loc[~_baseline_mask(manifest), "environment"])),
            "sessions": sorted(
                int(value) for value in manifest.loc[~_baseline_mask(manifest), "session"].unique()
            ),
            "baseline": BASELINE,
            "reference": {
                "pca_components": reference[0],
                "lag_frames": reference[1],
                "source_history": reference[2],
                "target_history": reference[3],
            },
            "conditioning": condition,
            "n_target_sessions": int(len(qc)),
            "n_complete_target_sessions": int(qc["candidate_complete"].sum()),
            "elapsed_seconds": time.perf_counter() - started,
            "output_directory": str(output),
        },
    )
    return output


def _run_one(task: tuple[str, str, bool]) -> tuple[str, str]:
    config_value, output_value, overwrite = task
    config_path = Path(config_value)
    output = Path(output_value)
    summary = output / "run_summary.json"
    network = config_path.stem
    if summary.is_file() and not overwrite:
        return network, "reused_complete"
    run_network(config_path, output, overwrite=overwrite)
    return network, "complete"


def run_batch(
    settings: Mapping[str, Any],
    config_path: Path,
    *,
    workers: int,
    overwrite: bool,
) -> None:
    tasks: list[tuple[str, str, bool]] = []
    for network in settings["networks"]:
        runtime, output = prepare_network(str(network), settings, config_path)
        tasks.append((str(runtime), str(output), overwrite))
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            network, status = _run_one(task)
            print(f"[network {index}/{len(tasks)}] {network}: {status}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            network, status = future.result()
            print(f"[network {index}/{len(tasks)}] {network}: {status}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.familiarization.pca2.json",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    settings = read_json(config_path)
    if args.mode == "prepare":
        for network in settings["networks"]:
            runtime, output = prepare_network(str(network), settings, config_path)
            print(f"{network}: {runtime} -> {output}")
        return 0
    run_batch(
        settings,
        config_path,
        workers=max(1, int(args.workers)),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
