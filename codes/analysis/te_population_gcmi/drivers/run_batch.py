"""Run symmetric population-PCA TE for all data_simu networks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SENSITIVITY_DIR = PROJECT / "data_simu_parameter_sensitivity"
if str(SENSITIVITY_DIR) not in sys.path:
    sys.path.insert(0, str(SENSITIVITY_DIR))

from run_ofat_sensitivity import run as run_ofat


TRAJECTORIES = ("trajectory_00", "trajectory_01")


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


def build_raw_manifest(
    network: str,
    settings: Mapping[str, Any],
    config_path: Path,
    destination: Path,
) -> Path:
    data_root = resolve(settings["data_root"], config_path.parent)
    rows: list[dict[str, Any]] = []
    conditions = [("baseline", "fam_100", 1.0)] + [
        ("initial_exposure", environment, environment_width(environment))
        for environment in settings["environments"]
    ]
    missing: list[str] = []
    for stage, environment, width in conditions:
        for trajectory in TRAJECTORIES:
            source = (
                data_root
                / network
                / stage
                / environment
                / trajectory
                / "transient_activity.npz"
            )
            if not source.is_file():
                missing.append(str(source))
            rows.append(
                {
                    "network_id": network,
                    "stage": stage,
                    "environment": environment,
                    "session": 0,
                    "trajectory_id": trajectory,
                    "width_m": width,
                    "height_m": 1.0,
                    "deformed_axis": "x",
                    "data_path": str(source.resolve()),
                }
            )
    if missing:
        raise FileNotFoundError(
            "Missing transient activity files:\n" + "\n".join(missing)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def build_raw_base_config(
    settings: Mapping[str, Any],
    config_path: Path,
    destination: Path,
) -> Path:
    payload = deepcopy(
        read_json(resolve(settings["base_config"], config_path.parent))
    )
    payload["data"]["array_keys"] = {
        "positions": "positions",
        "place": {"key": "place", "time_axis": 1},
        "grid_modules": {
            "module_1": {"key": "grid_m1", "time_axis": 1},
            "module_2": {"key": "grid_m2", "time_axis": 1},
            "module_3": {"key": "grid_m3", "time_axis": 1}
        },
        "dt_s": "record_dt_s"
    }
    payload["data"]["require_bvc"] = False
    payload["data"]["minimum_trajectories_per_group"] = 2
    write_json(destination, payload)
    return destination


def prepare_network_config(
    network: str,
    mode: str,
    settings: Mapping[str, Any],
    config_path: Path,
) -> tuple[Path, Path]:
    output_root = resolve(settings["output_root"], config_path.parent)
    shared = output_root / "shared"
    manifest = build_raw_manifest(
        network,
        settings,
        config_path,
        shared / "manifests" / f"{network}.csv",
    )
    base_config = build_raw_base_config(
        settings,
        config_path,
        shared / "base_config.raw_data_simu.json",
    )
    output = output_root / mode / network
    ranges_key = (
        "default_ranges" if mode == "default" else "sensitivity_ranges"
    )
    payload = {
        "schema_version": 1,
        "network_id": network,
        "prepared_manifests": {
            environment: str(manifest)
            for environment in settings["environments"]
        },
        "base_config": str(base_config),
        "conditioning": str(settings["conditioning"]),
        "reference": deepcopy(settings["reference"]),
        "ranges": deepcopy(settings[ranges_key]),
        "qc": deepcopy(settings["qc"]),
        "sensitivity_nulls": deepcopy(settings["sensitivity_nulls"]),
        "output_directory": str(output),
    }
    network_config = shared / "configs" / mode / f"{network}.json"
    write_json(network_config, payload)
    return network_config, output


def _run_one(task: tuple[str, str, str, bool]) -> tuple[str, str]:
    network, config_value, output_value, overwrite = task
    output = Path(output_value)
    summary = output / "run_summary.json"
    if summary.is_file() and not overwrite:
        return network, "reused_complete"
    run_ofat(Path(config_value), output, overwrite=overwrite)
    return network, "complete"


def seed_reference_checkpoints(
    network: str,
    settings: Mapping[str, Any],
    config_path: Path,
) -> int:
    """Reuse the identical default reference estimates in the OFAT output."""

    root = resolve(settings["output_root"], config_path.parent)
    source = root / "default" / network / "checkpoints"
    destination = root / "sensitivity" / network / "checkpoints"
    if not source.is_dir():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for checkpoint in source.glob("*__reference.csv"):
        target = destination / checkpoint.name
        if target.is_file():
            continue
        shutil.copy2(checkpoint, target)
        copied += 1
    return copied


def run_batch(
    mode: str,
    settings: Mapping[str, Any],
    config_path: Path,
    *,
    workers: int,
    overwrite: bool,
) -> None:
    tasks: list[tuple[str, str, str, bool]] = []
    for network in settings["networks"]:
        if mode == "sensitivity" and not overwrite:
            copied = seed_reference_checkpoints(
                str(network), settings, config_path
            )
            if copied:
                print(
                    f"[sensitivity] {network}: reused {copied} "
                    "default reference checkpoints",
                    flush=True,
                )
        network_config, output = prepare_network_config(
            str(network), mode, settings, config_path
        )
        tasks.append(
            (str(network), str(network_config), str(output), overwrite)
        )
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            network, status = _run_one(task)
            print(
                f"[{mode} {index}/{len(tasks)}] {network}: {status}",
                flush=True,
            )
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, task): task[0] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            network = futures[future]
            try:
                _, status = future.result()
            except Exception as exc:
                print(
                    f"[{mode} {completed}/{len(tasks)}] "
                    f"{network}: FAILED: {exc}",
                    flush=True,
                )
                raise
            print(
                f"[{mode} {completed}/{len(tasks)}] {network}: {status}",
                flush=True,
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "default", "sensitivity"))
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    settings = read_json(config_path)
    if args.mode == "prepare":
        for network in settings["networks"]:
            for mode in ("default", "sensitivity"):
                config, output = prepare_network_config(
                    str(network), mode, settings, config_path
                )
                print(f"{network} {mode}: {config} -> {output}")
        return 0
    run_batch(
        args.mode,
        settings,
        config_path,
        workers=max(1, int(args.workers)),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
