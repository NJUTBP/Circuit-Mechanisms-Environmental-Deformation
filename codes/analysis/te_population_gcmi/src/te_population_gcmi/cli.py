"""Command-line interface for validation, analysis, and synthetic checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .config import Config
from .io import iter_trajectories, load_manifest
from .pipeline import run_analysis, validate_manifest_design
from .synthetic import run_synthetic_check


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="te-gcmi",
        description=(
            "Standalone behavior-adjusted population transfer-entropy analysis"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate the JSON configuration, manifest, and referenced NPZ files",
    )
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument(
        "--manifest",
        type=Path,
        help="override data.manifest_path from the configuration",
    )
    validate.add_argument(
        "--skip-file-check",
        action="store_true",
        help="validate manifest metadata without requiring referenced NPZ files",
    )

    run = subparsers.add_parser("run", help="run the complete prespecified analysis")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--manifest",
        type=Path,
        help="override data.manifest_path from the configuration",
    )
    run.add_argument("--output", required=True, type=Path)
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known output files in a non-empty output directory",
    )

    synthetic = subparsers.add_parser(
        "synthetic-check",
        help="run data-independent estimator software checks",
    )
    synthetic.add_argument(
        "--output",
        type=Path,
        help="JSON path or directory in which synthetic_check.json is written",
    )
    synthetic.add_argument("--seed", type=int, default=20260909)
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_referenced_arrays(manifest, config: Config) -> dict[str, object]:
    """Load every NPZ once and validate frozen-PCA dimensional compatibility."""

    requested_components = max(
        [
            int(config.representation["primary_components"]),
            *[
                int(value)
                for value in config.representation.get(
                    "sensitivity_components", []
                )
            ],
        ]
    )
    signatures: dict[str, tuple[int, tuple[tuple[str, int], ...]]] = {}
    baseline_samples: dict[str, int] = {}
    minimum_place_cells: int | None = None
    minimum_grid_cells: int | None = None
    n_loaded = 0
    for trajectory in iter_trajectories(manifest, config):
        n_loaded += 1
        signature = (
            int(trajectory.place_activity.shape[1]),
            tuple(
                sorted(
                    (name, int(values.shape[1]))
                    for name, values in trajectory.grid_activity.items()
                )
            ),
        )
        previous = signatures.setdefault(trajectory.network_id, signature)
        if previous != signature:
            raise ValueError(
                "Population feature dimensions must be constant across all "
                f"conditions of network {trajectory.network_id!r}; found "
                f"{previous} and {signature}"
            )
        place_cells = signature[0]
        grid_cells = min(count for _, count in signature[1])
        if place_cells < requested_components or grid_cells < requested_components:
            raise ValueError(
                f"Network {trajectory.network_id!r} requests "
                f"{requested_components} PCA components but a population has only "
                f"{min(place_cells, grid_cells)} cells"
            )
        minimum_place_cells = (
            place_cells
            if minimum_place_cells is None
            else min(minimum_place_cells, place_cells)
        )
        minimum_grid_cells = (
            grid_cells
            if minimum_grid_cells is None
            else min(minimum_grid_cells, grid_cells)
        )
        if "behavior_bvc_conditioned" in config.conditioning["branches"]:
            if trajectory.bvc_activity is None:
                raise ValueError("The BVC-conditioned branch requires BVC arrays")
            if trajectory.bvc_activity.shape[1] < int(
                config.conditioning["bvc_components"]
            ):
                raise ValueError(
                    f"Trajectory {trajectory.trajectory_id!r} has fewer BVC cells "
                    "than conditioning.bvc_components"
                )
        if (
            str(trajectory.stage) == str(config.baseline["stage"])
            and str(trajectory.environment) == str(config.baseline["environment"])
            and str(trajectory.session) == str(config.baseline["session"])
        ):
            baseline_samples[trajectory.network_id] = (
                baseline_samples.get(trajectory.network_id, 0) + trajectory.n_time
            )
    insufficient = {
        network: count
        for network, count in baseline_samples.items()
        if count <= requested_components
    }
    if insufficient:
        raise ValueError(
            "Baseline sample counts cannot support the requested PCA rank: "
            f"{insufficient}"
        )
    return {
        "trajectory_arrays_checked": n_loaded,
        "maximum_requested_pca_components": requested_components,
        "minimum_place_cells": minimum_place_cells,
        "minimum_grid_cells_per_module": minimum_grid_cells,
    }


def _validate_command(args: argparse.Namespace) -> int:
    config = Config.from_json(args.config)
    manifest_source: Path | Config = args.manifest if args.manifest else config
    manifest = load_manifest(
        manifest_source,
        config=None if isinstance(manifest_source, Config) else config,
        check_files=not args.skip_file_check,
    )
    design = validate_manifest_design(manifest, config)
    array_summary = (
        {
            "trajectory_arrays_checked": 0,
            "maximum_requested_pca_components": max(
                [
                    int(config.representation["primary_components"]),
                    *[
                        int(value)
                        for value in config.representation.get(
                            "sensitivity_components", []
                        )
                    ],
                ]
            ),
            "minimum_place_cells": None,
            "minimum_grid_cells_per_module": None,
        }
        if args.skip_file_check
        else _validate_referenced_arrays(manifest, config)
    )
    group_columns = ["network_id", "stage", "environment", "session"]
    _print_json(
        {
            "status": "valid",
            "config": str(args.config.resolve()),
            "manifest": manifest.attrs["manifest_path"],
            "referenced_files_checked": not args.skip_file_check,
            "n_networks": int(manifest["network_id"].nunique()),
            "n_groups": int(manifest[group_columns].drop_duplicates().shape[0]),
            "n_trajectories": int(len(manifest)),
            "minimum_trajectories_in_group": int(design["n_trajectories"].min()),
            "primary_targets": config.statistics["primary_targets"],
            "independent_unit": "network_id",
            **array_summary,
        }
    )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    result = run_analysis(
        args.config,
        manifest_path=args.manifest,
        output_directory=args.output,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "status": "complete",
            "output_directory": str(result.output_directory),
            "selected_parameters": result.selected_parameters,
            "n_estimate_rows": result.n_estimate_rows,
            "n_primary_network_effects": result.n_primary_network_effects,
            "n_input_trajectories": result.n_input_trajectories,
        }
    )
    return 0


def _synthetic_command(args: argparse.Namespace) -> int:
    report, destination = run_synthetic_check(args.output, seed=args.seed)
    if destination is not None:
        report = {**report, "output_path": str(destination)}
    _print_json(report)
    return 0 if bool(report["passed"]) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate_command(args)
        if args.command == "run":
            return _run_command(args)
        if args.command == "synthetic-check":
            return _synthetic_command(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"te-gcmi: error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


__all__ = ["main"]
