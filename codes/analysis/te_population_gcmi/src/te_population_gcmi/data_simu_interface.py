"""One-call interface for running full two-trajectory TE on ``data_simu``.

The interface only adapts the cluster ``transient_activity.npz`` files to the
canonical input contract.  All TE estimation is delegated to the existing
cross-fitted :func:`te_population_gcmi.pipeline.run_analysis` pipeline.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .pipeline import run_analysis


TRAJECTORIES = ("trajectory_00", "trajectory_01")
DIRECTIONS = ("place_to_grid", "grid_to_place")
STATES = ("approach", "departure")
AXIS_ROLES = ("deformed", "orthogonal")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalise_network_id(value: str | int) -> str:
    text = str(value).strip()
    if text.lower().startswith("network_"):
        suffix = text.split("_", 1)[1]
    else:
        suffix = text
    if not suffix.isdigit():
        raise ValueError("network must be an integer or have the form network_###")
    return f"network_{int(suffix):03d}"


def _environment_width(environment: str) -> float:
    text = str(environment).strip()
    if text == "fam_100":
        return 1.0
    prefix, separator, suffix = text.rpartition("_")
    if prefix not in {"compressed", "elongated"} or not separator or not suffix.isdigit():
        raise ValueError(
            "environment must be fam_100, compressed_XX, or elongated_XX"
        )
    width = int(suffix) / 100.0
    if width <= 0:
        raise ValueError("environment width must be positive")
    return width


def _time_major(activity: np.ndarray, *, n_time: int, name: str) -> np.ndarray:
    values = np.asarray(activity)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {values.shape}")
    if values.shape[0] == n_time:
        result = values
    elif values.shape[1] == n_time:
        result = values.T
    else:
        raise ValueError(
            f"{name} has no time axis of length {n_time}: shape={values.shape}"
        )
    return np.ascontiguousarray(result)


def _convert_transient(source_path: Path, output_path: Path) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as source:
        required = {"positions", "place", "grid_m1", "grid_m2", "grid_m3", "record_dt_s"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"{source_path} is missing arrays: {missing}")
        positions = np.asarray(source["positions"])
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"{source_path}: positions has shape {positions.shape}")
        n_time = int(positions.shape[0])
        dt_s = float(np.asarray(source["record_dt_s"]).item())
        place = _time_major(source["place"], n_time=n_time, name="place")
        grid_1 = _time_major(source["grid_m1"], n_time=n_time, name="grid_m1")
        grid_2 = _time_major(source["grid_m2"], n_time=n_time, name="grid_m2")
        grid_3 = _time_major(source["grid_m3"], n_time=n_time, name="grid_m3")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        positions=np.ascontiguousarray(positions),
        place_activity=place,
        grid_module_1=grid_1,
        grid_module_2=grid_2,
        grid_module_3=grid_3,
        time=np.arange(n_time, dtype=np.float64) * dt_s,
    )
    return {
        "source": str(source_path),
        "output": str(output_path),
        "n_time": n_time,
        "record_dt_s": dt_s,
        "activity_input_orientation": "cell,time or time,cell (detected)",
        "activity_output_orientation": "time,cell",
    }


def prepare_selected_inputs(
    network: str | int,
    environment: str,
    *,
    data_root: str | Path,
    prepared_directory: str | Path,
    overwrite: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    """Convert baseline plus one target environment and write a manifest."""

    network_id = _normalise_network_id(network)
    target_environment = str(environment).strip()
    if target_environment == "fam_100":
        raise ValueError("Select an initial-exposure environment, not fam_100")
    target_width = _environment_width(target_environment)
    root = Path(data_root).resolve()
    prepared = Path(prepared_directory).resolve()
    canonical = prepared / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)

    conditions = (
        ("baseline", "fam_100", 1.0, root / network_id / "baseline" / "fam_100"),
        (
            "initial_exposure",
            target_environment,
            target_width,
            root / network_id / "initial_exposure" / target_environment,
        ),
    )
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for stage, condition, width_m, source_root in conditions:
        for trajectory in TRAJECTORIES:
            source = source_root / trajectory / "transient_activity.npz"
            destination = canonical / (
                f"{network_id}__{stage}__{condition}__{trajectory}.npz"
            )
            if overwrite or not destination.exists():
                audit.append(_convert_transient(source, destination))
            else:
                audit.append(
                    {
                        "source": str(source),
                        "output": str(destination),
                        "status": "reused_existing_canonical_file",
                    }
                )
            rows.append(
                {
                    "network_id": network_id,
                    "stage": stage,
                    "environment": condition,
                    "session": 0,
                    "trajectory_id": trajectory,
                    "width_m": width_m,
                    "height_m": 1.0,
                    "deformed_axis": "x",
                    "data_path": str(Path("canonical") / destination.name),
                }
            )

    manifest_path = prepared / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    (prepared / "conversion_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return manifest_path, audit


def _apply_representation(
    config: dict[str, Any], *, run_sensitivity: bool
) -> None:
    """Pin the run to the requested representation dimension.

    Only ``primary_components`` is estimated unless sensitivity runs are asked
    for; a sensitivity entry equal to the primary value is always dropped
    because ``Config.validate`` rejects that overlap.
    """
    section = config["representation"]
    primary = int(section["primary_components"])
    kept: list[int] = []
    if run_sensitivity:
        for value in section.get("sensitivity_components", []) or []:
            value = int(value)
            if value == primary or value in kept:
                continue
            kept.append(value)
    section["sensitivity_components"] = kept


def _write_generated_config(
    base_config: str | Path,
    destination: Path,
    *,
    manifest_path: Path,
    environment: str,
    pca_components: int = 3,
    run_sensitivity: bool,
    bias_shuffles: int | None,
    cross_null_repetitions: int | None,
    min_pool_samples: int | None = None,
    min_effective_samples_per_dimension: float | None = None,
) -> Path:
    with Path(base_config).resolve().open("r", encoding="utf-8") as handle:
        config = deepcopy(json.load(handle))
    config["data"]["manifest_path"] = str(manifest_path.resolve())
    config["data"]["minimum_trajectories_per_group"] = 2
    config["representation"]["primary_components"] = int(pca_components)
    _apply_representation(config, run_sensitivity=run_sensitivity)
    config["conditioning"]["primary"] = "behavior_conditioned"
    branches = list(config["conditioning"].get("branches", []))
    if "behavior_conditioned" not in branches:
        branches.insert(0, "behavior_conditioned")
    config["conditioning"]["branches"] = branches
    config["te"]["effect_value"] = "te_raw_bits"
    if min_pool_samples is not None:
        config["te"]["minimum_pool_samples"] = int(min_pool_samples)
    if min_effective_samples_per_dimension is not None:
        config["te"]["minimum_effective_samples_per_dimension"] = float(
            min_effective_samples_per_dimension
        )
    if bias_shuffles is not None:
        config["te"]["bias_shuffles"] = int(bias_shuffles)
    if cross_null_repetitions is not None:
        config["cross_trajectory_null"]["repetitions"] = int(
            cross_null_repetitions
        )
        config["cross_trajectory_null"]["enabled"] = (
            int(cross_null_repetitions) > 0
        )
    config["statistics"]["primary_targets"] = [
        {
            "stage": "initial_exposure",
            "environment": str(environment),
            "session": 0,
        }
    ]
    config["execution"]["run_primary"] = True
    config["execution"]["run_sensitivity"] = bool(run_sensitivity)
    config["execution"]["fail_on_qc"] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return destination


def _directionality(pg: float, gp: float, floor: float) -> tuple[float, bool, str]:
    if not (np.isfinite(pg) and np.isfinite(gp)):
        return np.nan, False, "non_finite_te"
    if pg < 0.0 or gp < 0.0:
        return np.nan, False, "negative_te"
    denominator = pg + gp
    if denominator <= floor:
        return np.nan, False, "directionality_denominator_too_small"
    return float((gp - pg) / denominator), True, "ok"


def _scope(frame: pd.DataFrame, condition: Mapping[str, Any]) -> pd.DataFrame:
    return frame.loc[
        frame["stage"].astype(str).eq(str(condition["stage"]))
        & frame["environment"].astype(str).eq(str(condition["environment"]))
        & frame["session"].astype(str).eq(str(condition["session"]))
    ].copy()


def build_id_outputs(
    estimates: pd.DataFrame,
    *,
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    target_axis: str = "x",
    denominator_floor: float = 1e-12,
    analysis_set: str = "primary",
    conditioning: str = "behavior_conditioned",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build module cells, network cells, module contrasts and network contrasts.

    The optional selectors default to the confirmatory data-simu branch for
    backward compatibility, while also allowing sensitivity summaries.
    """

    required = {
        "network_id", "stage", "environment", "session", "module", "state",
        "wall_axis", "direction", "conditioning", "analysis_set", "te_raw_bits",
    }
    missing = sorted(required.difference(estimates.columns))
    if missing:
        raise ValueError(f"TE estimates are missing columns: {missing}")
    primary = estimates.loc[
        estimates["analysis_set"].astype(str).eq(str(analysis_set))
        & estimates["conditioning"].astype(str).eq(str(conditioning))
    ].copy()
    base = _scope(primary, baseline)
    deform = _scope(primary, target)
    keys = ["network_id", "module", "state", "wall_axis"]

    def direction_table(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        selected = frame.loc[:, [*keys, "direction", "te_raw_bits"]].copy()
        duplicates = selected.duplicated([*keys, "direction"], keep=False)
        if duplicates.any():
            raise ValueError(f"Duplicate primary TE rows found for {prefix}")
        table = selected.pivot(index=keys, columns="direction", values="te_raw_bits")
        table = table.reindex(columns=DIRECTIONS).reset_index()
        return table.rename(
            columns={
                "place_to_grid": f"{prefix}_TE_PG_bits",
                "grid_to_place": f"{prefix}_TE_GP_bits",
            }
        )

    cells = direction_table(deform, "target").merge(
        direction_table(base, "baseline"), on=keys, how="outer", validate="one_to_one"
    )
    cells["target_stage"] = str(target["stage"])
    cells["target_environment"] = str(target["environment"])
    cells["target_session"] = target["session"]
    cells["baseline_stage"] = str(baseline["stage"])
    cells["baseline_environment"] = str(baseline["environment"])
    cells["baseline_session"] = baseline["session"]
    cells["axis_role"] = np.where(
        cells["wall_axis"].astype(str).eq(str(target_axis)),
        "deformed",
        "orthogonal",
    )

    id_rows: list[dict[str, Any]] = []
    for _, row in cells.iterrows():
        target_id, target_valid, target_reason = _directionality(
            float(row["target_TE_PG_bits"]),
            float(row["target_TE_GP_bits"]),
            denominator_floor,
        )
        baseline_id, baseline_valid, baseline_reason = _directionality(
            float(row["baseline_TE_PG_bits"]),
            float(row["baseline_TE_GP_bits"]),
            denominator_floor,
        )
        valid = bool(target_valid and baseline_valid)
        id_rows.append(
            {
                **row.to_dict(),
                "target_I_D": target_id,
                "baseline_I_D": baseline_id,
                "delta_I_D": target_id - baseline_id if valid else np.nan,
                "delta_TE_PG_bits": (
                    float(row["target_TE_PG_bits"] - row["baseline_TE_PG_bits"])
                    if valid else np.nan
                ),
                "delta_TE_GP_bits": (
                    float(row["target_TE_GP_bits"] - row["baseline_TE_GP_bits"])
                    if valid else np.nan
                ),
                "cell_valid": valid,
                "cell_reason": (
                    "ok" if valid else f"target:{target_reason}|baseline:{baseline_reason}"
                ),
            }
        )
    cells = pd.DataFrame(id_rows)

    expected_modules = int(cells["module"].nunique())
    network_cell_rows: list[dict[str, Any]] = []
    cell_values = [
        "target_TE_PG_bits", "target_TE_GP_bits", "baseline_TE_PG_bits",
        "baseline_TE_GP_bits", "target_I_D", "baseline_I_D", "delta_I_D",
        "delta_TE_PG_bits", "delta_TE_GP_bits",
    ]
    for identity, group in cells.groupby(
        ["network_id", "target_stage", "target_environment", "target_session",
         "state", "axis_role", "wall_axis"],
        dropna=False,
        sort=False,
    ):
        identity_keys = [
            "network_id", "target_stage", "target_environment", "target_session",
            "state", "axis_role", "wall_axis",
        ]
        complete = bool(
            group["module"].nunique() == expected_modules
            and group["cell_valid"].astype(bool).all()
        )
        result = {
            **dict(zip(identity_keys, identity)),
            "n_expected_modules": expected_modules,
            "n_valid_modules": int(group["cell_valid"].astype(bool).sum()),
            "network_cell_valid": complete,
        }
        for column in cell_values:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            result[column] = float(np.mean(values)) if complete else np.nan
        network_cell_rows.append(result)
    network_cells = pd.DataFrame(network_cell_rows)

    module_rows: list[dict[str, Any]] = []
    for identity, group in cells.groupby(
        ["network_id", "module", "target_stage", "target_environment", "target_session"],
        dropna=False,
        sort=False,
    ):
        indexed = group.set_index(["state", "axis_role"])
        complete = bool(
            all(cell in indexed.index for cell in (
                ("approach", "deformed"), ("departure", "deformed"),
                ("approach", "orthogonal"), ("departure", "orthogonal"),
            ))
            and group["cell_valid"].astype(bool).all()
        )
        contrast = np.nan
        absolute_contrast = np.nan
        if complete:
            contrast = float(
                indexed.loc[("departure", "deformed"), "delta_I_D"]
                - indexed.loc[("approach", "deformed"), "delta_I_D"]
                - indexed.loc[("departure", "orthogonal"), "delta_I_D"]
                + indexed.loc[("approach", "orthogonal"), "delta_I_D"]
            )
            absolute_contrast = float(
                indexed.loc[("departure", "deformed"), "target_I_D"]
                - indexed.loc[("approach", "deformed"), "target_I_D"]
                - indexed.loc[("departure", "orthogonal"), "target_I_D"]
                + indexed.loc[("approach", "orthogonal"), "target_I_D"]
            )
        module_rows.append(
            {
                **dict(zip(
                    ["network_id", "module", "target_stage", "target_environment", "target_session"],
                    identity,
                )),
                "module_contrast_valid": complete,
                "C_I_D": absolute_contrast,
                "C_delta_I_D": contrast,
            }
        )
    module_effects = pd.DataFrame(module_rows)

    network_rows: list[dict[str, Any]] = []
    for identity, group in module_effects.groupby(
        ["network_id", "target_stage", "target_environment", "target_session"],
        dropna=False,
        sort=False,
    ):
        valid = group["module_contrast_valid"].astype(bool)
        complete = bool(group["module"].nunique() == expected_modules and valid.all())
        values = pd.to_numeric(group.loc[valid, "C_delta_I_D"], errors="coerce").to_numpy(float)
        absolute_values = pd.to_numeric(
            group.loc[valid, "C_I_D"], errors="coerce"
        ).to_numpy(float)
        network_rows.append(
            {
                **dict(zip(
                    ["network_id", "target_stage", "target_environment", "target_session"],
                    identity,
                )),
                "n_expected_modules": expected_modules,
                "n_valid_modules": int(valid.sum()),
                "network_contrast_valid": complete,
                "C_I_D": (
                    float(np.mean(absolute_values))
                    if complete and absolute_values.size == expected_modules
                    and np.isfinite(absolute_values).all()
                    else np.nan
                ),
                "C_delta_I_D": (
                    float(np.mean(values))
                    if complete and values.size == expected_modules and np.isfinite(values).all()
                    else np.nan
                ),
                "contrast_definition": (
                    "(departure-approach on deformed axis) - "
                    "(departure-approach on orthogonal axis)"
                ),
            }
        )
    network_effects = pd.DataFrame(network_rows)
    return cells, network_cells, module_effects, network_effects


def run_data_simu_te(
    network: str | int,
    environment: str,
    *,
    data_root: str | Path | None = None,
    output_directory: str | Path | None = None,
    prepared_directory: str | Path | None = None,
    base_config: str | Path | None = None,
    pca_components: int = 3,
    run_sensitivity: bool = False,
    bias_shuffles: int | None = None,
    cross_null_repetitions: int | None = None,
    min_pool_samples: int | None = None,
    min_effective_samples_per_dimension: float | None = None,
    overwrite: bool = False,
    overwrite_inputs: bool = False,
) -> dict[str, Any]:
    """Run full TE for one network and one target environment.

    The familiar ``fam_100`` baseline is added automatically.  Both trajectory
    recordings are used for leave-one-trajectory-out behavioral correction.
    Only ``pca_components`` (default 3) is estimated; sensitivity dimensions
    are run only when ``run_sensitivity`` is requested.
    """

    project = _project_root()
    network_id = _normalise_network_id(network)
    target_environment = str(environment).strip()
    _environment_width(target_environment)
    root = Path(data_root).resolve() if data_root else project.parent / "data_simu"
    output = (
        Path(output_directory).resolve()
        if output_directory
        else project / "outputs" / "data_simu_te" / network_id / target_environment
    )
    prepared = (
        Path(prepared_directory).resolve()
        if prepared_directory
        else project / "prepared_data_simu" / network_id / target_environment
    )
    template = Path(base_config).resolve() if base_config else project / "config.example.json"

    manifest_path, conversion_audit = prepare_selected_inputs(
        network_id,
        target_environment,
        data_root=root,
        prepared_directory=prepared,
        overwrite=overwrite_inputs,
    )
    generated_config = _write_generated_config(
        template,
        prepared / "config.generated.json",
        manifest_path=manifest_path,
        environment=target_environment,
        pca_components=pca_components,
        run_sensitivity=run_sensitivity,
        bias_shuffles=bias_shuffles,
        cross_null_repetitions=cross_null_repetitions,
        min_pool_samples=min_pool_samples,
        min_effective_samples_per_dimension=min_effective_samples_per_dimension,
    )
    effective_config = json.loads(generated_config.read_text(encoding="utf-8"))
    pipeline_result = run_analysis(
        generated_config,
        manifest_path=manifest_path,
        output_directory=output,
        overwrite=overwrite,
    )

    estimates = pd.read_csv(output / "te_estimates.csv.gz")
    target = {
        "stage": "initial_exposure",
        "environment": target_environment,
        "session": 0,
    }
    baseline = {"stage": "baseline", "environment": "fam_100", "session": 0}
    cells, network_cells, module_effects, network_effects = build_id_outputs(
        estimates,
        baseline=baseline,
        target=target,
        target_axis="x",
    )
    cells.to_csv(output / "id_state_axis_cells.csv", index=False)
    network_cells.to_csv(output / "id_network_state_axis.csv", index=False)
    module_effects.to_csv(output / "id_module_effects.csv", index=False)
    network_effects.to_csv(output / "id_network_effects.csv", index=False)

    summary = {
        "status": "complete",
        "network_id": network_id,
        "environment": target_environment,
        "data_root": str(root),
        "manifest": str(manifest_path),
        "generated_config": str(generated_config),
        "representation_effective": {
            "primary_components": int(
                effective_config["representation"]["primary_components"]
            ),
            "sensitivity_components": [
                int(value)
                for value in effective_config["representation"].get(
                    "sensitivity_components", []
                )
            ],
        },
        "te_qc_effective": {
            "minimum_pool_samples": int(
                effective_config["te"]["minimum_pool_samples"]
            ),
            "minimum_effective_samples_per_dimension": float(
                effective_config["te"]["minimum_effective_samples_per_dimension"]
            ),
            "select_parameters_on_baseline": bool(
                effective_config["te"]["select_parameters_on_baseline"]
            ),
        },
        "output_directory": str(output),
        "n_input_files_converted_or_reused": len(conversion_audit),
        "selected_parameters": pipeline_result.selected_parameters,
        "id_definition": "(TE_G_to_P - TE_P_to_G) / (TE_G_to_P + TE_P_to_G)",
        "delta_id_definition": "target I_D - matched familiar-baseline I_D",
        "contrast_definition": "(departure-approach)_deformed - (departure-approach)_orthogonal",
        "primary_id_source": "te_raw_bits",
        "id_network_effects": network_effects.to_dict("records"),
    }
    (output / "id_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full two-trajectory behavior-adjusted TE for one data_simu target"
    )
    parser.add_argument("--network", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-directory", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help=(
            "Number of PCA components to estimate (default 3). Only this "
            "dimension is run; the sensitivity grid of the base configuration "
            "is dropped unless --run-sensitivity is given."
        ),
    )
    parser.add_argument("--run-sensitivity", action="store_true")
    parser.add_argument("--bias-shuffles", type=int)
    parser.add_argument("--cross-null-repetitions", type=int)
    parser.add_argument(
        "--min-pool-samples",
        type=int,
        default=None,
        help=(
            "Override te.minimum_pool_samples. Lower it when the pooled "
            "per-cell sample count is small; see baseline_selection_scores.csv."
        ),
    )
    parser.add_argument(
        "--min-effective-samples-per-dimension",
        type=float,
        default=None,
        help=(
            "Override te.minimum_effective_samples_per_dimension. The QC "
            "requirement is max(minimum_pool_samples, this value x joint dimension)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-inputs", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_data_simu_te(
        args.network,
        args.environment,
        data_root=args.data_root,
        output_directory=args.output,
        prepared_directory=args.prepared_directory,
        base_config=args.base_config,
        pca_components=args.pca_components,
        run_sensitivity=args.run_sensitivity,
        bias_shuffles=args.bias_shuffles,
        cross_null_repetitions=args.cross_null_repetitions,
        min_pool_samples=args.min_pool_samples,
        min_effective_samples_per_dimension=args.min_effective_samples_per_dimension,
        overwrite=args.overwrite,
        overwrite_inputs=args.overwrite_inputs,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_id_outputs", "prepare_selected_inputs", "run_data_simu_te"]
