"""Scalar-pair TE between raw place cells and grid-module PCA components."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from te_population_gcmi.config import Config
from te_population_gcmi.copula import rank_gaussianize
from te_population_gcmi.io import TrajectoryData, load_manifest
from te_population_gcmi.lagged import build_lagged_samples
from te_population_gcmi.metadata import derive_seed
from te_population_gcmi.pca import PCAProjector, _sample_indices, _stack_selected
from te_population_gcmi.pipeline import (
    _load_group,
    _manifest_group_rows,
    validate_manifest_design,
)
from te_population_gcmi.segments import (
    MovementSegment,
    SegmentationResult,
    segment_trajectory,
    segmentation_qc_table,
)
from te_population_gcmi.types import Segment


STATES = ("approach", "departure")
AXES = ("x", "y")


@dataclass(frozen=True)
class GridOnlyPCA:
    network_id: str
    grid: dict[str, PCAProjector]
    fitted_sample_count: int
    trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedGroup:
    trajectories: tuple[TrajectoryData, ...]
    grid_scores: dict[str, dict[str, np.ndarray]]
    segmentations: dict[str, SegmentationResult]

    @property
    def first(self) -> TrajectoryData:
        return self.trajectories[0]

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(self.grid_scores))


def _resolve(value: str | Path, parent: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _absolute_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["data_path"] = [
        str(_resolve(value, path.parent))
        for value in frame["data_path"].astype(str)
    ]
    return frame


def build_combined_manifest(
    settings: Mapping[str, Any],
    config_dir: Path,
    output: Path,
) -> Path:
    frames: list[pd.DataFrame] = []
    for environment, value in settings["prepared_manifests"].items():
        frame = _absolute_manifest(_resolve(value, config_dir))
        selected = (
            frame["stage"].astype(str).eq("baseline")
            | (
                frame["stage"].astype(str).eq("initial_exposure")
                & frame["environment"].astype(str).eq(environment)
            )
        )
        frames.append(frame.loc[selected].copy())
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        [
            "network_id",
            "stage",
            "environment",
            "session",
            "trajectory_id",
        ],
        keep="first",
    )
    combined = combined.loc[
        combined["network_id"].astype(str).eq(str(settings["network_id"]))
    ].copy()
    destination = output / "combined_manifest.csv"
    combined.to_csv(destination, index=False)
    return destination


def build_runtime_config(
    settings: Mapping[str, Any],
    config_dir: Path,
    manifest_path: Path,
    output: Path,
) -> Config:
    payload = deepcopy(
        _read_json(_resolve(settings["base_config"], config_dir))
    )
    payload["data"]["manifest_path"] = str(manifest_path)
    payload["representation"]["primary_components"] = int(
        settings["grid_pca_components"]
    )
    payload["representation"]["sensitivity_components"] = []
    payload["conditioning"]["primary"] = "unconditioned"
    payload["conditioning"]["branches"] = ["unconditioned"]
    payload["te"]["compute_cv_score"] = False
    payload["te"]["effect_value"] = "te_raw_bits"
    payload["te"]["lag_frames"] = [int(settings["lag_frames"])]
    payload["te"]["source_history_frames"] = [
        int(settings["source_history"])
    ]
    payload["te"]["target_history_frames"] = [
        int(settings["target_history"])
    ]
    payload["te"]["bias_shuffles"] = 0
    payload["te"]["sensitivity_bias_shuffles"] = 0
    payload["te"]["minimum_pool_samples"] = int(
        settings["minimum_pool_samples"]
    )
    payload["te"]["minimum_effective_samples_per_dimension"] = float(
        settings["minimum_effective_samples_per_dimension"]
    )
    payload["te"]["select_parameters_on_baseline"] = False
    payload["cross_trajectory_null"]["enabled"] = False
    payload["cross_trajectory_null"]["repetitions"] = 0
    payload["statistics"]["primary_targets"] = [
        {
            "stage": "initial_exposure",
            "environment": environment,
            "session": 0,
        }
        for environment in settings["prepared_manifests"]
    ]
    payload["execution"]["run_primary"] = True
    payload["execution"]["run_sensitivity"] = False
    payload["execution"]["fail_on_qc"] = False
    destination = output / "config.resolved.json"
    _write_json(destination, payload)
    return Config.from_dict(payload, source_path=destination)


def fit_grid_only_pca(
    trajectories: Sequence[TrajectoryData],
    *,
    n_components: int,
    max_samples: int,
    seed: int,
) -> GridOnlyPCA:
    ordered = tuple(
        sorted(trajectories, key=lambda item: str(item.trajectory_id))
    )
    networks = {str(item.network_id) for item in ordered}
    if len(networks) != 1:
        raise ValueError("Grid PCA requires exactly one network")
    modules = set(ordered[0].grid_activity)
    if any(set(item.grid_activity) != modules for item in ordered):
        raise ValueError("Grid modules differ across baseline trajectories")
    indices = _sample_indices(
        [item.n_time for item in ordered],
        max_samples,
        seed,
    )
    projectors = {
        module: PCAProjector.fit(
            _stack_selected(
                [item.grid_activity[module] for item in ordered],
                indices,
                f"grid baseline {module}",
            ),
            n_components,
        )
        for module in sorted(modules)
    }
    return GridOnlyPCA(
        network_id=next(iter(networks)),
        grid=projectors,
        fitted_sample_count=int(sum(index.size for index in indices)),
        trajectory_ids=tuple(str(item.trajectory_id) for item in ordered),
    )


def save_grid_pca(model: GridOnlyPCA, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for module, projector in model.grid.items():
        np.savez_compressed(
            output / f"{module}.npz",
            mean=projector.mean_,
            components=projector.components_,
            explained_variance=projector.explained_variance_,
            explained_variance_ratio=projector.explained_variance_ratio_,
            n_samples=np.asarray(projector.n_samples_seen_),
        )
        cumulative = np.cumsum(projector.explained_variance_ratio_)
        for component, (variance, ratio, total) in enumerate(
            zip(
                projector.explained_variance_,
                projector.explained_variance_ratio_,
                cumulative,
            ),
            start=1,
        ):
            rows.append(
                {
                    "network_id": model.network_id,
                    "module": module,
                    "component": component,
                    "explained_variance": float(variance),
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance_ratio": float(total),
                    "fit_samples": model.fitted_sample_count,
                }
            )
    return pd.DataFrame(rows)


def prepare_group(
    trajectories: Sequence[TrajectoryData],
    pca: GridOnlyPCA,
    config: Config,
) -> PreparedGroup:
    ordered = tuple(
        sorted(trajectories, key=lambda item: str(item.trajectory_id))
    )
    grid_scores = {
        module: {
            item.trajectory_id: projector.transform(
                item.grid_activity[module]
            )
            for item in ordered
        }
        for module, projector in pca.grid.items()
    }
    segmentations = {
        item.trajectory_id: segment_trajectory(
            item,
            config.segmentation,
        )
        for item in ordered
    }
    return PreparedGroup(
        trajectories=ordered,
        grid_scores=grid_scores,
        segmentations=segmentations,
    )


def _movement_segments(
    prepared: PreparedGroup,
    *,
    module: str,
    state: str,
    wall_axis: str,
) -> tuple[Segment, ...]:
    selected: list[tuple[TrajectoryData, MovementSegment]] = []
    for trajectory in prepared.trajectories:
        segmentation = prepared.segmentations[trajectory.trajectory_id]
        for movement in segmentation.accepted:
            if (
                movement.direction == state
                and movement.wall_axis == wall_axis
            ):
                selected.append((trajectory, movement))
    counts: dict[str, int] = {}
    for trajectory, _ in selected:
        key = str(trajectory.trajectory_id)
        counts[key] = counts.get(key, 0) + 1
    segments: list[Segment] = []
    for trajectory, movement in selected:
        key = str(trajectory.trajectory_id)
        start, stop = movement.start, movement.stop
        segments.append(
            Segment(
                source=trajectory.place_activity[start:stop],
                target=prepared.grid_scores[module][
                    trajectory.trajectory_id
                ][start:stop],
                conditioning=None,
                segment_id=movement.segment_id,
                trajectory_id=trajectory.trajectory_id,
                weight=1.0 / counts[key],
                metadata=movement.metadata,
            )
        )
    return tuple(segments)


def _weighted_center(
    values: np.ndarray,
    normalized_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(
        values * normalized_weight[:, None],
        axis=0,
    )
    centered = values - mean
    variance = np.sum(
        centered * centered * normalized_weight[:, None],
        axis=0,
    )
    return centered, variance


def scalar_cmi_matrix(
    source: np.ndarray,
    future: np.ndarray,
    target_past: np.ndarray,
    sample_weight: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    """Return scalar CMI for every source column x matched target column.

    Marginal Gaussianization is applied column-wise exactly as in the
    production estimator. The covariance algebra is vectorized across pairs;
    each CMI still uses its own coherent 3 x 3 shrunk covariance.
    """

    if not 0.0 < shrinkage <= 1.0:
        raise ValueError(
            "The fast scalar estimator requires numeric shrinkage in (0, 1]"
        )
    weight = np.asarray(sample_weight, dtype=float)
    weight = weight / np.sum(weight)
    x = rank_gaussianize(source, sample_weight=sample_weight)
    y = rank_gaussianize(future, sample_weight=sample_weight)
    z = rank_gaussianize(target_past, sample_weight=sample_weight)
    if y.shape != z.shape:
        raise ValueError(
            "future and target_past must have matched target columns"
        )
    xc, vx = _weighted_center(x, weight)
    yc, vy = _weighted_center(y, weight)
    zc, vz = _weighted_center(z, weight)
    cxy = xc.T @ (yc * weight[:, None])
    cxz = xc.T @ (zc * weight[:, None])
    cyz = np.sum(yc * zc * weight[:, None], axis=0)

    one_minus = 1.0 - shrinkage
    mean_variance = (
        vx[:, None] + vy[None, :] + vz[None, :]
    ) / 3.0
    ax = one_minus * vx[:, None] + shrinkage * mean_variance
    ay = one_minus * vy[None, :] + shrinkage * mean_variance
    az = one_minus * vz[None, :] + shrinkage * mean_variance
    xy = one_minus * cxy
    xz = one_minus * cxz
    yz = one_minus * cyz[None, :]

    determinant_xz = ax * az - xz * xz
    determinant_yz = ay * az - yz * yz
    determinant_full = (
        ax * ay * az
        + 2.0 * xy * xz * yz
        - ax * yz * yz
        - ay * xz * xz
        - az * xy * xy
    )
    denominator = az * determinant_full
    valid = (
        (determinant_xz > 0.0)
        & (determinant_yz > 0.0)
        & (denominator > 0.0)
    )
    result = np.full(cxy.shape, np.nan, dtype=float)
    result[valid] = (
        0.5
        * np.log(
            determinant_xz[valid]
            * determinant_yz[valid]
            / denominator[valid]
        )
        / np.log(2.0)
    )
    small_negative = (result < 0.0) & (result > -1e-10)
    result[small_negative] = 0.0
    return result


def _pooled_column_tau(
    arrays: Sequence[np.ndarray],
    *,
    max_lag: int,
) -> np.ndarray:
    if not arrays:
        raise ValueError("arrays cannot be empty")
    n_features = arrays[0].shape[1]
    result = np.ones(n_features, dtype=float)
    for feature in range(n_features):
        positive_sum = 0.0
        usable_max = min(
            max_lag,
            max(array.shape[0] - 1 for array in arrays),
        )
        for lag in range(1, usable_max + 1):
            numerator = 0.0
            left_energy = 0.0
            right_energy = 0.0
            n_pairs = 0
            for array in arrays:
                if array.shape[0] <= lag:
                    continue
                column = np.asarray(array[:, feature], dtype=float)
                centered = column - np.mean(column)
                left = centered[:-lag]
                right = centered[lag:]
                numerator += float(np.dot(left, right))
                left_energy += float(np.dot(left, left))
                right_energy += float(np.dot(right, right))
                n_pairs += left.size
            if (
                n_pairs < 2
                or left_energy <= 0.0
                or right_energy <= 0.0
            ):
                rho = 0.0
            else:
                rho = numerator / np.sqrt(
                    left_energy * right_energy
                )
            if not np.isfinite(rho) or rho <= 0.0:
                break
            positive_sum += min(float(rho), 0.999)
        result[feature] = 1.0 + 2.0 * positive_sum
    return result


def estimate_pool(
    segments_pg: Sequence[Segment],
    *,
    lag_frames: int,
    source_history: int,
    target_history: int,
    shrinkage: float,
    minimum_pool_samples: int,
    minimum_effective_samples_per_dimension: float,
    autocorrelation_max_lag: int,
) -> pd.DataFrame:
    if not segments_pg:
        return pd.DataFrame()
    pg = build_lagged_samples(
        segments_pg,
        source_history=source_history,
        target_history=target_history,
        prediction_delay=lag_frames,
        equal_segment_weight=True,
    )
    segments_gp = tuple(
        Segment(
            source=segment.target,
            target=segment.source,
            conditioning=None,
            segment_id=segment.segment_id,
            trajectory_id=segment.trajectory_id,
            weight=segment.weight,
            metadata=segment.metadata,
        )
        for segment in segments_pg
    )
    gp = build_lagged_samples(
        segments_gp,
        source_history=source_history,
        target_history=target_history,
        prediction_delay=lag_frames,
        equal_segment_weight=True,
    )
    if source_history != 1 or target_history != 1:
        raise ValueError(
            "This scalar-pair version currently requires both histories = 1"
        )
    if not np.array_equal(pg.time_index, gp.time_index):
        raise RuntimeError("PG and GP lagged rows are not aligned")

    te_pg = scalar_cmi_matrix(
        pg.source_past,
        pg.target_future,
        pg.target_past,
        pg.sample_weight,
        shrinkage=shrinkage,
    )
    te_gp_transposed = scalar_cmi_matrix(
        gp.source_past,
        gp.target_future,
        gp.target_past,
        gp.sample_weight,
        shrinkage=shrinkage,
    )
    te_gp = te_gp_transposed.T
    place_tau = _pooled_column_tau(
        [np.asarray(segment.source) for segment in segments_pg],
        max_lag=autocorrelation_max_lag,
    )
    grid_tau = _pooled_column_tau(
        [np.asarray(segment.target) for segment in segments_pg],
        max_lag=autocorrelation_max_lag,
    )
    pair_tau = np.maximum(
        place_tau[:, None],
        grid_tau[None, :],
    )
    temporal_n = pg.n_samples / pair_tau
    weights = np.asarray(pg.sample_weight, dtype=float)
    kish_n = float(
        np.sum(weights) ** 2 / np.dot(weights, weights)
    )
    joint_dimension = 1 + source_history + target_history
    required_n = max(
        float(minimum_pool_samples),
        float(minimum_effective_samples_per_dimension)
        * joint_dimension,
    )
    quality_ok = (
        pg.n_samples >= minimum_pool_samples
    ) & (np.minimum(kish_n, temporal_n) >= required_n)
    quality = np.broadcast_to(quality_ok, te_pg.shape)
    valid_te = (
        quality
        & np.isfinite(te_pg)
        & np.isfinite(te_gp)
        & (te_pg >= 0.0)
        & (te_gp >= 0.0)
    )
    denominator = te_pg + te_gp
    valid_id = valid_te & (denominator > 1e-12)
    directionality = np.full(te_pg.shape, np.nan, dtype=float)
    directionality[valid_id] = (
        te_pg[valid_id] - te_gp[valid_id]
    ) / denominator[valid_id]

    place_index, component_index = np.indices(te_pg.shape)
    trajectory_ids = {
        str(value) for value in pg.trajectory_id
    }
    return pd.DataFrame(
        {
            "place_cell_index": place_index.ravel(),
            "place_cell_id": [
                f"place_{value:03d}"
                for value in place_index.ravel()
            ],
            "grid_component": component_index.ravel() + 1,
            "TE_PG_bits": te_pg.ravel(),
            "TE_GP_bits": te_gp.ravel(),
            "I_D": directionality.ravel(),
            "pair_valid": valid_id.ravel(),
            "pair_reason": np.where(
                valid_id.ravel(),
                "ok",
                np.where(
                    quality.ravel(),
                    "invalid_te_or_directionality_denominator",
                    "insufficient_effective_samples",
                ),
            ),
            "n_segments": len(segments_pg),
            "n_trajectories": len(trajectory_ids),
            "n_samples": pg.n_samples,
            "kish_effective_n": kish_n,
            "temporal_effective_n": temporal_n.ravel(),
            "maximum_autocorrelation_time": pair_tau.ravel(),
            "joint_dimension": joint_dimension,
            "minimum_required_n": required_n,
        }
    )


def estimate_group(
    prepared: PreparedGroup,
    config: Config,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    first = prepared.first
    frames: list[pd.DataFrame] = []
    for module in prepared.modules:
        for state in STATES:
            for wall_axis in AXES:
                segments = _movement_segments(
                    prepared,
                    module=module,
                    state=state,
                    wall_axis=wall_axis,
                )
                estimates = estimate_pool(
                    segments,
                    lag_frames=int(settings["lag_frames"]),
                    source_history=int(settings["source_history"]),
                    target_history=int(settings["target_history"]),
                    shrinkage=float(
                        config.te["covariance_shrinkage"]
                    ),
                    minimum_pool_samples=int(
                        settings["minimum_pool_samples"]
                    ),
                    minimum_effective_samples_per_dimension=float(
                        settings[
                            "minimum_effective_samples_per_dimension"
                        ]
                    ),
                    autocorrelation_max_lag=int(
                        config.te["autocorrelation_max_lag"]
                    ),
                )
                estimates.insert(0, "wall_axis", wall_axis)
                estimates.insert(0, "state", state)
                estimates.insert(0, "module", module)
                estimates.insert(0, "session", first.session)
                estimates.insert(0, "environment", first.environment)
                estimates.insert(0, "stage", first.stage)
                estimates.insert(0, "network_id", first.network_id)
                frames.append(estimates)
    return pd.concat(frames, ignore_index=True)


def _complete_mean(
    group: pd.DataFrame,
    *,
    value: str,
    valid: str,
    expected: int,
) -> tuple[float, bool, int]:
    valid_mask = (
        group[valid].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(group[value], errors="coerce"))
    )
    complete = bool(len(group) == expected and valid_mask.all())
    return (
        (
            float(
                pd.to_numeric(
                    group.loc[valid_mask, value],
                    errors="coerce",
                ).mean()
            )
            if complete
            else np.nan
        ),
        complete,
        int(valid_mask.sum()),
    )


def build_effects(
    estimates: pd.DataFrame,
    *,
    target_environment: str,
    denominator_floor: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    baseline = estimates.loc[
        estimates["stage"].astype(str).eq("baseline")
        & estimates["environment"].astype(str).eq("fam_100")
    ].copy()
    target = estimates.loc[
        estimates["stage"].astype(str).eq("initial_exposure")
        & estimates["environment"]
        .astype(str)
        .eq(target_environment)
    ].copy()
    keys = [
        "network_id",
        "module",
        "place_cell_index",
        "place_cell_id",
        "grid_component",
        "state",
        "wall_axis",
    ]
    baseline = baseline.rename(
        columns={
            "TE_PG_bits": "baseline_TE_PG_bits",
            "TE_GP_bits": "baseline_TE_GP_bits",
            "I_D": "baseline_I_D",
            "pair_valid": "baseline_pair_valid",
            "pair_reason": "baseline_pair_reason",
        }
    )
    target = target.rename(
        columns={
            "TE_PG_bits": "target_TE_PG_bits",
            "TE_GP_bits": "target_TE_GP_bits",
            "I_D": "target_I_D",
            "pair_valid": "target_pair_valid",
            "pair_reason": "target_pair_reason",
        }
    )
    baseline_columns = [
        *keys,
        "baseline_TE_PG_bits",
        "baseline_TE_GP_bits",
        "baseline_I_D",
        "baseline_pair_valid",
        "baseline_pair_reason",
    ]
    target_columns = [
        *keys,
        "target_TE_PG_bits",
        "target_TE_GP_bits",
        "target_I_D",
        "target_pair_valid",
        "target_pair_reason",
    ]
    cells = target.loc[:, target_columns].merge(
        baseline.loc[:, baseline_columns],
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    cells["target_environment"] = target_environment
    cells["axis_role"] = np.where(
        cells["wall_axis"].astype(str).eq("x"),
        "deformed",
        "orthogonal",
    )
    valid = (
        cells["target_pair_valid"].fillna(False).astype(bool)
        & cells["baseline_pair_valid"].fillna(False).astype(bool)
    )
    for column in (
        "target_TE_PG_bits",
        "target_TE_GP_bits",
        "baseline_TE_PG_bits",
        "baseline_TE_GP_bits",
        "target_I_D",
        "baseline_I_D",
    ):
        valid &= np.isfinite(
            pd.to_numeric(cells[column], errors="coerce")
        )
    target_denominator = (
        cells["target_TE_PG_bits"]
        + cells["target_TE_GP_bits"]
    )
    baseline_denominator = (
        cells["baseline_TE_PG_bits"]
        + cells["baseline_TE_GP_bits"]
    )
    valid &= (
        target_denominator > denominator_floor
    ) & (baseline_denominator > denominator_floor)
    cells["cell_valid"] = valid
    cells["delta_I_D"] = np.where(
        valid,
        cells["target_I_D"] - cells["baseline_I_D"],
        np.nan,
    )
    cells["delta_TE_PG_bits"] = np.where(
        valid,
        cells["target_TE_PG_bits"]
        - cells["baseline_TE_PG_bits"],
        np.nan,
    )
    cells["delta_TE_GP_bits"] = np.where(
        valid,
        cells["target_TE_GP_bits"]
        - cells["baseline_TE_GP_bits"],
        np.nan,
    )

    contrast_rows: list[dict[str, Any]] = []
    contrast_keys = [
        "network_id",
        "target_environment",
        "module",
        "place_cell_index",
        "place_cell_id",
        "grid_component",
    ]
    expected_conditions = {
        ("approach", "deformed"),
        ("departure", "deformed"),
        ("approach", "orthogonal"),
        ("departure", "orthogonal"),
    }
    for identity, group in cells.groupby(
        contrast_keys,
        sort=False,
        dropna=False,
    ):
        indexed = group.set_index(["state", "axis_role"])
        complete = bool(
            set(indexed.index) == expected_conditions
            and group["cell_valid"].astype(bool).all()
        )
        value = np.nan
        if complete:
            value = float(
                indexed.loc[
                    ("approach", "deformed"),
                    "delta_I_D",
                ]
                - indexed.loc[
                    ("departure", "deformed"),
                    "delta_I_D",
                ]
                - indexed.loc[
                    ("approach", "orthogonal"),
                    "delta_I_D",
                ]
                + indexed.loc[
                    ("departure", "orthogonal"),
                    "delta_I_D",
                ]
            )
        contrast_rows.append(
            {
                **dict(zip(contrast_keys, identity)),
                "pair_contrast_valid": complete,
                "C_delta_I_D": value,
            }
        )
    pair_effects = pd.DataFrame(contrast_rows)

    n_components = int(
        pair_effects["grid_component"].nunique()
    )
    n_modules = int(pair_effects["module"].nunique())
    n_place_cells = int(
        pair_effects["place_cell_id"].nunique()
    )
    module_rows: list[dict[str, Any]] = []
    for identity, group in pair_effects.groupby(
        [
            "network_id",
            "target_environment",
            "place_cell_index",
            "place_cell_id",
            "module",
        ],
        sort=False,
        dropna=False,
    ):
        value, complete, n_valid = _complete_mean(
            group,
            value="C_delta_I_D",
            valid="pair_contrast_valid",
            expected=n_components,
        )
        module_rows.append(
            {
                **dict(
                    zip(
                        [
                            "network_id",
                            "target_environment",
                            "place_cell_index",
                            "place_cell_id",
                            "module",
                        ],
                        identity,
                    )
                ),
                "n_expected_components": n_components,
                "n_valid_components": n_valid,
                "place_module_valid": complete,
                "C_delta_I_D": value,
            }
        )
    place_module_effects = pd.DataFrame(module_rows)

    place_rows: list[dict[str, Any]] = []
    for identity, group in place_module_effects.groupby(
        [
            "network_id",
            "target_environment",
            "place_cell_index",
            "place_cell_id",
        ],
        sort=False,
        dropna=False,
    ):
        value, complete, n_valid = _complete_mean(
            group,
            value="C_delta_I_D",
            valid="place_module_valid",
            expected=n_modules,
        )
        place_rows.append(
            {
                **dict(
                    zip(
                        [
                            "network_id",
                            "target_environment",
                            "place_cell_index",
                            "place_cell_id",
                        ],
                        identity,
                    )
                ),
                "n_expected_modules": n_modules,
                "n_valid_modules": n_valid,
                "place_cell_valid": complete,
                "C_delta_I_D": value,
            }
        )
    place_effects = pd.DataFrame(place_rows)

    network_rows: list[dict[str, Any]] = []
    for identity, group in place_effects.groupby(
        ["network_id", "target_environment"],
        sort=False,
        dropna=False,
    ):
        valid_values = pd.to_numeric(
            group.loc[
                group["place_cell_valid"].astype(bool),
                "C_delta_I_D",
            ],
            errors="coerce",
        ).dropna()
        n_valid = int(len(valid_values))
        coverage = (
            float(n_valid / n_place_cells)
            if n_place_cells > 0
            else np.nan
        )
        value = (
            float(valid_values.mean())
            if n_valid > 0
            else np.nan
        )
        network_rows.append(
            {
                "network_id": identity[0],
                "target_environment": identity[1],
                "n_expected_place_cells": n_place_cells,
                "n_valid_place_cells": n_valid,
                "place_cell_coverage": coverage,
                "all_place_cells_complete": bool(
                    n_valid == n_place_cells
                ),
                "network_contrast_valid": bool(n_valid > 0),
                "C_delta_I_D": value,
                "place_cell_sd_descriptive": (
                    float(valid_values.std(ddof=1))
                    if len(valid_values) > 1
                    else np.nan
                ),
            }
        )
    network_effects = pd.DataFrame(network_rows)

    eligible_places = place_effects.loc[
        place_effects["place_cell_valid"].astype(bool),
        [
            "network_id",
            "target_environment",
            "place_cell_index",
            "place_cell_id",
        ],
    ].drop_duplicates()
    state_axis_cells = cells.merge(
        eligible_places,
        on=[
            "network_id",
            "target_environment",
            "place_cell_index",
            "place_cell_id",
        ],
        how="inner",
        validate="many_to_one",
    )
    state_axis_rows: list[dict[str, Any]] = []
    for identity, group in state_axis_cells.groupby(
        [
            "network_id",
            "target_environment",
            "state",
            "axis_role",
        ],
        sort=False,
        dropna=False,
    ):
        n_complete_places = int(
            group["place_cell_id"].nunique()
        )
        value, complete, n_valid = _complete_mean(
            group,
            value="delta_I_D",
            valid="cell_valid",
            expected=(
                n_complete_places * n_modules * n_components
            ),
        )
        state_axis_rows.append(
            {
                "network_id": identity[0],
                "target_environment": identity[1],
                "state": identity[2],
                "axis_role": identity[3],
                "n_complete_place_cells": n_complete_places,
                "n_expected_pairs": (
                    n_complete_places * n_modules * n_components
                ),
                "n_valid_pairs": n_valid,
                "network_cell_valid": complete,
                "delta_I_D": value,
            }
        )
    state_axis = pd.DataFrame(state_axis_rows)
    return (
        cells,
        pair_effects,
        place_module_effects,
        place_effects,
        network_effects,
        state_axis,
    )


def run(
    config_path: Path,
    output_override: Path | None,
    *,
    overwrite: bool,
) -> Path:
    started = time.perf_counter()
    settings = _read_json(config_path)
    config_dir = config_path.parent
    output = (
        output_override.resolve()
        if output_override is not None
        else _resolve(settings["output_directory"], config_dir)
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = build_combined_manifest(
        settings,
        config_dir,
        output,
    )
    config = build_runtime_config(
        settings,
        config_dir,
        manifest_path,
        output,
    )
    manifest = load_manifest(config, check_files=True)
    validate_manifest_design(manifest, config).to_csv(
        output / "design_qc.csv",
        index=False,
    )
    baseline_rows = manifest.loc[
        manifest["stage"].astype(str).eq("baseline")
        & manifest["environment"].astype(str).eq("fam_100")
    ]
    baseline_trajectories = _load_group(
        baseline_rows,
        config,
        manifest.attrs["manifest_dir"],
    )
    pca = fit_grid_only_pca(
        baseline_trajectories,
        n_components=int(settings["grid_pca_components"]),
        max_samples=int(
            config.representation["pca_fit_max_samples"]
        ),
        seed=derive_seed(
            config.random_seed,
            "grid_only_pca",
            settings["network_id"],
        ),
    )
    save_grid_pca(pca, output / "grid_pca").to_csv(
        output / "grid_pca_explained_variance.csv",
        index=False,
    )

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    estimate_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []
    groups = list(_manifest_group_rows(manifest))
    for index, (key, rows) in enumerate(groups, start=1):
        group_name = "__".join(str(value) for value in key)
        checkpoint = checkpoint_dir / f"{group_name}.csv.gz"
        if checkpoint.is_file() and not overwrite:
            print(
                f"[{index}/{len(groups)}] reuse {group_name}",
                flush=True,
            )
            estimates = pd.read_csv(checkpoint)
        else:
            print(
                f"[{index}/{len(groups)}] run   {group_name}",
                flush=True,
            )
            trajectories = _load_group(
                rows,
                config,
                manifest.attrs["manifest_dir"],
            )
            prepared = prepare_group(trajectories, pca, config)
            segment_frames.append(
                segmentation_qc_table(
                    prepared.segmentations.values()
                )
            )
            estimates = estimate_group(
                prepared,
                config,
                settings,
            )
            estimates.to_csv(
                checkpoint,
                index=False,
                compression="gzip",
            )
        estimate_frames.append(estimates)
    estimates = pd.concat(estimate_frames, ignore_index=True)
    estimates.to_csv(
        output / "pair_te_estimates.csv.gz",
        index=False,
        compression="gzip",
    )
    if segment_frames:
        pd.concat(
            segment_frames,
            ignore_index=True,
        ).drop_duplicates(
            [
                "network_id",
                "stage",
                "environment",
                "session",
                "segment_id",
            ]
        ).to_csv(
            output / "segment_qc.csv.gz",
            index=False,
            compression="gzip",
        )

    cell_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    module_frames: list[pd.DataFrame] = []
    place_frames: list[pd.DataFrame] = []
    network_frames: list[pd.DataFrame] = []
    state_axis_frames: list[pd.DataFrame] = []
    for environment in settings["prepared_manifests"]:
        (
            cells,
            pairs,
            modules,
            places,
            networks,
            state_axis,
        ) = build_effects(
            estimates,
            target_environment=environment,
            denominator_floor=float(
                settings["directionality_denominator_floor"]
            ),
        )
        cell_frames.append(cells)
        pair_frames.append(pairs)
        module_frames.append(modules)
        place_frames.append(places)
        network_frames.append(networks)
        state_axis_frames.append(state_axis)
    cells = pd.concat(cell_frames, ignore_index=True)
    pair_effects = pd.concat(pair_frames, ignore_index=True)
    module_effects = pd.concat(module_frames, ignore_index=True)
    place_effects = pd.concat(place_frames, ignore_index=True)
    network_effects = pd.concat(network_frames, ignore_index=True)
    state_axis = pd.concat(state_axis_frames, ignore_index=True)
    cells.to_csv(
        output / "pair_state_axis_delta_id.csv.gz",
        index=False,
        compression="gzip",
    )
    pair_effects.to_csv(
        output / "pair_C_delta_I_D.csv.gz",
        index=False,
        compression="gzip",
    )
    module_effects.to_csv(
        output / "place_module_C_delta_I_D.csv",
        index=False,
    )
    place_effects.to_csv(
        output / "place_cell_C_delta_I_D.csv",
        index=False,
    )
    network_effects.to_csv(
        output / "network_C_delta_I_D.csv",
        index=False,
    )
    state_axis.to_csv(
        output / "network_state_axis_delta_I_D.csv",
        index=False,
    )
    summary = {
        "status": "complete",
        "analysis": (
            "raw scalar place cell paired with scalar grid PCA "
            "component"
        ),
        "network_id": settings["network_id"],
        "environments": list(settings["prepared_manifests"]),
        "grid_pca_components": int(
            settings["grid_pca_components"]
        ),
        "place_representation": (
            "raw individual place-cell activity; no PCA"
        ),
        "conditioning": "unconditioned",
        "lag_frames": int(settings["lag_frames"]),
        "lag_ms": float(settings["lag_frames"]) * 10.0,
        "source_history": int(settings["source_history"]),
        "target_history": int(settings["target_history"]),
        "directionality_order": (
            "TE_PG and TE_GP -> I_D within place-cell/module/"
            "component/state/axis -> target minus baseline -> "
            "state-by-axis contrast -> component mean -> module "
            "mean -> place-cell mean"
        ),
        "n_pair_estimate_rows": int(len(estimates)),
        "network_effects": network_effects.to_dict("records"),
        "elapsed_seconds": time.perf_counter() - started,
        "output_directory": str(output),
    }
    _write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "config.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
