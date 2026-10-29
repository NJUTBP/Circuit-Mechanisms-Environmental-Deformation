"""End-to-end orchestration for the standalone population TE analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .behavior import (
    BehaviorDesignSpec,
    CrossFittedPopulationResiduals,
    cross_fit_population_residuals,
    derive_kinematics,
    residualize_population_from_training_fold,
)
from .config import Config
from .contrasts import (
    DEFAULT_UNIT_COLUMNS,
    aggregate_to_network,
    build_direction_difference,
    compute_familiar_deltas,
    compute_primary_did,
)
from .estimator import (
    PredictiveTEInputFold,
    estimate_cross_validated_predictive_te,
    estimate_cross_validated_predictive_te_from_folds,
    estimate_gcmi_te,
)
from .io import TrajectoryData, load_manifest, load_trajectory_npz
from .lagged import build_lagged_samples
from .matching import match_cross_trajectory_segments
from .metadata import atomic_write_json, derive_seed, runtime_metadata
from .nulls import circular_shift_test, plus_one_pvalue
from .pca import (
    FrozenPopulationPCA,
    PopulationScores,
    fit_population_pca,
    save_pca_collection,
)
from .qc import TEQuality, assess_te_quality
from .segments import (
    MovementSegment,
    SegmentationResult,
    segment_trajectory,
    segmentation_qc_table,
)
from .selection import SelectionResult, select_shared_parameters
from .statistics import holm_adjust, max_t_sign_flip, network_inference_table
from .types import Segment


GROUP_COLUMNS = ("network_id", "stage", "environment", "session")
PARAMETER_COLUMNS = (
    "conditioning",
    "pca_components",
    "lag_frames",
    "source_history",
    "target_history",
)

KNOWN_OUTPUT_FILES = (
    "config.resolved.json",
    "run_metadata.json",
    "design_qc.csv",
    "pca_explained_variance.csv",
    "baseline_selection_scores.csv",
    "baseline_selection_summary.csv",
    "selected_parameters.json",
    "te_estimates.csv.gz",
    "te_qc.csv.gz",
    "input_qc.csv",
    "segment_qc.csv.gz",
    "null_distributions.csv.gz",
    "cross_trajectory_matches.csv.gz",
    "network_effects.csv",
    "network_effect_qc.csv",
    "primary_statistics.csv",
    "direction_component_statistics.csv",
    "sensitivity_effects.csv.gz",
    "sensitivity_statistics.csv.gz",
)


@dataclass(frozen=True)
class SignalVariant:
    """One conditioning branch represented at the maximum fitted PCA rank."""

    place: dict[str, np.ndarray]
    grid: dict[str, dict[str, np.ndarray]]
    feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedGroup:
    """Loaded and transformed trajectories for one network-condition-session."""

    trajectories: tuple[TrajectoryData, ...]
    scores: dict[str, PopulationScores]
    variants: dict[str, SignalVariant]
    predictive_variants: dict[str, dict[str, SignalVariant]]
    segmentations: dict[str, SegmentationResult]
    speed: dict[str, np.ndarray]
    heading: dict[str, np.ndarray]

    @property
    def first(self) -> TrajectoryData:
        return self.trajectories[0]

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(next(iter(self.scores.values())).grid))


@dataclass(frozen=True)
class RunResult:
    output_directory: Path
    selected_parameters: dict[str, int | str]
    n_estimate_rows: int
    n_primary_network_effects: int
    n_input_trajectories: int


def _atomic_write_csv(
    frame: pd.DataFrame, path: str | Path, *, compression: str | None = None
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False, compression=compression)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _prepare_output_directory(output: Path, *, overwrite: bool) -> None:
    """Create an output directory and remove only known stale artifacts."""

    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --overwrite to replace "
            "known output files without deleting unrelated files."
        )
    output.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for name in KNOWN_OUTPUT_FILES:
        candidate = output / name
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink()
    pca_directory = output / "pca"
    if pca_directory.is_symlink() or pca_directory.is_file():
        pca_directory.unlink()
    elif pca_directory.is_dir():
        shutil.rmtree(pca_directory)


def _session_equal(series: pd.Series, value: object) -> pd.Series:
    return series.astype(str).eq(str(value))


def _target_mask(frame: pd.DataFrame, target: Mapping[str, Any]) -> pd.Series:
    return (
        frame["stage"].astype(str).eq(str(target["stage"]))
        & frame["environment"].astype(str).eq(str(target["environment"]))
        & _session_equal(frame["session"], target["session"])
    )


def validate_manifest_design(manifest: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Validate replication, familiar baselines, and prespecified target coverage."""

    rows: list[dict[str, Any]] = []
    network_ids = sorted(manifest["network_id"].astype(str).unique())
    baseline_mask = (
        manifest["stage"].astype(str).eq(str(config.baseline["stage"]))
        & manifest["environment"].astype(str).eq(str(config.baseline["environment"]))
        & _session_equal(manifest["session"], config.baseline["session"])
    )
    baseline_networks = set(manifest.loc[baseline_mask, "network_id"].astype(str))
    missing_baseline = sorted(set(network_ids).difference(baseline_networks))
    if missing_baseline:
        raise ValueError(
            "Every independent network requires the declared familiar baseline; "
            f"missing {missing_baseline}"
        )

    for key, group in manifest.groupby(list(GROUP_COLUMNS), dropna=False, sort=True):
        n_trajectories = int(group["trajectory_id"].nunique())
        row = dict(zip(GROUP_COLUMNS, key))
        row.update(
            {
                "n_trajectories": n_trajectories,
                "status": "ok" if n_trajectories >= 2 else "insufficient_trajectories",
            }
        )
        rows.append(row)
        if n_trajectories < 2:
            raise ValueError(
                "Leave-one-trajectory-out conditioning and predictive TE require "
                f"at least two trajectories in group {key}; found {n_trajectories}"
            )

    targets = config.statistics.get("primary_targets", [])
    if not targets:
        raise ValueError("statistics.primary_targets must prespecify the primary family")
    expected_networks = set(network_ids)
    for target in targets:
        selected = manifest.loc[_target_mask(manifest, target)]
        observed_networks = set(selected["network_id"].astype(str))
        missing = sorted(expected_networks.difference(observed_networks))
        if missing:
            raise ValueError(
                f"Primary target {dict(target)} is missing networks {missing}"
            )
        axes = set(selected["deformed_axis"].astype(str).str.lower())
        if not axes.issubset({"x", "y"}) or len(axes) != 1:
            raise ValueError(
                f"Primary target {dict(target)} must declare one shared x/y deformed_axis; "
                f"found {sorted(axes)}"
            )
    return pd.DataFrame(rows)


def _input_qc_rows(trajectories: Sequence[TrajectoryData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in trajectories:
        rows.append(
            {
                "network_id": item.network_id,
                "stage": item.stage,
                "environment": item.environment,
                "session": item.session,
                "trajectory_id": item.trajectory_id,
                "data_path": str(item.data_path) if item.data_path else "",
                "n_time": item.n_time,
                "duration_s": (item.n_time - 1) * item.dt_s,
                "dt_s": item.dt_s,
                "n_place_cells": item.place_activity.shape[1],
                "grid_modules": ";".join(sorted(item.grid_activity)),
                "grid_cells_by_module": json.dumps(
                    {key: value.shape[1] for key, value in item.grid_activity.items()},
                    sort_keys=True,
                ),
                "n_bvc_cells": (
                    0 if item.bvc_activity is None else item.bvc_activity.shape[1]
                ),
                "status": "ok",
            }
        )
    return rows


def _manifest_group_rows(
    manifest: pd.DataFrame,
) -> Iterable[tuple[tuple[str, str, str, str], pd.DataFrame]]:
    ordered = manifest.copy()
    ordered["_session_text"] = ordered["session"].astype(str)
    ordered = ordered.sort_values(
        ["network_id", "stage", "environment", "_session_text", "trajectory_id"],
        kind="mergesort",
    )
    for key, group in ordered.groupby(
        ["network_id", "stage", "environment", "_session_text"],
        sort=True,
        dropna=False,
    ):
        yield tuple(str(value) for value in key), group.drop(columns="_session_text")


def _load_group(
    group: pd.DataFrame, config: Config, manifest_dir: str | Path
) -> tuple[TrajectoryData, ...]:
    trajectories = tuple(
        load_trajectory_npz(row, config, manifest_dir=manifest_dir)
        for _, row in group.sort_values("trajectory_id", kind="mergesort").iterrows()
    )
    if len({item.trajectory_id for item in trajectories}) < 2:
        raise ValueError("A TE group must contain at least two trajectory IDs")
    return trajectories


def _fit_pcas_from_manifest(
    manifest: pd.DataFrame,
    config: Config,
    output_directory: Path,
) -> tuple[dict[str, FrozenPopulationPCA], list[dict[str, Any]]]:
    baseline = manifest.loc[
        manifest["stage"].astype(str).eq(str(config.baseline["stage"]))
        & manifest["environment"].astype(str).eq(str(config.baseline["environment"]))
        & _session_equal(manifest["session"], config.baseline["session"])
    ]
    manifest_dir = manifest.attrs["manifest_dir"]
    component_counts = [
        int(config.representation["primary_components"]),
        *[int(v) for v in config.representation.get("sensitivity_components", [])],
    ]
    max_components = max(component_counts)
    models: dict[str, FrozenPopulationPCA] = {}
    input_qc: list[dict[str, Any]] = []
    for network_id, rows in baseline.groupby("network_id", sort=True):
        trajectories = _load_group(rows, config, manifest_dir)
        input_qc.extend(_input_qc_rows(trajectories))
        models[str(network_id)] = fit_population_pca(
            trajectories,
            n_components=max_components,
            max_samples=config.representation["pca_fit_max_samples"],
            random_seed=derive_seed(config.random_seed, "pca", network_id),
            baseline=config.baseline,
        )
    save_pca_collection(models, output_directory / "pca")
    return models, input_qc


def _pca_variance_table(models: Mapping[str, FrozenPopulationPCA]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for network_id, model in sorted(models.items()):
        populations = [("place", "", model.place)] + [
            ("grid", module, projector)
            for module, projector in sorted(model.grid.items())
        ]
        for population, module, projector in populations:
            cumulative = np.cumsum(projector.explained_variance_ratio_)
            for index, (variance, ratio, total) in enumerate(
                zip(
                    projector.explained_variance_,
                    projector.explained_variance_ratio_,
                    cumulative,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "network_id": network_id,
                        "population": population,
                        "module": module,
                        "component": index,
                        "explained_variance": float(variance),
                        "explained_variance_ratio": float(ratio),
                        "cumulative_explained_variance_ratio": float(total),
                        "fit_samples": model.fitted_sample_count,
                    }
                )
    return pd.DataFrame(rows)


def _raw_variant(scores: Mapping[str, PopulationScores]) -> SignalVariant:
    modules = sorted(next(iter(scores.values())).grid)
    return SignalVariant(
        place={key: value.place for key, value in scores.items()},
        grid={
            module: {key: value.grid[module] for key, value in scores.items()}
            for module in modules
        },
    )


def _residual_variant(
    residuals: CrossFittedPopulationResiduals,
) -> SignalVariant:
    return SignalVariant(
        place=residuals.place,
        grid=residuals.grid,
        feature_names=residuals.feature_names,
    )


def prepare_group(
    trajectories: Sequence[TrajectoryData],
    pca: FrozenPopulationPCA,
    config: Config,
) -> PreparedGroup:
    """Project, cross-fit nuisance models, and segment one analysis group."""

    ordered = tuple(sorted(trajectories, key=lambda item: item.trajectory_id))
    max_components = pca.max_components
    scores = {
        item.trajectory_id: pca.transform(item, max_components) for item in ordered
    }
    variants: dict[str, SignalVariant] = {}
    configured_branches = tuple(config.conditioning["branches"])
    if "unconditioned" in configured_branches:
        variants["unconditioned"] = _raw_variant(scores)
    spec = BehaviorDesignSpec.from_config(config.conditioning)
    if "behavior_conditioned" in configured_branches:
        residuals = cross_fit_population_residuals(
            ordered,
            scores,
            spec=spec,
            ridge_alpha=float(config.conditioning["ridge"]),
            include_bvc=False,
        )
        variants["behavior_conditioned"] = _residual_variant(residuals)
    if "behavior_bvc_conditioned" in configured_branches:
        if not bool(config.conditioning.get("include_bvc_sensitivity", False)):
            raise ValueError(
                "behavior_bvc_conditioned branch requires include_bvc_sensitivity=true"
            )
        residuals = cross_fit_population_residuals(
            ordered,
            scores,
            spec=spec,
            ridge_alpha=float(config.conditioning["ridge"]),
            include_bvc=True,
            bvc_components=int(config.conditioning["bvc_components"]),
        )
        variants["behavior_bvc_conditioned"] = _residual_variant(residuals)
    if config.conditioning["primary"] not in variants:
        raise ValueError("The configured primary conditioning branch was not prepared")

    predictive_variants: dict[str, dict[str, SignalVariant]] = {}
    if bool(config.te["compute_cv_score"]):
        trajectory_ids = tuple(item.trajectory_id for item in ordered)
        for branch in configured_branches:
            if branch == "unconditioned":
                continue
            predictive_variants[branch] = {}
            for held_out in trajectory_ids:
                training_ids = tuple(
                    value for value in trajectory_ids if value != held_out
                )
                fold_residuals = residualize_population_from_training_fold(
                    ordered,
                    scores,
                    training_trajectory_ids=training_ids,
                    spec=spec,
                    ridge_alpha=float(config.conditioning["ridge"]),
                    include_bvc=(branch == "behavior_bvc_conditioned"),
                    bvc_components=int(config.conditioning["bvc_components"]),
                )
                predictive_variants[branch][held_out] = _residual_variant(
                    fold_residuals
                )

    segmentations = {
        item.trajectory_id: segment_trajectory(item, config.segmentation)
        for item in ordered
    }
    speed: dict[str, np.ndarray] = {}
    heading: dict[str, np.ndarray] = {}
    for item in ordered:
        speed[item.trajectory_id], heading[item.trajectory_id] = derive_kinematics(item)
    return PreparedGroup(
        trajectories=ordered,
        scores=scores,
        variants=variants,
        predictive_variants=predictive_variants,
        segmentations=segmentations,
        speed=speed,
        heading=heading,
    )


def _movement_pool(
    prepared: PreparedGroup, state: str, wall_axis: str
) -> list[tuple[TrajectoryData, SegmentationResult, MovementSegment]]:
    result: list[tuple[TrajectoryData, SegmentationResult, MovementSegment]] = []
    for trajectory in prepared.trajectories:
        segmentation = prepared.segmentations[trajectory.trajectory_id]
        for movement in segmentation.accepted:
            if movement.direction == state and movement.wall_axis == wall_axis:
                result.append((trajectory, segmentation, movement))
    return result


def _core_segments(
    prepared: PreparedGroup,
    *,
    branch: str,
    module: str,
    state: str,
    wall_axis: str,
    direction: str,
    n_components: int,
    variant_override: SignalVariant | None = None,
) -> tuple[Segment, ...]:
    variant = prepared.variants[branch] if variant_override is None else variant_override
    if direction not in {"place_to_grid", "grid_to_place"}:
        raise ValueError(f"Unknown TE direction {direction!r}")
    segments: list[Segment] = []
    movement_pool = _movement_pool(prepared, state, wall_axis)
    segments_per_trajectory: dict[str, int] = {}
    for trajectory, _, _ in movement_pool:
        trajectory_id = str(trajectory.trajectory_id)
        segments_per_trajectory[trajectory_id] = (
            segments_per_trajectory.get(trajectory_id, 0) + 1
        )
    for trajectory, segmentation, movement in movement_pool:
        trajectory_id = trajectory.trajectory_id
        place = variant.place[trajectory_id][:, :n_components]
        grid = variant.grid[module][trajectory_id][:, :n_components]
        source, target = (
            (place, grid) if direction == "place_to_grid" else (grid, place)
        )
        start, stop = movement.start, movement.stop
        heading = prepared.heading[trajectory_id][start:stop]
        valid_heading = np.isfinite(heading)
        heading_sin = np.zeros_like(heading)
        heading_cos = np.zeros_like(heading)
        heading_sin[valid_heading] = np.sin(heading[valid_heading])
        heading_cos[valid_heading] = np.cos(heading[valid_heading])
        metadata = {
            **movement.metadata,
            "state": movement.direction,
            "wall": movement.wall,
            "wall_axis": movement.wall_axis,
            "axis_relation": movement.axis_relation,
            "position": trajectory.positions[start:stop],
            "speed": prepared.speed[trajectory_id][start:stop],
            "heading_sin": heading_sin,
            "heading_cos": heading_cos,
            "wall_distances": segmentation.wall_distances_m[start:stop],
        }
        segments.append(
            Segment(
                source=source[start:stop],
                target=target[start:stop],
                conditioning=None,
                segment_id=movement.segment_id,
                trajectory_id=trajectory_id,
                # build_lagged_samples normalizes rows within each segment.
                # This second level makes every trajectory contribute total
                # weight one, irrespective of its number of accepted segments.
                weight=1.0 / segments_per_trajectory[str(trajectory_id)],
                metadata=metadata,
            )
        )
    return tuple(segments)


def _predictive_fold_inputs(
    prepared: PreparedGroup,
    *,
    branch: str,
    module: str,
    state: str,
    axis_label: str,
    direction: str,
    n_components: int,
    lag_frames: int,
    source_history: int,
    target_history: int,
) -> tuple[PredictiveTEInputFold, ...]:
    """Build outer-fold samples with nuisance models trained without test data."""

    if branch not in prepared.predictive_variants:
        raise ValueError(f"No fold-specific predictive variants for {branch!r}")
    folds: list[PredictiveTEInputFold] = []
    for held_out, variant in sorted(prepared.predictive_variants[branch].items()):
        segments = _core_segments(
            prepared,
            branch=branch,
            module=module,
            state=state,
            wall_axis=axis_label,
            direction=direction,
            n_components=n_components,
            variant_override=variant,
        )
        samples = build_lagged_samples(
            segments,
            source_history=source_history,
            target_history=target_history,
            prediction_delay=lag_frames,
            equal_segment_weight=True,
            min_valid_samples_per_segment=1,
        )
        test_mask = np.asarray(samples.trajectory_id, dtype=str) == str(held_out)
        if not np.any(test_mask) or np.all(test_mask):
            raise ValueError(
                f"Predictive fold {held_out!r} lacks training or test samples"
            )
        folds.append(
            PredictiveTEInputFold(
                held_out_group=held_out,
                training_samples=samples.subset(~test_mask),
                test_samples=samples.subset(test_mask),
            )
        )
    return tuple(folds)


def _empty_quality(reason: str, segments: Sequence[Segment]) -> TEQuality:
    return TEQuality(
        status="insufficient_data",
        reason=reason,
        n_segments=len(segments),
        n_trajectories=len({str(segment.trajectory_id) for segment in segments}),
        n_samples=0,
        kish_effective_n=0.0,
        temporal_effective_n=0.0,
        joint_dimension=0,
        minimum_required_n=0.0,
        maximum_autocorrelation_time=float("nan"),
    )


def _estimate_pool(
    segments: Sequence[Segment],
    *,
    lag_frames: int,
    source_history: int,
    target_history: int,
    config: Config,
    n_bias_shuffles: int,
    seed: int,
    compute_cross_null: bool,
    compute_cv_score: bool,
    predictive_fold_builder: Callable[[], tuple[PredictiveTEInputFold, ...]] | None,
    require_multiple_trajectories: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Estimate one direction and return estimate, null values, and match audit."""

    segment_tuple = tuple(segments)
    null_records: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    if not segment_tuple:
        quality = _empty_quality("no accepted segments in this state/axis pool", ())
        return {
            **quality.as_dict(),
            "te_raw_bits": np.nan,
            "te_null_mean_bits": np.nan,
            "te_null_sd_bits": np.nan,
            "te_bias_corrected_bits": np.nan,
            "te_bits": np.nan,
            "surrogate_p": np.nan,
            "te_cv_bits": np.nan,
            "n_cv_folds": 0,
            "cv_status": "not_run_qc",
            "cv_reason": quality.reason,
            "bias_correction_applied": False,
            "cross_null_mean_bits": np.nan,
            "cross_null_sd_bits": np.nan,
            "cross_null_p": np.nan,
            "cross_null_repetitions_valid": 0,
        }, null_records, match_records

    try:
        samples = build_lagged_samples(
            segment_tuple,
            source_history=source_history,
            target_history=target_history,
            prediction_delay=lag_frames,
            equal_segment_weight=True,
            min_valid_samples_per_segment=1,
        )
    except ValueError as exc:
        quality = _empty_quality(str(exc), segment_tuple)
        return {
            **quality.as_dict(),
            "te_raw_bits": np.nan,
            "te_null_mean_bits": np.nan,
            "te_null_sd_bits": np.nan,
            "te_bias_corrected_bits": np.nan,
            "te_bits": np.nan,
            "surrogate_p": np.nan,
            "te_cv_bits": np.nan,
            "n_cv_folds": 0,
            "cv_status": "not_run_qc",
            "cv_reason": quality.reason,
            "bias_correction_applied": False,
            "cross_null_mean_bits": np.nan,
            "cross_null_sd_bits": np.nan,
            "cross_null_p": np.nan,
            "cross_null_repetitions_valid": 0,
        }, null_records, match_records

    quality = assess_te_quality(
        samples,
        segment_tuple,
        minimum_pool_samples=int(config.te["minimum_pool_samples"]),
        minimum_effective_samples_per_dimension=float(
            config.te["minimum_effective_samples_per_dimension"]
        ),
        autocorrelation_max_lag=int(config.te.get("autocorrelation_max_lag", 100)),
        require_multiple_trajectories=require_multiple_trajectories,
    )
    base: dict[str, Any] = quality.as_dict()
    if not quality.estimable:
        base.update(
            {
                "te_raw_bits": np.nan,
                "te_null_mean_bits": np.nan,
                "te_null_sd_bits": np.nan,
                "te_bias_corrected_bits": np.nan,
                "te_bits": np.nan,
                "surrogate_p": np.nan,
                "te_cv_bits": np.nan,
                "n_cv_folds": 0,
                "cv_status": "not_run_qc",
                "cv_reason": quality.reason,
                "bias_correction_applied": False,
                "cross_null_mean_bits": np.nan,
                "cross_null_sd_bits": np.nan,
                "cross_null_p": np.nan,
                "cross_null_repetitions_valid": 0,
            }
        )
        return base, null_records, match_records

    shrinkage = float(config.te["covariance_shrinkage"])
    observed = estimate_gcmi_te(samples, gaussianize=True, shrinkage=shrinkage)
    cv_value = np.nan
    n_folds = 0
    cv_status = "not_requested"
    cv_reason = ""
    if compute_cv_score:
        try:
            if predictive_fold_builder is None:
                cv = estimate_cross_validated_predictive_te(
                    samples,
                    ridge=float(config.te["regression_ridge"]),
                    gaussianize=True,
                    shrinkage=shrinkage,
                )
            else:
                cv = estimate_cross_validated_predictive_te_from_folds(
                    predictive_fold_builder(),
                    ridge=float(config.te["regression_ridge"]),
                    gaussianize=True,
                    shrinkage=shrinkage,
                )
            cv_value = cv.value_bits_per_sample
            n_folds = len(cv.folds)
            cv_status = "ok"
        except (ValueError, np.linalg.LinAlgError) as exc:
            cv_status = "failed"
            cv_reason = str(exc)

    null_mean = np.nan
    null_sd = np.nan
    corrected = observed.value_bits
    surrogate_p = np.nan
    if n_bias_shuffles > 0:
        def scalar_estimator(candidate_segments: Sequence[Segment]) -> float:
            candidate_samples = build_lagged_samples(
                candidate_segments,
                source_history=source_history,
                target_history=target_history,
                prediction_delay=lag_frames,
                equal_segment_weight=True,
            )
            return estimate_gcmi_te(
                candidate_samples, gaussianize=True, shrinkage=shrinkage
            ).value_bits

        shifted = circular_shift_test(
            segment_tuple,
            scalar_estimator,
            n_surrogates=n_bias_shuffles,
            seed=seed,
            min_circular_distance=int(config.te["minimum_shift_frames"]),
            alternative="greater",
        )
        null_mean = shifted.null_mean
        null_sd = shifted.null_standard_deviation
        corrected = shifted.bias_corrected
        surrogate_p = shifted.p_value
        null_records.extend(
            {
                "null_type": "circular_shift",
                "repetition": index,
                "te_bits": float(value),
            }
            for index, value in enumerate(shifted.null_distribution)
        )

    cross_distribution: list[float] = []
    if compute_cross_null and bool(config.cross_trajectory_null["enabled"]):
        repetitions = int(config.cross_trajectory_null["repetitions"])
        for repetition in range(repetitions):
            try:
                matched = match_cross_trajectory_segments(
                    segment_tuple,
                    segment_tuple,
                    length_tolerance=float(
                        config.cross_trajectory_null["length_tolerance_fraction"]
                    ),
                    caliper=float(config.cross_trajectory_null["match_caliper_sd"]),
                    randomize_within_caliper=bool(
                        config.cross_trajectory_null["randomize_within_caliper"]
                    ),
                    seed=derive_seed(seed, "matched_null", repetition),
                )
                matched_samples = build_lagged_samples(
                    matched.segments,
                    source_history=source_history,
                    target_history=target_history,
                    prediction_delay=lag_frames,
                    equal_segment_weight=True,
                )
                matched_quality = assess_te_quality(
                    matched_samples,
                    matched.segments,
                    minimum_pool_samples=int(config.te["minimum_pool_samples"]),
                    minimum_effective_samples_per_dimension=float(
                        config.te["minimum_effective_samples_per_dimension"]
                    ),
                    autocorrelation_max_lag=int(
                        config.te.get("autocorrelation_max_lag", 100)
                    ),
                )
                if not matched_quality.estimable:
                    continue
                value = estimate_gcmi_te(
                    matched_samples, gaussianize=True, shrinkage=shrinkage
                ).value_bits
            except (ValueError, np.linalg.LinAlgError):
                continue
            cross_distribution.append(float(value))
            null_records.append(
                {
                    "null_type": "matched_cross_trajectory",
                    "repetition": repetition,
                    "te_bits": float(value),
                    "n_matches": matched.audit.n_matches,
                }
            )
            if repetition == 0:
                match_records.extend(matched.audit.records())

    cross_mean = float(np.mean(cross_distribution)) if cross_distribution else np.nan
    cross_sd = (
        float(np.std(cross_distribution, ddof=1))
        if len(cross_distribution) > 1
        else np.nan
    )
    cross_p = (
        plus_one_pvalue(observed.value_bits, cross_distribution, alternative="greater")
        if cross_distribution
        else np.nan
    )
    base.update(
        {
            "te_raw_bits": observed.value_bits,
            "te_null_mean_bits": null_mean,
            "te_null_sd_bits": null_sd,
            "te_bias_corrected_bits": corrected,
            "te_bits": (
                corrected
                if config.te["effect_value"] == "te_bias_corrected_bits"
                else observed.value_bits
            ),
            "surrogate_p": surrogate_p,
            "te_cv_bits": cv_value,
            "n_cv_folds": n_folds,
            "cv_status": cv_status,
            "cv_reason": cv_reason,
            "bias_correction_applied": bool(n_bias_shuffles > 0),
            "cross_null_mean_bits": cross_mean,
            "cross_null_sd_bits": cross_sd,
            "cross_null_p": cross_p,
            "cross_null_repetitions_valid": len(cross_distribution),
        }
    )
    return base, null_records, match_records


def _parameter_grid(
    component_counts: Sequence[int],
    lag_frames: Sequence[int],
    source_histories: Sequence[int],
    target_histories: Sequence[int],
) -> list[tuple[int, int, int, int]]:
    return [
        tuple(int(value) for value in values)
        for values in product(
            component_counts, lag_frames, source_histories, target_histories
        )
    ]


def estimate_prepared_group(
    prepared: PreparedGroup,
    config: Config,
    *,
    analysis_set: str,
    parameters: Sequence[tuple[int, int, int, int]],
    branches: Sequence[str],
    n_bias_shuffles: int,
    cross_null_for_primary: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate all requested population directions for one prepared group."""

    first = prepared.first
    estimate_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for branch in branches:
        if branch not in prepared.variants:
            raise ValueError(f"Conditioning branch {branch!r} was not prepared")
        for n_components, lag, source_history, target_history in parameters:
            for module, state, wall_axis, direction in product(
                prepared.modules,
                ("approach", "departure"),
                ("x", "y"),
                ("place_to_grid", "grid_to_place"),
            ):
                segments = _core_segments(
                    prepared,
                    branch=branch,
                    module=module,
                    state=state,
                    wall_axis=wall_axis,
                    direction=direction,
                    n_components=n_components,
                )
                row_seed = derive_seed(
                    config.random_seed,
                    first.network_id,
                    first.stage,
                    first.environment,
                    first.session,
                    branch,
                    n_components,
                    lag,
                    source_history,
                    target_history,
                    module,
                    state,
                    wall_axis,
                    direction,
                    analysis_set,
                )
                compute_cross = bool(
                    cross_null_for_primary
                    and branch == config.conditioning["primary"]
                )
                compute_cv = bool(config.te["compute_cv_score"]) and (
                    analysis_set != "sensitivity"
                )
                predictive_builder = None
                if compute_cv and branch != "unconditioned":
                    predictive_builder = lambda: _predictive_fold_inputs(
                        prepared,
                        branch=branch,
                        module=module,
                        state=state,
                        axis_label=wall_axis,
                        direction=direction,
                        n_components=n_components,
                        lag_frames=lag,
                        source_history=source_history,
                        target_history=target_history,
                    )
                values, null_values, matches = _estimate_pool(
                    segments,
                    lag_frames=lag,
                    source_history=source_history,
                    target_history=target_history,
                    config=config,
                    n_bias_shuffles=n_bias_shuffles,
                    seed=row_seed,
                    compute_cross_null=compute_cross,
                    compute_cv_score=compute_cv,
                    predictive_fold_builder=predictive_builder,
                    require_multiple_trajectories=not (
                        branch == "unconditioned"
                        and bool(
                            config.data.get(
                                "allow_single_trajectory_unconditioned",
                                False,
                            )
                        )
                    ),
                )
                identity = {
                    "network_id": first.network_id,
                    "stage": first.stage,
                    "environment": first.environment,
                    "session": first.session,
                    "trajectory_id": "pooled_crossfit",
                    "module": module,
                    "state": state,
                    "wall_axis": wall_axis,
                    "axis_role": (
                        "baseline"
                        if first.deformed_axis is None
                        else "deformed" if wall_axis == first.deformed_axis else "orthogonal"
                    ),
                    "direction": direction,
                    "conditioning": branch,
                    "pca_components": n_components,
                    "lag_frames": lag,
                    "lag_ms": lag * first.dt_s * 1000.0,
                    "source_history": source_history,
                    "target_history": target_history,
                    "analysis_set": analysis_set,
                    "seed": row_seed,
                }
                estimate_rows.append({**identity, **values})
                null_rows.extend({**identity, **record} for record in null_values)
                match_rows.extend({**identity, **record} for record in matches)
    return (
        pd.DataFrame(estimate_rows),
        pd.DataFrame(null_rows),
        pd.DataFrame(match_rows),
    )


def _relative_axis_table(
    estimates: pd.DataFrame,
    config: Config,
    target: Mapping[str, Any],
    target_axis: str,
) -> pd.DataFrame:
    baseline_mask = (
        estimates["stage"].astype(str).eq(str(config.baseline["stage"]))
        & estimates["environment"].astype(str).eq(str(config.baseline["environment"]))
        & _session_equal(estimates["session"], config.baseline["session"])
    )
    selected = estimates.loc[baseline_mask | _target_mask(estimates, target)].copy()
    selected["wall_axis"] = np.where(
        selected["wall_axis"].astype(str).eq(target_axis),
        "deformed",
        "orthogonal",
    )
    return selected


def _effects_for_target(
    estimates: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Config,
    target: Mapping[str, Any],
    *,
    sensitivity_columns: Sequence[str],
) -> pd.DataFrame:
    target_manifest = manifest.loc[_target_mask(manifest, target)]
    axes = sorted(set(target_manifest["deformed_axis"].astype(str).str.lower()))
    if len(axes) != 1 or axes[0] not in {"x", "y"}:
        raise ValueError(f"Target {dict(target)} has ambiguous deformed_axis {axes}")
    relative = _relative_axis_table(estimates, config, target, axes[0])
    paired = build_direction_difference(
        relative,
        value_col="te_bits",
        direction_col="direction",
        unit_cols=DEFAULT_UNIT_COLUMNS,
        sensitivity_cols=sensitivity_columns,
        missing="drop",
    )
    n_modules = int(relative["module"].nunique())
    network = aggregate_to_network(
        paired,
        sensitivity_cols=sensitivity_columns,
        expected_trajectories=1,
        expected_modules=n_modules,
        missing="drop",
    )
    complete = network.loc[
        network["complete_trajectory_design"] & network["complete_module_design"]
    ].copy()
    if complete.empty:
        return pd.DataFrame()
    deltas = compute_familiar_deltas(
        complete,
        sensitivity_cols=sensitivity_columns,
        baseline_environment=str(config.baseline["environment"]),
        target_environments=[str(target["environment"])],
        missing="drop",
    )
    if deltas.empty:
        return pd.DataFrame()
    return compute_primary_did(
        deltas,
        sensitivity_cols=sensitivity_columns,
        missing="drop",
    )


def build_network_effects(
    estimates: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Config,
    *,
    sensitivity: bool,
) -> pd.DataFrame:
    """Construct complete-design network effects for every non-baseline group."""

    parameter_columns = list(PARAMETER_COLUMNS) if sensitivity else []
    targets = list(config.statistics["primary_targets"])
    results: list[pd.DataFrame] = []
    for target in targets:
        effect = _effects_for_target(
            estimates,
            manifest,
            config,
            target,
            sensitivity_columns=parameter_columns,
        )
        if not effect.empty:
            results.append(effect)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _filter_primary_targets(
    frame: pd.DataFrame, targets: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = np.zeros(len(frame), dtype=bool)
    for target in targets:
        mask |= _target_mask(frame, target).to_numpy()
    return frame.loc[mask].copy()


def _annotate_network_coverage(
    statistics: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    value_column: str,
    hypothesis_columns: Sequence[str],
    expected_network_ids: Sequence[str],
) -> pd.DataFrame:
    """Attach planned-versus-available independent-unit counts to inference."""

    result = statistics.copy()
    coverage_columns = (
        "n_networks_expected",
        "n_networks_available",
        "n_networks_missing",
        "missing_network_ids",
        "network_coverage_complete",
    )
    if result.empty:
        for column in coverage_columns:
            if column not in result:
                result[column] = pd.Series(dtype="object")
        return result

    expected = {str(value) for value in expected_network_ids}
    records: list[dict[str, Any]] = []
    numeric_values = pd.to_numeric(effects[value_column], errors="coerce")
    for _, row in result.iterrows():
        mask = np.ones(len(effects), dtype=bool)
        for column in hypothesis_columns:
            mask &= effects[column].astype(str).eq(str(row[column])).to_numpy()
        available = set(
            effects.loc[mask & np.isfinite(numeric_values), "network_id"].astype(str)
        )
        missing = sorted(expected.difference(available))
        records.append(
            {
                "n_networks_expected": len(expected),
                "n_networks_available": len(available),
                "n_networks_missing": len(missing),
                "missing_network_ids": ";".join(missing),
                "network_coverage_complete": not missing,
            }
        )
    return pd.concat(
        [result.reset_index(drop=True), pd.DataFrame(records)], axis=1
    )


def _network_effect_qc(
    manifest: pd.DataFrame,
    effects: pd.DataFrame,
    primary_estimates: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Audit every planned primary target-network effect before inference."""

    expected_networks = sorted(manifest["network_id"].astype(str).unique())
    if primary_estimates.empty:
        baseline_mask = pd.Series(dtype=bool)
    else:
        baseline_mask = (
            primary_estimates["stage"].astype(str).eq(
                str(config.baseline["stage"])
            )
            & primary_estimates["environment"].astype(str).eq(
                str(config.baseline["environment"])
            )
            & _session_equal(
                primary_estimates["session"], config.baseline["session"]
            )
        )
    rows: list[dict[str, Any]] = []
    analysis_requested = bool(config.execution["run_primary"])
    for target in config.statistics["primary_targets"]:
        for network_id in expected_networks:
            if effects.empty:
                effect_rows = effects
            else:
                effect_rows = effects.loc[
                    effects["network_id"].astype(str).eq(network_id)
                    & _target_mask(effects, target)
                ]

            if primary_estimates.empty:
                upstream = primary_estimates
            else:
                target_upstream = _target_mask(primary_estimates, target)
                upstream = primary_estimates.loc[
                    primary_estimates["network_id"].astype(str).eq(network_id)
                    & (baseline_mask | target_upstream)
                ]
            if upstream.empty:
                failed_upstream = upstream
            else:
                finite_effect = np.isfinite(
                    pd.to_numeric(upstream["te_bits"], errors="coerce")
                )
                failed_upstream = upstream.loc[
                    ~upstream["status"].astype(str).eq("ok") | ~finite_effect
                ]
            reasons = sorted(
                {
                    str(value)
                    for value in failed_upstream.get(
                        "reason", pd.Series(dtype=object)
                    ).dropna()
                    if str(value)
                }
            )

            if not analysis_requested:
                status = "not_requested"
                reason = "execution.run_primary is false"
            elif len(effect_rows) == 0:
                status = "missing_network_effect"
                reason = (
                    "complete paired baseline/target state-axis-direction design "
                    "was not available"
                )
            elif len(effect_rows) > 1:
                status = "duplicate_network_effect"
                reason = "more than one primary network effect was produced"
            elif not np.all(
                np.isfinite(
                    pd.to_numeric(
                        effect_rows[["C", "C_P_to_G", "C_G_to_P"]].iloc[0],
                        errors="coerce",
                    )
                )
            ):
                status = "nonfinite_network_effect"
                reason = "one or more primary contrast components are non-finite"
            else:
                status = "ok"
                reason = ""
            rows.append(
                {
                    "network_id": network_id,
                    "stage": target["stage"],
                    "environment": target["environment"],
                    "session": target["session"],
                    "status": status,
                    "reason": reason,
                    "n_effect_rows": int(len(effect_rows)),
                    "n_upstream_requested_cells": int(len(upstream)),
                    "n_upstream_failed_cells": int(len(failed_upstream)),
                    "upstream_failure_reasons": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _primary_statistics(
    effects: pd.DataFrame,
    config: Config,
    *,
    expected_network_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = config.statistics["primary_targets"]
    primary = _filter_primary_targets(effects, targets)
    if primary.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["stage", "environment", "session"]
    statistics = network_inference_table(
        primary,
        value_col="C",
        hypothesis_cols=group_cols,
        adjustment="holm",
        confidence_level=float(config.statistics["confidence_level"]),
        exact_max_networks=int(
            config.statistics["exact_sign_flip_max_networks"]
        ),
        sign_resamples=int(config.statistics["monte_carlo_sign_flips"]),
        bootstrap_resamples=int(config.statistics["bootstrap_draws"]),
        random_state=derive_seed(config.random_seed, "primary_inference"),
        missing="drop",
    )
    statistics["contrast_component"] = "C_net_direction"
    statistics = _annotate_network_coverage(
        statistics,
        primary,
        value_column="C",
        hypothesis_columns=group_cols,
        expected_network_ids=expected_network_ids,
    )

    direction_tables: list[pd.DataFrame] = []
    for value_col, label in (
        ("C_P_to_G", "C_place_to_grid"),
        ("C_G_to_P", "C_grid_to_place"),
    ):
        table = network_inference_table(
            primary,
            value_col=value_col,
            hypothesis_cols=group_cols,
            adjustment="holm",
            confidence_level=float(config.statistics["confidence_level"]),
            exact_max_networks=int(
                config.statistics["exact_sign_flip_max_networks"]
            ),
            sign_resamples=int(config.statistics["monte_carlo_sign_flips"]),
            bootstrap_resamples=int(config.statistics["bootstrap_draws"]),
            random_state=derive_seed(config.random_seed, "direction_inference", label),
            missing="drop",
        )
        table["contrast_component"] = label
        table = _annotate_network_coverage(
            table,
            primary,
            value_column=value_col,
            hypothesis_columns=group_cols,
            expected_network_ids=expected_network_ids,
        )
        direction_tables.append(table)
    direction = pd.concat(direction_tables, ignore_index=True)
    direction["p_holm_direction_family"] = holm_adjust(
        direction["p_value"].to_numpy(dtype=float)
    )
    return statistics, direction


def _sensitivity_statistics(
    effects: pd.DataFrame,
    config: Config,
    *,
    expected_network_ids: Sequence[str],
) -> pd.DataFrame:
    primary_targets = _filter_primary_targets(
        effects, config.statistics["primary_targets"]
    )
    if primary_targets.empty:
        return pd.DataFrame()
    hypothesis_columns = [
        "stage",
        "environment",
        "session",
        *PARAMETER_COLUMNS,
    ]
    method = str(config.statistics["sensitivity_multiplicity"])
    if method == "max_t":
        statistics = max_t_sign_flip(
            primary_targets,
            value_col="C",
            hypothesis_cols=hypothesis_columns,
            family_cols=(),
            exact_max_networks=int(config.statistics["exact_sign_flip_max_networks"]),
            n_resamples=int(config.statistics["monte_carlo_sign_flips"]),
            random_state=derive_seed(config.random_seed, "sensitivity_max_t"),
            missing="complete_case",
        )
    elif method == "fdr_bh":
        statistics = network_inference_table(
            primary_targets,
            value_col="C",
            hypothesis_cols=hypothesis_columns,
            adjustment="bh",
            confidence_level=float(config.statistics["confidence_level"]),
            exact_max_networks=int(
                config.statistics["exact_sign_flip_max_networks"]
            ),
            sign_resamples=int(config.statistics["monte_carlo_sign_flips"]),
            bootstrap_resamples=int(config.statistics["bootstrap_draws"]),
            random_state=derive_seed(config.random_seed, "sensitivity_fdr_bh"),
            missing="drop",
        )
    else:
        raise ValueError(f"Unknown sensitivity multiplicity method {method!r}")
    return _annotate_network_coverage(
        statistics,
        primary_targets,
        value_column="C",
        hypothesis_columns=hypothesis_columns,
        expected_network_ids=expected_network_ids,
    )


def run_analysis(
    config_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    output_directory: str | Path,
    overwrite: bool = False,
) -> RunResult:
    """Run baseline selection, primary TE, nulls, contrasts, and inference."""

    config = Config.from_json(config_path)
    manifest_source = (
        config.resolve_path(config.data["manifest_path"])
        if manifest_path is None
        else Path(manifest_path).resolve()
    )
    manifest = load_manifest(manifest_source, config=config, check_files=True)
    design_qc = validate_manifest_design(manifest, config)
    output = Path(output_directory).resolve()
    _prepare_output_directory(output, overwrite=overwrite)
    atomic_write_json(output / "config.resolved.json", config.to_dict())
    atomic_write_json(
        output / "run_metadata.json",
        {
            **runtime_metadata(config.to_dict(), manifest_source),
            "random_seed": int(config.random_seed),
            "primary_targets": config.statistics["primary_targets"],
        },
    )
    _atomic_write_csv(design_qc, output / "design_qc.csv")

    pcas, baseline_input_qc = _fit_pcas_from_manifest(manifest, config, output)
    _atomic_write_csv(_pca_variance_table(pcas), output / "pca_explained_variance.csv")

    manifest_dir = manifest.attrs["manifest_dir"]
    groups = list(_manifest_group_rows(manifest))
    primary_k = int(config.representation["primary_components"])
    selection_parameters = _parameter_grid(
        [primary_k],
        config.te["lag_frames"],
        config.te["source_history_frames"],
        config.te["target_history_frames"],
    )

    baseline_selection_frames: list[pd.DataFrame] = []
    input_qc_rows = list(baseline_input_qc)
    segment_qc_frames: list[pd.DataFrame] = []
    for _, rows in groups:
        is_baseline = (
            str(rows.iloc[0]["stage"]) == str(config.baseline["stage"])
            and str(rows.iloc[0]["environment"]) == str(config.baseline["environment"])
            and str(rows.iloc[0]["session"]) == str(config.baseline["session"])
        )
        if not is_baseline:
            continue
        trajectories = _load_group(rows, config, manifest_dir)
        prepared = prepare_group(trajectories, pcas[trajectories[0].network_id], config)
        segment_qc_frames.append(segmentation_qc_table(prepared.segmentations.values()))
        estimates, _, _ = estimate_prepared_group(
            prepared,
            config,
            analysis_set="baseline_selection",
            parameters=selection_parameters,
            branches=[str(config.conditioning["primary"])],
            n_bias_shuffles=0,
            cross_null_for_primary=False,
        )
        baseline_selection_frames.append(estimates)
    baseline_selection = pd.concat(baseline_selection_frames, ignore_index=True)
    _atomic_write_csv(baseline_selection, output / "baseline_selection_scores.csv")
    if bool(config.te["select_parameters_on_baseline"]):
        selection: SelectionResult = select_shared_parameters(
            baseline_selection,
            value_column=str(config.te["selection_value"]),
            primary_conditioning=str(config.conditioning["primary"]),
            fixed_pca_components=primary_k,
            expected_baseline_stage=str(config.baseline["stage"]),
            expected_network_ids=sorted(pcas),
        )
        selected = selection.parameters
        _atomic_write_csv(selection.scores, output / "baseline_selection_summary.csv")
    else:
        selected = {
            "conditioning": str(config.conditioning["primary"]),
            "pca_components": primary_k,
            "lag_frames": int(config.te["lag_frames"][0]),
            "source_history": int(config.te["source_history_frames"][0]),
            "target_history": int(config.te["target_history_frames"][0]),
        }
    atomic_write_json(output / "selected_parameters.json", selected)

    primary_parameters = [
        (
            int(selected["pca_components"]),
            int(selected["lag_frames"]),
            int(selected["source_history"]),
            int(selected["target_history"]),
        )
    ]
    all_components = sorted(
        {
            primary_k,
            *[int(value) for value in config.representation.get("sensitivity_components", [])],
        }
    )
    sensitivity_parameters = _parameter_grid(
        all_components,
        config.te["lag_frames"],
        config.te["source_history_frames"],
        config.te["target_history_frames"],
    )

    estimate_frames: list[pd.DataFrame] = []
    null_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    for _, rows in groups:
        trajectories = _load_group(rows, config, manifest_dir)
        input_qc_rows.extend(_input_qc_rows(trajectories))
        prepared = prepare_group(trajectories, pcas[trajectories[0].network_id], config)
        segment_qc_frames.append(segmentation_qc_table(prepared.segmentations.values()))
        if bool(config.execution["run_primary"]):
            estimates, nulls, matches = estimate_prepared_group(
                prepared,
                config,
                analysis_set="primary",
                parameters=primary_parameters,
                branches=config.conditioning["branches"],
                n_bias_shuffles=int(config.te["bias_shuffles"]),
                cross_null_for_primary=True,
            )
            estimate_frames.append(estimates)
            if not nulls.empty:
                null_frames.append(nulls)
            if not matches.empty:
                match_frames.append(matches)
        if bool(config.execution["run_sensitivity"]):
            estimates, nulls, _ = estimate_prepared_group(
                prepared,
                config,
                analysis_set="sensitivity",
                parameters=sensitivity_parameters,
                branches=config.conditioning["branches"],
                n_bias_shuffles=int(config.te["sensitivity_bias_shuffles"]),
                cross_null_for_primary=False,
            )
            estimate_frames.append(estimates)
            if not nulls.empty:
                null_frames.append(nulls)

    estimates = pd.concat(estimate_frames, ignore_index=True) if estimate_frames else pd.DataFrame()
    if estimates.empty:
        raise RuntimeError("No TE estimates were requested or produced")
    _atomic_write_csv(estimates, output / "te_estimates.csv.gz", compression="gzip")
    te_qc_columns = [
        *[column for column in estimates.columns if column not in {
            "te_raw_bits", "te_null_mean_bits", "te_null_sd_bits",
            "te_bias_corrected_bits", "te_bits", "surrogate_p", "te_cv_bits",
            "cross_null_mean_bits", "cross_null_sd_bits", "cross_null_p",
        }],
    ]
    _atomic_write_csv(estimates.loc[:, te_qc_columns], output / "te_qc.csv.gz", compression="gzip")
    input_qc = pd.DataFrame(input_qc_rows).drop_duplicates(
        ["network_id", "stage", "environment", "session", "trajectory_id"]
    )
    _atomic_write_csv(input_qc, output / "input_qc.csv")
    if segment_qc_frames:
        segment_qc = pd.concat(segment_qc_frames, ignore_index=True).drop_duplicates(
            ["network_id", "stage", "environment", "session", "segment_id"]
        )
        _atomic_write_csv(segment_qc, output / "segment_qc.csv.gz", compression="gzip")
    if null_frames and bool(config.execution["write_intermediate_scores"]):
        _atomic_write_csv(
            pd.concat(null_frames, ignore_index=True),
            output / "null_distributions.csv.gz",
            compression="gzip",
        )
    if match_frames:
        _atomic_write_csv(
            pd.concat(match_frames, ignore_index=True),
            output / "cross_trajectory_matches.csv.gz",
            compression="gzip",
        )

    primary_estimates = estimates.loc[
        estimates["analysis_set"].eq("primary")
        & estimates["conditioning"].astype(str).eq(
            str(config.conditioning["primary"])
        )
    ].copy()
    primary_effects = build_network_effects(
        primary_estimates, manifest, config, sensitivity=False
    )
    _atomic_write_csv(primary_effects, output / "network_effects.csv")
    expected_network_ids = sorted(manifest["network_id"].astype(str).unique())
    effect_qc = _network_effect_qc(
        manifest, primary_effects, primary_estimates, config
    )
    _atomic_write_csv(effect_qc, output / "network_effect_qc.csv")

    if bool(config.execution["fail_on_qc"]) and bool(
        config.execution["run_primary"]
    ):
        baseline_scope_mask = (
            primary_estimates["stage"].astype(str).eq(
                str(config.baseline["stage"])
            )
            & primary_estimates["environment"].astype(str).eq(
                str(config.baseline["environment"])
            )
            & _session_equal(
                primary_estimates["session"], config.baseline["session"]
            )
        ).to_numpy()
        target_scope_mask = np.zeros(len(primary_estimates), dtype=bool)
        for target in config.statistics["primary_targets"]:
            target_scope_mask |= _target_mask(primary_estimates, target).to_numpy()
        primary_scope = primary_estimates.loc[
            baseline_scope_mask | target_scope_mask
        ].copy()
        finite_primary = np.isfinite(
            pd.to_numeric(primary_scope["te_bits"], errors="coerce")
        )
        failed_te = primary_scope.loc[
            ~primary_scope["status"].astype(str).eq("ok") | ~finite_primary
        ]
        failed_effects = effect_qc.loc[~effect_qc["status"].eq("ok")]
        if not failed_te.empty or not failed_effects.empty:
            identifiers = [
                "network_id",
                "stage",
                "environment",
                "session",
                "module",
                "state",
                "wall_axis",
                "direction",
                "reason",
            ]
            example = failed_te.loc[:, identifiers].head(8).to_dict("records")
            raise RuntimeError(
                "Primary QC failed before inference: "
                f"{len(failed_te)} TE cells and {len(failed_effects)} planned "
                f"network effects failed; TE examples: {example}. See te_qc.csv.gz "
                "and network_effect_qc.csv."
            )
    primary_statistics, direction_statistics = _primary_statistics(
        primary_effects,
        config,
        expected_network_ids=expected_network_ids,
    )
    _atomic_write_csv(primary_statistics, output / "primary_statistics.csv")
    _atomic_write_csv(
        direction_statistics, output / "direction_component_statistics.csv"
    )

    if bool(config.execution["run_sensitivity"]):
        sensitivity_estimates = estimates.loc[
            estimates["analysis_set"] == "sensitivity"
        ].copy()
        sensitivity_effects = build_network_effects(
            sensitivity_estimates, manifest, config, sensitivity=True
        )
        _atomic_write_csv(
            sensitivity_effects,
            output / "sensitivity_effects.csv.gz",
            compression="gzip",
        )
        sensitivity_stats = _sensitivity_statistics(
            sensitivity_effects,
            config,
            expected_network_ids=expected_network_ids,
        )
        _atomic_write_csv(
            sensitivity_stats,
            output / "sensitivity_statistics.csv.gz",
            compression="gzip",
        )

    return RunResult(
        output_directory=output,
        selected_parameters=selected,
        n_estimate_rows=len(estimates),
        n_primary_network_effects=len(primary_effects),
        n_input_trajectories=len(input_qc),
    )
