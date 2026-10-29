"""Behavioral nuisance design and leave-one-trajectory-out residualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .io import TrajectoryData
from .pca import PCAProjector, PopulationScores


@dataclass(frozen=True)
class BehaviorDesignSpec:
    """Fixed, condition-independent behavioral feature definition."""

    rbf_grid: tuple[int, int] = (5, 5)
    rbf_width: float = 0.3
    speed_polynomial_degree: int = 3

    @classmethod
    def from_config(cls, conditioning: Mapping[str, Any]) -> "BehaviorDesignSpec":
        grid = conditioning["position_rbf_grid"]
        shape = (int(grid), int(grid)) if isinstance(grid, int) else (int(grid[0]), int(grid[1]))
        return cls(
            rbf_grid=shape,
            rbf_width=float(conditioning["position_rbf_width"]),
            speed_polynomial_degree=int(conditioning["speed_polynomial_degree"]),
        )

    def __post_init__(self) -> None:
        if len(self.rbf_grid) != 2 or min(self.rbf_grid) < 2:
            raise ValueError("rbf_grid must contain two integers >= 2")
        if not np.isfinite(self.rbf_width) or self.rbf_width <= 0:
            raise ValueError("rbf_width must be finite and positive")
        if self.speed_polynomial_degree < 1:
            raise ValueError("speed_polynomial_degree must be positive")


@dataclass(frozen=True)
class BehaviorDesign:
    values: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("Behavior design shape does not match feature_names")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Behavior design contains non-finite values")
        object.__setattr__(self, "values", matrix)


def _arena_bounds(trajectory: TrajectoryData) -> tuple[float, float, float, float]:
    # The canonical contract uses a lower-left origin unless explicit bounds are
    # carried as extra manifest columns.  Positions are never re-centered from data.
    metadata = trajectory.metadata
    xmin = float(metadata.get("x_min_m", 0.0))
    ymin = float(metadata.get("y_min_m", 0.0))
    xmax = float(metadata.get("x_max_m", xmin + trajectory.width_m))
    ymax = float(metadata.get("y_max_m", ymin + trajectory.height_m))
    if not np.isclose(xmax - xmin, trajectory.width_m, rtol=1e-6, atol=1e-9):
        raise ValueError("Explicit x bounds disagree with width_m")
    if not np.isclose(ymax - ymin, trajectory.height_m, rtol=1e-6, atol=1e-9):
        raise ValueError("Explicit y bounds disagree with height_m")
    return xmin, xmax, ymin, ymax


def derive_kinematics(trajectory: TrajectoryData) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar speed and heading without guessing signal columns."""

    if trajectory.speed is not None:
        if trajectory.speed.shape[1] != 1:
            raise ValueError("Declared speed array must have exactly one feature")
        speed = trajectory.speed[:, 0].copy()
    else:
        velocity = np.gradient(trajectory.positions, trajectory.dt_s, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
    if trajectory.heading is not None:
        if trajectory.heading.shape[1] != 1:
            raise ValueError("Declared heading array must have exactly one feature")
        heading = trajectory.heading[:, 0].copy()
    else:
        velocity = np.gradient(trajectory.positions, trajectory.dt_s, axis=0)
        heading = np.arctan2(velocity[:, 1], velocity[:, 0])
        stationary = np.linalg.norm(velocity, axis=1) <= np.finfo(float).eps
        heading[stationary] = np.nan
        # Undefined heading at exact zero speed is represented by sin=cos=0 in
        # build_behavior_design; retaining NaN here makes that choice explicit.
    if not np.all(np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("Speed must be finite and non-negative")
    return speed, heading


def build_behavior_design(
    trajectory: TrajectoryData,
    spec: BehaviorDesignSpec,
    *,
    bvc_scores: np.ndarray | None = None,
) -> BehaviorDesign:
    """Construct nonlinear position, movement, heading, and wall features.

    Position RBFs use arena-normalized coordinates and fixed centers. Four wall
    distances remain separate (left/right/bottom/top), and their nonlinear
    minimum is included as nearest-wall distance. Feature standardization is
    learned only inside each ridge training fold by :class:`RidgeResidualizer`.
    """

    positions = np.asarray(trajectory.positions, dtype=np.float64)
    xmin, xmax, ymin, ymax = _arena_bounds(trajectory)
    x_normalized = (positions[:, 0] - xmin) / (xmax - xmin)
    y_normalized = (positions[:, 1] - ymin) / (ymax - ymin)
    tolerance = 1e-6
    if (
        np.any(x_normalized < -tolerance)
        or np.any(x_normalized > 1 + tolerance)
        or np.any(y_normalized < -tolerance)
        or np.any(y_normalized > 1 + tolerance)
    ):
        raise ValueError(
            "Positions fall outside explicit arena bounds; provide x_min_m/y_min_m "
            "manifest columns rather than allowing bounds to be inferred"
        )
    x_normalized = np.clip(x_normalized, 0.0, 1.0)
    y_normalized = np.clip(y_normalized, 0.0, 1.0)

    nx, ny = spec.rbf_grid
    centers_x = np.linspace(0.0, 1.0, nx)
    centers_y = np.linspace(0.0, 1.0, ny)
    center_pairs = np.asarray(
        [(x, y) for y in centers_y for x in centers_x], dtype=np.float64
    )
    points = np.column_stack((x_normalized, y_normalized))
    squared_distance = np.sum(
        (points[:, None, :] - center_pairs[None, :, :]) ** 2, axis=2
    )
    rbfs = np.exp(-0.5 * squared_distance / (spec.rbf_width**2))
    values: list[np.ndarray] = [rbfs]
    names: list[str] = [
        f"position_rbf_x{ix:02d}_y{iy:02d}"
        for iy in range(ny)
        for ix in range(nx)
    ]

    speed, heading = derive_kinematics(trajectory)
    speed_terms = np.column_stack(
        [speed**degree for degree in range(1, spec.speed_polynomial_degree + 1)]
    )
    values.append(speed_terms)
    names.extend(
        f"speed_power_{degree}"
        for degree in range(1, spec.speed_polynomial_degree + 1)
    )

    valid_heading = np.isfinite(heading)
    heading_sin = np.zeros_like(speed)
    heading_cos = np.zeros_like(speed)
    heading_sin[valid_heading] = np.sin(heading[valid_heading])
    heading_cos[valid_heading] = np.cos(heading[valid_heading])
    values.append(np.column_stack((heading_sin, heading_cos)))
    names.extend(("heading_sin", "heading_cos"))

    walls = np.column_stack(
        (
            positions[:, 0] - xmin,
            xmax - positions[:, 0],
            positions[:, 1] - ymin,
            ymax - positions[:, 1],
        )
    )
    # The nonlinear nearest-wall term is not recoverable by a linear combination
    # of the four individual distances, so it is retained alongside them.
    values.append(np.column_stack((walls, np.min(walls, axis=1))))
    names.extend(
        (
            "wall_left_m",
            "wall_right_m",
            "wall_bottom_m",
            "wall_top_m",
            "nearest_wall_m",
        )
    )

    if bvc_scores is not None:
        bvc = np.asarray(bvc_scores, dtype=np.float64)
        if bvc.ndim != 2 or bvc.shape[0] != trajectory.n_time:
            raise ValueError("bvc_scores must have shape (n_time, n_components)")
        if not np.all(np.isfinite(bvc)):
            raise ValueError("bvc_scores contains non-finite values")
        values.append(bvc)
        names.extend(f"bvc_pc_{index + 1:02d}" for index in range(bvc.shape[1]))

    return BehaviorDesign(np.column_stack(values), tuple(names))


@dataclass(frozen=True)
class RidgeResidualizer:
    """Training-fold-standardized multivariate ridge nuisance model."""

    x_mean_: np.ndarray
    x_scale_: np.ndarray
    y_mean_: np.ndarray
    coefficients_: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls, design: np.ndarray, response: np.ndarray, alpha: float
    ) -> "RidgeResidualizer":
        x = np.asarray(design, dtype=np.float64)
        y = np.asarray(response, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("design and response must be aligned two-dimensional arrays")
        if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError("Insufficient data for ridge residualization")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Ridge inputs must be finite")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Ridge alpha must be finite and non-negative")
        x_mean = x.mean(axis=0)
        x_scale = x.std(axis=0, ddof=0)
        x_scale[x_scale <= np.finfo(float).eps] = 1.0
        standardized = (x - x_mean) / x_scale
        y_mean = y.mean(axis=0)
        centered_y = y - y_mean
        gram = standardized.T @ standardized / x.shape[0]
        cross = standardized.T @ centered_y / x.shape[0]
        penalized = gram + float(alpha) * np.eye(gram.shape[0])
        try:
            coefficients = np.linalg.solve(penalized, cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(penalized, hermitian=True) @ cross
        return cls(x_mean, x_scale, y_mean, coefficients, float(alpha))

    def predict(self, design: np.ndarray) -> np.ndarray:
        x = np.asarray(design, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coefficients_.shape[0]:
            raise ValueError("Prediction design has the wrong number of features")
        if not np.all(np.isfinite(x)):
            raise ValueError("Prediction design contains non-finite values")
        return self.y_mean_ + ((x - self.x_mean_) / self.x_scale_) @ self.coefficients_

    def residualize(self, design: np.ndarray, response: np.ndarray) -> np.ndarray:
        y = np.asarray(response, dtype=np.float64)
        prediction = self.predict(design)
        if y.shape != prediction.shape:
            raise ValueError("Response shape does not match ridge prediction")
        return y - prediction


def cross_fitted_ridge_residuals(
    designs: Mapping[str, np.ndarray],
    responses: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Residualize each trajectory using all other trajectories for training."""

    keys = sorted(designs)
    if set(keys) != set(responses):
        raise ValueError("Design and response trajectory keys differ")
    if len(keys) < 2:
        raise ValueError("Leave-one-trajectory-out residualization needs >=2 trajectories")
    result: dict[str, np.ndarray] = {}
    for held_out in keys:
        training = [key for key in keys if key != held_out]
        x_train = np.concatenate([np.asarray(designs[key]) for key in training], axis=0)
        y_train = np.concatenate([np.asarray(responses[key]) for key in training], axis=0)
        model = RidgeResidualizer.fit(x_train, y_train, alpha)
        result[held_out] = model.residualize(designs[held_out], responses[held_out])
    return result


@dataclass(frozen=True)
class CrossFittedPopulationResiduals:
    """Out-of-trajectory behavioral residuals for symmetric population scores."""

    place: dict[str, np.ndarray]
    grid: dict[str, dict[str, np.ndarray]]
    feature_names: tuple[str, ...]
    include_bvc: bool
    fold_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


def residualize_population_from_training_fold(
    trajectories: Sequence[TrajectoryData],
    scores: Mapping[str, PopulationScores],
    *,
    training_trajectory_ids: Sequence[str],
    spec: BehaviorDesignSpec,
    ridge_alpha: float,
    include_bvc: bool = False,
    bvc_components: int = 10,
    frozen_bvc_pca: PCAProjector | None = None,
) -> CrossFittedPopulationResiduals:
    """Fit nuisance models on an outer training fold and apply them to all rows.

    This function is used by held-out predictive TE.  In contrast to ordinary
    cross-fitted residuals, *every* residual array in one returned fold is
    generated by the same nuisance model fitted exclusively to the declared
    outer-training trajectories.  Consequently, the held-out trajectory never
    influences the residuals used to fit the TE prediction models.
    """

    if len(trajectories) < 2:
        raise ValueError("At least two trajectories are required for an outer fold")
    by_id = {str(item.trajectory_id): item for item in trajectories}
    if len(by_id) != len(trajectories):
        raise ValueError("trajectory_id values must be unique within a fold")
    keys = tuple(sorted(by_id))
    if set(scores) != set(keys):
        raise ValueError("PCA score keys must match trajectory_id values")
    training = tuple(dict.fromkeys(str(value) for value in training_trajectory_ids))
    if not training or not set(training).issubset(by_id):
        raise ValueError("training_trajectory_ids must be a non-empty known subset")
    held_out = tuple(key for key in keys if key not in set(training))
    if not held_out:
        raise ValueError("At least one trajectory must be held out")
    group_identity = {
        (str(item.network_id), str(item.stage), str(item.environment), str(item.session))
        for item in trajectories
    }
    if len(group_identity) != 1:
        raise ValueError(
            "An outer nuisance fold cannot mix networks, stages, environments, "
            "or sessions"
        )
    modules = set(next(iter(scores.values())).grid)
    if any(set(item.grid) != modules for item in scores.values()):
        raise ValueError("Grid module names differ across PCA score objects")

    bvc_projector = frozen_bvc_pca
    if include_bvc:
        if any(by_id[key].bvc_activity is None for key in keys):
            raise ValueError("BVC-conditioned analysis requires BVC activity everywhere")
        if bvc_projector is None:
            bvc_training = np.concatenate(
                [by_id[key].bvc_activity for key in training], axis=0  # type: ignore[list-item]
            )
            bvc_projector = PCAProjector.fit(bvc_training, int(bvc_components))

    designs: dict[str, BehaviorDesign] = {}
    for key in keys:
        bvc_scores = None
        if include_bvc:
            assert bvc_projector is not None
            assert by_id[key].bvc_activity is not None
            bvc_scores = bvc_projector.transform(
                by_id[key].bvc_activity, int(bvc_components)
            )
        designs[key] = build_behavior_design(by_id[key], spec, bvc_scores=bvc_scores)
    feature_names = designs[keys[0]].feature_names
    if any(designs[key].feature_names != feature_names for key in keys[1:]):
        raise RuntimeError("Behavior feature definitions differ across trajectories")

    def residualize(responses: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        design_train = np.concatenate(
            [designs[key].values for key in training], axis=0
        )
        response_train = np.concatenate(
            [np.asarray(responses[key], dtype=np.float64) for key in training],
            axis=0,
        )
        model = RidgeResidualizer.fit(design_train, response_train, ridge_alpha)
        return {
            key: model.residualize(designs[key].values, responses[key])
            for key in keys
        }

    place = residualize({key: scores[key].place for key in keys})
    grid = {
        module: residualize({key: scores[key].grid[module] for key in keys})
        for module in sorted(modules)
    }
    return CrossFittedPopulationResiduals(
        place=place,
        grid=grid,
        feature_names=feature_names,
        include_bvc=bool(include_bvc),
        fold_metadata={
            key: {
                "training_trajectory_ids": list(training),
                "held_out": key in held_out,
                "nuisance_fit_scope": "outer_training_fold_only",
            }
            for key in keys
        },
    )


def cross_fit_population_residuals(
    trajectories: Sequence[TrajectoryData],
    scores: Mapping[str, PopulationScores],
    *,
    spec: BehaviorDesignSpec,
    ridge_alpha: float,
    include_bvc: bool = False,
    bvc_components: int = 10,
    frozen_bvc_pca: PCAProjector | None = None,
) -> CrossFittedPopulationResiduals:
    """Cross-fit nuisance models for place and every grid population.

    All trajectories must describe the same network/condition/session and have
    unique trajectory IDs.  If a baseline-fitted BVC PCA is supplied it remains
    frozen.  Otherwise BVC PCA is fitted inside each training fold, never on the
    held-out trajectory.
    """

    if len(trajectories) < 2:
        raise ValueError("At least two independent trajectories are required")
    by_id = {str(item.trajectory_id): item for item in trajectories}
    if len(by_id) != len(trajectories):
        raise ValueError("trajectory_id values must be unique within a cross-fit group")
    keys = sorted(by_id)
    if set(scores) != set(keys):
        raise ValueError("PCA score keys must match trajectory_id values")
    group_identity = {
        (str(item.network_id), str(item.stage), str(item.environment), str(item.session))
        for item in trajectories
    }
    if len(group_identity) != 1:
        raise ValueError(
            "Cross-fitting must not mix networks, stages, environments, or sessions"
        )
    modules = set(next(iter(scores.values())).grid)
    if any(set(item.grid) != modules for item in scores.values()):
        raise ValueError("Grid module names differ across PCA score objects")

    place_residuals: dict[str, np.ndarray] = {}
    grid_residuals: dict[str, dict[str, np.ndarray]] = {
        module: {} for module in sorted(modules)
    }
    fold_metadata: dict[str, dict[str, Any]] = {}
    common_feature_names: tuple[str, ...] | None = None

    for held_out in keys:
        training = [key for key in keys if key != held_out]
        bvc_projector = frozen_bvc_pca
        if include_bvc:
            if any(by_id[key].bvc_activity is None for key in keys):
                raise ValueError("BVC-conditioned analysis requires BVC activity everywhere")
            if bvc_projector is None:
                bvc_training = np.concatenate(
                    [by_id[key].bvc_activity for key in training], axis=0  # type: ignore[list-item]
                )
                bvc_projector = PCAProjector.fit(bvc_training, int(bvc_components))
        fold_designs: dict[str, BehaviorDesign] = {}
        for key in keys:
            bvc_scores = None
            if include_bvc:
                assert bvc_projector is not None
                assert by_id[key].bvc_activity is not None
                bvc_scores = bvc_projector.transform(
                    by_id[key].bvc_activity, int(bvc_components)
                )
            fold_designs[key] = build_behavior_design(
                by_id[key], spec, bvc_scores=bvc_scores
            )
        names = fold_designs[held_out].feature_names
        if any(design.feature_names != names for design in fold_designs.values()):
            raise RuntimeError("Behavior feature definitions differ across trajectories")
        if common_feature_names is None:
            common_feature_names = names
        elif common_feature_names != names:
            raise RuntimeError("Behavior feature definitions differ across folds")
        x_train = np.concatenate([fold_designs[key].values for key in training], axis=0)
        x_test = fold_designs[held_out].values

        place_model = RidgeResidualizer.fit(
            x_train,
            np.concatenate([scores[key].place for key in training], axis=0),
            ridge_alpha,
        )
        place_residuals[held_out] = place_model.residualize(
            x_test, scores[held_out].place
        )
        for module in sorted(modules):
            grid_model = RidgeResidualizer.fit(
                x_train,
                np.concatenate([scores[key].grid[module] for key in training], axis=0),
                ridge_alpha,
            )
            grid_residuals[module][held_out] = grid_model.residualize(
                x_test, scores[held_out].grid[module]
            )
        fold_metadata[held_out] = {
            "held_out_trajectory": held_out,
            "training_trajectories": training,
            "n_training_samples": int(x_train.shape[0]),
            "n_test_samples": int(x_test.shape[0]),
            "n_behavior_features": int(x_train.shape[1]),
            "bvc_pca_policy": (
                "frozen_baseline"
                if include_bvc and frozen_bvc_pca is not None
                else "training_fold"
                if include_bvc
                else "not_included"
            ),
        }
    assert common_feature_names is not None
    return CrossFittedPopulationResiduals(
        place=place_residuals,
        grid=grid_residuals,
        feature_names=common_feature_names,
        include_bvc=include_bvc,
        fold_metadata=fold_metadata,
    )
