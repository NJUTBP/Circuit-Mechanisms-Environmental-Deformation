"""Population Gaussian-copula and cross-validated predictive TE estimators."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Hashable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .copula import (
    GaussianCopulaTransformer,
    Shrinkage,
    gaussian_conditional_mutual_information,
    regularized_covariance,
    stable_logdet,
)
from .types import LaggedSamples


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GCMITEResult:
    """Result of an in-sample Gaussian-copula conditional TE estimate."""

    value_bits: float
    n_samples: int
    effective_n: float
    source_dimension: int
    target_dimension: int
    condition_dimension: int
    gaussianized: bool
    shrinkage: Shrinkage


@dataclass(frozen=True, slots=True)
class PredictiveTEFold:
    """Held-out log-predictive improvement for one trajectory fold."""

    held_out_group: Hashable
    value_bits_per_sample: float
    n_train: int
    n_test: int
    test_weight: float


@dataclass(frozen=True, slots=True)
class CrossValidatedTEResult:
    """Trajectory-fold cross-validated Gaussian predictive TE result.

    Values may be negative: adding source history can reduce held-out predictive
    log likelihood.  This is useful evidence against over-interpreting a small,
    positive in-sample conditional mutual information.
    """

    value_bits_per_sample: float
    folds: tuple[PredictiveTEFold, ...]
    ridge: float
    gaussianized: bool
    shrinkage: Shrinkage


@dataclass(frozen=True, slots=True)
class PredictiveTEInputFold:
    """Fold-specific samples after training-only preprocessing."""

    held_out_group: Hashable
    training_samples: LaggedSamples
    test_samples: LaggedSamples


def estimate_gcmi_te(
    samples: LaggedSamples,
    *,
    gaussianize: bool = True,
    shrinkage: Shrinkage = "oas",
) -> GCMITEResult:
    """Estimate population TE as Gaussian(-copula) conditional MI in bits."""

    _require_finite(samples)
    value = gaussian_conditional_mutual_information(
        samples.source_past,
        samples.target_future,
        samples.conditioning_state,
        sample_weight=samples.sample_weight,
        gaussianize=gaussianize,
        shrinkage=shrinkage,
        base=2.0,
    )
    return GCMITEResult(
        value_bits=value,
        n_samples=samples.n_samples,
        effective_n=samples.effective_sample_size,
        source_dimension=samples.source_past.shape[1],
        target_dimension=samples.target_future.shape[1],
        condition_dimension=samples.conditioning_state.shape[1],
        gaussianized=gaussianize,
        shrinkage=shrinkage,
    )


def estimate_gaussian_copula_te(
    samples: LaggedSamples,
    *,
    gaussianize: bool = True,
    shrinkage: Shrinkage = "oas",
) -> GCMITEResult:
    """Descriptive public alias for :func:`estimate_gcmi_te`.

    Keeping the long name at the package boundary makes it clear that this is
    the Gaussian-copula estimator, not a histogram or k-nearest-neighbour TE.
    """

    return estimate_gcmi_te(
        samples,
        gaussianize=gaussianize,
        shrinkage=shrinkage,
    )


def estimate_cross_validated_predictive_te(
    samples: LaggedSamples,
    *,
    groups: ArrayLike | None = None,
    ridge: float = 1e-3,
    gaussianize: bool = True,
    shrinkage: Shrinkage = "oas",
    min_train_samples: int = 10,
    min_test_samples: int = 2,
) -> CrossValidatedTEResult:
    """Estimate held-out Gaussian predictive TE with leave-one-group-out folds.

    The restricted model predicts ``target_future`` from ``target_past`` and
    conditioning variables.  The full model additionally receives
    ``source_past``.  Each model and residual covariance is fitted on all but
    one trajectory, and its Gaussian log likelihood is evaluated on the held-
    out trajectory.  If ``groups`` is omitted, ``samples.trajectory_id`` is
    used.  Copula transforms, when requested, are also fitted on training rows
    only.
    """

    _require_finite(samples)
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if min_train_samples < 2 or min_test_samples < 1:
        raise ValueError("invalid minimum fold sample sizes")

    group_array = (
        np.asarray(samples.trajectory_id, dtype=object).reshape(-1)
        if groups is None
        else np.asarray(groups, dtype=object).reshape(-1)
    )
    if group_array.shape[0] != samples.n_samples:
        raise ValueError("groups must contain one label per sample")
    unique_groups = _unique_preserving_order(group_array)
    if len(unique_groups) < 2:
        raise ValueError("trajectory-fold TE requires at least two distinct groups")

    fold_results: list[PredictiveTEFold] = []
    weighted_sum = 0.0
    total_test_weight = 0.0
    for held_out in unique_groups:
        test_mask = np.fromiter(
            (label == held_out for label in group_array),
            dtype=bool,
            count=group_array.size,
        )
        train_mask = ~test_mask
        n_train = int(np.sum(train_mask))
        n_test = int(np.sum(test_mask))
        if n_train < min_train_samples or n_test < min_test_samples:
            raise ValueError(
                f"fold {held_out!r} has {n_train} train and {n_test} test rows; "
                "increase data or lower explicit minimums"
            )

        train_weight = samples.sample_weight[train_mask]
        test_weight = samples.sample_weight[test_mask]
        train_arrays, test_arrays = _fold_arrays(
            samples,
            train_mask,
            test_mask,
            gaussianize=gaussianize,
            train_weight=train_weight,
        )
        y_train, source_train, condition_train = train_arrays
        y_test, source_test, condition_test = test_arrays

        restricted = _fit_gaussian_regression(
            condition_train,
            y_train,
            train_weight,
            ridge=ridge,
            shrinkage=shrinkage,
        )
        full = _fit_gaussian_regression(
            np.concatenate((condition_train, source_train), axis=1),
            y_train,
            train_weight,
            ridge=ridge,
            shrinkage=shrinkage,
        )
        restricted_log_likelihood = _gaussian_log_likelihood(
            y_test,
            restricted.predict(condition_test),
            restricted.residual_covariance,
        )
        full_log_likelihood = _gaussian_log_likelihood(
            y_test,
            full.predict(np.concatenate((condition_test, source_test), axis=1)),
            full.residual_covariance,
        )
        improvement_bits = (
            full_log_likelihood - restricted_log_likelihood
        ) / np.log(2.0)
        fold_weight = float(np.sum(test_weight))
        fold_value = float(np.dot(test_weight, improvement_bits) / fold_weight)
        fold_results.append(
            PredictiveTEFold(
                held_out_group=held_out,
                value_bits_per_sample=fold_value,
                n_train=n_train,
                n_test=n_test,
                test_weight=fold_weight,
            )
        )
        weighted_sum += fold_weight * fold_value
        total_test_weight += fold_weight

    return CrossValidatedTEResult(
        value_bits_per_sample=weighted_sum / total_test_weight,
        folds=tuple(fold_results),
        ridge=ridge,
        gaussianized=gaussianize,
        shrinkage=shrinkage,
    )


def estimate_cross_validated_predictive_te_from_folds(
    folds: Sequence[PredictiveTEInputFold],
    *,
    ridge: float = 1e-3,
    gaussianize: bool = True,
    shrinkage: Shrinkage = "oas",
    min_train_samples: int = 10,
    min_test_samples: int = 2,
) -> CrossValidatedTEResult:
    """Evaluate predictive TE after fold-specific training-only preprocessing."""

    fold_inputs = tuple(folds)
    if len(fold_inputs) < 2:
        raise ValueError("at least two predictive folds are required")
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if min_train_samples < 2 or min_test_samples < 1:
        raise ValueError("invalid minimum fold sample sizes")

    fold_results: list[PredictiveTEFold] = []
    weighted_sum = 0.0
    total_test = 0.0

    for item in fold_inputs:
        train = item.training_samples
        test = item.test_samples
        _require_finite(train)
        _require_finite(test)
        if train.n_samples < min_train_samples or test.n_samples < min_test_samples:
            raise ValueError(
                f"fold {item.held_out_group!r} has {train.n_samples} train and "
                f"{test.n_samples} test rows"
            )
        train_dimensions = (
            train.target_future.shape[1],
            train.source_past.shape[1],
            train.conditioning_state.shape[1],
        )
        test_dimensions = (
            test.target_future.shape[1],
            test.source_past.shape[1],
            test.conditioning_state.shape[1],
        )
        if train_dimensions != test_dimensions:
            raise ValueError(
                f"fold {item.held_out_group!r} training/test dimensions differ"
            )
        train_arrays, test_arrays = _fold_sample_arrays(
            train,
            test,
            gaussianize=gaussianize,
        )
        y_train, source_train, condition_train = train_arrays
        y_test, source_test, condition_test = test_arrays

        restricted = _fit_gaussian_regression(
            condition_train,
            y_train,
            train.sample_weight,
            ridge=ridge,
            shrinkage=shrinkage,
        )
        full = _fit_gaussian_regression(
            np.concatenate((condition_train, source_train), axis=1),
            y_train,
            train.sample_weight,
            ridge=ridge,
            shrinkage=shrinkage,
        )
        restricted_log_likelihood = _gaussian_log_likelihood(
            y_test,
            restricted.predict(condition_test),
            restricted.residual_covariance,
        )
        full_log_likelihood = _gaussian_log_likelihood(
            y_test,
            full.predict(np.concatenate((condition_test, source_test), axis=1)),
            full.residual_covariance,
        )
        improvement_bits = (
            full_log_likelihood - restricted_log_likelihood
        ) / np.log(2.0)
        fold_weight = float(np.sum(test.sample_weight))
        fold_value = float(
            np.dot(test.sample_weight, improvement_bits) / fold_weight
        )
        fold_results.append(
            PredictiveTEFold(
                held_out_group=item.held_out_group,
                value_bits_per_sample=fold_value,
                n_train=train.n_samples,
                n_test=test.n_samples,
                test_weight=fold_weight,
            )
        )
        weighted_sum += fold_weight * fold_value
        total_test += fold_weight

    return CrossValidatedTEResult(
        value_bits_per_sample=weighted_sum / total_test,
        folds=tuple(fold_results),
        ridge=ridge,
        gaussianized=gaussianize,
        shrinkage=shrinkage,
    )


@dataclass(frozen=True, slots=True)
class _GaussianRegression:
    intercept: FloatArray
    coefficient: FloatArray
    residual_covariance: FloatArray

    def predict(self, design: FloatArray) -> FloatArray:
        return self.intercept + design @ self.coefficient


def _fit_gaussian_regression(
    design: FloatArray,
    response: FloatArray,
    sample_weight: FloatArray,
    *,
    ridge: float,
    shrinkage: Shrinkage,
) -> _GaussianRegression:
    normalized_weight = sample_weight / np.sum(sample_weight)
    x_mean = np.sum(design * normalized_weight[:, np.newaxis], axis=0)
    y_mean = np.sum(response * normalized_weight[:, np.newaxis], axis=0)
    x_centered = design - x_mean
    y_centered = response - y_mean

    if design.shape[1] == 0:
        coefficient = np.empty((0, response.shape[1]), dtype=np.float64)
    else:
        gram = x_centered.T @ (x_centered * normalized_weight[:, np.newaxis])
        cross = x_centered.T @ (y_centered * normalized_weight[:, np.newaxis])
        penalty_scale = max(float(np.trace(gram)) / design.shape[1], 1e-12)
        penalized = gram + ridge * penalty_scale * np.eye(design.shape[1])
        try:
            coefficient = np.linalg.solve(penalized, cross)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.pinv(penalized, rcond=1e-12) @ cross

    intercept = y_mean - x_mean @ coefficient
    residual = response - (intercept + design @ coefficient)
    covariance, _ = regularized_covariance(
        residual,
        sample_weight=sample_weight,
        shrinkage=shrinkage,
    )
    return _GaussianRegression(intercept, coefficient, covariance)


def _gaussian_log_likelihood(
    observed: FloatArray,
    predicted_mean: FloatArray,
    covariance: FloatArray,
) -> FloatArray:
    residual = observed - predicted_mean
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    if np.any(eigenvalues <= 0.0):
        raise np.linalg.LinAlgError("residual covariance is not positive definite")
    rotated = residual @ eigenvectors
    quadratic = np.sum((rotated * rotated) / eigenvalues, axis=1)
    dimension = observed.shape[1]
    return -0.5 * (
        dimension * np.log(2.0 * np.pi) + stable_logdet(covariance) + quadratic
    )


def _fold_arrays(
    samples: LaggedSamples,
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
    *,
    gaussianize: bool,
    train_weight: FloatArray,
) -> tuple[tuple[FloatArray, FloatArray, FloatArray], tuple[FloatArray, FloatArray, FloatArray]]:
    blocks = (
        samples.target_future,
        samples.source_past,
        samples.conditioning_state,
    )
    train_blocks: list[FloatArray] = []
    test_blocks: list[FloatArray] = []
    for block in blocks:
        train = block[train_mask]
        test = block[test_mask]
        if gaussianize and block.shape[1] > 0:
            transformer = GaussianCopulaTransformer.fit(
                train,
                sample_weight=train_weight,
            )
            train = transformer.transform(train)
            test = transformer.transform(test)
        train_blocks.append(train)
        test_blocks.append(test)
    return (
        (train_blocks[0], train_blocks[1], train_blocks[2]),
        (test_blocks[0], test_blocks[1], test_blocks[2]),
    )


def _fold_sample_arrays(
    train_samples: LaggedSamples,
    test_samples: LaggedSamples,
    *,
    gaussianize: bool,
) -> tuple[
    tuple[FloatArray, FloatArray, FloatArray],
    tuple[FloatArray, FloatArray, FloatArray],
]:
    train_source = (
        train_samples.target_future,
        train_samples.source_past,
        train_samples.conditioning_state,
    )
    test_source = (
        test_samples.target_future,
        test_samples.source_past,
        test_samples.conditioning_state,
    )
    train_blocks: list[FloatArray] = []
    test_blocks: list[FloatArray] = []
    for train, test in zip(train_source, test_source):
        if gaussianize and train.shape[1] > 0:
            transformer = GaussianCopulaTransformer.fit(
                train,
                sample_weight=train_samples.sample_weight,
            )
            train = transformer.transform(train)
            test = transformer.transform(test)
        train_blocks.append(train)
        test_blocks.append(test)
    return (
        (train_blocks[0], train_blocks[1], train_blocks[2]),
        (test_blocks[0], test_blocks[1], test_blocks[2]),
    )


def _unique_preserving_order(values: NDArray[np.object_]) -> tuple[Hashable, ...]:
    result: list[Hashable] = []
    for value in values:
        if not any(value == existing for existing in result):
            result.append(value)
    return tuple(result)


def _require_finite(samples: LaggedSamples) -> None:
    for name, value in (
        ("target_future", samples.target_future),
        ("source_past", samples.source_past),
        ("target_past", samples.target_past),
        ("conditioning", samples.conditioning),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} contains non-finite values; build samples with "
                "drop_nonfinite=True or clean the data first"
            )
