"""Gaussian-copula transforms and numerically stable Gaussian CMI.

Only NumPy is required.  The inverse-normal approximation is Peter J.
Acklam's rational approximation, implemented here to avoid a SciPy runtime
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Shrinkage = Literal["oas"] | float | None


def _as_2d_finite(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _normal_ppf(probability: ArrayLike) -> FloatArray:
    """Approximate the standard-normal quantile for probabilities in (0, 1)."""

    p = np.asarray(probability, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0) | ~np.isfinite(p)):
        raise ValueError("normal quantiles require finite probabilities in (0, 1)")

    # Coefficients for Acklam's inverse-normal approximation.
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    result = np.empty_like(p)
    low = p < 0.02425
    high = p > 1.0 - 0.02425
    central = ~(low | high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        result[low] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(high):
        q = np.sqrt(-2.0 * np.log1p(-p[high]))
        result[high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(central):
        q = p[central] - 0.5
        r = q * q
        result[central] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return result


def rank_gaussianize(
    value: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> FloatArray:
    """Map each column to a standard-normal marginal using average ranks.

    Stable sorting gives deterministic handling of tied observations.  A
    constant column maps to zero and therefore contributes no information.
    If weights are supplied, ties use the midpoint of their cumulative
    probability mass; this keeps equal-segment weighting coherent with the
    downstream covariance estimate.
    """

    array = _as_2d_finite(value, "value")
    n_rows, n_columns = array.shape
    if n_rows < 2:
        raise ValueError("at least two rows are required for rank Gaussianization")
    if sample_weight is None:
        weights = np.ones(n_rows, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_rows:
            raise ValueError("sample_weight length does not match the data")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("sample weights must be finite and strictly positive")
    transformed = np.empty_like(array)

    for column_index in range(n_columns):
        column = array[:, column_index]
        order = np.argsort(column, kind="mergesort")
        sorted_column = column[order]
        sorted_weight = weights[order]
        cumulative_weight = np.concatenate(
            (np.array([0.0]), np.cumsum(sorted_weight, dtype=np.float64))
        )
        total_weight = cumulative_weight[-1]
        boundaries = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.flatnonzero(np.diff(sorted_column) != 0.0) + 1,
                np.array([n_rows], dtype=np.int64),
            )
        )
        probabilities = np.empty(n_rows, dtype=np.float64)
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            probability = (
                cumulative_weight[start] + cumulative_weight[stop]
            ) / (2.0 * total_weight)
            probabilities[order[start:stop]] = probability
        transformed[:, column_index] = _normal_ppf(probabilities)
    return transformed


@dataclass(frozen=True, slots=True)
class GaussianCopulaTransformer:
    """Empirical marginal transform that can be fitted on training data only."""

    sorted_training_columns: tuple[FloatArray, ...]
    cumulative_training_weights: tuple[FloatArray, ...]

    @classmethod
    def fit(
        cls,
        value: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "GaussianCopulaTransformer":
        array = _as_2d_finite(value, "value")
        if array.shape[0] < 2:
            raise ValueError("at least two training rows are required")
        if sample_weight is None:
            weights = np.ones(array.shape[0], dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if weights.shape[0] != array.shape[0]:
                raise ValueError("sample_weight length does not match the data")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("sample weights must be finite and strictly positive")

        columns: list[FloatArray] = []
        cumulative_weights: list[FloatArray] = []
        for index in range(array.shape[1]):
            order = np.argsort(array[:, index], kind="mergesort")
            columns.append(array[order, index])
            cumulative_weights.append(
                np.concatenate(
                    (
                        np.array([0.0]),
                        np.cumsum(weights[order], dtype=np.float64),
                    )
                )
            )
        return cls(tuple(columns), tuple(cumulative_weights))

    @property
    def n_features(self) -> int:
        return len(self.sorted_training_columns)

    def transform(self, value: ArrayLike) -> FloatArray:
        array = _as_2d_finite(value, "value")
        if array.shape[1] != self.n_features:
            raise ValueError(
                f"value has {array.shape[1]} columns; expected {self.n_features}"
            )
        transformed = np.empty_like(array)
        for index, (training, cumulative_weight) in enumerate(
            zip(self.sorted_training_columns, self.cumulative_training_weights)
        ):
            left = np.searchsorted(training, array[:, index], side="left")
            right = np.searchsorted(training, array[:, index], side="right")
            total_weight = cumulative_weight[-1]
            probabilities = (
                cumulative_weight[left] + cumulative_weight[right]
            ) / (2.0 * total_weight)
            # Values outside the training range receive the most extreme
            # finite training plotting positions.
            edge = 0.5 * np.min(np.diff(cumulative_weight)) / total_weight
            probabilities = np.clip(probabilities, edge, 1.0 - edge)
            transformed[:, index] = _normal_ppf(probabilities)
        return transformed

    def fit_transform(self, value: ArrayLike) -> FloatArray:
        """Transform data; equivalent to ``transform`` on the fitted sample."""

        return self.transform(value)


def effective_sample_size(sample_weight: ArrayLike | None, n_rows: int) -> float:
    if sample_weight is None:
        return float(n_rows)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n_rows:
        raise ValueError("sample_weight length does not match the data")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample weights must be finite and strictly positive")
    total = float(np.sum(weights))
    return total * total / float(np.dot(weights, weights))


def regularized_covariance(
    value: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    shrinkage: Shrinkage = "oas",
    eigenvalue_floor: float = 1e-10,
) -> tuple[FloatArray, float]:
    """Estimate a positive-definite covariance matrix.

    ``shrinkage='oas'`` uses the Oracle Approximating Shrinkage formula with
    Kish effective sample size for weighted observations.  A numeric value in
    ``[0, 1]`` mixes the empirical covariance with its spherical target.
    """

    array = _as_2d_finite(value, "value")
    n_rows, n_features = array.shape
    if n_rows < 2:
        raise ValueError("at least two observations are required")
    if n_features == 0:
        return np.empty((0, 0), dtype=np.float64), 0.0

    if sample_weight is None:
        normalized_weight = np.full(n_rows, 1.0 / n_rows)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_rows:
            raise ValueError("sample_weight length does not match the data")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("sample weights must be finite and strictly positive")
        normalized_weight = weights / np.sum(weights)

    mean = np.sum(array * normalized_weight[:, np.newaxis], axis=0)
    centered = array - mean
    empirical = centered.T @ (centered * normalized_weight[:, np.newaxis])
    empirical = (empirical + empirical.T) * 0.5
    mean_variance = float(np.trace(empirical)) / n_features

    if shrinkage == "oas":
        n_effective = effective_sample_size(sample_weight, n_rows)
        alpha = float(np.mean(empirical * empirical))
        denominator = (n_effective + 1.0) * (
            alpha - (mean_variance * mean_variance / n_features)
        )
        if denominator <= np.finfo(np.float64).eps:
            coefficient = 1.0
        else:
            coefficient = min(
                (alpha + mean_variance * mean_variance) / denominator,
                1.0,
            )
    elif shrinkage is None:
        coefficient = 0.0
    else:
        coefficient = float(shrinkage)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("numeric shrinkage must lie in [0, 1]")

    covariance = (1.0 - coefficient) * empirical
    covariance += coefficient * mean_variance * np.eye(n_features)

    # Guarantee strict positive definiteness even for constant features.
    scale = max(mean_variance, 1.0)
    minimum = max(float(eigenvalue_floor) * scale, np.finfo(np.float64).eps)
    smallest = float(np.min(np.linalg.eigvalsh(covariance)))
    if smallest < minimum:
        covariance += (minimum - smallest) * np.eye(n_features)
    return covariance, coefficient


def stable_logdet(matrix: ArrayLike) -> float:
    """Return the log determinant of a symmetric positive-definite matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if array.shape[0] == 0:
        return 0.0
    eigenvalues = np.linalg.eigvalsh((array + array.T) * 0.5)
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise np.linalg.LinAlgError("matrix is not positive definite")
    return float(np.sum(np.log(eigenvalues)))


def gaussian_conditional_mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    condition: ArrayLike | None = None,
    *,
    sample_weight: ArrayLike | None = None,
    gaussianize: bool = True,
    shrinkage: Shrinkage = "oas",
    base: float = 2.0,
) -> float:
    """Estimate ``I(X; Y | condition)`` under a Gaussian copula model.

    The covariance of the complete joint vector is regularized once; all
    determinant terms are principal submatrices of that same covariance.  The
    result is therefore coherent even when the empirical covariance is rank
    deficient.
    """

    x_array = _as_2d_finite(x, "x")
    y_array = _as_2d_finite(y, "y")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must contain the same number of rows")
    if condition is None:
        z_array = np.empty((x_array.shape[0], 0), dtype=np.float64)
    else:
        z_array = _as_2d_finite(condition, "condition")
        if z_array.shape[0] != x_array.shape[0]:
            raise ValueError("condition must contain the same number of rows")
    if x_array.shape[1] == 0 or y_array.shape[1] == 0:
        raise ValueError("x and y must each contain at least one feature")
    if base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError("logarithm base must be positive and different from one")

    joint = np.concatenate((x_array, y_array, z_array), axis=1)
    if gaussianize:
        joint = rank_gaussianize(joint, sample_weight=sample_weight)
    covariance, _ = regularized_covariance(
        joint,
        sample_weight=sample_weight,
        shrinkage=shrinkage,
    )

    x_stop = x_array.shape[1]
    y_stop = x_stop + y_array.shape[1]
    z_stop = y_stop + z_array.shape[1]
    x_indices = np.arange(0, x_stop)
    y_indices = np.arange(x_stop, y_stop)
    z_indices = np.arange(y_stop, z_stop)

    def submatrix(indices: Sequence[int] | NDArray[np.int64]) -> FloatArray:
        index = np.asarray(indices, dtype=np.int64)
        return covariance[np.ix_(index, index)]

    if z_array.shape[1] == 0:
        log_ratio = (
            stable_logdet(submatrix(x_indices))
            + stable_logdet(submatrix(y_indices))
            - stable_logdet(covariance)
        )
    else:
        xz = np.concatenate((x_indices, z_indices))
        yz = np.concatenate((y_indices, z_indices))
        log_ratio = (
            stable_logdet(submatrix(xz))
            + stable_logdet(submatrix(yz))
            - stable_logdet(submatrix(z_indices))
            - stable_logdet(covariance)
        )

    value = 0.5 * log_ratio / np.log(base)
    # Any negative value here can only be floating-point noise because every
    # determinant comes from one positive-definite joint covariance.
    if value < 0.0 and value > -1e-10:
        return 0.0
    return float(value)
