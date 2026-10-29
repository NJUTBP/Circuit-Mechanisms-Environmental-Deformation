"""Numerical primitives for the adopted five-edge grid-rescaling estimator."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.fft import irfft, next_fast_len, rfft

FAMILIAR_SIZE_CM = 100.0
ARENA_HEIGHT_CM = 100.0
SEARCH_RANGE_CM = 10.0
SEARCH_STEP_CM = 2.5
MIN_VALID_PIXELS = 20


def _spatial_index(positions_cm: np.ndarray, width_cm: float, bin_cm: float):
    nx = int(round(float(width_cm) / float(bin_cm)))
    ny = int(round(ARENA_HEIGHT_CM / float(bin_cm)))
    bx = np.clip((positions_cm[:, 0] / bin_cm).astype(np.int64), 0, nx - 1)
    by = np.clip((positions_cm[:, 1] / bin_cm).astype(np.int64), 0, ny - 1)
    flat = by * nx + bx
    occupancy = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
    return flat, occupancy, nx, ny


def _raw_rate_maps(activity: np.ndarray, flat: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    ny, nx = occupancy.shape
    visited = occupancy > 0
    maps = np.full((activity.shape[0], ny, nx), np.nan, dtype=np.float64)
    for cell in range(activity.shape[0]):
        weighted = np.bincount(
            flat, weights=np.asarray(activity[cell], dtype=np.float64), minlength=nx * ny
        ).reshape(ny, nx)
        np.divide(weighted, occupancy, out=maps[cell], where=visited)
    return maps


def _smooth_rate_maps(raw_maps: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return raw_maps.copy()
    valid = np.isfinite(raw_maps)
    numerator = ndimage.gaussian_filter(
        np.where(valid, raw_maps, 0.0),
        sigma=(0.0, sigma_bins, sigma_bins), mode="constant",
    )
    denominator = ndimage.gaussian_filter(
        valid.astype(np.float64),
        sigma=(0.0, sigma_bins, sigma_bins), mode="constant",
    )
    smoothed = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=smoothed, where=denominator > 1e-12)
    smoothed[~valid] = np.nan
    return smoothed


def _resize_x_batch(rate_maps: np.ndarray, target_size_cm: float, bin_cm: float) -> np.ndarray:
    target_bins = int(float(target_size_cm) / float(bin_cm))
    scale_x = target_bins / rate_maps.shape[2]
    if scale_x == 1.0:
        return rate_maps.copy()
    missing = ~np.isfinite(rate_maps)
    resized = ndimage.zoom(
        np.where(missing, 0.0, rate_maps), zoom=(1.0, 1.0, scale_x), order=1
    )
    if np.any(missing):
        resized_missing = ndimage.zoom(
            missing.astype(np.float64), zoom=(1.0, 1.0, scale_x), order=0
        ) > 0.5
        resized[resized_missing] = np.nan
    return resized


def _fft_cross_x(first: np.ndarray, second: np.ndarray, n_fft: int) -> np.ndarray:
    product = rfft(first, n=n_fft, axis=2) * rfft(second[:, :, ::-1], n=n_fft, axis=2)
    n_out = first.shape[2] + second.shape[2] - 1
    return irfft(product, n=n_fft, axis=2)[:, :, :n_out].sum(axis=1)


def correlations_all_x_lags(first: np.ndarray, second: np.ndarray):
    """Pearson correlations at every x lag, using pairwise finite pixels."""
    height = min(first.shape[1], second.shape[1])
    first = np.asarray(first[:, :height, :], dtype=np.float64)
    second = np.asarray(second[:, :height, :], dtype=np.float64)
    w1, w2 = first.shape[2], second.shape[2]
    n_out = w1 + w2 - 1
    n_fft = next_fast_len(n_out)
    valid1 = np.isfinite(first)
    valid2 = np.isfinite(second)
    mask1 = valid1.astype(np.float64)
    mask2 = valid2.astype(np.float64)
    value1 = np.where(valid1, first, 0.0)
    value2 = np.where(valid2, second, 0.0)
    count = np.rint(_fft_cross_x(mask1, mask2, n_fft))
    sum1 = _fft_cross_x(value1, mask2, n_fft)
    sum2 = _fft_cross_x(mask1, value2, n_fft)
    sum11 = _fft_cross_x(value1 * value1, mask2, n_fft)
    sum22 = _fft_cross_x(mask1, value2 * value2, n_fft)
    sum12 = _fft_cross_x(value1, value2, n_fft)
    safe_count = np.maximum(count, 1.0)
    covariance = sum12 - (sum1 * sum2 / safe_count)
    variance1 = sum11 - (sum1 * sum1 / safe_count)
    variance2 = sum22 - (sum2 * sum2 / safe_count)
    denominator = np.sqrt(np.maximum(variance1, 0.0) * np.maximum(variance2, 0.0))
    correlations = np.full_like(covariance, np.nan)
    usable = (count >= MIN_VALID_PIXELS) & (denominator > 0.0)
    np.divide(covariance, denominator, out=correlations, where=usable)
    correlations = np.clip(correlations, -1.0, 1.0)
    # Convolution index i corresponds to lag=(w2-1)-i.
    lags = (w2 - 1) - np.arange(n_out, dtype=int)
    overlaps = np.convolve(np.ones(w1, dtype=int), np.ones(w2, dtype=int))
    order = np.argsort(lags)
    return correlations[:, order], lags[order], overlaps[order].astype(int)


def _row_nanmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, -np.inf)
    result = np.max(safe, axis=1)
    result[~np.any(finite, axis=1)] = np.nan
    return result


def _candidate_scales(current_size_cm: float) -> np.ndarray:
    low = max(min(FAMILIAR_SIZE_CM, current_size_cm) - SEARCH_RANGE_CM, 20.0)
    high = max(FAMILIAR_SIZE_CM, current_size_cm) + SEARCH_RANGE_CM
    return np.arange(low, high + SEARCH_STEP_CM, SEARCH_STEP_CM, dtype=float)


def _update_best(candidate_corr: np.ndarray, scale_cm: float,
                 best_corr: np.ndarray, best_scale: np.ndarray) -> None:
    improve = np.isfinite(candidate_corr) & (candidate_corr > best_corr)
    best_corr[improve] = candidate_corr[improve]
    best_scale[improve] = scale_cm
