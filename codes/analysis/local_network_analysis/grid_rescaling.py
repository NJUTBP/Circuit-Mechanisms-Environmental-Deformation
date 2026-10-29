"""Reusable recommended grid-rescaling estimator.

Input maps must be occupancy-normalized 2.5-cm rate maps smoothed with a
normalized-mask Gaussian kernel (sigma=1.5 bins), with unvisited pixels stored
as NaN.  The function searches the Methods physical scale grid and restricts
translation matching to the first/last five complete-overlap positions.
"""

from __future__ import annotations

import numpy as np

import grid_rescale_primitives as core


def compute_recommended_rescale_batch(
    familiar_maps: np.ndarray,
    current_maps: np.ndarray,
    current_size_cm: float,
    *,
    edge_positions_per_side: int = 5,
    bin_cm: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-cell rescaling factors and best Pearson correlations.

    Parameters
    ----------
    familiar_maps, current_maps
        Arrays shaped ``(cells, y_bins, x_bins)``.  Corresponding cells must
        appear in the same order.
    current_size_cm
        Physical x size of the current environment (80 or 120 in v4).
    edge_positions_per_side
        Number of complete-overlap translations retained at each boundary.
        The adopted value is five.
    bin_cm
        Spatial bin size.  The recommended value is 2.5 cm.
    """
    familiar_maps = np.asarray(familiar_maps, dtype=float)
    current_maps = np.asarray(current_maps, dtype=float)
    if familiar_maps.ndim != 3 or current_maps.ndim != 3:
        raise ValueError("Maps must have shape (cells, y_bins, x_bins)")
    if familiar_maps.shape[0] != current_maps.shape[0]:
        raise ValueError("Familiar and current maps must contain the same cells")
    if edge_positions_per_side < 1:
        raise ValueError("edge_positions_per_side must be at least one")

    cells = familiar_maps.shape[0]
    best_corr = np.full(cells, -np.inf)
    best_scale = np.full(cells, core.FAMILIAR_SIZE_CM)
    for scale_cm in core._candidate_scales(float(current_size_cm)):
        scaled = core._resize_x_batch(familiar_maps, scale_cm, float(bin_cm))
        correlations, _, overlaps = core.correlations_all_x_lags(
            scaled, current_maps
        )
        max_overlap = min(scaled.shape[2], current_maps.shape[2])
        full = np.flatnonzero(overlaps == max_overlap)
        columns = np.unique(
            np.concatenate(
                (
                    full[:edge_positions_per_side],
                    full[-edge_positions_per_side:],
                )
            )
        )
        candidate = core._row_nanmax(correlations[:, columns])
        core._update_best(candidate, scale_cm, best_corr, best_scale)

    best_corr[~np.isfinite(best_corr)] = np.nan
    return best_scale / core.FAMILIAR_SIZE_CM, best_corr


__all__ = ["compute_recommended_rescale_batch"]
