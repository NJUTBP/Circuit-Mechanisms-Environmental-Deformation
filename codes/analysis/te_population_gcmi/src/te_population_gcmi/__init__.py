"""Cross-fitted behavior-adjusted population transfer-entropy analysis.

This package is intentionally independent of the legacy TE implementations in
the surrounding repository.  Its public API is kept small; the command-line
pipeline is the recommended entry point for complete analyses.
"""

from .estimator import (
    PredictiveTEInputFold,
    estimate_cross_validated_predictive_te,
    estimate_cross_validated_predictive_te_from_folds,
    estimate_gaussian_copula_te,
    estimate_gcmi_te,
)
from .lagged import build_lagged_samples
from .types import LaggedSamples, Segment

__all__ = [
    "LaggedSamples",
    "PredictiveTEInputFold",
    "Segment",
    "build_lagged_samples",
    "estimate_cross_validated_predictive_te",
    "estimate_cross_validated_predictive_te_from_folds",
    "estimate_gaussian_copula_te",
    "estimate_gcmi_te",
]
__version__ = "0.1.0"
