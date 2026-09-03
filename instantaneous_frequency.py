"""Utilities for estimating instantaneous frequency from SST matrices."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ridge_frequency(
    transform: ArrayLike, sample_rate: float
) -> NDArray[np.float64]:
    """Estimate instantaneous frequency from the strongest spectral ridge.

    The estimate at each time sample is the frequency bin having the greatest
    magnitude, converted to hertz using ``sample_rate``.
    """

    matrix = np.asarray(transform)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("transform must be a non-empty two-dimensional matrix")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive finite number")

    signal_length = matrix.shape[1]
    ridge_bins = np.argmax(np.abs(matrix), axis=0)
    return ridge_bins.astype(np.float64) * float(sample_rate) / signal_length
