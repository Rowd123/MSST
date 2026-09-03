"""Shared implementation helpers for the SST transforms."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_signal(x: ArrayLike) -> NDArray[np.complex128]:
    """Return *x* as a non-empty, one-dimensional complex signal."""

    signal = np.asarray(x)
    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal[:, 0]
    elif signal.ndim != 1:
        raise ValueError("x must be a one-dimensional signal or column vector")
    if signal.size == 0:
        raise ValueError("x must not be empty")
    return signal.astype(np.complex128, copy=False)


def positive_integer(value: object, name: str) -> int:
    """Validate and return a positive integer argument."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def gaussian_window(length: int) -> tuple[NDArray[np.float64], int]:
    """Build the odd-length Gaussian window used by both transforms."""

    length = length if length % 2 else length + 1
    grid = np.linspace(-0.5, 0.5, length)
    return np.exp(-np.pi / 0.32**2 * grid**2), length


def windowed_signal(
    signal: NDArray[np.complex128], window: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Build the windowed signal matrix consumed by the column-wise FFT."""

    n = signal.size
    n_frequencies = (n + 1) // 2
    half_window = window.size // 2
    result = np.zeros((n, n), dtype=np.complex128)

    for time in range(n):
        tau_min = -min(n_frequencies - 1, half_window, time)
        tau_max = min(n_frequencies - 1, half_window, n - time - 1)
        tau = np.arange(tau_min, tau_max + 1)
        result[np.mod(tau, n), time] = (
            signal[time + tau] * window[half_window + tau]
        )

    return result
