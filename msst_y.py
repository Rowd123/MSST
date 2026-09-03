"""Multi-synchrosqueezing transform translated from MATLAB ``MSST_Y``."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sst_y import _matlab_round


def _validate_signal(x: ArrayLike) -> NDArray[np.complex128]:
    signal = np.asarray(x)
    if signal.ndim == 2:
        if signal.shape[1] != 1:
            raise ValueError("x must be a one-dimensional signal or column vector")
        signal = signal[:, 0]
    elif signal.ndim != 1:
        raise ValueError("x must be a one-dimensional signal or column vector")
    if signal.size < 2:
        raise ValueError("x must contain at least two samples")
    return signal.astype(np.complex128, copy=False)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _reassign(
    transform: NDArray[np.complex128], omega: NDArray[np.int64]
) -> NDArray[np.complex128]:
    """Reassign a time-frequency matrix using MATLAB's one-based bins."""

    result = np.zeros_like(transform)
    n_frequencies, n_times = transform.shape
    for time in range(n_times):
        destinations = omega[:, time]
        valid = (destinations >= 1) & (destinations <= n_frequencies)
        np.add.at(result[:, time], destinations[valid] - 1, transform[valid, time])
    return result


def msst_y(
    x: ArrayLike, hlength: int, num: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compute the first-order multi-synchrosqueezing transform.

    Parameters
    ----------
    x:
        One-dimensional signal or column vector of shape ``(n, 1)``.
    hlength:
        Positive window length. An even length is increased by one, matching
        the MATLAB implementation.
    num:
        Positive number of synchrosqueezing iterations.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Ts, tfr1)``, where ``Ts`` is the normalized MSST and ``tfr1`` is
        the original STFT. Both have shape ``(round(n / 2), n)``.
    """

    signal = _validate_signal(x)
    hlength = _positive_integer(hlength, "hlength")
    num = _positive_integer(num, "num")

    n = signal.size
    hlength += 1 - hlength % 2
    ht = np.linspace(-0.5, 0.5, hlength)
    window = np.exp(-np.pi / 0.32**2 * ht**2)
    half_window = (hlength - 1) // 2
    n_frequencies = int(np.floor(n / 2 + 0.5))

    tfr = np.zeros((n, n), dtype=np.complex128)
    for time in range(n):
        tau_min = -min(n_frequencies - 1, half_window, time)
        tau_max = min(n_frequencies - 1, half_window, n - time - 1)
        tau = np.arange(tau_min, tau_max + 1)
        indices = np.mod(tau, n)
        tfr[indices, time] = signal[time + tau] * np.conj(
            window[half_window + tau]
        )

    tfr1 = np.fft.fft(tfr, axis=0)[:n_frequencies]
    phase = np.unwrap(np.angle(tfr1), axis=1)
    omega_values = np.diff(phase, axis=1) * n / (2 * np.pi)
    omega_values = np.concatenate((omega_values, omega_values[:, -1:]), axis=1)
    omega = _matlab_round(omega_values)

    transform = tfr1
    for _ in range(num):
        transform = _reassign(transform, omega)

    return transform / (n / 2), tfr1


# Preserve the original MATLAB function name for straightforward migration.
MSST_Y = msst_y
