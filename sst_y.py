"""Synchrosqueezing transform translated from the MATLAB ``SST_Y`` routine."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matlab_round(values: NDArray[np.floating]) -> NDArray[np.int64]:
    """Round halves away from zero, as MATLAB does."""

    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(
        np.int64
    )


def sst_y(x: ArrayLike, hlength: int | None = None) -> NDArray[np.complex128]:
    """Compute the synchrosqueezing transform of a signal.

    Parameters
    ----------
    x:
        One-dimensional signal, or a two-dimensional column vector of shape
        ``(n, 1)``.
    hlength:
        Window length.  As in the MATLAB implementation, an even value is
        increased by one.  The default is ``round(n / 5)``.

    Returns
    -------
    numpy.ndarray
        Complex SST matrix with shape ``(round(n / 2), n)``.  Rows are
        frequency bins and columns are time samples.

    Raises
    ------
    ValueError
        If the input is empty, is not a vector, or ``hlength`` is invalid.
    """

    signal = np.asarray(x)
    if signal.ndim == 2:
        if signal.shape[1] != 1:
            raise ValueError("x must be a one-dimensional signal or column vector")
        signal = signal[:, 0]
    elif signal.ndim != 1:
        raise ValueError("x must be a one-dimensional signal or column vector")

    n = signal.size
    if n == 0:
        raise ValueError("x must not be empty")

    if hlength is None:
        # n / 5 is non-negative, so this is MATLAB's round rather than
        # NumPy's ties-to-even rounding.
        hlength = max(1, int(np.floor(n / 5 + 0.5)))
    elif isinstance(hlength, (bool, np.bool_)) or not isinstance(
        hlength, (int, np.integer)
    ):
        raise ValueError("hlength must be a positive integer")
    if hlength < 1:
        raise ValueError("hlength must be a positive integer")

    # MATLAB: hlength = hlength + 1 - rem(hlength, 2)
    hlength += 1 - hlength % 2
    ht = np.linspace(-0.5, 0.5, hlength)
    window = np.exp(-np.pi / 0.32**2 * ht**2)
    window_derivative = -2 * np.pi / 0.32**2 * ht * window
    half_window = (hlength - 1) // 2
    n_frequencies = int(np.floor(n / 2 + 0.5))

    tfr1 = np.zeros((n, n), dtype=np.complex128)
    tfr2 = np.zeros_like(tfr1)
    signal = signal.astype(np.complex128, copy=False)

    for time in range(n):
        tau_min = -min(n_frequencies - 1, half_window, time)
        tau_max = min(n_frequencies - 1, half_window, n - time - 1)
        tau = np.arange(tau_min, tau_max + 1)
        indices = np.mod(tau, n)
        samples = signal[time + tau]
        tfr1[indices, time] = samples * np.conj(window[half_window + tau])
        tfr2[indices, time] = samples * np.conj(
            window_derivative[half_window + tau]
        )

    tfr1 = np.fft.fft(tfr1, axis=0)[:n_frequencies]
    tfr2 = np.fft.fft(tfr2, axis=0)[:n_frequencies]

    frequencies = np.arange(n_frequencies)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        omega_values = frequencies + np.real(
            (n / hlength) * 1j * tfr2 / (2 * np.pi * tfr1)
        )
    # MATLAB's conversion of NaN/Inf through round is harmless here because
    # those bins fail the bounds check during reassignment.
    finite = np.isfinite(omega_values)
    omega = np.zeros(omega_values.shape, dtype=np.int64)
    omega[finite] = _matlab_round(omega_values[finite])

    transform = np.zeros_like(tfr1)
    for time in range(n):
        active = np.abs(tfr1[:, time]) > 1e-4
        destinations = omega[:, time]
        # MATLAB uses one-based destination bins and accepts 1..n_frequencies.
        valid = active & finite[:, time] & (destinations >= 1) & (
            destinations <= n_frequencies
        )
        np.add.at(transform[:, time], destinations[valid] - 1, tfr1[valid, time])

    return transform / (n / 2)


# Keep the original MATLAB name available for callers porting existing code.
SST_Y = sst_y
