"""Synchrosqueezing transform implemented with NumPy."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from _transform_utils import (
    as_signal,
    gaussian_window,
    positive_integer,
    windowed_signal,
)


def sst_y(x: ArrayLike, hlength: int | None = None) -> NDArray[np.complex128]:
    """Compute the synchrosqueezing transform of a signal.

    Parameters
    ----------
    x:
        One-dimensional signal, or a two-dimensional column vector of shape
        ``(n, 1)``.
    hlength:
        Window length. An even value is increased by one. The default is
        ``round(n / 5)``.

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

    signal = as_signal(x)
    n = signal.size
    if hlength is None:
        hlength = max(1, round(n / 5))
    else:
        hlength = positive_integer(hlength, "hlength")

    window, hlength = gaussian_window(hlength)
    grid = np.linspace(-0.5, 0.5, hlength)
    window_derivative = -2 * np.pi / 0.32**2 * grid * window
    n_frequencies = (n + 1) // 2

    tfr1 = windowed_signal(signal, window)
    tfr2 = windowed_signal(signal, window_derivative)

    tfr1 = np.fft.fft(tfr1, axis=0)[:n_frequencies]
    tfr2 = np.fft.fft(tfr2, axis=0)[:n_frequencies]

    frequencies = np.arange(n_frequencies)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        omega_values = frequencies + np.real(
            (n / hlength) * 1j * tfr2 / (2 * np.pi * tfr1)
        )
    finite = np.isfinite(omega_values)
    omega = np.zeros(omega_values.shape, dtype=np.int64)
    omega[finite] = np.rint(omega_values[finite]).astype(np.int64)

    transform = np.zeros_like(tfr1)
    valid = (
        (np.abs(tfr1) > 1e-4)
        & finite
        & (omega >= 0)
        & (omega < n_frequencies)
    )
    times = np.broadcast_to(np.arange(n), omega.shape)
    np.add.at(transform, (omega[valid], times[valid]), tfr1[valid])

    return transform / (n / 2)
