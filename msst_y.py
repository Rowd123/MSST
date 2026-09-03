"""Multi-synchrosqueezing transform implemented with NumPy."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from _transform_utils import (
    as_signal,
    gaussian_window,
    positive_integer,
    windowed_signal,
)


def _reassign(
    transform: NDArray[np.complex128], omega: NDArray[np.int64]
) -> NDArray[np.complex128]:
    """Reassign a time-frequency matrix to zero-based frequency bins."""

    result = np.zeros_like(transform)
    n_frequencies, n_times = transform.shape
    valid = (omega >= 0) & (omega < n_frequencies)
    times = np.broadcast_to(np.arange(n_times), omega.shape)
    np.add.at(result, (omega[valid], times[valid]), transform[valid])
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
        Positive window length. An even length is increased by one.
    num:
        Positive number of synchrosqueezing iterations.

    Returns
    -------
    tuple of numpy.ndarray
        ``(Ts, tfr1)``, where ``Ts`` is the normalized MSST and ``tfr1`` is
        the original STFT. Both have shape ``(round(n / 2), n)``.
    """

    signal = as_signal(x)
    if signal.size < 2:
        raise ValueError("x must contain at least two samples")
    hlength = positive_integer(hlength, "hlength")
    num = positive_integer(num, "num")

    n = signal.size
    window, _ = gaussian_window(hlength)
    n_frequencies = (n + 1) // 2
    tfr = windowed_signal(signal, window)

    tfr1 = np.fft.fft(tfr, axis=0)[:n_frequencies]
    phase = np.unwrap(np.angle(tfr1), axis=1)
    omega_values = np.diff(phase, axis=1) * n / (2 * np.pi)
    omega_values = np.concatenate((omega_values, omega_values[:, -1:]), axis=1)
    omega = np.rint(omega_values).astype(np.int64)

    transform = tfr1
    for _ in range(num):
        transform = _reassign(transform, omega)

    return transform / (n / 2), tfr1
