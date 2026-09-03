import numpy as np
import pytest

from msst_y import MSST_Y, msst_y


def test_returns_msst_and_original_stft():
    signal = np.sin(2 * np.pi * np.arange(20) / 5)

    transform, stft = msst_y(signal, hlength=4, num=2)

    assert transform.shape == (10, 20)
    assert stft.shape == (10, 20)
    assert np.iscomplexobj(transform)
    assert np.all(np.isfinite(transform))
    assert np.all(np.isfinite(stft))


def test_alias_accepts_a_column_vector():
    signal = np.arange(9, dtype=float)

    actual = MSST_Y(signal[:, None], 3, 1)
    expected = msst_y(signal, 3, 1)

    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_iterations_preserve_the_original_stft():
    signal = np.cos(2 * np.pi * np.arange(16) / 4)

    _, stft_once = msst_y(signal, 5, 1)
    _, stft_thrice = msst_y(signal, 5, 3)

    np.testing.assert_allclose(stft_once, stft_thrice)


@pytest.mark.parametrize("signal", [[], [1], np.ones((4, 2)), np.ones((2, 2, 1))])
def test_rejects_invalid_signal(signal):
    with pytest.raises(ValueError):
        msst_y(signal, 3, 1)


@pytest.mark.parametrize("hlength", [0, -1, 2.5, True])
def test_rejects_invalid_window_length(hlength):
    with pytest.raises(ValueError):
        msst_y(np.ones(8), hlength, 1)


@pytest.mark.parametrize("num", [0, -1, 1.5, True])
def test_rejects_invalid_iteration_count(num):
    with pytest.raises(ValueError):
        msst_y(np.ones(8), 3, num)
