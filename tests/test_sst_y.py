import numpy as np
import pytest

from sst_y import sst_y


def test_returns_expected_shape_and_complex_values():
    signal = np.sin(2 * np.pi * np.arange(20) / 5)

    result = sst_y(signal, hlength=4)

    assert result.shape == (10, 20)
    assert np.iscomplexobj(result)
    assert np.all(np.isfinite(result))


def test_accepts_column_vector():
    signal = np.arange(9, dtype=float)

    np.testing.assert_allclose(sst_y(signal[:, None], 3), sst_y(signal, 3))


def test_preserves_the_zero_frequency_bin():
    result = sst_y(np.ones(20), hlength=5)

    assert np.any(np.abs(result[0]) > 0)


@pytest.mark.parametrize("signal", [[], np.ones((4, 2)), np.ones((2, 2, 1))])
def test_rejects_invalid_signal(signal):
    with pytest.raises(ValueError):
        sst_y(signal)


@pytest.mark.parametrize("hlength", [0, -1, 2.5, True])
def test_rejects_invalid_window_length(hlength):
    with pytest.raises(ValueError):
        sst_y(np.ones(8), hlength)
