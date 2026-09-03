import numpy as np
import pytest

from instantaneous_frequency import ridge_frequency


def test_converts_strongest_bins_to_hertz():
    transform = np.zeros((4, 5), dtype=complex)
    transform[[0, 1, 2, 3, 1], np.arange(5)] = 2

    np.testing.assert_allclose(ridge_frequency(transform, 100), [0, 20, 40, 60, 20])


@pytest.mark.parametrize("sample_rate", [0, -1, np.inf, np.nan])
def test_rejects_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError):
        ridge_frequency(np.ones((2, 3)), sample_rate)


@pytest.mark.parametrize("transform", [[], [1, 2], np.empty((0, 2))])
def test_rejects_invalid_transform(transform):
    with pytest.raises(ValueError):
        ridge_frequency(transform, 100)
