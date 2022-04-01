# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.custom.hawc_plus.channels.gain_provider.pol_imbalance \
    import HawcPlusPolImbalance


class DummyChannelData(object):
    def __init__(self):
        self.pol = np.zeros(10, dtype=int)

    @property
    def size(self):
        return self.pol.size


def test_init():
    provider = HawcPlusPolImbalance()
    assert provider.ave_g == 0


def test_get_relative_gain():
    data = DummyChannelData()
    provider = HawcPlusPolImbalance()
    gain = provider.get_relative_gain(data)
    assert np.allclose(gain, np.ones(10, dtype=int))
    data.pol.fill(1)
    gain = provider.get_relative_gain(data)
    assert np.allclose(gain, -np.ones(10, dtype=int))


def test_set_raw_gain():
    provider = HawcPlusPolImbalance()
    with pytest.raises(NotImplementedError) as err:
        provider.set_raw_gain(None, None)
    assert "Cannot set polarization imbalance gains" in str(err.value)
