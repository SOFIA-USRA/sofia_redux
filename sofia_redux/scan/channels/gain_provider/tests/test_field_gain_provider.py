# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.gain_provider.field_gain_provider \
    import FieldGainProvider


class TestFieldGainProvider(object):

    def test_init(self):
        provider = FieldGainProvider('test')
        assert provider.field == 'test'

    def test_get_gain(self, populated_data):
        provider = FieldGainProvider('test')
        with pytest.raises(ValueError) as err:
            provider.get_gain(populated_data)
        assert 'does not contain test field' in str(err)

        provider = FieldGainProvider('flag')
        gain = provider.get_gain(populated_data)
        assert gain.size == populated_data.size
        assert np.all(gain == 0)

    def test_set_gain(self, populated_data):
        nchannel = populated_data.size
        provider = FieldGainProvider('test')

        with pytest.raises(ValueError) as err:
            provider.set_gain(populated_data, np.arange(nchannel))
        assert 'does not have a test field' in str(err)

        with pytest.raises(ValueError) as err:
            provider.set_gain(populated_data, np.arange(10))
        assert 'does not match channel size' in str(err)

        provider = FieldGainProvider('variance')
        provider.set_gain(populated_data, np.arange(nchannel))
        assert np.allclose(populated_data.variance, np.arange(nchannel))

        single = populated_data[0]
        provider.set_gain(single, 1 * units.dimensionless_unscaled)
        assert np.allclose(single.variance, 1)
