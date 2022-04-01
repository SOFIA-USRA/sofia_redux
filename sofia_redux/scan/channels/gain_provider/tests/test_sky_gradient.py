# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from sofia_redux.scan.channels.gain_provider.sky_gradient \
    import SkyGradient


class TestSkyGradient(object):

    def test_init(self):
        provider = SkyGradient(horizontal=True)
        assert provider.horizontal is True
        provider = SkyGradient(horizontal=False)
        assert provider.horizontal is False

    def test_get_relative_gain(self, populated_data):
        pos = populated_data.position
        provider = SkyGradient(horizontal=True)
        assert np.allclose(provider.get_relative_gain(populated_data), pos.x)
        provider = SkyGradient(horizontal=False)
        assert np.allclose(provider.get_relative_gain(populated_data), pos.y)

        populated_data.position = None
        assert np.all(np.isnan(provider.get_relative_gain(populated_data)))

        with pytest.raises(TypeError) as err:
            provider.get_relative_gain('test')
        assert "does not have 'position' attribute" in str(err)
