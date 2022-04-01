# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.channel_data.color_arrangement_data \
    import ColorArrangementData


class TestColorArrangementData(object):

    def test_init(self, populated_data):
        assert isinstance(populated_data, ColorArrangementData)

    def test_set_beam_size(self, populated_data):
        with pytest.raises(ValueError) as err:
            populated_data.set_beam_size(10)
        assert 'must be' in str(err)

        populated_data.set_beam_size(1 * units.deg)
        assert np.all(populated_data.resolution == 1 * units.deg)
        assert populated_data.resolution.size == populated_data.size

        populated_data.set_beam_size(10 * units.arcsec)
        assert np.all(populated_data.resolution == 10 * units.arcsec)

    def test_apply_info(self, capsys, populated_data):
        info = populated_data.info

        # set beam configuration as float
        populated_data.configuration.set_option('beam', 20)
        populated_data.apply_info(info)
        assert populated_data.info.instrument.resolution == 20 * units.arcsec

        # set as alias
        populated_data.configuration.set_option('beam', 'other')

        # missing alias
        populated_data.apply_info(info)
        assert np.isnan(populated_data.info.instrument.resolution)
        assert 'Could not parse' in capsys.readouterr().err

        # with alias set
        populated_data.configuration.set_option('other', 15)
        populated_data.apply_info(info)
        assert populated_data.info.instrument.resolution == 15 * units.arcsec
