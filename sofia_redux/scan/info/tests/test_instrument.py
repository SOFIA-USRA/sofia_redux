# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.flags.mounts import Mount
from sofia_redux.scan.info.instrument import InstrumentInfo
from sofia_redux.scan.source_models.beams.instant_focus import InstantFocus


class TestInstrumentInfo(object):
    def test_init(self):
        info = InstrumentInfo()
        assert info.name is None
        assert info.gain == 1.0
        assert info.mount == Mount.UNKNOWN
        assert info.log_id == 'inst'
        assert np.isnan(info.resolution)
        assert info.get_size_unit() == units.arcsec

    def test_set_configuration(self):
        info = InstrumentInfo()
        assert info.configuration is None
        config = Configuration()
        info.set_configuration(config)
        assert info.configuration is config

    def test_set_mount(self):
        info = InstrumentInfo()

        info.set_mount(2)
        assert info.mount == Mount.CASSEGRAIN

        info.set_mount('LEFT_NASMYTH')
        assert info.mount == Mount.LEFT_NASMYTH

        info.set_mount(Mount.RIGHT_NASMYTH)
        assert info.mount == Mount.RIGHT_NASMYTH

        with pytest.raises(ValueError) as err:
            info.set_mount('bad')
        assert 'bad is not a valid Mount' in str(err)

        with pytest.raises(ValueError) as err:
            info.set_mount([1, 2, 3])
        assert 'is not a valid Mount' in str(err)

    def test_get_source_size(self):
        info = InstrumentInfo()
        assert np.isnan(info.get_source_size())

        info.configuration = Configuration()
        assert np.isnan(info.get_source_size())

        # configure source size
        info.configuration.set_option('sourcesize', 10)
        assert np.isnan(info.get_source_size())

        # configure resolution
        info.resolution = 2 * units.arcsec
        assert info.get_source_size() == np.hypot(10, 2) * units.arcsec

    def test_get_stability(self):
        info = InstrumentInfo()
        assert info.get_stability() == 10 * units.s

        info.configuration = Configuration()
        assert info.get_stability() == 10 * units.s

        info.configuration.set_option('stability', 20)
        assert info.get_stability() == 20 * units.s

    def test_get_point_size(self):
        info = InstrumentInfo()
        assert np.isnan(info.get_point_size())
        info.resolution = 10 * units.arcsec
        assert info.get_point_size() == 10 * units.arcsec

    def test_get_spectral_size(self):
        info = InstrumentInfo()
        assert np.isnan(info.get_spectral_size())
        assert info.get_spectral_size().unit == 'um'

    def test_get_spectral_unit(self):
        assert InstrumentInfo.get_spectral_unit() == 'um'

    def test_get_data_unit(self):
        info = InstrumentInfo()
        assert info.get_data_unit() == units.count

        info.configuration = Configuration()
        assert info.get_data_unit() == units.count

        info.configuration.set_option('dataunit', 'Jy')
        assert info.get_data_unit() == units.Jy

    def test_jansky_per_beam(self):
        info = InstrumentInfo()
        assert info.jansky_per_beam() == 1 * units.Jy / units.beam

        info.configuration = Configuration()
        assert info.jansky_per_beam() == 1 * units.Jy / units.beam

        info.configuration.set_option('jansky', 100)
        assert info.jansky_per_beam() == 100 * units.Jy / units.beam

        info.configuration.set_option('jansky.inverse', True)
        assert info.jansky_per_beam() == .01 * units.Jy / units.beam

        info.configuration.set_option('dataunit', 'mJy')
        assert info.jansky_per_beam() == 1e-5 * units.Jy / units.beam

    def test_kelvin(self):
        info = InstrumentInfo()
        assert np.isnan(info.kelvin())

        info.configuration = Configuration()
        assert np.isnan(info.kelvin())

        info.configuration.set_option('k2jy', 10)
        assert info.kelvin() == 10 * units.Kelvin

        info.configuration.set_option('kelvin', 100)
        assert info.kelvin() == 100 * units.Kelvin

    def test_edit_image_header(self):
        info = InstrumentInfo()
        info.name = 'test'

        hdr = fits.Header()
        info.edit_image_header(hdr)
        assert hdr['INSTRUME'] == 'test'
        assert hdr['V2JY'] == 1
        assert len(hdr) == 2

    def test_validate_scan(self, populated_scan):
        info = InstrumentInfo()
        info.configuration = Configuration()
        inst = populated_scan.info.instrument

        # nothing configured: no op
        info.validate_scan(populated_scan)
        assert np.isnan(inst.frequency)
        assert inst.resolution == 10 * units.arcsec
        assert inst.gain == 1.0

        # configure wavelength, resolution, gain
        info.configuration.set_option('wavelength', 5)
        info.configuration.set_option('resolution', 2)
        info.configuration.set_option('gain', 0.5)
        info.validate_scan(populated_scan)
        assert np.isclose(inst.frequency, 5.9958e13 * units.Hz)
        assert inst.resolution == 2 * units.arcsec
        assert inst.gain == 0.5

        # configure frequency directly instead
        info.configuration.set_option('frequency', 5)
        info.validate_scan(populated_scan)
        assert np.isclose(inst.frequency, 5 * units.Hz)

    def test_get_focus_string(self):
        msg = InstrumentInfo.get_focus_string(None)
        assert msg == 'No instant focus'

        focus = InstantFocus()
        focus.x = 1 * units.cm
        focus.x_weight = 1 * units.cm ** -2
        msg = InstrumentInfo.get_focus_string(focus)
        assert msg == '\n  Focus.dX --> 1.0 cm +- 1.0 cm'

        focus.x = 1
        focus.x_weight = 1
        msg = InstrumentInfo.get_focus_string(focus)
        assert msg == '\n  Focus.dX --> 1.0 mm +- 1.0 mm'
