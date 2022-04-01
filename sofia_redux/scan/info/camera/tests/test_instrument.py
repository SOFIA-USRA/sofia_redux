# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.info.camera.instrument import CameraInstrumentInfo


class TestCameraInstrumentInfo(object):
    def test_init(self):
        info = CameraInstrumentInfo()
        assert info.rotation == 0 * units.deg

    def test_parse_image_header(self):
        info = CameraInstrumentInfo()
        header = fits.Header()

        assert np.isnan(info.resolution)
        info.parse_image_header(header)
        assert np.isnan(info.resolution)

        header['BEAM'] = 10
        info.parse_image_header(header)
        assert info.resolution == 10 * units.arcsec

        header['BEAM'] = 20
        info.parse_image_header(header)
        assert info.resolution == 20 * units.arcsec

    def test_edit_image_header(self):
        info = CameraInstrumentInfo()
        header = fits.Header()

        assert np.isnan(info.resolution)
        info.edit_image_header(header)
        assert header['BEAM'] == -9999

        info.resolution = 1 * units.deg
        info.edit_image_header(header)
        assert header['BEAM'] == 3600
