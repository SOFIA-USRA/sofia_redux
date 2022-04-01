# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.info.camera.info import CameraInfo
from sofia_redux.scan.info.camera.instrument import CameraInstrumentInfo
from sofia_redux.scan.source_models.astro_intensity_map \
    import AstroIntensityMap
from sofia_redux.scan.source_models.pixel_map import PixelMap


class CameraInfoCheck(CameraInfo):
    """Unabstracted class for testing"""
    def max_pixels(self):
        return 10


class TestCameraInfo(object):
    def test_init(self):
        info = CameraInfoCheck()
        assert isinstance(info, CameraInfo)
        assert info.name == 'camera'
        assert isinstance(info.instrument, CameraInstrumentInfo)

    def test_get_source_model_instance(self, populated_scan):
        info = CameraInfoCheck()
        info.configuration = Configuration()

        model = info.get_source_model_instance([populated_scan])
        assert model is None

        info.configuration.set_option('source.type', 'map')
        model = info.get_source_model_instance([populated_scan])
        assert isinstance(model, AstroIntensityMap)

        info.configuration.set_option('source.type', 'pixelmap')
        model = info.get_source_model_instance([populated_scan])
        assert isinstance(model, PixelMap)
        assert model.info is info

    def test_get_rotation_angle(self):
        info = CameraInfoCheck()
        assert info.get_rotation_angle() == 0 * units.deg

    def test_parse_image_header(self):
        info = CameraInfoCheck()
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
        info = CameraInfoCheck()
        header = fits.Header()

        assert np.isnan(info.resolution)
        info.edit_image_header(header)
        assert header['BEAM'] == -9999

        info.resolution = 1 * units.deg
        info.edit_image_header(header)
        assert header['BEAM'] == 3600

    def test_set_pointing(self, populated_scan):
        info = CameraInfoCheck()
        info.configuration = Configuration()

        info.set_pointing()
        assert info.configuration.has_option('point')
        assert info.configuration.get_bool('point', True)

        # no change if repeated
        info.set_pointing()
        assert info.configuration.get_bool('point', True)

        # also set in scan if needed
        del populated_scan.configuration['point']
        info.configuration.set_option('point', False)

        info.set_pointing(scan=populated_scan)
        assert info.configuration.get_bool('point', True)
        assert populated_scan.configuration.get_bool('point', True)

    def test_get_pointing_center_offset(self):
        info = CameraInfoCheck()
        offset = info.get_pointing_center_offset()
        assert offset.x == 0 * units.arcsec
        assert offset.y == 0 * units.arcsec

    def test_get_pointing_offset(self, mocker):
        info = CameraInfoCheck()
        offset = info.get_pointing_offset()
        assert offset.x == 0 * units.arcsec
        assert offset.y == 0 * units.arcsec

        info.instrument.set_mount('CASSEGRAIN')
        offset = info.get_pointing_offset()
        assert offset.x == 0 * units.arcsec
        assert offset.y == 0 * units.arcsec

        offset = info.get_pointing_offset(rotation_angle=90 * units.deg)
        assert offset.x == 0 * units.arcsec
        assert offset.y == 0 * units.arcsec

        coord = Coordinate2D([1, 2], unit=units.arcsec)
        mocker.patch.object(info, 'get_pointing_center_offset',
                            return_value=coord)
        offset = info.get_pointing_offset(rotation_angle=90 * units.deg)
        assert offset.x == 3 * units.arcsec
        assert offset.y == 3 * units.arcsec
