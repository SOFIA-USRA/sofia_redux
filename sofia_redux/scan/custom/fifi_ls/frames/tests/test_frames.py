# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.index_3d import Index3D
from sofia_redux.scan.custom.fifi_ls.info.info import FifiLsInfo
from sofia_redux.scan.custom.fifi_ls.frames.frames import FifiLsFrames
from sofia_redux.scan.custom.fifi_ls.flags.frame_flags import FifiLsFrameFlags


def test_init():
    frames = FifiLsFrames()
    assert frames.flagspace is FifiLsFrameFlags


def test_copy():
    frames = FifiLsFrames()
    frames_2 = frames.copy()
    assert isinstance(frames_2, FifiLsFrames)
    assert frames_2 is not frames


def test_default_channel_fields():
    frames = FifiLsFrames()
    assert frames.default_channel_fields['map_index'] == (Index3D, -1)


def test_info(fifi_simulated_frames):
    assert isinstance(fifi_simulated_frames.info, FifiLsInfo)


def test_site():
    frames = FifiLsFrames()
    location = GeodeticCoordinates([1, 2])
    frames.sofia_location = location
    assert frames.site == location


def test_detector_coordinates_to_equatorial_offsets(fifi_simulated_frames):
    frames = fifi_simulated_frames.copy()
    coordinates = Coordinate2D(np.arange(20).reshape((2, 10)), unit='arcsec')
    offsets = frames.detector_coordinates_to_equatorial_offsets(coordinates)
    assert np.allclose(offsets.x, -np.arange(10) * units.Unit('arcsec'))
    assert np.allclose(offsets.y, np.arange(10, 20) * units.Unit('arcsec'))


def test_detector_coordinates_to_equatorial(fifi_simulated_frames):
    frames = fifi_simulated_frames.copy()
    coordinates = Coordinate2D(np.arange(10).reshape((2, 5)), unit='arcsec')
    equatorial = frames.detector_coordinates_to_equatorial(coordinates)
    assert np.allclose(
        equatorial.ra, [266.41500888, 266.41469126, 266.41437364,
                        266.41405602, 266.41373841] * units.Unit('degree'),
        atol=1e-3)
    assert np.allclose(
        equatorial.dec, [-29.00472222, -29.00444444, -29.00416667,
                         -29.00388889, -29.00361111] * units.Unit('degree'),
        atol=1e-3)


def test_read_hdul(fifi_simulated_frames, fifi_simulated_hdul):
    frames = fifi_simulated_frames.copy()
    frames.configuration.parse_key_value('fifi_ls.uncorrected', 'False')
    frames.configuration.parse_key_value('lab', 'False')
    frames.data.fill(np.nan)
    frames.relative_weight.fill(0)
    frames.has_telescope_info.fill(False)
    frames.instrument_vpa.fill(np.nan)
    frames.telescope_vpa.fill(np.nan)
    frames.utc.fill(0)
    frames.mjd.fill(0)
    frames.site.zero()
    frames.equatorial.zero()
    frames.object_equatorial.zero()
    frames.horizontal_offset.zero()
    frames.horizontal.zero()
    frames.valid.fill(False)
    frames.read_hdul(fifi_simulated_hdul)
    assert not np.isnan(frames.data).all()
    assert np.allclose(frames.instrument_vpa, 0 * units.Unit('degree'))
    assert np.allclose(frames.telescope_vpa, 0 * units.Unit('degree'))
    assert not np.allclose(frames.utc, 0)
    assert not np.allclose(frames.mjd, 0)
    assert not frames.site.is_null().any()
    assert not np.allclose(frames.lst, 0 * units.Unit('hourangle'))
    assert not frames.equatorial.is_null().any()
    assert not frames.object_equatorial.is_null().any()
    assert not frames.horizontal_offset.is_null().all()
    assert not frames.horizontal.is_null().any()
    assert frames.valid.all()

    frames.data.fill(np.nan)
    frames.configuration.parse_key_value('fifi_ls.uncorrected', 'True')
    frames.configuration.parse_key_value('lab', 'True')
    frames.equatorial.zero()
    frames.read_hdul(fifi_simulated_hdul)
    assert frames.equatorial.is_null().all()
    assert not np.isnan(frames.data).all()


def test_validate(fifi_simulated_frames):
    frames = fifi_simulated_frames.copy()
    assert not frames.sample_flag.any()
    frames.data[0, 0] = np.nan
    frames.validate()
    assert frames.sample_flag[0, 0] != 0
