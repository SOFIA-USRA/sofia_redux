# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.custom.fifi_ls.info.info import (
    FifiLsInfo, normalize_scan_coordinates)
from sofia_redux.scan.reduction.reduction import Reduction


second = units.Unit('second')
um = units.Unit('um')


class DummyScan(object):  # pragma: no cover
    def __init__(self):
        self.info = FifiLsInfo()
        self.info.instrument.wavelength = 10 * um
        self.info.instrument.instrument_config = 'config1'

    @staticmethod
    def is_valid():
        return False

    @staticmethod
    def get_id():
        return ''

    @staticmethod
    def get_observing_time():
        return 1 * units.Unit('hour')


def test_normalize_scan_coordinates():
    frames, pixels = 4, 3
    shape = (frames, pixels)
    valid = np.full(frames, True)
    valid[-1] = False
    flags = np.zeros(shape, dtype=int)
    flags[1, 1] = 1
    ra, dec, x, y, data, error = np.empty((6, frames, pixels))
    channel_valid = np.full(pixels, True)
    channel_valid[2] = False

    ra[...] = np.arange(pixels)[None]
    dec[...] = np.arange(pixels)[None] + 10
    x[...] = np.arange(pixels)[None] + 20
    y[...] = np.arange(pixels)[None] + 30
    data[...] = np.arange(pixels)[None] + 40
    error[...] = np.arange(pixels)[None] + 50
    z = np.arange(pixels) + 100
    ra2, dec2, z2, x2, y2, data2, error2 = normalize_scan_coordinates(
        ra=ra, dec=dec, x=x, y=y, z=z, data=data, error=error,
        valid=valid, flags=flags, channel_valid=channel_valid)

    expected = np.asarray([0, 0, 0, 1, 1])
    assert np.allclose(ra2, expected)
    assert np.allclose(dec2, expected + 10)
    assert np.allclose(x2, expected + 20)
    assert np.allclose(y2, expected + 30)
    assert np.allclose(data2, expected + 40)
    assert np.allclose(error2, expected + 50)
    assert np.allclose(z2, expected + 100)


def test_init():
    info = FifiLsInfo()
    assert info.name == 'fifi_ls'
    assert isinstance(info.astrometry, InfoBase)
    assert isinstance(info.detector_array, InfoBase)
    assert isinstance(info.instrument, InfoBase)
    assert info.spectroscopy is None
    assert isinstance(info.scanning, InfoBase)


def test_get_file_id():
    assert FifiLsInfo.get_file_id() == 'FIFI'


def test_edit_header():
    h = fits.Header()
    info = FifiLsInfo()
    info.detector_array.sampling_interval = 0.1 * second
    info.detector_array.boresight_index = Coordinate2D([0, 0])
    info.sampling_interval = 0.1 * second
    info.edit_header(h)
    assert h['SMPLFREQ'] == 10


def test_validate_scans():
    info = FifiLsInfo()
    scan1 = DummyScan()
    scan2 = DummyScan()
    scans = [scan1, scan2]
    scans0 = scans.copy()
    info.validate_scans(None)  # does nothing

    info.validate_scans(scans)
    assert len(scans) == 2

    scan2.info.instrument.instrument_config = 'config2'
    info.validate_scans(scans)
    assert len(scans) == 1
    scan2.info.instrument.instrument_config = 'config1'
    scan2.info.instrument.wavelength = -1 * um

    scans = scans0
    info.validate_scans(scans)
    assert len(scans) == 1


def test_max_pixels():
    assert FifiLsInfo().max_pixels() == 400


def test_get_si_pixel_size():
    info = FifiLsInfo()
    info.detector_array.pixel_sizes = 'a'
    assert info.get_si_pixel_size() == 'a'


def test_perform_reduction(fifi_simulated_hdul, capsys):
    reduction = Reduction('fifi_ls')
    info = reduction.info
    info.configuration.parse_key_value('fifi_ls.resample', 'False')
    info.configuration.parse_key_value('rounds', '1')
    info.perform_reduction(reduction, [fifi_simulated_hdul])
    out = capsys.readouterr().out
    assert 'Performing reduction for subsequent FIFI-LS resampling' not in out
    info.configuration.parse_key_value('fifi_ls.resample', 'True')
    info.configuration.parse_key_value('fifi_ls.insert_source', 'False')
    info.perform_reduction(reduction, [fifi_simulated_hdul])
    out = capsys.readouterr().out
    assert 'Removing decorrelations and offsets from original data' in out
    info.configuration.parse_key_value('fifi_ls.insert_source', 'True')
    info.perform_reduction(reduction, [fifi_simulated_hdul])
    out = capsys.readouterr().out
    assert 'Reinserting source back into cleaned data' in out
