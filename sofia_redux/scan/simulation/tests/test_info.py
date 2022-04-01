# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.simulation.info import SimulationInfo
from sofia_redux.scan.custom.example.frames.frames import ExampleFrames
from sofia_redux.scan.simulation.source_models.single_gaussian import \
    SingleGaussian
from sofia_redux.scan.simulation.source_models.sky import Sky


arcsec = units.Unit('arcsec')


def test_init():
    info = SimulationInfo()
    assert info.name == 'simulation'
    for attribute in ['instrument', 'telescope', 'astrometry', 'observation',
                      'detector_array']:
        assert isinstance(getattr(info, attribute), InfoBase)
    assert info.resolution == 10 * arcsec


def test_get_file_id():
    assert SimulationInfo.get_file_id() == 'SIML'


def test_max_pixels():
    assert SimulationInfo().max_pixels() == 121


def test_copy():
    info = SimulationInfo()
    info.resolution = 11 * arcsec
    info2 = info.copy()
    assert info is not info2 and info2.resolution == 11 * arcsec


def test_write_simulated_hdul(tmpdir):
    filename = str(tmpdir.mkdir('test_write_simulated_hdul').join('test.fits'))
    info = SimulationInfo()
    info.write_simulated_hdul(filename, fwhm=10 * units.Unit('arcsec'))
    hdul = fits.open(filename)
    assert isinstance(hdul[1], fits.BinTableHDU)


def test_simulated_hdul():
    info = SimulationInfo()
    hdul = info.simulated_hdul(fwhm=10 * arcsec)
    assert isinstance(hdul[1], fits.BinTableHDU)


def test_simulated_observation_header():
    h = SimulationInfo.simulated_observation_header()
    for key, value in {'OBJECT': 'Simulation', 'SCANID': 1,
                       'DATE-OBS': '2021-12-06T18:48:25.876',
                       'SCANPATT': 'daisy', 'OBSRA': 17.76100059166667,
                       'OBSDEC': -29.00611111111111, 'SITELON': -122.0644,
                       'SITELAT': 37.4089, 'LST': 15.72123722613777,
                       'OBSAZ': 152.1496509553599,
                       'OBSEL': 17.661336430535634}.items():
        if isinstance(value, float):
            assert np.isclose(h[key], value, rtol=1e-2)
        else:
            assert h[key] == value

    h = SimulationInfo.simulated_observation_header(ra=15, dec=30)
    assert np.isclose(h['OBSRA'], 1)
    assert np.isclose(h['OBSDEC'], 30)


def test_scan_hdu_from_header():
    info = SimulationInfo()
    h = info.simulated_observation_header()
    h['SCANPATT'] = 'foo'
    with pytest.raises(NotImplementedError) as err:
        _ = info.scan_hdu_from_header(h)
    assert 'FOO scanning pattern not supported' in str(err.value)

    for pattern in ['daisy', 'lissajous', 'skydip']:
        h['SCANPATT'] = pattern
        hdu = info.scan_hdu_from_header(h)
        table = Table(hdu.data)
        for column in ['RA', 'DEC', 'DMJD', 'LST', 'AZ', 'EL']:
            assert column in table.columns


def test_set_frames_coordinates():
    info = SimulationInfo()
    h = info.simulated_observation_header()
    hdu = info.scan_hdu_from_header(h)
    frames = ExampleFrames()
    lon = h['SITELON'] * units.Unit('degree')
    lat = h['SITELAT'] * units.Unit('degree')
    ra = h['OBSRA'] * units.Unit('hourangle')
    dec = h['OBSDEC'] * units.Unit('degree')
    info.astrometry.equatorial = EquatorialCoordinates([ra, dec])
    info.astrometry.site = GeodeticCoordinates([lon, lat])
    info.set_frames_coordinates(frames, hdu.data)
    assert isinstance(frames.horizontal_offset, Coordinate2D)
    assert frames.horizontal_offset.size == 1100
    assert not np.any(frames.horizontal_offset.is_nan())
    assert not np.all(frames.horizontal_offset.is_null())


def test_simulated_data():
    source = SingleGaussian(fwhm=10 * arcsec)
    info = SimulationInfo()
    h = info.simulated_observation_header()
    hdu = info.scan_hdu_from_header(h)
    hdu2 = info.simulated_data(hdu, h, source)
    dac = hdu2.data['DAC']
    assert dac.shape == (1100, 11, 11)
    assert np.all(np.isfinite(dac))
    assert not np.all(dac == 0)


def test_create_data():
    source = SingleGaussian(fwhm=10 * arcsec)
    info = SimulationInfo()
    h = info.simulated_observation_header()
    hdu = info.scan_hdu_from_header(h)
    data = info.create_data(hdu, h, source)
    assert data.shape == (1100, 11, 11)
    assert np.all(np.isfinite(data))
    assert not np.all(data == 0)

    source = Sky()
    data2 = info.create_data(hdu, h, source)
    assert data2.shape == (1100, 11, 11)
    assert np.all(np.isfinite(data2))
    assert not np.all(data2 == 0)
    assert not np.allclose(data, data2)


def test_modify_data():
    data = np.ones((100, 100))
    data0 = data.copy()
    SimulationInfo.modify_data(data, s2n=1)
    assert not np.allclose(data, data0)
    dev = np.std(data)
    assert np.isclose(dev, 1, atol=0.1)

    data = np.ones((3, 3))
    SimulationInfo.modify_data(data, s2n=3, seed=2)
    assert np.allclose(data,
                       [[0.86108072, 0.98124439, 0.28793463],
                        [1.54675694, 0.40218814, 0.71941754],
                        [1.16762714, 0.58490397, 0.64734926]],
                       atol=1e-3)
