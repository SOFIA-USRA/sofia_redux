# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.detector_array import \
    SofiaDetectorArrayInfo


arcsec = units.Unit('arcsec')


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['DETECTOR'] = 'HAWC'
    h['DETSIZE'] = '64,40'
    h['PIXSCAL'] = 9.43
    h['SUBARRNO'] = 3
    h['SIBS_X'] = 15.5
    h['SIBS_Y'] = 19.5
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['SUBARR01'] = 10
    h['SUBARR02'] = 11
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


def test_class():
    assert SofiaDetectorArrayInfo.subarrays == 0


def test_init():
    info = SofiaDetectorArrayInfo()
    assert info.detector_name is None
    assert info.detector_size_string is None
    assert np.isnan(info.pixel_size) and info.pixel_size.unit == arcsec
    assert info.subarray_size is None
    assert info.boresight_index.size == 0
    assert info.grid is None


def test_log_id():
    info = SofiaDetectorArrayInfo()
    assert info.log_id == 'sofscan/array'


def test_apply_configuration(sofia_configuration):
    info = SofiaDetectorArrayInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.detector_name == 'HAWC'
    assert info.detector_size_string == '64,40'
    assert info.pixel_size == 9.43 * arcsec
    assert info.subarray_size == ['10', '11', None]
    assert info.boresight_index.x == 15.5
    assert info.boresight_index.y == 19.5
    assert info.grid.projection.get_full_name() == 'Gnomonic'
    del info.configuration.fits.options['SUBARRNO']
    info.configuration.fits.header['CTYPE1'] = 1
    info.configuration.fits.header['CTYPE2'] = 2
    info.apply_configuration()
    assert info.grid is None
    assert info.subarray_size == []
    del info.configuration.fits.header['CTYPE1']
    del info.configuration.fits.header['CTYPE2']
    info.apply_configuration()
    assert info.grid is None

    info = SofiaDetectorArrayInfo()
    info.apply_configuration()
    assert info.detector_name is None


def test_edit_header(sofia_configuration):
    info = SofiaDetectorArrayInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    info.subarrays = 3
    h = fits.Header()
    info.edit_header(h)
    expected = {'DETECTOR': 'HAWC',
                'DETSIZE': '64,40',
                'PIXSCAL': 9.43,
                'SUBARRNO': 3,
                'SUBARR00': '10',
                'SUBARR01': '11',
                'SIBS_X': 15.5,
                'SIBS_Y': 19.5,
                'CTYPE1': 'RA---TAN',
                'CTYPE2': 'DEC--TAN',
                'CRPIX1': 1.0,
                'CRPIX2': 1.0,
                'CRVAL1': 0.0,
                'CRVAL2': 0.0,
                'RADESYS': 'FK5',
                'EQUINOX': 2000.0,
                'WCSNAME': 'Equatorial Coordinates',
                'CDELT1': -1.0,
                'CDELT2': 1.0,
                'CUNIT1': 'deg',
                'CUNIT2': 'deg'}
    for key, value in expected.items():
        assert h[key] == value

    info.boresight_index = None
    info.edit_header(h)
    assert h['SIBS_X'] == -9999
    assert h['SIBS_Y'] == -9999


def test_get_table_entry():
    info = SofiaDetectorArrayInfo()
    info.boresight_index.x = 15.5
    info.boresight_index.y = 19.5
    assert info.get_table_entry('sibsx') == 15.5
    assert info.get_table_entry('sibsy') == 19.5
    assert info.get_table_entry('foo') is None
