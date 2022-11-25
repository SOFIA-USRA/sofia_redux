# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.exes import despike, diff_arr
from sofia_redux.instruments.exes.tests import resources
from sofia_redux.toolkit.image.adjust import rotate90


@pytest.fixture
def raw_low_hdul():
    hdul = resources.raw_low_nod_on()
    yield hdul
    hdul.close()


@pytest.fixture
def rdc_low_hdul():
    hdul = resources.raw_low_nod_on(coadded=True)
    yield hdul
    hdul.close()


@pytest.fixture
def rdc_low_flat_hdul():
    hdul = resources.raw_low_flat(coadded=True)
    yield hdul
    hdul.close()


@pytest.fixture
def rdc_high_low_flat_hdul():
    hdul = resources.raw_high_low_flat(coadded=True)
    yield hdul
    hdul.close()


@pytest.fixture
def raw_high_low_hdul():
    hdul = resources.raw_high_low_nod_off()
    yield hdul
    hdul.close()


@pytest.fixture
def rdc_high_low_hdul():
    hdul = resources.raw_high_low_nod_off(coadded=True)
    yield hdul
    hdul.close()


@pytest.fixture
def rdc_high_med_flat_hdul():
    hdul = resources.raw_high_med_flat(coadded=True)
    yield hdul
    hdul.close()


@pytest.fixture
def rdc_data(rdc_low_hdul):
    data = rdc_low_hdul['FLUX'].data
    var = rdc_low_hdul['ERROR'].data ** 2
    return data, var


@pytest.fixture
def rdc_header(rdc_low_hdul):
    header = rdc_low_hdul[0].header
    return header


@pytest.fixture
def rdc_beams():
    abeams = np.array([1, 3])
    bbeams = np.array([0, 2])
    beams = {'a': {'beam': abeams}, 'b': {'beam': bbeams}}
    return beams


@pytest.fixture
def rdc_read_noise(rdc_header):
    read_noise = despike.read_noise_contribution(rdc_header)
    return read_noise


@pytest.fixture
def rdc_b_info(rdc_header, rdc_data, rdc_beams):
    data, var = rdc_data
    header = rdc_header.copy()
    header['INSTMODE'] = 'MAP'
    info = diff_arr.check_beams(abeams=rdc_beams['a']['beam'],
                                bbeams=rdc_beams['b']['beam'],
                                header=header, data=data,
                                variance=var, do_var=True,
                                nz=2*len(rdc_beams['a']['beam']))
    return info


@pytest.fixture
def rdc_dark():
    data_loc = os.path.join(os.path.dirname(__file__), 'data')
    data_file = glob.glob(os.path.join(data_loc,
                                       'dark_2015.02.13.fits'))[0]
    hdul = fits.open(data_file)
    dark = hdul[0].data[0]
    hdul.close()
    return dark


@pytest.fixture
def nsb_cross_dispersed_hdul():
    hdul = resources.nodsub_hdul(mode='high_low')
    yield hdul
    hdul.close()


@pytest.fixture
def nsb_single_order_hdul():
    hdul = resources.nodsub_hdul(mode='low')
    yield hdul
    hdul.close()


@pytest.fixture
def cross_dispersed_flat():
    hdul = resources.flat_hdul(mode='high_low')
    yield hdul
    hdul.close()


@pytest.fixture
def single_order_flat():
    hdul = resources.flat_hdul(mode='low')
    yield hdul
    hdul.close()


@pytest.fixture
def und_cross_dispersed_hdul():
    # for first pass undistortion, just rotate image to
    # align spectra along x
    hdul = resources.nodsub_hdul(mode='high_low', do_flat=False)
    for ext in hdul:
        for i, data in enumerate(ext.data):
            ext.data[i] = rotate90(data, 3)
    yield hdul
    hdul.close()
