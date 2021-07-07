# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import numpy as np
import numpy.testing as npt
from astropy import constants
import astropy.units as u
import pytest

from sofia_redux.calibration.standard_model import modconvert
from sofia_redux.calibration.pipecal_error import PipeCalError


@pytest.fixture(scope='function')
def infile():
    data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))))
    data_loc = os.path.join(data_loc, 'data', 'models', 'nep_esa5_2_i.dat')
    return data_loc


@pytest.fixture(scope='function')
def infile_large_header(infile):
    data = np.genfromtxt(infile)
    header = '\n'.join(['Header'] * 4)
    filename = 'large_header.dat'
    np.savetxt(filename, data, header=header)
    return filename


@pytest.fixture
def outfile():
    return 'scaled_flux.out'


@pytest.fixture
def wave():
    return np.linspace(20, 300, 50)


@pytest.fixture
def freq(wave):
    return (constants.c.to('um/s') / (wave * u.um)).value


@pytest.fixture
def flux(wave):
    return np.log10(wave)


@pytest.fixture
def brightness_temp(wave):
    return np.log2(wave)


@pytest.fixture
def rj_temp(wave):
    return np.log(wave)


def monotonically_increasing(x):
    return np.all(np.diff(x) > 0)


def monotonically_decreasing(x):
    return np.all(np.diff(x) < 0)


def test_read_infile(infile):
    index, freq, temp, flux, trj = modconvert.read_infile(infile)
    assert len(index) == 19983
    npt.assert_array_equal(index, np.arange(1, 19984))
    npt.assert_allclose(np.min(freq), 2.)
    npt.assert_allclose(np.max(temp), 297.8497)
    npt.assert_allclose(np.mean(flux), 111.240534)
    npt.assert_allclose(np.median(trj), 0.0986)


def test_read_infile_large_header(infile, infile_large_header):
    correct = modconvert.read_infile(infile)
    data = modconvert.read_infile(infile_large_header)
    npt.assert_array_equal(data, correct)
    os.remove(infile_large_header)


def test_read_infile_bad():
    dummy_file = 'dummy_file.txt'
    with pytest.raises(PipeCalError) as excinfo:
        modconvert.read_infile(dummy_file)
    assert dummy_file in str(excinfo.value)


def test_sort_spectrum(freq, flux, brightness_temp):
    w, f, t = modconvert.sort_spectrum(freq, flux, brightness_temp)
    assert monotonically_increasing(w)


def test_plot_scaled_spectrum(wave, flux, infile):
    scaled_flux = 0.15 * flux
    filename = 'scaled_flux_nep_esa5_2_i.png'
    modconvert.plot_scaled_spectrum(wave, scaled_flux, 0.15, infile)
    assert os.path.isfile(filename)
    os.remove(filename)


def test_write_scaled_spectrum(wave, flux, brightness_temp, infile, outfile):
    fscale = 0.15
    scaled_flux = fscale * flux
    modconvert.write_scaled_spectrum(wave, scaled_flux, fscale,
                                     brightness_temp, infile, outfile)
    assert os.path.isfile(outfile)
    os.remove(outfile)
