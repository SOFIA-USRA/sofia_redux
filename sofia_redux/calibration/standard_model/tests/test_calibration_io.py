# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pytest
import numpy.testing as npt
import numpy as np
import pandas as pd
import astropy.io.fits as pf

from sofia_redux.calibration.standard_model import calibration_io as cio
from sofia_redux.calibration.standard_model.tests import resources
from sofia_redux.calibration.pipecal_error import PipeCalError


def strictly_increasing(arr):
    return all(x < y for x, y in zip(arr, arr[1:]))


def strictly_decreasing(arr):
    return all(x > y for x, y in zip(arr, arr[1:]))


@pytest.fixture(scope='module')
def result():
    s = dict()
    s['lambda_c'] = 50 * np.ones(5)
    s['lambda_mean'] = 50 * np.ones(5)
    s['lambda_1'] = 50 * np.ones(5)
    s['lambda_pivot'] = 50 * np.ones(5)
    s['lambda_eff'] = 50 * np.ones(5)
    s['lambda_eff_jv'] = 50 * np.ones(5)
    s['isophotal_wt'] = 50 * np.ones(5)
    s['width'] = 50 * np.ones(5)
    s['response'] = 50 * np.ones(5)
    s['flux_mean'] = 50 * np.ones(5)
    s['flux_nu_mean'] = 50 * np.ones(5)
    s['color_term_k0'] = 50 * np.ones(5)
    s['color_term_k1'] = 50 * np.ones(5)
    s['source_rate'] = 50 * np.ones(5)
    s['source_size'] = 50 * np.ones(5)
    s['source_fwhm'] = 50 * np.ones(5)
    s['background_power'] = 50 * np.ones(5)
    s['nep'] = 50 * np.ones(5)
    s['nefd'] = 50 * np.ones(5)
    s['mdcf'] = 50 * np.ones(5)
    s['npix_mean'] = 50 * np.ones(5)
    s['lambda_prime'] = 50 * np.ones(5)
    s['lamcorr'] = 50 * np.ones(5)
    s = pd.DataFrame(s)
    return s


@pytest.fixture(scope='module')
def wavelengths():
    return np.linspace(5, 300, 5)


@pytest.fixture(scope='module')
def fluxes(wavelengths):
    return np.log10(wavelengths)


@pytest.fixture(scope='module')
def shuffled_data(wavelengths, fluxes):
    rand = np.random.RandomState(42)
    p = rand.permutation(len(wavelengths))
    return wavelengths[p], fluxes[p]


@pytest.fixture(scope='function')
def ascii_model(wavelengths, fluxes, tmpdir):
    filename = str(tmpdir.join('model.spec'))
    model = np.array([wavelengths, fluxes]).T
    np.savetxt(filename, model, header='#wavelength\tflux')
    return filename


@pytest.fixture(scope='function')
def dataframe_model(ascii_model):
    files = [ascii_model, 'dummy_file.txt']
    calfiles = ['dataframe.out', 'dummy_file.out']
    df = pd.DataFrame({'model_file': files, 'cal_file': calfiles})
    return df


@pytest.fixture(scope='function')
def bad_dataframe_model(ascii_model):
    files = [ascii_model, 'dummy_file.txt']
    df = pd.DataFrame({'Model_file': files})
    return df


@pytest.fixture(scope='function')
def fits_model(wavelengths, fluxes, tmpdir):
    filename = str(tmpdir.join('model.fits'))
    model = np.array([wavelengths, fluxes])
    model = np.expand_dims(model, axis=0)
    hdu = pf.PrimaryHDU(model)
    hdul = pf.HDUList(hdu)
    hdul.writeto(filename, overwrite=True)
    return filename


def test_model_spectrum_ascii(ascii_model, wavelengths, fluxes):
    w, f, pl, bb = cio.model_spectrum(ascii_model, txt=True)
    assert not pl
    assert not bb
    npt.assert_array_equal(w, wavelengths)
    npt.assert_array_equal(f, fluxes)


def test_model_spectrum_dataframe(dataframe_model, wavelengths,
                                  fluxes, ascii_model):
    w, f, pl, bb = cio.model_spectrum(dataframe_model, dataframe=True)
    assert not pl
    assert not bb
    npt.assert_array_equal(w, wavelengths)
    npt.assert_array_equal(f, fluxes)


def test_model_spectrum_dataframe_bad(bad_dataframe_model, ascii_model):
    with pytest.raises(PipeCalError):
        cio.model_spectrum(bad_dataframe_model, dataframe=True)


def test_model_spectrum_powerlaw():
    w, f, pl, bb = cio.model_spectrum('PowerLaw', alpha=3)
    assert pl
    assert not bb


def test_model_spectrum_powerlaw_no_alpha():
    with pytest.raises(PipeCalError):
        cio.model_spectrum('powerlaw')


def test_model_spectrum_blackbody():
    w, r, pl, bb = cio.model_spectrum('BLACKBODY', temp=250)
    assert not pl
    assert bb


def test_model_spectrum_blackbody_no_temp():
    with pytest.raises(PipeCalError):
        cio.model_spectrum('blackbody')


def test_model_fits(fits_model, wavelengths, fluxes):
    w, f, pl, bb = cio.model_spectrum(fits_model)
    assert not pl
    assert not bb
    npt.assert_array_equal(w, wavelengths)
    npt.assert_array_equal(f, fluxes)


@pytest.mark.parametrize('alpha,wmin,wmax,direction', [(2, 20, 200, 'd'),
                                                       (-0.5, 100, 150, 'u')])
def test_generate_power_law(alpha, wmin, wmax, direction):
    dw = 0.005
    w, f = cio.generate_power_law(alpha=alpha, wmin=wmin, wmax=wmax)
    npt.assert_allclose(w[1:] - w[:-1], dw, atol=0.0001)
    assert np.min(w) >= wmin
    assert np.max(w) <= wmax
    assert len(w) == len(f)
    assert strictly_increasing(w)
    if direction == 'u':
        assert strictly_increasing(f)
    else:
        assert strictly_decreasing(f)


@pytest.mark.parametrize('temp,wmin,wmax,direction', [(100, 20, 200, 'd'),
                                                      (500, 10, 150, 'u')])
def test_generate_blackbody(temp, wmin, wmax, direction):
    dw = 0.005
    w, f = cio.generate_blackbody(temp=temp, wmin=wmin, wmax=wmax)
    npt.assert_allclose(w[1:] - w[:-1], dw, atol=0.0001)
    assert np.min(w) >= wmin
    assert np.max(w) <= wmax
    assert len(w) == len(f)
    assert strictly_increasing(w)
    assert not strictly_increasing(f)
    assert not strictly_decreasing(f)


def test_read_fits_no_atm(wavelengths):
    w, t, afile = cio.read_atran('dummy_file.out', ws=wavelengths, no_atm=True)
    npt.assert_array_equal(w, wavelengths)
    npt.assert_allclose(t, 1.)
    assert afile is None


def test_read_atran(wavelengths, tmpdir):
    atran_file = resources.atran(tmpdir)
    w, t, afile = cio.read_atran(atran_file, ws=wavelengths)
    assert atran_file == afile
    assert len(w) == len(t)
    assert np.min(t) >= 0
    assert np.max(t) <= 1


def test_calibration_data_path():
    caldata = cio.calibration_data_path()
    true_path = os.path.join('sofia_redux', 'calibration', 'data', 'models')
    assert true_path in caldata
    assert os.path.isdir(caldata)


def test_open_outfile_and_header(capsys, tmpdir):
    with tmpdir.as_cwd():
        outfile = 'test.out'
        afile = 'atran.fits'
        infile = 'test.in'
        outf = cio.open_outfile_and_header(outfile=outfile, no_atm=False,
                                           afile=afile, infile=infile)
        captured = capsys.readouterr()
        assert 'Using ATRAN' in captured.out

        assert os.path.isfile(outfile)
        assert not outf.closed
        assert os.path.basename(outf.name) == outfile
        outf.close()

        with open(outfile, 'r') as f:
            lines = f.read()
        assert len(lines.split('\n')) == 7
        assert afile in lines


def test_open_outfile_and_header_no_atm(capsys, tmpdir):
    with tmpdir.as_cwd():
        outfile = 'test.out'
        infile = 'test.in'
        outf = cio.open_outfile_and_header(outfile=outfile, no_atm=True,
                                           infile=infile)
        captured = capsys.readouterr()
        assert 'No atmosphere' in captured.out

        assert os.path.isfile(outfile)
        assert not outf.closed
        assert os.path.basename(outf.name) == outfile
        outf.close()

        with open(outfile, 'r') as f:
            lines = f.read()
        assert 'No atmosphere' in captured.out
        assert len(lines.split('\n')) == 7


def test_open_outfile_and_header_dataframe(capsys, dataframe_model,
                                           ascii_model, tmpdir):
    with tmpdir.as_cwd():
        afile = 'atran.fits'
        outf = cio.open_outfile_and_header(outfile=None, no_atm=False,
                                           afile=afile, infile=dataframe_model)
        captured = capsys.readouterr()
        assert 'Using ATRAN' in captured.out

        outfile = 'dataframe.out'

        assert os.path.isfile(outfile)
        assert not outf.closed
        assert os.path.basename(outf.name) == outfile
        outf.close()

        with open(outfile, 'r') as f:
            lines = f.read()
        assert len(lines.split('\n')) == 7
        assert afile in lines


def test_open_outfile_and_header_no_outfile(capsys, tmpdir):
    with tmpdir.as_cwd():
        outfile = 'flux_values.out'
        afile = 'atran.fits'
        infile = 'test.in'
        outf = cio.open_outfile_and_header(outfile=None, no_atm=False,
                                           afile=afile, infile=infile)
        captured = capsys.readouterr()
        assert 'Using ATRAN' in captured.out

        assert os.path.isfile(outfile)
        assert not outf.closed
        assert os.path.basename(outf.name) == outfile
        outf.close()

        with open(outfile, 'r') as f:
            lines = f.read()
        assert len(lines.split('\n')) == 7
        assert afile in lines


def test_report_result(result, tmp_path, capsys):
    filter_name = 'HAWC_bandA.dat'
    outfile = tmp_path / 'test.out'
    outf = open(outfile, 'w')
    cio.report_result(result.iloc[0], filter_name, outf)
    outf.close()
    captured = capsys.readouterr()
    with open(outfile, 'r') as f:
        lines = f.read()
    assert lines.strip() == captured.out.strip()
    assert filter_name in captured.out


@pytest.mark.parametrize('outfile,outflag,bb',
                         [('test.png', True, True),
                          ('spectrum.png', False, False)])
def test_plot_spectrum(wavelengths, fluxes, capsys, result,
                       outfile, outflag, bb, tmpdir):
    with tmpdir.as_cwd():
        if outflag:
            cio.plot_spectrum(model_wave=wavelengths, model_flux=fluxes,
                              power_law=False, blackbody=bb,
                              isophotal_weight=wavelengths,
                              calibration_results=result,
                              outfile=outfile)
        else:
            cio.plot_spectrum(model_wave=wavelengths, model_flux=fluxes,
                              power_law=False, blackbody=bb,
                              isophotal_weight=wavelengths,
                              calibration_results=result)
        captured = capsys.readouterr()
        assert os.path.isfile(outfile)
        assert outfile in captured.out
        assert len(captured.out.split('\n')) == 10


def test_unique_wavelengths(shuffled_data):
    assert not strictly_increasing(shuffled_data[0])
    assert not strictly_increasing(shuffled_data[1])
    w, f = cio.unique_wavelengths(shuffled_data[0], shuffled_data[1])
    assert strictly_increasing(w)
    assert strictly_increasing(f)
