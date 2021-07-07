# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import dill as pickle
import numpy as np
import pytest

from sofia_redux.instruments.fifi_ls.get_resolution import get_resolution
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_wsh_files, get_scm_files, get_cal_files
from sofia_redux.instruments.fifi_ls.resample \
    import (resample, combine_files, get_grid_info,
            generate_exposure_map, make_hdul, rbf_mean_combine)


class TestResample(FIFITestCase):

    def test_success(self):
        files = get_wsh_files()
        result = resample(files, write=False)
        assert isinstance(result, fits.HDUList)
        assert result[0].header['PRODTYPE'] == 'resampled'
        for extname in ['FLUX', 'ERROR',
                        'UNCORRECTED_FLUX', 'UNCORRECTED_ERROR',
                        'WAVELENGTH', 'X', 'Y', 'TRANSMISSION',
                        'RESPONSE', 'EXPOSURE_MAP',
                        'UNSMOOTHED_TRANSMISSION']:
            assert extname in result
        assert not np.all(np.isnan(result[1].data))

    def test_write(self, tmpdir):
        files = get_wsh_files()
        result = resample(files, write=True, outdir=str(tmpdir))
        assert os.path.isfile(result)

    def test_interp(self):
        files = get_wsh_files()
        rsmp = resample(files, write=False, edge_threshold=0, order=(2, 2, 0))
        interp = resample(files, write=False, interp=True, order=0)

        # data should be similar for all extensions
        for i in range(len(rsmp)):
            if i == 0:
                continue
            elif i == 2 or i == 4:
                # error sum should be higher for interp data
                assert np.nansum(interp[i].data) > np.nansum(rsmp[i].data)
            elif i == 5 or i == 6 or i == 7:
                # same for x, y, wvlen
                assert np.allclose(rsmp[i].data, interp[i].data)
            else:
                # mean should be in the ballpark for flux
                assert np.allclose(np.nanmean(rsmp[i].data),
                                   np.nanmean(interp[i].data),
                                   rtol=5)

    @pytest.mark.parametrize('algorithm', ['shaped', 'scaled'])
    def test_adaptive(self, algorithm):
        files = get_wsh_files()

        # default smoothing: spatial sigma = 1.0 FWHM for the channel
        rsmp = resample(files, write=False, smoothing=(1.0, 1.0, 0.25),
                        adaptive_algorithm=None)

        # adaptive smoothing requires beam sigma = 2 sqrt(2 * log(2))
        sigma = 2 * np.sqrt(2 * np.log(2))
        adapt = resample(files, write=False, smoothing=(sigma, sigma, 0.25),
                         adaptive_algorithm=algorithm,
                         adaptive_threshold=(1, 1, 0))

        for i in range(len(rsmp)):
            if i == 0:
                # first extension is empty
                continue
            elif i < 5:
                # data shape is the same for flux and errors
                assert rsmp[i].data.shape == adapt[i].data.shape
            else:
                # data is same for x, y, wvlen, etc
                assert np.allclose(rsmp[i].data, adapt[i].data)

    def test_combine_files_error(self, capsys):
        # bad file
        with pytest.raises(ValueError):
            combine_files(['bad_file.fits'])
        capt = capsys.readouterr()
        assert 'Could not read file' in capt.err

    def test_combine_files_method(self):
        # default files
        files = [get_wsh_files()[0]]
        default = combine_files(files)
        assert isinstance(default, dict)
        keys = ['PRIMEHEAD', 'method', 'X', 'Y', 'FLUX', 'ERROR',
                'WAVELENGTH', 'UNCORRECTED_FLUX', 'UNCORRECTED_ERROR',
                'UNCORRECTED_LAMBDA', 'UNSMOOTHED_TRANSMISSION']
        for key in keys:
            assert key in default
            if key == 'PRIMEHEAD':
                assert isinstance(default[key], fits.Header)
            elif key == 'method':
                # always interp for a single file
                assert default[key] == 'interpolate'
            else:
                assert isinstance(default[key], np.ndarray)

        # multiple uncal files
        files = get_scm_files()
        uncal_fit = combine_files(files)
        assert isinstance(uncal_fit, dict)
        for key in keys:
            if 'UN' in key:
                assert key not in uncal_fit
                continue
            else:
                assert key in uncal_fit
            if key == 'PRIMEHEAD':
                assert isinstance(uncal_fit[key], fits.Header)
            elif key == 'method':
                # fit for a multiple files with dlam/dbet
                assert uncal_fit[key] == 'resample'
            else:
                assert isinstance(uncal_fit[key], np.ndarray)

        # multiple files, but dlam/dbet are all zero
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul[0].header['DLAM_MAP'] = 0.0
            hdul[0].header['DBET_MAP'] = 0.0
            inp.append(hdul)
        dmap_zero = combine_files(inp)
        assert dmap_zero['method'] == 'interpolate'

    def test_combine_files_rotate(self):
        # check for appropriate behavior when SKY_ANGL is not zero
        files = get_wsh_files()
        default = combine_files([files[0]])

        # set a non-zero angle
        angle = 45.0
        hdul = fits.open(files[0])
        hdul[0].header['SKY_ANGL'] = angle
        rotated = combine_files([hdul])

        # for one file, result is the same, except that sky_angl is set
        for key in default:
            if key == 'PRIMEHEAD':
                assert default[key]['SKY_ANGL'] == 0
                assert rotated[key]['SKY_ANGL'] == angle
            elif key == 'method':
                assert default[key] == rotated[key]
            else:
                assert np.allclose(default[key], rotated[key], equal_nan=True)

        # for multiple files, all are rotated to the first
        default = combine_files(files)
        inp = []
        for i, fname in enumerate(files):
            hdul = fits.open(fname)
            hdul[0].header['SKY_ANGL'] = angle + i * 10
            inp.append(hdul)
        rotated = combine_files(inp)

        for i in range(len(rotated['X'])):
            if i == 0:
                # first matches default
                assert np.allclose(rotated['X'][i], default['X'][i])
                assert np.allclose(rotated['Y'][i], default['Y'][i])
            else:
                # rest are rotated by angle delta
                xd = default['X'][i]
                yd = default['Y'][i]
                xs = rotated['X'][i]
                ys = rotated['Y'][i]
                a = np.deg2rad(-i * 10)
                assert np.allclose(xs, xd * np.cos(a) - yd * np.sin(a))
                assert np.allclose(ys, xd * np.sin(a) + yd * np.cos(a))

    def test_combine_files_offset(self):
        # check for appropriate behavior when there are
        # map center offsets between files
        files = get_wsh_files()

        # default: dlam_map and dbet_map match
        default = combine_files(files)

        inp = []
        for i, fname in enumerate(files):
            hdul = fits.open(fname)
            hdul[0].header['OBSLAM'] += i * 10 / 3600.
            hdul[0].header['OBSBET'] += i * 10 / 3600.
            inp.append(hdul)
        offset = combine_files(inp)

        for i in range(len(offset['X'])):
            if i == 0:
                # first matches default
                assert np.allclose(offset['X'][i], default['X'][i])
                assert np.allclose(offset['Y'][i], default['Y'][i])
            else:
                # rest are offset by difference in base position
                xd = default['X'][i]
                yd = default['Y'][i]
                xs = offset['X'][i]
                ys = offset['Y'][i]
                lamoff = -i * 10
                betoff = i * 10
                assert np.allclose(xs, xd + lamoff)
                assert np.allclose(ys, yd + betoff)

    def test_grid_info(self, capsys, mocker):
        files = [get_wsh_files()[0]]
        combined = combine_files(files)

        # mock resolution to return constant value for spatial grid
        def mock_res(header, wmean=None, spatial=False):
            if spatial and header['CHANNEL'] == 'BLUE':
                return 5.0
            elif spatial and header['CHANNEL'] == 'RED':
                return 10.0
            else:
                return get_resolution(header, wmean, spatial)
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.get_resolution',
            mock_res)

        # default value
        default = get_grid_info(combined)

        # test oversample parameter
        result = get_grid_info(combined, oversample=(5.0, 8.0))
        assert np.allclose(result['grid'][0], default['grid'][0])

        # test blue file
        hdul = fits.open(files[0])
        hdul[0].header['CHANNEL'] = 'BLUE'
        blue = combine_files([hdul])
        result = get_grid_info(blue)
        rshape = np.array(result['shape'])
        dshape = np.array(default['shape'])
        # input size is same
        assert np.allclose(rshape[0], dshape[0])
        # x/y output size are doubled for pix_size / 2
        assert np.allclose(rshape[2:], dshape[2:] * 2, atol=1)

        # test large file
        hdul = fits.open(files[0])
        hdul['XS'].data *= 100.0
        too_big = combine_files([hdul])
        result = get_grid_info(too_big)
        assert result is None
        capt = capsys.readouterr()
        assert 'Spatial range too large' in capt.err

    def test_exposure_map(self):
        files = [get_wsh_files()[0]]
        combined = combine_files(files)
        grid = get_grid_info(combined)

        # det angle non zero -- will have some zero pixels in the map
        default = generate_exposure_map(combined, grid)
        assert np.any(default == 0)

        # set det angle to zero -- all pixels are 2, except for first frame
        hdul = fits.open(files[0])
        hdul[0].header['DET_ANGL'] = 0
        combined = combine_files([hdul])
        grid = get_grid_info(combined)
        unrot = generate_exposure_map(combined, grid)
        assert np.all(unrot[1:] == 2)

    def test_rbf_parameters(self):
        files = [get_wsh_files()[0]]
        combined = combine_files(files)
        grid_info = get_grid_info(combined)

        # default
        rbf_mean_combine(combined, grid_info)
        default = combined['GRID_FLUX']

        # test 3D parameters as if from resample

        # same values
        rbf_mean_combine(combined, grid_info, window=(3, 3, 0.5),
                         smoothing=(2, 2, 0.25), order=(3, 3, 0))
        assert np.allclose(combined['GRID_FLUX'], default, equal_nan=True)

        # different values
        rbf_mean_combine(combined, grid_info, window=(3, 3, 3),
                         smoothing=(2, 2, 2), order=(3, 3, 3))
        assert not np.allclose(combined['GRID_FLUX'], default, equal_nan=True)

    def test_rbf_errors(self, mocker, capsys):
        files = [get_wsh_files()[0]]
        combined = combine_files(files)
        grid_info = get_grid_info(combined)

        # math error in resampler sets pixels to NaN
        def bad_function(*args, **kwargs):
            raise ValueError('test')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.Resample',
            bad_function)
        rbf_mean_combine(combined, grid_info)
        assert np.all(np.isnan(combined['GRID_FLUX']))
        capt = capsys.readouterr()
        assert 'Math error' in capt.out

        # exposure map returns no good idx
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.generate_exposure_map',
            return_value=np.array([False]))
        rbf_mean_combine(combined, grid_info)
        assert np.all(np.isnan(combined['GRID_FLUX']))
        capt = capsys.readouterr()
        assert 'No good values' in capt.out

    def test_make_hdul(self, mocker, capsys):
        # resample a single file with defaults
        files = [get_wsh_files()[0]]
        combined = combine_files(files)
        grid_info = get_grid_info(combined)
        rbf_mean_combine(combined, grid_info)
        cc = pickle.dumps(combined)

        # set the rotation explicitly
        combined['PRIMEHEAD']['SKY_ANGL'] = 0

        default = make_hdul(combined, grid_info)
        assert isinstance(default, fits.HDUList)

        # check bunit in all extensions, assuming
        # all extensions are present
        expected = ['Jy/pixel', 'Jy/pixel', 'Jy/pixel', 'Jy/pixel', 'Jy/pixel',
                    'um', 'arcsec', 'arcsec', '', 'adu/(s Hz Jy)',
                    '', '']
        for i, ext in enumerate(default):
            assert ext.header.get('BUNIT') == expected[i]

        # check spexlwid in primary -- should match cdelt3
        assert np.allclose(default[0].header['SPEXLWID'],
                           default[0].header['CDELT3'],)

        # check for SPECSYS key
        assert default[0].header['SPECSYS'] == 'BARYCENT'

        # apply rotation
        combined = pickle.loads(cc)
        angle = 45.0
        combined['PRIMEHEAD']['SKY_ANGL'] = angle

        # expected value is rotated -- check crval2
        # (crval1 has a cos(dec) that makes it harder to check)
        xref, yref = grid_info['xout'][0], grid_info['yout'][0]
        nyref = (xref * np.sin(np.deg2rad(angle))) + \
                (yref * np.cos(np.deg2rad(angle)))

        rotated = make_hdul(combined, grid_info)
        diff = rotated[0].header['CRVAL2'] - default[0].header['CRVAL2']
        assert np.allclose(diff, (nyref - yref) / 3600.)
        assert not np.allclose(rotated[0].header['CRVAL1'],
                               default[0].header['CRVAL1'],
                               atol=1e-4, rtol=0)
        assert np.allclose(rotated[0].header['CRPIX1'],
                           default[0].header['CRPIX1'])
        assert np.allclose(rotated[0].header['CRPIX2'],
                           default[0].header['CRPIX2'])

        # mock an error in interpolation -- will leave set atran
        # and response to 1, but leave unsmoothed atran
        def bad_interp(*args, **kwargs):
            raise ValueError('test')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.np.interp',
            bad_interp)
        no_atran = make_hdul(combined, grid_info)
        assert np.allclose(no_atran['TRANSMISSION'].data, 1)
        assert np.allclose(no_atran['RESPONSE'].data, 1)
        assert not np.allclose(no_atran['UNSMOOTHED_TRANSMISSION'].data[1], 1)
        capt = capsys.readouterr()
        assert 'Problem in interpolation' in capt.err

    def test_uncalib(self):
        # check that uncalibrated files still resample okay
        files = get_scm_files()

        # with local fits
        result = resample(files, write=False)
        assert isinstance(result, fits.HDUList)

        # bunit is not Jy
        assert result[1].header['BUNIT'] == 'ADU/(s Hz)'
        assert result[2].header['BUNIT'] == 'ADU/(s Hz)'
        assert 'UNCORRECTED_FLUX' not in result
        assert 'UNCORRECTED_ERROR' not in result

        # and also with interp
        result = resample(files[0], write=False, interp=True)
        assert isinstance(result, fits.HDUList)
        assert result[1].header['BUNIT'] == 'ADU/(s Hz)'

        # check that un-wave-shifted files also resample okay,
        # but BUNIT is Jy
        files = get_cal_files()
        result = resample(files, write=False)
        assert isinstance(result, fits.HDUList)
        assert result[1].header['BUNIT'] == 'Jy/pixel'

    def test_resample_failure(self, capsys, mocker):
        # bad files
        result = resample(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input" in capt.err

        # single file, bad outdir
        files = get_wsh_files()
        result = resample(files[0], write=False, outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert "Output directory badval does not exist" in capt.err

        # problem in interp
        def mock_error(*args, **kwargs):
            raise ValueError('test')
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.rbf_mean_combine',
            mock_error)
        result = resample(files[0], write=False, interp=True)
        assert result is None
        capt = capsys.readouterr()
        assert 'test' in capt.err

        # problem in fit
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.local_surface_fit',
            mock_error)
        result = resample(files, write=False, interp=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'test' in capt.err

        # problem in gridding
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.get_grid_info',
            return_value=None)
        result = resample(files[0], write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem in grid calculation' in capt.err

    def test_resample_nancube(self, capsys):
        # test for the case where all input data is NaN -- this
        # can happen after telluric correction in particular wavelength
        # ranges

        # set all corrected data to NaN, leave uncorrected data
        files = get_cal_files()
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul[1].data[:] = np.nan
            inp.append(hdul)

        # step should succeed but warn the user
        result = resample(inp, write=False)
        assert isinstance(result, fits.HDUList)

        # output flux is all NaN, but matches shape of uncorrected flux
        assert np.all(np.isnan(result['FLUX'].data))
        assert not np.all(np.isnan(result['UNCORRECTED_FLUX'].data))
        assert result['FLUX'].data.shape \
            == result['UNCORRECTED_FLUX'].data.shape

        capt = capsys.readouterr()
        assert 'Primary flux cube contains only NaN' in capt.err
        assert 'Uncorrected flux cube contains only NaN' not in capt.err

        # now set uncorrected flux to NaN too, for good measure
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul['FLUX'].data[:] = np.nan
            hdul['UNCORRECTED_FLUX'].data[:] = np.nan
            inp.append(hdul)
        result = resample(inp, write=False)
        assert np.all(np.isnan(result['UNCORRECTED_FLUX'].data))
        capt = capsys.readouterr()
        assert 'Uncorrected flux cube contains only NaN' in capt.err

        # also check for warning in interpolate case
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul['FLUX'].data[:] = np.nan
            hdul['UNCORRECTED_FLUX'].data[:] = np.nan
            inp.append(hdul)
        result = resample(inp, write=False, interp=True)
        assert np.all(np.isnan(result['FLUX'].data))
        assert np.all(np.isnan(result['UNCORRECTED_FLUX'].data))
        capt = capsys.readouterr()
        assert 'Primary flux cube contains only NaN' in capt.err
        assert 'Uncorrected flux cube contains only NaN' in capt.err

    def test_output_pixel_size(self, capsys, mocker):
        files = get_wsh_files()

        # test default: oversample (5.0, 8.0) assumed
        result = resample(files, write=False, oversample=None,
                          spectral_size=None, spatial_size=None)
        capt = capsys.readouterr()
        assert 'Output spectral pixel scale: 0.019' in capt.out
        assert 'Output spatial pixel scale: 2.38' in capt.out
        assert 'Spectral oversample: 8' in capt.out
        assert 'Spatial oversample: 5' in capt.out
        assert result[1].data.shape == (56, 32, 39)

        # test pixel size override
        result = resample(files, write=False, oversample=None,
                          spectral_size=0.05, spatial_size=1.0)
        capt = capsys.readouterr()
        assert 'Output spectral pixel scale: 0.05' in capt.out
        assert 'Output spatial pixel scale: 1.0' in capt.out
        assert 'Spectral oversample: 3' in capt.out
        assert 'Spatial oversample: 11' in capt.out
        assert result[1].data.shape == (23, 76, 92)

        # mock resolution to return a different value:
        # confirm that oversample returns different value, but
        # pixel size doesn't
        def mock_res(header, wmean=None, spatial=False):
            if spatial and header['CHANNEL'] == 'BLUE':
                return 10.0
            elif spatial and header['CHANNEL'] == 'RED':
                return 20.0
            else:
                return 200
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.get_resolution',
            mock_res)

        # test red file
        result = resample(files, write=False, oversample=None,
                          spectral_size=None, spatial_size=None)
        assert result[1].data.shape == (16, 20, 24)
        result = resample(files, write=False, oversample=None,
                          spectral_size=0.05, spatial_size=1.0)
        assert result[1].data.shape == (23, 76, 92)

        # test blue files
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul[0].header['CHANNEL'] = 'BLUE'
            inp.append(hdul.copy())
        result = resample(inp, write=False, oversample=None,
                          spectral_size=None, spatial_size=None)
        assert result[1].data.shape == (16, 38, 47)
        result = resample(inp, write=False, oversample=None,
                          spectral_size=0.05, spatial_size=1.0)
        assert result[1].data.shape == (23, 76, 92)

    def test_otf_resample(self):
        files = get_wsh_files()
        hdul = fits.open(files[0])

        # mock OTF data: should have 3D flux, stddev, xs, ys
        nramp = 15
        nw = hdul['FLUX'].data.shape[0]
        flux = np.repeat(hdul['FLUX'].data.reshape(1, nw, 25),
                         nramp, axis=0)
        stddev = np.repeat(hdul['STDDEV'].data.reshape(1, nw, 25),
                           nramp, axis=0)
        hdul['FLUX'].data = flux
        hdul['STDDEV'].data = stddev
        hdul['UNCORRECTED_FLUX'].data = flux.copy() * 0.5
        hdul['UNCORRECTED_STDDEV'].data = stddev.copy() * 0.5

        # linear scan offsets: should end up with nramp x nramp cube
        # at spatial size 1.0
        xs = np.zeros((nramp, nw, 25))
        ys = np.zeros((nramp, nw, 25))
        xs += np.arange(nramp, dtype=float)[:, None, None]
        ys += np.arange(nramp, dtype=float)[:, None, None]
        # add small faked offsets for pixel positions
        xs += ((np.arange(25, dtype=float) - 12.5) / 12.5 / 10)[None, None, :]
        ys += ((np.arange(25, dtype=float) - 12.5) / 12.5 / 10)[None, None, :]

        hdul['XS'].data = xs
        hdul['YS'].data = ys

        result = resample([hdul.copy()], write=False, outdir=None,
                          spatial_size=1.0, spectral_size=0.1)
        assert result['FLUX'].data.shape == (12, 15, 15)
        assert result['ERROR'].data.shape == (12, 15, 15)
        assert result['UNCORRECTED_FLUX'].data.shape == (12, 15, 15)
        assert result['UNCORRECTED_ERROR'].data.shape == (12, 15, 15)

        result2 = resample([hdul.copy()], write=False, outdir=None,
                           spatial_size=1.0, spectral_size=0.1,
                           interp=True)
        # same size as resampled
        assert result2['FLUX'].data.shape == (12, 15, 15)
        assert result2['ERROR'].data.shape == (12, 15, 15)
        assert result2['UNCORRECTED_FLUX'].data.shape == (12, 15, 15)
        assert result2['UNCORRECTED_ERROR'].data.shape == (12, 15, 15)

        # but different values (interp is not default for single
        # file if OTF)
        assert not np.allclose(result2['FLUX'].data, result['FLUX'].data,
                               equal_nan=True)

    def test_append_weights(self):
        files = [get_wsh_files()[0]]
        combined = combine_files(files)
        grid = get_grid_info(combined)
        # fake resample step
        combined['GRID_FLUX'] = np.zeros(10)
        combined['GRID_ERROR'] = np.zeros(10)
        combined['GRID_COUNTS'] = np.zeros(10)

        default = make_hdul(combined, grid, append_weights=False)
        assert 'IMAGE_WEIGHTS' not in default
        assert 'UNCORRECTED_IMAGE_WEIGHTS' not in default

        # test append weights with none available
        no_wts = make_hdul(combined, grid, append_weights=True)
        assert 'IMAGE_WEIGHTS' not in no_wts
        assert 'UNCORRECTED_IMAGE_WEIGHTS' not in no_wts

        # add weights - should now be stored
        combined['GRID_WEIGHTS'] = np.arange(10)
        with_wts = make_hdul(combined, grid, append_weights=True)
        assert 'IMAGE_WEIGHTS' in with_wts
        assert 'UNCORRECTED_IMAGE_WEIGHTS' not in with_wts
        combined['GRID_UNCORRECTED_WEIGHTS'] = np.arange(10) + 10
        with_wts = make_hdul(combined, grid, append_weights=True)
        assert 'IMAGE_WEIGHTS' in with_wts
        assert 'UNCORRECTED_IMAGE_WEIGHTS' in with_wts

        assert np.all(with_wts['IMAGE_WEIGHTS'].data == np.arange(10))
        assert np.all(with_wts['UNCORRECTED_IMAGE_WEIGHTS'].data
                      == np.arange(10) + 10)
