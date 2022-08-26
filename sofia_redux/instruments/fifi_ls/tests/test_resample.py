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
    import (resample, combine_files, get_grid_info, extract_info_from_file,
            generate_exposure_map, make_hdul, rbf_mean_combine,
            analyze_input_files, normalize_spherical_coordinates,
            cleanup_scan_reduction)


class TestResample(FIFITestCase):

    def test_extract_info_from_file(self):

        with pytest.raises(ValueError) as err:
            extract_info_from_file(None)
        assert 'Could not read file' in str(err.value)

        # Check flip sign
        filename = get_wsh_files()[0]
        hdul = fits.open(filename)
        hdul[0].header['DATE-OBS'] = '2020-01-01T12:00:00'
        assert not extract_info_from_file(hdul)['flip_sign']
        hdul = fits.open(filename)
        hdul[0].header['DATE-OBS'] = '2010-01-01T12:00:00'
        assert extract_info_from_file(hdul)['flip_sign']
        hdul = fits.open(filename)
        del hdul[0].header['DATE-OBS']
        assert not extract_info_from_file(hdul)['flip_sign']

        # Simulate OTF
        hdul = fits.open(filename)
        otf_shape = (15, 25, 16)
        for key in ['FLUX', 'LAMBDA', 'STDDEV', 'UNCORRECTED_FLUX',
                    'UNCORRECTED_STDDEV', 'UNCORRECTED_LAMBDA']:
            hdul[key].data = np.zeros(otf_shape)
        info = extract_info_from_file(hdul)
        for i_key in ['flux', 'u_flux', 'error', 'u_error', 'wave', 'u_wave']:
            assert info[i_key].shape == otf_shape
        assert isinstance(info['atran'], np.ndarray)

        hdul = fits.open(filename)
        for key in ['UNCORRECTED_FLUX', 'UNCORRECTED_LAMBDA',
                    'UNSMOOTHED_ATRAN']:
            del hdul[key]
        info = extract_info_from_file(hdul)
        for i_key in ['u_flux', 'u_error', 'u_wave', 'atran']:
            assert info[i_key] is None

    def test_analyze_input_files(self):
        files = get_wsh_files()
        filename = files[0]

        with pytest.raises(ValueError) as err:
            analyze_input_files(None)
        assert 'Could not read file' in str(err.value)

        info = analyze_input_files(filename)
        assert not info['otf']
        assert info['nonsidereal_values']
        assert not info['definite_nonsidereal']
        assert info['can_interpolate']
        assert info['uncorrected']
        assert isinstance(info['atran'], np.ndarray)

        hdul = fits.open(filename)
        hdul['FLUX'].data = np.zeros((15, 25, 16))
        hdul[0].header['NAIF_ID'] = 'foo'
        hdul[0].header['OBSLAM'] = 1.0
        hdul[0].header['DLAM_MAP'] = 1.0
        del hdul['UNCORRECTED_FLUX']
        del hdul['UNSMOOTHED_ATRAN']
        info = analyze_input_files(hdul)
        hdul.close()
        assert info['otf']
        assert not info['nonsidereal_values']
        assert info['definite_nonsidereal']
        assert not info['can_interpolate']
        assert not info['uncorrected']
        assert info['atran'] is None

    def test_normalize_spherical_coordinates(self):
        x, y = np.meshgrid(np.arange(-2, 3, dtype=float),
                           np.arange(-2, 3, dtype=float))
        angles = [15, 105, 15]
        headers = [fits.Header(), fits.Header(), fits.Header()]
        beta = [0, 1 / 3600, 0]
        lam = [0, 2 / 3600, 0]

        d = {}
        for i, (a, b, l) in enumerate(zip(angles, beta, lam)):
            d[i] = {'sky_angle': a, 'obs_bet': b, 'obs_lam': l,
                    'xs': x.copy(), 'ys': y.copy(), 'header': headers[i]}

        normalize_spherical_coordinates(d)
        assert np.allclose(d[0]['lon'], x) and np.allclose(d[0]['lat'], y)
        assert np.allclose(d[2]['lon'], x) and np.allclose(d[2]['lat'], y)
        assert d[0]['header']['SKY_ANGL'] == 15
        assert d[1]['header']['SKY_ANGL'] == 15
        assert d[2]['header']['SKY_ANGL'] == 15
        assert d[1]['header']['DLAM_MAP'] == 2
        assert d[1]['header']['DBET_MAP'] == 1

        assert np.allclose(d[1]['lon'],
                           [[-0.33, -0.33, -0.33, -0.33, -0.33],
                            [0.67, 0.67, 0.67, 0.67, 0.67],
                            [1.67, 1.67, 1.67, 1.67, 1.67],
                            [2.67, 2.67, 2.67, 2.67, 2.67],
                            [3.67, 3.67, 3.67, 3.67, 3.67]],
                           atol=1e-2)
        assert np.allclose(d[1]['lat'],
                           [[0.52, -0.48, -1.48, -2.48, -3.48],
                            [0.52, -0.48, -1.48, -2.48, -3.48],
                            [0.52, -0.48, -1.48, -2.48, -3.48],
                            [0.52, -0.48, -1.48, -2.48, -3.48],
                            [0.52, -0.48, -1.48, -2.48, -3.48]],
                           atol=1e-2)

    def test_success(self):
        files = get_wsh_files()
        result = resample(files, write=False)
        assert isinstance(result, fits.HDUList)
        assert result[0].header['PRODTYPE'] == 'resampled'

        for extname in ['FLUX', 'ERROR',
                        'UNCORRECTED_FLUX', 'UNCORRECTED_ERROR',
                        'WAVELENGTH', 'XS', 'YS', 'TRANSMISSION',
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
                assert np.nanmedian(interp[i].data
                                    ) > np.nanmedian(rsmp[i].data)
            elif i in [5, 6, 7, 8, 9, 10, 11]:
                # same for x, y, w, transmission, response, exposure map
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
        keys = ['PRIMEHEAD', 'method', 'RA', 'DEC', 'FLUX', 'ERROR',
                'WAVE', 'UNCORRECTED_FLUX', 'UNCORRECTED_ERROR',
                'UNCORRECTED_WAVE', 'UNSMOOTHED_TRANSMISSION', 'SAMPLES']
        for key in keys:
            assert key in default
            if key == 'PRIMEHEAD':
                assert isinstance(default[key], fits.Header)
            elif key == 'method':
                # always interp for a single file
                assert default[key] == 'interpolate'
            else:
                assert isinstance(default[key], (list, np.ndarray))

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
                assert isinstance(uncal_fit[key], (list, np.ndarray))

        # multiple files, but dlam/dbet are all zero
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul[0].header['DLAM_MAP'] = 0.0
            hdul[0].header['DBET_MAP'] = 0.0
            inp.append(hdul)
        dmap_zero = combine_files(inp)
        assert dmap_zero['method'] == 'interpolate'

        files = get_wsh_files()
        inp = []
        for i, f in enumerate(files):
            hdul = fits.open(f)
            if i > 0 and 'UNCORRECTED_FLUX' in hdul:
                del hdul['UNCORRECTED_FLUX']
            inp.append(hdul)
        bad_uncor = combine_files(inp)
        assert bad_uncor.get('UNCORRECTED_FLUX') is None

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
        # w,x,y output size are increased for pix_size / 2
        assert np.all(rshape[1:] > dshape[1:])

        # test large file
        hdul = fits.open(files[0])
        hdul['RA'].data *= 100.0
        too_big = combine_files([hdul])
        result = get_grid_info(too_big)
        assert result is None
        capt = capsys.readouterr()
        assert 'Spatial range too large' in capt.err

        hdul = fits.open(files[0])
        combined = combine_files([hdul])
        combined['PRIMEHEAD']['SKY_ANGL'] = 10.0
        combined['PRIMEHEAD']['DET_ANGL'] = 5.0
        # check target RA/DEC/WAVE
        target_ra = np.min([x.min() for x in combined['RA']])  # hourangle
        target_dec = np.min([x.min() for x in combined['DEC']])  # degree
        target_wave = np.min([x.min() for x in combined['WAVE']])  # um
        grid_info = get_grid_info(combined, target_x=target_ra,
                                  target_y=target_dec,
                                  target_wave=target_wave,
                                  detector_coordinates=False)
        wcs = grid_info['wcs'].to_header()
        assert np.isclose(wcs['CRVAL1'], target_ra * 15)
        assert np.isclose(wcs['CRVAL2'], target_dec)
        assert np.isclose(wcs['CRVAL3'], target_wave * 1e-6)
        assert wcs['CDELT1'] < 0  # east to west
        assert grid_info['delta'][2] < 0
        assert grid_info['det_angle'] == 0

        # Check detector coordinates rotation propagation
        combined['PRIMEHEAD']['SKY_ANGL'] = 0
        grid_info = get_grid_info(combined, target_x=0,
                                  target_y=0,
                                  target_wave=target_wave,
                                  detector_coordinates=True)
        wcs = grid_info['wcs'].to_header()
        assert wcs['CDELT1'] > 0
        assert grid_info['delta'][2] > 0
        assert grid_info['det_angle'].value == -5

    def test_exposure_map(self):
        files = [get_wsh_files()[0]]
        combined = combine_files(files)
        grid = get_grid_info(combined)

        # det angle non zero -- will have some zero pixels in the map
        default = generate_exposure_map(combined, grid)
        assert np.any(default == 0)
        assert np.max(default) == 2

        # Test vertical lines in the coordinates by using a regular x,y grid
        ndat, nspax = combined['FLUX'][0].shape
        yg, xg = np.indices((5, 5)).reshape((2, 25))
        xg += 5
        yg += 10
        x = np.empty((ndat, nspax))
        y = np.empty((ndat, nspax))
        x[...], y[...] = xg[None], yg[None]
        grid['coordinates'][:2] = x.ravel(), y.ravel()
        exposure = generate_exposure_map(combined, grid)

        # Check all planes are equal
        assert np.allclose(exposure, exposure[0][None])
        # check single exposure
        plane = exposure[0]
        ind_y, ind_x = np.nonzero(plane)
        x0, x1 = np.min(ind_x), np.max(ind_x)
        y0, y1 = np.min(ind_y), np.max(ind_y)

        # Check everything inside the coordinate bounds is filled
        filled = plane[y0:y1 + 1, x0:x1 + 1]
        assert np.allclose(filled, 2)

        # Check everything outside the coordinate bounds is zero
        filled.fill(0)
        assert np.allclose(plane, 0)

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
            return_value=[(np.array([False]), (0, 1), (0, 1))])
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
        expected = ['Jy/pixel', 'Jy/pixel', 'Jy/pixel', 'Jy/pixel',
                    'Jy/pixel', 'um', 'degree', 'degree', '', 'adu/(s Hz Jy)',
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

        wcs = grid_info['wcs']
        wcs_info = wcs.to_header()
        # Check reference pixel is at an integer location
        cp = np.asarray([wcs_info[f'CRPIX{i}'] for i in range(1, 4)])
        assert np.allclose(cp, cp.astype(int))

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

    def test_resample_detector_coordinates(self, capsys):
        filename = get_wsh_files()[0]
        hdul = fits.open(filename)
        h = hdul[0].header
        h['NAIF_ID'] = 'foo'
        _ = resample([hdul], detector_coordinates=None)
        capt = capsys.readouterr()
        assert "Resampling using detector coordinates: nonsidereal" in capt.out
        hdul.close()
        hdul = fits.open(filename)
        h = hdul[0].header
        if 'NAIF_ID' in h:  # pragma: no cover
            del h['NAIF_ID']
        h['OBSLAM'] = 0.0
        h['OBSBET'] = 0.0
        _ = resample([hdul], detector_coordinates=None)
        capt = capsys.readouterr()
        assert "possible non-sidereal observation" in capt.out
        hdul.close()
        hdul = fits.open(filename)
        h = hdul[0].header
        if 'NAIF_ID' in h:  # pragma: no cover
            del h['NAIF_ID']
        h['OBSLAM'] = 0.01
        h['OBSBET'] = 0.01
        _ = resample([hdul], detector_coordinates=None)
        capt = capsys.readouterr()
        assert "Resampling using equatorial coordinates" in capt.out

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
        assert result[1].data.shape == (55, 33, 39)

        # test pixel size override
        result = resample(files, write=False, oversample=None,
                          spectral_size=0.05, spatial_size=1.0)
        capt = capsys.readouterr()
        assert 'Output spectral pixel scale: 0.05' in capt.out
        assert 'Output spatial pixel scale: 1.0' in capt.out
        assert 'Spectral oversample: 3' in capt.out
        assert 'Spatial oversample: 11' in capt.out
        assert result[1].data.shape == (23, 76, 93)

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
        assert result[1].data.shape == (23, 76, 93)

        # test blue files
        inp = []
        for f in files:
            hdul = fits.open(f)
            hdul[0].header['CHANNEL'] = 'BLUE'
            inp.append(hdul.copy())
        result = resample(inp, write=False, oversample=None,
                          spectral_size=None, spatial_size=None)
        assert result[1].data.shape == (16, 39, 47)
        result = resample(inp, write=False, oversample=None,
                          spectral_size=0.05, spatial_size=1.0)
        assert result[1].data.shape == (23, 76, 93)

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

        # Do a diagonal scan in 0.2" increments (RA and DEC)
        ra = np.zeros((nramp, nw, 25))
        dec = np.zeros_like(ra)
        ra[...] = hdul['RA'].data[None].copy()
        dec[...] = hdul['DEC'].data[None].copy()
        drift = np.arange(nramp)[:, None, None] / (5 * 3600)
        ra += drift / 15  # to hour angle
        dec += drift

        hdul['RA'].data = ra
        hdul['DEC'].data = dec

        result = resample([hdul.copy()], write=False, outdir=None,
                          spatial_size=1.0, spectral_size=0.1,
                          detector_coordinates=False)
        assert result['FLUX'].data.shape == (12, 69, 67)
        assert result['ERROR'].data.shape == (12, 69, 67)
        assert result['UNCORRECTED_FLUX'].data.shape == (12, 69, 67)
        assert result['UNCORRECTED_ERROR'].data.shape == (12, 69, 67)

        result2 = resample([hdul.copy()], write=False, outdir=None,
                           spatial_size=1.0, spectral_size=0.1,
                           interp=True, detector_coordinates=False)
        # same size as resampled
        assert result['FLUX'].data.shape == (12, 69, 67)
        assert result['ERROR'].data.shape == (12, 69, 67)
        assert result['UNCORRECTED_FLUX'].data.shape == (12, 69, 67)
        assert result['UNCORRECTED_ERROR'].data.shape == (12, 69, 67)

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

    def test_large_data(self, capsys, mocker):
        files = get_wsh_files()

        # mock resample to return faster
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.Resample.'
            '__call__', return_value=(10, 10, 10))

        # mock data size estimates
        class MockMem:
            total = 1000
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.psutil.virtual_memory',
            return_value=MockMem)
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.Resample.'
            'estimate_max_bytes', return_value=10)

        # test with small data, check_memory=True and False should be same
        resample(files, write=False, check_memory=True)
        capt = capsys.readouterr()
        assert 'Maximum expected memory needed: 10.00 B' in capt.out
        assert 'Splitting data tree into blocks.' not in capt.out
        resample(files, write=False, check_memory=False)
        capt = capsys.readouterr()
        assert 'Maximum expected memory needed: 10.00 B' in capt.out
        assert 'Splitting data tree into blocks.' not in capt.out

        # test with large data, check_memory=True
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.Resample.'
            'estimate_max_bytes', return_value=10000)
        resample(files, write=False, check_memory=True)
        capt = capsys.readouterr()
        assert 'Maximum expected memory needed: 9.77 kB' in capt.out
        assert 'Splitting data tree into blocks.' not in capt.out

        # test with large data, check_memory=False:
        # large_data is explicitly set
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.Resample.'
            'estimate_max_bytes', return_value=10000)
        resample(files, write=False, check_memory=False)
        capt = capsys.readouterr()
        assert 'Maximum expected memory needed: 9.77 kB' in capt.out
        assert 'Splitting data tree into blocks.' in capt.out

        # test with borderline data: 1/10 of memory
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.resample.Resample.'
            'estimate_max_bytes', return_value=100)
        resample(files, write=False, check_memory=False)
        capt = capsys.readouterr()
        assert 'Maximum expected memory needed: 100.00 B' in capt.out
        assert 'Splitting data tree into blocks.' in capt.out

    def test_cleanup_scan_reduction(self, tmpdir):
        directory = tmpdir.mkdir('test_cleanup')
        reduction_file = str(directory.join('reduction.p'))
        u_reduction_file = str(directory.join('u_reduction.p'))
        for filename in [reduction_file, u_reduction_file]:
            with open(filename, 'w') as f:
                print('hello', file=f)
        combined = {'reduction_file': reduction_file,
                    'uncorrected_reduction_file': u_reduction_file}
        directory = str(directory)
        cleanup_scan_reduction(combined)
        assert not os.path.isdir(directory)
