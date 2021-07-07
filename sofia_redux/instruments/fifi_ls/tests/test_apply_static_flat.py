# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, raw_testdata, get_xyc_files


class TestApplyStaticFlat(FIFITestCase):

    def test_flat_cache(self, tmpdir):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import clear_flat_cache, get_flat_from_cache, store_flat_in_cache

        tempdir = str(tmpdir.mkdir('test_apply_static_flat'))
        specfile = os.path.join(tempdir, 'specfile')
        spatfile = os.path.join(tempdir, 'spatfile')
        obsdate = 20200201

        with open(specfile, 'w') as f:
            print('this is the specfile', file=f)

        with open(spatfile, 'w') as f:
            print('this is the spatfile', file=f)

        spat_flat = spec_flat = spec_wave = spec_err = np.arange(10)

        filename = 'FLATFILE_header_value'
        files = specfile, spatfile, obsdate
        store = (specfile, spatfile, obsdate,
                 filename, spat_flat, spec_flat, spec_wave, spec_err)

        clear_flat_cache()
        assert get_flat_from_cache(*files) is None
        store_flat_in_cache(*store)

        # It should be in there now
        result = get_flat_from_cache(*files)
        assert result[0] == filename
        for r in result[1:]:
            assert np.allclose(r, spat_flat)

        # Check it's still in there
        assert get_flat_from_cache(*files) is not None

        # Check that different date is not there
        assert get_flat_from_cache(specfile, spatfile, obsdate + 1) is None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(specfile, 'w') as f:
            print('a modification', file=f)

        assert get_flat_from_cache(*files) is None

        # Store the data again
        store_flat_in_cache(*store)

        # Make sure it's there
        assert get_flat_from_cache(*files) is not None

        # Check clear works
        clear_flat_cache()
        assert get_flat_from_cache(*files) is None

        # Store then delete the spec file -- check that bad file
        # can't be retrieved
        store_flat_in_cache(*store)
        assert get_flat_from_cache(*files) is not None
        os.remove(specfile)
        assert get_flat_from_cache(*files) is None

    def test_bad_header(self, capsys):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import get_flat

        # start with empty header
        header = fits.Header()
        result = get_flat(header)
        capt = capsys.readouterr()
        assert result is None
        assert 'Cannot determine DATE-OBS' in capt.err

        # add invalid date; still missing others
        header['DATE-OBS'] = 'BADVAL'
        for key in ['CHANNEL', 'DICHROIC', 'G_ORD_B']:
            result = get_flat(header)
            capt = capsys.readouterr()
            assert result is None
            assert 'Header is missing {}'.format(key) in capt.err
            header[key] = 'TESTVAL'

        # now it has all others, will complain about bad date
        result = get_flat(header)
        capt = capsys.readouterr()
        assert result is None
        assert 'Invalid DATE-OBS' in capt.err

    def test_channel(self):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import get_flat

        # get a valid fifi header
        header = raw_testdata()[0].header

        # check red flat, d130
        header['CHANNEL'] = 'RED'
        header['DICHROIC'] = 130
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatr' in result[0].lower()
        assert 'spectralflatsr1d130' in result[0].lower()

        # check red flat, d105
        header['CHANNEL'] = 'RED'
        header['DICHROIC'] = 105
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatr' in result[0].lower()
        assert 'spectralflatsr1d105' in result[0].lower()

        # check blue 1 flat, d130
        header['CHANNEL'] = 'BLUE'
        header['DICHROIC'] = 130
        header['G_ORD_B'] = 1
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatb1' in result[0].lower()
        assert 'spectralflatsb1d130' in result[0].lower()

        # check blue 1 flat, d105
        header['CHANNEL'] = 'BLUE'
        header['DICHROIC'] = 105
        header['G_ORD_B'] = 1
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatb1' in result[0].lower()
        assert 'spectralflatsb1d105' in result[0].lower()

        # check blue 2 flat, d130
        header['CHANNEL'] = 'BLUE'
        header['DICHROIC'] = 130
        header['G_ORD_B'] = 2
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatb2' in result[0].lower()
        assert 'spectralflatsb2d130' in result[0].lower()

        # check blue 2 flat, d105
        header['CHANNEL'] = 'BLUE'
        header['DICHROIC'] = 105
        header['G_ORD_B'] = 2
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatb2' in result[0].lower()
        assert 'spectralflatsb2d105' in result[0].lower()

        # check blue 2 filter 1 flat, d130
        header['CHANNEL'] = 'BLUE'
        header['DICHROIC'] = 130
        header['G_ORD_B'] = 2
        header['G_FLT_B'] = 1
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatb2' in result[0].lower()
        assert 'spectralflatsb21d130' in result[0].lower()

        # check blue 2 filter 1 flat, d105
        header['CHANNEL'] = 'BLUE'
        header['DICHROIC'] = 105
        header['G_ORD_B'] = 2
        header['G_FLT_B'] = 1
        result = get_flat(header)
        assert result is not None
        assert 'spatialflatb2' in result[0].lower()
        assert 'spectralflatsb21d105' in result[0].lower()

        # also explicitly check blue 2 filter 2 - should
        # same as if FLT is not specified
        header['DICHROIC'] = 130
        header['G_FLT_B'] = 2
        result = get_flat(header)
        assert 'spectralflatsb2d130' in result[0].lower()
        header['DICHROIC'] = 105
        result = get_flat(header)
        assert 'spectralflatsb2d105' in result[0].lower()

        # and also the same for an invalid value
        header['G_FLT_B'] = -9999
        result = get_flat(header)
        assert 'spectralflatsb2d105' in result[0].lower()

    def test_missing_files(self, tmpdir, mocker, capsys):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import get_flat

        # get a valid fifi header
        header = raw_testdata()[0].header
        result = get_flat(header)
        assert result is not None

        # mock the data path to make sure flats aren't found
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.fifi_ls.__file__',
                     str(mock_file))
        result = get_flat(header)
        assert result is None
        capt = capsys.readouterr()
        assert 'Cannot locate spatial flat' in capt.err

        # make a spatial flat to check that spectral flat fails too
        os.makedirs(tmpdir.join('data', 'flat_files'))
        spatflat = tmpdir.join('data', 'flat_files', 'spatialFlatR.txt')
        spatflat.write('date test1\n10000000 1\n')
        result = get_flat(header)
        assert result is None
        capt = capsys.readouterr()
        assert 'Cannot locate spectral flat' in capt.err

        # make the spectral flat -- will then fail because date not found
        # in spatial flat
        specflat = tmpdir.join('data', 'flat_files',
                               'spectralFlatsR1D105.fits')
        specflat.write('test data\n')
        result = get_flat(header)
        assert result is None
        capt = capsys.readouterr()
        assert 'No spatial flat found' in capt.err

    def test_success(self):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import apply_static_flat
        files = get_xyc_files()

        # test on a filename
        result = apply_static_flat(
            files[0], write=False, outdir=None)
        assert isinstance(result, fits.HDUList)
        assert 'FLATFILE' in result[0].header

        # test on an HDUList
        hdul = fits.open(files[0])
        result2 = apply_static_flat(
            hdul, write=False, outdir=None)
        assert isinstance(result2, fits.HDUList)
        assert 'FLATFILE' in result2[0].header

        for i in range(hdul[0].header['NGRATING']):
            # flux and stddev should have same factors
            assert np.allclose(result[f'FLUX_G{i}'].data
                               / hdul[f'FLUX_G{i}'].data,
                               result[f'STDDEV_G{i}'].data
                               / hdul[f'STDDEV_G{i}'].data,
                               equal_nan=True)
            # ratio between input and output should be flat
            good = ~np.isnan(result2[f'FLUX_G{i}'].data)
            assert np.allclose(hdul[f'FLUX_G{i}'].data[good]
                               / result2[f'FLUX_G{i}'].data[good],
                               result2[f'FLAT_G{i}'].data[good])
            # check bunit
            assert result2[f'FLUX_G{i}'].header['BUNIT'] == 'adu/(Hz s)'
            assert result2[f'STDDEV_G{i}'].header['BUNIT'] == 'adu/(Hz s)'
            assert result2[f'LAMBDA_G{i}'].header['BUNIT'] == 'um'
            assert result2[f'XS_G{i}'].header['BUNIT'] == 'arcsec'
            assert result2[f'YS_G{i}'].header['BUNIT'] == 'arcsec'
            assert result2[f'FLAT_G{i}'].header['BUNIT'] == ''
            assert result2[f'FLATERR_G{i}'].header['BUNIT'] == ''

    def test_write(self, tmpdir):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import apply_static_flat
        files = get_xyc_files()
        result = apply_static_flat(
            files[0], write=True, outdir=str(tmpdir))
        failure = False
        if not isinstance(result, str) or not os.path.isfile(result):
            print("error: not a file (%s)" % repr(result))
            failure = True
        assert not failure

    def test_parallel(self, tmpdir):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import wrap_apply_static_flat
        files = get_xyc_files()
        result = wrap_apply_static_flat(files, outdir=str(tmpdir),
                                        allow_errors=False,
                                        jobs=-1, write=True)
        failure = False
        for f in result:
            if not isinstance(f, str) or not os.path.isfile(f):
                print("error: not a file (%s)" % repr(f))
                failure = True
        assert not failure

    def test_serial(self, tmpdir):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import wrap_apply_static_flat
        files = get_xyc_files()
        result = wrap_apply_static_flat(
            files, outdir=str(tmpdir),
            allow_errors=False, write=True)
        failure = False
        for f in result:
            if not isinstance(f, str) or not os.path.isfile(f):
                print("error: not a file (%s)" % repr(f))
                failure = True
        assert not failure

    def test_stripnans(self):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import stripnans
        x = np.array([10, 1, 10, 2, 10, 3, 10, 4], dtype=float)
        y = np.array([np.nan, 1, np.nan, 2, np.nan, 3, np.nan, 4], dtype=float)
        e = np.array([10, 1, 10, 2, 10, 3, 10, 4], dtype=float)
        expected = np.array([1, 2, 3, 4], dtype=float)

        e1, e2, e3 = stripnans(x, y, e)
        assert np.allclose(e1, expected)
        assert np.allclose(e2, expected)
        assert np.allclose(e3, expected)

    def test_calculate_flat(self):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import calculate_flat
        wave = np.repeat((np.arange(16, dtype=float) + 2).reshape(16, 1), 25,
                         axis=1)
        data = np.ones((10, 16, 25))
        var = np.ones((10, 16, 25))
        spatdata = (np.arange(25) + 1) / 25.0
        specdata = (np.arange(100 * 25 * 16) + 1).reshape((100, 25, 16)) \
            / (100 * 25 * 16)
        specwave = np.arange(100, dtype=float)
        specerr = np.ones_like(specdata)

        # construct expected flat data from spec and spat
        expected_flat = np.ones((10, 16, 25))
        expected_flat_err = np.ones((10, 16, 25))
        expected_flat *= spatdata[None, None, :]
        expected_flat_err *= spatdata[None, None, :]
        for i in range(16):
            spec = specdata[int(wave[i][0]), :, :]
            expected_flat[:, i, :] *= spec[None, :, i]

        result = calculate_flat(wave, data, var, spatdata, specdata,
                                specwave, specerr, True)

        flat_corr, var_corr, flat_store, flat_err_store = result

        # returned flat and error should be as expected
        assert np.allclose(flat_store, expected_flat)
        assert np.allclose(flat_err_store, expected_flat_err)
        # corrected data should be divided by it
        assert np.allclose(flat_corr, 1 / expected_flat)
        # variance should be divided by it, squared
        assert np.allclose(var_corr, 1 / expected_flat ** 2)

        # propagate flat error
        result = calculate_flat(wave, data, var, spatdata, specdata,
                                specwave, specerr, False)
        # all should be same except variance, which should be higher
        assert np.allclose(result[0], flat_corr)
        assert np.all(result[1] > var_corr)
        assert np.allclose(result[2], flat_store)
        assert np.allclose(result[3], flat_err_store)

    def test_bad_parameters(self, capsys, mocker):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import apply_static_flat
        files = get_xyc_files()

        # bad output directory
        result = apply_static_flat(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = apply_static_flat('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

        # no flat data
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.apply_static_flat.get_flat',
            return_value=None)
        result = apply_static_flat(files[0], write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'No flat found' in capt.err

    def test_wrap_failure(self, capsys, mocker):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import wrap_apply_static_flat

        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.apply_static_flat.multitask',
            return_value=['test', None])

        # bad files
        result = wrap_apply_static_flat(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_xyc_files()
        wrap_apply_static_flat(files[0], write=False,
                               allow_errors=False)

        # allow errors
        result = wrap_apply_static_flat(files, write=False,
                                        allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = wrap_apply_static_flat(files, write=False,
                                        allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_otf_reshape(self):
        from sofia_redux.instruments.fifi_ls.apply_static_flat \
            import apply_static_flat
        files = get_xyc_files()
        hdul = fits.open(files[0])

        # mock OTF data: should have 3D flux, stddev that get modified;
        # rest of extensions are copied as is
        # lambda extension is used, but should match 2D format
        # of non-otf data
        header = hdul[0].header
        header['NGRATING'] = 1
        nramp = 15
        flux = np.repeat(hdul['FLUX_G0'].data.reshape(1, 16, 25),
                         nramp, axis=0)
        stddev = np.repeat(hdul['STDDEV_G0'].data.reshape(1, 16, 25),
                           nramp, axis=0)
        hdul['FLUX_G0'].data = flux
        hdul['STDDEV_G0'].data = stddev
        assert hdul['FLUX_G0'].data.shape == (nramp, 16, 25)
        assert hdul['STDDEV_G0'].data.shape == (nramp, 16, 25)

        result = apply_static_flat(hdul, write=False, outdir=None)
        assert result['FLUX_G0'].data.shape == (nramp, 16, 25)
        assert result['STDDEV_G0'].data.shape == (nramp, 16, 25)
        # flat is 2D
        assert result['FLAT_G0'].data.shape == (16, 25)
        for ext in result[1:]:
            name = ext.header['EXTNAME']
            if name == 'FLUX_G0' or name == 'STDDEV_G0':
                assert not np.allclose(ext.data, hdul[name].data)
            elif 'FLAT' in name:
                continue
            else:
                assert np.allclose(ext.data, hdul[name].data)
        # flux and stddev should have same factors
        assert np.allclose(result['FLUX_G0'].data / hdul['FLUX_G0'].data,
                           result['STDDEV_G0'].data / hdul['STDDEV_G0'].data,
                           equal_nan=True)
        # same flat should be applied to all ramps
        for ramp in range(nramp):
            data = result['FLUX_G0'].data[ramp]
            good = ~np.isnan(data)
            assert np.allclose(hdul['FLUX_G0'].data[ramp][good] / data[good],
                               result['FLAT_G0'].data[good])
