# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import logging

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.exes import readraw as rr


@pytest.fixture
def bpm_file(tmpdir):
    filename = os.path.join(tmpdir.mkdir('test_readraw'), 'bpm.fits')
    ny, nx = 100, 100
    data = np.zeros((ny, nx), dtype=int)
    start = 5
    stop = 95
    data[:, :start] = 2
    data[:, stop:] = 2
    fits.HDUList(fits.PrimaryHDU(data)).writeto(filename)
    return filename


class TestReadRaw(object):

    def test_check_header(self):
        header = fits.Header()
        header['OTPAT'] = 'HELLO'
        header['NINT'] = 10
        header['FRAMETIM'] = 2.0
        header['READNOIS'] = 3.0
        header['DARKVAL'] = 4.0
        header['PAGAIN'] = 5.0
        header['EPERADU'] = 6.0
        rr._check_header(header)
        assert header['PAGAIN'] == 5
        assert header['EPERADU'] == 6
        header['PAGAIN'] = 'a'
        del header['EPERADU']
        rr._check_header(header)
        assert header['PAGAIN'] == 1
        assert header['EPERADU'] == 1

    def test_check_data(self):
        data = np.empty(10)
        with pytest.raises(ValueError):
            rr._check_data(data)
        data = rr._check_data(np.empty((10, 10)))
        assert data.shape == (1, 10, 10)
        data = rr._check_data(np.empty((2, 10, 10)))
        assert data.shape == (2, 10, 10)

    def test_get_data_subarray(self, bpm_file):
        ny, nx = 100, 100
        data = np.empty((2, ny, nx))
        header = fits.Header()
        header['BPM'] = bpm_file
        header['ECTPAT'] = '0 0 0 %i 0 %i' % (ny // 2, nx)
        header['FRAMETIM'] = 1.0

        result, mask = rr._get_data_subarray(data, header)
        assert result.shape == (2, 100, 90)
        assert mask.shape == result.shape[-2:]
        assert header['DETSEC'] == '[1:90,1:100]'
        assert header['NSPAT'] == 90
        assert header['NSPEC'] == 100

        # frametime is updated for subarray
        assert header['FRAMETIM'] == 100 / 1024

        data = np.empty((2, ny + 1, nx + 1))
        with pytest.raises(Exception) as err:
            rr._get_data_subarray(data, header)
        assert 'too small' in str(err).lower()

        # Check array size can be extracted from data size if ectpat is missing
        data = np.empty((2, ny, nx))
        del header['ECTPAT']
        result, mask = rr._get_data_subarray(data, header)
        assert result.shape == (2, 100, 90)
        assert mask.shape == result.shape[-2:]
        assert header['DETSEC'] == '[1:90,1:100]'
        assert header['NSPAT'] == 90
        assert header['NSPEC'] == 100

        # Check nothing is done if badpix file does not exist
        header['BPM'] = '__does_not_exist__'
        result, mask = rr._get_data_subarray(data, header)
        assert result.shape == (2, 100, 100)
        assert mask.shape == result.shape[-2:]
        assert header['DETSEC'] == '[1:100,1:100]'
        assert header['NSPAT'] == 100
        assert header['NSPEC'] == 100

    def test_check_readout_pattern(self):
        header = fits.Header()

        # spin, trash, dest, coadd, nondest
        otpat = 'S0 T1 D2 C3 N0'  # Fowler mode with 4 frame pattern
        header['OTPAT'] = otpat

        # in this case, coadd (3 + 1) = 4 determines the number of patterns
        nframes = 8  # 2 patterns (4 * 2)
        data = np.empty((nframes, 10, 10))
        result, readout = rr._check_readout_pattern(data, header)
        assert readout['spin'] == 1
        assert readout['trash'] == 2
        assert readout['nondest'] == 1
        assert readout['dest'] == 7  # dest + coadd
        assert readout['coadd'] == 4
        assert readout['nread'] == 1  # number of 'N's altogether
        assert readout['npass'] == 9  # sum of all except T
        assert readout['nframes'] == 4
        assert readout['npattern'] == 2
        assert readout['mode'] == 'fowler'

        header['OTPAT'] = 'D0'  # Destructive read
        nframes = 1
        data = np.empty((nframes, 10, 10))
        result, readout = rr._check_readout_pattern(data, header)
        assert readout['mode'] == 'destructive'

        header['OTPAT'] = 'D0 N2'  # Sample-up-the-ramp
        nframes = 8
        data = np.empty((nframes, 10, 10))
        result, readout = rr._check_readout_pattern(data, header)
        assert readout['mode'] == 'sample-up-the-ramp'
        assert result.shape == (8, 10, 10)

        # Check drops frames with incomplete pattern
        nframes = 7
        data = np.empty((nframes, 10, 10))
        result, _ = rr._check_readout_pattern(data, header)
        assert result.shape == (4, 10, 10)

        # Check bad frames
        with pytest.raises(Exception) as err:
            rr._check_readout_pattern(data[:1], header)
        assert "a full pattern is not present" in str(err).lower()

        # Check unrecognized readmode
        header['OTPAT'] = 'D0 N1 C2'
        with pytest.raises(Exception) as err:
            rr._check_readout_pattern(data, header)
        assert "unrecognized readmode" in str(err).lower()

        # Check unreadable readmode
        header['OTPAT'] = 'bad readmode'
        with pytest.raises(Exception) as err:
            rr._check_readout_pattern(data, header)
        assert "unreadable ot pattern" in str(err).lower()

    def test_process_fowler(self):
        header = fits.Header()
        nframes = 8  # 2 patterns (4 * 2)
        data = np.ones((nframes, 10, 10))

        # Fowler mode with 4 frame pattern, 2 read
        otpat = 'S0 T1 D1 N0 N0'
        header['OTPAT'] = otpat

        header['FRAMETIM'] = 1.0
        header['PAGAIN'] = 2.0
        header['EPERADU'] = 3.0
        # gives zeroval = 2.0
        header['DARKVAL'] = 4.0
        header['READNOIS'] = 0.1
        data, readmode = rr._check_readout_pattern(data, header)
        result, variance = rr._process_fowler(data, header, readmode)
        assert np.allclose(result, 2)
        assert np.allclose(variance, 0.1853086)
        assert result.shape == (2, 10, 10)
        assert header['NFRAME'] == 2
        assert header['BEAMTIME'] == 3.0

        # Fowler mode with 1 frame pattern, 1 coadd
        otpat = 'S0 T1 C0 N0'
        # In this case coadd/subtraction would have been done in hardware
        header['OTPAT'] = otpat
        data, readmode = rr._check_readout_pattern(data, header)
        result, variance = rr._process_fowler(data, header, readmode)
        assert np.allclose(result, 1.75)
        assert np.allclose(variance, 0.292222)
        assert result.shape == (8, 10, 10)
        assert header['BEAMTIME'] == 2.0

    def test_process_destructive(self, tmpdir):
        header = fits.Header()
        nframes = 4  # 4 patterns (4 * 1)
        data = np.ones((nframes, 10, 10))

        otpat = 'D0'  # Fowler mode with 4 frame pattern, 2 read
        header['OTPAT'] = otpat

        header['FRAMETIM'] = 1.0
        header['PAGAIN'] = 2.0
        header['EPERADU'] = 3.0
        header['DARKVAL'] = 4.0  # gives zeroval = 2.0
        header['READNOIS'] = 0.1
        data, readmode = rr._check_readout_pattern(data, header)

        # At this point DRKFILE (dark file) has not been defined, so it
        # raises an error
        with pytest.raises(ValueError) as err:
            rr._process_destructive(data, header, readmode)
        assert 'Cannot open dark' in str(err)

        # Try with a dark file
        darkfile = str(tmpdir.mkdir('test_readraw').join(
            'test_process_destructive.fits'))
        dark = np.full(data.shape[1:], 0.5)
        fits.HDUList(fits.PrimaryHDU(dark)).writeto(darkfile, overwrite=True)
        header['DRKFILE'] = darkfile
        header['DETSEC'] = '[1,10,1,10]'
        result, variance = rr._process_destructive(data, header, readmode)
        assert np.allclose(result, 1.75)
        assert np.allclose(variance, 0.58444444)

        assert header['NFRAME'] == 1
        assert header['BEAMTIME'] == 1.0

    def test_process_nondestructive1(self):
        header = fits.Header()
        # 2 patterns (2 * 2)
        nframes = 4
        orig_data = np.ones((nframes, 10, 10))

        # Fowler mode with 1 nondestructive, 1 destructive
        otpat = 'N0 D0'
        header['OTPAT'] = otpat

        header['FRAMETIM'] = 1.0
        header['PAGAIN'] = 2.0
        header['EPERADU'] = 3.0
        # gives zeroval = 2.0
        header['DARKVAL'] = 4.0
        header['READNOIS'] = 0.1
        orig_header = header.copy()

        data, readmode = rr._check_readout_pattern(orig_data.copy(), header)
        result, variance = rr._process_nondestructive1(data, header, readmode)
        # frames subtract, leaving zeroval
        assert np.allclose(result, 2)
        assert np.allclose(variance, 0.6688889)

        assert header['NFRAME'] == 1
        assert header['BEAMTIME'] == 1.0

        # process fowler should give the same result for this pattern
        fowler_header = orig_header.copy()
        data, readmode = rr._check_readout_pattern(orig_data.copy(),
                                                   fowler_header)
        fowler_result, fowler_var = rr._process_fowler(data, fowler_header,
                                                       readmode)
        assert np.allclose(fowler_result, result)
        assert np.allclose(fowler_var, variance)
        assert fowler_header['NFRAME'] == 1
        assert fowler_header['BEAMTIME'] == 1.0

        # if missing nondestructive, should raise error
        header['OTPAT'] = 'D0'
        data, readmode = rr._check_readout_pattern(orig_data.copy(), header)
        with pytest.raises(ValueError) as err:
            rr._process_nondestructive1(data, header, readmode)
        assert 'OTPAT is not suitable' in str(err)

    def test_process_nondestructive2(self):
        header = fits.Header()
        # 2 patterns (2 * 4)
        nframes = 8
        data = np.ones((nframes, 10, 10))
        data[:4] = np.arange(4)[:, None, None]
        data[4:] = np.arange(4)[:, None, None]

        # Fowler mode with 3 nondestructive, one descructive
        otpat = 'N2 D0'
        header['OTPAT'] = otpat

        header['FRAMETIM'] = 1.0
        header['PAGAIN'] = 2.0
        header['EPERADU'] = 3.0
        # gives zeroval = 2
        header['DARKVAL'] = 4.0
        header['READNOIS'] = 0.1
        data, readmode = rr._check_readout_pattern(data, header)

        # result is zero - (3rd - 2nd) /gain = 2 - 1/2 = 1.5
        result, variance = rr._process_nondestructive2(data, header, readmode)
        assert np.allclose(result, 1.5)
        assert np.allclose(variance, 0.50222222)

        assert header['NFRAME'] == 1
        assert header['BEAMTIME'] == 1.0

        # if missing nondestructives, should raise error
        header['OTPAT'] = 'N0 D0'
        data, readmode = rr._check_readout_pattern(data, header)
        with pytest.raises(ValueError) as err:
            rr._process_nondestructive2(data, header, readmode)
        assert 'OTPAT is not suitable' in str(err)

    def test_process_sample_up_the_ramp(self):
        header = fits.Header()
        header['OTPAT'] = 'D0 N2'  # Sample-up-the-ramp
        header['FRAMETIM'] = 1.0
        header['PAGAIN'] = 2.0
        header['EPERADU'] = 3.0
        header['DARKVAL'] = 4.0  # gives zeroval = 2.0
        header['READNOIS'] = 0.1

        nframes = 8
        data = np.empty((nframes, 10, 10))
        data[:] = (np.arange(nframes) % (nframes // 2) + 1)[..., None, None]

        data, readout = rr._check_readout_pattern(data, header)

        # no coadd mode
        result, variance = rr._process_sample_up_the_ramp(
            data, header, readout)
        assert np.allclose(result, 7 / 6)
        assert np.allclose(variance, 0.13244444)

        # Change to coadd mode
        readout['coadd'] = 1
        result, variance = rr._process_sample_up_the_ramp(
            data, header, readout)
        assert np.allclose(result[0], 1.83333333)
        assert np.allclose(result[1], 1.66666667)
        assert np.allclose(variance[0], 0.208)
        assert np.allclose(variance[1], 0.18911111)

        assert header['NFRAME'] == 4
        assert header['BEAMTIME'] == 3.0

    @pytest.mark.parametrize('copy_int,instmode,factor',
                             [(True, 'NOD_OFF_SLIT', 2),
                              (False, 'NOD_ON_SLIT', 1)])
    def test_combine_nods(self, copy_int, instmode, factor):
        header = fits.Header()
        header['INSTMODE'] = instmode
        header['BEAMTIME'] = 2.0
        readmode = {'npattern': 4}
        data = np.empty((8, 10, 10))
        data[:] = (np.arange(8) % (4) + 1)[..., None, None]
        variance = data.copy()
        toss_nint = 0

        # if NINT is 1, no change
        header['NINT'] = 1
        coadd, mvar = rr._combine_nods(data, variance, header,
                                       readmode, toss_nint, copy_int)
        assert coadd is data
        assert mvar is variance

        header['NINT'] = 2
        coadd, mvar = rr._combine_nods(data, variance, header,
                                       readmode, toss_nint, copy_int)
        assert coadd.shape == (2, 10, 10)
        assert mvar.shape == (2, 10, 10)
        assert np.allclose(coadd[0], 1.5)
        assert np.allclose(coadd[1], 3.5)
        assert np.allclose(mvar[0], 0.75)
        assert np.allclose(mvar[1], 1.75)

        assert header['EXPTIME'] == 8.0 / factor
        assert header['NEXP'] == 4

        # Check warnings and errors
        header['NINT'] = 3
        coadd, mvar = rr._combine_nods(data, variance, header,
                                       readmode, toss_nint, copy_int)
        assert coadd.shape == (1, 10, 10)
        assert mvar.shape == (1, 10, 10)
        assert np.allclose(coadd, 2)
        assert np.allclose(mvar, 2 / 3)

        with pytest.raises(Exception) as err:
            header['NINT'] = 100
            rr._combine_nods(data, variance, header, readmode, toss_nint,
                             copy_int)
        assert 'data does not match nint' in str(err).lower()

    def test_combine_nods_tossnint(self, caplog):
        caplog.set_level(logging.INFO)
        header = fits.Header()
        header['NINT'] = 2
        header['BEAMTIME'] = 2.0
        header['NODN'] = 2

        readmode = {'npattern': 8}
        data = np.empty((8, 10, 10))
        data[:] = (np.arange(8) % 4 + 1)[..., None, None]
        variance = data.copy()

        copy_int = False
        toss_nint = 1
        coadd, mvar = rr._combine_nods(data, variance, header,
                                       readmode, toss_nint, copy_int)
        assert 'Dropping the first 1 frames' in caplog.text
        assert coadd.shape == (4, 10, 10)
        assert np.allclose(coadd[0], 2.)
        assert np.allclose(coadd[1], 3.5)
        assert np.allclose(coadd[2], 1.5)
        assert np.allclose(coadd[3], 3.5)
        assert np.allclose(mvar[0], 2.)
        assert np.allclose(mvar[1], 1.75)
        assert np.allclose(mvar[2], 0.75)
        assert np.allclose(mvar[3], 1.75)

        # toss too high: ignored
        toss_nint = 3
        coadd, mvar = rr._combine_nods(data, variance, header,
                                       readmode, toss_nint, copy_int)
        assert 'TOSS_NINT is set higher than NINT' in caplog.text
        assert np.allclose(coadd[0], 1.5)
        assert np.allclose(coadd[1], 3.5)
        assert np.allclose(coadd[2], 1.5)
        assert np.allclose(coadd[3], 3.5)
        assert np.allclose(mvar[0], 0.75)
        assert np.allclose(mvar[1], 1.75)
        assert np.allclose(mvar[2], 0.75)
        assert np.allclose(mvar[3], 1.75)

        # allow subbing the first frames from the next B nod
        # if toss == nint
        toss_nint = 2
        data[4:] += 1
        coadd, mvar = rr._combine_nods(data, variance, header,
                                       readmode, toss_nint, copy_int)
        assert 'Copying the first integration(s)' in caplog.text
        assert np.allclose(coadd[0], 2.5)
        assert np.allclose(coadd[1], 3.5)
        assert np.allclose(coadd[2], 2.5)
        assert np.allclose(coadd[3], 4.5)
        assert np.allclose(mvar[0], 0.75)
        assert np.allclose(mvar[1], 1.75)
        assert np.allclose(mvar[2], 0.75)
        assert np.allclose(mvar[3], 1.75)

    def test_readraw_default(self, raw_low_hdul, rdc_low_hdul, capsys):
        data = raw_low_hdul[0].data
        header = raw_low_hdul[0].header

        # default
        coadd, var, mask = rr.readraw(data, header)
        capt = capsys.readouterr()
        assert 'Recommended readout mode: fowler' in capt.out
        assert 'Fowler mode' in capt.out
        assert 'Linear correction not applied' in capt.out
        assert coadd.shape == (4, 1024, 1024)
        assert var.shape == (4, 1024, 1024)
        assert mask.shape == (1024, 1024)
        assert header['NFRAME'] == 1
        assert header['BEAMTIME'] == 1.0

        # mask has some bad pixels from default bpm
        badmask = ~mask
        assert badmask.sum() == 3003
        # check that bad pixels are in the right place for the default
        assert np.all(badmask[:, 0])
        assert np.all(badmask[647, 776:793])

        # coadd data should come out same as synthetic
        assert np.allclose(np.median(coadd), np.median(rdc_low_hdul[0].data))

        # coadd error should be around .004 of data
        assert np.allclose(np.median(np.sqrt(var)), 0.004 * np.median(coadd),
                           atol=0.5)

    @pytest.mark.parametrize('algorithm,name,beamtime,error',
                             [(0, 'last destructive', 2.0, None),
                              (1, 'destructive', 2.0, None),
                              (2, 'first/last nd', 1.0, None),
                              (3, 'second/penultimate nd', 1.0,
                               'not suitable'),
                              (4, 'fowler', 1.0, None),
                              (5, 'sample-up-the-ramp', 1.0, 'not suitable'),
                              (6, 'bad', 1.0, 'Invalid algorithm')])
    def test_readraw_algorithm(self, raw_low_hdul, rdc_low_hdul, capsys,
                               algorithm, name, beamtime, error):
        data = raw_low_hdul[0].data
        header = raw_low_hdul[0].header

        if error is not None:
            with pytest.raises(ValueError) as err:
                rr.readraw(data, header, algorithm=algorithm)
            assert error in str(err)
        else:
            coadd, var, mask = rr.readraw(data, header, algorithm=algorithm)
            capt = capsys.readouterr()
            assert f'Using read mode algorithm: {name}' in capt.out
            assert coadd.shape == (4, 1024, 1024)
            assert var.shape == (4, 1024, 1024)
            assert mask.shape == (1024, 1024)
            assert header['BEAMTIME'] == beamtime

            # coadd data should come out same as synthetic
            data_median = np.median(coadd)
            error_median = np.median(np.sqrt(var))
            assert np.allclose(data_median, np.median(rdc_low_hdul[0].data))

            # coadd error should be < 1% of data and > 0
            assert error_median < 0.01 * data_median
            assert np.all(var) > 0

    def test_readraw_lincor(self, raw_low_hdul, rdc_low_hdul, capsys):
        data = raw_low_hdul[0].data
        header = raw_low_hdul[0].header

        coadd, var, mask = rr.readraw(data, header, do_lincor=True)
        assert coadd.shape == (4, 1024, 1024)
        assert var.shape == (4, 1024, 1024)
        assert mask.shape == (1024, 1024)
        assert 'Applying linear correction' in capsys.readouterr().out

        # mask has default bad pix (3003) + some saturated
        badmask = ~mask
        assert np.allclose(badmask.sum(), 5770, atol=10)

        # coadd data should come out slightly higher than synthetic
        synth_median = np.median(rdc_low_hdul[0].data)
        data_median = np.median(coadd)
        error_median = np.median(np.sqrt(var))
        assert np.allclose(data_median, synth_median, rtol=0.1)
        assert data_median > synth_median

        # coadd error should be < 1% of data and > 0
        assert error_median < 0.01 * data_median
        assert np.all(var) > 0
