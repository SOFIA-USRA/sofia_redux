# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
from astropy.table import Table
import numpy as np

import sofia_redux.instruments.fifi_ls.fit_ramps as u
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_split_files


class TestFitRamps(FIFITestCase):

    def test_get_readout_range(self):
        header = fits.Header()
        header['RAMPLN_R'] = 100
        header['RAMPLN_B'] = 200
        assert u.get_readout_range(header) == (2, 200)
        header['CHANNEL'] = 'RED'
        assert u.get_readout_range(header) == (2, 100)
        header['DATE-OBS'] = '2014-06-01T00:00:00'
        assert u.get_readout_range(header) == (3, 100)
        del header['RAMPLN_R']
        assert u.get_readout_range(header) == (3, None)

    def test_resize_data(self):
        readout_range = 2, 32
        datashape = 960, 18, 26
        data = np.full(datashape, 1, dtype=np.int16)
        indpos = 1234567
        result, _ = u.resize_data(data, readout_range, indpos,
                                  remove_first_ramps=False)
        assert result.shape == (30, 29, 16, 25)
        result, _ = u.resize_data(data, readout_range, indpos,
                                  remove_first_ramps=True)
        assert result.shape == (28, 29, 16, 25)
        readout_range = 10, 30
        result, _ = u.resize_data(data, readout_range, indpos,
                                  remove_first_ramps=True)
        assert result.shape == (30, 19, 16, 25)
        assert u.resize_data(None, readout_range, indpos) is None
        assert u.resize_data(data, [1, 2, 3], indpos) is None
        assert u.resize_data(np.empty((959, 18, 26), dtype=np.int16),
                             readout_range, indpos) is None
        assert u.resize_data(np.empty((1, 960, 18, 26, 26), dtype=np.int16),
                             readout_range, indpos) is None

        # check bias subtraction: add a different value to all spaxels
        data += np.arange(26)
        result, _ = u.resize_data(data, readout_range, indpos,
                                  subtract_bias=False)
        assert not np.allclose(result, result[0, 0, 0, 0])

        # with bias subtraction, all data should be the same
        result, _ = u.resize_data(data, readout_range,
                                  indpos, subtract_bias=True)
        assert np.allclose(result, result[0, 0, 0, 0])

    def test_fit_data(self):
        # seed the random module for consistent tests
        rand = np.random.RandomState(42)

        data = np.zeros((30, 30, 18, 25))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):
                    data[i, :, j, k] = np.arange(30)
        noise = (rand.random(data.shape) - 0.5) / 10
        data += noise
        slopes, mvar = u.fit_data(data, s2n=10, threshold=5)
        assert np.allclose(slopes, 1, rtol=np.sqrt(mvar[0]))
        assert not np.allclose(mvar, mvar[0])
        assert not np.isnan(slopes).any()
        assert not np.any(mvar == 0)
        data -= noise
        badnoise = noise * 1000
        data += badnoise
        slopes2, mvar2 = u.fit_data(data, s2n=10, threshold=5)
        assert np.isnan(slopes2).any()
        assert np.any(mvar2 == 0)

    def test_process_extension(self):
        filename = get_split_files()[0]
        hdul = fits.open(filename)
        hdu = hdul[1]
        readout_range = u.get_readout_range(hdul[0].header)
        result = u.process_extension(hdu, readout_range)
        assert len(result) == 2
        assert isinstance(result[0], fits.ImageHDU)
        assert isinstance(result[1], fits.ImageHDU)
        flux = result[0]
        stddev = result[1]
        assert isinstance(flux.header, fits.Header)
        assert 'BGLEVL_A' in flux.header
        assert flux.data.ndim == 2
        assert flux.data.shape == (16, 25)
        assert stddev.data.ndim == 2
        assert stddev.data.shape == (16, 25)
        assert not np.isnan(flux.data).all()
        badmask = []
        for i in range(16):
            for j in range(25):
                badmask.append([i, j])
        badmask = np.array(badmask)
        result = u.process_extension(hdu, readout_range, badmask=badmask)
        assert np.isnan(result[0].data).all()
        assert np.isnan(result[1].data).all()

    def test_process_bad_extension(self, capsys):
        filename = get_split_files()[0]
        hdul = fits.open(filename)
        readout_range = u.get_readout_range(hdul[0].header)

        # no data
        result = u.process_extension(hdul[0], readout_range)
        assert result is None
        capt = capsys.readouterr()
        assert 'No data in HDU' in capt.err

    def test_process_extension_posdata(self, capsys):
        filename = get_split_files()[0]
        hdul = fits.open(filename)
        readout_range = u.get_readout_range(hdul[0].header)

        # attempt to average ramps with posdata
        result = u.process_extension(hdul[1], readout_range,
                                     average_ramps=True, posdata=10)
        assert 'Incompatible arguments' in capsys.readouterr().err
        assert len(result) == 2

        # provide posdata, but mismatched to actual data
        header = hdul[0].header
        nsamp = hdul[1].data.shape[0]
        tstart = header['START'] + header['FIFISTRT'] * header['ALPHA']
        tab = Table()
        tab['DLAM_MAP'] = header['DLAM_MAP'] + np.arange(nsamp - 10,
                                                         dtype=float)
        tab['DBET_MAP'] = header['DBET_MAP'] + np.arange(nsamp - 10,
                                                         dtype=float)
        tab['FLAG'] = np.full(nsamp - 10, True)
        tab['UNIXTIME'] = tstart + np.arange(nsamp - 10, dtype=float)

        result = u.process_extension(hdul[1], readout_range,
                                     average_ramps=False, posdata=tab)
        assert 'Number of readouts does not match header' \
            in capsys.readouterr().err
        assert result is None

        # provide good posdata -- should now return three hdu results
        tab = Table()
        tab['DLAM_MAP'] = header['DLAM_MAP'] + np.arange(nsamp, dtype=float)
        tab['DBET_MAP'] = header['DBET_MAP'] + np.arange(nsamp, dtype=float)
        tab['FLAG'] = np.full(nsamp, True)
        tab['UNIXTIME'] = tstart + np.arange(nsamp, dtype=float)
        result = u.process_extension(hdul[1], readout_range,
                                     average_ramps=False, posdata=tab)
        assert len(result) == 3
        assert isinstance(result[0], fits.ImageHDU)
        assert isinstance(result[1], fits.ImageHDU)
        assert isinstance(result[2], fits.BinTableHDU)

    def test_bad_parameters(self, capsys):
        files = get_split_files()

        # bad output directory
        result = u.fit_ramps(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = u.fit_ramps('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_hdul_input(self):
        filename = get_split_files()[0]
        hdul = fits.open(filename)
        result = u.fit_ramps(hdul)
        assert isinstance(result, fits.HDUList)

    def test_fit_ramps(self):
        filename = get_split_files()[0]
        result = u.fit_ramps(filename)
        assert isinstance(result, fits.HDUList)
        assert len(result) > 1
        phead = result[0].header
        assert phead['PRODTYPE'] == 'ramps_fit'

        # check BUNIT
        for ext in result[1:]:
            assert ext.header['BUNIT'] == 'adu/s'

    def test_fit_failure(self, mocker, capsys):
        filename = get_split_files()[0]

        # mock failure in resize data
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.fit_ramps.resize_data',
            return_value=None)
        result = u.fit_ramps(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Failed to process extension' in capt.err

        # mock failure in process extension
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.fit_ramps.process_extension',
            return_value=None)
        result = u.fit_ramps(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Failed to process extension' in capt.err

        # mock failure in readout range
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.fit_ramps.get_readout_range',
            return_value=None)
        result = u.fit_ramps(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid readout range' in capt.err

    def test_wrap_fit_ramps(self, tmpdir):
        # for this case
        filenames = get_split_files() * 2
        assert len(filenames) > 1
        t0 = time.time()
        result = u.wrap_fit_ramps(filenames, outdir=str(tmpdir), jobs=None)
        assert len(result) > 0
        for f in result:
            if isinstance(f, str) and os.path.isfile(f):
                os.remove(f)
        t1 = time.time()
        dseries = t1 - t0
        print("series time = %f seconds" % (dseries))
        t2 = time.time()
        result = u.wrap_fit_ramps(filenames, outdir=str(tmpdir), jobs=-1)
        assert len(result) > 0

        t3 = time.time()
        dparallel = (t3 - t2)
        print("parallel time = %f seconds" % (dparallel))

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.fit_ramps.multitask',
            return_value=['test', None])

        # bad files
        result = u.wrap_fit_ramps(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_split_files()
        u.wrap_fit_ramps(files[0], write=False,
                         allow_errors=False)

        # allow errors
        result = u.wrap_fit_ramps(files, write=False,
                                  allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = u.wrap_fit_ramps(files, write=False,
                                  allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_otf_ramps(self, capsys):
        filename = get_split_files()[0]

        # mock OTF data
        hdul = fits.open(filename)
        header = hdul[0].header
        header['INSTMODE'] = 'OTF_TP'
        header['C_AMP'] = 0
        header['NGRATING'] = 1
        nsamp = hdul[1].data.shape[0]
        tab = Table()
        tab['DLAM_MAP'] = header['DLAM_MAP'] + np.arange(nsamp, dtype=float)
        tab['DBET_MAP'] = header['DBET_MAP'] + np.arange(nsamp, dtype=float)
        tab['FLAG'] = np.full(nsamp, True)
        tab['UNIXTIME'] = header['START'] \
            + header['FIFISTRT'] * header['ALPHA'] \
            + np.arange(nsamp, dtype=float)
        hcopy = hdul.copy()
        hdul.append(fits.BinTableHDU(tab, name='SCANPOS_G0'))

        result = u.fit_ramps(hdul)
        assert isinstance(result, fits.HDUList)
        assert 'SCANPOS_G0' in result
        rlen = len(result['SCANPOS_G0'].data)
        assert rlen == result['FLUX_G0'].data.shape[0]
        assert rlen == result['STDDEV_G0'].data.shape[0]
        assert 'Trimming' not in capsys.readouterr().out

        # output positions should be averaged over ramps
        assert np.allclose(result['SCANPOS_G0'].data['DLAM_MAP'][0],
                           np.mean(tab['DLAM_MAP'][:32]))
        assert np.allclose(result['SCANPOS_G0'].data['DBET_MAP'][0],
                           np.mean(tab['DBET_MAP'][:32]))

        # header should have ramp start and end times
        assert np.allclose(result['FLUX_G0'].header['RAMPSTRT'],
                           np.mean(tab['UNIXTIME'][:32]))
        assert np.allclose(result['FLUX_G0'].header['RAMPEND'],
                           np.mean(tab['UNIXTIME'][:-32]))

        # bad-flagged data should be trimmed out
        hdul = hcopy.copy()
        tab['FLAG'][:10] = False
        tab['FLAG'][-10:] = False
        hdul.append(fits.BinTableHDU(tab, name='SCANPOS_G0'))
        trimmed = u.fit_ramps(hdul.copy())
        assert 'Trimming 2 ramps' in capsys.readouterr().out
        newlen = len(trimmed['SCANPOS_G0'].data)
        assert newlen == rlen - 2
        assert newlen == trimmed['FLUX_G0'].data.shape[0]
        assert newlen == trimmed['STDDEV_G0'].data.shape[0]

    def test_subtract_bias(self):
        filename = get_split_files()[0]
        hdul = fits.open(filename)

        # make some data with known slopes
        rand = np.random.RandomState(42)
        data = np.zeros((15, 32, 18, 26))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):
                    data[i, :, j, k] = np.arange(32)

        # add correlated noise to each spexel in a spaxel
        noise = (rand.random((15, 32, 26)) - 0.5) / 10
        for spex in range(18):
            data[:, :, spex, :] += noise
        hdul[1].data = data.reshape((480, 18, 26))

        # run with and without bias subtraction
        no_sub = u.fit_ramps(hdul, subtract_bias=False, indpos_sigma=-1)
        sub = u.fit_ramps(hdul, subtract_bias=True, indpos_sigma=-1)

        # output slopes are the same
        good = ~np.isnan(no_sub['FLUX_G0'].data) \
            & ~np.isnan(sub['FLUX_G0'].data)
        assert np.allclose(no_sub['FLUX_G0'].data[good], 1.0,
                           equal_nan=True, atol=0.001)
        assert np.allclose(sub['FLUX_G0'].data[good], 1.0,
                           equal_nan=True, atol=0.001)

        # output error should be lower for bias subtracted data
        good = ~np.isnan(no_sub['STDDEV_G0'].data) \
            & ~np.isnan(sub['STDDEV_G0'].data)
        assert np.all(no_sub['STDDEV_G0'].data[good]
                      > sub['STDDEV_G0'].data[good])

    def test_grating_instability_rejection(self, capsys):
        rand = np.random.RandomState(42)
        filename = get_split_files()[0]
        hdul = fits.open(filename)
        # make indpos small to make the math easier
        indpos = 40000
        hdul[1].header['INDPOS'] = indpos

        # make some data with known slopes
        data = np.zeros((15, 32, 18, 26), dtype=np.int16)
        for i in range(data.shape[0]):
            if i % 2 == 0:
                sign = 1
            else:
                sign = -1
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):
                    data[i, :, j, k] = np.arange(32, dtype=np.int16) \
                        + rand.random_integers(-1, 1, 32)

                    # set the values in the 26th spaxel to the indpos, with
                    # a little wiggle
                    if k == 25:
                        data[i, :, 2::4, k] = indpos + sign * i
                        data[i, :, 3::4, k] = 0

        # reshape back to expected size
        hdul[1].data = data.reshape((480, 18, 26))

        # test resize_data with and without rejection - should give
        # same answer (no flags) for these data
        readout_range = u.get_readout_range(hdul[0].header)
        no_rej, bad = u.resize_data(hdul[1].data, readout_range, indpos,
                                    remove_first_ramps=False,
                                    subtract_bias=False,
                                    indpos_sigma=-1)
        assert bad.shape == (no_rej.shape[0],)
        assert not np.any(bad)
        assert 'Expected INDPOS' not in capsys.readouterr().out
        assert 'Bad ramp index' not in capsys.readouterr().out

        with_rej, bad = u.resize_data(hdul[1].data, readout_range, indpos,
                                      remove_first_ramps=False,
                                      subtract_bias=False,
                                      indpos_sigma=3.0)
        assert np.allclose(no_rej, with_rej)
        assert not np.any(bad)
        assert f'Expected INDPOS: {indpos}' in capsys.readouterr().out
        assert 'Bad ramp index' not in capsys.readouterr().out

        # now insert a bad value in the data for two ramps
        data[3, ::2, 2::4, 25] = indpos + 10000
        data[10, ::2, 2::4, 25] = indpos - 10000
        hdul[1].data = data.reshape((480, 18, 26))

        # these ramps should be flagged
        with_rej, bad = u.resize_data(hdul[1].data, readout_range, indpos,
                                      indpos_sigma=3.0,
                                      remove_first_ramps=False,
                                      subtract_bias=False)
        assert np.allclose(no_rej, with_rej)
        assert np.sum(bad) == 2
        assert bad[3]
        assert bad[10]
        assert 'Bad ramp index: [ 3 10]' in capsys.readouterr().out

        # lower sigma rejects more, higher sigma rejects less
        _, lower = u.resize_data(hdul[1].data, readout_range, indpos,
                                 indpos_sigma=1.0,
                                 remove_first_ramps=False,
                                 subtract_bias=False)
        assert np.sum(lower) > 2
        _, higher = u.resize_data(hdul[1].data, readout_range, indpos,
                                  indpos_sigma=1000,
                                  remove_first_ramps=False,
                                  subtract_bias=False)
        assert np.sum(higher) < 2
        capsys.readouterr()

        # run the full fit with and without rejection
        no_rej = u.fit_ramps(hdul, subtract_bias=False,
                             remove_first=False, indpos_sigma=-1)
        assert 'Bad ramp index' not in capsys.readouterr().out
        rej = u.fit_ramps(hdul, subtract_bias=False,
                          remove_first=False, indpos_sigma=3)
        assert 'Bad ramp index: [ 3 10]' in capsys.readouterr().out

        # output slopes are the same either way
        good = ~np.isnan(no_rej['FLUX_G0'].data) \
            & ~np.isnan(rej['FLUX_G0'].data)
        assert np.allclose(no_rej['FLUX_G0'].data[good], 1.0,
                           equal_nan=True, atol=0.1)
        assert np.allclose(rej['FLUX_G0'].data[good], 1.0,
                           equal_nan=True, atol=0.1)

        # output error should be lower for non-filtered data,
        # since the number of ramps used is higher
        good = ~np.isnan(no_rej['STDDEV_G0'].data) \
            & ~np.isnan(rej['STDDEV_G0'].data)
        assert np.all(no_rej['STDDEV_G0'].data[good]
                      < rej['STDDEV_G0'].data[good])

    def test_write(self, tmpdir):
        filename = get_split_files()
        result = u.wrap_fit_ramps(filename, write=True, outdir=str(tmpdir))
        for fn in result:
            assert isinstance(fn, str)
            assert fn.startswith(str(tmpdir))
            assert os.path.isfile(fn)
