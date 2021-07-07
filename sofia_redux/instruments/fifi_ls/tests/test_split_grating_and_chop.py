# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.fifi_ls.split_grating_and_chop as u
from sofia_redux.instruments.fifi_ls.make_header import make_header
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, test_files, get_wxy_files, MockHDU


class TestSplitGratingAndChop(FIFITestCase):

    def fake_hdul(self):
        """

        Returns
        -------
        2-tuple
        """
        data = np.recarray(7680, dtype=[('header', '>i2', (8,)),
                                        ('data', '>i2', (18, 26))])
        header = make_header()
        header['SIMPLE'] = True
        header['EXTEND'] = True
        header['CHANNEL'] = 'RED'
        header['C_SCHEME'] = '2POINT'
        header['C_AMP'] = 135.0
        header['C_CHOPLN'] = 64
        header['RAMPLN_R'] = 32
        header['G_PSUP_R'] = 4
        header['G_SZUP_R'] = 510
        header['G_STRT_R'] = 901405
        header['G_PSDN_R'] = 0
        header['G_SZDN_R'] = 0
        header['G_CYC_R'] = 1
        header['C_CYC_R'] = 15
        header['PRIMARAY'] = 'BLUE'
        header['G_STRT_B'] = 733726
        header['G_PSUP_B'] = 4
        header['G_SZUP_B'] = 4
        hdulist = fits.HDUList([fits.PrimaryHDU(header=header),
                                fits.BinTableHDU(data)])
        hdulist[1].header['EXTNAME'] = 'FIFILS_rawdata'
        return hdulist

    def test_get_channel(self):
        hdul = self.fake_hdul()
        header = hdul[0].header
        data = hdul[1].data
        del header['CHANNEL']
        data['header'][:, 3] = 3
        assert u.get_channel(hdul, channel_index=3) == 'BLUE'
        data['header'][:, 3] = 0
        assert u.get_channel(hdul, channel_index=3) == 'RED'

        for chan in ['blue', 'BLUE ', '1', 1]:
            header['CHANNEL'] = chan
            assert u.get_channel(hdul, channel_index=3) == 'BLUE'
        for chan in ['red', 'RED ', '0', 0]:
            header['CHANNEL'] = chan
            assert u.get_channel(hdul, channel_index=3) == 'RED'
        for chan in ['foo', '-1', '3', 3]:
            header['CHANNEL'] = chan
            assert u.get_channel(hdul, channel_index=3) == 'UNKNOWN'

    def test_get_split_params(self):
        hdul = self.fake_hdul()
        header = hdul[0].header
        header['C_CHOPLN'] = 4
        header['RAMPLN_R'] = 2
        assert u.get_split_params(hdul)['success']

        header['C_CHOPLN'] = 2
        header['RAMPLN_R'] = 3
        assert not u.get_split_params(hdul)['success']

        header['C_CHOPLN'] = 2
        header['RAMPLN_R'] = 4
        assert not u.get_split_params(hdul)['success']

        header['C_CHOPLN'] = 2
        header['RAMPLN_R'] = 2
        assert not u.get_split_params(hdul)['success']

        header['C_CHOPLN'] = 4
        header['RAMPLN_R'] = 2
        params = u.get_split_params(hdul)
        assert params['C_AMP'] == 135

        del header['C_AMP']
        params = u.get_split_params(hdul)
        assert params['C_AMP'] == 0
        assert params['success']

        del header['G_SZDN_R']
        params = u.get_split_params(hdul)
        assert not params['success']
        assert params['G_SZDN'] is None

        hdul = self.fake_hdul()
        header = hdul[0].header
        header['CHANNEL'] = 'foo'
        header['C_CHOPLN'] = 4
        header['RAMPLN_R'] = 2
        assert not u.get_split_params(hdul)['success']

        hdul = self.fake_hdul()
        header = hdul[0].header
        header['C_CHOPLN'] = 4
        header['RAMPLN_R'] = 2
        header['C_SCHEME'] = '4POINT'
        params = u.get_split_params(hdul)
        assert not params['success']
        assert params['C_SCHEME'] == 4

        # check bad value in c_scheme
        hdul = self.fake_hdul()
        hdul[0].header['C_SCHEME'] = 'UNKNOWN'
        params = u.get_split_params(hdul)
        assert not params['success']
        assert params['C_SCHEME'] is None

        # check bad value in rampln
        hdul = self.fake_hdul()
        hdul[0].header['RAMPLN_R'] = 'UNKNOWN'
        params = u.get_split_params(hdul)
        assert not params['success']

    def test_trim_data(self):
        hdul = self.fake_hdul()
        data = hdul[1].data

        params = u.get_split_params(hdul)
        ramplength = params['RAMPLN']
        si = params['sample_index']
        ri = params['ramp_index']
        # Test partial ramp removal
        # full data set
        ndata = len(hdul[1].data)
        data['header'][:, si] = ramplength - 1
        data['header'][:, ri] = 1
        data['header'][0, si] = 0
        hdul[1].data = data
        h2 = u.trim_data(hdul, params)
        assert len(h2[1].data) == ndata
        # trim one from start
        data = hdul[1].data
        data['header'][0, si] = 1
        data['header'][1, si] = 0
        hdul[1].data = data
        h2 = u.trim_data(hdul, params)
        assert len(h2[1].data) == (ndata - 1)
        # trim one from end
        data = hdul[1].data
        data['header'][-1, si] += 1
        data['header'][-1, ri] += 1
        hdul[1].data = data
        h2 = u.trim_data(hdul, params)
        assert len(h2[1].data) == (ndata - 2)

        # Test unpaired chops
        hdul = self.fake_hdul()
        data = hdul[1].data
        data['header'][:, si] = ramplength - 1  # no ramp removal
        data['header'][0: 10, ri] = 2
        data['header'][10:-10, ri] = 0
        data['header'][-11, ri] = 2
        data['header'][-10:, ri] = 0
        hdul[1].data = data
        h2 = u.trim_data(hdul, params)
        assert len(h2[1].data) == (ndata - 20)

    def test_name_output_files(self):
        hdul = self.fake_hdul()
        header = hdul[0].header
        header['FILENUM'] = '777'
        header['AOR_ID'] = 'testaor'
        header['MISSN-ID'] = 'foo_F123'
        header['CHANNEL'] = 'BLUE'
        files = u.name_output_files(hdul)
        assert files == ('F0123_FI_IFS_testaor_BLU_CP0_777.fits',
                         'F0123_FI_IFS_testaor_BLU_CP1_777.fits')

        # check unknown channel
        header['CHANNEL'] = 'foobar'
        files = u.name_output_files(hdul)
        assert files == ('F0123_FI_IFS_testaor_UN_CP0_777.fits',
                         'F0123_FI_IFS_testaor_UN_CP1_777.fits')

        # check blank aor_id
        header['AOR_ID'] = ''
        files = u.name_output_files(hdul)
        assert files == ('F0123_FI_IFS_UNKNOWN_UN_CP0_777.fits',
                         'F0123_FI_IFS_UNKNOWN_UN_CP1_777.fits')

    def test_separate_chops(self):
        hdul = self.fake_hdul()
        params = u.get_split_params(hdul)
        result = u.separate_chops(hdul, params)
        hdul2 = result[0]

        assert len(hdul2) == (params['G_PSUP'] + 1)
        assert hdul2[0].data is None
        assert len(hdul2[0].header) > len(hdul2[1].header)
        assert len(hdul2[1].header) == len(hdul2[2].header)
        assert isinstance(hdul2[1].data, np.ndarray)

        imhdr = hdul2[1].header
        assert 'DATE' in imhdr
        assert imhdr['CHOPNUM'] == 0
        assert imhdr['NODPOS'] == 0

    def test_separate_chops_error(self, capsys):
        hdul = self.fake_hdul()
        params = u.get_split_params(hdul)
        pcopy = params.copy()

        # test rampln > chopln
        params['RAMPLN'] = params['CHOPLN'] * 2
        result = u.separate_chops(hdul, params)
        assert result is None
        assert 'not accounted for' in capsys.readouterr().err

        # test rampln = chopln and not c_amp = 0
        params['RAMPLN'] = params['CHOPLN']
        result = u.separate_chops(hdul, params)
        assert result is None
        assert 'not accounted for' in capsys.readouterr().err

        # with c_amp = 0 is okay
        hdul[0].header['C_AMP'] = 0
        params = u.get_split_params(hdul)
        params['RAMPLN'] = params['CHOPLN']
        result = u.separate_chops(hdul, params)
        assert result is not None

        # test mismatch between prime and pos
        params = pcopy.copy()
        params['G_PSUP'] *= 2
        hdul = self.fake_hdul()
        result = u.separate_chops(hdul, params)
        assert result is not None

        # test mismatch between header and data

        # mock HDU to make an error easier to trigger
        hdul = [hdul[0], MockHDU()]
        hdul[1].data['HEADER'] = np.zeros((10, 10))
        result = u.separate_chops(hdul, params)
        assert result is None
        assert 'does not match' in capsys.readouterr().err

    def test_split_grating_and_chop(self, tmpdir):
        hdul = self.fake_hdul()
        fname = os.path.join(tmpdir, 'testfits.fits')
        hdul.writeto(fname)
        result = u.split_grating_and_chop(fname)
        assert isinstance(result[0], fits.HDUList)
        assert len(result[0]) > 1

    def test_total_power(self):
        # total power mode:
        # when c_amp = 0, should get back only a CP0 file
        filename = test_files()[0]
        hdul = fits.open(filename)
        hdul[0].header['C_AMP'] = 0
        result = u.split_grating_and_chop(hdul)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], fits.HDUList)
        assert 'CP0' in result[0][0].header['FILENAME']

    def test_hdul_input(self):
        filename = test_files()[0]
        hdul = fits.open(filename)
        result = u.split_grating_and_chop(hdul)
        assert isinstance(result, list)
        assert len(result) == 2
        for hdul in result:
            assert isinstance(hdul, fits.HDUList)
            # check bunit
            for ext in hdul[1:]:
                assert ext.header['BUNIT'] == 'adu'

    def test_bad_parameters(self, capsys):
        files = test_files()

        # bad output directory
        result = u.split_grating_and_chop(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = u.split_grating_and_chop('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_split_failure(self, mocker, capsys):
        filename = test_files()[0]

        # test bad data

        # missing extension
        hdul = fits.open(filename)
        result = u.split_grating_and_chop([hdul[0]])
        assert result is None
        capt = capsys.readouterr()
        assert 'HDUList missing extension' in capt.err

        # wrong extension type
        imgfile = get_wxy_files()[0]
        result = u.split_grating_and_chop(imgfile)
        assert result is None
        capt = capsys.readouterr()
        assert 'Expected BINTABLE' in capt.err

        # wrong extension name
        hdul = fits.open(filename)
        hdul[1].header['EXTNAME'] = 'TEST'
        result = u.split_grating_and_chop(hdul)
        assert result is None
        capt = capsys.readouterr()
        assert 'Can only split FIFILS_rawdata' in capt.err

        # missing data column
        hdul = fits.open(filename)
        hdul[1].data.columns.del_col('DATA')
        result = u.split_grating_and_chop(hdul)
        assert result is None
        capt = capsys.readouterr()
        assert 'Missing expected DATA' in capt.err

        # mock failure in separate_chops
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.split_grating_and_chop.'
            'separate_chops',
            return_value=None)
        result = u.split_grating_and_chop(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem separating chops' in capt.err

        # mock failure in trim_data
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.split_grating_and_chop.trim_data',
            return_value=None)
        result = u.split_grating_and_chop(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem trimming data' in capt.err

        # mock failure in params
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.split_grating_and_chop.'
            'get_split_params',
            return_value={'success': False})
        result = u.split_grating_and_chop(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem getting split parameters' in capt.err

        # mock failure in make_header
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.split_grating_and_chop.'
            'make_header',
            return_value=None)
        result = u.split_grating_and_chop(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem updating header' in capt.err

    def test_wrapper(self, tmpdir):
        files = test_files()
        for _ in range(3):
            files += files
        t0 = time.time()
        result = u.wrap_split_grating_and_chop(
            files, outdir=str(tmpdir), allow_errors=True, jobs=None)

        t1 = time.time()
        d0 = t1 - t0
        print("processing time in series: %f seconds" % d0)
        assert len(result) > 0
        t0 = time.time()
        # Note that for parallel processing errors are expected
        # since we are recycling the same few files and writing and
        # deleting at the same time.
        result = u.wrap_split_grating_and_chop(
            files, outdir=str(tmpdir), allow_errors=True, jobs=-1)
        t1 = time.time()
        d1 = t1 - t0
        print("processing time in parallel: %f seconds" % d1)
        assert len(result) > 0

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.split_grating_and_chop.multitask',
            return_value=[['test'], None])

        # bad files
        result = u.wrap_split_grating_and_chop(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = test_files()
        u.wrap_split_grating_and_chop(files[0], write=False,
                                      allow_errors=False)

        # allow errors
        result = u.wrap_split_grating_and_chop(files, write=False,
                                               allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = u.wrap_split_grating_and_chop(files, write=False,
                                               allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err

    def test_otf_posdata(self, capsys):
        filename = test_files()[0]
        hdul = fits.open(filename)
        hdul[0].header['INSTMODE'] = 'OTF_TP'
        hdul[0].header['C_AMP'] = 0

        # without other keys, should fail
        result = u.split_grating_and_chop(hdul, write=False)
        assert result is None
        assert 'Missing required keywords' in capsys.readouterr().err

        # add UNIXSTRT, OTFSTART, TRK_DRTN to cover full set of ramps
        hdul[0].header['UNIXSTRT'] = 0
        hdul[0].header['OTFSTART'] = 0
        hdul[0].header['TRK_DRTN'] = hdul[0].header['EXPTIME']
        hdul[0].header['OBSLAMV'] = 1.0
        hdul[0].header['OBSBETV'] = 0.0
        result = u.split_grating_and_chop(hdul, write=False)
        assert len(result) == 1
        for hdul in result:
            assert isinstance(hdul, fits.HDUList)
            assert 'SCANPOS_G0' in hdul
            assert isinstance(hdul['SCANPOS_G0'], fits.BinTableHDU)
            assert hdul['SCANPOS_G0'].data.shape[0] == hdul[1].data.shape[0]

    def test_derive_positions(self, capsys):
        filename = test_files()[0]
        hdul = fits.open(filename)
        nsamp = hdul[1].data.shape[0]
        header = hdul[0].header
        header['INSTMODE'] = 'OTF_TP'
        header['C_AMP'] = 0
        header['UNIXSTRT'] = 0.0
        header['OTFSTART'] = 0.0
        header['TRK_DRTN'] = (nsamp - 1) * header['ALPHA']
        header['OBSLAMV'] = 1.0 / header['ALPHA']
        header['OBSBETV'] = 0.0
        params = u.get_split_params(hdul)

        result = u._derive_positions(hdul.copy(), params)
        scanpos = result['SCANPOS'].data
        # no errors, use all data
        assert 'Bad OTF keywords' not in capsys.readouterr().err
        assert len(scanpos) == nsamp
        # obsbet speed is 0, so dbet is same as header
        assert np.allclose(scanpos['DBET_MAP'], header['DBET_MAP'])
        # obslam speed is 1/frame
        assert np.allclose(scanpos['DLAM_MAP'],
                           header['DLAM_MAP'] + np.arange(nsamp, dtype=float))
        assert scanpos['FLAG'].sum() == nsamp

        # also check obslam=0, obsbet=1
        header['OBSLAMV'] = 0.0
        header['OBSBETV'] = 1.0 / header['ALPHA']
        result = u._derive_positions(hdul.copy(), params)
        scanpos = result['SCANPOS'].data
        assert np.allclose(scanpos['DBET_MAP'],
                           header['DBET_MAP'] + np.arange(nsamp, dtype=float))
        assert np.allclose(scanpos['DLAM_MAP'], header['DLAM_MAP'])
        assert scanpos['FLAG'].sum() == nsamp

        # change time keys to generate warnings - still uses all data
        header['OTFSTART'] -= 10
        result = u._derive_positions(hdul.copy(), params)
        scanpos = result['SCANPOS'].data
        assert 'Bad OTF keywords: ' \
               'calculated scan start' in capsys.readouterr().err
        assert scanpos['FLAG'].sum() == nsamp

        header['TRK_DRTN'] += 10
        result = u._derive_positions(hdul.copy(), params)
        scanpos = result['SCANPOS'].data
        assert 'Bad OTF keywords: ' \
               'calculated scan end' in capsys.readouterr().err
        assert scanpos['FLAG'].sum() == nsamp

        # change time keys to discard some data
        header['OTFSTART'] = 35 * header['ALPHA']
        header['TRK_DRTN'] = (nsamp - 70) * header['ALPHA']
        result = u._derive_positions(hdul.copy(), params)
        scanpos = result['SCANPOS'].data
        assert 'Bad OTF keywords' not in capsys.readouterr().err
        # two 32-sample ramps less in useful data
        assert scanpos['FLAG'].sum() == nsamp - 64

        # if less than one ramp, data is not discarded
        header['OTFSTART'] = 11 * header['ALPHA']
        header['TRK_DRTN'] = (nsamp - 20) * header['ALPHA']
        result = u._derive_positions(hdul.copy(), params)
        scanpos = result['SCANPOS'].data
        assert 'Bad OTF keywords' not in capsys.readouterr().err
        assert scanpos['FLAG'].sum() == nsamp

    def test_write(self, tmpdir):
        hdul = self.fake_hdul()
        result = u.split_grating_and_chop(hdul, write=True, outdir=str(tmpdir))
        for fn in result:
            assert isinstance(fn, str)
            assert fn.startswith(str(tmpdir))
            assert os.path.isfile(fn)
