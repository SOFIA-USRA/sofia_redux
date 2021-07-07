# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.fifi_ls.subtract_chops as u
from sofia_redux.instruments.fifi_ls.tests.resources \
    import FIFITestCase, get_chop0_file, get_chop_files


class TestSubtractChops(FIFITestCase):

    def test_subtract_chops(self):
        chop0_file = get_chop0_file()
        u.subtract_chops(chop0_file)

    def test_write(self, tmpdir):
        files = get_chop0_file()
        result = u.subtract_chops(
            files, write=True, outdir=str(tmpdir))
        assert os.path.isfile(result)
        result = u.subtract_chops(
            files, write=False, outdir=str(tmpdir))
        assert isinstance(result, fits.HDUList)

    def test_get_chop_pair(self, tmpdir, capsys):
        chop0_file = get_chop0_file()
        hduls = u.get_chop_pair(chop0_file)
        assert len(hduls) == 2
        for hdul in hduls:
            assert isinstance(hdul, fits.HDUList)

        # bad input, file does not exist
        assert u.get_chop_pair(1) is None
        assert u.get_chop_pair('not/a_file.foobar') is None

        # file is not named correctly
        badfile = tmpdir.join('test.fits')
        badfile.write('test\n')
        assert u.get_chop_pair(str(badfile)) is None
        assert "must contain 'RP0'" in capsys.readouterr().err

        # RP0 is invalid
        rp0 = tmpdir.join('RP0.fits')
        rp0.write('test\n')
        assert u.get_chop_pair(str(rp0)) is None
        assert 'Invalid RP0' in capsys.readouterr().err

        # RP0 is fits, RP1 is missing
        hduls[0][1] = fits.ImageHDU()
        hduls[0].writeto(str(rp0), overwrite=True)
        assert u.get_chop_pair(str(rp0)) is None
        assert 'No RP1 file' in capsys.readouterr().err

        # RP0 is fits, RP1 is invalid
        rp1 = tmpdir.join('RP1.fits')
        rp1.write('test\n')
        assert u.get_chop_pair(str(rp0)) is None
        assert 'Invalid RP1' in capsys.readouterr().err

        # both are fits, but extension is invalid
        hduls[1].writeto(str(rp1), overwrite=True)
        assert u.get_chop_pair(str(rp0)) is None
        assert 'Missing extension: FLUX' in capsys.readouterr().err

        # both have valid flux, one missing a stddev extension
        hduls = u.get_chop_pair(chop0_file)
        hduls[0].writeto(str(rp0), overwrite=True)
        del hduls[1]['STDDEV_G1']
        hduls[1].writeto(str(rp1), overwrite=True)
        assert u.get_chop_pair(str(rp0)) is None
        assert 'Missing extension: STDDEV_G1' in capsys.readouterr().err

        # both valid, one mismatched grating extension
        hduls = u.get_chop_pair(chop0_file)
        del hduls[0][-2:]
        hduls[0][0].header['NGRATING'] -= 1
        hduls[0].writeto(str(rp0), overwrite=True)
        hduls[1].writeto(str(rp1), overwrite=True)
        assert u.get_chop_pair(str(rp0)) is None
        assert 'Differing number of inductosyn' in capsys.readouterr().err

    def test_subtract_extensions(self, capsys):
        chop0_file = get_chop0_file()
        hduls = u.get_chop_pair(chop0_file)

        # default result
        default = u.subtract_extensions(hduls[0], hduls[1], add_only=False)
        assert isinstance(default, fits.HDUList)
        mean_val = np.nanmean(default[1].data)

        # add instead of subtract
        added = u.subtract_extensions(hduls[0], hduls[1], add_only=True)
        added_val = np.nanmean(added[1].data)
        assert added_val > mean_val

        # mismatched inductosyn positions
        hdr = hduls[0][1].header
        hdr['INDPOS'] += 10
        result = u.subtract_extensions(hduls[0], hduls[1])
        assert result is None
        assert 'Inductosyn positions do ' \
               'not line up' in capsys.readouterr().err

        # chopnum = 1 instead of 0: sign swap
        hdr['INDPOS'] -= 10
        hdr['CHOPNUM'] = 1
        result = u.subtract_extensions(hduls[0], hduls[1])
        pos_val = np.nanmean(result[1].data)
        assert np.allclose(pos_val, -mean_val)

    def test_total_power(self, capsys, tmpdir):
        files = get_chop_files()

        # make files with c_amp = 0
        inp = []
        for fn in files:
            hdul = fits.open(fn)
            hdul[0].header['C_AMP'] = 0
            inp.append(hdul)

        # two hduls with c_amp = 0
        result = u.subtract_chops(inp[:2], write=False)
        assert len(result) == 2
        assert result == inp[:2]
        assert 'No chop subtraction' in capsys.readouterr().out

        # four hduls with c_amp = 0, via wrap
        result = u.wrap_subtract_chops(inp, write=False)
        assert len(result) == 4
        assert list(result) == inp
        assert 'No chop subtraction' in capsys.readouterr().out

        # one hdul with c_amp = 0
        result = u.subtract_chops(inp[0], write=False)
        assert len(result) == 1
        assert result[0] == inp[0]
        assert 'No chop subtraction' in capsys.readouterr().out

        # one hdul in list form with c_amp = 0
        result = u.subtract_chops([inp[0]], write=False)
        assert len(result) == 1
        assert result[0] == inp[0]
        assert 'No chop subtraction' in capsys.readouterr().out

        # one chop0 file on disk
        fname = str(tmpdir.join('testRP0.fits'))
        inp[0].writeto(fname)
        result = u.subtract_chops(fname, write=False)
        assert len(result) == 1
        assert isinstance(result[0], fits.HDUList)
        assert 'No chop subtraction' in capsys.readouterr().out

        # set write = True
        result = u.subtract_chops(fname, write=True, outdir=str(tmpdir))
        assert result[0] == fname

    def test_hdul_input(self):
        filename = get_chop_files()
        inp = []
        for fn in filename[0:2]:
            inp.append(fits.open(fn))
        result = u.subtract_chops(inp)
        assert isinstance(result, fits.HDUList)
        # check bunit
        for ext in result[1:]:
            assert ext.header['BUNIT'] == 'adu/s'

    def test_bad_parameters(self, capsys):
        files = get_chop_files()

        # bad output directory
        result = u.subtract_chops(files[0], outdir='badval')
        assert result is None
        capt = capsys.readouterr()
        assert 'does not exist' in capt.err

        # bad filename
        result = u.subtract_chops('badfile.fits', write=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'not a file' in capt.err

    def test_subtract_failure(self, mocker, capsys):
        filename = get_chop_files()[0]

        # mock failure in calculate offsets
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.subtract_chops.'
            'subtract_extensions',
            return_value=None)
        result = u.subtract_chops(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem subtracting extensions' in capt.err

        # mock failure in chop_pair
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.subtract_chops.get_chop_pair',
            return_value=None)
        result = u.subtract_chops(filename)
        assert result is None
        capt = capsys.readouterr()
        assert 'Problem retrieving chop pair' in capt.err

    def test_wrap(self):
        files = get_chop_files()
        output = u.wrap_subtract_chops(files, write=False)
        assert len(output) > 0
        output = u.wrap_subtract_chops(files, write=False, jobs=-1)
        assert len(output) > 0

    def test_wrap_failure(self, capsys, mocker):
        # mock a partial failure
        mocker.patch(
            'sofia_redux.instruments.fifi_ls.subtract_chops.multitask',
            return_value=['test', None])

        # bad files
        result = u.wrap_subtract_chops(None, write=False)
        assert result is None
        capt = capsys.readouterr()
        assert "Invalid input files type" in capt.err

        # real files, but pass only one
        files = get_chop_files()
        u.wrap_subtract_chops(files[0], write=False,
                              allow_errors=False)

        # allow errors
        result = u.wrap_subtract_chops(files, write=False,
                                       allow_errors=True)
        assert len(result) == 1
        assert result[0] == 'test'

        # don't allow errors
        result = u.wrap_subtract_chops(files, write=False,
                                       allow_errors=False)
        assert result is None
        capt = capsys.readouterr()
        assert 'Errors were encountered' in capt.err
