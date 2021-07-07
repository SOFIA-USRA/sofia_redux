# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.fifi_ls.get_response \
    import (clear_response_cache, get_response_from_cache,
            store_response_in_cache, get_response)
from sofia_redux.instruments.fifi_ls.tests.resources import FIFITestCase


class TestGetResponse(FIFITestCase):

    def get_header(self):
        header = fits.Header()
        header['CHANNEL'] = 'RED'
        header['G_WAVE_R'] = 100.0
        header['G_WAVE_B'] = 100.0
        header['G_ORD_B'] = 1
        header['DICHROIC'] = 105
        return header

    def test_success(self):
        header = self.get_header()
        red = get_response(header)
        assert isinstance(red, np.ndarray)
        header['CHANNEL'] = 'BLUE'
        blue = get_response(header)
        assert isinstance(blue, np.ndarray)
        assert not np.allclose(np.nansum(red), np.nansum(blue))

    def test_error(self, capsys):
        # bad header
        result = get_response(None)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid header' in capt.err

        # now start with empty header
        header = fits.Header()

        # add missing keys one at a time
        header['DATE-OBS'] = 'BADVAL'
        for key in ['CHANNEL', 'G_ORD_B', 'DICHROIC']:
            result = get_response(header)
            assert result is None
            capt = capsys.readouterr()
            assert 'Header missing {}'.format(key) in capt.err
            header[key] = 'TESTVAL'

        # now it has all others, will complain about bad blue order
        header['CHANNEL'] = 'BLUE'
        result = get_response(header)
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid blue grating order' in capt.err

    def test_filename(self, tmpdir, capsys):
        header = self.get_header()

        # default
        default = get_response(header)

        # missing file
        result = get_response(header, filename='test.fits')
        capt = capsys.readouterr()
        assert 'Could not find file' in capt.err
        assert 'Using default' in capt.err
        assert np.allclose(result, default)

        # provide a good filename, bad file
        respfile = tmpdir.join('test.fits')
        respfile.write('Test data\n')
        result = get_response(header, filename=str(respfile))
        assert result is None
        capt = capsys.readouterr()
        assert 'Invalid data' in capt.err

    def test_default_table(self, tmpdir, mocker, capsys):
        # test for missing/bad defaults file
        os.makedirs(tmpdir.join('data', 'response_files'))

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.fifi_ls.__file__',
                     str(mock_file))

        # missing default file
        result = get_response(self.get_header())
        assert result is None
        capt = capsys.readouterr()
        assert 'Cannot read response default file' in capt.err

        # write a file that is unreadable
        default = tmpdir.join('data', 'response_files', 'response_default.txt')
        default.write('test\n')
        result = get_response(self.get_header())
        assert result is None
        capt = capsys.readouterr()
        assert 'Cannot read response default file' in capt.err

        # write a file that is readable but contains no useful data
        default.write('19990101\tb1\t105\tresponse_files/test.fits\n')
        result = get_response(self.get_header())
        assert result is None
        capt = capsys.readouterr()
        assert 'Unable to find response file' in capt.err

    def test_response_cache(self, tmpdir):

        tempdir = str(tmpdir.mkdir('test_get_response'))
        responsefile = os.path.join(tempdir, 'test01')

        with open(responsefile, 'w') as f:
            print('this is the response file', file=f)

        response = np.arange(10)
        store = responsefile, response

        clear_response_cache()
        assert get_response_from_cache(responsefile) is None
        store_response_in_cache(*store)

        # It should be in there now
        result = get_response_from_cache(responsefile)
        assert np.allclose(result, response)

        # Check it's still in there
        assert get_response_from_cache(responsefile) is not None

        # Modify the file - the result should be None,
        # indicating it was removed from the file and
        # should be processed and stored again.
        time.sleep(0.5)
        with open(responsefile, 'w') as f:
            print('a modification', file=f)

        assert get_response_from_cache(responsefile) is None

        # Store the data again
        store_response_in_cache(*store)

        # Make sure it's there
        assert get_response_from_cache(responsefile) is not None

        # Check clear works
        clear_response_cache()
        assert get_response_from_cache(responsefile) is None

        # Store then delete the response file -- check that bad file
        # can't be retrieved
        store_response_in_cache(*store)
        assert get_response_from_cache(responsefile) is not None
        os.remove(responsefile)
        assert get_response_from_cache(responsefile) is None

    def test_order_filter(self, tmpdir, mocker, capsys):
        # make a default file with all potential orders/filters
        os.makedirs(tmpdir.join('data', 'response_files'))

        # mock the data path
        mock_file = tmpdir.join('test_file')
        mocker.patch('sofia_redux.instruments.fifi_ls.__file__',
                     str(mock_file))

        # write a defaults file
        default = tmpdir.join('data', 'response_files', 'response_default.txt')
        default.write('99999999\tr\t105\tresponse_files/r.fits\n'
                      '99999999\tb1\t105\tresponse_files/b1.fits\n'
                      '99999999\tb2\t105\tresponse_files/b2.fits\n'
                      '99999999\tb12\t105\tresponse_files/b12.fits\n'
                      '99999999\tb21\t105\tresponse_files/b21.fits\n')

        # write some test fits files with appropriate names
        order_list = ['r', 'b1', 'b2', 'b12', 'b21']
        for order in order_list:
            ffile = str(tmpdir.join('data', 'response_files',
                                    '{}.fits'.format(order)))
            hdul = fits.HDUList(fits.PrimaryHDU(data=np.zeros(10)))
            hdul.writeto(ffile, overwrite=True)

        # check that each mode is retrieved correctly
        header = self.get_header()
        for order in order_list:
            if order[0] == 'r':
                header['CHANNEL'] = 'RED'
            else:
                header['CHANNEL'] = 'BLUE'
            if len(order) > 1:
                header['G_ORD_B'] = order[1]
            if len(order) > 2:
                header['G_FLT_B'] = order[2]
            elif 'G_FLT_B' in header:
                del header['G_FLT_B']

            get_response(header)
            assert '{}.fits'.format(order) in header['RSPNFILE']

        # also explicitly check blue 2 filter 2 - should
        # same as if FLT is not specified
        header['CHANNEL'] = 'BLUE'
        header['G_ORD_B'] = 2
        header['G_FLT_B'] = 2
        get_response(header)
        assert 'b2.fits' in header['RSPNFILE']

        # and also the same for an invalid value
        header['G_FLT_B'] = -9999
        get_response(header)
        assert 'b2.fits' in header['RSPNFILE']
