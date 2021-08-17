# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import kref, href

from sofia_redux.instruments.forcast.readfits import readfits, addparent
from sofia_redux.instruments.forcast.tests.resources import \
    raw_testdata, nmc_testdata


class TestReadfits(object):

    def test_success(self, tmpdir):
        test_hdul = raw_testdata()
        testfile = str(tmpdir.join('test.fits'))
        test_hdul.writeto(testfile, overwrite=True)

        data, header = readfits(
            testfile, fitshead=True, variance=True)
        assert isinstance(data, np.ndarray)
        assert not (data == 0).all()
        assert isinstance(header, fits.header.Header)
        assert len(header) > 0
        assert '---------- PIPELINE HISTORY -----------' in [*header.values()]
        assert href in header
        assert kref in header

        # check update header
        upd_hdr = fits.header.Header()
        data, header = readfits(
            testfile, fitshead=True, variance=True,
            update_header=upd_hdr, key='TESTKEY')
        assert 'PARENT1' in upd_hdr
        assert 'TESTKEY' in upd_hdr

        # check stddev
        data_err = readfits(testfile, stddev=True)
        assert np.allclose(data_err[1], np.sqrt(data[1]))

        # check hdul return
        # 1 extension without var/stddev
        data_hdul = readfits(testfile, fitshdul=True)
        assert isinstance(data_hdul, fits.HDUList)
        assert len(data_hdul) == 1
        # 2 extensions with var/stddev
        data_hdul = readfits(testfile, fitshdul=True, variance=True)
        assert len(data_hdul) == 2
        assert not np.allclose(data_hdul['VARIANCE'].data, 0)
        data_hdul = readfits(testfile, fitshdul=True, stddev=True)
        assert len(data_hdul) == 2
        assert not np.allclose(data_hdul['ERROR'].data, 0)

    def test_failure(self):
        assert readfits('this file does not exist') is None

    def test_one_parent(self):
        header = nmc_testdata()['header']
        assert 'PARENT1' not in header
        addparent('TESTING_ADDPARENT', header)
        assert 'PARENT1' in header
        assert header['PARENT1'] == 'TESTING_ADDPARENT'

    def test_two_parents(self):
        header = nmc_testdata()['header']
        assert 'PARENT1' not in header
        addparent('FOO', header)
        assert 'PARENT1' in header
        # adding a parent with the same value as one that already
        # exists should not result in the addition of a new parent
        addparent('FOO', header)
        assert 'PARENT2' not in header
        # test we can add a parent if the value is new
        addparent('BAR', header)
        assert 'PARENT2' in header

    def test_comments(self):
        header = nmc_testdata()['header']
        assert 'PARENT1' not in header
        addparent('FOO', header)
        # test the default comment
        assert header.comments['PARENT1'] == 'id or name of file ' \
                                             'used in the processing'
        # add a new parent with a non-default message
        addparent('BAR', header, comment='test comment')
        assert 'PARENT2' in header
        assert header.comments['PARENT2'] == 'test comment'
