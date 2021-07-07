# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.hdcheck as u
from sofia_redux.instruments.forcast.hdrequirements import hdrequirements


def fake_fits(tmpdir, data=None, keywords=None):
    tempfile = str(tmpdir.join('test.fits'))
    if data is None:
        data = np.arange(100)
    hdul = fits.HDUList(fits.PrimaryHDU(data))
    if keywords is not None:
        for keyword, value in keywords.items():
            hdul[0].header[keyword] = value
    hdul.writeto(tempfile, overwrite=True)
    return tempfile


class TestHdcheck(object):

    def test_validate_condition(self):
        equalities = ['==', '!=', '<=', '<', '>=', '>']
        header = fits.header.Header()
        header['TESTBOOL'] = True
        header['TESTINT'] = 2
        header['TESTFLT'] = 2.0
        header['TESTSTR'] = 'bar'

        expected_true = [True, False, True, False, True, False]
        expected_false = [False, True, False, False, True, True]
        for equality, ok, fail in zip(equalities, expected_true,
                                      expected_false):
            assert u.validate_condition(header,
                                        ('TESTBOOL', equality, '1')) == ok
            assert u.validate_condition(header,
                                        ('TESTBOOL', equality, '0')) == fail

        expected_gt = [False, True, True, True, False, False]
        expected_lt = [False, True, False, False, True, True]
        for equality, lt, gt in zip(equalities, expected_gt, expected_lt):
            assert u.validate_condition(header,
                                        ('TESTINT', equality, '3')) == lt
            assert u.validate_condition(header,
                                        ('TESTINT', equality, '0')) == gt
            assert u.validate_condition(header,
                                        ('TESTFLT', equality, '3')) == lt
            assert u.validate_condition(header,
                                        ('TESTFLT', equality, '-1')) == gt

        assert u.validate_condition(header, ('TESTSTR', '==', 'bar'))
        assert not u.validate_condition(header, ('TESTSTR', '==', 'baz'))
        assert u.validate_condition(header, ('TESTSTR', '>', ''))

    def test_validate_condition_errors(self, capsys):
        header = fits.header.Header()
        header['TESTBOOL'] = True
        header['TESTINT'] = 2
        header['TESTFLT'] = 2.0
        header['TESTSTR'] = 'bar'

        # empty condition - returns true
        assert u.validate_condition(header, ())

        # bad header - return false
        assert not u.validate_condition(None, ('TESTINT', '==', '2'))
        capt = capsys.readouterr()
        assert 'invalid fits header' in capt.err.lower()

        # bad condition - return false
        assert not u.validate_condition(header, ('TESTINT', 'BADVAL'))
        capt = capsys.readouterr()
        assert 'invalid condition' in capt.err.lower()

        # bad comparison - return false
        assert not u.validate_condition(header, ('TESTINT', 'BADVAL', '2'))
        capt = capsys.readouterr()
        assert 'invalid comparison' in capt.err.lower()

        # bad comparison type -- return false
        assert not u.validate_condition(header, ('TESTINT', '==', 'BADVAL'))
        capt = capsys.readouterr()
        assert 'could not convert' in capt.err.lower()

    def test_validate_compound_condition(self):
        header = fits.header.Header()
        header['TTRUE'] = True
        header['TFALSE'] = False

        compound = [[('TTRUE', '==', '1'), ('TFALSE', '==', '1')]]  # False
        assert not u.validate_compound_condition(header, compound)

        compound = [[('TTRUE', '==', '1'), ('TFALSE', '==', '0')]]  # True
        assert u.validate_compound_condition(header, compound)

        compound = [[('TTRUE', '==', '1'), ('TFALSE', '==', '1')],  # False
                    [('TTRUE', '==', '0'), ('TFALSE', '==', '0')]]  # False
        assert not u.validate_compound_condition(header, compound)

        compound = [[('TTRUE', '==', '1'), ('TFALSE', '==', '0')],  # True
                    [('TTRUE', '==', '1'), ('TFALSE', '==', '0')]]  # True
        assert u.validate_compound_condition(header, compound)

        compound = [[('TTRUE', '==', '0'), ('TFALSE', '==', '1')],  # False
                    [('TTRUE', '==', '1'), ('TFALSE', '==', '0')]]  # True
        assert u.validate_compound_condition(header, compound)

        compound = [[('TTRUE', '==', '1'), ('TFALSE', '==', '0')],  # True
                    [('TTRUE', '==', '0'), ('TFALSE', '==', '1')]]  # False
        assert u.validate_compound_condition(header, compound)

    def test_validate_keyrow(self):
        header = fits.header.Header()
        keywords = hdrequirements()
        header['TTRUE'] = True
        header['TFALSE'] = False
        header['CHPAMP1'] = 90
        header['CHOPPING'] = True
        keyrow = keywords.loc['CHPAMP1'].copy()
        assert u.validate_keyrow(header, keyrow, dripconf=True)
        assert u.validate_keyrow(header, keyrow)
        keyrow.loc['type'] = str
        assert not u.validate_keyrow(header, keyrow)
        assert not u.validate_keyrow(header, keyrow, dripconf=True)
        header['CHPAMP1'] = 90
        keyrow.loc['type'] = int
        keyrow['enum'] = ['90', '91']
        assert u.validate_keyrow(header, keyrow)
        keyrow['enum'] = ['91', '92', '93']
        assert not u.validate_keyrow(header, keyrow)
        keyrow['enum'] = []
        keyrow['min'] = 80.
        keyrow['max'] = 100.
        assert u.validate_keyrow(header, keyrow)
        keyrow['min'] = 91.
        assert not u.validate_keyrow(header, keyrow)
        keyrow['min'] = 80.
        keyrow['max'] = 89.
        assert not u.validate_keyrow(header, keyrow)

    def test_validate_keyrow_errors(self, capsys):
        header = fits.header.Header()
        keywords = hdrequirements()
        header['TTRUE'] = True
        header['TFALSE'] = False
        header['CHPAMP1'] = 90
        header['CHOPPING'] = True
        keyrow = keywords.loc['CHPAMP1'].copy()

        # bad header - return false
        assert not u.validate_keyrow(None, keyrow)
        capt = capsys.readouterr()
        assert 'not an astropy header' in capt.err.lower()

        # bad keyrow - return false
        assert not u.validate_keyrow(header, None)
        capt = capsys.readouterr()
        assert 'not a pandas series' in capt.err.lower()

        # missing columns - return false
        bad_keyrow = keyrow.drop('required')
        assert not u.validate_keyrow(header, bad_keyrow)
        capt = capsys.readouterr()
        assert 'columns missing' in capt.err.lower()

        # bad enum values
        """
        keyrow['type'] = int
        keyrow['enum'] = ['a', 'b', 'c']
        assert not u.validate_keyrow(header, keyrow)
        capt = capsys.readouterr()
        assert 'cannot be converted' in capt.err
        """

    def test_validate_header(self):
        header = fits.header.Header()
        keywords = hdrequirements()
        keywords['required'] = False
        keywords.at['NAXIS1', 'required'] = True
        assert not u.validate_header(header, keywords)
        header['NAXIS1'] = 256
        assert u.validate_header(header, keywords)

    def test_validate_header_errors(self, capsys):
        header = fits.header.Header()
        keywords = hdrequirements()
        keywords['required'] = False
        keywords.at['NAXIS1', 'required'] = True

        # bad header
        assert not u.validate_header(None, keywords)
        capt = capsys.readouterr()
        assert 'header' in capt.err.lower()

        # bad keywords
        assert not u.validate_header(header, None)
        capt = capsys.readouterr()
        assert 'keywords' in capt.err.lower()

    def test_validate_file(self, tmpdir):
        testfile = fake_fits(tmpdir)
        keywords = hdrequirements()
        keywords['required'] = False
        keywords.at['NAXIS2', 'required'] = True
        keywords.at['NAXIS2', 'enum'] = [10]
        assert not u.validate_file(testfile, keywords)
        testfile = fake_fits(tmpdir, data=np.zeros((10, 10)))
        assert u.validate_file(testfile, keywords)
        os.remove(testfile)

    def test_validate_file_errors(self, tmpdir, capsys):
        testfile = fake_fits(tmpdir)
        keywords = hdrequirements()
        keywords['required'] = False
        keywords.at['NAXIS2', 'required'] = True
        keywords.at['NAXIS2', 'enum'] = [10]

        # bad filename type
        assert not u.validate_file(1, keywords)
        capt = capsys.readouterr()
        assert 'filename must be a string' in capt.err.lower()

        # bad keywords
        assert not u.validate_file(testfile, 1)
        capt = capsys.readouterr()
        assert 'keywords must be' in capt.err.lower()

        # bad file
        assert not u.validate_file(testfile + 'tmp.fits', keywords)
        capt = capsys.readouterr()
        assert 'not a file' in capt.err.lower()

        os.remove(testfile)

    def test_hdcheck(self, tmpdir):
        testfile = [fake_fits(tmpdir, data=np.zeros((10, 10)))]
        keywords = hdrequirements()
        keywords['required'] = False
        keywords.at['NAXIS2', 'required'] = True
        keywords.at['NAXIS2', 'enum'] = [10]
        files = testfile * 2
        assert len(files) > 1
        assert not u.hdcheck(files)
        os.remove(testfile[0])

    def test_hdcheck_errors(self, capsys, tmpdir):
        fname = fake_fits(tmpdir, data=np.zeros((10, 10)))
        reqfile = tmpdir.join('req.txt')
        reqfile.write('name  condition    type    enum    format  min max')
        reqfile.write('NAXIS1  1  3 .  . 1 9999')

        # test with string value -- okay
        assert u.hdcheck(fname, kwfile=str(reqfile))

        # bad filename types
        flist = [1, 2, 3]
        assert not u.hdcheck(flist, kwfile=str(reqfile))
        capt = capsys.readouterr()
        assert 'must specify file name(s) as strings' in capt.err.lower()
        flist = 1
        assert not u.hdcheck(flist, kwfile=str(reqfile))
        capt = capsys.readouterr()
        assert 'must specify file name(s) as strings' in capt.err.lower()

        # missing file
        assert not u.hdcheck('badfile.fits', kwfile=str(reqfile))
        capt = capsys.readouterr()
        assert 'file does not exist' in capt.err.lower()

        # non-fits file
        badfile = tmpdir.join('badfile.fits')
        badfile.write('BADVAL')
        assert not u.hdcheck(str(badfile), kwfile=str(reqfile))
        capt = capsys.readouterr()
        assert 'could not read fits header' in capt.err.lower()

        # dohdcheck -- skip validation, returns True
        # (using NAXIS1 here as proxy for doinhdch keyword in
        # config -- it shouldn't return '1' from getpar)
        assert u.hdcheck(fname, dohdcheck='NAXIS1',
                         kwfile=str(reqfile))

        os.remove(fname)
