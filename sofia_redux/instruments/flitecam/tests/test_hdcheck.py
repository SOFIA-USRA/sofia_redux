# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import configobj
import pytest

import sofia_redux.instruments.flitecam as fdrp
from sofia_redux.instruments.flitecam.hdcheck import hdcheck
from sofia_redux.instruments.flitecam.tests.resources import \
    raw_testdata, raw_specdata


@pytest.fixture(scope='function')
def kwfile():
    pth = os.path.join(os.path.dirname(fdrp.__file__), 'data', 'keyword_files')
    kwfile = os.path.join(pth, 'header_req_ima.cfg')
    return kwfile


@pytest.fixture(scope='function')
def gri_kwfile():
    pth = os.path.join(os.path.dirname(fdrp.__file__), 'data', 'keyword_files')
    kwfile = os.path.join(pth, 'header_req_gri.cfg')
    return kwfile


class TestHdcheck(object):

    def test_ima_data(self, kwfile):
        hdul = raw_testdata()
        result = hdcheck([hdul[0].header], kwfile)
        assert result is True

    def test_gri_data(self, gri_kwfile):
        hdul = raw_specdata()
        result = hdcheck([hdul[0].header], gri_kwfile)
        assert result is True

    def test_list(self, kwfile):
        hdul = raw_testdata()
        hdr = []
        for i in range(4):
            hdr.append(hdul[0].header.copy())
        result = hdcheck(hdr, kwfile)
        assert result is True

    def test_mismatch(self, gri_kwfile):
        hdul1 = raw_testdata()
        hdul2 = raw_specdata()
        hdr = [hdul1[0].header, hdul2[0].header]
        result = hdcheck(hdr, gri_kwfile)
        assert result is False

    def test_errors(self, kwfile, tmpdir, capsys):
        hdul = raw_testdata()
        hdr = hdul[0].header
        hdrl = [hdr]

        # bad header
        assert not hdcheck('bad', kwfile)
        assert 'Could not read FITS header' in capsys.readouterr().err

        # missing keyword file
        with pytest.raises(IOError):
            hdcheck(hdrl, 'badfile')
        assert 'invalid file name' in capsys.readouterr().err

        # bad file format
        badfile = tmpdir.join('badfile.txt')
        badfile.write('badval\n')
        with pytest.raises(configobj.ParseError):
            hdcheck(hdrl, badfile)
        assert 'Error while loading' in capsys.readouterr().err

        # good file, but various missing keywords
        goodfile = tmpdir.join('goodfile.txt')
        goodfile.write('[TEST1]\n')

        # keyword is assumed required, not found
        assert not hdcheck(hdrl, goodfile)
        assert 'Required keyword TEST1 not found' in capsys.readouterr().err

        # error in enum spec in config file
        goodfile.write('[TEST1]\n'
                       'dtype = float\n'
                       '[[drange]]\n'
                       'enum = a, b, c')
        hdr['TEST1'] = 1.0
        with pytest.raises(ValueError):
            hdcheck(hdrl, goodfile)
        assert 'Error in header configuration' in capsys.readouterr().err

        # error in min spec
        goodfile.write('[TEST1]\n'
                       'dtype = float\n'
                       '[[drange]]\n'
                       'min = a')
        with pytest.raises(ValueError):
            hdcheck(hdrl, goodfile)
        assert 'Error in header configuration' in capsys.readouterr().err

        # error in max spec
        goodfile.write('[TEST1]\n'
                       'dtype = float\n'
                       '[[drange]]\n'
                       'max = a')
        with pytest.raises(ValueError):
            hdcheck(hdrl, goodfile)
        assert 'Error in header configuration' in capsys.readouterr().err

    def test_dtypes(self, tmpdir, capsys):
        hdul = raw_testdata()
        hdr = hdul[0].header
        hdrl = [hdr]
        reqfile = tmpdir.join('reqfile.txt')

        # dtype bool
        reqfile.write('[TEST1]\n'
                      'dtype = bool')

        # wrong type
        hdr['TEST1'] = 'strval'
        assert not hdcheck(hdrl, reqfile)
        assert 'wrong type str; should be bool' in capsys.readouterr().err

        # right type
        hdr['TEST1'] = True
        assert hdcheck(hdrl, reqfile)

        # dtype float
        reqfile.write('[TEST1]\n'
                      'dtype = float')
        # wrong type
        hdr['TEST1'] = 'strval'
        assert not hdcheck(hdrl, reqfile)
        assert 'wrong type str; should be float' in capsys.readouterr().err

        # right type: should be float, but int is okay
        hdr['TEST1'] = 1
        assert hdcheck(hdrl, reqfile)

        # dtype int
        reqfile.write('[TEST1]\n'
                      'dtype = int')
        # wrong type
        hdr['TEST1'] = 1.0
        assert not hdcheck(hdrl, reqfile)
        assert 'wrong type float; should be int' in capsys.readouterr().err

        # right type
        hdr['TEST1'] = 1
        assert hdcheck(hdrl, reqfile)

    def test_drange_enum(self, tmpdir, capsys):
        hdul = raw_testdata()
        hdr = hdul[0].header
        hdrl = [hdr]
        reqfile = tmpdir.join('reqfile.txt')

        # dtype bool, enum
        reqfile.write('[TEST1]\n'
                      'dtype = bool\n'
                      '[[drange]]\n'
                      'enum = True\n')

        # wrong value
        hdr['TEST1'] = False
        assert not hdcheck(hdrl, reqfile)
        assert 'wrong value False; should be in [True]' \
            in capsys.readouterr().err

        # right value
        hdr['TEST1'] = True
        assert hdcheck(hdrl, reqfile)

        # dtype str, enum
        reqfile.write('[TEST1]\n'
                      'dtype = str\n'
                      '[[drange]]\n'
                      'enum = val1, val2\n')

        # wrong value
        hdr['TEST1'] = 'val3'
        assert not hdcheck(hdrl, reqfile)
        assert "wrong value val3; should be in " \
               "['VAL1', 'VAL2']" in capsys.readouterr().err

        # right value
        hdr['TEST1'] = 'val2'
        assert hdcheck(hdrl, reqfile)

    def test_drange_minmax(self, tmpdir, capsys):
        hdul = raw_testdata()
        hdr = hdul[0].header
        hdrl = [hdr]
        reqfile = tmpdir.join('reqfile.txt')

        # dtype bool, enum
        reqfile.write('[TEST1]\n'
                      ' dtype = int\n'
                      ' [[drange]]\n'
                      '  min = 1\n'
                      '  max = 3\n')

        # wrong value - min
        hdr['TEST1'] = 0
        assert not hdcheck(hdrl, reqfile)
        assert 'wrong value 0; should be >= 1' in capsys.readouterr().err

        # wrong value - max
        hdr['TEST1'] = 4
        assert not hdcheck(hdrl, reqfile)
        assert 'wrong value 4; should be <= 3' in capsys.readouterr().err

        # right value
        for val in [1, 2, 3]:
            hdr['TEST1'] = val
            assert hdcheck(hdrl, reqfile)

    def test_req_category(self, tmpdir, capsys):
        hdul = raw_testdata()
        hdr = hdul[0].header
        hdrl = [hdr]
        reqfile = tmpdir.join('reqfile.txt')

        # nodding keyword only
        reqfile.write('[TEST1]\n'
                      'requirement = nodding\n')

        # missing value, mismatched category: passes
        hdr['NODDING'] = False
        assert hdcheck(hdrl, reqfile)

        # missing value, matched category: fails
        hdr['NODDING'] = True
        assert not hdcheck(hdrl, reqfile)
        assert 'Required keyword TEST1 not found' in capsys.readouterr().err

        # value present, matched category: passes
        hdr['TEST1'] = 'present'
        assert hdcheck(hdrl, reqfile)
