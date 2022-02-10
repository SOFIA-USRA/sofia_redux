# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import configobj
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepcheckhead \
    import StepCheckhead, HeaderValidationError
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data, scan_raw_data


class TestCheckhead(DRPTestCase):
    def test_pol_data(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepCheckhead()
        out = step(df)

        assert isinstance(out, DataFits)

    def test_scan_data(self, tmpdir):
        hdul = scan_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepCheckhead()
        out = step(df)

        assert isinstance(out, DataFits)

    def test_errors(self, tmpdir, capsys):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepCheckhead()

        # missing header def file
        step.runstart(df, {'headerdef': 'badfile.txt'})
        with pytest.raises(IOError):
            step.run()

        # bad file format
        badfile = tmpdir.join('badfile.txt')
        badfile.write('badval\n')
        step.runstart(df, {'headerdef': str(badfile)})
        with pytest.raises(configobj.ParseError):
            step.run()

        # good file, but various missing keywords
        goodfile = tmpdir.join('goodfile.txt')
        goodfile.write('[TEST1]\n')
        step.runstart(df, {'headerdef': str(goodfile)})

        # keyword is assumed required, not found
        with pytest.raises(HeaderValidationError):
            step.run()
        capt = capsys.readouterr()
        assert 'Required keyword <TEST1> not found' in capt.err

        # same, but don't abort (no error raised)
        step.runstart(df, {'headerdef': str(goodfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'Required keyword <TEST1> not found' in capt.err

        # error in enum spec in config file
        goodfile.write('[TEST1]\n'
                       'dtype = float\n'
                       '[[drange]]\n'
                       'enum = a, b, c')
        df.setheadval('TEST1', 1.0)
        step.datain = df
        with pytest.raises(ValueError):
            step.run()
        capt = capsys.readouterr()
        assert 'Error in header configuration file' in capt.err

        # error in min spec
        goodfile.write('[TEST1]\n'
                       'dtype = float\n'
                       '[[drange]]\n'
                       'min = a')
        df.setheadval('TEST1', 1.0)
        step.datain = df
        with pytest.raises(ValueError):
            step.run()
        capt = capsys.readouterr()
        assert 'Error in header configuration file' in capt.err

        # error in max spec
        goodfile.write('[TEST1]\n'
                       'dtype = float\n'
                       '[[drange]]\n'
                       'max = a')
        df.setheadval('TEST1', 1.0)
        step.datain = df
        with pytest.raises(ValueError):
            step.run()
        capt = capsys.readouterr()
        assert 'Error in header configuration file' in capt.err

    def test_dtypes(self, tmpdir, capsys):
        df = DataFits()
        step = StepCheckhead()
        reqfile = tmpdir.join('reqfile.txt')

        # dtype bool
        reqfile.write('[TEST1]\n'
                      'dtype = bool')

        # wrong type
        df.setheadval('TEST1', 'strval')
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'wrong type <str>; should be <bool>' in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # right type
        df.setheadval('TEST1', True)
        step.datain = df
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        # no error
        step.run()

        # dtype float
        reqfile.write('[TEST1]\n'
                      'dtype = float')
        # wrong type
        df.setheadval('TEST1', 'strval')
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'wrong type <str>; should be <float>' in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # right type: float, int okay
        df.setheadval('TEST1', 1)
        step.datain = df
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        # no error
        step.run()

        # dtype long
        reqfile.write('[TEST1]\n'
                      'dtype = long')
        # wrong type
        df.setheadval('TEST1', 1.0)
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'wrong type <float>; should be <long>' in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # right type
        df.setheadval('TEST1', 1)
        step.datain = df
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        # no error
        step.run()

    def test_drange_enum(self, tmpdir, capsys):
        df = DataFits()
        step = StepCheckhead()
        reqfile = tmpdir.join('reqfile.txt')

        # dtype bool, enum
        reqfile.write('[TEST1]\n'
                      'dtype = bool\n'
                      '[[drange]]\n'
                      'enum = True\n')

        # wrong value
        df.setheadval('TEST1', False)
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'wrong value <False>; should be in [True]' in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # right value
        df.setheadval('TEST1', True)
        step.datain = df
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        # no error
        step.run()

        # dtype str, enum
        reqfile.write('[TEST1]\n'
                      'dtype = str\n'
                      '[[drange]]\n'
                      'enum = val1, val2\n')

        # wrong value
        df.setheadval('TEST1', 'val3')
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert "wrong value <val3>; should be in " \
               "['VAL1', 'VAL2']" in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # right value
        df.setheadval('TEST1', 'val2')
        step.datain = df
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        # no error
        step.run()

    def test_drange_minmax(self, tmpdir, capsys):
        df = DataFits()
        step = StepCheckhead()
        reqfile = tmpdir.join('reqfile.txt')

        # dtype bool, enum
        reqfile.write('[TEST1]\n'
                      ' dtype = int\n'
                      ' [[drange]]\n'
                      '  min = 1\n'
                      '  max = 3\n')

        # wrong value - min
        df.setheadval('TEST1', 0)
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'wrong value <0>; should be >= 1' in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # wrong value - max
        df.setheadval('TEST1', 4)
        step.datain = df

        # abort false
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': False})
        step.run()
        capt = capsys.readouterr()
        assert 'wrong value <4>; should be <= 3' in capt.err

        # abort true
        step.runstart(df, {'headerdef': str(reqfile),
                           'abort': True})
        with pytest.raises(HeaderValidationError):
            step.run()

        # right value
        for val in [1, 2, 3]:
            df.setheadval('TEST1', val)
            step.datain = df
            step.runstart(df, {'headerdef': str(reqfile),
                               'abort': True})
            # no error
            step.run()

    def test_filename(self, tmpdir):
        # pol data filenames
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('testpol_1.fits'))
        hdul.writeto(ffile, overwrite=True)
        poldf = DataFits(ffile)
        poldf.config['data'] = {'filenum': r'.*(\d)+.fits'}

        step = StepCheckhead()
        step.datain = poldf
        step.runstart(poldf, {})
        step.run()
        assert os.path.basename(step.dataout.filename) == \
            'F0001_HA_POL_90000101_HAWDHWPD_RAW_1.fits'

        # imaging data filenames
        hdul = scan_raw_data()
        ffile = str(tmpdir.join('testpol_2.fits'))
        hdul.writeto(ffile, overwrite=True)
        scandf = DataFits(ffile)
        scandf.config['data'] = {'filenum': r'.*(\d)+.fits'}

        step.datain = scandf
        step.runstart(scandf, {})
        step.run()
        assert os.path.basename(step.dataout.filename) == \
            'F0001_HA_IMA_90000101_HAWDHWPD_RAW_2.fits'

        # cal data filenames
        caldf = scandf.copy()
        caldf.setheadval('CALMODE', 'INT_CAL')
        step.datain = caldf
        step.runstart(caldf, {})
        step.run()
        assert os.path.basename(step.dataout.filename) == \
            'F0001_HA_CAL_90000101_HAWDHWPD_RAW_2.fits'
