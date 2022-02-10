# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.tests.resources import DRPTestCase, \
    pol_raw_data, intcal_raw_data, add_col, del_col


class TestPrepare(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepPrepare()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_read_pixscal(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepPrepare()
        step.datain = df
        step.runstart(df, {})

        expected = step.getarg('pixscalist')[-1]

        # test defaults
        result = step.read_pixscal()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_pixscal()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_pixscal()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'pixscalist': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_pixscal()
        capt = capsys.readouterr()
        assert 'Need pixscal values for all wavebands' in capt.err

    def test_colnames(self, tmpdir, capsys):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepPrepare()

        # default output
        out = step(df.copy(), colrename='')

        # invalid column name format
        out1 = step(df.copy(), colrename='badval')
        capt = capsys.readouterr()
        assert 'Invalid colrename item' in capt.err
        assert out1.table.names == out.table.names

        # good format, missing column
        with pytest.raises(ValueError):
            step(df.copy(), colrename='badval->newval')

        # bad detcounts table name
        with pytest.raises(ValueError):
            step(df.copy(), colrename='', detcounts='badval')
        capt = capsys.readouterr()
        assert 'Table badval not found in raw data' in capt.err
        step = StepPrepare()

        # try to rename a multi-dim column -- raises error
        with pytest.raises(ValueError):
            step(df.copy(), colrename='FluxJumps->test')
        capt = capsys.readouterr()
        assert 'cannot rename' in capt.err

        # try to rename to a column already in data
        out1 = step(df.copy(), colrename='sofiaChopR->sofiaChopS')
        capt = capsys.readouterr()
        assert 'Column sofiaChopS already in output data, ' \
               'replace is ignored' in capt.err
        assert out1.table.names == out.table.names

        # missing hwpcounts column
        with pytest.raises(ValueError):
            step(df.copy(), colrename='', hwpcounts='badval')
        capt = capsys.readouterr()
        assert 'Column badval not found' in capt.err

    def test_replacenod(self, tmpdir, capsys):
        hdul = pol_raw_data()

        # add a 'Nod Offset' column to raw data --
        # is replaced if replacenod=True
        hdul = add_col(hdul, 'Nod Offset', 'NOD_OFF')

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepPrepare()

        # set replacenod True -- replaces nod offset column
        out1 = step(df.copy(), colrename='', replacenod=True)
        capt = capsys.readouterr()
        assert 'Using original Nod Offset' not in capt.err
        assert not np.allclose(out1.table['Nod Offset'],
                               df.table['Nod Offset'])

        # also check that it doesn't take it from a rename parameter
        # and that it will use obsra/obsdec if telra/teldec not present
        df1 = df.copy()
        df1.tabledelcol('Nod Offset')
        df1.delheadval(['TELRA', 'TELDEC'])
        out1 = step(df1, colrename='ai23->Nod Offset', replacenod=True)
        capt = capsys.readouterr()
        assert 'Using original Nod Offset' not in capt.err
        assert not np.allclose(out1.table['Nod Offset'], df.table['ai23'])
        assert 'TELRA AND TELDEC NOT DEFINED' in capt.err
        assert 'WILL USE OBSRA AND OBSDEC' in capt.err

        # now replacenod=False, should be used as is
        out2 = step(df.copy(), colrename='', replacenod=False)
        capt = capsys.readouterr()
        assert 'Using original Nod Offset' in capt.err
        assert np.allclose(out2.table['Nod Offset'], df.table['Nod Offset'])

        # with replacenod=True, missing RA/Dec columns should raise an error
        df1 = df.copy()
        df1.tabledelcol('RA')
        with pytest.raises(ValueError):
            step(df1, replacenod=True)
        capt = capsys.readouterr()
        assert 'RA and DEC columns not found' in capt.err

    def test_replacechop(self, tmpdir, capsys):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepPrepare()

        # default output
        out = step(df.copy(), colrename='', chpoffsofiars=True)
        capt = capsys.readouterr()
        assert 'Assigning Chop Offset to SofiaChopR + SofiaChopS' in capt.out
        assert 'Chop Offset' in out.table.names

        # turn off chop replace --
        # raises error without colrename, okay with
        with pytest.raises(ValueError):
            step(df.copy(), colrename='', chpoffsofiars=False)
        capt = capsys.readouterr()
        step = StepPrepare()
        assert 'Use colrename to define Chop Offset' in capt.err
        out1 = step(df.copy(), colrename='sofiaChopR->Chop Offset',
                    chpoffsofiars=False)
        assert 'Chop Offset' in out1.table.names

        # turn on chop replace, but missing columns entirely
        df1 = df.copy()
        df1.tabledelcol('sofiaChopR')
        with pytest.raises(ValueError):
            step(df1, colrename='', chpoffsofiars=True)
        capt = capsys.readouterr()
        assert 'SofiaChopR and S columns not found' in capt.err

        # check for zero-ed chopsync signal --
        # delete, add column to set to zero
        hdul = del_col(hdul, 'sofiaChopSync')
        hdul = add_col(hdul, 'sofiaChopSync', 'sofiaChopR', fill=0.0)
        hdul.writeto(ffile, overwrite=True)
        df1 = DataFits(ffile)
        step(df1, colrename='', chpoffsofiars=True)
        capt = capsys.readouterr()
        assert 'Chop Sync signal may be missing' in capt.err

        # set chopr to zero, chops matches sync
        hdul = del_col(hdul, 'sofiaChopR')
        hdul = del_col(hdul, 'sofiaChopSync')
        hdul = add_col(hdul, 'sofiaChopR', 'sofiaChopS', fill=0.0)
        hdul = add_col(hdul, 'sofiaChopSync', 'sofiaChopS')
        hdul.writeto(ffile, overwrite=True)
        df1 = DataFits(ffile)
        step(df1, colrename='', chpoffsofiars=True)
        capt = capsys.readouterr()
        assert 'Assigning Chop Offset to SofiaChopS - SofiaChopR' in capt.out

        # same, but swap R and S
        hdul = del_col(hdul, 'sofiaChopR')
        hdul = del_col(hdul, 'sofiaChopS')
        hdul = add_col(hdul, 'sofiaChopS', 'sofiaChopSync', fill=0.0)
        hdul = add_col(hdul, 'sofiaChopR', 'sofiaChopSync')
        hdul.writeto(ffile, overwrite=True)
        df1 = DataFits(ffile)
        step(df1, colrename='', chpoffsofiars=True)
        capt = capsys.readouterr()
        assert 'Assigning Chop Offset to SofiaChopR - SofiaChopS' in capt.out

        # chop sync out of phase with both
        hdul = del_col(hdul, 'sofiaChopS')
        hdul = del_col(hdul, 'sofiaChopSync')
        hdul = add_col(hdul, 'sofiaChopS', 'sofiaChopR')
        hdul = add_col(hdul, 'sofiaChopSync', 'sofiaChopR',
                       fill=-1 * df.table['sofiaChopSync'])
        hdul.writeto(ffile, overwrite=True)
        df1 = DataFits(ffile)
        step(df1, colrename='', chpoffsofiars=True)
        capt = capsys.readouterr()
        assert 'Assigning Chop Offset to ' \
               '-1*(SofiaChopR + SofiaChopS)' in capt.out

    def test_labmode(self, tmpdir, capsys):
        hdul = intcal_raw_data()

        # add some nans to azimuth
        hdul[2].data['AZ'] *= np.nan

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        step = StepPrepare()

        # default
        out = step(df.copy(), labmode=True)
        assert out.getheadval('OBSRA') == 1.0
        assert out.getheadval('OBSDEC') == 1.0
        capt = capsys.readouterr()
        assert 'NaNs and were substituted by zeros' in capt.err

        # run with traceshift
        out1 = step(df.copy(), labmode=True, traceshift=2)
        # most columns are shifted
        assert np.allclose(out1.table['Chop Offset'],
                           out.table['Chop Offset'][:-2])
        # flux data is not
        assert np.allclose(out1.table['R Array'],
                           out.table['R Array'][2:])

    def test_dropouts(self, tmpdir, capsys):
        hdul = pol_raw_data()

        # add some dropouts to RA/Dec
        hdul[2].data['RA'][::10] = 0.0

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        step = StepPrepare()

        # no dropout handling
        out = step(df.copy(), removedropouts=False)
        assert len(out.table) == len(df.table)
        capt = capsys.readouterr()
        assert 'Remove Dropouts' not in capt.err

        # with dropout handling
        out = step(df.copy(), removedropouts=True)
        assert len(out.table) < len(df.table)
        capt = capsys.readouterr()
        assert 'Remove Dropouts' in capt.err
