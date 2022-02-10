# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.steppolvec import StepPolVec
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestPolVec(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        step = StepPolVec()
        out = step(df)

        assert isinstance(out, DataFits)

    def test_read_eff(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepPolVec()
        step.datain = df
        step.runstart(df, {})

        expected = step.getarg('eff')[-1]

        # test defaults
        result = step.read_eff()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_eff()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_eff()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'eff': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_eff()
        capt = capsys.readouterr()
        assert 'Need efficiency values for all wavebands' in capt.err

    def test_nhwp(self, capsys):
        df = DataFits()
        df.setheadval('NHWP', 1)
        step = StepPolVec()
        step(df)
        capt = capsys.readouterr()
        assert 'Only 1 HWP, so skipping step' in capt.out
