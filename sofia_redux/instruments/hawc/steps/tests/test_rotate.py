# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.steprotate import StepRotate
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestRotate(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_bgs_data()

        # add a couple necessary keywords from other steps
        hdul[0].header['VPOS_ANG'] = 194.64
        hdul[0].header['HWPINIT'] = 5.0

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepRotate()
        out = step(inp)

        assert isinstance(out, DataFits)

    def test_read_angle(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepRotate()
        step.datain = df
        step.runstart(df, {})

        expected = step.getarg('gridangle')[-1]

        # test defaults
        result = step.read_angle()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_angle()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_angle()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'gridangle': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_angle()
        capt = capsys.readouterr()
        assert 'Need grid angle values for all wavebands' in capt.err

    def test_no_hwp(self, capsys):
        df = DataFits()
        df.setheadval('NHWP', 1)
        step = StepRotate()
        # no-op with nhwp=1
        step(df)
        capt = capsys.readouterr()
        assert 'Only 1 HWP, so skipping step' in capt.out

    def test_angles(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        hdul[0].header['VPOS_ANG'] = 194.64
        # 5 degrees apart, will wrap around to negative
        hdul[0].header['HWPINIT'] = 275.0
        hdul[0].header['HWPSTART'] = 280.0

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepRotate()

        # test commanded
        step(inp, hwpzero_tol=3.0,
             hwpzero_option='commanded')
        capt = capsys.readouterr()
        assert 'difference is above the tolerance' in capt.err
        assert 'Will use the commanded value (-80' in capt.err

        # test actual
        step(inp, hwpzero_tol=3.0,
             hwpzero_option='actual')
        capt = capsys.readouterr()
        assert 'difference is above the tolerance' in capt.err
        assert 'Will use the actual value (-85' in capt.err

        # bad value
        with pytest.raises(ValueError):
            step(inp, hwpzero_tol=3.0,
                 hwpzero_option='badval')
        capt = capsys.readouterr()
        assert 'difference is above the tolerance' in capt.err
        assert 'hwpzero_option parameter value must be ' \
               'commanded or actual' in capt.err
