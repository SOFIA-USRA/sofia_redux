# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.steps.stepsplit import StepSplit
from sofia_redux.instruments.hawc.steps.stepcombine import StepCombine
from sofia_redux.instruments.hawc.steps.stepnodpolsub import StepNodPolSub
from sofia_redux.instruments.hawc.steps.stepstokes import StepStokes
from sofia_redux.instruments.hawc.steps.stepwcs import StepWcs
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, pol_raw_data


class TestWCS(DRPTestCase):
    def make_pre_wcs(self, tmpdir):
        """Minimal steps to make data that will pass stokes step."""
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        steps = [StepPrepare, StepDemodulate, StepFlat,
                 StepSplit, StepCombine, StepNodPolSub, StepStokes]
        inp = DataFits(ffile)
        for step in steps:
            s = step()
            inp = s(inp)

        hdul = inp.to_hdulist()
        hdul.writeto(ffile, overwrite=True)
        return ffile

    def test_siso(self, tmpdir):
        ffile = self.make_pre_wcs(tmpdir)
        inp = DataFits(ffile)

        step = StepWcs()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_badfile(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepWcs()
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'No valid table' in capt.err

    def test_read_sibs(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepWcs()
        step.datain = df
        step.runstart(df, {})

        expected = (step.getarg('offsibs_x')[-1],
                    step.getarg('offsibs_y')[-1])

        # test defaults
        result = step.read_sibs()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_sibs()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_sibs()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart(df, {'offsibs_x': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_sibs()
        capt = capsys.readouterr()
        assert 'Need offsibs_x/y values for all wavebands' in capt.err

    def test_erf(self, tmpdir, capsys):
        ffile = self.make_pre_wcs(tmpdir)
        inp = DataFits(ffile)
        step = StepWcs()

        # test a few non-standard keyword conditions
        inp.table['Array VPA'] = 270
        inp.header['CHPANGLE'] = 90
        inp.header['CHPAMP1'] = 100
        inp.header['CHPCRSYS'] = 'erf'
        out = step(inp)
        assert out.header['VPOS_ANG'] == 90

        # erf moves crval, sirf moves crpix
        inp.header['CHPCRSYS'] = 'sirf'
        out1 = step(inp)
        # with chpang=90, only RA/CRPIX1 changes
        assert out1.header['CRVAL1'] > out.header['CRVAL1']
        assert out1.header['CRVAL2'] == out.header['CRVAL2']
        assert out1.header['CRPIX1'] < out.header['CRPIX1']
        assert out1.header['CRPIX2'] == out.header['CRPIX2']

        # invalid chop system
        inp.header['CHPCRSYS'] = 'tarf'
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'not a valid chopping coordinate system' in capt.err

    def test_labmode(self):
        # bare df -- should still pass in lab mode
        df = DataFits()
        df.setheadval('PIXSCAL', 1.0)
        step = StepWcs()

        out = step(df, labmode=True)
        assert 'CRPIX1' in out.header
        assert 'CRVAL1' in out.header
        assert 'CDELT1' in out.header
