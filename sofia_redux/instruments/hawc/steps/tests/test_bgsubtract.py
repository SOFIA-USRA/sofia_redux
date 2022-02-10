# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepbgsubtract import StepBgSubtract
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestBgSubtract(DRPTestCase):
    def make_input(self, nfiles, tmpdir, offset=100, flat=False):
        inp = []
        for i in range(nfiles):
            hdul = pol_bgs_data(idx=i)

            if flat:
                # set all data to same value
                hdul['STOKES I'].data[:, :] = i * offset
                hdul['STOKES Q'].data[:, :] = i * offset
                hdul['STOKES U'].data[:, :] = i * offset
                hdul['ERROR I'].data[:, :] = i * offset + 0.1
                hdul['ERROR Q'].data[:, :] = i * offset + 0.1
                hdul['ERROR U'].data[:, :] = i * offset + 0.1
            else:
                # add an artificial background that differs by file
                hdul['STOKES I'].data += i * offset
                hdul['STOKES Q'].data += i * offset
                hdul['STOKES U'].data += i * offset

            ffile = str(tmpdir.join('test{}.fits'.format(i + i)))
            hdul.writeto(ffile, overwrite=True)
            inp.append(DataFits(ffile))
        return inp

    def test_mimo(self, tmpdir, capsys):
        inp = self.make_input(3, tmpdir)

        step = StepBgSubtract()
        out = step(inp)

        assert isinstance(out, list)
        assert len(out) == len(inp)
        for df in out:
            assert isinstance(df, DataFits)

        capt = capsys.readouterr()
        assert 'Stokes I offset' in capt.out
        assert 'Stokes Q offset' in capt.out
        assert 'Stokes U offset' in capt.out

    def test_read_fwhm(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepBgSubtract()
        step.datain = [df]
        step.runstart([df], {})

        expected = (step.getarg('fwhm')[-1],
                    step.getarg('radius')[-1],
                    step.getarg('cdelt')[-1])

        # test defaults
        result = step.read_fwhm_radius_cdelt()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_fwhm_radius_cdelt()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_fwhm_radius_cdelt()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart([df], {'cdelt': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_fwhm_radius_cdelt()
        capt = capsys.readouterr()
        assert 'Missing radius/fwhm values' in capt.err

    def test_one_file(self, tmpdir, capsys):
        inp = self.make_input(1, tmpdir)
        step = StepBgSubtract()

        # try to run on a single file: should issue message and return
        one_file = step(inp)
        assert np.allclose(one_file[0].data, inp[0].data, equal_nan=True)
        capt = capsys.readouterr()
        assert 'One file only' in capt.out

    def test_combine_flags(self, tmpdir, capsys):
        offset = 100
        inp = self.make_input(2, tmpdir, offset=offset)
        step = StepBgSubtract()

        # run with and without flags: should all give
        # answers offset by ~100 for introduced background
        default = step(inp, fitflag=False, chauvenet=False)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.5)

        chflag = step(inp, fitflag=False, chauvenet=True)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.5)

        fitflag = step(inp, fitflag=True, chauvenet=False)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.5)

        allflag = step(inp, fitflag=True, chauvenet=True)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.5)

        assert not np.allclose(default[0].data, chflag[0].data)
        assert not np.allclose(default[0].data, fitflag[0].data)
        assert not np.allclose(default[0].data, allflag[0].data)

        # run on flat data to make sure all give same answer
        inp = self.make_input(2, tmpdir, offset=100, flat=True)
        default = step(inp, fitflag=False, chauvenet=False)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        chflag = step(inp, fitflag=False, chauvenet=True)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        fitflag = step(inp, fitflag=True, chauvenet=False)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        allflag = step(inp, fitflag=True, chauvenet=True)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        assert np.allclose(default[0].data, chflag[0].data, atol=1e-4)
        assert np.allclose(default[0].data, fitflag[0].data, atol=1e-4)
        assert np.allclose(default[0].data, allflag[0].data, atol=1e-4)

        # run without widow pixels -- should give same answer
        no_widow = step(inp, fitflag=False, chauvenet=False,
                        widowstokesi=False)
        for stokes in ['I', 'Q', 'U']:
            s_off = step.offsets[stokes]
            assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        assert np.allclose(default[0].data, no_widow[0].data, atol=1e-4)

        # run without qubgsubtract: should give zero for q and u, but
        # still do stokes I
        step(inp, fitflag=False, chauvenet=False,
             qubgsubtract=False)
        s_off = step.offsets['I']
        assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        assert np.allclose(step.offsets['Q'], 0)
        assert np.allclose(step.offsets['U'], 0)

        # run with nhwp = 1: should do I only
        capsys.readouterr()
        for df in inp:
            df.setheadval('NHWP', 1)
        step(inp, fitflag=False, chauvenet=False, qubgsubtract=True)
        s_off = step.offsets['I']
        assert np.allclose(s_off[1] - s_off[0], offset, rtol=0.1)
        assert np.allclose(step.offsets['Q'], 0)
        assert np.allclose(step.offsets['U'], 0)
        capt = capsys.readouterr()
        assert 'Stokes Q offset' not in capt.out
        assert 'Stokes U offset' not in capt.out
