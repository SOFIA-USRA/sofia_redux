# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepfluxjump import StepFluxjump
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestFluxJump(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepFluxjump()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_run(self, tmpdir, capsys):
        nframe = 80
        hdul = pol_raw_data(nframe)

        # mark a pixel for jump correction
        # with two jumps, in opposite directions
        # looks like:      ^-----^
        #             ----v       v----
        j1 = nframe // 3
        j2 = 2 * nframe // 3
        fjdata = hdul[2].data['FluxJumps']
        fjdata[0:j1, 15, 15] = -14
        fjdata[j1, 15, 15] = -60
        fjdata[j1 + 1, 15, 15] = 60
        fjdata[j1 + 2:j2, 15, 15] = 14
        fjdata[j2, 15, 15] = 60
        fjdata[j2 + 1, 15, 15] = -60
        fjdata[j2 + 2:, 15, 15] = 14

        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        # jumpmap -- all zero
        jmap = fits.HDUList(
            [fits.PrimaryHDU(data=np.zeros((41, 128), dtype=int))])
        jfile = str(tmpdir.join('jmap.fits'))
        jmap.writeto(jfile, overwrite=True)

        step = StepFluxjump()
        out = step(inp.copy(), jumpmap=jfile)
        capt = capsys.readouterr()
        assert 'No correction' in capt.out
        # zero means no correction
        assert np.allclose(out.table['SQ1Feedback'],
                           inp.table['SQ1Feedback'])

        # add a correction to the one pixel
        jmap[0].data[15, 15] = 6477
        jmap.writeto(jfile, overwrite=True)
        out = step(inp.copy(), jumpmap=jfile)
        assert np.allclose(out.table['SQ1Feedback'][:, 0:14, 0:14],
                           inp.table['SQ1Feedback'][:, 0:14, 0:14])
        assert not np.allclose(out.table['SQ1Feedback'][j1, 15, 15],
                               inp.table['SQ1Feedback'][j1, 15, 15])
        assert not np.allclose(out.table['SQ1Feedback'][j2, 15, 15],
                               inp.table['SQ1Feedback'][j2, 15, 15])

        # test with numerical value instead (floored to int)
        # data before and after bump is corrected by factor * jump
        out = step(inp.copy(), jumpmap=10.5)
        assert np.allclose(out.table['SQ1Feedback'][:j1, 15, 15],
                           10 * fjdata[:j1, 15, 15]
                           + inp.table['SQ1Feedback'][:j1, 15, 15])
        assert np.allclose(out.table['SQ1Feedback'][j2 + 3:, 15, 15],
                           10 * fjdata[j2 + 3:, 15, 15]
                           + inp.table['SQ1Feedback'][j2 + 3:, 15, 15])

        # try a bad value
        with pytest.raises(ValueError):
            step(inp.copy(), jumpmap='bad')
        capt = capsys.readouterr()
        assert 'Bad fluxjump value' in capt.err
