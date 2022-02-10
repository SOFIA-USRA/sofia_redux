# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepskydip import StepSkydip
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, scan_raw_data


class TestSkyDip(DRPTestCase):
    def make_data(self, tmpdir, run_scanmap=False, suffix=''):
        hdul = scan_raw_data()
        hdul[0].header['INSTCFG'] = 'TOTAL_INTENSITY'
        hdul[0].header['CALMODE'] = 'SKY_DIP'
        hdul[0].header['OBSMODE'] = 'SkyDip'
        ffile = str(tmpdir.join(f'test{suffix}.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        if run_scanmap:
            StepScanMap()([inp], options='-rounds=1')
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)
        return dmd

    def test_mimo(self, tmpdir):
        with tmpdir.as_cwd():
            dmd = self.make_data(tmpdir)

            step = StepSkydip()
            out = step([dmd])
            assert isinstance(out, list)
            assert isinstance(out[0], DataFits)
            assert os.path.isfile('test.DMD_skydiplot.png')

    def test_scanmap_plots(self, tmpdir, mocker):
        with tmpdir.as_cwd():
            dmd = self.make_data(tmpdir, run_scanmap=True)
            dmd.filename = os.path.basename(dmd.filename)

            # mock os remove for an OSError -- should still complete
            tmp = os.remove

            def mock_remove(*args, **kwargs):
                raise OSError('test error')
            mocker.patch(
                'sofia_redux.instruments.hawc.steps.stepskydip.os.remove',
                mock_remove)

            step = StepSkydip()
            step([dmd])
            assert os.path.isfile('test.DMD_skydiplot.png')
            assert os.path.isfile('test.DMD_skymodel.png')

            # temporary dat files are not removed
            assert len(glob.glob('*.dat')) > 0

            # unmock
            mocker.patch(
                'sofia_redux.instruments.hawc.steps.stepskydip.os.remove', tmp)
            step = StepSkydip()
            step([dmd])

            # temporary dat files are now removed
            assert len(glob.glob('*.dat')) == 0

    def test_threadsafe(self, tmpdir, capsys):
        with tmpdir.as_cwd():

            dmd1 = self.make_data(tmpdir, suffix='1')
            dmd2 = self.make_data(tmpdir, suffix='2')
            inp = [dmd1, dmd2]

            def _try_plot(i):
                dmd = inp[i]
                step = StepSkydip()
                step([dmd])

            # this will crash with a fatal Python error
            # if plots are not thread safe

            from threading import Thread
            t1 = Thread(target=_try_plot, args=(0,))
            t1.setDaemon(True)
            t1.start()
            t2 = Thread(target=_try_plot, args=(1,))
            t2.setDaemon(True)
            t2.start()

            # let both finish
            t1.join()
            t2.join()

            # check for output from both threads
            fname = ['test1.DMD_skydiplot.png',
                     'test2.DMD_skydiplot.png']
            for fn in fname:
                assert os.path.exists(fn)
