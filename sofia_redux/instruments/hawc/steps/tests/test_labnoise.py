# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil

from matplotlib.testing.compare import compare_images
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepnoisefft import StepNoiseFFT
from sofia_redux.instruments.hawc.steps.stepnoiseplots import StepNoisePlots
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestLabNoise(DRPTestCase):
    def make_data(self, tmpdir, suffix=''):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join(f'test{suffix}.fits'))
        hdul[0].header['SMPLFREQ'] = 203.
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        step = StepPrepare()
        return step(df)

    def test_siso(self, tmpdir):
        df = self.make_data(tmpdir)

        # move to tmpdir -- writes auxfiles to cwd
        with tmpdir.as_cwd():
            step = StepNoiseFFT()
            out = step(df)
            assert isinstance(out, DataFits)

            step = StepNoisePlots()
            out2 = step(out)
            assert isinstance(out2, DataFits)
            assert np.allclose(out.image, out2.image, equal_nan=True)

            fname = ['test.NPL_med.png', 'test.NPL_specmap.png',
                     'test.NPL_8-12Hz.png']
            for fn in fname:
                assert os.path.exists(fn)

    def test_threadsafe(self, tmpdir, capsys):
        with tmpdir.as_cwd():

            inp = []
            df1 = self.make_data(tmpdir, suffix='1')
            df2 = self.make_data(tmpdir, suffix='2')

            step1 = StepNoiseFFT()
            inp.append(step1(df1))
            inp.append(step1(df2))

            def _try_plot(i):
                step2 = StepNoisePlots()
                step2(inp[i])

            # this will crash with a fatal Python error
            # if plots are not thread safe

            from threading import Thread
            t1 = Thread(target=_try_plot, args=(0,))
            t1.start()
            t2 = Thread(target=_try_plot, args=(1,))
            t2.start()

            # let both finish
            t1.join()
            t2.join()

            # check for output from both threads
            fname = ['test1.NPL_med.png', 'test1.NPL_specmap.png',
                     'test1.NPL_8-12Hz.png',
                     'test2.NPL_med.png', 'test2.NPL_specmap.png',
                     'test2.NPL_8-12Hz.png', ]
            for fn in fname:
                assert os.path.exists(fn)

    def test_missing_array(self, tmpdir):
        outfile = 'test.NPL_med.png'

        # move to tmpdir -- writes auxfiles to cwd
        with tmpdir.as_cwd():
            df = self.make_data(tmpdir)
            step1 = StepNoiseFFT()
            fft = step1(df)

            step2 = StepNoisePlots()
            step2(fft)
            assert os.path.isfile(outfile)
            shutil.copyfile(outfile, 'tmp1.png')

            # set FFT to zero for missing array
            img = fft.image
            tmp = img.reshape(41, 128, img.shape[1])
            tmp[:, 96:, :] = 0
            fft.image = tmp.reshape(img.shape)
            step2 = StepNoisePlots()
            step2(fft)

            assert os.path.isfile(outfile)
            shutil.copyfile(outfile, 'tmp2.png')

            # image is not the same
            assert (compare_images('tmp1.png', 'tmp2.png', 0) is not None)
