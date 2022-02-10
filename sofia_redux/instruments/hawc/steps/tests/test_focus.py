# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.modeling.models import Gaussian2D
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap
from sofia_redux.instruments.hawc.steps.stepfocus import StepFocus
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, scan_raw_data, pol_bgs_data


class TestFocus(DRPTestCase):
    def make_data(self, suffix=''):
        nsize = 100
        pos = nsize / 2.0

        # get a sofia header to use with fake data,
        # update with some useful keywords
        hdul = pol_bgs_data()
        hdr = hdul[0].header
        hdr['PRODTYPE'] = 'stdphotcal'
        hdr['INSTCFG'] = 'TOTAL_INTENSITY'
        hdr['INSTMODE'] = 'OTFMAP'
        hdr['CALMODE'] = 'FOCUS'
        hdr['CRPIX1'] = pos
        hdr['CRPIX2'] = pos

        # make three Gaussian data sets to fit
        y, x = np.mgrid[:nsize, :nsize]
        g1 = Gaussian2D(amplitude=10., x_stddev=3., y_stddev=3.,
                        x_mean=pos, y_mean=pos)
        df1 = DataFits()
        df1.imageset(g1(x, y), imageheader=hdr, imagename='STOKES I')
        df1.setheadval('FOCUS_ST', 100)
        df1.setheadval('FOCUS_EN', 100)
        df1.setheadval('FCSTOFF', -100)
        df1.tableset(None, tablename='EXTRA')
        df1.setheadval('SRCFWHM', 5., dataname='EXTRA')
        df1.setheadval('SRCPEAK', 10., dataname='EXTRA')
        df1.filename = f'test1{suffix}.fits'

        g2 = Gaussian2D(amplitude=20., x_stddev=2., y_stddev=2.,
                        x_mean=pos, y_mean=pos)
        df2 = DataFits()
        df2.imageset(g2(x, y), imageheader=hdr, imagename='STOKES I')
        df2.setheadval('FOCUS_ST', 200)
        df2.setheadval('FOCUS_EN', 200)
        df2.setheadval('FCSTOFF', 0)
        df2.tableset(None, tablename='EXTRA')
        df2.setheadval('SRCFWHM', 4., dataname='EXTRA')
        df2.setheadval('SRCPEAK', 20., dataname='EXTRA')
        df2.filename = f'test2{suffix}.fits'

        g3 = Gaussian2D(amplitude=8., x_stddev=4., y_stddev=4.,
                        x_mean=pos, y_mean=pos)
        df3 = DataFits()
        df3.imageset(g3(x, y), imageheader=hdr, imagename='STOKES I')
        df3.setheadval('FOCUS_ST', 300)
        df3.setheadval('FOCUS_EN', 300)
        df3.setheadval('FCSTOFF', 100)
        df3.tableset(None, tablename='EXTRA')
        df3.setheadval('SRCFWHM', 8., dataname='EXTRA')
        df3.setheadval('SRCPEAK', 8., dataname='EXTRA')
        df3.filename = f'test3{suffix}.fits'

        inp = [df1, df2, df3]
        return inp

    def test_mimo(self, tmpdir):
        hdul = scan_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            # raw data fails
            step = StepFocus()
            with pytest.raises(ValueError):
                step([df])

            # scanmapped data works
            smap = StepScanMap()([df])
            out = step(smap)

            assert isinstance(out, list)
            assert isinstance(out[0], DataFits)

    def test_moments(self):
        step = StepFocus()

        # zero data
        data = np.zeros((11, 11), dtype=float)
        h, x, y, wx, wy, bg = step.moments(data)
        assert np.allclose([h, bg], 0)
        assert np.allclose([x, y], 5.5)
        assert np.all(~np.isfinite([wx, wy]))

        # one pixel
        data[5, 5] = 1.0
        h, x, y, wx, wy, bg = step.moments(data)
        assert np.allclose(h, 1.0)
        assert np.allclose([x, y], 5.0)
        assert np.allclose([wx, wy, bg], 0)

        # Gaussian
        y, x = np.mgrid[:11, :11]
        g = Gaussian2D(amplitude=10., x_stddev=2., y_stddev=1.,
                       x_mean=5., y_mean=5.)
        data = g(x, y)

        h, x, y, wx, wy, bg = step.moments(data)
        assert np.allclose(h, 10.0)
        assert np.allclose([x, y], 5.0)
        assert np.allclose(wx, 2., atol=0.1)
        assert np.allclose(wy, 1., atol=0.1)
        assert np.allclose(bg, 0., atol=0.1)

    def test_fitgaussian(self):
        step = StepFocus()

        y, x = np.mgrid[:21, :15]
        g = Gaussian2D(amplitude=10., x_stddev=2., y_stddev=1.,
                       x_mean=5., y_mean=7.)
        data = g(x, y)

        # fit parameters should be same as input
        result = list(step.fitgaussian(data, [], True))
        assert np.allclose(result[:-1], [10, 7, 5, 1, 2, 0])
        assert result[-1] in [1, 2, 3, 4]

        # add a nan patch and mask it
        data[10:12, 10:12] = np.nan
        idx = np.where(np.isnan(data))
        result = list(step.fitgaussian(data, idx, False))
        assert np.allclose(result[:-1], [10, 7, 5, 1, 2, 0], atol=.01)
        assert result[-1] in [1, 2, 3, 4]

    def test_run_mult(self, tmpdir, capsys):
        inp = self.make_data()
        with tmpdir.as_cwd():
            step = StepFocus()

            step(inp)
            capt = capsys.readouterr()
            assert 'Best focus position' in capt.out
            imglist = ['test3_autofocus_image1.png',
                       'test3_autofocus_image2.png',
                       'test3_autofocus_image3.png',
                       'test3_autofocus_FWHM_X.png',
                       'test3_autofocus_FWHM_Y.png',
                       'test3_autofocus_FWHM_XY.png',
                       'test3_autofocus_Amplitude.png',
                       'test3_autofocus_Peak.png',
                       'test3_autofocus_FWHM-C.png']
            for img in imglist:
                assert os.path.isfile(img)

            # test various cropping conditions
            step(inp, autocrop=False, xyboxcent=[90, 90])
            capt = capsys.readouterr()
            assert 'Crop box x2 invalid' in capt.err
            assert 'Crop box y2 invalid' in capt.err

            step(inp, autocrop=False, xyboxcent=[10, 10])
            capt = capsys.readouterr()
            assert 'Crop box x1 invalid' in capt.err
            assert 'Crop box y1 invalid' in capt.err

            # specify HDU to use directly;
            # add a bad pixel mask with widow pixels
            inp2 = []
            for df in inp:
                df2 = df.copy()
                bpm = np.zeros_like(df.image)
                bpm[10:12, 10:12] = 1
                df2.imageset(np.zeros_like(df.image),
                             imagename='BAD PIXEL MASK')
                inp2.append(df2)
            step(inp2, primaryimg='STOKES I', widowisgood=False)
            capt = capsys.readouterr()
            assert 'Using specified image: STOKES I' in capt.out
            assert 'Bad Pixel Mask found' in capt.out
            assert 'Best focus position' in capt.out

            # add an image mask instead;
            # turn off median averaging to propagate nans
            inp2 = []
            for df in inp:
                df2 = df.copy()
                bpm = np.zeros_like(df.image)
                bpm[10:12, 10:12] = np.nan
                df2.imageset(bpm, imagename='IMAGE MASK')
                inp2.append(df2)
            step(inp2, medianaverage=False)
            capt = capsys.readouterr()
            assert 'Using Image Mask' in capt.out
            assert 'Best focus position' in capt.out

            # remove FCSTOFF keyword from headers - will use
            # focus value instead
            # also delete some scanmap keys -- will not make extra plots
            for df in inp:
                df.delheadval('FCSTOFF')
                df.delheadval('SRCFWHM', dataname='EXTRA')
                df.delheadval('SRCPEAK', dataname='EXTRA')
                df.delheadval('SRCINT', dataname='EXTRA')
            step(inp)
            capt = capsys.readouterr()
            assert 'FCSTOFF not found' in capt.out
            assert 'Scan map fit keys not found' in capt.out

    def test_gaussfail(self, tmpdir, capsys, mocker):
        inp = self.make_data()

        with tmpdir.as_cwd():
            # mock fitgaussian to return a failure
            mocker.patch.object(StepFocus, 'fitgaussian',
                                return_value=(0, 0, 0, 0, 0, 0, 0))

            step = StepFocus()
            step(inp)
            capt = capsys.readouterr()
            assert 'Gaussian fit was unsuccessful' in capt.out

    def test_focusplot(self, tmpdir, capsys):
        with tmpdir.as_cwd():
            step = StepFocus()

            focus = [-5, -4, -3]
            values = [1, 2, 1]
            difftot = 1
            label = 'label'
            lbl = 'lbl'
            step.focusplot(focus, values, difftot, label, lbl, -1)
            capt = capsys.readouterr()
            assert 'Best focus position' in capt.out

            values = [1, 2, 3]
            step.focusplot(focus, values, difftot, label, lbl, -1)
            capt = capsys.readouterr()
            assert 'No local maximum' in capt.err

            step.focusplot(focus, values, difftot, label, lbl, 1)
            capt = capsys.readouterr()
            assert 'No local minimum' in capt.err

    def test_threadsafe(self, tmpdir, capsys):
        with tmpdir.as_cwd():

            def _try_plot(i):
                inp = self.make_data(suffix=i)
                step = StepFocus()
                step(inp)

            # this will crash with a fatal Python error
            # if plots are not thread safe

            from threading import Thread
            t1 = Thread(target=_try_plot, args=(1,))
            t1.setDaemon(True)
            t1.start()
            t2 = Thread(target=_try_plot, args=(2,))
            t2.setDaemon(True)
            t2.start()

            # let both finish
            t1.join()
            t2.join()

            # check for output from both threads
            fname = ['test31_autofocus_Amplitude.png',
                     'test31_autofocus_FWHM-C.png',
                     'test31_autofocus_FWHM_X.png',
                     'test31_autofocus_FWHM_XY.png',
                     'test31_autofocus_FWHM_Y.png',
                     'test31_autofocus_Peak.png',
                     'test31_autofocus_image1.png',
                     'test31_autofocus_image2.png',
                     'test31_autofocus_image3.png',
                     'test32_autofocus_Amplitude.png',
                     'test32_autofocus_FWHM-C.png',
                     'test32_autofocus_FWHM_X.png',
                     'test32_autofocus_FWHM_XY.png',
                     'test32_autofocus_FWHM_Y.png',
                     'test32_autofocus_Peak.png',
                     'test32_autofocus_image1.png',
                     'test32_autofocus_image2.png',
                     'test32_autofocus_image3.png']
            for fn in fname:
                assert os.path.exists(fn)
