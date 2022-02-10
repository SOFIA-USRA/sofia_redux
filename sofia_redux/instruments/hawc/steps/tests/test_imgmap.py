# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepimgmap import StepImgMap
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestImgMap(DRPTestCase):
    def make_data(self, tmpdir, name='test.fits', datatype='nod'):
        # note -- should cd to tmpdir before calling this.
        hdul = pol_bgs_data()

        # add some necessary keywords from other steps
        hdul[0].header['BMIN'] = 0.003778
        hdul[0].header['BMAJ'] = 0.003778
        hdul[0].header['BPA'] = 0.0

        if datatype == 'scan':
            hdul[0].header['EXTNAME'] = 'PRIMARY IMAGE'
            hdul[1].header['EXTNAME'] = 'NOISE'

        ffile = str(tmpdir.join(name))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        return df

    def test_siso(self, tmpdir):
        with tmpdir.as_cwd():
            df = self.make_data(tmpdir)
            step = StepImgMap()

            # passes on input data with defaults
            out = step(df)
            assert isinstance(out, DataFits)
            assert os.path.isfile('test.png')

    def test_run_options(self, tmpdir, capsys):
        with tmpdir.as_cwd():
            df = self.make_data(tmpdir)
            step = StepImgMap()

            # output file name
            fname = 'test.png'

            # use center cropping
            par = [df.getheadval('CRVAL1'),
                   df.getheadval('CRVAL2'),
                   10 * df.getheadval('CDELT1'),
                   10 * df.getheadval('CDELT1')]
            step(df, centercrop=True, centercropparams=par,
                 maphdu='STOKES I')
            capt = capsys.readouterr()
            assert 'Using center cropping' in capt.out
            assert 'Mapping extension STOKES I' in capt.out
            assert os.path.isfile(fname)

            # scale low/high parameter
            # bad input
            with pytest.raises(TypeError):
                step(df, lowhighscale="['a', 'b']")
            # okay input
            step(df, lowhighscale=[0, 100])

            # test mapping for non-scan names to scan convention
            df = self.make_data(tmpdir, datatype='scan')
            step(df, maphdu='STOKES I')
            assert 'Mapping extension PRIMARY IMAGE' \
                in capsys.readouterr().out
            step(df, maphdu='PRIMARY IMAGE')
            assert 'Mapping extension PRIMARY IMAGE' \
                in capsys.readouterr().out

            step(df, maphdu='ERROR I')
            assert 'Mapping extension NOISE' \
                in capsys.readouterr().out
            step(df, maphdu='NOISE')
            assert 'Mapping extension NOISE' \
                in capsys.readouterr().out

    def test_threadsafe(self, tmpdir, capsys):
        # set log to error level to ignore warnings
        from astropy import log
        log.setLevel('ERROR')

        with tmpdir.as_cwd():

            def _try_plot(i):
                df = self.make_data(tmpdir, name=f'test{i}.fits')
                step = StepImgMap()
                step(df)

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

            # check for output
            assert os.path.exists('test1.png')
            assert os.path.exists('test2.png')
