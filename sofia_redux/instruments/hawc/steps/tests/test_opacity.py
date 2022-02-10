# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepopacity import StepOpacity
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, scan_raw_data


class TestOpacity(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepOpacity()
        out = step(inp)

        assert isinstance(out, DataFits)

    def test_opc_scan(self, capsys):
        # make a test data set with a scan header, but
        # a small map-like image
        hdul = scan_raw_data()
        hdr = hdul[0].header
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE',
                    imageheader=hdr)
        df.imageset(np.zeros((10, 10)), imagename='NOISE')
        df.imageset(np.zeros((10, 10)), imagename='S/N')

        step = StepOpacity()
        step(df)
        capt = capsys.readouterr()
        assert 'Opacity correction factor' in capt.out
        for im in df.imgnames:
            if 'S/N' in im:
                continue
            assert 'Correcting extension: {}'.format(im) in capt.out

    def test_nod(self, capsys):
        # make a small chop/nod-like file
        hdul = pol_bgs_data()
        hdr = hdul[0].header
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='STOKES I',
                    imageheader=hdr)
        df.imageset(np.zeros((10, 10)), imagename='ERROR I')

        # set nhwp to 1
        df.setheadval('NHWP', 1)

        step = StepOpacity()

        out = step(df)
        capt = capsys.readouterr()
        assert 'Opacity correction factor' in capt.out
        assert 'STOKES I' in out.imgnames

        # same result if missing nhwp keys
        df.delheadval('NHWP')
        out = step(df)
        capt = capsys.readouterr()
        assert 'Opacity correction factor' in capt.out
        assert 'STOKES I' in out.imgnames
