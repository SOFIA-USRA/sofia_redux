# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps import basemap
from sofia_redux.instruments.hawc.steps.stepmerge import StepMerge
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestBaseMap(DRPTestCase):
    def test_addgap(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        bmap = basemap.BaseMap()

        data = DataFits(ffile)
        naxis1, naxis2 = data.getheadval('NAXIS1'), data.getheadval('NAXIS2')
        pix2, pix1 = np.mgrid[0:naxis2, 0:naxis1]
        x = pix1.copy()
        y = pix2.copy()

        # for default data, gapx = 4, all others zero --
        # x is adjusted by 4 pixels, y is not
        bmap.addgap(data, x, y)
        assert np.all(x[:, -1] == pix1[:, -1] + 4)
        assert np.all(y == pix2)

        # delete keywords from header -- warns, uses 0 for all
        data.delheadval('ALNGAPX')
        data.delheadval('ALNGAPY')
        data.delheadval('ALNROTA')
        x = pix1.copy()
        y = pix2.copy()
        bmap.addgap(data, x, y)
        assert np.all(x == pix1)
        assert np.all(y == pix2)
        capt = capsys.readouterr()
        assert 'Missing ALNGAPX' in capt.out
        assert 'Missing ALNGAPY' in capt.out
        assert 'Missing ALNROTA' in capt.out

    def test_checkvalid(self, tmpdir, capsys):
        inp = []
        for i in range(2):
            hdul = pol_bgs_data(idx=i)
            ffile = str(tmpdir.join('test.fits'))
            hdul.writeto(ffile, overwrite=True)
            inp.append(DataFits(ffile))

        bmap = basemap.BaseMap()
        bmap.datain = inp

        # default: should match
        bmap.checkvalid()
        capt = capsys.readouterr()
        assert 'Finished checking valid' in capt.out

        # modify spectel
        inp[0].setheadval('SPECTEL1', 'BADVAL')
        with pytest.raises(ValueError):
            bmap.checkvalid()
        capt = capsys.readouterr()
        assert 'SPECTEL1 not the same' in capt.err

        # modify nhwp
        inp[0].setheadval('SPECTEL1', inp[1].getheadval('SPECTEL1'))
        inp[0].setheadval('NHWP', 2)
        with pytest.raises(ValueError):
            bmap.checkvalid()
        capt = capsys.readouterr()
        assert 'Number of HWP angles not the same' in capt.err

    def test_exptime(self):
        inp = []
        expected = 0
        for i in range(3):
            df = DataFits()
            etime = i * 10 + 1
            df.setheadval('EXPTIME', etime)
            expected += etime
            inp.append(df)
        bmap = basemap.BaseMap()
        bmap.datain = inp
        bmap.dataout = DataFits()

        bmap.sumexptime()
        assert np.allclose(bmap.dataout.getheadval('EXPTIME'), expected)

    def test_map(self, tmpdir, capsys):
        """Test makemaps: allocates space for maps."""
        inp = []
        for i in range(2):
            hdul = pol_bgs_data(idx=i)
            ffile = str(tmpdir.join('test{}.fits'.format(i + 1)))
            hdul.writeto(ffile, overwrite=True)
            inp.append(DataFits(ffile))

        # use a subclass that has appropriate functions defined (eg getarg)
        def reset(indata):
            bmapper = StepMerge()
            bmapper.datain = indata
            bmapper.runstart(indata, {})
            bmapper.nhwp = 4
            bmapper.nfiles = 2
            bmapper.read_resample_data(0)
            return bmapper

        # check the default works
        bmap = reset(inp)
        bmap.make_resample_map(7.0)
        assert bmap.pmap['xout'].size < bmap.getarg('sizelimit')

        # check that an error is raised on
        # attempt to make a big map
        inp[0].setheadval('CRVAL1', 0.0)
        inp[0].setheadval('CRVAL2', 0.0)
        bmap = reset(inp)
        with pytest.raises(ValueError):
            bmap.make_resample_map(7.0)
        capt = capsys.readouterr()
        assert 'Output map is too large' in capt.err
