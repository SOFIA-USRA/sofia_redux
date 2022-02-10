# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepmerge import StepMerge
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, scan_smp_data


class TestMerge(DRPTestCase):
    def make_data(self, tmpdir, nfiles=2, scan=False):
        inp = []
        for i in range(nfiles):
            if scan:
                hdul = scan_smp_data()
                hdul[0].header['NHWP'] = 1
            else:
                hdul = pol_bgs_data(idx=i)
            ffile = str(tmpdir.join('test{}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            df = DataFits(ffile)
            inp.append(df)
        return inp

    def test_miso(self, tmpdir):
        # works for c2n style data
        inp = self.make_data(tmpdir, 2)
        step = StepMerge()
        out = step(inp)
        assert isinstance(out, DataFits)

        # and for scan data
        inp = self.make_data(tmpdir, 2, scan=True)
        step = StepMerge()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_read_fwhm(self, capsys):
        df = DataFits()
        df.setheadval('SPECTEL1', 'HAWE')
        step = StepMerge()
        step.datain = [df]
        step.runstart([df], {})

        expected = (step.getarg('fwhm')[-1],
                    step.getarg('radius')[-1],
                    step.getarg('cdelt')[-1],
                    step.getarg('beamsize')[-1])

        # test defaults
        result = step.read_fwhm_radius_cdelt_beam()
        assert result == expected

        # bad spectel
        df.setheadval('SPECTEL1', 'HAWQ')
        with pytest.raises(ValueError):
            step.read_fwhm_radius_cdelt_beam()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        df.setheadval('SPECTEL1', '')
        with pytest.raises(ValueError):
            step.read_fwhm_radius_cdelt_beam()
        capt = capsys.readouterr()
        assert 'Cannot parse waveband' in capt.err

        # bad arglist
        df.setheadval('SPECTEL1', 'HAWE')
        step.runstart([df], {'cdelt': [1, 2, 3]})
        with pytest.raises(IndexError):
            step.read_fwhm_radius_cdelt_beam()
        capt = capsys.readouterr()
        assert 'Missing radius/fwhm values' in capt.err

    def test_tables(self, tmpdir, capsys):
        test_rec = np.array([(1, 2, 3)],
                            dtype=[('x', int), ('y', int), ('z', int)])
        test_tab = fits.TableHDU(test_rec).data

        # make some input data with tables to merge
        inp = []
        nfiles = 2
        for i in range(nfiles):
            hdul = pol_bgs_data(idx=i)
            ffile = str(tmpdir.join('test.fits'))
            hdul.writeto(ffile, overwrite=True)
            df = DataFits(ffile)
            df.tableset(test_tab, tablename='TABLE DATA')
            inp.append(df)

        step = StepMerge()
        out = step(inp)
        assert 'x' in out.table.names
        assert len(out.table['x']) == nfiles
        # some columns are added
        assert 'Right Ascension' in out.table.names
        assert 'Declination' in out.table.names
        assert 'Filename' in out.table.names

        # various table mismatch conditions

        # different units
        col1 = fits.Column(name='x', format='D',
                           unit='deg', array=np.array([1.0]))
        col = fits.Column(name='test1', format='D',
                          unit='deg', array=np.array([10.0]))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1, col]))
        inp[0].table = tbhdu.data
        col = fits.Column(name='test1', format='D',
                          unit='rad', array=np.array([10.0]))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1, col]))
        inp[1].table = tbhdu.data

        out = step(inp)
        capt = capsys.readouterr()
        assert 'different units' in capt.err
        assert 'x' in out.table.names
        assert 'test1' not in out.table.names

        # different dimensions
        xcol = fits.Column(name='x', format='D', dim='(1)',
                           unit='deg', array=np.array([10.0]))
        col = fits.Column(name='test1', format='D', dim='(1)',
                          unit='rad', array=np.array([10.0]))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([xcol, col]))
        inp[1].table = tbhdu.data
        out = step(inp)
        capt = capsys.readouterr()
        assert 'different dimension' in capt.err
        assert 'x' not in out.table.names
        assert 'test1' not in out.table.names

        # different format
        col = fits.Column(name='test1', format='E',
                          unit='rad', array=np.array([10.0]))
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1, col]))
        inp[1].table = tbhdu.data
        out = step(inp)
        capt = capsys.readouterr()
        assert 'different format' in capt.err
        assert 'x' in out.table.names
        assert 'test1' not in out.table.names

        # missing column
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1]))
        inp[1].table = tbhdu.data
        out = step(inp)
        capt = capsys.readouterr()
        assert 'name not found' in capt.err
        assert 'x' in out.table.names
        assert 'test1' not in out.table.names

    def test_merge_options(self, tmpdir, capsys):
        inp = self.make_data(tmpdir)
        step = StepMerge()

        # set some default parameters to ensure consistent
        # reductions
        kwargs = {'cdelt': [1.00, 1.55, 1.55, 2.75, 3.7],
                  'fwhm': [2.57, 4.02, 4.02, 6.93, 9.43],
                  'radius': [2.57, 4.02, 4.02, 6.93, 9.43],
                  'fit_order': 0,
                  'adaptive_algorithm': None,
                  'edge_threshold': 0}

        # set nhwp = 1 to just do stokes i
        # and add some bad pixels to mask
        for df in inp:
            df.setheadval('LATPOLE', 0)
            df.setheadval('LONPOLE', 0)
            df.setheadval('NHWP', 1)
            imlist = ['STOKES Q', 'STOKES U',
                      'ERROR Q', 'ERROR U',
                      'COVAR Q I', 'COVAR U I',
                      'COVAR Q U']
            for im in imlist:
                df.imagedel(im)

            bpm = df.imageget('BAD PIXEL MASK')
            bpm[10:12, 10:12] = 1
            bpm[12:14, 12:14] = 2
            bpm[14:16, 14:16] = 3

        # test flux conservation
        out1 = step(inp, conserveflux=True, **kwargs)
        out2 = step(inp, conserveflux=False, **kwargs)
        flux_factor = 2.75 ** 2 / inp[0].getheadval('PIXSCAL') ** 2

        # difference is flux factor
        assert np.allclose(np.nanmean(out1.image / out2.image), flux_factor)

        # since oversampled, out2 will overestimate flux
        assert np.nansum(out1.image) < np.nansum(out2.image)

        # with flux conservation, summed flux should be similar
        assert np.allclose(np.nansum(out1.image),
                           np.nansum(inp[0].image),
                           rtol=0.3)
        assert not np.allclose(np.nansum(out2.image),
                               np.nansum(inp[0].image),
                               rtol=0.3)

        # also check that lat and lon pole are no longer in header
        assert 'LATPOLE' not in out1.header
        assert 'LONPOLE' not in out1.header

        # test widow pix -- if used, should be higher total
        # value in image map
        out1 = step(inp, widowstokesi=True, **kwargs)
        out2 = step(inp, widowstokesi=False, **kwargs)
        capsys.readouterr()
        assert np.nansum(out1.imageget('IMAGE MASK')) \
            > np.nansum(out2.imageget('IMAGE MASK'))

        # run with/without error weighting
        step(inp, errflag=True, **kwargs)
        capt = capsys.readouterr()
        assert 'Uncertainties used for weighting' in capt.out
        step(inp, errflag=False, **kwargs)
        capt = capsys.readouterr()
        assert 'Uncertainties NOT used for weighting' in capt.err

    def test_adaptive_fwhm(self, tmpdir, capsys):
        inp = self.make_data(tmpdir)
        step = StepMerge()

        # standard kwargs
        kwargs = {'cdelt': [1.00, 1.55, 1.55, 2.75, 3.7],
                  'radius': [2.57, 4.02, 4.02, 6.93, 9.43],
                  'fit_order': 1,
                  'edge_threshold': 0}

        # run with adaptive, fwhm = beam -- no warning
        expected = step(inp, adaptive_algorithm='scaled',
                        fwhm=[4.84, 7.80, 7.80, 13.6, 18.2],
                        beamsize=[4.84, 7.80, 7.80, 13.6, 18.2],
                        **kwargs)
        assert 'Setting smoothing FWHM to beam' not in capsys.readouterr().err

        # run with fwhm != beam -- warns
        testval = step(inp, adaptive_algorithm='scaled',
                       fwhm=[2.57, 4.02, 4.02, 6.93, 9.43],
                       beamsize=[4.84, 7.80, 7.80, 13.6, 18.2],
                       **kwargs)
        assert 'Setting smoothing FWHM to beam' in capsys.readouterr().err
        assert np.allclose(expected.image, testval.image, equal_nan=True)

        # run without adaptive, fwhm != beam -- no warning
        testval2 = step(inp, adaptive_algorithm=None,
                        fwhm=[2.57, 4.02, 4.02, 6.93, 9.43],
                        beamsize=[4.84, 7.80, 7.80, 13.6, 18.2],
                        **kwargs)
        assert 'Setting smoothing FWHM to beam' not in capsys.readouterr().err
        # result is not the same without adaptive
        assert not np.allclose(expected.image, testval2.image, equal_nan=True)

    def test_bin_cdelt(self, tmpdir, capsys):
        inp = self.make_data(tmpdir)
        step = StepMerge()

        # default: no binning, bin_cdelt on
        default = step(inp, bin_cdelt=True, fit_order=2)
        capt = capsys.readouterr().out
        assert 'Multiplying cdelt' not in capt
        assert 'Reducing fit order' not in capt

        # set binning to 2, bin_cdelt off - same result
        inp[0].setheadval('PIXELBIN', 2)
        bin_off = step(inp, bin_cdelt=False, fit_order=2)
        capt = capsys.readouterr().out
        assert 'Multiplying cdelt' not in capt
        assert 'Reducing fit order' not in capt
        assert np.allclose(default.image, bin_off.image, equal_nan=True)

        # turn bin_cdelt on: result is half the size, fit order is reduced
        bin_on = step(inp, bin_cdelt=True, fit_order=2)
        capt = capsys.readouterr().out
        assert 'Multiplying cdelt and radius by binning factor 2' in capt
        assert 'Reducing fit order to 1' in capt
        assert tuple([s // 2 for s
                      in default.image.shape]) == bin_on.image.shape
        # total flux should be pretty close
        assert np.allclose(np.nansum(default.image),
                           np.nansum(bin_on.image),
                           rtol=0.1)
