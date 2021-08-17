# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.wcs import WCS
import numpy as np
import pytest

from sofia_redux.instruments.forcast.coadd import coadd, _target_xy
from sofia_redux.instruments.forcast.tests.resources import nmc_testdata


class TestCoadd(object):
    def make_test_data(self, nfiles=3, spec_style=True):
        test = nmc_testdata()
        h, d, v, e = [], [], [], []
        if spec_style:
            test['header']['CRPIX1A'] = test['header']['CRPIX1']
            test['header']['CRPIX2A'] = test['header']['CRPIX2']
            test['header']['CRPIX3A'] = 1
            test['header']['CRVAL1A'] = test['header']['CRVAL1']
            test['header']['CRVAL2A'] = test['header']['CRVAL2']
            test['header']['CRVAL3A'] = 1
            test['header']['CDELT1A'] = test['header']['CDELT1']
            test['header']['CDELT2A'] = test['header']['CDELT2']
            test['header']['CDELT3A'] = 1
        for i in range(nfiles):
            h0 = test['header'].copy()
            if spec_style:
                h0['CRVAL3A'] += i
            h0['FILENAME'] = f'test{i+1}.fits'

            d0 = np.full((20, 20), i + 1, dtype=np.float64)
            v0 = d0.copy()
            e0 = d0.copy().astype(int) * 0 + 1
            h.append(h0)
            d.append(d0)
            v.append(v0)
            e.append(e0)
        return h, d, v, e

    def test_algorithms(self):
        nfiles = 3
        vals = np.arange(nfiles) + 1
        h, d, v, e = self.make_test_data(nfiles=nfiles)

        # mean combine
        mh, md, mv, me = coadd(h, d, v, e, method='mean', weighted=False,
                               robust=False)

        # output should be the same size as input for no shifts
        assert md.shape == d[0].shape
        assert mv.shape == d[0].shape
        assert me.shape == d[0].shape

        # expected values for mean of data = 0, 1, 2...nfiles=1
        assert np.allclose(md[~np.isnan(md)], np.mean(vals))
        assert np.allclose(mv[~np.isnan(mv)], np.mean(vals) / nfiles)
        assert np.allclose(me[me != 0], nfiles)

        # median combine
        dh, dd, dv, de = coadd(h, d, v, e, method='median', robust=False)

        # expected values for median of data = 0, 1, 2...nfiles=1
        assert np.allclose(dd[~np.isnan(dd)], np.median(vals))
        assert np.allclose(dv[~np.isnan(dv)],
                           np.mean(vals) / nfiles * np.pi / 2)
        assert np.allclose(de[de != 0], nfiles)

        # resample -- should report mean values
        rh, rd, rv, re = coadd(h, d, v, e, method='resample',
                               weighted=False, fit_order=0, window=0.5)
        assert np.allclose(rd[~np.isnan(rd)], np.mean(vals))
        assert np.allclose(rv[~np.isnan(rv)], np.mean(vals) / nfiles)
        assert np.allclose(re[re != 0], nfiles)

        # weighted result for adaptive
        rh, rd, rv, re = coadd(h, d, v, e, method='resample',
                               weighted=True, fit_order=1, window=0.5,
                               adaptive_algorithm='scaled')
        assert np.allclose(rd[~np.isnan(rd)], 1.63636364)
        assert np.allclose(rv[~np.isnan(rv)], 0.54545455)
        assert np.allclose(re[re != 0], nfiles)

    def test_cube(self, capsys):
        nfiles = 3
        vals = np.arange(nfiles) + 1
        h, d, v, e = self.make_test_data(nfiles=nfiles, spec_style=True)

        # try to make a cube with a non-3d WCS
        with pytest.raises(ValueError):
            coadd(h, d, v, e, cube=True,
                  weighted=False, fit_order=0, window=0.5)
        assert 'wcs is not 3d' in capsys.readouterr().err.lower()

        # try to make a cube with a 3d non-secondary WCS
        h[0]['CRPIX3'] = 1
        h[0]['CRVAL3'] = 1
        with pytest.raises(ValueError):
            coadd(h, d, v, e, cube=True,
                  weighted=False, fit_order=0, window=0.5)
        assert 'unexpected input wcs' in capsys.readouterr().err.lower()

        # now make a cube with the secondary WCS
        rh, rd, rv, re = coadd(h, d, v, e, cube=True, wcskey='A',
                               weighted=False, fit_order=0, window=0.5)
        assert rd.shape == (20, 20, 3)
        for plane in range(3):
            td = rd[:, :, plane]
            tv = rd[:, :, plane]
            assert np.allclose(td[~np.isnan(td)], vals[plane])
            assert np.allclose(tv[~np.isnan(tv)], vals[plane])
        assert np.allclose(re[re != 0], nfiles)

        # check for modified header keys
        assert 'CRPIX1A' not in rh
        assert 'CRPIX2A' not in rh
        assert 'CRPIX3A' not in rh
        assert rh['CRPIX1'] == 1
        assert rh['CRPIX2'] == 250
        assert rh['CRPIX3'] == 250

        # close result with adaptive
        rh, rd, rv, re = coadd(h, d, v, e, cube=True, wcskey='A',
                               weighted=True, fit_order=1, window=0.5,
                               adaptive_algorithm='scaled')
        assert rd.shape == (20, 20, 3)
        for plane in range(3):
            td = rd[:, :, plane]
            assert np.allclose(td[~np.isnan(td)], vals[plane])
        assert np.allclose(re[re != 0], nfiles)

        # and now make a 2D image with the secondary WCS: should mean combine
        rh, rd, rv, re = coadd(h, d, v, e, cube=False, wcskey='A',
                               weighted=False, spectral=True)
        assert rd.shape == (20, 20)
        assert np.allclose(rd[~np.isnan(rd)], np.mean(vals))
        assert np.allclose(rv[~np.isnan(rv)], np.mean(vals) / nfiles)
        assert np.allclose(re[re != 0], nfiles)

        # check that secondary keys are still in header
        assert rh['CRPIX1'] == h[0]['CRPIX1']
        assert rh['CRPIX2'] == h[0]['CRPIX2']
        assert rh['CRPIX1A'] == h[0]['CRPIX1A']
        assert rh['CRPIX2A'] == h[0]['CRPIX2A']
        assert rh['CRPIX3A'] == h[0]['CRPIX3A']

    def test_target_xy(self, capsys):
        h, d, v, e = self.make_test_data(nfiles=2, spec_style=True)

        # make sure pixel scale is 1
        h[0]['CDELT1'] = 1
        h[0]['CDELT1A'] = 1

        outwcs_img = WCS(h[0], key=' ')
        outwcs_spc = WCS(h[0], key='A')

        # add TGTRA/DEC and verify they are retrieved
        # pixel scale is 1.0, so 1 degree movement is 1 pixel
        # (with RA recorded in hours)
        h[0]['TGTRA'] = 10.0 / 15.
        h[0]['TGTDEC'] = 10.0
        h[1]['TGTRA'] = 11.0 / 15.
        h[1]['TGTDEC'] = 11.0

        assert np.allclose(_target_xy(h[0], outwcs_img), (209, 209))
        assert np.allclose(_target_xy(h[0], outwcs_spc), (9, 209))
        assert np.allclose(_target_xy(h[1], outwcs_img), (210, 210))
        assert np.allclose(_target_xy(h[1], outwcs_spc), (10, 210))

        # missing one or the other => warning, return (None, None)
        del h[0]['TGTRA']
        assert _target_xy(h[0], outwcs_img) == (None, None)
        del h[1]['TGTDEC']
        assert _target_xy(h[1], outwcs_img) == (None, None)
        del h[1]['TGTRA']
        assert _target_xy(h[1], outwcs_img) == (None, None)

    def test_coadd_target(self, capsys):
        # for spectral cube
        h, d, v, e = self.make_test_data(nfiles=2, spec_style=True)

        # add TGTRA/DEC
        # pixel scale is 1.0, so 1 degree movement is 1 pixel
        # (with RA recorded in hours)
        h[0]['TGTRA'] = 10.0 / 15.
        h[0]['TGTDEC'] = 10.0
        h[1]['TGTRA'] = 11.0 / 15.
        h[1]['TGTDEC'] = 11.0

        # output shape is 1 off when correcting for target motion
        first = coadd(h, d, v, e, reference='first', cube=True, wcskey='A')
        assert first[1].shape == (20, 20, 2)
        target = coadd(h, d, v, e, reference='target', cube=True, wcskey='A')
        assert target[1].shape == (20, 21, 1)

        # for 2D image
        h, d, v, e = self.make_test_data(nfiles=2, spec_style=False)

        # add TGTRA/DEC
        h[0]['TGTRA'] = 10.0 / 15.
        h[0]['TGTDEC'] = 10.0
        h[1]['TGTRA'] = 11.0 / 15.
        h[1]['TGTDEC'] = 11.0

        # output shape is 1 off
        first = coadd(h, d, v, e, reference='first')
        assert first[1].shape == (20, 20)
        target = coadd(h, d, v, e, reference='target')
        assert target[1].shape == (21, 21)

        # missing keys in first data
        del h[0]['TGTRA']
        target = coadd(h, d, v, e, reference='target')
        assert target[1].shape == (20, 20)
        assert 'Missing TGTRA or TGTDEC; ' \
               'cannot reference to target' in capsys.readouterr().err

        # missing key in subsequent data
        h[0]['TGTRA'] = 10.0
        del h[1]['TGTRA']
        target = coadd(h, d, v, e, reference='target')
        assert target[1].shape == (20, 20)
        assert 'Missing target RA/Dec in file ' \
               'test2.fits' in capsys.readouterr().err

    def test_bad_data(self, capsys):
        h, d, v, e = self.make_test_data(nfiles=3, spec_style=False)

        # default: average of 1, 2, 3
        default = coadd(h, d, v, e, weighted=False, robust=False)
        assert np.allclose(np.nanmean(default[1]), 2)

        # all errors bad in third file: average of 1, 2
        vcopy = v[2].copy()
        v[2] *= np.nan
        result1 = coadd(h, d, v, e, weighted=False, robust=False)
        assert np.allclose(np.nanmean(result1[1]), 1.5)
        assert 'No good data in test3.fits' in capsys.readouterr().err

        # all data bad in third file
        v[2] = vcopy
        d[2] *= np.nan
        result2 = coadd(h, d, v, e, weighted=False, robust=False)
        assert np.allclose(np.nanmean(result2[1]), 1.5)
        assert 'No good data in test3.fits' in capsys.readouterr().err

        # all data bad in second and third file:
        # only one image remaining, but allows output anyway
        d[1] *= np.nan
        result = coadd(h, d, v, e, weighted=False, robust=False)
        assert np.allclose(np.nanmean(result[1]), 1)

    def test_rotate_keys(self):
        h, d, v, e = self.make_test_data(nfiles=2, spec_style=False)

        # no rotate: crota2 out matches in
        no_rot = coadd(h, d, v, e, rotate=False)
        assert np.allclose(no_rot[0]['CROTA2'], h[0]['CROTA2'])
        assert not np.allclose(no_rot[0]['CROTA2'], 0.0)

        # rotate: crota2 out is 0
        rot = coadd(h, d, v, e, rotate=True)
        assert np.allclose(rot[0]['CROTA2'], 0.0)

        # rotate: cdelt1 out is negative
        h[0]['CDELT1'] = np.abs(h[0]['CDELT1'])
        rot = coadd(h, d, v, e, rotate=True)
        assert rot[0]['CDELT1'] < 0.0

        # make spectral style data
        h, d, v, e = self.make_test_data(nfiles=2, spec_style=True)

        # rotate with PC* keys: removed from out if rotate
        kset = ['PC2_2A', 'PC2_3A', 'PC3_2A', 'PC3_3A']
        for key in kset:
            h[0][key] = np.sqrt(2) / 2
        h[0]['PC2_3A'] *= -1

        no_rot = coadd(h, d, v, e, rotate=False, spectral=True, wcskey='A')
        for key in kset:
            assert np.allclose(no_rot[0][key], h[0][key])

        rot = coadd(h, d, v, e, rotate=True, spectral=True, wcskey='A')
        for key in kset:
            assert key not in rot[0]
