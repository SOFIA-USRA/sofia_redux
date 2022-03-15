# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
import sofia_redux.instruments.forcast.undistort as u
from sofia_redux.toolkit.image.warp import warp_image


def fake_data():
    """
    This is hard to test as I'm not sure what the expected
    results should be.  It depends on the algorithm.  For now
    test contraction warp.  At least that way I know what to expect
    """
    imgnx, imgny = 256, 256  # make it 2^n
    nx, ny = 12, 12  # something that does not integer divide imgnx, imgny
    yy, xx = np.mgrid[:imgny, :imgnx]
    factor = 2  # something that can integer divide imgnx, imgny
    dyidx, dxidx = imgny // (ny - 1), imgnx // (nx - 1)
    xin, yin, xout, yout = [], [], [], []
    for iy in range(ny):
        for ix in range(nx):
            xin.append(xx[iy * dyidx, ix * dxidx])
            yin.append(yy[iy * dyidx, ix * dxidx])
    xin = np.array(xin)
    yin = np.array(yin)
    xout, yout = xin / factor, yin / factor

    data = np.zeros_like(xx, dtype=float)
    expected = np.zeros_like(data)
    spacing = imgny // (factor * 4), imgnx // (factor * 4)
    sexp = tuple(x // factor for x in spacing)
    newx0, newy0 = imgnx / factor / 2, imgny / factor / 2

    for iy in range(factor * 4):
        for ix in range(factor * 4):
            data[iy * spacing[0] + (spacing[0] // 2),
                 ix * spacing[1] + (spacing[1] // 2)] = 1
            expected[iy * sexp[0] + (sexp[0] // 2),
                     ix * sexp[1] + (sexp[1] // 2)] = 1

    pinpos = {
        'model': np.stack((xout, yout), axis=1),
        'pins': np.stack((xin, yin), axis=1),
        'nx': 256, 'ny': 256,
        'dx': 1.0, 'dy': 2.0,  # The ratio dy/dx is used in rebin_image()
        'angle': 0.0,  # not used but will appear in header under PIN_MOD
        'order': 3,
    }

    return {
        'data': data, 'expected': expected, 'imgnx': imgnx, 'imgny': imgny,
        'xin': xin, 'yin': yin, 'xout': xout, 'yout': yout, 'pinpos': pinpos,
        'newx0': newx0, 'newy0': newy0, 'factor': factor}


class TestUndistort(object):

    def test_addhist(self):
        header = fits.header.Header()
        u.addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Distortion: test history message'

    def test_default_pinpos(self):
        results = u.default_pinpos()
        assert isinstance(results, dict)
        keys = ['model', 'pins', 'dx', 'dy', 'nx', 'ny', 'order', 'angle']
        for key in keys:
            assert key in results
        for key in ['model', 'pins']:
            assert isinstance(results[key], np.ndarray)
            assert len(results[key].shape) == 2
            assert results[key].shape[1] == 2
        assert results['model'].shape == results['pins'].shape
        for key in ['dx', 'dy', 'nx', 'ny']:
            assert results[key] > 0
        for key in ['nx', 'ny']:
            assert isinstance(results[key], int)
        assert isinstance(results['angle'], (int, float))

    def test_get_pinpos_integrity(self):
        dripconfig.load()
        dripconfig.configuration['fpinhole'] = 'pinhole_locs.txt'
        drip_header = fits.header.Header()
        default_header = fits.header.Header()
        user_header = fits.header.Header()

        # Test defaults
        drip_pinpos = u.get_pinpos(drip_header)
        default_pinpos = u.get_pinpos(default_header, pinpos='default')
        assert isinstance(drip_pinpos, dict)
        assert isinstance(default_pinpos, dict)
        testarr = np.array([[*(range(10))], [*(range(10))]]).T
        user_dict = {
            'model': testarr.copy(), 'pins': testarr.copy(),
            'dx': 1.0, 'dy': 1.0, 'order': 2, 'angle': 90.0
        }
        user_pinpos = u.get_pinpos(user_header, pinpos=user_dict)
        assert isinstance(user_pinpos, dict)

        # test PIN_MOD in header as [float, float, float, int]
        assert 'PIN_MOD' in user_header
        pinmod = re.split(r'[\[\],]', user_header['PIN_MOD'])
        pinmod = [x for x in pinmod if x != '']
        assert len(pinmod) == 4
        for val in pinmod[:3]:
            assert '.' in val
        assert '.' not in pinmod[3]

        # test failures - pinpos
        failure_pinpos_keyvals = [
            'a',
            {'model': testarr.copy()},
            {'pins': testarr.copy()},
            {'model': testarr.copy(), 'pins': 'a'},
            {'pins': testarr.copy(), 'model': 'a'},
            {'pins': testarr.copy()[:, 0], 'model': testarr.copy()},
            {'pins': testarr.copy()[:2, :], 'model': testarr.copy()[:3, :]}
        ]
        msg = 'Distortion: correction was not applied (Invalid pinpos)'
        fails = 0
        for pinpos in failure_pinpos_keyvals:
            pp = u.get_pinpos(user_header, pinpos=pinpos)
            assert pp is None
            fails += 1
            assert list(user_header.values()).count(msg) == fails

        # PIN_MOD errors
        msg = 'Distortion: correction was not applied (Invalid coeffs)'
        user_dict = {'model': testarr.copy(), 'pins': testarr.copy(),
                     'order': 2, 'angle': 90.0}
        del user_header['PIN_MOD']
        assert msg not in user_header.values()
        pp = u.get_pinpos(user_header, pinpos=user_dict)
        assert pp is None
        assert msg in user_header.values()
        user_header['PIN_MOD'] = 'foo'
        pp = u.get_pinpos(user_header)
        assert pp is None
        assert list(user_header.values()).count(msg) == 2
        user_header['PIN_MOD'] = '[1,2,3]'
        pp = u.get_pinpos(user_header)
        assert pp is None
        assert list(user_header.values()).count(msg) == 3

    def test_get_pinpos_from_file(self, tmpdir):
        header = fits.header.Header()
        testfile = tmpdir.join('testfile')

        # non-existent file returns None
        assert u.get_pinpos(header, pinpos=str(testfile)) is None

        # write a file
        contents = ['xid\tyid\txpos\typos\n']
        for i in range(5):
            contents.append('0\t{}\t1\t2\n'.format(i))
        testfile.write(''.join(contents))

        assert u.get_pinpos(header, pinpos=str(testfile)) is not None

    def test_get_pinpos_rotation(self):
        header = fits.header.Header()
        header['NAXIS1'] = 4
        header['NAXIS2'] = 4
        testarr = np.zeros((3, 2))
        expect = np.zeros_like(testarr)
        # 0 0 0 1          1 0 0 0
        # 0 0 0 0  90 deg  0 0 0 0
        # 0 1 0 0  ----->  0 0 1 0
        # 1 0 0 0          0 0 0 1
        angle = 90
        testarr[0], expect[0] = [0, 0], [3, 0]
        testarr[1], expect[1] = [1, 1], [2, 1]
        testarr[2], expect[2] = [3, 3], [0, 3]

        pinpos = {
            'model': testarr.copy(), 'pins': testarr.copy(),
            'dx': 1.0, 'dy': 1.0, 'order': 2, 'angle': -1}
        pp = u.get_pinpos(header, pinpos=pinpos, rotate=True)
        assert (pp['model'] == pinpos['model']).all()
        header['NODANGLE'] = angle
        pp = u.get_pinpos(header, pinpos=pinpos, rotate=True)
        # check memory references
        assert not (pp['model'] == pinpos['model']).all()
        assert (pp['model'] == expect).all()

    def test_find_pixat11(self):
        for ttype in ['polynomial', 'piecewise-affine']:
            test = fake_data()
            data = test['data']
            xin, yin, xout, yout = \
                test['xin'], test['yin'], test['xout'], test['yout']
            imgnx, imgny = test['imgnx'], test['imgny']
            xrange = 0, imgnx - 1
            yrange = 0, imgny - 1
            eps = 1e-6
            x0, y0 = imgnx / 2, imgny / 2
            _, transform = warp_image(data, xin, yin, xout, yout, order=4,
                                      get_transform=True, transform=ttype)

            p11 = u.find_pixat11(transform, x0, y0, epsilon=eps,
                                 xrange=xrange, yrange=yrange)
            assert np.allclose(p11['x1ref'], test['newx0'], atol=eps)
            assert np.allclose(p11['y1ref'], test['newy0'], atol=eps)
            assert np.allclose(abs(p11['x01'] - p11['x0']), test['factor'])
            assert np.allclose(abs(p11['y01'] - p11['y0']), test['factor'])

            # one iteration -- fails
            p11 = u.find_pixat11(transform, x0, y0, epsilon=eps, xrange=xrange,
                                 yrange=yrange, maxiter=1, direct=False)
            assert p11 is None

            # test xrange, yrange
            t1 = 100
            t2 = 150
            p11 = u.find_pixat11(transform, x0, y0, epsilon=eps,
                                 xrange=(t1, t2), yrange=(t1, t2),
                                 maxiter=100, direct=False)
            assert t1 <= p11['x1ref'] <= t2
            assert t1 <= p11['y1ref'] <= t2
            assert not np.allclose(p11['x1ref'], test['newx0'], atol=eps)
            assert not np.allclose(p11['y1ref'], test['newy0'], atol=eps)

            # test epsilon
            p11 = u.find_pixat11(transform, x0, y0, epsilon=0.1,
                                 direct=False,
                                 xrange=xrange, yrange=yrange)
            assert int(p11['x1ref']) == p11['x1ref']

            # xrange, yrange None => no bound
            from astropy import log
            log.error(ttype)
            p11_ub = u.find_pixat11(transform, x0, y0, epsilon=eps,
                                    direct=False,
                                    xrange=None, yrange=None)
            # will either fail or be correct (bounds are recommended!)
            assert (p11_ub is None) or \
                   (np.allclose(p11['x1ref'], test['newx0'], atol=eps))

    def test_update_wcs(self):
        test = fake_data()
        data = test['data']
        xin, yin, xout, yout = \
            test['xin'], test['yin'], test['xout'], test['yout']
        imgnx, imgny = test['imgnx'], test['imgny']
        x0, y0 = imgnx / 2, imgny / 2
        _, transform = warp_image(data, xin, yin, xout, yout, order=4,
                                  get_transform=True)
        header = fits.header.Header()
        dxy = u.update_wcs('a', transform)
        assert dxy is None
        dxy = u.update_wcs(header, transform)
        assert dxy is None
        msg = 'Distortion: CRPIX1 or CRPIX2 are not in header. ' \
              'Skipping WCS update'
        assert msg in header['HISTORY']

        header = fits.header.Header()
        header['CRPIX1'], header['CRPIX2'] = x0, y0
        dxy = u.update_wcs(header, transform)
        assert np.allclose(header['CRPIX1'], x0 / test['factor'], atol=1)
        assert np.allclose(header['CRPIX2'], y0 / test['factor'], atol=1)
        msgs = [
            'Distortion: CDELT from stack= [-1,-1]',
            'Distortion: CROT from stack= [ - ,-1]']
        for msg in msgs:
            assert msg in header['HISTORY']
        assert not dxy['update_cdelt']

        header = fits.header.Header()
        header['CRPIX1'], header['CRPIX2'] = x0, y0
        header['CDELT1'], header['CDELT2'], header['CROTA2'] = 1.0, 1.0, 0
        dxy = u.update_wcs(header, transform)
        assert dxy['update_cdelt']
        assert header['CDELT1'] == dxy['x01'] - dxy['x0']
        assert header['CDELT2'] == dxy['y01'] - dxy['y0']

    def test_transform_image(self):
        test = fake_data()
        data = test['data']
        expected = test['expected']
        xin, yin, xout, yout = \
            test['xin'], test['yin'], test['xout'], test['yout']
        header = fits.header.Header()
        variance = np.ones_like(data)

        dout, vout = u.transform_image(data, xin, yin, xout, yout, order=3,
                                       header=header, variance=variance)
        delta = np.abs(expected - dout)
        assert np.allclose(np.nanmax(delta), 0, atol=1e-6)
        assert np.allclose(np.nanmax(delta), 0, atol=1e-6)
        assert np.isnan(vout).any()
        assert 'Distortion: correction model uses order 3' in header['HISTORY']

        # none variance, check dxy
        _, vout, dxy = u.transform_image(
            data, xin, yin, xout, yout, get_dxy=True)
        assert dxy is None
        assert vout is None

        # bad variance
        dout, vout = u.transform_image(data, xin, yin, xout, yout, order=3,
                                       header=header, variance=np.zeros(10))
        assert dout is not None
        assert vout is None

    def test_rebin_image(self):
        n = 100
        image = np.ones((n, n), dtype=float)
        flux = image.sum()
        variance = np.full_like(image, 2)
        factor = 2  # dy/dx to correct, so we're going to squish y if > 1
        header = fits.header.Header()
        header['CRPIX1'] = n / 2
        header['CRPIX2'] = n / 2

        imout, vout = u.rebin_image(
            image, factor, variance=variance, header=header,
            platescale=0.768)
        assert np.allclose(imout.sum(), flux)
        assert imout.shape[0] * factor == imout.shape[1]
        assert vout.shape == imout.shape
        assert not (vout == imout).any()
        assert header['CRPIX1'] != n / 2
        assert header['CRPIX2'] != n / 2
        assert header['CRPIX1'] > 0
        assert header['CRPIX2'] > 0
        assert 'CDELT1' not in header
        header['CDELT1'] = 1
        header['CDELT2'] = 1

        u.rebin_image(image, factor, variance=variance,
                      header=header, platescale=1)
        assert header['CDELT1'] == 1 / 3600
        assert header['CDELT2'] == 1 / 3600

        u.rebin_image(image, factor, variance=variance, header=header)
        assert header['CDELT1'] != 1 / 3600
        assert header['CDELT2'] != 1 / 3600
        fullhist = ''.join(header['HISTORY'])
        assert "new image size:" in fullhist
        assert "CDELT after rebin" in fullhist

        # bad variance
        d, v = u.rebin_image(
            image, factor, variance=np.zeros(10), header=header)
        assert d is not None
        assert v is None

        # none header -- no impact on data
        d, v = u.rebin_image(
            image, factor, variance=variance, header=None)
        assert np.allclose(d, imout)
        assert np.allclose(v, vout)

    def test_frame_image(self):
        n = 100
        image = np.ones((n, n), dtype=float)
        shape = (200, 200)
        border = 10
        variance = np.full_like(image, 2)
        dripconfig.load()
        default_border = dripconfig.configuration.get('border')
        if default_border is not None:
            del dripconfig.configuration['border']
        frame, var = u.frame_image(image, shape, border=border)
        assert frame.shape == (shape[0] + border * 2, shape[1] + border * 2)
        assert np.isnan(frame).any()
        assert not np.isnan(frame).all()
        assert var is None

        # Check NAXIS is updated
        header = fits.header.Header()
        frame, _ = u.frame_image(image, shape, border=border, header=header)
        assert header['NAXIS1'] == frame.shape[1]
        assert header['NAXIS2'] == frame.shape[0]

        # Check CRPIX is updated
        header['CRPIX1'] = shape[1] / 2
        header['CRPIX2'] = shape[0] / 2
        header['CRVAL1'] = 55
        header['CRVAL2'] = 55
        _ = u.frame_image(image, shape, border=border, header=header)
        assert header['CRPIX1'] != shape[1] / 2
        assert header['CRPIX2'] != shape[0] / 2
        hstr = ''.join(header['HISTORY'])
        assert 'CRPIX after border' in hstr
        assert 'CRVAL = [55' in hstr

        # unless WCS is false
        header = fits.header.Header()
        header['CRPIX1'] = shape[1] / 2
        header['CRPIX2'] = shape[0] / 2
        _ = u.frame_image(image, shape, border=border, header=header,
                          wcs=False)
        assert header['CRPIX1'] == shape[1] / 2
        assert header['CRPIX2'] == shape[0] / 2
        assert 'HISTORY' not in header

        # Check variance is updated
        _, var = u.frame_image(image, shape, border=border,
                               variance=variance)
        assert var.shape != (200, 200)
        assert np.nanmax(var) == 2
        assert np.isnan(var).any()

        # unless it's bad
        d, v = u.frame_image(image, shape, border=border,
                             variance=np.zeros(10))
        assert d is not None
        assert v is None

        # Check drip configuration will override border parameter
        dripconfig.configuration['border'] = 50
        frame, _ = u.frame_image(image, shape, border=10)
        assert frame.shape, (shape[0] + 100 == shape[1] + 100)
        dripconfig.load()

    def test_find_source(self, mocker):
        n = 100
        test = np.zeros((n, n))
        test[n // 2, n // 2] = 1
        dripconfig.load()
        dripconfig.configuration['border'] = 0
        header = fits.header.Header()

        u.find_source(test, header)
        assert header.get('SRCPOSX') == n // 2
        assert header.get('SRCPOSY') == n // 2

        # peakfind failure
        mocker.patch('sofia_redux.instruments.forcast.undistort.peakfind',
                     return_value=None)
        header = fits.header.Header()
        u.find_source(test, header)
        assert 'SRCPOSX' not in header
        assert 'SRCPOSY' not in header

        dripconfig.load()

    def test_undistort(self):

        data = fake_data()['data'].copy()
        pinpos = fake_data()['pinpos']
        variance = np.full_like(data, 2)

        # Check the two algorithms
        header1 = fits.header.Header()
        header2 = fits.header.Header()
        result1, var1 = u.undistort(
            data, header1, pinhole=pinpos,
            transform_type='piecewise-affine')
        result2, var2 = u.undistort(
            data, header2, pinhole=pinpos,
            transform_type='polynomial', variance=variance)
        assert result1.shape == result2.shape
        assert var2.shape == result2.shape
        assert var1 is None
        assert 'PRODTYPE' in header1

    def test_undistort_errors(self, mocker, capsys):
        testdata = fake_data()
        data = testdata['data'].copy()
        pinpos = testdata['pinpos']
        header = fits.header.Header()
        dripconfig.configuration['fpinhole'] = 'pinhole_locs.txt'

        # bad header
        result = u.undistort(data, 10, pinhole=pinpos)
        assert result is not None

        # bad shape keys, otherwise okay
        header['NAXIS1'] = 5
        header['NAXIS2'] = 5
        header['CRPIX1'] = data.shape[1] / 2
        header['CRPIX2'] = data.shape[0] / 2
        header['CRVAL1'] = 55
        header['CRVAL2'] = 55
        result = u.undistort(data, header, pinhole=pinpos)
        assert result is not None
        assert header['NAXIS1'] == result[0].shape[1]
        assert header['NAXIS2'] == result[0].shape[0]

        # bad data
        result = u.undistort(np.zeros(10), header, pinhole=pinpos)
        assert result is None

        # bad variance
        result = u.undistort(data, header, pinhole=pinpos,
                             variance=np.zeros(10))
        assert result is not None
        assert result[0] is not None
        assert result[1] is None

        # platescale is None
        result = u.undistort(data, header, pinhole=pinpos,
                             default_platescale=None)
        assert result is not None

        # find_source is called for NMC standards
        def mock_find(*args, **kwargs):
            print('find_source called')
        mocker.patch('sofia_redux.instruments.forcast.undistort.find_source',
                     mock_find)
        header['INSTMODE'] = 'C2N'
        header['SKYMODE'] = 'NMC'
        header['OBSTYPE'] = 'STANDARD_FLUX'
        result = u.undistort(data, header)
        assert result is not None
        capt = capsys.readouterr()
        assert 'find_source called' in capt.out

        # pinpos error
        mocker.patch('sofia_redux.instruments.forcast.undistort.get_pinpos',
                     return_value=None)
        result = u.undistort(data, header, pinhole=pinpos)
        assert result is None
        capt = capsys.readouterr()
        assert 'failed to find a valid pinhole model' in capt.err
