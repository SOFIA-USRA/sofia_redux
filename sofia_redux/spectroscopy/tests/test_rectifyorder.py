# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.rectifyorder \
    import get_rect_xy, trim_xy, reconstruct_slit, rectifyorder, update_wcs


@pytest.fixture
def grids():
    y1, x1 = np.mgrid[:64, :64]
    x2 = x1.astype(float)
    y2 = y1.astype(float)
    return x1, y1, x2, y2


@pytest.fixture
def data():
    y1, x1 = np.mgrid[:64, :64]
    x2 = x1.astype(float)
    x2 += y1 / 10
    y2 = y1.astype(float)
    wavecal, spatcal, xgrid, ygrid = get_rect_xy(x2, y2, x1, y1, dx=1, dy=1)
    imshape = wavecal.shape  # [0] + 1, wavecal.shape[1] + 1
    ordermask = np.full(imshape, 0)
    image = np.full(imshape, 2.0)
    variance = np.full(imshape, 3.0)
    badpix_mask = np.full(imshape, True)
    badpix_mask[10] = False
    return (image, ordermask, wavecal, spatcal,
            xgrid, ygrid, variance, badpix_mask)


def test_get_rect_xy(grids):
    x1, y1, x2, y2 = grids
    assert get_rect_xy(x1, y1[:-1], x2, y2) is None
    assert get_rect_xy(x1, y1, x2[:-1], y2) is None
    assert get_rect_xy(x1, y1, x2, y2[:-1]) is None
    mask = np.full(x1.shape, True)
    assert get_rect_xy(x1, y1, x2, y2, mask=mask[:-1]) is None
    assert get_rect_xy(x1, y1, x2, y2, mask=~mask) is None
    ix, iy, xout, yout = get_rect_xy(x1, y1, x2, y2, dx=1, dy=1)
    assert np.allclose(ix, x2)
    assert np.allclose(iy, y2)
    assert np.allclose(xout, x1[0])
    assert np.allclose(yout, y1[:, 0])
    ix, iy, xout, yout = get_rect_xy(x1, y1, x2, y2)
    expected_y, expected_x = np.mgrid[:ix.shape[0], :ix.shape[1]]
    assert np.allclose(expected_y, iy)
    assert np.allclose(expected_x, ix)


def test_trim_xy(grids):
    x1, y1, x2, y2 = grids
    xarray, yarray, xgrid, ygrid = get_rect_xy(x2, y2, x1, y1)
    ix, iy, xg, yg = trim_xy(xarray, yarray, xgrid, ygrid,
                             xbuffer=2, ybuffer=3)
    for arr in [ix, iy]:
        assert arr.shape == (xarray.shape[0] - 6, xarray.shape[1] - 4)
    assert len(xg) == len(xgrid) - 4
    assert len(yg) == len(ygrid) - 6
    ix, iy, xg, yg = trim_xy(xarray, yarray, xgrid, ygrid,
                             xrange=(10.5, 21), yrange=(10.5, 41))

    assert np.allclose(ix.shape, (30, 10), atol=1)
    assert ix.shape == iy.shape

    assert len(xg) == ix.shape[1]
    assert len(yg) == ix.shape[0]


def test_reconstruct_slit(data):
    (image, ordermask, wavecal, spatcal, xgrid,
     ygrid, variance, badpix_mask) = data
    result = reconstruct_slit(image, wavecal, spatcal, xgrid, ygrid,
                              badfrac=0.01, badpix_mask=badpix_mask,
                              variance=variance)
    mask = result['mask']
    assert mask.any()
    assert not mask[9].any()
    assert not mask[10].any()
    assert np.allclose(result['pixsum'][mask], 1)
    assert np.allclose(result['image'][mask], 2)
    assert np.allclose(result['variance'][mask], 3)
    for i in range(2, 7):
        assert not mask[i * 10, :i].any()
        assert mask[i * 10, i:].all()

    # if mask is None, it is still updated, but more pixels are used
    result2 = reconstruct_slit(image, wavecal, spatcal, xgrid, ygrid,
                               badfrac=0.01, badpix_mask=None,
                               variance=variance)
    assert np.sum(result2['mask']) > np.sum(result['mask'])


def test_rectify(data):
    (image, ordermask, wavecal, spatcal, xgrid,
     ygrid, variance, badpix_mask) = data
    bitmask = np.full_like(ordermask, 1)
    result = rectifyorder(image, ordermask, wavecal, spatcal, 0,
                          variance=variance, mask=badpix_mask,
                          bitmask=bitmask, dw=1, ds=1, ybuffer=0)
    s1 = image.shape
    s2 = result['image'].shape
    assert np.allclose(s2[0], s1[0] - 5, atol=1)
    assert rectifyorder(
        image, ordermask, wavecal[:-1], spatcal, 0) is None


def test_update_wcs(data):
    spatcal, wavecal = np.mgrid[:10, :10]

    result = {'header': fits.Header(),
              'wave': np.arange(10, dtype=float),
              'spatial': np.arange(10, dtype=float)}

    # with minimal header, only primary keys appear in result
    expected = {
        'CTYPE1': 'LINEAR',
        'CTYPE2': 'LINEAR',
        'CUNIT1': 'um',
        'CUNIT2': 'arcsec',
        'CRPIX1': 6.0,
        'CRPIX2': 6.0,
        'CRVAL1': 5.0,
        'CRVAL2': 5.0,
        'CDELT1': 1.0,
        'CDELT2': 1.0,
        'CROTA2': 0.0,
        'SPECSYS': 'TOPOCENT'
    }
    update_wcs(result, spatcal)
    for key in result['header']:
        if isinstance(result['header'][key], str):
            assert result['header'][key] == expected[key]
        else:
            assert np.allclose(result['header'][key], expected[key])

    # with a non-trivial header 2ndary WCS is added
    header = fits.Header({'CRPIX1': 1.0,
                          'CRPIX2': 2.0,
                          'CRVAL1': 1.0,
                          'CRVAL2': 2.0,
                          'CROTA2': 10.0,
                          'XUNITS': 'm',
                          'SPECSYS': 'BARYCENT'})
    result = {'header': header,
              'wave': np.arange(10, dtype=float),
              'spatial': np.arange(10, dtype=float)}

    expected2 = {
        'SPECSYS': 'BARYCENT',
        'SPECSYSA': 'BARYCENT',
        'CTYPE1A': 'WAVE',
        'CTYPE2A': 'DEC--TAN',
        'CTYPE3A': 'RA---TAN',
        'CUNIT1A': 'm',
        'CUNIT2A': 'deg',
        'CUNIT3A': 'deg',
        'CRPIX1A': 6.0,
        'CRPIX2A': 2.0,
        'CRPIX3A': 1.0,
        'CRVAL1A': 5.0,
        'CRVAL2A': 2.0,
        'CRVAL3A': 1.0,
        'CDELT1A': 1.0,
        'CDELT2A': 1.0 / 3600.,
        'CDELT3A': -1.0 / 3600.,
        'PC2_2A': np.cos(np.radians(10)),
        'PC2_3A': -np.sin(np.radians(10)),
        'PC3_2A': np.sin(np.radians(10)),
        'PC3_3A': np.cos(np.radians(10)),
        'RADESYSA': 'FK5',
        'EQUINOXA': 2000.,
        'XUNITS': 'm'
    }
    expected.update(expected2)
    expected['CUNIT1'] = 'm'

    update_wcs(result, spatcal)

    for key in result['header']:
        if isinstance(result['header'][key], str):
            assert result['header'][key] == expected[key]
        else:
            assert np.allclose(result['header'][key], expected[key])
