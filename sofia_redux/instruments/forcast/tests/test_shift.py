# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
from sofia_redux.instruments.forcast.shift import shift, symmetric_ceil


def fake_data(shape=(10, 10), level=1.0):
    s = np.array(shape)
    crpix = np.fix(s / 2)
    data = np.zeros(s)
    data[tuple(crpix.astype(int))] = level
    return data, np.flip(crpix)


class TestShift(object):

    def test_errors(self, mocker):
        data, _ = fake_data()

        assert shift(None, [1, 1]) is None
        assert shift(data, [1]) is None
        assert shift(data, [1, 1], order=-1) is None

        badvar = np.zeros(5)
        shifted, var = shift(data, [1, 1], variance=badvar)
        assert var is None
        assert isinstance(shifted, np.ndarray)
        assert not np.allclose(shifted, data)

    def test_symmetric_ceil(self):
        assert symmetric_ceil(1.5) == 2
        assert symmetric_ceil(-1.5) == -2
        assert np.all(np.array(symmetric_ceil([-1, -0.5])) == -1)
        assert np.all(np.array(symmetric_ceil([1, 1e-6])) == 1)

    def test_shift(self, capsys):
        level = 1
        data, crpix = fake_data(level=level)
        variance = data.copy() * 2
        # use linear interpolation for all tests
        crnew = crpix.copy()
        offset = np.array([0.5, 0.5])
        d, v = shift(data, offset, order=1, mode='wrap',
                     variance=variance, crpix=crnew)
        # Check flux conservation
        assert np.nansum(d) == np.nansum(data)
        # Check something happened
        assert not np.allclose(d, data)

        # Check data and variance were handled appropriately
        d0, _ = shift(data, offset, order=0, mode='wrap')
        assert np.allclose(d0 * 2, v)

        # Check new CRPIX is as expected
        assert np.allclose(offset + crpix, crnew)
        c = tuple(crpix.astype(int))
        # Check values are as expected (0.5 pixel offset
        # should split the value over 4 pixels)
        for i in range(2):
            for j in range(2):
                assert d[c[0] + j, c[1] + i] == level / 4

        # check bad crpix input type
        bad_crpix = 10
        d1, _ = shift(data, offset, order=0, mode='wrap',
                      crpix=bad_crpix)
        capt = capsys.readouterr()
        assert 'invalid crpix' in capt.err
        assert bad_crpix == 10
        assert np.allclose(d1, d0)

    def test_resize(self):
        data, crpix = fake_data(level=1.0)
        header = fits.header.Header()
        header['CRPIX1'] = crpix[0]
        header['CRPIX2'] = crpix[1]
        crpix2 = crpix + 1
        variance = data.copy() * 2
        crnew = crpix2.copy()
        offset = np.array([3, -4])
        d, v = shift(data, offset, order=1, resize=True, header=header,
                     variance=variance, crpix=crnew)
        assert np.sum(~np.isnan(d)) == data.size
        assert d.size > data.size
        assert np.nansum(d) == np.nansum(data)
        yx = np.unravel_index(np.argmax(data), data.shape)
        dy = header['CRPIX2'] - crpix[1]
        dx = header['CRPIX1'] - crpix[0]
        yx2 = int(yx[0] + dy), int(yx[1] + dx)
        yxmax = np.unravel_index(np.nanargmax(d), d.shape)
        assert yx2 == yxmax
        assert v.shape == d.shape
        vmax = np.unravel_index(np.nanargmax(v), v.shape)
        assert yx2 == vmax

    def test_resize_only(self):
        data, crpix = fake_data(level=1.0)
        header = fits.header.Header()
        header['CRPIX1'] = crpix[0]
        header['CRPIX2'] = crpix[1]
        crpix2 = crpix + 1
        variance = data.copy() * 2
        crnew = crpix2.copy()
        offset = np.array([3, -4])
        d, v = shift(data, offset, order=1, resize=True, header=header,
                     variance=variance, crpix=crnew, no_shift=True)
        assert np.sum(~np.isnan(d)) == data.size
        assert d.size > data.size
        assert np.nansum(d) == np.nansum(data)
        yx = np.unravel_index(np.argmax(data), data.shape)
        dy = header['CRPIX2'] - crpix[1]
        dx = header['CRPIX1'] - crpix[0]
        yx2 = int(yx[0] + dy), int(yx[1] + dx)
        yxmax = np.unravel_index(np.nanargmax(d), d.shape)
        assert yx2 == yxmax
        assert v.shape == d.shape
        vmax = np.unravel_index(np.nanargmax(v), v.shape)
        assert yx2 == vmax
        data, crpix = fake_data(level=1.0)
        variance = data.copy() * 2
        ds, vs = shift(data, offset, order=1, resize=True, header=header,
                       variance=variance, crpix=crnew, no_shift=False)
        assert np.nanargmax(ds) != np.nanargmax(d)
        assert np.nanargmax(vs) != np.nanargmax(v)
