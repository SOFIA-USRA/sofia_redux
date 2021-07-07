# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from sofia_redux.instruments.forcast.flat import flat, addhist
import numpy as np


def fake_data(shape=(256, 256), data=3.0, dark=1.0, flatsum=2.0,
              vflat=0.1, vdark=0.2, var=0.3):

    test = np.full(shape, data)
    header = fits.header.Header()
    header['ICONFIG'] = 'SPECTROSCOPY'
    flatsum = np.full(shape[-2:], flatsum)
    darksum = np.full(shape[-2:], dark)
    variance = np.full(shape, var)
    flatvar = np.full(shape[-2:], vflat)
    darkvar = np.full(shape[-2:], vdark)
    result = (data - dark) / flatsum
    resultvar = (var + vdark + ((result ** 2) * vflat)) / (flatsum ** 2)
    return {
        'data': test, 'variance': variance, 'header': header,
        'flatsum': flatsum, 'darksum': darksum, 'flatvar': flatvar,
        'darkvar': darkvar, 'expect': result, 'vexpect': resultvar
    }


class TestFlat(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Flat: test history message'

    def test_failure(self):
        t = fake_data()
        assert flat(t['data'], t['flatsum']) is None
        header = t['header']
        assert flat(np.zeros(10), t['flatsum'], header=header) is None
        assert flat(t['data'], np.zeros(10), header=header) is None
        assert flat(t['data'], t['flatsum'], header=header,
                    darksum=np.zeros(10)) is None

    def test_success(self):
        for shape in ((256, 256), (3, 256, 256)):
            t = fake_data(shape=shape)
            f, v = flat(t['data'], t['flatsum'], darksum=t['darksum'],
                        header=t['header'], variance=t['variance'],
                        flatvar=t['flatvar'], darkvar=t['darkvar'])
            assert isinstance(f, np.ndarray)
            assert isinstance(v, np.ndarray)
            assert f.shape == v.shape
            assert np.allclose(f, t['expect'])
            assert np.allclose(v, t['vexpect'])

    def test_no_dark(self):
        for shape in ((256, 256), (3, 256, 256)):
            t = fake_data(shape=shape, dark=0, vdark=0)
            f, v = flat(t['data'], t['flatsum'], header=t['header'],
                        variance=t['variance'], flatvar=t['flatvar'])
            assert isinstance(f, np.ndarray)
            assert isinstance(v, np.ndarray)
            assert f.shape == v.shape
            assert np.allclose(f, t['expect'])
            assert np.allclose(v, t['vexpect'])

    def test_no_variance(self):
        for shape in ((256, 256), (3, 256, 256)):
            t = fake_data(shape=shape)
            f, v = flat(t['data'], t['flatsum'], darksum=t['darksum'],
                        header=t['header'], variance=t['variance'],
                        flatvar=t['flatvar'], darkvar=None)
            assert v is None
            assert isinstance(f, np.ndarray)
            f, v = flat(t['data'], t['flatsum'], header=t['header'],
                        variance=t['variance'], flatvar=None)
            assert v is None
            assert isinstance(f, np.ndarray)
            f, v = flat(t['data'], t['flatsum'], header=t['header'],
                        variance=None, flatvar=t['flatvar'])
            assert v is None
            assert isinstance(f, np.ndarray)
            f, v = flat(t['data'], t['flatsum'], header=t['header'],
                        variance=np.zeros((10, 10)), flatvar=t['flatvar'])
            assert v is None
            assert isinstance(f, np.ndarray)
