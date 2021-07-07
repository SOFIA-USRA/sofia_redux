# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
import sofia_redux.instruments.forcast.imgnonlin as u


def fake_data(shape=(3, 256, 256), value=2.0):
    ndim = 1 if len(shape) == 2 else shape[0]
    data = np.full(shape, value)
    header = fits.header.Header()
    header['DETCHAN'] = 1  # LWC
    header['EPERADU'] = 136  # LO
    header['NLRLWCLO'] = 0.5  # refsig
    header['NLSLWCLO'] = 3.0  # scale
    header['NLCLWCLO'] = str([1.0, 0.5, 0.0])
    header['LIMLWCLO'] = str([-9999999, 9999999])
    header['NLINSLEV'] = str([4] * ndim)
    return data, header


class TestImgnonlin(object):

    def test_addhist(self):
        header = fits.header.Header()
        u.addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Image nonlin: test history message'

    def test_camcap(self):
        _, header = fake_data()
        assert u.get_camera_and_capacitance(header) == 'LWCLO'
        _, header = fake_data()
        del header['EPERADU']
        assert u.get_camera_and_capacitance(header) is None
        _, header = fake_data()
        header['EPERADU'] = 999
        assert u.get_camera_and_capacitance(header) is None

    def test_get_reference_scale(self):
        dripconfig.load()
        for key in ['nlrlwclo', 'nlslwclo']:
            if key in dripconfig.configuration:
                del dripconfig.configuration[key]
        _, header = fake_data()
        camcap = u.get_camera_and_capacitance(header)
        assert u.get_reference_scale(header, camcap) == (0.5, 3)
        del header['NLRLWCLO']
        del header['NLSLWCLO']
        assert u.get_reference_scale(header, camcap) == (9000, 9000)

    def test_get_coefficients(self):
        dripconfig.load()
        if 'nlclwclo' in dripconfig.configuration:
            del dripconfig.configuration['nlclwclo']
        _, header = fake_data()
        camcap = u.get_camera_and_capacitance(header)
        del header['NLCLWCLO']
        assert u.get_coefficients(header, camcap, update=True) is None
        _, header = fake_data()
        result = u.get_coefficients(header, camcap, update=True)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert header['NLINC0'] == 1
        assert header['NLINC1'] == 0.5
        assert header['NLINC2'] == 0

    def test_get_limits(self):
        dripconfig.load()
        for key in ['limlwclo']:
            if key in dripconfig.configuration:
                del dripconfig.configuration[key]
        _, header = fake_data()

        camcap = u.get_camera_and_capacitance(header)
        result = u.get_coeff_limits(header, camcap, update=True)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0] == -9999999
        assert result[1] == 9999999
        assert "level limits are" in repr(header)

        # missing entirely
        del header['LIMLWCLO']
        assert u.get_coeff_limits(header, camcap) is None

        # set to a single value
        header['LIMLWCLO'] = str([-9999999])
        assert u.get_coeff_limits(header, camcap) is None

    def test_errors(self):
        dripconfig.load()
        data, header = fake_data()
        for key in header:
            if key.lower() in dripconfig.configuration:
                del dripconfig.configuration[key.lower()]
        ndim = 1 if len(data.shape) == 2 else data.shape[0]
        assert u.imgnonlin(np.zeros(10), header) is None
        del header['NLINSLEV']
        assert u.imgnonlin(data, header) is None
        header['NLINSLEV'] = str([1] * (ndim + 1))
        assert u.imgnonlin(data, header) is None
        data, header = fake_data()
        del header['EPERADU']
        assert u.imgnonlin(data, header) is None
        data, header = fake_data()
        del header['NLCLWCLO']
        assert u.imgnonlin(data, header) is None
        data, header = fake_data()
        del header['LIMLWCLO']
        assert u.imgnonlin(data, header) is None
        data, header = fake_data()
        header['LIMLWCLO'] = str([0, 0])
        assert u.imgnonlin(data, header) is None

        # bad header
        assert u.imgnonlin(data, None) is None

        # bad variance
        assert u.imgnonlin(data, header, variance=np.zeros(10)) is None

    def test_success(self):
        dripconfig.load()
        data, header = fake_data(value=2.0)
        for key in header:
            if key.lower() in dripconfig.configuration:
                del dripconfig.configuration[key.lower()]
        camcap = u.get_camera_and_capacitance(header)
        refscale = u.get_reference_scale(header, camcap)
        siglev = u.get_siglev(header)
        coeffs = u.get_coefficients(header, camcap)
        xval = (siglev - refscale[0]) / refscale[1]
        variance = np.full_like(data, 1.0)
        expected = coeffs[0] + coeffs[1] * xval + coeffs[2] * xval ** 2
        dexp = 2.0 / expected[0]
        vexp = 1 / expected[0] ** 2
        d, v = u.imgnonlin(data, header, variance=variance)
        assert np.allclose(d, dexp)
        assert np.allclose(v, vexp)

        # single plane data
        data, header = fake_data(shape=(256, 256))
        d, v = u.imgnonlin(data, header, variance=np.full_like(data, 1.0))
        assert np.allclose(d, dexp)
        assert np.allclose(v, vexp)
