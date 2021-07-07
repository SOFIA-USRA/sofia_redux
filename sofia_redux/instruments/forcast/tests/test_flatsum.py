# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from tempfile import mkdtemp

from astropy.io import fits
import numpy as np

import sofia_redux.instruments.forcast.configuration as dripconfig
import sofia_redux.instruments.forcast.flatsum as u
from sofia_redux.instruments.forcast.tests.resources \
    import random_mask, add_jailbars

dripconfig.load()


def fake_data(shape=(4, 256, 256), value=2.0):
    header = fits.header.Header()
    data = np.full(shape, float(value))
    nframes = 1 if len(shape) == 2 else shape[0]
    header['OTMODE'] = 'AD'
    header['OTSTACKS'] = 2
    header['BGSCALE'] = True
    header['EPERADU'] = 1e6  # to get original data back
    header['FRMRATE'] = 1.0
    header['BGSUB'] = True
    header['BGSCALE'] = True
    header['CHOPTSAC'] = -1
    header['INSTMODE'] = 'STARE'
    header['JBCLEAN'] = 'MEDIAN'
    header['DETCHAN'] = 1  # LWC
    header['EPERADU'] = 136  # LO
    header['NLRLWCLO'] = 0.5  # refsig
    header['NLSLWCLO'] = 3.0  # scale
    header['NLCLWCLO'] = str([1.0, 0.5, 0.0])
    header['LIMLWCLO'] = str([-9999999, 9999999])
    header['NLINSLEV'] = str([4] * nframes)
    return data, header


def header_priority(header):
    dripconfig.load()
    for key in header:
        if key.lower() in dripconfig.configuration:
            del dripconfig.configuration[key.lower()]


class TestFlatsum(object):

    def test_addhist(self):
        header = fits.header.Header()
        u.addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Flatsum: test history message'

    def test_clean_flat(self, mocker):
        # test for when input is not an array
        assert u.clean_flat(None) is None

        # test good conditions
        for shape in ((64, 64), (2, 64, 64)):
            data, header = fake_data(shape, value=2.0)
            header_priority(header)
            add_jailbars(data)
            assert not np.allclose(np.nanmean(data), 2.0, atol=1e-2)
            mask = random_mask(data.shape[-2:], frac=0.1)  # 10 percent are bad
            variance = np.full_like(data, 3.0)
            if len(shape) == 2:
                data[mask] = -1
                variance[mask] = -1
            else:
                for d, v in zip(data, variance):
                    d[mask] = -1
                    v[mask] = -1
            d, v = u.clean_flat(data, header=header, variance=variance,
                                badmap=mask, jailbar=True)
            nans = np.isnan(d)
            assert nans.sum() < mask.sum()
            assert np.allclose(np.nanmean(d), 2.0, atol=0.2)
            assert np.allclose(np.nanmean(v), 3.0, atol=0.2)
            assert d.shape == data.shape
            assert d.shape == v.shape

        # test jbclean failure
        mocker.patch('sofia_redux.instruments.forcast.flatsum.jbclean',
                     return_value=None)
        data, header = fake_data((10, 10), value=2.0)
        assert u.clean_flat(data, jailbar=True) is None

    def test_get_flatbias(self):
        """
        Note that this also tests the current configuration file.
        A failure may indicate a problem there...
        """
        _, header = fake_data()
        header_priority(header)
        latest_vals = []
        for ilowcap in [True, False]:
            header['ILOWCAP'] = ilowcap
            for detchan in [0, 1]:
                header['DETCHAN'] = detchan
                value = u.get_flatbias(header)
                latest_vals.append(value)
        assert 0 not in latest_vals
        assert u.get_flatbias(header, pathcal='/does/not/exist') == 0

        pathcal = mkdtemp()
        biasfile = os.path.join(pathcal, 'biaslevels.txt')
        with open(biasfile, 'w') as f:
            print('# date   SWC_Lo SWC_Hi LWC_Lo LWC_Hi', file=f)
            print('20130101 1 2 3 4', file=f)
            print('20140101 11 22 33 44', file=f)
            print('20150101 111 222 333 444', file=f)
            print('20160101 1111 2222 3333 4444', file=f)
            print('99999999 11111 22222 33333 44444', file=f)

        header['DATE-OBS'] = '2014-06-01T00:00:00'
        fake_vals = []
        for detchan in [0, 1]:
            header['DETCHAN'] = detchan
            for ilowcap in [True, False]:
                header['ILOWCAP'] = ilowcap
                fake_vals.append(u.get_flatbias(header, pathcal=pathcal))
        assert np.allclose(fake_vals, [111, 222, 333, 444])

        # test bad date -- defaults to 99999999
        # (for last modified header)
        header['DATE-OBS'] = 'BADVAL-BADVAL-BADVAL'
        assert u.get_flatbias(header, pathcal=pathcal) == 44444

        if os.path.isfile(biasfile):
            os.remove(biasfile)
        if os.path.isdir(pathcal):
            os.rmdir(pathcal)

    def test_2d_masterflat(self):
        data, header = fake_data((64, 64), 2)
        flatbias = u.get_flatbias(header)
        variance = np.full_like(data, 0.5)
        data += flatbias
        dark = np.full_like(data, flatbias + 1)
        darkvar = np.full_like(variance, 1.0)
        d, v = u.create_master_flat(data, header, variance=variance)
        assert np.allclose(d, 2)
        assert np.allclose(v, 0.5)
        d, v = u.create_master_flat(
            data, header, variance=variance, dark=dark, darkvar=darkvar)
        assert np.allclose(d, 1)
        assert np.allclose(v, 1.5)

    def test_4n_masterflat(self):
        data, header = fake_data((4, 64, 64), 2)
        data[:2] += 3
        variance = np.full_like(data, 0.5)
        variance[:2] += 0.5
        d, v = u.create_master_flat(data, header, variance=variance)
        assert np.allclose(d, 3)
        assert np.allclose(v, 0.75)

    def test_nn_masterflat(self):
        data, header = fake_data((3, 64, 64), 2.0)
        data += u.get_flatbias(header)
        variance = np.full_like(data, 2.0)
        variance[1] += 1
        variance[2] += 2
        d, v = u.create_master_flat(data, header,
                                    variance=variance)
        assert np.allclose(d, 2)
        assert np.allclose(v, 1)
        dark = np.zeros((64, 64)) + u.get_flatbias(header)
        d, v = u.create_master_flat(data, header,
                                    variance=variance, dark=dark)
        assert np.allclose(d, 2)
        assert np.allclose(v, 1)

    def test_normalize_masterflat(self):
        data, header = fake_data((64, 64), 2.0)
        flatbias = u.get_flatbias(header)
        variance = np.full_like(data, 1.0)
        data += flatbias
        d, v = u.create_master_flat(
            data, header, variance=variance, normflat=True)
        assert np.allclose(d, 1)
        assert np.allclose(v, 0.25)

    def test_create_masterflat(self, capsys):
        data, header = fake_data((64, 64), 2)
        variance = np.full_like(data, 0.5)
        dark = np.full_like(data, 2)
        data += u.get_flatbias(header)
        _, v = u.create_master_flat(
            data, header, dark=dark, variance=variance, darkvar=np.zeros(10))
        assert np.allclose(variance, v)
        assert u.create_master_flat(np.zeros(10), header) is None

        # test bad dark shape
        dark = np.zeros(10)
        d, v = u.create_master_flat(data, header, dark=dark)
        capt = capsys.readouterr()
        assert 'not subtracting dark' in capt.err
        assert d is not None

    def test_test_hotpixels(self):
        data, header = fake_data((64, 64), 2)
        variance = np.full_like(data, 0.5)
        dark = np.full_like(data, 0.5)
        dark[32, 32] = 10
        d, v = u.create_master_flat(data, header, dark=dark, variance=variance)
        assert np.allclose(d[32, 32], 1.5)
        assert np.allclose(v[32, 32], 0.5)
        dark.fill(10)
        dark[32, 32] = 0
        variance[32, 32] = 0.1
        d, v = u.create_master_flat(data, header, dark=dark, variance=variance)
        assert np.allclose(d, 2)
        assert np.allclose(v, 0.1)

    def test_flatsum(self):
        data, header = fake_data()
        header_priority(header)

        assert u.flatsum(np.zeros(10), header) is None
        assert u.flatsum(np.zeros(10), None) is None
        extra = {}
        result = u.flatsum(data, header, flatvar=np.zeros(10),
                           extra=extra)

        assert isinstance(result[0], np.ndarray)
        assert result[1] is None
        assert len(extra.keys()) == 4
        assert extra['droopedvar'] is None
        assert isinstance(extra['drooped'], np.ndarray)
        data, header = fake_data()
        add_jailbars(data)
        data, _ = u.flatsum(data, header, jailbar=True)
        assert np.allclose(data, 0)
        data, header = fake_data()
        data[2:] += 2
        extra = {}
        result = u.flatsum(data, header, extra=extra, imglin=True)
        assert len(extra) == 6
        result = u.flatsum(data, header, normflat=True)
        assert np.allclose(result[0], 1)
        mask = np.full(data.shape[-2:], False)
        result = u.flatsum(data, header, normflat=True, ordermask=mask)
        assert "invalid median value" in str(header)

    def test_flatsum_fail(self, mocker):
        mocker.patch(
            'sofia_redux.instruments.forcast.flatsum.create_master_flat',
            return_value=None)
        data, header = fake_data()
        result = u.flatsum(data, header)
        assert result is None
        assert 'Could not create master flat' in str(header)
