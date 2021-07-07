# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.image.adjust import shift, rotate

from sofia_redux.instruments.forcast.register \
    import (addhist, coadd_centroid, coadd_correlation,
            coadd_header, coadd_user, register)
import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.tests.resources \
    import nmc_testdata, npc_testdata


class TestCoaddCentroid(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Register: test history message'

    def test_coadd_centroid(self):
        test = nmc_testdata()
        data = test['data']
        data[data < 0] = 0
        header = test['header']
        variance = np.full_like(data, 2.0)
        offset = 4.5, 5.75
        crpix = [0, 0]
        reference = shift(data, offset, order=3)
        shifted, variance = coadd_centroid(
            data, reference, variance=variance, crpix=crpix, header=header)
        for pix, off in zip(crpix, np.flip(offset)):
            assert np.allclose(pix, off, atol=0.01)
        assert np.allclose(np.nanmax(abs(reference - shifted)), 0, atol=0.01)
        assert np.isnan(variance).any()
        assert not np.isnan(variance).all()
        assert header['COADX0'] == crpix[0]
        assert header['COADY0'] == crpix[1]

    def test_errors(self, mocker, capsys):
        test = nmc_testdata()
        data = test['data']
        data[data < 0] = 0
        offset = 4.5, 5.75
        reference = shift(data, offset, order=3)
        bad = np.array([0])
        assert coadd_centroid(np.array([0]), reference) is None
        assert coadd_centroid(data, bad) is None
        _, testvar = coadd_centroid(data, reference, variance=bad)
        assert testvar is None
        assert coadd_centroid(data, reference,
                              border=data.shape[0]) is None

        # mock peakfind failure
        mocker.patch('sofia_redux.instruments.forcast.register.peakfind',
                     return_value=[])
        assert coadd_centroid(data, reference) is None
        capt = capsys.readouterr()
        assert 'peakfind failed' in capt.err

    def test_rotation(self):
        test = nmc_testdata()
        data = test['data']
        data[data < 0] = 0
        variance = np.full_like(data, 2.0)
        offset = 4.5, 5.75
        rotation_order = 1
        angle = 20
        crpix = [0, 0]
        reference = rotate(data, angle, order=rotation_order)
        reference = shift(reference, offset, order=3)
        shifted, var = coadd_centroid(
            data, reference, variance=variance, rot_angle=angle,
            rotation_order=rotation_order, crpix=crpix)
        for pix, off in zip(crpix, np.flip(offset)):
            assert np.allclose(pix, off, atol=0.01)

        # test that NaNs are in the same place in data and variance,
        # to within 1% of pixels --
        # they may differ slightly near the edges
        diff = np.sum(np.logical_xor(np.isnan(shifted), np.isnan(var)))
        assert diff / shifted.size < 0.05

    def test_get_offsets(self):
        test = nmc_testdata()
        data = test['data']
        result = coadd_centroid(data, shift(data, [5, 5]), get_offsets=True)
        assert np.allclose(result, 5)

        # test border -- should be same
        result2 = coadd_centroid(data, shift(data, [5, 5]),
                                 get_offsets=True, border=10)
        assert np.allclose(result2, result)


class TestCoaddCorrelation(object):

    def test_addhist(self):
        header = fits.header.Header()
        addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Register: test history message'

    def test_coadd_correlation(self):
        test = npc_testdata()
        data = test['data']
        header = test['header']
        variance = np.full_like(data, 2.0)
        offset = 4.5, 5.75
        crpix = [0, 0]
        reference = shift(data, offset, order=3)
        shifted, variance = coadd_correlation(
            data, reference, variance=variance,
            crpix=crpix, header=header)
        for pix, off in zip(crpix, np.flip(offset)):
            assert np.allclose(pix, off, atol=0.01)
        assert np.allclose(np.nanmax(abs(reference - shifted)),
                           0, atol=0.01)
        assert np.isnan(variance).any()
        assert not np.isnan(variance).all()
        assert header['COADX0'] == crpix[0]
        assert header['COADY0'] == crpix[1]
        assert 'XYSHIFT' in header

    def test_errors(self):
        test = npc_testdata()
        data = test['data']
        data[data < 0] = 0
        header = test['header']
        offset = 4.5, 5.75
        reference = shift(data, offset, order=3)
        assert coadd_correlation(np.array([0]), reference) is None
        assert coadd_correlation(data, np.array([0])) is None
        testvar = np.array([0.])
        _, testvar = coadd_correlation(data, reference, variance=testvar)
        assert testvar is None
        assert coadd_correlation(data, reference,
                                 border=data.shape[0]) is None
        dripconfig.load()
        header['XYSHIFT'] = 2
        dripconfig.configuration['xyshift'] = 2
        result = coadd_correlation(data, reference, header=header)
        dripconfig.load()
        assert result is None

    def test_rotation(self):
        test = npc_testdata()
        data = test['data']
        variance = np.full_like(data, 2.0)
        offset = 4.5, 5.75
        rotation_order = 1
        angle = 20
        crpix = [0, 0]
        reference = rotate(data, angle, order=rotation_order)
        reference = shift(reference, offset, order=3)
        shifted, variance = coadd_correlation(
            data, reference, variance=variance, rot_angle=angle,
            rotation_order=rotation_order, crpix=crpix)
        for pix, off in zip(crpix, np.flip(offset)):
            assert np.allclose(pix, off, atol=0.01)

        # test that NaNs are in the same place in data and variance,
        # to within 1% of pixels --
        # they may differ slightly near the edges
        diff = np.sum(np.logical_xor(np.isnan(shifted), np.isnan(variance)))
        assert diff / shifted.size < 0.05

    def test_upsample(self):
        test = npc_testdata()
        data = test['data']
        offset = 4.5, 5.75
        reference = shift(data, offset, order=3)
        crpix = [0, 0]
        coadd_correlation(data, reference, crpix=crpix, upsample=1)
        for val in crpix:
            assert int(val) == val

    def test_get_offsets(self):
        test = npc_testdata()
        data = test['data']
        result = coadd_correlation(data, shift(data, [5, 5]), get_offsets=True)
        assert np.allclose(result, 5)

        # test border -- should be same
        result2 = coadd_correlation(data, shift(data, [5, 5]),
                                    get_offsets=True, border=10)
        assert np.allclose(result2, result)


class TestCoaddHeader(object):

    def test_errors(self, mocker, capsys):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        variance = data * 2
        assert coadd_header(data, None) is None
        assert coadd_header(data[0], header) is None
        d, v = coadd_header(data, header, variance=variance[0])
        assert v is None
        assert isinstance(d, np.ndarray)

        # mock imgshift failure
        mocker.patch(
            'sofia_redux.instruments.forcast.register.imgshift_header',
            return_value={})
        result = coadd_header(data, header)
        assert result is None
        capt = capsys.readouterr()
        assert 'invalid header shift' in capt.err

    def test_success(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        variance = data * 2
        crpix = [0, 0]
        d, v = coadd_header(data, header, variance=variance, crpix=crpix)
        assert d.shape == v.shape
        assert not np.allclose(d, v)
        assert not np.allclose(d, data)
        assert not np.allclose(v, variance)
        assert 'COADX0' in header
        assert 'COADY0' in header
        assert not (crpix[0] == 0 and crpix[1] == 0)

    def test_offsets(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        offsets = coadd_header(data, header, get_offsets=True)
        assert len(offsets) == 2
        assert not np.allclose(offsets, 0)
        assert 'COADX0' not in header
        assert 'COADY0' not in header


class TestCoaddUser(object):

    def test_errors(self):
        test = npc_testdata()
        data = test['data'].copy()
        variance = data * 2
        reference = [2, 2]
        position = [1, 1.0]
        assert coadd_user(data[0], reference, position) is None
        assert coadd_user(data, reference[0], position) is None
        assert coadd_user(data, reference, position[0]) is None
        d, v = coadd_user(data, reference, position, variance=variance[0])
        assert v is None
        assert isinstance(d, np.ndarray)

        # too large shift - returns 0
        reference = [1e5, 1e5]
        offsets = coadd_user(data, reference, position, get_offsets=True)
        assert np.all(offsets == 0)

    def test_success(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        variance = data * 2
        crpix = [0, 0]
        reference = [2, 2]
        position = [1, 1.0]
        d, v = coadd_user(data, reference, position, header=header,
                          variance=variance, crpix=crpix)
        assert d.shape == v.shape
        assert not np.allclose(d, v)
        assert not np.allclose(d, data)
        assert not np.allclose(v, variance)
        assert 'COADX0' in header
        assert 'COADY0' in header
        assert not (crpix[0] == 0 and crpix[1] == 0)

    def test_offsets(self):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        reference = [2, 2]
        position = [1, 1.0]
        offsets = coadd_user(data, reference, position,
                             header=header, get_offsets=True)
        assert len(offsets) == 2
        assert np.allclose(offsets, 1)
        assert 'COADX0' not in header
        assert 'COADY0' not in header


class TestRegister(object):

    def test_errors(self, mocker, capsys):
        test = npc_testdata()
        data = test['data'].copy()
        header = test['header'].copy()
        variance = data * 2

        # none header is okay, other type is not
        assert register(data, None) is not None
        assert register(data, 0) is None

        # bad data, variance
        assert register(data[0], header) is None
        d, v = register(data, header, variance=variance[0])
        assert isinstance(d, np.ndarray)
        assert v is None

    def test_algorithms(self):
        dripconfig.load()
        if 'corcoadd' in dripconfig.configuration:
            del dripconfig.configuration['corcoadd']
        test = npc_testdata()
        data = test['data'].copy()
        variance = data * 2
        header = test['header'].copy()
        if 'CORCOADD' in header:
            del header['CORCOADD']
        dh, vh = register(data, header, missing=0, variance=variance)
        assert not np.allclose(dh, data)
        assert not np.allclose(vh, variance)
        assert "Used header registration" in str(header)

        for missing_ref in ['CORCOADD', 'CENTROID']:
            header = test['header'].copy()
            header['CORCOADD'] = missing_ref
            d, v = register(data, header, variance=variance, missing=0)
            assert "Used header registration" in str(header)
            assert np.allclose(d, dh)
            assert np.allclose(v, vh)

        header = test['header'].copy()
        header['CORCOADD'] = 'NOSHIFT'
        d, v = register(data, header, variance=variance, missing=0)
        assert "No shift applied" in str(header)
        assert np.allclose(d, data)
        assert np.allclose(v, variance)

        header = test['header'].copy()
        header['CORCOADD'] = 'CENTROID'
        ref = shift(data, (5, 5))
        register(data, header, variance=variance, reference=ref)
        assert "Used centroid registration" in str(header)

        header = test['header'].copy()
        header['CORCOADD'] = 'XCOR'
        ref = shift(data, (5, 5))
        register(data, header, variance=variance, reference=ref)
        assert "Used correlation registration" in str(header)

        register(data, header, reference=[0, 0], position=[2, 2])
        assert header['COADX0'] == -2
        assert header['COADY0'] == -2

        dripconfig.load()

    def test_get_offsets(self):
        dripconfig.load()
        if 'corcoadd' in dripconfig.configuration:
            del dripconfig.configuration['corcoadd']
        test = npc_testdata()
        data = test['data'].copy()
        variance = data * 2
        header = test['header'].copy()

        header['CORCOADD'] = 'XCOR'
        ref = shift(data, (5, 5))
        offset = register(data, header, variance=variance,
                          reference=ref, get_offsets=True)
        assert np.allclose(offset, 5)
        assert len(offset) == 2

        header['CORCOADD'] = 'XCOR'
        offset = register(data, header, variance=variance,
                          reference=ref, get_offsets=True)
        assert np.allclose(offset, 5)
        assert len(offset) == 2

        headershifts = imgshift_header(header, nod=False, chop=False)
        dx = headershifts['ditherx']
        dy = headershifts['dithery']
        header['CORCOADD'] = 'HEADER'
        offset = register(data, header, variance=variance,
                          reference=ref, get_offsets=True)
        assert offset[0] == dx
        assert offset[1] == dy
        offset = register(data, header, reference=[0, 0],
                          position=[2, 2], get_offsets=True)
        assert np.allclose(offset, -2)
        assert len(offset) == 2

        header['CORCOADD'] = 'NOSHIFT'
        offset = register(data, header, reference=[0, 0],
                          get_offsets=True)
        assert np.allclose(offset, 0)
        assert len(offset) == 2

        # unknown algorithm
        header['CORCOADD'] = 'BADVAL'
        offset = register(data, header, reference=[0, 0],
                          get_offsets=True)
        assert offset is None
        assert 'Coadd registration failed' in str(header)
