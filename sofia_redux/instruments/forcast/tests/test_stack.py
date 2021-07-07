# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.tests.resources import add_jailbars

import sofia_redux.instruments.forcast.configuration as dripconfig
import sofia_redux.instruments.forcast.readmode
import sofia_redux.instruments.forcast.stack as u


def fake_data(shape=(4, 256, 256), value=2.0):
    header = fits.header.Header()
    data = np.full(shape, float(value))
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
    return data, header


def header_priority(header):
    dripconfig.load()
    for key in header:
        if key.lower() in dripconfig.configuration:
            del dripconfig.configuration[key.lower()]


class TestStack(object):

    def test_addhist(self):
        header = fits.header.Header()
        u.addhist(header, 'test history message')
        assert 'HISTORY' in header
        assert header['HISTORY'] == 'Stack: test history message'

    def test_add_stacks(self, capsys):
        data, header = fake_data(value=2.0)

        # check missing header
        assert u.add_stacks(data, None) is None
        capt = capsys.readouterr()
        assert 'invalid header' in capt.err

        header_priority(header)
        variance = np.full_like(data, 3.0)
        posdata, posvar = u.add_stacks(data, header, variance=variance)
        assert np.allclose(posdata, 4)
        assert np.allclose(posvar, 6)
        assert posdata.shape == (2, 256, 256)
        assert posvar.shape == posdata.shape

        posdata, posvar = u.add_stacks(data, header)
        assert posvar is None

        data, header = fake_data()
        del header['OTSTACKS']
        assert u.add_stacks(data, header) is None

        data, header = fake_data()
        del header['OTMODE']
        assert u.add_stacks(data, header) is None

        data, header = fake_data()
        assert u.add_stacks(data[0], header) is None

        posdata, posvar = u.add_stacks(
            data[:3], header, variance=variance)
        assert posvar is None
        assert posdata.shape == (1, 256, 256)
        assert np.allclose(posdata, 4)
        dripconfig.load()

    def test_background_scale(self):
        data, header = fake_data(value=2)
        header_priority(header)
        bglevel = u.background_scale(data, header)
        assert len(bglevel) == data.shape[0]
        assert np.allclose(bglevel, 2)
        mask = np.full(data.shape[-2:], False)
        assert u.background_scale(data, header, mask=mask) is None
        header['BGSCALE'] = False
        assert u.background_scale(data, header) is None

    def test_stack_c2nc2(self):
        data, header = fake_data(value=2.0)
        bgscale = [1, 2, 3, 4]  # scale = [1/2, 3/4] = [0.5, 0.75]
        variance = np.full_like(data, 3)
        extra = {}
        chop, var = u.stack_c2nc2(data, header, variance=variance,
                                  bglevel=bgscale, extra=extra)
        dexpect = (2 - 0.5 * 2) + (2 - 0.75 * 2)
        vexpect = (3 + 0.5 * 0.5 * 3) + (3 + 0.75 * 0.75 * 3)
        assert np.allclose(chop, dexpect)
        assert np.allclose(var, vexpect)
        assert chop.shape == (256, 256)
        assert chop.shape == var.shape
        assert extra['chopsub'].shape != data.shape
        assert len(extra['chopsub'].shape) == 3

        # check errors
        assert u.stack_c2nc2(data[0], None) is None
        _, var = u.stack_c2nc2(data, header, variance=variance[0],
                               bglevel=bgscale, extra=extra)
        assert var is None
        assert u.stack_c2nc2(data, header, bglevel=bgscale[:2]) is None

    def test_stack_map(self):
        d, v = 2.0, 3.0
        data, header = fake_data(shape=(8, 256, 256), value=d)
        bgscale = [1, 2, 3, 4, 1, 2, 3, 4]  # s1,s2,s3 = [1/2, 3/4, 1/3]
        variance = np.full_like(data, v)
        extra = {}
        s1 = bgscale[0] / bgscale[1]
        s2 = bgscale[2] / bgscale[3]
        s3 = bgscale[0] / bgscale[2]
        dexpect = (d - s1 * d) - ((d - s2 * d) * s3)
        dexpect *= 2  # 2 sets of 4
        vexpect = (v + s1 * s1 * v) + ((v + s2 * s2 * v) * s3 * s3)
        vexpect *= 2  # 2 sets of 4
        result, var = u.stack_map(data, header, variance=variance,
                                  bglevel=bgscale, extra=extra)
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)
        assert result.shape == (256, 256)
        assert result.shape == var.shape
        assert extra['chopsub'].shape != data.shape
        assert len(extra['chopsub'].shape) == 3
        assert extra['nodsub'].shape != data.shape
        assert len(extra['nodsub'].shape) == 3

        # check clipping
        data, header = fake_data(shape=(10, 256, 256), value=d)
        variance = np.full_like(data, v)
        bgscale = bgscale + [1, 2]
        result, var = u.stack_map(data, header, variance=variance,
                                  bglevel=bgscale, extra=extra)
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)

        # check NODBEAM
        header['NODBEAM'] = 'B'
        result, var = u.stack_map(data, header, variance=variance,
                                  bglevel=bgscale, extra=extra)
        assert np.allclose(result, -dexpect)

        # check errors
        assert u.stack_map(data[0], None) is None
        _, var = u.stack_map(data, header, variance=variance[0],
                             bglevel=bgscale, extra=extra)
        assert var is None
        assert u.stack_map(data, header, bglevel=bgscale[:2]) is None

    def test_stack_c3d(self):
        data, header = fake_data(shape=(3, 256, 256), value=1)
        variance = np.full_like(data, 1.0)
        extra = {}
        dexpect = -1  # 1 - 1 - 1
        vexpect = 3  # 1 + 1 + 1
        result, var = u.stack_c3d(
            data, header, variance=variance, extra=extra)
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)
        assert result.shape == (256, 256)
        assert result.shape == var.shape
        assert np.allclose(result, extra['chopsub'])

        # check errors
        assert u.stack_c3d(data[:1], None) is None
        _, var = u.stack_c3d(
            data, header, variance=variance[0], extra=extra)
        assert var is None

    def test_stack_cm(self):
        data, header = fake_data(shape=(5, 256, 256), value=1)
        variance = np.full_like(data, 1.0)
        extra = {}
        dexpect = -3  # 1 - 1 - 1 - 1 - 1
        vexpect = 5  # 1 + 1 + 1
        header['CHPNPOS'] = 5
        header_priority(header)
        result, var = u.stack_cm(
            data, header, variance=variance, extra=extra)
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)
        assert result.shape == (256, 256)
        assert result.shape == var.shape
        assert np.allclose(result, extra['chopsub'])

        # check errors
        assert u.stack_cm(10, None) is None
        _, var = u.stack_cm(
            data, header, variance=variance[0], extra=extra)
        assert var is None

        # Check clipping
        header['CHPNPOS'] = 3
        result, var = u.stack_cm(
            data, header, variance=variance, extra=extra)
        dexpect = -1  # 1 - 1 -1
        vexpect = 3  # 1 + 1 + 1
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)

    def test_stack_stare(self):
        data, header = fake_data(shape=(5, 256, 256), value=2.0)
        header_priority(header)
        variance = np.full_like(data, 1 / np.pi)
        dexpect = 2  # choptsaconv effect
        vexpect = 5 / (2 * (5**2))
        result, var = u.stack_stare(data, header, variance=variance)
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)
        assert result.shape == (256, 256)
        assert result.shape == var.shape
        # check errors
        assert u.stack_stare(data[0], None) is None
        _, var = u.stack_stare(data, header, variance=variance[0])
        assert var is None
        # Check NaN handling
        variance[0] = np.nan
        vexpect = 4 / (2 * (4**2))
        result, var = u.stack_stare(data, header, variance=variance)
        assert np.allclose(result, dexpect)
        assert np.allclose(var, vexpect)

    def test_convert_to_electrons(self):
        data, header = fake_data(shape=(5, 256, 256), value=2.0)
        header_priority(header)
        variance = np.full_like(data, 0.1)
        result, var = u.convert_to_electrons(data, header, variance=variance)
        scale = header['EPERADU'] * header['FRMRATE'] * 1e-6
        dexpect = 2.0 * scale
        vexpect = 0.1 * scale ** 2
        assert np.allclose(data, dexpect)
        assert np.allclose(var, vexpect)
        del header['FRMRATE']
        assert u.convert_to_electrons(data, header) is None

    def test_subtract_background(self):
        data, header = fake_data(shape=(256, 256), value=2.0)
        header_priority(header)

        result = u.subtract_background(data, header=header)
        assert result.shape == data.shape
        assert np.allclose(result, 0)

        stat = 'mode'
        data, header = fake_data(shape=(3, 256, 256), value=2.0)
        data[1:] += 1
        result = u.subtract_background(data, header=header, stat=stat)
        assert result.shape == data.shape
        assert np.allclose(result, 0)

        # invalid statistics fall back to 'mode'
        stat = 'invalid'
        result2 = u.subtract_background(data, header=header, stat=stat)
        assert np.allclose(result, result2)

        stat = 'median'
        result = u.subtract_background(data, header=header, stat=stat)
        assert result.shape == data.shape
        assert np.allclose(result, 0)

        assert u.subtract_background(np.zeros(10), stat=stat) is None
        header['BGSUB'] = False
        assert u.subtract_background(data) is None
        data, header = fake_data(shape=(256, 256), value=2.0)
        mask = np.full(data.shape[-2:], False)
        assert u.subtract_background(data, mask=mask) is None

    def test_jailbars(self):
        testval = 2.0
        data, header = fake_data(shape=(4, 256, 256), value=testval)
        header['BGSUB'] = False
        header['OTSTACKS'] = 1
        header_priority(header)
        add_jailbars(data)
        header_priority(header)
        result = u.stack(data, header)
        assert np.allclose(result[0], testval)
        # Check no JBCLEAN if FFT
        header['JBCLEAN'] = 'FFT'
        result = u.stack(data, header)
        assert not np.allclose(result[0], testval)
        assert np.allclose(data[0], result[0])

    def test_background_mask(self):
        data, header = fake_data()
        header['OTSTACKS'] = 1
        header['BGSUB'] = 1
        mask = np.full(data.shape[-2:], True)
        header_priority(header)
        d, v = u.stack(data, header, mask=mask)
        assert np.allclose(d, 0)
        mask.fill(False)
        d, _ = u.stack(data, header, mask=mask)
        assert not np.allclose(d, 0)

    def test_stack_success(self):
        data, header = fake_data(shape=(4, 256, 256))
        header['INSTMODE'] = 'MAP'
        header['OTSTACKS'] = 1
        header_priority(header)
        extra = {}
        variance = data.copy()

        d, v = u.stack(data, header, extra=extra,
                       variance=variance, stat='median')
        assert d.shape == v.shape
        assert extra['posdata'].shape[0] == data.shape[0]
        assert extra['chopsub'].shape[0] == data.shape[0] / 2
        assert extra['nodsub'].shape[0] == data.shape[0] / 4
        assert header['PRODTYPE'] == 'STACKED'

        # single plane
        dripconfig.load()
        dripconfig.configuration['bgsub'] = False
        dripconfig.configuration['bgscale'] = False
        dripconfig.configuration['jbclean'] = False
        data, header = fake_data(shape=(256, 256))
        header['OTSTACKS'] = 1
        variance = data.copy()
        d, v = u.stack(data, header, extra=extra,
                       variance=variance, stat='median')
        assert np.allclose(d, data, equal_nan=True)
        assert np.allclose(v, variance, equal_nan=True)

    def test_stack_errors(self, capsys, mocker):
        data, header = fake_data()
        assert u.stack(data, None) is None
        assert u.stack(np.zeros(10), header) is None
        header['INSTMODE'] = 'FOO'
        header_priority(header)
        assert u.stack(data, header) is None

        # bad variance
        data, header = fake_data()
        d, v = u.stack(data, header, variance=10)
        capt = capsys.readouterr()
        assert 'invalid variance' in capt.err
        assert d is not None
        assert v is None

        # bad mask
        d, _ = u.stack(data, header, mask=10)
        capt = capsys.readouterr()
        assert 'mask invalid' in capt.err
        assert d is not None

        # jbclean failure
        mocker.patch('sofia_redux.instruments.forcast.stack.jbclean',
                     return_value=None)
        header['JBCLEAN'] = 'MEDIAN'
        assert u.stack(data, header) is not None
        capt = capsys.readouterr()
        assert 'Jailbars not removed' in capt.err

        # conversion failure
        mocker.patch(
            'sofia_redux.instruments.forcast.stack.convert_to_electrons',
            return_value=None)
        header['JBCLEAN'] = 'NONE'
        assert u.stack(data, header) is not None
        capt = capsys.readouterr()
        assert 'Could not convert' in capt.err

        # add_stacks failure
        mocker.patch('sofia_redux.instruments.forcast.stack.add_stacks',
                     return_value=None)

        assert u.stack(data, header) is None
        capt = capsys.readouterr()
        assert 'stack failed' in capt.err

    @pytest.mark.parametrize('skymode',
                             ['C2', 'C2NC2', 'NAS', 'NOS', 'C2NC4',
                              'NXCAC', 'NMC', 'NPC', 'NPCCAS', 'NPCNAS',
                              'C2ND', 'SLITSCAN', 'MAP', 'C3D',
                              'CM', 'STARE'])
    def test_stack_modes(self, skymode, capsys, mocker):
        c2_modes = ['C2', 'C2NC2']
        map_modes = ['NAS', 'NOS', 'C2NC4', 'NXCAC', 'C2N', 'NMC',
                     'NPC', 'NPCCAS', 'NPCNAS', 'C2ND', 'SLITSCAN',
                     'MAP']

        # fake data
        data, header = fake_data()
        header['INSTMODE'] = skymode
        header['SKYMODE'] = skymode
        assert sofia_redux.instruments.forcast.readmode.readmode(header) \
            == skymode

        # mock all the stack modes to verify which is called
        def mock_c2nc2(*args, **kwargs):
            print('stack C2NC2')

        def mock_map(*args, **kwargs):
            print('stack MAP')

        def mock_c3d(*args, **kwargs):
            print('stack C3D')

        def mock_cm(*args, **kwargs):
            print('stack CM')

        def mock_stare(*args, **kwargs):
            print('stack STARE')

        mocker.patch('sofia_redux.instruments.forcast.stack.stack_c2nc2',
                     mock_c2nc2)
        mocker.patch('sofia_redux.instruments.forcast.stack.stack_map',
                     mock_map)
        mocker.patch('sofia_redux.instruments.forcast.stack.stack_c3d',
                     mock_c3d)
        mocker.patch('sofia_redux.instruments.forcast.stack.stack_cm',
                     mock_cm)
        mocker.patch('sofia_redux.instruments.forcast.stack.stack_stare',
                     mock_stare)

        u.stack(data, header)
        capt = capsys.readouterr()
        if skymode in c2_modes:
            assert 'stack C2NC2' in capt.out
        elif skymode in map_modes:
            assert 'stack MAP' in capt.out
        else:
            assert 'stack {}'.format(skymode) in capt.out

    def test_tsaconv(self, capsys):
        dripconfig.load()
        try:
            del dripconfig.configuration['choptsac']
            del dripconfig.configuration['bgsub']
            del dripconfig.configuration['bgscale']
        except KeyError:
            pass

        dval = 10.0
        data, header = fake_data(shape=(4, 256, 256), value=dval)
        header['INSTMODE'] = 'C2N'
        header['SKYMODE'] = 'NMC'
        header['OTSTACKS'] = 1
        del header['CHOPTSAC']
        del header['BGSUB']
        del header['BGSCALE']
        data[0] *= 2.0
        data[3] *= 2.0
        # expected value, on - off * 2
        exval = dval * 2

        # chop tsa None or non-int - uses -1 (ie. no sign switch)
        d, _ = u.stack(data, header)
        assert np.allclose(d, exval)
        capt = capsys.readouterr()
        assert 'chop tsa convention was not found' in capt.out
        header['CHOPTSAC'] = 'BADVAL'
        d, _ = u.stack(data, header)
        assert np.allclose(d, exval)
        capt = capsys.readouterr()
        assert 'chop tsa convention was not found' in capt.out

        # chop tsa 1 (sign switch)
        header['CHOPTSAC'] = 1
        d, _ = u.stack(data, header)
        assert np.allclose(d, -1 * exval)
        capt = capsys.readouterr()
        assert 'chop tsa convention is 1' in capt.out

        # chop tsa -1 (no sign switch)
        header['CHOPTSAC'] = -1
        d, _ = u.stack(data, header)
        assert np.allclose(d, exval)
        capt = capsys.readouterr()
        assert 'chop tsa convention is -1' in capt.out

        # unexpected value (no sign switch)
        header['CHOPTSAC'] = 2
        d, _ = u.stack(data, header)
        assert np.allclose(d, exval)
        capt = capsys.readouterr()
        assert 'chop tsa convention (2) is different from -1, 1' in capt.out
