# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np
import pytest

import sofia_redux.instruments.forcast.configuration as dripconfig
from sofia_redux.instruments.forcast import distcorr_model
from sofia_redux.instruments.forcast.distcorr_model import pinhole_defaults


def pinhole_test_pos(nximg=256, nyimg=256,
                     nxpts=5, nypts=5, angle=1.0,
                     spx=None, spy=None):
    theta = np.deg2rad(angle)
    yid, xid = np.mgrid[:nypts, :nxpts].astype(int)
    ypos, xpos = np.mgrid[:nypts, :nxpts].astype(float)
    ypos -= nypts / 2
    xpos -= nxpts / 2
    xp = (xpos * np.cos(theta)) + (ypos * np.sin(theta))
    yp = (xpos * -np.sin(theta)) + (ypos * np.cos(theta))
    xpos = xp + nxpts / 2
    ypos = yp + nypts / 2

    xpos = (xpos - xpos.min()) * (nximg - 1) / np.ptp(xpos)
    ypos = (ypos - ypos.min()) * (nyimg - 1) / np.ptp(ypos)

    if spx is not None:
        for i, j in zip(spx, spy):
            xpos[j, i] = np.nan
            ypos[j, i] = np.nan

    return xpos, ypos, xid, yid


class TestDistcorrModel(object):
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def test_defaults(self):
        defaults = pinhole_defaults()
        print(defaults)
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        for value in defaults.values():
            assert value is not None
        assert isinstance(defaults['order'], int)
        # may be undefined or not, depending on config
        assert isinstance(defaults['fpinhole'], str)

        try:
            del dripconfig.configuration['fpinhole']
        except KeyError:
            pass
        defaults2 = pinhole_defaults()
        fname2 = defaults2['fpinhole']
        assert os.path.basename(fname2) == 'undefined'
        dripconfig.load()

    def test_read_pinhole_file(self, tmpdir):
        func = distcorr_model.read_pinhole_file
        testfile = tmpdir.join('testfile')

        # non-existent file returns None
        assert func(pinhole_file=testfile) is None

        # write a file
        contents = ['xid\tyid\txpos\typos\n']
        for i in range(5):
            contents.append('0\t{}\t1\t2\n'.format(i))
        testfile.write(''.join(contents))

        table = func(str(testfile))
        columns = ['xid', 'yid', 'xpos', 'ypos']
        for col in columns:
            assert col in table.columns
        assert len(table) == 5
        assert (table.xpos.values == 1).all()
        assert (table.ypos.values == 2).all()

    def test_default_pinhole_model(self):
        # test default pinhole model
        dripconfig.configuration['fpinhole'] = 'pinhole_locs.txt'
        func = distcorr_model.pinhole_model
        kwargs = pinhole_defaults()
        del kwargs['order']
        table = distcorr_model.read_pinhole_file(kwargs.pop('fpinhole'))
        result = func(table.xpos.values, table.ypos.values,
                      table.xid.values, table.yid.values)
        assert isinstance(result, dict)
        for key in ['avgdx', 'avgdy', 'angle']:
            assert key in result
            assert isinstance(result[key], float)
        for key in ['xmodel', 'ymodel']:
            assert key in result
            assert isinstance(result[key], np.ndarray)

    def test_bad_arguments(self, capsys):
        func = distcorr_model.pinhole_model
        nx = 5
        testx, testy, xid, yid = pinhole_test_pos(
            nxpts=nx, nypts=nx,
            spx=[nx // 2], spy=[nx // 2],
            nximg=nx, nyimg=nx)

        model_ok = func(testx, testy, xid, yid)
        assert model_ok is not None

        badx = testx.copy()
        badx[2] = -1
        bad_positions_x = func(badx, testy, xid, yid)
        assert bad_positions_x is None
        capt = capsys.readouterr()
        assert 'X positions' in capt.err

        bady = testy.copy()
        bady[2] = -1
        bad_positions_y = func(testx, bady, xid, yid)
        assert bad_positions_y is None
        capt = capsys.readouterr()
        assert 'Y positions' in capt.err

    def test_model_value(self, capsys):
        func = distcorr_model.pinhole_model
        params = {
            'nxpts': 5, 'nypts': 5,
            'spx': [2], 'spy': [2],
            'nximg': 5, 'nyimg': 5,
            'angle': 10
        }
        pixfrac = 0.001
        testx, testy, xid, yid = pinhole_test_pos(**params)

        result = func(testx, testy, xid, yid)
        assert abs((testx[~np.isnan(testx)]
                    - result['xmodel']) < pixfrac).all()
        assert abs((testy[~np.isnan(testy)]
                    - result['ymodel']) < pixfrac).all()

        # test for small angle: won't be corrected; should still be close
        params['angle'] = 0.5
        pixfrac = 0.1
        testx, testy, xid, yid = pinhole_test_pos(**params)
        result = func(testx, testy, xid, yid)
        assert abs((testx[~np.isnan(testx)]
                    - result['xmodel']) < pixfrac).all()
        assert abs((testy[~np.isnan(testy)]
                    - result['ymodel']) < pixfrac).all()
        capt = capsys.readouterr()
        assert 'ignoring angle correction' in capt.out

    def test_view_model(self, tmpdir, mocker):
        func = distcorr_model.view_model
        nx = 256
        ny = 256
        x, y, xid, yid = pinhole_test_pos(nximg=nx, nyimg=ny)
        args = x[~np.isnan(x)], y[~np.isnan(y)]

        tempfile = str(tmpdir.join('test_model.fits'))

        # check writing to file and force option
        func(*args, write_file=tempfile, fwhm=1, amplitude=1)
        assert os.path.isfile(tempfile)

        data1 = fits.getdata(tempfile).copy()
        t1 = os.stat(tempfile).st_mtime
        func(*args, write_file=tempfile)
        t2 = os.stat(tempfile).st_mtime
        assert t1 == t2

        func(*args, write_file=tempfile, force=True,
             fwhm=2, amplitude=1)
        t3 = os.stat(tempfile).st_mtime
        data3 = fits.getdata(tempfile).copy()
        assert t3 != t1
        assert data1.sum() < data3.sum()
        assert np.allclose(data1.max(), data3.max(), atol=0.01)
        func(*args, write_file=tempfile, force=True,
             fwhm=1, amplitude=2)
        data4 = fits.getdata(tempfile).copy()
        assert not np.allclose(data1.max(), data4.max(), atol=0.01)

    def test_distcorr_model_function(self):
        func = distcorr_model.distcorr_model
        header = fits.header.Header()
        dripconfig.configuration['fpinhole'] = 'pinhole_locs.txt'

        results = func(basehead=header)
        assert isinstance(results, dict)
        assert 'PIN_MOD' in header
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

    def test_distcorr_args(self, capsys, mocker, tmpdir):
        func = distcorr_model.distcorr_model
        dripconfig.configuration['fpinhole'] = 'pinhole_locs.txt'

        # test pandas pinhole
        pfile = pinhole_defaults()['fpinhole']
        pinhole_pd = distcorr_model.read_pinhole_file(pfile)
        results = func(pinhole=pinhole_pd)
        assert results is not None

        # test bad pinhole file
        results = func(pinhole='badval')
        assert results is None
        capt = capsys.readouterr()
        assert 'invalid pinhole data' in capt.err.lower()

        # write to file
        fname = str(tmpdir.join('test_model.fits'))
        results = func(viewpin=fname)
        assert results is not None
        assert os.path.isfile(fname)
