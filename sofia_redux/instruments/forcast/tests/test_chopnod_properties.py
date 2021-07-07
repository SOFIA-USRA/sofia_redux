# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.chopnod_properties \
    import chopnod_properties
from sofia_redux.instruments.forcast.tests.resources \
    import nmc_testdata, npc_testdata, raw_testdata


class TestChopnodProperties(object):

    def test_defaults(self):
        header = fits.header.Header()
        result = chopnod_properties(header, update_header=True)
        assert isinstance(result, dict)
        assert len(result) > 0
        for k in ['chop', 'nod']:
            assert (result[k]['dxdy'] == 0).all()
            assert result[k]['angle'] == 0
            assert result[k]['coordsys'] == 'SIRF'
        assert 'HISTORY' in header
        assert result['nmc']
        assert not result['c2nc2']

    def test_chopnod(self):
        header = nmc_testdata()['header']
        result = chopnod_properties(header)
        assert result['nmc']
        assert result['chop']['distance'] == result['nod']['distance']
        assert result['chop']['distance'] != 0
        dang = result['chop']['angle'] - result['nod']['angle']
        assert result['chop']['angle'] != 0
        assert int(abs(dang / np.pi)) == abs(dang / np.pi)

        header = npc_testdata()['header']
        result = chopnod_properties(header)
        assert not result['nmc']
        assert result['chop']['distance'] == result['nod']['distance']
        assert result['chop']['distance'] != 0
        dang = result['chop']['angle'] - result['nod']['angle']
        assert result['chop']['angle'] != 0
        assert int(abs(dang / np.pi)) != abs(dang / np.pi)

    def test_erf(self):
        header = nmc_testdata()['header']
        sirf = chopnod_properties(header)
        scdx, scdy = sirf['chop']['dxdy']
        sndx, sndy = sirf['nod']['dxdy']

        header['CHPCRSYS'] = 'ERF'
        header['NODCRSYS'] = 'ERF'
        result = chopnod_properties(header)
        assert len(result) > 0
        assert result['chop']['coordsys'] == 'ERF'
        assert result['nod']['coordsys'] == 'ERF'

        # check that dx, dy are rotated by sky angle,
        # compared to sirf version
        sky_angle = np.radians(header['SKY_ANGL'])
        cosa = np.cos(sky_angle)
        sina = np.sin(sky_angle)
        assert result['chop']['dxdy'][0] == scdx * cosa + scdy * sina
        assert result['chop']['dxdy'][1] == scdy * cosa - scdx * sina
        assert result['nod']['dxdy'][0] == sndx * cosa + sndy * sina
        assert result['nod']['dxdy'][1] == sndy * cosa - sndx * sina

    def test_c2nc2(self):
        # single plane test data
        header = nmc_testdata()['header']
        header['SKYMODE'] = 'C2NC2'
        result = chopnod_properties(header)
        assert not result['nmc']
        assert result['c2nc2'] == 1

        # 4 plane test data
        header = raw_testdata()[0].header
        header['SKYMODE'] = 'C2NC2'
        result = chopnod_properties(header)
        assert not result['nmc']
        assert result['c2nc2'] == 2
