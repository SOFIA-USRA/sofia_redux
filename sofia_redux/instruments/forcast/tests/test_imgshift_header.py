# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.tests.resources import npc_testdata


class TestImgshiftHeader(object):

    def test_error(self):
        shift = imgshift_header(None)
        assert isinstance(shift, dict)
        assert len(shift) > 0
        for key, val in shift.items():
            if 'coord' in key:
                assert val == ''
            else:
                assert val == 0

    def test_switches(self):
        header = npc_testdata()['header']
        shift = imgshift_header(header, chop=False, nod=False, dither=False)
        for k, v in shift.items():
            if 'coord' in k:
                assert v == ''
            elif k != 'sky_angle':
                assert np.allclose(v, 0)
        shift = imgshift_header(header, chop=True, nod=True, dither=True)
        for k, v in shift.items():
            if 'coord' in k:
                assert v != ''
            else:
                assert v != 0

    def test_plate_scale(self):
        header = npc_testdata()['header']
        header['PLTSCALE'] = 0.768 / 3600
        shift = imgshift_header(header)
        cx1 = shift['chopx']
        del header['PLTSCALE']
        header['TELESCOP'] = 'SOFIA'
        shift = imgshift_header(header)
        cx2 = shift['chopx']
        assert cx2 == cx1
        header['TELESCOP'] = 'PIXELS'
        shift = imgshift_header(header)
        cx3 = shift['chopx']
        assert cx3 / 0.768 == cx1

    def test_sky_angle(self):
        header = npc_testdata()['header']
        header['ANGLCONV'] = 'POSITIVE'
        shift_pos = imgshift_header(header, dripconf=False)
        header['ANGLCONV'] = 'NEGATIVE'
        shift_neg = imgshift_header(header, dripconf=False)
        assert shift_neg['chopx'] == -shift_pos['chopx']
        header['ANGLCONV'] = 'FOO'
        shift_foo = imgshift_header(header, dripconf=False)
        assert shift_neg['chopx'] == shift_foo['chopx']

    def test_distances(self):
        header = npc_testdata()['header']
        shift = imgshift_header(header)
        header['NODAMP'] *= 4
        header['CHPAMP1'] *= 4
        header['DITHERX'] *= 4
        header['DITHERY'] *= 4
        shift4 = imgshift_header(header)
        for k in ['chop', 'nod', 'dither']:
            for c in ['x', 'y']:
                assert shift[k + c] != 0
                assert shift4[k + c] == shift[k + c] * 4

    def test_angles(self):
        header = npc_testdata()['header']
        shift = imgshift_header(header)
        header['CHPANGLE'] += 180
        header['NODANGLE'] += 180
        shift_rot = imgshift_header(header)
        for key in ['chopx', 'chopy', 'nodx', 'nody']:
            assert np.allclose(shift[key], -shift_rot[key], atol=1e-5)
        header['SKY_ANGL'] += 180
        shift_rot_dith = imgshift_header(header)
        assert np.allclose(shift['ditherx'], -shift_rot_dith['ditherx'])
        assert np.allclose(shift['dithery'], -shift_rot_dith['dithery'])

    def test_erf(self):
        header = npc_testdata()['header']
        shift = imgshift_header(header)
        header['CHPCRSYS'] = 'ERF'
        header['NODCRSYS'] = 'ERF'
        header['DTHCRSYS'] = 'ERF'
        shift_erf = imgshift_header(header)
        for key in ['chopx', 'chopy', 'nodx', 'nody']:
            assert shift[key] != shift_erf[key]
        assert shift_erf['ditherx'] == -1.0
        assert shift_erf['dithery'] == 1.0

    def test_backup_coord(self):
        header = npc_testdata()['header']
        shift = imgshift_header(header)
        for k in ['CHP', 'NOD', 'DTH']:
            kold = k + 'CRSYS'
            if kold in header:
                del header[kold]
        shift_del = imgshift_header(header)
        assert shift == shift_del
        for k in ['CHPCOORD', 'NODCOORD', 'DITHERCS']:
            header[k] = 2
        shift_erf = imgshift_header(header)
        for key in ['chopx', 'chopy', 'nodx', 'nody', 'ditherx', 'dithery']:
            assert shift[key] != shift_erf[key]
