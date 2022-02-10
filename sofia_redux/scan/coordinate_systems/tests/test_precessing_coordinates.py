# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pytest


from sofia_redux.scan.coordinate_systems.precessing_coordinates import \
    PrecessingCoordinates
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.epoch.precession import Precession


class Precessing(PrecessingCoordinates):  # pragma: no cover
    """Class for testing PrecessingCoordinates"""
    def precess_to_epoch(self, new_epoch):
        if self.epoch == new_epoch:
            return
        precession = Precession(self.epoch, new_epoch)
        precession.precess(self)

    def get_equatorial_pole(self):
        return EquatorialCoordinates([0, 90], epoch='J2000')

    def get_zero_longitude(self):
        return 0 * units.Unit('degree')

    @property
    def ra(self):
        return self.x

    @property
    def dec(self):
        return self.y

    def set_ra(self, ra, copy=False):
        self.x = ra

    def set_dec(self, dec, copy=False):
        self.y = dec


@pytest.fixture
def precessing():
    x, y = np.meshgrid([-2, -1, 0, 1, 2], [-60, -45, -30, 0, 30, 45, 60])
    return Precessing([x.ravel(), y.ravel()], unit='degree', epoch='J2000')


def test_init():
    p = Precessing()
    assert str(p.epoch) == 'J2000.0'
    p = Precessing(epoch='B1950')
    assert str(p.epoch) == 'B1950.0'


def test_copy(precessing):
    p = precessing.copy()
    assert p == precessing


def test_empty_copy(precessing):
    p = precessing.empty_copy()
    assert str(p.epoch) == 'J2000.0'
    b1950 = p.epoch.get_epoch('B1950').get_epoch([1950, 1951])
    p.set_epoch(b1950)
    p2 = p.empty_copy()
    assert str(p2.epoch) == 'B1950.0'
    j2000 = p.epoch.get_epoch('J2000').get_epoch([1950, 1951])
    p.set_epoch(j2000)
    assert str(p.empty_copy().epoch) == 'J2000.0'


def test_eq(precessing):
    p = precessing.copy()
    p2 = p.copy()
    p2.set_epoch('B1950')
    assert p == p
    assert p != p2
    p.rotate(1 * units.Unit('degree'))
    assert p != precessing


def test_getitem(precessing):
    p = precessing[1]
    assert np.allclose(p.coordinates.value, [-1, -60])


def test_str(precessing):
    p = Precessing()
    assert str(p) == 'Empty coordinates (J2000.0)'

    p = precessing[0]
    assert str(p) == 'LON=-2.0 deg LAT=-60.0 deg (J2000.0)'

    assert str(precessing) == (
        'LON=-2.0 deg->2.0 deg DEC=-60.0 deg->60.0 deg (J2000.0)')


def test_empty_copy_skip_attributes(precessing):
    assert 'epoch' in precessing.empty_copy_skip_attributes


def test_copy_coordinates(precessing):
    p = precessing.copy()
    p2 = p.copy()
    p2.zero()
    p2.copy_coordinates(p)
    assert p2 == p
    c2 = p.get_class_for('2d')()
    p.convert_to(c2)
    p2.copy_coordinates(c2)
    assert p2.epoch is None
    p2.copy_coordinates(p2)
    assert p2.epoch is None
    p2.copy_coordinates(p)
    assert str(p2.epoch) == 'J2000.0'


def test_set_epoch(precessing):
    p = precessing.copy()
    p.set_epoch(2021)
    assert str(p.epoch) == 'J2021.0'


def test_precess(precessing):
    p = precessing.copy()
    p0 = p.copy()
    p.precess(p.epoch)
    assert p == p0
    p.epoch = None
    with pytest.raises(ValueError) as err:
        p.precess(p0.epoch)
    assert "Undefined from epoch" in str(err.value)
    p = precessing.copy()
    with pytest.raises(ValueError) as err:
        p.precess(None)
    assert "Undefined to epoch" in str(err.value)
    assert p == p0
    p.set_epoch(2001)
    p.precess(p0.epoch)
    assert p != p0


def test_edit_header():
    h = fits.Header()
    p = Precessing([1, 1])
    p.edit_header(h, 'X')
    assert h['RADESYS'] == 'FK5'
    p = Precessing([1, 1], epoch='B1950')
    p.edit_header(h, 'X')
    assert h['RADESYS'] == 'FK4'


def test_parse_header():
    h = fits.Header()
    p = Precessing()
    h['X1'] = 1
    h['X2'] = 2
    p.parse_header(h, 'X')
    assert p.x.value == 1 and p.y.value == 2 and str(p.epoch) == 'J2000.0'
    h['EQUINOX'] = 1983
    p.parse_header(h, 'X')
    assert p.x.value == 1 and p.y.value == 2 and str(p.epoch) == 'B1983.0'
    h['EQUINOX'] = '1982'
    p.parse_header(h, 'X')
    assert p.x.value == 1 and p.y.value == 2 and str(p.epoch) == 'B1982.0'
    h['EQUINOX'] = 'J1984'
    p.parse_header(h, 'X')
    assert p.x.value == 1 and p.y.value == 2 and str(p.epoch) == 'J1984.0'
    h['RADESYS'] = 'FK4'
    p.parse_header(h, 'X')
    assert str(p.epoch)[:10] == 'B1984.0009'


def test_convert():
    p1 = Precessing([1, 1], epoch='B1950')
    p2 = Precessing(epoch='J2000')
    p1.convert(p1, p2)
    assert np.allclose(p2.coordinates.value, [1.64049288, 1.27826545])

    c1 = p1.get_class_for('2d')([2, 2], unit='degree')
    p1.convert(p1, c1)
    assert np.allclose(c1.coordinates.value, 1)
    c1 = p1.get_class_for('2d')([2, 2], unit='degree')
    p1.convert(c1, p1)
    assert np.allclose(p1.coordinates.value, 2)


def test_get_indices():
    p = Precessing(np.arange(10).reshape(2, 5), epoch=np.arange(10))
    p2 = p.get_indices(2)
    assert np.allclose(p2.coordinates.value, [2, 7])
    assert p2.epoch.year == 2.0

    p = Precessing(np.arange(10).reshape(2, 5))
    p2 = p.get_indices(2)
    assert np.allclose(p2.coordinates.value, [2, 7])
    assert p2.epoch.year == 2000

    p = Precessing(epoch='B1950')
    p2 = p.get_indices(0)
    assert p2.coordinates is None and p2.epoch.year == 1950


def test_insert_blanks():
    p = Precessing(np.arange(10).reshape(2, 5), epoch=np.arange(5) + 10)
    p.insert_blanks([1, 1])
    n = np.nan
    assert np.allclose(p.x.value, [0, n, n, 1, 2, 3, 4], equal_nan=True)
    assert np.allclose(p.y.value, [5, n, n, 6, 7, 8, 9], equal_nan=True)
    assert np.allclose(p.epoch.equinox.jyear, [10, 0, 0, 11, 12, 13, 14])

    p = Precessing(np.arange(10).reshape(2, 5))
    p.insert_blanks([1, 1])
    assert np.allclose(p.x.value, [0, n, n, 1, 2, 3, 4], equal_nan=True)
    assert np.allclose(p.y.value, [5, n, n, 6, 7, 8, 9], equal_nan=True)
    assert p.epoch.equinox.jyear == 2000


def test_precession_required():
    pj = Precessing(np.arange(10).reshape(2, 5), epoch='J2000')
    pb = Precessing(np.arange(10).reshape(2, 5), epoch='B1950')
    precess, ej, eb = pj.precession_required(pj.epoch, pb.epoch)
    assert precess and ej == pj.epoch and eb == pb.epoch
    precess, e1, e2 = pj.precession_required(None, None)
    assert not precess and e1 is None and e2 is None
    precess, e1, e2 = pj.precession_required(ej, None)
    assert not precess and e1 == ej and e2 == ej
    precess, e1, e2 = pj.precession_required(None, ej)
    assert not precess and e1 == ej and e2 == ej

    em = ej.copy()
    em.equinox = Time(np.arange(5) + 2000.0, format='jyear')
    precess, e1, e2 = pj.precession_required(ej, em)
    assert precess and e1 == ej and e2 == em
    precess, e1, e2 = pj.precession_required(em, None)
    assert precess and e1 == em and e2 == ej
    precess, e1, e2 = pj.precession_required(None, em)
    assert precess and e1 == ej and e2 == em


def test_merge():
    pj = Precessing(np.arange(10).reshape(2, 5), epoch='J2000')
    pb = Precessing(np.arange(10).reshape(2, 5), epoch='B1950')

    p1 = pj.copy()
    p2 = pb.copy()
    p1.merge(p2)
    assert np.allclose(
        p1.x.value,
        [0., 1., 2., 3., 4.,
         0.64050527, 1.64105529, 2.64177711, 3.64267157, 4.64373964])
    assert np.allclose(
        p1.y.value,
        [5., 6., 7., 8., 9.,
         5.27833499, 6.27826542, 7.27811102, 8.27787182, 9.27754788])

    p1, p2 = pj.copy(), pj.copy()
    p1.merge(p2)
    assert np.allclose(p1.x.value, [0, 1, 2, 3, 4] * 2)
    assert np.allclose(p1.y.value, [5, 6, 7, 8, 9] * 2)

    p1, p2 = pj.copy(), pj.copy()
    p1.epoch = None
    p1.merge(p2)
    assert np.allclose(p1.x.value, [0, 1, 2, 3, 4] * 2)
    assert np.allclose(p1.y.value, [5, 6, 7, 8, 9] * 2)
    assert p1.epoch.year == 2000.0

    p1, p2 = pj.copy(), pb.copy()
    p1.epoch.equinox = Time(np.arange(p1.size) + 2001.0, format='jyear')
    p1.merge(p2)
    assert np.allclose(p1.x.value, [0, 1, 2, 3, 4] * 2)
    assert np.allclose(p1.y.value, [5, 6, 7, 8, 9] * 2)
    assert np.allclose(p1.epoch.equinox.jyear[:5],
                       [2001, 2002, 2003, 2004, 2005])
    assert np.allclose(p1.epoch.equinox.jyear[5:], 1949.99979044)

    p1, p2 = pj.copy(), pb.copy()
    p2.epoch.equinox = Time(np.arange(p1.size) + 2001.0, format='jyear')
    p1.merge(p2)
    assert p1.epoch.equinox.jyear == 2000
    assert np.allclose(
        p1.x.value,
        [0., 1., 2., 3., 4.,
         -0.01280915, 0.97436137, 1.96150136, 2.94860057, 3.93564865])


def test_paste():
    pj = Precessing(np.arange(10).reshape(2, 5), epoch='J2000')
    p_insert = Precessing((np.full((2, 2), 10.0)), epoch='B1950')

    p1 = pj.copy()
    p2 = p_insert.copy()
    p1.paste(p2, indices=np.array([2, 3]))
    assert np.allclose(p1.x.value,
                       [0., 1., 10.64928448, 10.64928448, 4.])
    assert np.allclose(p1.y.value,
                       [5., 6., 10.27383252, 10.27383252, 9.])

    p_insert = Precessing((np.full((2, 2), 10.0)), epoch='J2000')
    p1 = pj.copy()
    p2 = p_insert.copy()
    p1.paste(p2, indices=np.array([2, 3]))
    assert np.allclose(p1.x.value, [0, 1, 10, 10, 4])
    assert np.allclose(p1.y.value, [5, 6, 10, 10, 9])

    p = pj.copy()
    p.epoch.equinox = Time(
        np.arange(p.size) + 2001.0, format='jyear')
    p.paste(p2, indices=np.array([2, 3]))
    assert np.allclose(p.x.value, [0, 1, 10, 10, 4])
    assert np.allclose(p.y.value, [5, 6, 10, 10, 9])
    assert np.allclose(p.epoch.equinox.jyear, [2001, 2002, 2000, 2000, 2005])
