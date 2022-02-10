# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.time import Time
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.epoch.epoch import Epoch, J2000, B1950
from sofia_redux.scan.coordinate_systems.epoch.precession import Precession
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates


@pytest.fixture
def b2j():
    return Precession(B1950, J2000)


@pytest.fixture
def j2b():
    return Precession(J2000, B1950)


@pytest.fixture
def to_2001():
    return Precession(J2000, Epoch(2001))


@pytest.fixture
def from_2001():
    return Precession(Epoch(2001), J2000)


@pytest.fixture
def matrix_1950_2000():
    return np.asarray(
        [[9.99925744e-01, -1.11761854e-02, -4.85784291e-03],
         [1.11761854e-02, 9.99937544e-01, -2.71494197e-05],
         [4.85784293e-03, -2.71447492e-05, 9.99988200e-01]])


def test_init(matrix_1950_2000):
    p = Precession(J2000, J2000)
    assert p.from_epoch == J2000 and p.to_epoch == J2000
    assert p.p is None

    p = Precession(B1950, J2000)
    assert p.from_epoch == B1950 and p.to_epoch == J2000
    assert np.allclose(p.p, matrix_1950_2000)


def test_copy():
    b = Precession(B1950, J2000)
    b2 = b.copy()
    assert b2 == b and b2 is not b
    assert np.allclose(b2.p, b.p)


def test_eq(b2j, j2b):
    assert b2j == b2j
    assert b2j is not None
    assert b2j != j2b
    b2 = Precession(b2j.from_epoch, b2j.from_epoch)
    assert b2 != b2j
    assert b2j == b2j.copy()


def test_singular_epoch(b2j):
    assert b2j.singular_epoch
    p = Precession(Epoch([2001, 2002]), Epoch([2002, 2003]))
    assert not p.singular_epoch


def test_r2():
    phi = 30 * units.Unit('degree')
    r2 = Precession.r2(phi)
    c30 = np.sqrt(3) / 2
    assert np.allclose(
        r2,
        [[c30, 0, -0.5],
         [0, 1, 0],
         [0.5, 0, c30]])
    r2_30 = r2

    phi = [30, 60] * units.Unit('degree')
    r2 = Precession.r2(phi)
    assert np.allclose(r2[0], r2_30)
    assert np.allclose(
        r2[1],
        [[0.5, 0, -c30],
         [0, 1, 0],
         [c30, 0, 0.5]])


def test_r3():
    phi = 30 * units.Unit('degree')
    r3 = Precession.r3(phi)
    c30 = np.sqrt(3) / 2
    assert np.allclose(
        r3,
        [[c30, 0.5, 0],
         [-0.5, c30, 0],
         [0, 0, 1]])
    r3_30 = r3

    phi = [30, 60] * units.Unit('degree')
    r3 = Precession.r3(phi)
    assert np.allclose(r3[0], r3_30)
    assert np.allclose(
        r3[1],
        [[0.5, c30, 0],
         [-c30, 0.5, 0],
         [0, 0, 1]])


def test_calculate_matrix(to_2001):
    p = Precession(J2000, J2000)
    assert p.p is None
    p.calculate_matrix()
    assert np.allclose(p.p, np.eye(3))

    p = to_2001.copy()
    assert p.p is not None
    p.p = None
    p.calculate_matrix()
    assert np.allclose(
        p.p,
        [[9.99999970e-01, -2.23562852e-04, -9.71482991e-05],
         [2.23562852e-04, 9.99999975e-01, -1.08593943e-08],
         [9.71482991e-05, -1.08593569e-08, 9.99999995e-01]])


def test_precess(from_2001, to_2001):
    c = EquatorialCoordinates([1, 1])
    c0 = c.copy()
    p_forward = to_2001
    p_reverse = from_2001
    p = Precession(J2000, J2000)
    # Test no precession
    p.precess(c)
    assert c == c0
    assert c.epoch == J2000

    p_forward.precess(c)
    assert c.epoch != J2000 and c.epoch.year == 2001
    # Note x coordinates are negative RA
    assert np.allclose(c.coordinates.value, [[-1.01281092, 1.00556533]])
    c2001 = c.copy()
    p_reverse.precess(c)
    assert c == c0
    assert c.epoch == J2000

    c = EquatorialCoordinates([[1, 1], [1, 1]])
    from_epoch = Epoch(Time([2000, 2002], format='jyear', scale='tt'))
    precession = Precession(from_epoch, to_2001.to_epoch)
    precession.precess(c)
    assert c[0] == c2001
    assert np.allclose(c[1].coordinates.value, [-0.98718903, 0.99443467])
    assert c.epoch == c2001.epoch
