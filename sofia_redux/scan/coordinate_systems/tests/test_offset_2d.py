# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.offset_2d import Offset2D


def test_init():
    c = Offset2D(reference=np.ones(2), copy=True, unit='degree')
    assert c.coordinates is None
    assert np.allclose(c.reference.coordinates.value, [1, 1])
    assert c.unit == 'degree'
    assert c.reference.unit == 'degree'
    assert c.reference.coordinates.unit == 'degree'

    c = Offset2D(reference=np.ones(2) * units.Unit('degree'), copy=False)
    assert c.unit == 'degree'
    assert c.reference.unit == 'degree'
    assert c.reference.coordinates.unit == 'degree'

    reference = c.reference.copy()
    c = Offset2D(reference=reference, copy=False)
    assert c.unit == 'degree'
    assert c.reference is reference

    c = Offset2D(reference=reference, copy=True)
    assert c.reference == reference
    assert c.reference is not reference

    x = np.arange(10).reshape(2, 5)

    c = Offset2D(reference=reference, coordinates=x)
    assert np.allclose(c.coordinates.value, x)
    assert c.coordinates.unit == 'degree'

    c = Offset2D(reference=np.ones(2), coordinates=x * units.Unit('degree'))
    assert c.unit == 'degree' and c.reference.unit == 'degree'
    assert np.allclose(c.coordinates, x * units.Unit('degree'))


def test_eq():
    c = Offset2D(reference=np.ones(2))
    assert c == c
    c2 = c.copy()
    assert c2 == c
    other = c.get_instance('equatorial')
    assert c != other

    c2 = Offset2D(reference=np.zeros(2))
    assert c != c2

    x = np.arange(10).reshape(2, 5)
    c = Offset2D(reference=np.zeros(2), coordinates=x)
    c2 = Offset2D(reference=np.zeros(2), coordinates=x.copy())
    assert c == c2
    c2 = Offset2D(reference=np.zeros(2), coordinates=x - 1)
    assert c != c2


def test_get_coordinate_class():
    c = Offset2D(reference=np.zeros(2))
    assert c.get_coordinate_class().__name__ == 'Coordinate2D'
    reference = c.get_class('equatorial')([0, 0])
    c = Offset2D(reference=reference)
    assert c.get_coordinate_class().__name__ == 'EquatorialCoordinates'
