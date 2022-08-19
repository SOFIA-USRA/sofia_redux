# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.coordinate_axis import CoordinateAxis


def test_init():
    a = CoordinateAxis()
    assert a.label == 'unspecified axis'
    assert a.short_label == 'unspecified axis'
    assert a.unit == units.dimensionless_unscaled
    a = CoordinateAxis(label='foo', short_label='f', unit=units.Unit('degree'))
    assert a.label == 'foo'
    assert a.short_label == 'f'
    assert a.unit == units.Unit('degree')


def test_copy():
    a = CoordinateAxis()
    b = a.copy()
    assert a == b and a is not b


def test_eq():
    a = CoordinateAxis(label='foo', short_label='f', unit=units.Unit('degree'))
    b = a.copy()
    assert a == b
    assert a != 1
    b.unit = units.Unit('arcsec')
    assert a != b
    b = a.copy()
    b.short_label = '1'
    assert a != b
    b = a.copy()
    b.label = '1'
    assert a != b
    b = a.copy()
    b.reverse_from = 1.0
    assert a != b
    b = a.copy()
    b.reverse = True
    assert a != b


def test_str():
    a = CoordinateAxis(label='foo', short_label='f', unit=units.Unit('degree'))
    assert str(a) == 'foo (f)'
