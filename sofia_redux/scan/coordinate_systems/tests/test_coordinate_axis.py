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


def test_str():
    a = CoordinateAxis(label='foo', short_label='f', unit=units.Unit('degree'))
    assert str(a) == 'foo (f)'
