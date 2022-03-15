# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.source_models.beams.asymmetry_2d import Asymmetry2D
from sofia_redux.scan.source_models.beams.instant_focus import InstantFocus


@pytest.fixture
def instant_focus():
    focus = InstantFocus()
    focus.x = 0.3
    focus.y = 0.4
    focus.z = 0.5
    focus.x_weight = 9.0
    focus.x_weight = 16.0
    focus.z_weight = 25.0
    return focus


def test_init():
    focus = InstantFocus()
    assert focus.x is None and focus.x_weight is None
    assert focus.y is None and focus.y_weight is None
    assert focus.z is None and focus.z_weight is None


def test_copy(instant_focus):
    focus = instant_focus
    focus2 = focus.copy()
    assert focus2 == focus and focus2 is not focus


def test_copy_from(instant_focus):
    focus = InstantFocus()
    focus.copy_from(instant_focus)
    assert focus == instant_focus


def test_is_valid(instant_focus):
    f = InstantFocus()
    assert not f.is_valid
    assert instant_focus.is_valid


def test_is_complete(instant_focus):
    assert not InstantFocus().is_complete
    assert instant_focus.is_complete


def test_str():
    focus = InstantFocus()
    assert str(focus) == 'No focus results'
    mm = units.Unit('mm')
    imm2 = 1 / mm ** 2
    focus.x = 1 * mm
    focus.y = 2 * mm
    focus.z = 3 * mm
    focus.x_weight = 4 * imm2
    focus.y_weight = 9 * imm2
    focus.z_weight = 25 * imm2
    s = str(focus)
    assert s == ('Focus results: x=1.000000+-0.500000 mm '
                 'y=2.000000+-0.333333 mm z=3.000000+-0.200000 mm')
    focus.x = None
    focus.y_weight = None
    s = str(focus)
    assert s == 'Focus results: y=2.000000 mm z=3.000000+-0.200000 mm'


def test_repr():
    focus = InstantFocus()
    focus.x = 1 * units.Unit('mm')
    focus.y_weight = 1 / units.Unit('mm') ** 2
    s = repr(focus)
    assert 'InstantFocus object' in s
    assert s.endswith('Focus results: x=1.000000 mm')


def test_eq(instant_focus):
    f1 = instant_focus.copy()
    f2 = instant_focus.copy()
    assert f1 == f1
    assert f1 == f2
    assert f1 != 1
    f2.z = 10
    assert f1 != f2
    f2.z = f1.z
    f2.y = 10
    assert f1 != f2
    f2.y = f1.y
    f2.x = 10
    assert f1 != f2


def test_derive_from():
    c = Configuration()
    c.parse_key_value('focus.significance', '3.0')
    c.parse_key_value('focus.xcoeff', '2.0')
    c.parse_key_value('focus.ycoeff', '3.0')
    c.parse_key_value('focus.zcoeff', '4.0')
    c.parse_key_value('focus.elong0', '10.0')  # percent

    c.parse_key_value('focus.xscatter', '0.0')
    c.parse_key_value('focus.yscatter', '1.0')
    c.parse_key_value('focus.zscatter', '2.0')

    asymmetry = Asymmetry2D(x=1.0, y=1.0, x_weight=16, y_weight=25.0)
    focus = InstantFocus()
    focus.derive_from(c, asymmetry=asymmetry, elongation=0.4,
                      elongation_weight=400)

    mm = units.Unit('mm')
    imm2 = 1 / mm ** 2
    assert np.isclose(focus.x, -0.5 * mm)
    assert np.isclose(focus.x_weight, 64 * imm2)
    assert np.isclose(focus.y, -1/3 * mm)
    assert np.isclose(focus.y_weight, 0.995575 * imm2, atol=1e-6)
    assert np.isclose(focus.z, -0.075 * mm)
    assert np.isclose(focus.z_weight, 0.249990 * imm2, atol=1e-6)

    c.parse_key_value('focus.significance', '4.0')
    focus.derive_from(c, asymmetry=asymmetry, elongation=0.4,
                      elongation_weight=400)
    assert focus.x is None and focus.x_weight is None
    assert np.isclose(focus.y, -1/3 * mm)
    assert np.isclose(focus.y_weight, 0.995575 * imm2, atol=1e-6)
    assert np.isclose(focus.z, -0.075 * mm)
    assert np.isclose(focus.z_weight, 0.249990 * imm2, atol=1e-6)

    focus.derive_from(c, asymmetry=None, elongation=0.4,
                      elongation_weight=400)
    assert focus.x is None and focus.y is None
    assert np.isclose(focus.z, -0.075 * mm)
    assert np.isclose(focus.z_weight, 0.249990 * imm2, atol=1e-6)

    c.parse_key_value('focus.significance', '2.0')
    del c['focus.ycoeff']
    focus.derive_from(c, asymmetry=asymmetry)
    assert np.isclose(focus.x, -0.5 * mm)
    assert np.isclose(focus.x_weight, 64 * imm2)
    assert focus.y is None and focus.z is None

    asymmetry.x = None
    focus.derive_from(c, asymmetry=asymmetry)
    assert focus.x is None and focus.y is None and focus.z is None
