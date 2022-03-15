# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.source_models.beams.asymmetry_2d import Asymmetry2D


@pytest.fixture
def asymmetry2d():
    a = Asymmetry2D(x=0.2, y=0.3, x_weight=1.0, y_weight=2.0)
    return a


def test_init():
    a = Asymmetry2D()
    assert a.x is None and a.y is None
    assert a.x_weight == 0 and a.y_weight == 0
    a = Asymmetry2D(x=1.0, y=2.0, x_weight=3.0, y_weight=4.0)
    assert a.x == 1 and a.y == 2 and a.x_weight == 3 and a.y_weight == 4


def test_copy(asymmetry2d):
    a = asymmetry2d
    b = asymmetry2d.copy()
    assert a.x == b.x and a.y == b.y
    assert a.x_weight == b.x_weight and a.y_weight == b.y_weight
    assert a is not b


def test_eq(asymmetry2d):
    a = asymmetry2d
    assert a == a
    b = a.copy()
    assert a == b
    b.x = 0.9
    assert a != b
    assert a != 'a'


def test_str(asymmetry2d):
    a = asymmetry2d.copy()
    s = str(a)
    assert s == 'Asymmetry: x = 20.000% +- 100.000%, y = 30.000% +- 70.711%'
    a.x = None
    assert str(a) == 'Asymmetry: y = 30.000% +- 70.711%'
    a.y = None
    assert str(a) == "Asymmetry: empty"


def test_repr(asymmetry2d):
    s = repr(asymmetry2d)
    assert s.endswith(
        'Asymmetry: x = 20.000% +- 100.000%, y = 30.000% +- 70.711%')
    assert 'Asymmetry2D object' in s


def test_x_significance(asymmetry2d):
    a = asymmetry2d.copy()
    assert a.x_significance == 0.2
    a.x_weight = None
    assert a.x_significance == np.inf


def test_y_significance(asymmetry2d):
    a = asymmetry2d.copy()
    assert np.isclose(a.y_significance, 0.424264, atol=1e-6)
    a.y_weight = None
    assert a.y_significance == np.inf


def test_x_rms(asymmetry2d):
    a = asymmetry2d.copy()
    assert a.x_rms == 1
    a.x_weight = None
    assert a.x_rms == 0


def test_y_rms(asymmetry2d):
    a = asymmetry2d.copy()
    assert np.isclose(a.y_rms, 0.707107, atol=1e-6)
    a.y_weight = None
    assert a.y_rms == 0
