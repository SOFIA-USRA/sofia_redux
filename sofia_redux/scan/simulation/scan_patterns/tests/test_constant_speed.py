# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.simulation.scan_patterns.constant_speed import (
    to_constant_speed)


def test_to_constant_speed():
    arcsec = units.Unit('arcsec')
    x = np.arange(10, dtype=float) ** 2
    y = x + 2
    c = Coordinate2D([x, y], unit=arcsec)
    cc = to_constant_speed(c)
    assert c.max == cc.max and c.min == cc.min
    expected_x = np.arange(10) * 9 * arcsec
    assert np.allclose(cc.x, expected_x)
    assert np.allclose(cc.y, expected_x + 2 * arcsec)
