# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.projection.polyconic_projection \
    import PolyconicProjection


def test_init():
    p = PolyconicProjection()
    assert p.reference.size == 0


def test_get_fits_id():
    assert PolyconicProjection.get_fits_id() == 'PCO'


def test_get_full_name():
    assert PolyconicProjection.get_full_name() == 'Polyconic'


def test_get_phi_theta():
    p = PolyconicProjection()
    with pytest.raises(NotImplementedError) as err:
        _ = p.get_phi_theta(Coordinate2D([0, 0]))
    assert "Deprojection not implemented" in str(err.value)


def test_get_offsets():
    p = PolyconicProjection()
    # Test zero theta
    o = p.get_offsets(0, np.pi / 4)
    assert np.allclose(o.coordinates.value, [45, 0])
    o = p.get_offsets(np.pi / 4, np.pi / 6)
    assert np.allclose(o.coordinates.value, [20.73187096, 48.88233667])
