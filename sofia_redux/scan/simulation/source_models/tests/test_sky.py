# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.simulation.source_models.sky import Sky
from sofia_redux.scan.source_models.sky_dip_model import SkyDipModel


arcsec = units.Unit('arcsec')
kelvin = units.Unit('Kelvin')


def test_init():
    source = Sky()
    assert source.name == 'sky'
    assert isinstance(source.model, SkyDipModel)
    assert source.tau == 0.1
    assert source.t_offset == 0 * kelvin
    assert source.scaling == 1
    assert source.t_sky == 273 * kelvin

    source = Sky(tau=0.2, scaling=2, tsky=250, offset=1)
    assert source.tau == 0.2
    assert source.scaling == 2
    assert source.t_sky == 250 * kelvin
    assert source.t_offset == 1 * kelvin


def test_initialize_model():
    source = Sky()
    source.initialize_model(tau=0.2, scaling=2, tsky=250, offset=1)
    assert source.tau == 0.2
    assert source.scaling == 2
    assert source.t_sky == 250 * kelvin
    assert source.t_offset == 1 * kelvin


def test_apply_to_offsets():
    source = Sky()
    offsets = Coordinate2D()
    with pytest.raises(NotImplementedError) as err:
        _ = source(offsets)
    assert "Can only determine sky from horizontal coordinates" in str(
        err.value)


def test_apply_to_horizontal():
    source = Sky()
    el = np.arange(6) * 15 * units.Unit('degree')
    az = np.zeros(6) * units.Unit('degree')
    horizontal = HorizontalCoordinates([az, el])
    data = source(horizontal)
    assert np.allclose(
        data,
        [273, 87.491359, 49.486504, 36.002299, 29.7714, 26.849244], atol=1e-6)
