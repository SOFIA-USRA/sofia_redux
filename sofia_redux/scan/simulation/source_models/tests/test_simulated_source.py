# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.simulation.source_models.simulated_source import \
    SimulatedSource
from sofia_redux.scan.simulation.source_models.single_gaussian import \
    SingleGaussian


class SimSource(SimulatedSource):
    def apply_to_offsets(self, offsets):
        return 1

    def apply_to_horizontal(self, offsets):
        return 2


def test_init():
    source = SimSource()
    assert source.name == 'base'


def test_call():
    source = SimSource()
    assert source(Coordinate2D()) == 1
    assert source(HorizontalCoordinates()) == 2


def test_copy():
    source = SimSource()
    source2 = source.copy()
    assert isinstance(source2, SimSource) and source2 is not SimSource


def test_get_source_model():
    source = SimulatedSource.get_source_model('single_gaussian',
                                              fwhm=10 * units.Unit('arcsec'))
    assert isinstance(source, SingleGaussian)
