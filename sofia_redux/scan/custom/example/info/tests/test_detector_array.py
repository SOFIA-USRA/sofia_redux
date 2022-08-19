# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.custom.example.info.detector_array import \
    ExampleDetectorArrayInfo

arcsec = units.Unit('arcsec')


def test_class():
    assert ExampleDetectorArrayInfo.COLS == 11
    assert ExampleDetectorArrayInfo.ROWS == 11
    b = Coordinate2D([5, 5])
    assert ExampleDetectorArrayInfo.boresight_index == b
    assert ExampleDetectorArrayInfo.pixel_size == 2 * arcsec


def test_init():
    info = ExampleDetectorArrayInfo()
    assert info.pixels == 121
    assert info.pixel_sizes == Coordinate2D([2, 2], unit=arcsec)


def test_get_sibs_position():
    info = ExampleDetectorArrayInfo()
    p = info.get_sibs_position(3, 4)
    assert p == Coordinate2D([2, 4], unit='arcsec')


def test_initialize_channel_data(populated_scan):
    data = populated_scan.channels.data
    info = ExampleDetectorArrayInfo()
    data.position = None
    info.initialize_channel_data(data)
    index = np.arange(121)
    assert np.allclose(data.fixed_index, index)
    assert np.allclose(data.col, index % 11)
    assert np.allclose(data.row, index // 11)
    assert isinstance(data.position, Coordinate2D)


def test_equatorial_to_detector_coordinates():
    info = ExampleDetectorArrayInfo()
    equatorial = EquatorialCoordinates([1, 2])
    coordinates = info.equatorial_to_detector_coordinates(equatorial)
    assert coordinates.x == -1 * units.Unit('degree')
    assert coordinates.y == 2 * units.Unit('degree')
