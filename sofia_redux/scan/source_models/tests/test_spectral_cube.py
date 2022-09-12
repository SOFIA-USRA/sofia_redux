# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.index_3d import Index3D
from sofia_redux.scan.source_models.maps.observation_2d1 import Observation2D1
from sofia_redux.scan.source_models.spectral_cube import SpectralCube
from sofia_redux.scan.coordinate_systems.grid.spherical_grid_2d1 import \
    SphericalGrid2D1


arcsec = units.Unit('arcsec')
um = units.Unit('um')


@pytest.fixture
def fifi_info(fifi_simulated_reduction):
    return fifi_simulated_reduction.info


@pytest.fixture
def fifi_cube(fifi_simulated_reduction):
    fifi_simulated_reduction.validate()
    return fifi_simulated_reduction.source.copy()


@pytest.fixture
def processed_cube(fifi_cube):
    m = fifi_cube.copy()
    pipeline = m.reduction.pipeline
    scan_source = pipeline.scan_source.copy()
    scan_source.renew()
    scan = m.scans[0].copy()
    integration = scan[0]
    scan_source.add_integration(integration)
    scan_source.process_scan(scan)
    return scan_source


def test_init(fifi_info):
    m = SpectralCube(fifi_info.copy())
    assert np.all(m.smoothing.is_null())
    assert m.smoothing.xy_unit == 'arcsec'
    assert m.smoothing.z_unit == 'um'


def test_copy(fifi_cube):
    m = fifi_cube.copy()
    assert m is not fifi_cube
    assert m.smoothing == fifi_cube.smoothing


def test_create_grid(fifi_cube):
    m = fifi_cube.copy()
    base_resolution = m.info.instrument.resolution.copy()
    if 'grid' in m.configuration:
        del m.configuration['grid']
    m.grid = None
    m.create_grid()
    assert isinstance(m.grid, SphericalGrid2D1)
    res = base_resolution.copy()
    res.scale(0.2)
    assert m.grid.resolution == res
    m.configuration.parse_key_value('grid', '5')
    m.create_grid()
    assert m.grid.resolution.x == 5 * arcsec
    assert m.grid.resolution.y == 5 * arcsec
    assert m.grid.resolution.z == res.z
    m.configuration.parse_key_value('grid', '5,6')
    m.create_grid()
    assert m.grid.resolution.x == 5 * arcsec
    assert m.grid.resolution.y == 5 * arcsec
    assert m.grid.resolution.z == 6 * um
    m.configuration.parse_key_value('grid', '5,6,7')
    m.create_grid()
    assert m.grid.resolution.x == 5 * arcsec
    assert m.grid.resolution.y == 6 * arcsec
    assert m.grid.resolution.z == 7 * um


def test_reference(fifi_cube):
    m = fifi_cube.copy()
    ref = m.reference
    assert isinstance(ref, Coordinate2D1)
    m.grid = None
    assert m.reference is None
    m.create_grid()
    m.reference = ref
    assert m.reference == ref
    c = Coordinate2D1()
    with pytest.raises(ValueError) as err:
        m.reference = c
    assert 'Reference xy coordinates must be' in str(err.value)

    with pytest.raises(ValueError) as err:
        m.reference = Coordinate2D()
    assert 'Reference coordinates must be' in str(err.value)

    m.grid = None
    m.reference = ref
    assert m.reference is None


def test_size_x(fifi_cube):
    assert fifi_cube.size_x == 12


def test_size_y(fifi_cube):
    assert fifi_cube.size_y == 12


def test_size_z(fifi_cube):
    assert fifi_cube.size_z == 11


def test_get_reference(fifi_cube):
    assert fifi_cube.get_reference() == fifi_cube.reference


def test_create_from(fifi_cube):
    m = fifi_cube.copy()
    m.create_from(m.reduction.scans)
    assert m.shape == (11, 12, 12)


def test_create_map(fifi_cube):
    m = fifi_cube.copy()
    m.map = None
    m.create_map()
    assert isinstance(m.map, Observation2D1)


def test_set_size(fifi_cube):
    m = fifi_cube.copy()
    if 'grid' in m.configuration:
        del m.configuration['grid']
    if 'large' in m.configuration:
        del m.configuration['large']

    m.set_size()
    assert m.shape == (57, 72, 63)
    m.configuration.parse_key_value('grid', '5')
    m.set_size()
    assert m.shape == (57, 20, 18)
    m.configuration.parse_key_value('grid', '5,5')
    m.set_size()
    assert m.shape == (2, 20, 18)
    m.configuration.parse_key_value('grid', '5,6,7')
    m.set_size()
    assert m.shape == (2, 16, 18)
    m.configuration.parse_key_value('grid', '0.01,1000,1000')

    with pytest.raises(ValueError) as err:
        m.set_size()
    assert 'too large' in str(err.value)

    m.configuration.parse_key_value('grid', '-5')
    with pytest.raises(ValueError) as err:
        m.set_size()
    assert 'Negative image size' in str(err.value)


def test_search_corners(fifi_cube):
    m = fifi_cube.copy()
    r = m.search_corners(determine_scan_range=True)
    assert np.allclose(r.x, [-38.72975731, 37.72168199] * arcsec, atol=1e-4)
    assert np.allclose(r.y, [-43.57159237, 42.94501899] * arcsec, atol=1e-4)
    assert np.allclose(r.z, [-0.30252361, 0.31728493] * um, atol=1e-4)
    m.configuration.parse_key_value('map.size', '5,5')
    r = m.search_corners(determine_scan_range=False)
    assert np.allclose(r.x, [-2.5, 2.5] * arcsec, atol=1e-4)
    assert np.allclose(r.y, [-2.5, 2.5] * arcsec, atol=1e-4)
    assert np.allclose(r.z, [-0.30252361, 0.31728493] * um, atol=1e-4)
    m.configuration.parse_key_value('map.size', '5,5,0.2')
    r = m.search_corners(determine_scan_range=False)
    assert np.allclose(r.x, [-2.5, 2.5] * arcsec, atol=1e-4)
    assert np.allclose(r.y, [-2.5, 2.5] * arcsec, atol=1e-4)
    assert np.allclose(r.z, [-0.1, 0.1] * um, atol=1e-4)


def test_create_lookup(fifi_cube, capsys):
    m = fifi_cube.copy()
    integration = m.scans[0][0]
    integration.frames.map_index = None
    integration.frames.source_index = None
    m.create_lookup(integration)
    assert isinstance(integration.frames.map_index, Index3D)
    assert isinstance(integration.frames.source_index, np.ndarray)

    # With values already existing
    reference = m.reference.copy()
    reference.z -= 0.2 * um
    m.reference = reference
    m.create_lookup(integration)
    assert isinstance(integration.frames.map_index, Index3D)
    assert isinstance(integration.frames.source_index, np.ndarray)
    assert '19712 samples have bad map indices' in capsys.readouterr().err


def test_pixel_index_to_source_index(fifi_cube):
    m = fifi_cube.copy()
    m.set_data_shape((2, 3, 4))
    pixel_indices = np.stack([x.ravel() for x in np.indices((2, 3, 4))[::-1]])
    source_indices = m.pixel_index_to_source_index(pixel_indices)
    assert np.allclose(source_indices, np.arange(24))
    pixel_indices[0, 0] = -1
    pixel_indices[1, 1] = -1
    pixel_indices[2, 2] = -1
    source_indices = m.pixel_index_to_source_index(pixel_indices)
    assert np.allclose(source_indices[:3], -1)
    assert np.allclose(source_indices[3:], np.arange(21) + 3)


def test_source_index_to_pixel_index(fifi_cube):
    m = fifi_cube.copy()
    m.set_data_shape((2, 3, 4))
    pixel_indices = m.source_index_to_pixel_index(np.arange(24))
    expected = np.stack([x.ravel() for x in np.indices((2, 3, 4))[::-1]])
    assert np.allclose(pixel_indices, expected)
    bad_indices = np.array([1, 2, -1])
    bad = m.source_index_to_pixel_index(bad_indices)
    assert np.allclose(bad, [[1, 2, -1], [0, 0, -1], [0, 0, -1]])


def test_get_smoothing(fifi_cube):
    m = fifi_cube.copy()
    m.configuration.parse_key_value('grid', '1,0.01')
    m.create_grid()
    fwhm = m.info.instrument.resolution.copy()
    pix_smooth = m.get_pixelization_smoothing()
    smooth = m.get_smoothing(None)
    assert smooth == pix_smooth
    smooth = m.get_smoothing(1)
    assert smooth == pix_smooth
    smooth = m.get_smoothing(Coordinate2D1([5, 5] * arcsec, 2 * um))
    assert smooth == Coordinate2D1([5, 5] * arcsec, 2 * um)
    m.configuration.parse_key_value('smooth.optimal', '3')
    smooth = m.get_smoothing('optimal')
    assert smooth.x == smooth.y == 3 * arcsec
    assert smooth.z == fwhm.z
    m.configuration.parse_key_value('smooth.optimal', '4,5')
    smooth = m.get_smoothing('optimal')
    assert smooth.x == smooth.y == 4 * arcsec
    assert smooth.z == 5 * um
    m.configuration.parse_key_value('smooth.optimal', '4,5,6')
    smooth = m.get_smoothing('optimal')
    assert smooth.x == 4 * arcsec and smooth.y == 5 * arcsec
    assert smooth.z == 6 * um
    del m.configuration['smooth.optimal']
    smooth = m.get_smoothing('optimal')
    assert smooth == fwhm
    smooth = m.get_smoothing('minimal')
    expected = fwhm.copy()
    expected.scale(0.3)
    assert smooth == expected
    smooth = m.get_smoothing('2/3beam')
    expected = fwhm.copy()
    expected.scale(2/3)
    assert smooth == expected
    smooth = m.get_smoothing('halfbeam')
    expected = fwhm.copy()
    expected.scale(0.5)
    assert smooth == expected
    smooth = m.get_smoothing('beam')
    assert smooth == fwhm


def test_set_smoothing(fifi_cube):
    m = fifi_cube.copy()
    smoothing = Coordinate2D1([4, 5] * arcsec, 6 * um)
    m.set_smoothing(smoothing)
    assert m.smoothing == smoothing


def test_get_requested_smoothing(fifi_cube):
    m = fifi_cube.copy()
    smoothing = m.get_requested_smoothing(None)
    assert np.all(smoothing.is_null())
    smoothing = m.get_requested_smoothing('minimal')
    assert smoothing == m.get_pixelization_smoothing()


def test_get_pixelization_smoothing(fifi_cube):
    m = fifi_cube.copy()
    m.configuration.parse_key_value('grid', '1,2,3')
    m.create_grid()
    smoothing = m.get_pixelization_smoothing()
    assert np.isclose(smoothing.x, 1.3285649 * arcsec, atol=1e-5)
    assert smoothing.x == smoothing.y
    assert np.isclose(smoothing.z, 2.8183118 * um, atol=1e-5)


def test_get_point_size(fifi_cube):
    m = fifi_cube.copy()
    m.configuration.parse_key_value('grid', '1,1,0.01')
    m.create_grid()
    m.configuration.parse_key_value('smooth', 'minimal')
    point_size_0 = m.get_point_size()
    m.configuration.parse_key_value('smooth', 'beam')
    point_size_1 = m.get_point_size()
    assert point_size_1.x > point_size_0.x
    assert point_size_1.y > point_size_0.y
    assert point_size_1.z > point_size_0.z


def test_get_source_size(fifi_cube):
    m = fifi_cube.copy()
    m.configuration.parse_key_value('grid', '1,1,0.01')
    m.create_grid()
    m.configuration.parse_key_value('smooth', 'minimal')
    source_size_0 = m.get_source_size()
    m.configuration.parse_key_value('smooth', 'beam')
    source_size_1 = m.get_source_size()
    assert source_size_1.x > source_size_0.x
    assert source_size_1.y > source_size_0.y
    assert source_size_1.z > source_size_0.z


def test_stand_alone(fifi_cube):
    m = fifi_cube.copy()
    m.base.data.fill(1)
    m.stand_alone()
    assert m.base.shape == m.shape
    assert np.allclose(m.base.data, 0)


def test_post_process_scan(processed_cube):
    m = processed_cube.copy()
    m.post_process_scan(m.scans[0])  # does nothing


def test_get_peak_index(processed_cube):
    m = processed_cube.copy()
    m.configuration.parse_key_value('source.sign', '+')
    s = m.get_significance().data.copy()
    i = m.get_peak_index()
    assert s[i.z, i.y, i.x] == s.max()
    m.configuration.parse_key_value('source.sign', '-')
    i = m.get_peak_index()
    assert s[i.z, i.y, i.x] == s.min()
    m.configuration.parse_key_value('source.sign', '0')
    i = m.get_peak_index()
    assert s[i.z, i.y, i.x] == s.max()


def test_get_peak_coords(processed_cube):
    m = processed_cube.copy()
    m.configuration.parse_key_value('source.sign', '+')
    cp = m.get_peak_coords()
    m.configuration.parse_key_value('source.sign', '-')
    cm = m.get_peak_coords()
    assert isinstance(cp.xy_coordinates, EquatorialCoordinates)
    assert isinstance(cm.xy_coordinates, EquatorialCoordinates)
    assert cp != cm


def test_mask_integration_samples(processed_cube):
    m = processed_cube.copy()
    integration = m.scans[0][0]
    sample_flags = integration.frames.sample_flag
    s0 = sample_flags.copy()
    masked = m.is_masked()
    assert not masked.any()
    map_flags = m.map.flag
    m0 = map_flags.copy()

    m.mask_integration_samples(integration)
    assert np.allclose(map_flags, m0)
    assert np.allclose(sample_flags, s0)

    # Find first discarded map point
    sample_flags.fill(0)
    inds = np.nonzero(map_flags)
    sample_flag = integration.frames.flagspace.convert_flag(
        'SAMPLE_SKIP').value

    # This should not do anything
    m.map.flag[inds[0][0], inds[1][0], inds[2][0]] = 2
    # rebuild the frame map indices
    integration.frames.map_index = None
    m.mask_integration_samples(integration)
    assert np.allclose(sample_flags, 0)

    # Now mask the first good map point
    inds = np.nonzero(map_flags == 0)
    m.map.flag[inds[0][0], inds[1][0], inds[2][0]] = 2
    m.mask_integration_samples(integration)
    assert not np.allclose(sample_flags, 0)
    assert np.allclose(np.unique(sample_flags), [0, sample_flag])


def test_get_map_2d(processed_cube):
    m = processed_cube.copy()
    map2d = m.get_map_2d()
    assert map2d is m.map


def test_get_data(processed_cube):
    m = processed_cube.copy()
    data = m.get_data()
    assert data is m.map


def test_smooth_to(processed_cube):
    m = processed_cube.copy()
    d0 = m.map.data.copy()
    m.smooth_to(Coordinate2D1([1, 1] * arcsec, 0.1 * um))
    d1 = m.map.data.copy()
    assert not np.allclose(d0, d1)


def test_filter_source(processed_cube):
    m = processed_cube.copy()
    d0 = m.map.data.copy()
    filter_fwhm = Coordinate2D1([10, 10] * arcsec, 0.1 * um)
    m.filter_source(filter_fwhm)
    assert m.map.filter_fwhm == filter_fwhm
    assert not np.allclose(d0, m.map.data)


def test_set_filtering(fifi_cube):
    m = fifi_cube.copy()
    fwhm = Coordinate2D1([10, 10] * arcsec, 0.1 * um)
    m.set_filtering(fwhm)
    assert m.map.filter_fwhm == fwhm


def test_get_average_resolution(processed_cube):
    m = processed_cube.copy()
    scan1 = m.scans[0]
    scan2 = scan1.copy()
    integration = scan2[0].copy()
    scan2.integrations = [integration]
    m.scans = [scan1, scan2]

    integration.info.instrument.resolution.zero()

    average = m.get_average_resolution()
    assert average != scan1.info.instrument.resolution
    scan1.weight = 0
    scan2.weight = 0
    average = m.get_average_resolution()
    assert average == m.info.instrument.resolution
