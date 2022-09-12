# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from copy import deepcopy
import imageio
import numpy as np
import os
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid
from sofia_redux.scan.source_models.astro_model_2d import AstroModel2D
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


arcsec = units.Unit('arcsec')


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    have_matplotlib = False
    plt = None


class FunctionalAstroModel2D(AstroModel2D):  # pragma: no cover

    def __init__(self, info, reduction=None):
        super().__init__(info, reduction=reduction)
        self.input_data_shape = None
        self.use_input_shape = False
        self.add_to_master = False
        self.set_empty = True
        self.do_png_write = True
        self.png_data = None

    @property
    def shape(self):
        if self.use_input_shape:
            return self.input_data_shape
        return super().shape

    def is_empty(self):
        return self.set_empty

    def is_adding_to_master(self):
        if not self.add_to_master:
            return super().is_adding_to_master()
        return True

    def get_pixel_footprint(self):
        return 2

    def base_footprint(self, pixels):
        return pixels

    def process_final(self):
        pass

    def write_fits(self, filename):
        pass

    def get_map_2d(self):
        pass

    def merge_accumulate(self, other):
        print('merge accumulate')

    def set_data_shape(self, shape):
        self.input_data_shape = shape

    def covariant_points(self):
        return 1

    def add_points(self, frames, pixels, frame_gains, source_gains):
        return 999

    def sync_source_gains(self, frames, pixels, frame_gains, source_gains,
                          sync_gains):
        pass

    def calculate_coupling(self, integration, pixels, source_gains,
                           sync_gains):
        pass

    def count_points(self):
        return 1

    def get_source_name(self):
        return 'test'

    def get_default_core_name(self):
        return 'default_core_name'

    def get_unit(self):
        pass

    def clear_content(self):
        pass

    def add_model_data(self, source_model, weight=1.0):
        pass

    def process(self):
        pass

    def process_scan(self, scan):
        pass

    def set_base(self):
        pass

    def write_image_png(self, image, file_name):
        if self.do_png_write:
            super().write_image_png(image, file_name)
        else:
            self.png_data = image, file_name


@pytest.fixture
def example_reduction():
    reduction = Reduction('example')
    reduction.configuration.parse_key_value('parallel.cores', '1')
    reduction.configuration.parse_key_value('parallel.jobs', '1')
    reduction.configuration.parse_key_value('indexing.check_memory', 'False')
    reduction.update_parallel_config()
    return reduction


@pytest.fixture
def basic_source(example_reduction):
    source = FunctionalAstroModel2D(example_reduction.info,
                                    reduction=example_reduction)
    source.configuration.parse_key_value('indexing.check_memory', 'False')
    return source


@pytest.fixture
def basic_with_projection(basic_source, populated_scan):
    source = basic_source.copy()
    source.use_input_shape = True
    scans = [populated_scan]
    source.assign_scans(scans)
    projection = SphericalProjection.for_name('gnomonic')
    projection.set_reference(populated_scan.get_position_reference(
        'equatorial'))
    source.projection = projection
    return source


@pytest.fixture
def initialized_source(basic_source, populated_scan):
    source = basic_source.copy()
    source.use_input_shape = True
    source.allow_indexing = True
    source.configuration.parse_key_value('indexing', 'True')
    source.create_from([populated_scan])
    return source


@pytest.fixture
def obs2d(initialized_source):
    data = np.zeros((101, 101), dtype=float)
    data[50, 50] = 1
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 0.5)
    o.set_resolution(1 * units.Unit('arcsec'))
    o.grid = initialized_source.grid
    return o


def test_class():
    assert AstroModel2D.DEFAULT_PNG_SIZE == 300
    assert AstroModel2D.MAX_X_OR_Y_SIZE == 5000


def test_init(example_reduction):
    reduction = example_reduction
    info = reduction.info
    info.configuration.parse_key_value('grid', '3.0')
    model = FunctionalAstroModel2D(info)
    assert isinstance(model.grid, SphericalGrid)
    assert model.smoothing == 0 * arcsec
    assert model.allow_indexing
    assert model.index_shift_x == 0 and model.index_mask_y == 0
    assert model.grid.resolution.x == 3 * arcsec
    assert model.grid.resolution.y == 3 * arcsec
    assert model.reduction is None

    model = FunctionalAstroModel2D(info, reduction=reduction)
    assert model.reduction is reduction
    assert model.info is reduction.info

    del info.configuration['grid']
    model = FunctionalAstroModel2D(info)
    assert model.grid.resolution.x == 2 * arcsec
    assert model.grid.resolution.y == 2 * arcsec
    info.instrument.resolution = Coordinate2D([20, 30], unit='arcsec')
    model = FunctionalAstroModel2D(info)
    assert model.grid.resolution.x == 4 * arcsec
    assert model.grid.resolution.y == 6 * arcsec


def test_copy(basic_source):
    source = basic_source
    source.scans = [1, 2, 3]
    source.process_brief = 4
    source.integration_time = 5 * units.Unit('second')
    source2 = source.copy()
    assert source2.reduction is source.reduction
    assert source2.scans is source.scans
    assert source2.integration_time == source.integration_time
    assert source2.integration_time is not source.integration_time
    assert source2.process_brief is None

    source2 = source.copy(with_contents=False)
    assert source2.scans is source.scans
    assert source2.reduction is source.reduction
    assert source2.integration_time != source.integration_time


def test_clear_all_memory(initialized_source):
    source = initialized_source.copy()
    source.clear_all_memory()
    assert source.grid is None
    assert source.smoothing == 0


def test_projection(basic_source):
    source = basic_source.copy()
    projection = source.projection
    assert isinstance(projection, SphericalProjection)
    projection_copy = source.projection.copy()

    with pytest.raises(ValueError) as err:
        source.projection = 1
    assert "Projection must be" in str(err.value)
    source.projection = projection_copy
    assert source.projection is projection_copy
    assert source.projection is not projection
    source.grid = None
    assert source.projection is None


def test_reference(basic_source):
    source = basic_source.copy()
    reference = source.reference
    assert isinstance(reference, SphericalCoordinates)
    assert reference.size == 0
    new_reference = SphericalCoordinates([30, 45], unit='degree')
    source.reference = new_reference
    assert source.reference is new_reference
    with pytest.raises(ValueError) as err:
        source.reference = 1
    assert "Reference coordinates must be" in str(err.value)
    source.grid = None
    assert source.reference is None
    source.reference = new_reference
    assert source.reference is None


def test_shape(basic_source):
    assert basic_source.shape == (0, 0)


def test_size_x(basic_source):
    assert basic_source.size_x == 0


def test_size_y(basic_source):
    assert basic_source.size_y == 0


def test_get_memory_footprint(basic_source):
    assert basic_source.get_memory_footprint(5) == 15


def test_get_reduction_footprint(basic_source):
    assert basic_source.get_reduction_footprint(5) == 35


def test_pixels(basic_source):
    assert basic_source.pixels() == 0


def test_is_adding_to_master(basic_source):
    assert not basic_source.is_adding_to_master()


def test_get_reference(basic_source):
    assert basic_source.get_reference() is basic_source.grid.reference


def test_reset_processing(basic_source):
    source = basic_source.copy()
    source.configuration.parse_key_value('smooth', '4.5')
    source.generation = 1
    source.integration_time = 1 * units.Unit('second')
    source.reset_processing()
    assert source.generation == 0 and source.integration_time == 0
    assert source.smoothing == 4.5 * arcsec


def test_is_valid(basic_source):
    assert not basic_source.is_valid()


def test_get_default_filename(basic_source):
    assert basic_source.get_default_file_name().endswith('test.fits')


def test_get_core_name(basic_source):
    source = basic_source.copy()
    source.configuration.parse_key_value('name', 'a_name.fits')
    assert source.get_core_name() == 'a_name'
    del source.configuration['name']
    assert source.get_core_name() == 'default_core_name'


def test_create_from(basic_source, populated_scan):
    source = basic_source.copy()
    source.use_input_shape = True
    source.allow_indexing = True
    source.configuration.parse_key_value('indexing', 'True')
    scans = [populated_scan]
    source.create_from(scans)
    assert source.input_data_shape == (135, 120)
    assert np.isclose(source.reference.ra,
                      266.415 * units.Unit('degree'), atol=1e-3)
    # Check indexing occurred
    assert source.index_shift_x == 8
    assert source.index_mask_y == 255


def test_set_size(basic_source, populated_scan):
    source = basic_source.copy()
    source.use_input_shape = True
    scans = [populated_scan]
    source.configuration.parse_key_value('grid', '3.0')
    source.configuration.parse_key_value('large', 'False')
    source.assign_scans(scans)
    projection = SphericalProjection.for_name('gnomonic')
    projection.set_reference(populated_scan.get_position_reference(
        'equatorial'))
    source.projection = projection
    source.set_size()
    assert source.grid.resolution.x == 3 * arcsec
    assert source.grid.resolution.y == 3 * arcsec
    assert np.allclose(source.grid.reference_index.coordinates,
                       [39.5, 44.5])
    assert source.shape == (90, 80)

    source.configuration.parse_key_value('grid', '3.0, 4.0')
    source.set_size()
    assert source.grid.resolution.x == 3 * arcsec
    assert source.grid.resolution.y == 4 * arcsec
    assert source.shape == (68, 80)

    source.configuration.parse_key_value('grid', '-3.0, 4.0')
    with pytest.raises(ValueError) as err:
        source.set_size()
    assert 'Negative image size' in str(err.value)

    source.configuration.parse_key_value('grid', '0.00001')
    with pytest.raises(ValueError) as err:
        source.set_size()
    assert 'Map too large' in str(err.value)

    del source.configuration['grid']
    source.set_size()
    assert source.grid.resolution.x == 2 * arcsec
    assert source.grid.resolution.y == 2 * arcsec
    assert source.shape == (135, 120)


def test_search_corners(basic_with_projection):
    source = basic_with_projection.copy()
    source.configuration.parse_key_value('map.size', '20x20')
    samples = source.scans[0][0].frames.sample_flag.copy()
    assert np.allclose(samples, 0)
    map_range = source.search_corners()
    assert np.any(source.scans[0][0].frames.sample_flag != 0)
    assert np.allclose(map_range.coordinates, [[-10, 10], [-10, 10]] * arcsec)

    source.scans[0][0].frames.sample_flag = samples

    map_range = source.search_corners(determine_scan_range=True)
    assert np.allclose(map_range.coordinates,
                       [[-26.42276159, 36.034133],
                        [-28.32867433, 28.31612061]] * arcsec, atol=1e-6)

    del source.configuration['map.size']
    source.reduction.parent_reduction = source.reduction
    map_range = source.search_corners()
    assert np.allclose(map_range.coordinates,
                       [[-26.42276159, 36.034133],
                        [-28.32867433, 28.31612061]] * arcsec, atol=1e-6)


def test_parallel_safe_integration_range(basic_with_projection):
    source = basic_with_projection.copy()
    integrations = source.scans[0].integrations
    projection = source.projection
    args = integrations, projection
    integration_number = 0
    x0, x1, y0, y1 = AstroModel2D.parallel_safe_integration_range(
        args, integration_number)
    assert np.isclose(x0, -117.99752558 * arcsec, atol=1e-6)
    assert np.isclose(x1, 117.9864927 * arcsec, atol=1e-6)
    assert np.isclose(y0, -132.9945686 * arcsec, atol=1e-6)
    assert np.isclose(y1, 133.01127392 * arcsec, atol=1e-6)


def test_parallel_safe_flag_outside(basic_with_projection):
    source = basic_with_projection.copy()
    projection = source.projection
    integrations = source.scans[0].integrations
    map_range = Coordinate2D([[-10, 10], [-5, 5]], unit=arcsec)
    assert np.allclose(integrations[0].frames.sample_flag, 0)
    args = integrations, projection, map_range
    integration_number = 0
    AstroModel2D.parallel_safe_flag_outside(args, integration_number)
    assert not np.allclose(integrations[0].frames.sample_flag, 0)


def test_flag_outside(basic_with_projection):
    source = basic_with_projection.copy()
    projection = source.projection
    integration = source.scans[0][0]
    map_range = Coordinate2D([[-10, 10], [-5, 5]], unit=arcsec)

    channels = integration.channels.get_mapping_pixels()
    projector = AstroProjector(projection)
    offsets = integration.frames.project(channels.data.position, projector)
    offsets.change_unit('arcsec')

    assert np.allclose(integration.frames.sample_flag, 0)
    AstroModel2D.flag_outside(projection, integration, map_range)

    samples = integration.frames.sample_flag.copy()
    bad = samples != 0

    expected = offsets.x < map_range.x[0]
    expected |= offsets.x > map_range.x[1]
    expected |= offsets.y < map_range.y[0]
    expected |= offsets.y > map_range.y[1]
    assert np.allclose(expected, bad)


def test_index(basic_with_projection):
    source = basic_with_projection.copy()
    source.configuration.parse_key_value('indexing.check_memory', 'True')
    source.input_data_shape = (1, 1)
    frames = source.scans[0][0].frames
    frames.map_index.coordinates.fill(-1)
    assert np.allclose(frames.map_index.coordinates, -1)
    frames.source_index.fill(-1)
    assert np.allclose(frames.source_index, -1)
    source.configuration.parse_key_value('indexing.saturation', '0.0')
    source.index()
    assert np.allclose(frames.map_index.coordinates, -1)
    assert np.allclose(frames.source_index, -1)
    source.configuration.parse_key_value('indexing.saturation', '1.5')
    source.index()
    assert not np.allclose(frames.map_index.coordinates, -1)
    assert not np.allclose(frames.source_index, -1)


def test_create_lookup(basic_with_projection, capsys):
    source = basic_with_projection.copy()
    source.set_size()
    integration = source.scans[0][0]
    integration.frames.source_index = None
    integration.frames.map_index = None
    source.create_lookup(integration)
    assert integration.frames.map_index[0, 0].x == 66
    assert integration.frames.map_index[0, 0].y == 69
    assert source.index_shift_x == 8
    assert source.index_mask_y == 255
    assert integration.frames.source_index[0, 0] == 16965
    source.grid.reference_index.subtract(Coordinate2D([1, 1]))
    source.create_lookup(integration)
    assert "5 samples have bad map indices" in capsys.readouterr().err


def test_pixel_index_to_source_index(initialized_source):
    source = initialized_source.copy()
    pixel_indices = np.arange(10).reshape((2, 5))
    source_indices = source.pixel_index_to_source_index(pixel_indices)
    assert np.allclose(source_indices, [5, 262, 519, 776, 1033])

    pixel_indices = np.full((2, 3), -1)
    source_indices = source.pixel_index_to_source_index(pixel_indices)
    assert np.allclose(source_indices, -1)


def test_source_index_to_pixel_index(initialized_source):
    source = initialized_source.copy()
    source_indices = np.asarray([5, 262, 519, 776, 1033])
    pixel_indices = source.source_index_to_pixel_index(source_indices)
    assert np.allclose(pixel_indices, np.arange(10).reshape((2, 5)))
    source_indices = np.full(3, -1)
    pixel_indices = source.source_index_to_pixel_index(source_indices)
    assert np.allclose(pixel_indices, -1)


def test_find_outliers(initialized_source):
    source = initialized_source.copy()
    scan = source.scans[0]
    scan2 = deepcopy(scan)
    scan3 = deepcopy(scan)
    equatorial = scan3.equatorial.copy()
    equatorial.subtract(Coordinate2D([1, 1], unit='degree'))
    scan3.equatorial = equatorial
    source.scans = [scan, scan2, scan3]
    max_distance = 1 * units.Unit('arcmin')
    outliers = source.find_outliers(max_distance)
    assert len(outliers) == 1 and outliers[0] is scan3
    source.scans = [scan]
    assert source.find_outliers(max_distance) == []


def test_find_slewing(initialized_source):
    source = initialized_source.copy()
    slew_scans = source.find_slewing(1 * units.Unit('degree'))
    assert slew_scans == []
    slew_scans = source.find_slewing(1 * units.Unit('arcsec'))
    assert slew_scans == source.scans


def test_add_integration(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    integration.nefd = np.nan
    integration.comments = []
    source.add_integration(integration)
    assert np.isclose(integration.nefd, 0.316228, atol=1e-6)
    assert integration.comments == [
        'Map', '3.16e-01 beam / Jy', '[C~1.00]', ' ']

    source.id = 'foo'
    integration.channels.data.weight *= 0
    integration.nefd = np.nan
    integration.comments = []
    source.add_integration(integration)
    assert integration.comments == ['Map.foo', 'inf beam / Jy', '[C~inf]', ' ']

    integration.configuration.parse_key_value('mappingpixels', '1000')
    integration.comments = []
    source.add_integration(integration)
    assert integration.comments == ['Map.foo', '(!ch)']


def test_add_frames_from_integration(initialized_source, capsys):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    pixels = integration.channels.get_mapping_pixels(keep_flag=0)
    source_gains = integration.channels.get_source_gains()

    integration.frames.map_index = None
    mapping_frames = source.add_frames_from_integration(
        integration, pixels, source_gains)
    assert mapping_frames == 999
    assert 'merge accumulate' in capsys.readouterr().out

    source.add_to_master = True
    mapping_frames = source.add_frames_from_integration(
        integration, pixels, source_gains)
    assert mapping_frames == 999
    assert 'merge accumulate' not in capsys.readouterr().out


def test_add_pixels_from_integration(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    pixels = integration.channels.get_mapping_pixels()
    source_gains = integration.channels.get_source_gains(True)
    assert source.add_pixels_from_integration(
        integration, pixels, source_gains, None) == 999


def test_get_sample_points(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    frames = integration.frames
    pixels = integration.channels.get_mapping_pixels()
    frame_gains = np.full(frames.size, 0.5)
    source_gains = np.full(pixels.size, 3.0)
    n, data, gains, weights, indices = source.get_sample_points(
        frames, pixels, frame_gains, source_gains)
    assert n == 1100
    assert np.allclose(frames.data, data)
    assert gains.shape == (1100, 121) and np.allclose(gains, 1.5)
    assert np.allclose(frames.map_index.coordinates, indices)


def test_set_sync_gains(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    assert integration.source_sync_gain is None
    pixels = integration.channels.get_mapping_pixels()[:10]
    source_gains = np.full(integration.channels.size, 0.5)
    source.set_sync_gains(integration, pixels, source_gains)
    assert isinstance(integration.source_sync_gain, np.ndarray)
    sg = integration.source_sync_gain
    assert sg.shape == (121,)
    assert np.allclose(sg[:10], 0.5)
    assert np.allclose(sg[10:], 0)


def test_sync_integration(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    integration.scan.source_points = 1
    source.configuration.parse_key_value('source.coupling', 'True')
    integration.configuration.parse_key_value('crushbugs', 'True')

    source.sync_integration(integration)
    assert isinstance(integration.source_sync_gain, np.ndarray)
    assert integration.source_sync_gain.shape == (121,)
    assert np.allclose(integration.source_sync_gain, 1)
    parms = integration.get_dependents('source')
    assert np.allclose(parms.for_frame, 0.00090909, atol=1e-6)
    assert np.allclose(parms.for_channel[:-1], 0)
    assert np.isclose(parms.for_channel[-1], 0.00826446, atol=1e-6)

    del integration.dependents['source']
    integration.configuration.parse_key_value('crushbugs', 'False')

    source.sync_integration(integration)
    parms = integration.get_dependents('source')
    assert np.allclose(parms.for_frame, 0.00090909, atol=1e-6)
    assert np.allclose(parms.for_channel, 0.00826446, atol=1e-6)


def test_sync_pixels(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    pixels = integration.channels.get_mapping_pixels()
    integration.frames.map_index = None
    integration.source_sync_gain = None
    source_gains = np.ones(integration.channels.size, dtype=float)
    source.sync_pixels(integration, pixels, source_gains)
    assert isinstance(integration.source_sync_gain, np.ndarray)
    assert np.allclose(integration.source_sync_gain, 0)
    assert integration.source_sync_gain.shape == (121,)
    assert isinstance(integration.frames.map_index, Coordinate2D)


def test_get_table_entry(initialized_source):
    source = initialized_source.copy()
    source.smoothing = None
    assert source.get_table_entry('smooth') is None
    source.smoothing = 1 * units.Unit('degree')
    smooth = source.get_table_entry('smooth')
    assert isinstance(smooth, units.Quantity)
    assert smooth.unit == 'arcsec' and smooth.value == 3600
    assert source.get_table_entry('system') == 'EQ'
    source.grid.projection.reference = None
    assert source.get_table_entry('system') is None
    source.grid = None
    assert source.get_table_entry('system') is None
    assert source.get_table_entry('foo') is None


def test_write(initialized_source, tmpdir, capsys):
    source = initialized_source.copy()
    path = tmpdir.mkdir('testing_write')
    intermediate = path.join('intermediate..fits')
    with open(intermediate, 'w') as f:
        f.write('hello')
    assert os.path.isfile(intermediate)
    source.write(path)
    assert not os.path.isfile(intermediate)
    assert 'Source is empty' in capsys.readouterr().err
    source.id = 'foo'

    filename = path.join('default_core_name.foo.fits')
    with open(filename, 'w') as f:
        f.write('hello')
    source.write(path)
    assert not os.path.isfile(filename)
    assert 'Source foo is empty' in capsys.readouterr().err

    source.set_empty = False
    source.configuration.parse_key_value('write.png', 'True')
    source.write(path)


def test_write_png(obs2d, initialized_source):
    source = initialized_source.copy()
    source.do_png_write = False
    map_2d = obs2d.copy()

    smoothed = map_2d.copy()
    smoothed.smooth_to(5 * arcsec)

    file_name = 'test_write_png.fits'

    source.configuration.parse_key_value('write.png', 'False')
    source.write_png(map_2d, file_name)
    assert source.png_data is None

    source.configuration.parse_key_value('write.png', 'True')

    source.write_png(None, file_name)
    assert source.png_data is None

    source.configuration.parse_key_value('write.png.smooth', 'halfbeam')
    source.configuration.parse_key_value('write.png.crop', 'auto')
    source.configuration.parse_key_value('write.png.plane', 'image')

    source.write_png(map_2d, file_name)
    assert np.allclose(source.png_data[0].data, smoothed.data)
    assert source.png_data[1] == 'test_write_png.fits'

    map_2d = obs2d.copy()
    del source.configuration['write.png.plane']
    source.write_png(map_2d, file_name)
    assert np.allclose(source.png_data[0].data, smoothed.data)
    assert source.png_data[1] == 'test_write_png.fits'

    map_2d = obs2d.copy()
    del source.configuration['write.png.smooth']
    source.configuration.parse_key_value('write.png.plane', 's2n')
    source.write_png(map_2d, file_name)
    assert map_2d.get_significance().get_image() == source.png_data[0]
    source.configuration.parse_key_value('write.png.plane', 'time')
    source.write_png(map_2d, file_name)
    assert map_2d.get_exposures().get_image() == source.png_data[0]
    source.configuration.parse_key_value('write.png.plane', 'rms')
    source.write_png(map_2d, file_name)
    assert map_2d.get_noise().get_image() == source.png_data[0]
    source.configuration.parse_key_value('write.png.plane', 'weight')
    source.write_png(map_2d, file_name)
    assert map_2d.get_weights().get_image() == source.png_data[0]

    del source.configuration['write.png.plane']
    source.configuration.parse_key_value('write.png.crop', '1')  # arcsec
    m = obs2d.copy()
    source.write_png(m, file_name)
    assert source.png_data[0].shape == (2, 2)

    m = obs2d.copy()
    source.configuration.parse_key_value('write.png.crop', '1,2')  # arcsec
    source.write_png(m, file_name)
    assert source.png_data[0].shape == (3, 2)  # y, x

    m = obs2d.copy()
    source.configuration.parse_key_value('write.png.crop', '-1,-2,3')
    source.write_png(m, file_name)
    assert source.png_data[0].shape == (3, 3)  # y, x

    m = obs2d.copy()
    source.configuration.parse_key_value('write.png.crop', '-1,-2,3,4')
    source.write_png(m, file_name)
    assert source.png_data[0].shape == (4, 3)


@pytest.mark.skipif(not have_matplotlib, reason='matplotlib not installed')
def test_write_image_png(obs2d, initialized_source, tmpdir):
    source = initialized_source.copy()
    source.do_png_write = True
    map_2d = obs2d.copy()
    map_2d.smooth_to(10 * arcsec)
    image = map_2d.get_image()
    source.configuration.parse_key_value('write.png.size', '200x250')
    source.configuration.parse_key_value('write.png.crop', 'auto')
    del source.configuration['write.png.crop']
    source.configuration.parse_key_value('write.png.color', 'gray')
    path = tmpdir.mkdir('test_write_image_png')
    file_name = str(path.join('filename'))
    png_file = f'{file_name}.png'
    source.write_image_png(image, file_name)
    assert os.path.isfile(png_file)
    png = imageio.imread(png_file, format='png')
    assert png.shape == (250, 200, 4)

    os.remove(png_file)
    del source.configuration['write.png.color']
    source.write_image_png(image, png_file)
    assert os.path.isfile(png_file)
    png = imageio.imread(png_file, format='png')
    assert png.shape == (250, 200, 4)


def test_get_smoothing(initialized_source):
    source = initialized_source.copy()
    source.configuration.parse_key_value('smooth.optimal', '4.5')
    assert source.get_smoothing('beam') == 10 * arcsec
    assert source.get_smoothing('halfbeam') == 5 * arcsec
    assert source.get_smoothing('2/3beam') == 10 / 1.5 * arcsec
    assert source.get_smoothing('minimal') == 3 * arcsec
    assert source.get_smoothing('optimal') == 4.5 * arcsec
    del source.configuration['smooth.optimal']
    assert source.get_smoothing('optimal') == 10 * arcsec
    assert source.get_smoothing('3.5') == 3.5 * arcsec
    assert source.get_smoothing(3.5) == 3.5 * arcsec

    pix_fwhm = source.get_pixelization_smoothing()
    assert source.get_smoothing(0.0) == pix_fwhm
    assert source.get_smoothing('a') == pix_fwhm
    assert source.get_smoothing('None') == pix_fwhm


def test_set_smoothing(initialized_source):
    source = initialized_source.copy()
    source.set_smoothing(3.5 * arcsec)
    assert source.smoothing == 3.5 * arcsec


def test_update_smoothing(initialized_source):
    source = initialized_source.copy()
    source.configuration.parse_key_value('smooth', '4.5')
    source.update_smoothing()
    assert source.smoothing == 4.5 * arcsec
    del source.configuration['smooth']
    source.update_smoothing()
    assert source.smoothing == 4.5 * arcsec  # No change


def test_get_requested_smoothing(initialized_source):
    source = initialized_source.copy()
    assert source.get_requested_smoothing() == 0 * arcsec
    assert source.get_requested_smoothing('beam') == 10 * arcsec


def test_get_pixelization_smoothing(initialized_source):
    source = initialized_source.copy()
    assert source.grid.get_pixel_area() == 4 * units.Unit('arcsec2')
    assert np.isclose(source.get_pixelization_smoothing(),
                      1.87887456 * arcsec, atol=1e-6)


def test_get_point_size(initialized_source):
    source = initialized_source.copy()
    source.configuration.parse_key_value('smooth', 'halfbeam')
    assert np.isclose(source.get_point_size(),
                      np.sqrt(125) * arcsec)


def test_get_source_size(initialized_source):
    source = initialized_source.copy()
    source.configuration.parse_key_value('smooth', 'halfbeam')
    assert np.isclose(source.get_source_size(),
                      np.sqrt(125) * arcsec)
