# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import os
import pytest
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.source_models.pixel_map import PixelMap
from sofia_redux.scan.reduction.reduction import Reduction


arcsec = units.Unit('arcsec')


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
    source = PixelMap(example_reduction.info,
                      reduction=example_reduction)
    source.configuration.parse_key_value('indexing.check_memory', 'False')
    return source


@pytest.fixture
def initialized_source(basic_source, populated_scan):
    source = basic_source.copy()
    source.use_input_shape = True
    source.allow_indexing = True
    source.configuration.parse_key_value('indexing', 'True')
    source.create_from([populated_scan])
    source.reduction.available_reduction_jobs = 1
    return source


@pytest.fixture
def data_source(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    source.add_integration(integration)
    return source


@pytest.fixture
def short_source(data_source):
    source = data_source
    pixmaps = dict((i, source.pixel_maps[i]) for i in range(5))
    scan = source.scans[0]
    inds = np.arange(scan.channels.size)
    scan.channels.data.set_flags('BLIND', inds[5:])
    source.pixel_maps = pixmaps
    return source


def test_init(example_reduction):
    reduction = example_reduction
    info = reduction.info
    source = PixelMap(info, reduction=reduction)
    assert source.reduction is reduction
    assert source.info is reduction.info
    assert source.pixel_maps == {}
    assert isinstance(source.template, AstroIntensityMap)


def test_copy(data_source):
    source = data_source.copy(with_contents=True)
    assert len(source.pixel_maps) == 121
    assert source.pixel_maps.keys() == data_source.pixel_maps.keys()
    for v in source.pixel_maps.values():
        assert isinstance(v, AstroIntensityMap)
    source = data_source.copy(with_contents=False)
    assert source.pixel_maps == {}


def test_clear_all_memory(data_source):
    source = data_source.copy()
    source.clear_all_memory()
    assert source.pixel_maps == {} and source.template is None


def test_referenced_attributes(basic_source):
    assert 'pixel_maps' in basic_source.referenced_attributes


def test_shape(data_source):
    source = data_source
    assert source.shape == (122, 106)
    source.shape = (100, 100)
    assert source.shape == (100, 100)
    source.template = None
    assert source.shape == ()


def test_is_adding_to_master(basic_source):
    assert basic_source.is_adding_to_master()


def test_clear_process_brief(data_source):
    source = data_source
    source.clear_process_brief()
    assert source.process_brief == []
    for pixmap in source.pixel_maps.values():
        assert pixmap.process_brief == []


def test_create_from(basic_source, populated_scan):
    source = basic_source.copy()
    scans = [populated_scan.copy()]
    source.configuration.parse_key_value('indexing', 'True')
    source.allow_indexing = True
    source.use_input_shape = True
    source.create_from(scans)
    integration = scans[0][0]
    assert source.pixel_maps == {}
    assert integration.channels.data.independent.all()
    assert integration.channels.data.position.is_null().all()
    assert not np.allclose(integration.frames.map_index.coordinates, -1)


def test_reset_processing(data_source):
    source = data_source
    source.configuration.parse_key_value('smooth', '4.5')
    source.reset_processing()
    assert source.smoothing == 4.5 * arcsec
    for pixmap in source.pixel_maps.values():
        assert pixmap.smoothing == 4.5 * arcsec


def test_clear_content(data_source):
    source = data_source
    source.clear_content()
    for pixmap in source.pixel_maps.values():
        assert pixmap.is_empty()


def test_create_lookup(initialized_source, populated_scan, capsys):
    source = initialized_source
    integration = populated_scan[0]
    source.index_shift_x = 0
    source.index_mask_y = 0
    frames = integration.frames
    frames.source_index = None
    frames.map_index = None
    frames.equatorial.zero(0)  # Bad value in first frame
    source.create_lookup(integration)
    assert '1 frames have bad map indices' in capsys.readouterr().err
    assert isinstance(frames.map_index, Coordinate2D)
    assert isinstance(frames.source_index, np.ndarray)
    expected = frames.map_index.copy()
    source.create_lookup(integration)
    assert frames.map_index == expected


def test_add_model_data(data_source):
    source1 = data_source.copy()
    source2 = data_source.copy()
    source2.pixel_maps[0] = None
    source1.pixel_maps[1] = None
    source1.pixel_maps[2] = 1

    for k in source2.pixel_maps.keys():
        if k > 4:
            source2.pixel_maps[k] = None

    d0 = source1.pixel_maps[0].map.data.copy()
    d1 = source2.pixel_maps[1].map.data.copy()
    d3 = source1.pixel_maps[3].map.data.copy()

    source1.add_model_data(source2)
    assert np.allclose(source1.pixel_maps[0].map.data, d0)
    assert np.allclose(source1.pixel_maps[1].map.data, d1)
    assert source1.pixel_maps[2] == 1
    assert not np.allclose(source1.pixel_maps[3].map.data, d3)

    with pytest.raises(ValueError) as err:
        source1.add_model_data(1)
    assert "Cannot add" in str(err.value)


def test_add_points(initialized_source):
    source = initialized_source.copy()
    integration = source.scans[0][0]
    pixels = integration.channels.get_mapping_pixels()
    frames = integration.frames
    frame_gains = np.ones(frames.size, dtype=float)
    source_gains = np.ones(integration.channels.size, dtype=float)
    n = source.add_points(frames, pixels, frame_gains, source_gains)
    assert n == 1100
    assert len(source.pixel_maps) == 121
    p = source.pixel_maps[0]
    assert np.isclose(p.map.data[58, 46], 1.947716, atol=1e-6)
    expected = p.map.data.copy()
    source.pixel_maps = {}
    dt = pixels.info.instrument.sampling_interval
    pixels.info.instrument.sampling_interval = dt.decompose().value
    source.add_points(frames, pixels, frame_gains, source_gains)
    assert np.allclose(source.pixel_maps[0].map.data, expected)


def test_parallel_safe_add_points(initialized_source):
    source = initialized_source
    integration = initialized_source.scans[0][0]
    frames = integration.frames
    pixels = integration.channels.get_mapping_pixels()
    frame_gains = np.ones(frames.size, dtype=float)
    source_gains = np.ones(integration.channels.size, dtype=float)
    n, frame_data, sample_gains, sample_weights, sample_indices = (
        source.get_sample_points(frames, pixels, frame_gains, source_gains))
    dt = 0.1
    source.pixel_maps = {}
    args = (source.template, source.pixel_maps, pixels.fixed_index, frame_data,
            sample_gains, sample_weights, dt, sample_indices)
    pixel_number = 2
    fixed_index, pixmap = PixelMap.parallel_safe_add_points(args, pixel_number)
    assert fixed_index == 2
    assert isinstance(pixmap, AstroIntensityMap)
    assert np.isclose(pixmap.map.data[56, 47], 2.473776, atol=1e-6)


def test_calculate_coupling(basic_source):
    basic_source.calculate_coupling(None, None, None, None)


def test_count_points(data_source):
    assert data_source.count_points() == 110352


def test_covariance_points(data_source):
    assert data_source.covariant_points() == 1
    data_source.pixel_maps = {}
    assert data_source.covariant_points() == 1


def test_get_pixel_footprint(data_source):
    assert data_source.get_pixel_footprint() == 3872


def test_get_base_footprint(data_source):
    assert data_source.base_footprint(5) == 4840


def test_set_data_shape(data_source):
    source = data_source
    assert source.shape == (122, 106)
    source.set_data_shape((10, 10))
    assert source.shape == (10, 10)
    for pixmap in source.pixel_maps.values():
        assert pixmap.shape == (10, 10)


def test_sync_source_gains(data_source):
    source = data_source
    integration = source.scans[0][0]
    frames = integration.frames
    pixels = integration.channels.get_mapping_pixels()
    frame_gains = np.ones(frames.size, dtype=float)
    source_gains = np.ones(integration.channels.size, dtype=float)
    sync_gains = np.zeros(integration.channels.size, dtype=float)

    d0 = frames.data.copy()
    source.sync_source_gains(frames, pixels, frame_gains, source_gains,
                             sync_gains)
    # Check all pixels are updated
    different = np.any(d0 != frames.data, axis=0)
    assert different.all()


def test_parallel_safe_sync_source_gains(data_source):
    source = data_source
    integration = source.scans[0][0]
    frames = integration.frames
    pixels = integration.channels.get_mapping_pixels()
    frame_gains = np.ones(frames.size, dtype=float)
    source_gains = np.ones(integration.channels.size, dtype=float)
    sync_gains = np.zeros(integration.channels.size, dtype=float)
    args = (source.pixel_maps, frames, pixels, frame_gains, source_gains,
            sync_gains)
    pixel_number = 2
    d0 = frames.data.copy()
    d2 = PixelMap.parallel_safe_sync_source_gains(args, pixel_number)
    diff = d0[:, 2] - d2
    inds = np.nonzero(np.abs(diff) > 2)[0]
    assert np.allclose(inds, [249, 722, 797])
    assert np.allclose(d2[inds], [-2.47377644, -2.47377646, -2.47377646],
                       atol=1e-6)


def test_set_base(data_source):
    source = data_source
    source.set_base()
    for pixmap in source.pixel_maps.values():
        assert pixmap.base.data.any()


def test_process_scan(data_source):
    source = data_source
    scan = source.scans[0]
    d0 = source.pixel_maps[0].map.data.copy()
    d1 = source.pixel_maps[1].map.data.copy()
    source.process_scan(scan)
    assert not np.allclose(d0, source.pixel_maps[0].map.data)
    assert not np.allclose(d1, source.pixel_maps[1].map.data)


def test_write(short_source, tmpdir):
    source = short_source
    path = str(tmpdir.mkdir('test_write'))
    source.configuration.parse_key_value('pixelmap.writemaps', 'False')
    source.write(path)
    files = os.listdir(path)
    assert len(files) == 1 and files[0] == 'Simulation.Simulation.1.rcp'
    source.configuration.parse_key_value('write.png', 'False')
    source.configuration.parse_key_value('pixelmap.writemaps', '0,1')
    source.write(path)
    files = os.listdir(path)
    assert len(files) == 3
    assert 'Simulation.Simulation.1.rcp' in files
    assert 'Simulation.Simulation.1.0.fits' in files
    assert 'Simulation.Simulation.1.1.fits' in files
    source.configuration.parse_key_value('pixelmap.writemaps', 'True')
    source.write(path)
    files = os.listdir(path)
    assert len(files) == 6
    assert 'Simulation.Simulation.1.rcp' in files
    for i in range(5):
        assert f'Simulation.Simulation.1.{i}.fits' in files


def test_write_fits(short_source, tmpdir):
    filename = str(tmpdir.mkdir('test_write_fits').join('foo.fits'))
    short_source.write_fits('foo.fits')
    assert not os.path.isfile(filename)


def test_process(short_source):
    source = short_source
    source.configuration.parse_key_value('pixelmap.process', 'False')
    source.process()
    assert source.generation == 5
    source.configuration.parse_key_value('pixelmap.process', 'True')
    source.process()
    for pixmap in source.pixel_maps.values():
        assert pixmap.generation == 1


def test_calculate_pixel_data(short_source):
    source = short_source
    source.configuration.parse_key_value('pointing.reduce_degrees', 'False')
    for pixmap in source.pixel_maps.values():
        pixmap.configuration.parse_key_value('pointing.significance', '0.0')

    mapping_pixels = source.scans[0].channels.get_mapping_pixels(keep_flag=0)
    positions = mapping_pixels.position.copy()
    coupling = mapping_pixels.coupling.copy()
    source.calculate_pixel_data()
    # No update - all fits fail
    assert mapping_pixels.position == positions
    assert np.allclose(mapping_pixels.coupling, coupling)

    source.configuration.parse_key_value('pointing.reduce_degrees', 'True')
    source.calculate_pixel_data()
    assert not mapping_pixels.position == positions
    assert not np.allclose(mapping_pixels.coupling, coupling)
    assert np.allclose(
        mapping_pixels.position.coordinates,
        [[12.84525247, 11.18817902, 10.76833087, 4.42720224, 2.99269552],
         [4.61873447, 8.3826164, 9.01247487, 3.93474032, 4.00953595]] * arcsec,
        atol=2)


def test_parallel_safe_calculate_pixel_data(short_source):
    source = short_source
    smooth = False
    point_size = source.info.instrument.get_point_size()
    degree = 3
    reduce_degrees = False
    args = source.pixel_maps, smooth, point_size, degree, reduce_degrees
    pixel_number = 0

    pix_maps = args[0]
    for pixmap in pix_maps.values():
        pixmap.configuration.parse_key_value('pointing.significance', '0.0')

    peak = PixelMap.parallel_safe_calculate_pixel_data(args, pixel_number)
    assert peak is None

    smooth = True
    reduce_degrees = True
    args = source.pixel_maps, smooth, point_size, degree, reduce_degrees
    peak = PixelMap.parallel_safe_calculate_pixel_data(args, pixel_number)
    assert np.isclose(peak.peak, 1.25, atol=0.05)

    pix_maps[0] = None
    peak = PixelMap.parallel_safe_calculate_pixel_data(args, pixel_number)
    assert peak is None


def test_write_pixel_data(short_source, tmpdir):
    path = str(tmpdir.mkdir('test_write_pixel_data'))
    source = short_source
    source.reduction.work_path = path
    source.write_pixel_data()
    files = os.listdir(path)
    assert len(files) == 1
    assert files[0] == 'Simulation.Simulation.1.rcp'
    filename = os.path.join(path, files[0])

    with open(filename, 'r') as f:
        lines = f.readlines()
    indices = [int(x.split()[0]) for x in lines
               if (not x.startswith('#') and x.strip() != '')]
    assert indices == [0, 1, 2, 3, 4]


def test_parallel(short_source):
    source = short_source
    source.set_parallel(7)
    for pixmap in source.pixel_maps.values():
        assert pixmap.map.parallelism == 7

    assert source.get_parallel() == 7
    source.no_parallel()
    assert source.get_parallel() == 0
    source.pixel_maps = {}
    assert source.get_parallel() == 1


def test_merge_accumulate(short_source):
    source = short_source
    with pytest.raises(ValueError) as err:
        source.merge_accumulate(None)
    assert 'Cannot add None' in str(err.value)

    maps = [x.map.copy() for x in source.pixel_maps.values()]
    source.merge_accumulate(source)
    maps2 = [x.map.copy() for x in source.pixel_maps.values()]
    for i in range(len(maps)):
        assert maps[i] != maps2[i]
        assert np.allclose(maps2[i].data - maps[i].data, maps[i].data)


def test_parallel_safe_merge_accumulate(short_source):
    source = short_source
    maps = source.pixel_maps
    other_maps = maps.copy()
    other_maps[0] = other_maps[1]
    other_maps[1] = None
    args = maps, other_maps

    expected = maps[0].map.data + maps[1].map.data
    index, new_map = PixelMap.parallel_safe_merge_accumulate(args, 0)
    assert index == 0 and np.allclose(new_map.map.data, expected)

    map1 = maps[1].map.copy()
    index, new_map = PixelMap.parallel_safe_merge_accumulate(args, 1)
    assert index == 1 and np.allclose(map1.data, new_map.map.data)


def test_get_source_name(initialized_source):
    assert initialized_source.get_source_name() == 'Simulation'


def test_get_unit(initialized_source):
    assert initialized_source.get_unit() == 1 * units.Unit('count')


def test_is_empty(short_source):
    source = short_source
    assert not source.is_empty()
    source.pixel_maps = {1: None}
    assert source.is_empty()
    source.pixel_maps = {}
    assert source.is_empty()


def test_process_final(short_source):
    source = short_source
    for pixel_map in source.pixel_maps.values():
        pixel_map.map.history = ['a message']
    source.process_final()
    # Just check that history has been cleared (standard operation)
    for pixel_map in source.pixel_maps.values():
        assert 'a message' not in pixel_map.map.history


def test_parallel_safe_process_final(short_source):
    source = short_source
    pixel_maps = source.pixel_maps
    pixel_maps[1] = None
    pixel_maps[0].map.history = ['delete me']
    index, pixel_map = PixelMap.parallel_safe_process_final((pixel_maps,), 0)
    assert 'delete me' not in pixel_map.map.history and index == 0
    index, pixel_map = PixelMap.parallel_safe_process_final((pixel_maps,), 1)
    assert pixel_map is None and index == 1


def test_get_map_2d(basic_source):
    assert basic_source.get_map_2d() is None
