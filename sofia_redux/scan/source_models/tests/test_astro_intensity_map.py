# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.flags.array_flags import ArrayFlags
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
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
    source = AstroIntensityMap(example_reduction.info,
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
    return source


@pytest.fixture
def data_source(initialized_source):
    source = initialized_source.copy()
    o = source.get_data()
    data = o.data
    data[50, 50] = 1.0
    o.data = data
    o.weight.data = np.full_like(data, 16.0)
    o.exposure.data = np.full_like(data, 0.5)
    return source


def test_init(example_reduction):
    reduction = example_reduction
    info = reduction.info
    source = AstroIntensityMap(info, reduction=reduction)
    assert source.info is info and source.reduction is reduction
    assert isinstance(source.map, Observation2D)
    assert source.base is None


def test_copy(initialized_source):
    source = initialized_source.copy()
    source.base = np.arange(10)
    source2 = source.copy(with_contents=False)
    assert source2.map != source.map
    assert source2.base is source.base
    source2 = source.copy(with_contents=True)
    assert source2.map == source.map
    assert source2.base is source.base
    assert source2.grid is not source.grid
    assert source2.grid == source.grid


def test_clear_all_memory(data_source):
    source = data_source.copy()
    source.clear_all_memory()
    assert source.map is None and source.base is None


def test_referenced_attributes(basic_source):
    assert 'base' in basic_source.referenced_attributes


def test_flagspace(initialized_source):
    source = initialized_source.copy()
    assert source.flagspace == ArrayFlags
    source.map = None
    assert source.flagspace == ArrayFlags


def test_shape(initialized_source):
    source = initialized_source.copy()
    assert source.shape == (135, 120)
    source.map = None
    assert source.shape == ()
    source.shape = (3, 5)
    assert source.shape == ()
    source = initialized_source.copy()
    source.shape = (100, 100)
    assert source.shape == (100, 100)


def test_set_info(basic_source):
    source = basic_source.copy()
    info = source.info.copy()
    info.parent = None
    assert info is not source.info
    source.map.fits_properties.instrument_name = None
    source.set_info(info)
    assert source.info is info
    assert info.parent is source
    assert source.map.fits_properties.instrument_name == 'example'


def test_get_jansky_unit(initialized_source):
    source = initialized_source.copy()
    assert np.isclose(source.get_jansky_unit(),
                      2.6632636e-9 * units.Unit('Jy'),
                      rtol=1e-3)
    source.map.underlying_beam = None
    assert source.get_jansky_unit() == 0 * units.Unit('Jy')


def test_add_model_data(data_source):
    source = data_source.copy()
    source_model = source.copy()
    source.add_model_data(source_model)
    assert source.map.data[50, 50] == 17


def test_merge_accumulate(data_source):
    source = data_source.copy()
    source2 = source.copy()
    source.merge_accumulate(source2)
    assert source.map.data[50, 50] == 2


def test_stand_along(data_source):
    source = data_source.copy()
    source.base = None
    source.stand_alone()
    assert source.base.shape == (135, 120)


def test_create_map(data_source):
    source = data_source.copy()
    map1 = source.map
    source.map = None
    source.reduction.max_jobs = 2
    source.create_map()
    assert isinstance(source.map, Observation2D)
    assert source.map.grid == map1.grid
    assert source.map.validating_flags.name == 'DISCARD'
    assert 'ct' in source.map.local_units
    assert source.map.display_grid_unit == 1 * arcsec
    assert source.map.fits_properties.instrument_name == 'example'
    assert '(c)' in source.map.fits_properties.copyright
    assert source.map.parallelism == 2
    assert source.map.fits_properties.creator == 'Reduction'


def test_create_from(data_source):
    source = data_source.copy()
    scans = source.scans
    source.create_from(scans)
    assert source.map.fits_properties.object_name == 'Simulation'
    assert source.map.underlying_beam.fwhm == 10 * arcsec
    assert np.isclose(source.map.local_units['K'],
                      np.nan * units.Unit('K'), equal_nan=True)
    assert np.isclose(source.map.local_units['Jy'],
                      2.6632636e-09 * units.Unit('Jy'), rtol=1e-3)
    assert source.base.shape == (135, 120)


def test_post_process_scan(data_source):
    source = data_source.copy()
    scan = source.scans[0]
    source.configuration.parse_key_value('pointing.suggest', 'True')
    source.configuration.parse_key_value('smooth.optimal', '5.0')
    source.configuration.parse_key_value('pointing.exposureclip', '0.1')
    source.configuration.parse_key_value('pointing.radius', '50.0')  # arcsec
    source.configuration.parse_key_value('pointing.method', 'centroid')
    source.configuration.parse_key_value('pointing.lsq', 'False')
    source.configuration.parse_key_value('pointing.significance', '0.1')
    reference_index = Coordinate2D([51, 51])  # 1 pixel off
    source.map.grid.set_reference_index(reference_index)
    source.post_process_scan(scan)

    x = scan.pointing.x_mean
    y = scan.pointing.y_mean
    assert np.isclose(x, -266.4156441 * units.Unit('degree'))
    assert np.isclose(y, -29.00666667 * units.Unit('degree'))

    source = data_source.copy()
    scan.pointing = None
    del source.configuration['smooth.optimal']
    source.post_process_scan(scan)
    x = scan.pointing.x_mean
    y = scan.pointing.y_mean
    assert np.isclose(x, -266.4156441 * units.Unit('degree'))
    assert np.isclose(y, -29.00666667 * units.Unit('degree'))

    scan.pointing = None
    source.clear_content()
    source.post_process_scan(scan)
    assert scan.pointing is None


def test_get_peak_index(data_source):
    source = data_source.copy()
    source.configuration.parse_key_value('source.sign', '+')
    index = source.get_peak_index()
    assert np.allclose(index.coordinates, 50)
    source.configuration.parse_key_value('source.sign', '-')
    index = source.get_peak_index()
    assert np.allclose(index.coordinates, 0)
    del source.configuration['source.sign']
    index = source.get_peak_index()
    assert np.allclose(index.coordinates, 50)


def test_get_peak_coords(data_source):
    source = data_source.copy()
    c = source.get_peak_coords()
    assert np.allclose(c.coordinates,
                       [-266.4210441, -29.0152776] * units.Unit('degree'),
                       atol=1e-6)


def test_get_peak_source(data_source):
    source = data_source.copy()
    data = source.map.data
    data.fill(0)
    data[68, 60] = 1
    source.map.data = data
    source.map.smooth_to(10 * arcsec)
    source.configuration.parse_key_value('pointing.method', 'centroid')
    source.configuration.parse_key_value('pointing.lsq', 'False')
    source.configuration.parse_key_value('pointing.significance', '0.0')
    peak_source = source.get_peak_source()
    assert np.isclose(peak_source.x_mean, -266.414691 * units.Unit('degree'),
                      atol=1e-6)
    assert np.isclose(peak_source.y_mean, -29.005278 * units.Unit('degree'),
                      atol=1e-6)
    assert np.isclose(peak_source.fwhm, 0 * arcsec)  # deconvolved

    source.configuration.parse_key_value('pointing.significance', '100000')
    peak_source = source.get_peak_source()
    assert peak_source is None
    source.configuration.parse_key_value('pointing.significance', '0.0')

    source.map.smoothing_beam.fwhm = 0 * arcsec
    source.configuration.parse_key_value('pointing.lsq', 'True')
    peak_source = source.get_peak_source()
    assert np.isclose(peak_source.x_mean, -266.414691 * units.Unit('degree'),
                      atol=1e-6)
    assert np.isclose(peak_source.y_mean, -29.0052778 * units.Unit('degree'),
                      atol=1e-6)
    assert np.isclose(peak_source.fwhm, 10 * arcsec)  # deconvolved

    source.map.data *= 0
    peak_source = source.get_peak_source()
    assert peak_source is None

    source.configuration.parse_key_value('pointing.lsq', 'False')
    peak_source = source.get_peak_source()
    assert peak_source is None


def test_update_mask(data_source):
    source = data_source.copy()
    assert np.allclose(source.map.flag, 0)
    source.update_mask()
    flag_value = source.FLAG_MASK.value
    assert np.allclose(source.map.flag, flag_value)
    source.configuration.parse_key_value('source.sign', '+')
    source.update_mask(blanking_level=3)
    mask = np.full(source.map.shape, False)
    mask[50, 50] = True
    assert np.allclose(source.map.flag[mask], flag_value)
    assert np.allclose(source.map.flag[~mask], 0)
    source.configuration.parse_key_value('source.sign', '-')
    source.update_mask(blanking_level=3)
    assert np.allclose(source.map.flag, 0)


def test_merge_mask(data_source):
    source = data_source.copy()
    other_map = source.map.copy()
    flag_value = source.FLAG_MASK.value
    other_map.flag[30, 30] = flag_value
    assert np.allclose(source.map.flag, 0)
    source.merge_mask(other_map)
    mask = np.full(other_map.shape, False)
    mask[30, 30] = True
    assert np.allclose(source.map.flag[mask], flag_value)
    assert np.allclose(source.map.flag[~mask], 0)


def test_is_masked(data_source):
    source = data_source.copy()
    mask = np.full(source.map.shape, False)
    mask[2, 5:10] = True
    source.map.flag[mask] = source.FLAG_MASK.value
    assert np.allclose(source.is_masked(), mask)


def test_add_points(data_source):
    source = data_source.copy()
    integration = source.scans[0][0]
    frames = integration.frames
    pixels = integration.channels.get_mapping_pixels()
    source.map.fill(0)
    source.map.weight.fill(0)
    source.map.exposure.fill(0)
    frame_gains = np.full(frames.size, 0.5)
    source_gains = np.full(pixels.size, 0.6)
    frames.data.fill(1.0)
    n = source.add_points(frames, pixels, frame_gains, source_gains)
    assert n == 1100
    inds = np.arange(65, 67), np.arange(58, 60)
    assert np.allclose(source.map.data[inds], [21.3, 17.4])
    assert np.allclose(source.map.weight.data[inds], [6.39, 5.22])
    assert np.allclose(source.map.exposure.data[inds], [7.1, 5.8])
    pixels.info.instrument.sampling_interval = 0.2
    source.map.exposure.fill(0)
    source.add_points(frames, pixels, frame_gains, source_gains)
    assert np.allclose(source.map.exposure.data[inds], [14.2, 11.6])


def test_mask_samples(data_source):
    source = data_source.copy()
    assert np.allclose(source.map.flag, 0)
    source.map.flag.fill(source.mask_flag.value)
    assert source.is_masked().all()
    integration = source.scans[0][0]
    source.mask_samples()
    assert np.allclose(integration.frames.sample_flag,
                       integration.frames.flagspace.convert_flag(
                           'SAMPLE_SKIP').value)
    source.scans[0].integrations = None
    source.mask_samples()
    assert source.scans[0].integrations is None
    source.scans = None
    source.mask_samples()
    assert source.scans is None


def test_mask_integration_samples(data_source):
    source = data_source.copy()
    assert not source.is_masked().any()
    integration = source.scans[0][0]
    frames = integration.frames
    frames.source_index = None
    frames.map_index = None
    source.mask_integration_samples(integration)
    assert np.allclose(frames.sample_flag, 0)
    source.map.flag[50, 50] = source.mask_flag.value
    source.mask_integration_samples(integration)
    inds = np.nonzero(frames.sample_flag)
    assert np.allclose(inds[0],  # Frames
                       [27, 27, 28, 28, 102, 103, 177, 178, 422, 497, 498,
                        572, 573, 647, 648, 1052])
    assert np.allclose(inds[1],  # Channels
                       [106, 117, 30, 41, 115, 38, 113, 35, 31, 39, 117, 36,
                        115, 45, 113, 108])
    mask = np.full(integration.channels.size, True)
    mask[106] = False
    group = integration.channels.groups['obs-channels'][mask]

    integration.channels.groups['obs-channels'] = group
    frames.sample_flag *= 0

    source.mask_integration_samples(integration)
    inds2 = np.nonzero(frames.sample_flag)
    assert np.allclose(inds2[0], inds[0][1:])
    assert np.allclose(inds2[1], inds[1][1:])
    group.indices = np.arange(0)
    frames.sample_flag *= 0
    source.mask_integration_samples(integration)
    assert np.allclose(frames.sample_flag, 0)


def test_add_frames_from_integration(data_source):
    source = data_source.copy()
    integration = source.scans[0][0]
    source_gains = np.ones(integration.channels.size, dtype=float)
    pixels = integration.channels.get_mapping_pixels()
    source.map.fill(0)
    source.map.weight.fill(0)
    source.map.exposure.fill(0)
    integration.frames.data.fill(1)
    mapping_frames = source.add_frames_from_integration(
        integration, pixels, source_gains)
    assert mapping_frames == 1100
    inds = np.arange(65, 67), np.arange(58, 60)
    assert np.allclose(source.map.data[inds], [71, 58])
    assert np.allclose(source.map.weight.data[inds], [71, 58])
    assert np.allclose(source.map.exposure.data[inds],
                       [7.1, 5.8])
    assert np.isclose(source.integration_time, 110 * units.Unit('second'))


def test_sync_source_gains(data_source):
    source = data_source.copy()
    integration = source.scans[0][0]
    frames = integration.frames
    pixels = integration.channels.get_mapping_pixels()
    frame_gains = np.ones(frames.size, dtype=float)
    source_gains = np.ones(integration.channels.size, dtype=float)
    sync_gains = np.zeros(integration.channels.size, dtype=float)
    frames.data.fill(0.0)
    source.sync_source_gains(frames, pixels, frame_gains, source_gains,
                             sync_gains)
    inds = np.nonzero(frames.data)
    assert np.allclose(inds[0],  # Frames
                       [27, 27, 28, 28, 102, 103, 177, 178, 422, 497, 498,
                        572, 573, 647, 648, 1052])
    assert np.allclose(inds[1],  # Channels
                       [106, 117, 30, 41, 115, 38, 113, 35, 31, 39, 117, 36,
                        115, 45, 113, 108])
    assert np.allclose(frames.data[inds], -1)


def test_calculate_coupling(data_source):
    source = data_source.copy()
    integration = source.scans[0][0]
    pixels = integration.channels.get_mapping_pixels()
    sync_gains = np.zeros(integration.channels.size, dtype=float)
    integration.frames.map_index = None
    source_gains = np.ones(integration.frames.size, dtype=float)
    source.configuration.parse_key_value('source.coupling.s2n', '0:10')
    source.configuration.parse_key_value('source.coupling.range', '0.3:3.0')

    assert np.allclose(pixels.coupling, 1)
    source.calculate_coupling(integration, pixels, source_gains, sync_gains)
    mask = pixels.coupling == 0
    assert mask.sum() == 13
    assert np.allclose(pixels.coupling[~mask], 1.01444648, atol=1e-6)
    assert np.allclose(pixels.flag[mask],
                       pixels.flagspace.convert_flag('BLIND').value)

    pixels.coupling = np.ones(pixels.size, dtype=float)
    del source.configuration['source.coupling.s2n']
    pixels.flag = np.zeros(pixels.size, dtype=int)
    source.calculate_coupling(integration, pixels, source_gains, sync_gains)
    assert np.allclose(pixels.coupling, 1)


def test_process_final(data_source):
    source = data_source.copy()
    source.configuration.parse_key_value('extended', 'False')
    source.configuration.parse_key_value('deep', 'False')
    source.enable_level = True
    source.enable_weighting = True
    source.configuration.parse_key_value('regrid', '1.0')
    source.process_final()
    assert np.allclose(source.map.grid.resolution.coordinates, 1 * arcsec)
    assert source.map.shape == (270, 240)
    assert np.allclose(source.map.data[99:102, 99:102],
                       [[0.36057743, 0.60048094, 0.36057737],
                        [0.60048098, 1., 0.60048095],
                        [0.36057739, 0.6004809, 0.36057738]], atol=1e-6)


def test_get_table_entry(data_source):
    source = data_source.copy()
    assert source.get_table_entry('system') == 'EQ'
    source.grid.projection.reference = None
    assert source.get_table_entry('system') is None
    source.grid = None
    assert source.get_table_entry('system') is None
    assert source.get_table_entry('map.depth') == 16
    assert source.get_table_entry('foo') is None


def test_get_pixel_footprint(data_source):
    assert data_source.get_pixel_footprint() == 32


def test_base_footprint(data_source):
    assert data_source.base_footprint(1) == 8
    assert data_source.base_footprint(5) == 40


def test_set_data_shape(data_source):
    source = data_source.copy()
    source.set_data_shape((10, 12))
    assert source.map.shape == (10, 12)


def test_set_base(data_source):
    source = data_source.copy()
    source.base = None
    source.set_base()
    assert np.allclose(source.base.data, source.map.data)
    assert source.base.data is not source.map.data


def test_reset_processing(data_source):
    source = data_source.copy()
    source.generation = 1
    source.map.noise_rescale = 2.0
    source.reset_processing()
    assert source.generation == 0
    assert source.map.noise_rescale == 1.0


def test_covariant_points(data_source):
    source = data_source.copy()
    assert source.covariant_points() == 1.0


def test_get_map_2d(data_source):
    source = data_source.copy()
    assert source.get_map_2d() is source.map


def test_get_source_name(data_source):
    assert data_source.get_source_name() == 'Simulation'


def test_get_unit(data_source):
    assert data_source.get_unit() == 1 * units.Unit('count')


def test_get_data(data_source):
    assert data_source.get_data() is data_source.map


def test_add_base(data_source):
    source = data_source.copy()
    source.base.fill(1)
    d0 = source.map.data.copy()
    source.add_base()
    assert np.allclose(source.map.data, d0 + 1)


def test_smooth_to(data_source):
    source = data_source.copy()
    source.smooth_to(10 * arcsec)
    assert np.allclose(source.map.data[49:52, 49:52],
                       [[0.02907699, 0.03261946, 0.02907699],
                        [0.03261946, 0.03659351, 0.03261946],
                        [0.02907699, 0.03261946, 0.02907699]], atol=1e-6)


def test_filter_source(data_source):
    source = data_source.copy()
    source.smooth_to(10 * arcsec)
    source.filter_source(5 * arcsec, filter_blanking=None, use_fft=True)
    assert np.allclose(source.map.data[49:52, 49:52],
                       [[0.00432037, 0.00549613, 0.00432037],
                        [0.00549613, 0.00687721, 0.00549613],
                        [0.00432037, 0.00549613, 0.00432037]], atol=1e-6)
    assert np.isnan(source.map.filter_blanking)

    source = data_source.copy()
    source.map.data[100, 100] = 10.0
    source.smooth_to(10 * arcsec)
    source.filter_source(5 * arcsec, filter_blanking=1.0, use_fft=False)
    assert np.allclose(source.map.data[49:52, 49:52], 0)
    assert np.allclose(source.map.data[99:102, 99:102],
                       [[0.29076988, 0.32619458, 0.29076988],
                        [0.32619458, 0.36593509, 0.32619458],
                        [0.29076988, 0.32619458, 0.29076988]], atol=1e-6)
    assert source.map.filter_blanking == 1
    assert source.map.filter_fwhm == 5 * arcsec


def test_set_filtering(data_source):
    source = data_source.copy()
    source.set_filtering(12 * arcsec)
    assert source.map.filter_fwhm == 12 * arcsec


def test_reset_filtering(data_source):
    source = data_source.copy()
    source.set_filtering(10 * arcsec)
    source.reset_filtering()
    assert np.isnan(source.map.filter_fwhm)


def test_filter_beam_correct(data_source):
    source = data_source.copy()
    source.filter_beam_correct()
    assert source.map.correcting_fwhm == 10 * arcsec


def test_mem_correct(data_source):
    source = data_source.copy()
    source.mem_correct(0.1)
    assert np.isclose(source.map.data[50, 50], 0.96458483, atol=1e-6)


def test_get_clean_local_copy(data_source):
    source = data_source.copy()
    source.base.fill(1)
    new = source.get_clean_local_copy()
    assert new is not source and new.base is source.base
    new = source.get_clean_local_copy(full=True)
    assert new is not source and new.base is not source.base
