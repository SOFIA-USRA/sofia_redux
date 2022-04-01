# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import glob
import os
import pytest
import numpy as np

from sofia_redux.scan.flags.array_flags import ArrayFlags
from sofia_redux.scan.source_models.astro_data_2d import AstroData2D
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.reduction.reduction import Reduction


arcsec = units.Unit('arcsec')


@pytest.fixture
def example_reduction():
    return Reduction('example')


@pytest.fixture
def basic_source(example_reduction):
    source = AstroIntensityMap(example_reduction.info,
                               reduction=example_reduction)
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


def test_class():
    assert AstroData2D.FLAG_MASK.name == 'MASK'


def test_init(data_source):
    assert isinstance(data_source, AstroData2D)


def test_create_from(basic_source, populated_scan):
    source = basic_source.copy()
    scans = [populated_scan.copy()]
    source.configuration.parse_key_value('unit', 'Jy')
    source.create_from(scans)
    assert source.scans == scans
    assert source.get_data().unit == 1 * units.Unit('Jy')


def test_flagspace(basic_source):
    source = basic_source.copy()
    source.map = None
    assert source.flagspace == ArrayFlags


def test_mask_flag(basic_source):
    source = basic_source.copy()
    source.map = None
    assert source.mask_flag == ArrayFlags.flags.MASK


def test_get_weights(data_source):
    assert data_source.get_weights() == data_source.map.get_weights()


def test_get_noise(data_source):
    assert data_source.get_noise() == data_source.map.get_noise()


def test_get_significance(data_source):
    assert data_source.get_significance() == data_source.map.get_significance()


def test_get_exposures(data_source):
    assert data_source.get_exposures() == data_source.map.get_exposures()


def test_end_accumulation(data_source):
    source = data_source.copy()
    source.end_accumulation()
    assert np.isclose(source.get_data().data[50, 50], 0.0625)


def test_executor(data_source):
    source = data_source.copy()
    source.set_executor(1)
    assert source.get_executor() == 1


def test_parallel(data_source):
    source = data_source.copy()
    source.set_parallel(2)
    assert source.get_parallel() == 2
    source.no_parallel()
    assert source.get_parallel() == 0


def test_clear_content(data_source):
    source = data_source.copy()
    source.clear_content()
    assert np.allclose(source.get_data().data, 0)


def test_is_empty(data_source):
    source = data_source.copy()
    assert not source.is_empty()
    source.clear_content()
    assert source.is_empty()


def test_count_points(data_source):
    source = data_source.copy()
    assert source.count_points() == 16200
    source.clear_content()
    assert source.count_points() == 0


def test_get_chi2(data_source):
    source = data_source.copy()
    assert np.isclose(source.get_chi2(), 9.8765432e-4)
    assert source.get_chi2(robust=True) == 0


def test_smooth(data_source):
    source = data_source.copy()
    source.smoothing = 10 * arcsec
    source.smooth()
    assert np.isclose(source.get_data().data[50, 50], 0.0365935, atol=1e-6)


def test_filter(data_source):
    source = data_source.copy()
    data = source.get_data().data.copy()
    data[60, 60] = 11
    source.get_data().data = data.copy()
    source.configuration.parse_key_value('source.filter', 'True')
    source.configuration.parse_key_value('source.filter.blank', '10.0')
    source.configuration.parse_key_value('source.filter.type', 'fft')
    source.filter(allow_blanking=True)

    result = source.get_data().data
    assert np.allclose(result[49:52, 49:52],
                       [[-0.00128497, -0.00129021, -0.00128497],
                        [-0.00129021, 0.99870452, -0.00129021],
                        [-0.00128497, -0.00129021, -0.00128497]], atol=1e-6
                       )
    assert np.allclose(result[59:62, 59:62],
                       [[-6.70025825e-04, -6.20167133e-04, -5.69365200e-04],
                        [-6.20167133e-04, 1.09994260e+01, -5.26996976e-04],
                        [-5.69365200e-04, -5.26996976e-04, -4.83827218e-04]],
                       atol=1e-6)

    source.get_data().data = data.copy()
    source.configuration.parse_key_value('source.filter.type', 'convolution')
    source.filter(allow_blanking=False)

    result = source.get_data().data
    assert np.allclose(result[49:52, 49:52],
                       [[-0.00661084, -0.00709073, -0.00755165],
                        [-0.00709073, 0.99238708, -0.00811537],
                        [-0.00755165, -0.00811537, -0.0086587]], atol=1e-6
                       )
    assert np.allclose(result[59:62, 59:62],
                       [[-0.01481916, -0.01482903, -0.01471853],
                        [-0.01482903, 10.98515713, -0.01473589],
                        [-0.01471852, -0.01473588, -0.014633]],
                       atol=1e-6)

    source.get_data().data = data.copy()
    source.configuration.parse_key_value('source.filter', 'False')
    source.filter()
    assert np.allclose(source.get_data().data, data)


def test_get_filter_scale(data_source):
    source = data_source.copy()
    source.configuration.parse_key_value('source.filter.fwhm', '5.0')
    assert source.get_filter_scale() == 5 * arcsec
    del source.configuration['source.filter.fwhm']
    assert source.get_filter_scale() == 5 * source.get_source_size()


def test_process_scan(data_source, tmpdir):
    source = data_source.copy()
    scan = source.scans[0]
    path = str(tmpdir.mkdir('test_process_scan'))
    source.reduction.work_path = path
    source.configuration.parse_key_value('source.filter', 'True')
    source.configuration.parse_key_value('source.filter.type', 'fft')
    source.configuration.parse_key_value('source.despike', 'True')
    source.configuration.parse_key_value('weighting.scans', 'True')
    source.configuration.parse_key_value('weighting.scans.method', 'rms')
    source.configuration.parse_key_value('scanmaps', 'True')
    source.process_scan(scan)
    files = glob.glob(os.path.join(path, '*.fits'))
    assert len(files) == 1
    hdul = fits.open(files[0])
    assert np.isclose(hdul[0].data[50, 50], 0.062419, atol=1e-6)

    source.configuration.parse_key_value('scanmaps', 'False')
    assert np.isclose(scan.weight, 259704.7, atol=0.1)
    d = source.get_data().data
    d.fill(np.nan)
    source.get_data().data = d
    source.process_scan(scan)
    assert scan.weight == 0


def test_level(data_source):
    source = data_source.copy()
    d0 = source.get_data().data.copy()
    source.level(robust=False)
    assert np.isclose(source.get_data().data[50, 50], 0.99993827, atol=1e-6)
    source.level(robust=True)
    assert np.allclose(source.get_data().data, d0)


def test_process(data_source, tmpdir):
    source = data_source.copy()
    path = str(tmpdir.mkdir('test_process'))
    source.reduction.work_path = path
    source.enable_level = True
    source.enable_weighting = True
    source.enable_bias = True
    source.configuration.parse_key_value('source.despike', 'True')
    source.configuration.parse_key_value('source.filter', 'True')
    source.configuration.parse_key_value('weighting.scans', 'True')
    source.configuration.parse_key_value('source.redundancy', '1')
    source.configuration.parse_key_value('smooth', 'halfbeam')
    source.configuration.parse_key_value('smooth.external', 'False')
    source.configuration.parse_key_value('exposureclip', '0.1')
    source.configuration.parse_key_value('noiseclip', '1.1')
    source.configuration.parse_key_value('source.sign', '+')
    source.configuration.parse_key_value('clip', '100.0')
    source.configuration.parse_key_value('source.mem', 'True')
    source.configuration.parse_key_value('source.intermediates', 'True')
    source.configuration.parse_key_value('source.nosync', 'False')
    source.configuration.parse_key_value('blank', '30.0')
    configured_source = source.copy()
    source.process()
    assert source.process_brief == [
        '{level} ',
        '{despike} ',
        '{filter} ',
        '{1.00x}',
        '(check) ',
        '(smooth) ',
        '(filter) ',
        '(exposureclip) ',
        '(noiseclip) ',
        '(clip:100.0) ',
        '(MEM) ',
        'blank:30.0) ']
    files = glob.glob(os.path.join(path, '*.fits'))
    assert len(files) == 1 and files[0].endswith('intermediate.fits')
    hdul = fits.open(files[0])
    assert np.isnan(hdul[0].data).all()  # clipping
    hdul.close()
    os.remove(files[0])
    assert np.allclose(source.get_data().flag, 1)  # clipping

    source = configured_source.copy()
    source.configuration.parse_key_value('source.despike', 'False')
    source.configuration.parse_key_value('source.filter', 'False')
    del source.configuration['source.redundancy']
    del source.configuration['smooth']
    source.scans[0].weight = 0.0
    del source.configuration['exposureclip']
    del source.configuration['noiseclip']
    source.configuration.parse_key_value('source.sign', '-')
    del source.configuration['blank']
    source.configuration.parse_key_value('source.intermediates', 'False')
    source.process()
    files = glob.glob(os.path.join(path, '*.fits'))
    assert len(files) == 0
    assert source.process_brief == [
        '{level} ', '{inf}', '(clip:100.0) ', '(MEM) ']


def test_process_final(data_source):
    source = data_source.copy()
    source.get_data().history = ['foo']
    source.process_final()
    assert 'foo' not in source.get_data().history


def test_write_fits(data_source, tmpdir):
    source = data_source.copy()
    filename = str(tmpdir.mkdir('test_write_fits').join('filename.fits'))
    source.configuration.parse_key_value('write.source', 'False')
    source.hdul = None
    source.write_fits(filename)
    assert not os.path.isfile(filename)
    assert isinstance(source.hdul, fits.HDUList)
    source.configuration.parse_key_value('write.source', 'True')
    source.write_fits(filename)
    assert os.path.isfile(filename)
