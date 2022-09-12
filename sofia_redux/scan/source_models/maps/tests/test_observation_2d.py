# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D


arcsec = units.Unit('arcsec')


@pytest.fixture
def obs2d():
    data = np.zeros((10, 11), dtype=float)
    data[5, 6] = 1
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 0.5)
    o.filter_fwhm = 1 * arcsec
    o.grid.resolution = Coordinate2D([1, 1] * arcsec)
    return o


@pytest.fixture
def alternate_obs2d():
    data = np.zeros((12, 13), dtype=float)
    data[5, 6] = 1
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 0.5)
    o.filter_fwhm = 1 * arcsec
    o.grid.resolution = Coordinate2D([1, 1] * arcsec)
    return o


@pytest.fixture
def large_obs2d():
    data = np.zeros((101, 101), dtype=float)
    data[50, 50] = 1
    o = Observation2D(data=data, unit='Jy')
    o.weight.data = np.full(o.shape, 16.0)
    o.exposure.data = np.full(o.shape, 0.5)
    o.set_resolution(1 * units.Unit('arcsec'))
    return o


def test_init():
    data = np.ones((10, 11), dtype=float)
    o = Observation2D(data=None)
    assert o.data is None
    assert o.weight.size == 0 and o.exposure.size == 0
    assert o.noise_rescale == 1
    assert not o.is_zero_weight_valid
    assert o.weight_dtype == float
    assert o.dtype == float
    assert np.isnan(o.weight.blanking_value)
    assert np.isnan(o.exposure.blanking_value)

    o = Observation2D(data=data, unit='K', weight_dtype=int)
    assert o.weight.blanking_value == 0
    assert o.exposure.blanking_value == 0
    assert o.weight.dtype == int and o.exposure.dtype == int
    assert o.weight.shape == (10, 11)
    assert o.exposure.shape == (10, 11)
    assert o.unit == 1 * units.Unit('K')


def test_copy(obs2d):
    o = obs2d
    o2 = obs2d.copy()
    assert o == o2 and o is not o2


def test_eq(obs2d):
    o = obs2d.copy()
    o2 = o.copy()
    assert o2 == o
    o2.is_zero_weight_valid = True
    assert o2 != o
    o2 = o.copy()
    o2.weight = None
    assert o2 != o
    o2 = o.copy()
    o2.exposure = None
    assert o2 != o
    o2 = o.copy()
    o2.noise_rescale = 1.5
    assert o2 != o
    o2 = o.copy()
    o2.data = o.data * 2
    assert o2 != o


def test_copy_processing_from(obs2d):
    o = obs2d.copy()
    o2 = o.copy()
    o2.noise_rescale = 1.5
    o.copy_processing_from(o2)
    assert o.noise_rescale == 1.5


def test_reset_processing(obs2d):
    o = obs2d.copy()
    o.noise_rescale = 2.0
    o.reset_processing()
    assert o.noise_rescale == 1


def test_valid(obs2d):
    o = obs2d.copy()
    assert np.all(o.valid)
    data = o.data
    data[0, 0] = np.nan
    weight = o.weight.data
    weight[0, 1] = 0.0
    weight[0, 2] = -1.0
    o.data = data
    o.weight.data = weight
    assert np.allclose(np.nonzero(~o.valid)[0], [0, 0, 0])
    assert np.allclose(np.nonzero(~o.valid)[1], [0, 1, 2])
    o.is_zero_weight_valid = True
    assert np.allclose(np.nonzero(~o.valid)[0], [0, 0])
    assert np.allclose(np.nonzero(~o.valid)[1], [0, 2])

    o.weight = None
    assert not o.valid[0, 0]
    assert np.all(o.valid.ravel()[1:])
    o.destroy()
    assert o.valid.size == 0


def test_clear(obs2d):
    o = obs2d.copy()
    o.clear()
    assert np.allclose(o.data, 0)
    assert np.allclose(o.weight.data, 0)
    assert np.allclose(o.exposure.data, 0)


def test_discard(obs2d):
    o = obs2d.copy()
    o.discard()
    assert np.all(np.isnan(o.data))
    assert np.allclose(o.flag, 1)
    assert np.allclose(o.weight.data, 0)
    assert np.allclose(o.exposure.data, 0)


def test_destroy(obs2d):
    o = obs2d.copy()
    o.destroy()
    assert o.size == 0
    assert o.weight.size == 0
    assert o.exposure.size == 0


def test_set_data_shape(obs2d):
    o = obs2d.copy()
    o.set_data_shape((5, 5))
    assert o.data.shape == (5, 5)
    assert o.weight.shape == (5, 5)
    assert o.exposure.shape == (5, 5)
    assert o.weight.unit == 1 * units.Unit('Jy')
    assert o.exposure.unit == 1 * units.Unit('Jy')


def test_to_weight_image(obs2d):
    o = obs2d.copy()
    w = o.to_weight_image(None)
    assert w.shape == o.shape and np.allclose(w.data, 0)
    assert np.isnan(w.blanking_value)

    o = Observation2D(data=o.data, weight_dtype=int)
    w = o.to_weight_image(o.data)
    assert w.blanking_value == 0
    assert w.dtype == int
    assert np.allclose(w.data, o.data)


def test_get_weights(obs2d):
    o = obs2d.copy()
    w = o.get_weights()
    assert np.allclose(w.data, 16)


def test_get_weight_image(obs2d):
    o = obs2d.copy()
    assert o.get_weight_image() is o.weight


def test_weight_values(obs2d):
    o = obs2d.copy()
    assert np.allclose(o.weight_values(), 16)


def test_set_weight_image(obs2d):
    o = obs2d.copy()
    r = np.random.random(o.shape)
    o.set_weight_image(r)
    assert np.allclose(o.weight.data, r)


def test_get_exposures(obs2d):
    o = obs2d.copy()
    e = o.get_exposures()
    assert np.allclose(e.data, 0.5)


def test_exposure_values(obs2d):
    o = obs2d.copy()
    assert np.allclose(o.exposure_values(), 0.5)


def test_get_exposure_image(obs2d):
    o = obs2d.copy()
    assert o.get_exposure_image() is o.exposure


def test_get_noise(obs2d):
    o = obs2d.copy()
    noise = o.get_noise()
    assert np.allclose(noise.data, 0.25)


def test_noise_values(obs2d):
    o = obs2d.copy()
    noise = o.noise_values()
    assert np.allclose(noise, 0.25)


def test_set_noise(obs2d):
    o = obs2d.copy()
    noise = np.full(o.shape, 0.5)
    o.set_noise(noise)
    assert np.allclose(o.weight.data, 4)


def test_get_significance(obs2d):
    o = obs2d.copy()
    s2n = o.get_significance()
    assert s2n.data[5, 6] == 4
    mask = np.full(o.shape, True)
    mask[5, 6] = False
    assert np.allclose(s2n.data[mask], 0)


def test_significance_values(obs2d):
    o = obs2d.copy()
    assert np.allclose(o.significance_values(), o.data * 4)


def test_set_significance(obs2d):
    o = obs2d.copy()
    d0 = o.data.copy()
    s2n = o.significance_values() * 2
    o.set_significance(s2n)
    assert np.allclose(o.data, d0 * 2)


def test_scale(obs2d):
    o = obs2d.copy()
    d0 = o.data.copy()
    o.scale(2)
    assert np.allclose(o.data, d0 * 2)
    assert np.allclose(o.weight.data, 4)


def test_crop(obs2d):
    o = obs2d.copy()
    ranges = np.array([[4, 7], [1, 8]])
    o.crop(ranges)
    assert o.data.shape == (8, 4)
    assert o.weight.shape == (8, 4)
    assert o.exposure.shape == (8, 4)


def test_accumulate(obs2d):
    o = obs2d.copy()
    o2 = o.copy()
    o.clear()
    o.accumulate(o2)
    assert np.allclose(o.data, obs2d.data * 16)
    assert np.allclose(o.weight.data, 16)
    assert np.allclose(o.exposure.data, 0.5)
    o.clear()
    o.accumulate(o2, weight=2.0, gain=3.0)
    assert np.allclose(o.data, obs2d.data * 96)
    assert np.allclose(o.weight.data, 288)
    assert np.allclose(o.exposure.data, 0.5)


def test_accumulate_at(obs2d):
    o = obs2d.copy()
    o.clear()
    image = Image2D(data=np.ones(o.shape))
    gains = Image2D(data=np.full(o.shape, 2.0))
    weights = Image2D(data=np.full(o.shape, 3.0))
    times = Image2D(data=np.full(o.shape, 4.0))
    o.accumulate_at(image, gains, weights, times)
    assert np.allclose(o.data, 6)
    assert np.allclose(o.weight.data, 12)
    assert np.allclose(o.exposure.data, 4)


def test_merge_accumulate(obs2d):
    o = obs2d.copy()
    d0 = o.data.copy()
    o.data = o.data * o.weight.data
    o2 = o.copy()
    o.merge_accumulate(o2)
    assert np.allclose(o.data, d0 * 32)
    assert np.allclose(o.weight.data, 32)
    assert np.allclose(o.exposure.data, 1)


def test_end_accumulation(obs2d):
    o = obs2d.copy()
    d0 = o.data.copy()
    o.end_accumulation()
    assert np.allclose(o.data, d0 / 16)


def test_get_chi2(obs2d):
    o = obs2d.copy()
    assert o.get_chi2() == 0
    assert np.isclose(o.get_chi2(robust=False), 144 / 990)
    o.destroy()
    assert np.isnan(o.get_chi2())


def test_mean(obs2d):
    o = obs2d.copy()
    m, w = o.mean()
    assert np.isclose(m, 9 / 990)
    assert w == 1760


def test_median(obs2d):
    o = obs2d.copy()
    m, w = o.median()
    assert m == 0 and w == 1760


def test_reweight(obs2d):
    o = obs2d.copy()
    o.reweight(robust=False)
    assert np.allclose(o.weight.data, 110)
    assert np.isclose(o.noise_rescale, 0.381385, atol=1e-6)
    old = o.noise_rescale
    o.reweight(robust=True)
    assert o.noise_rescale == old  # Aborted


def test_unscale_weights(obs2d):
    o = obs2d.copy()
    w0 = o.weight.data.copy()
    o.reweight(robust=False)
    assert not np.allclose(o.weight.data, w0)
    o.unscale_weights()
    assert np.allclose(o.weight.data, w0)
    assert o.noise_rescale == 1


def test_mem_correct_observation(obs2d):
    o = obs2d.copy()
    model = o.get_noise()
    o.mem_correct_observation(model, 0.1)
    mask = np.full(o.shape, True)
    mask[5, 6] = False
    assert np.allclose(o.data[mask], 0.008664, atol=1e-6)
    assert np.allclose(o.data[~mask], 0.973249, atol=1e-6)
    o.mem_correct_observation(model.data, 0.1)
    assert np.allclose(o.data[mask], 0.017314, atol=1e-6)
    assert np.allclose(o.data[~mask], 0.947135, atol=1e-6)


def test_smooth(obs2d):
    o = obs2d.copy()
    beam_map = np.ones((3, 3), dtype=float)
    assert o.smoothing_beam is None
    o.exposure.data = o.data + 1
    o.weight.data = o.data + 1
    o_save = o.copy()
    o.smooth(beam_map)
    mask = np.full(o.shape, False)
    mask[4:7, 5:8] = True
    assert np.allclose(o.data[mask], 0.2)
    assert np.allclose(o.data[~mask], 0)
    assert np.allclose(o.weight.data[mask], 10)
    assert np.allclose(o.weight.data[0], [4] + [6] * 9 + [4])
    assert np.allclose(o.exposure.data[mask], 1.2)
    assert np.allclose(o.exposure.data[~mask], 1)
    assert np.isclose(o.smoothing_beam.fwhm, 2.818312 * arcsec, atol=1e-6)

    expected = o.copy()
    o = o_save.copy()
    reference_index = np.ones(2)
    o.smooth(beam_map, reference_index=reference_index)
    assert o == expected


def test_fast_smooth(large_obs2d):
    o = large_obs2d.copy()
    beam = o.smoothing_beam.copy()
    beam.fwhm = 6 * units.Unit('arcsec')
    beam_image = beam.get_beam_map(o.grid)
    steps = np.full(2, 2)
    o.fast_smooth(beam_image, steps)
    assert np.isclose(o.data.sum(), 1, atol=1e-6)
    line = o.data[50, 45:56]
    assert np.isclose(line[5], 0.024515, atol=1e-6)
    # FWHM location - about half the max
    assert np.isclose(line[2], 0.012352, atol=1e-6)
    assert np.isclose(line[8], line[2])  # Reflection value
    assert np.allclose(o.weight.data[50, 45:56], 652.659860, atol=1e-6)
    assert np.allclose(o.exposure.data, 0.5)
    assert np.isclose(o.smoothing_beam.fwhm, 6.0730999 * units.Unit('arcsec'),
                      atol=1e-6)


def test_filter_correct(obs2d):
    o = obs2d.copy()
    d0 = o.data.copy()
    o.filter_correct(5 * arcsec)
    assert np.allclose(o.data, d0 * 26)
    assert o.correcting_fwhm == 5 * arcsec


def test_undo_filter_correct(obs2d):
    o = obs2d.copy()
    d0 = o.data.copy()
    o.filter_correct(4 * arcsec)
    o.undo_filter_correct()
    assert np.allclose(o.data, d0)
    assert np.isnan(o.correcting_fwhm)


def test_fft_filter_above(obs2d):
    o = obs2d.copy()
    o.filter_fwhm = np.inf * arcsec
    o.fft_filter_above(0.01 * arcsec)
    assert np.allclose(o.data, 0, atol=1e-2)


def test_resample_from_map(obs2d, alternate_obs2d):
    o = obs2d.copy()
    o.clear()
    a = alternate_obs2d.copy()
    o.resample_from_map(a)
    assert np.allclose(o.weight.data, 16)
    assert np.allclose(o.exposure.data, 0.5)
    assert np.isclose(o.data.sum(), 1)
    assert np.allclose(o.data[4:7, 5:8],
                       [[0.00158212, 0.0366114, 0.00158212],
                        [0.0366114, 0.84721308, 0.0366114],
                        [0.00158212, 0.0366114, 0.00158212]],
                       atol=1e-6)

    with pytest.raises(ValueError) as err:
        o.resample_from_map(None)
    assert "cannot be resampled from None" in str(err.value)


def test_get_table_entry(obs2d):
    o = obs2d.copy()
    assert o.get_table_entry('depth') == 16
    assert o.get_table_entry('max') == 1


def test_get_hdus(obs2d):
    hdus = obs2d.get_hdus()
    assert np.allclose(hdus[0].data, obs2d.data)
    assert np.allclose(hdus[1].data, obs2d.exposure.data)
    assert hdus[1].header['EXTNAME'] == 'Exposure'
    assert np.allclose(hdus[2].data, obs2d.get_noise().data)
    assert hdus[2].header['EXTNAME'] == 'Noise'
    assert np.allclose(hdus[3].data, obs2d.get_significance().data)
    assert hdus[3].header['EXTNAME'] == 'S/N'


def test_get_info(obs2d):
    o = obs2d.copy()
    info = o.get_info()
    assert 'Noise re-scaling' not in info[-1]
    o.noise_rescale = 0.5
    info = o.get_info()
    assert 'Noise re-scaling' in info[-1]


def test_index_of_max(obs2d):
    o = obs2d.copy()
    value, index = o.index_of_max()
    assert value == 4  # significance
    assert index.x == 6 and index.y == 5
