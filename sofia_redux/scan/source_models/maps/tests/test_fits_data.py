# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.source_models.maps.fits_data import FitsData


ud = units.dimensionless_unscaled


@pytest.fixture
def data_array():
    shape = (10, 11)  # y, x
    d = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
    return d


@pytest.fixture
def fits_data(data_array):
    f = FitsData(data=data_array.copy(), unit='Jy')
    return f


@pytest.fixture
def ones(fits_data):
    f = fits_data.copy()
    f.fill(1.0)
    return f


@pytest.fixture
def beam_map():
    beam = np.array([0.5, 1, 0.5])
    beam = beam[None] * beam[:, None]
    return beam


def test_init(data_array):
    data = data_array.copy()
    f = FitsData(data=data)
    assert f.shape == (10, 11)
    assert np.allclose(f.data, data)
    assert f.history == ['new size 11x10']
    assert not f.verbose
    assert f.unit == 1 * ud
    assert f.local_units == {'': 1 * ud}
    assert f.alternate_unit_names == {'': ''}
    assert f.log_new_data
    assert f.parallelism == 0
    assert f.executor is None
    assert np.isnan(f.blanking_value)


def test_copy(fits_data):
    f = fits_data.copy()
    f2 = f.copy()
    assert f == f2 and f is not f2


def test_eq(fits_data):
    f = fits_data.copy()
    f2 = f.copy()
    assert f == f and f == f2
    assert f2.data is not f.data
    assert f2.local_units is f.local_units
    assert f2.alternate_unit_names is f.alternate_unit_names
    assert f != 1
    f2.unit = 'K'
    assert f != f2


def test_referenced_attributes():
    f = FitsData()
    assert 'local_units' in f.referenced_attributes
    assert 'alternate_unit_names' in f.referenced_attributes


def test_unit():
    f = FitsData()
    assert f.unit == 1 * ud
    f.unit = 'K'
    assert f.unit == 1 * units.Unit('K')


def test_fits_to_numpy():
    coordinates = Coordinate2D(np.arange(10).reshape(2, 5))
    new = FitsData.fits_to_numpy(coordinates)
    assert np.allclose(new, coordinates.coordinates[::-1])
    coordinates = coordinates.coordinates
    new = FitsData.fits_to_numpy(coordinates)
    assert np.allclose(new, coordinates[::-1])
    coordinates = np.random.random((2, 5)) > 0.5
    new = FitsData.fits_to_numpy(coordinates)
    assert np.allclose(new, coordinates)
    new = FitsData.fits_to_numpy([1, 2])
    assert new == [2, 1]
    new = FitsData.fits_to_numpy(1)
    assert new == 1


def test_numpy_to_fits():
    new = FitsData.numpy_to_fits([1, 2])
    assert new == [2, 1]


def test_get_size_string(fits_data):
    f = fits_data.copy()
    assert f.get_size_string() == '11x10'
    f._data = None
    assert f.get_size_string() == '0'


def test_set_data_shape(fits_data):
    f = fits_data.copy()
    f.set_data_shape((4, 5))
    assert f.data.shape == (4, 5)
    assert f.shape == (4, 5)
    assert f.history == ['new size 5x4']


def test_set_data(fits_data):
    f = fits_data.copy()
    f.set_data(np.ones((3, 4)))
    assert f.data.shape == (3, 4)
    assert f.unit == 1 * units.Unit('Jy')
    f2 = f.copy()
    f2.unit = 'K'
    f.set_data(f2)
    assert f.unit == 1 * units.Unit('K')


def test_unit_to_quantity():
    k = units.Unit('K')
    assert FitsData.unit_to_quantity(1 * k) == 1 * k
    assert FitsData.unit_to_quantity('K') == 1 * k
    assert FitsData.unit_to_quantity(k) == 1 * k
    assert FitsData.unit_to_quantity(ud) == 1 * ud
    with pytest.raises(ValueError) as err:
        _ = FitsData.unit_to_quantity(1)
    assert 'Unit must be a ' in str(err.value)


def test_add_local_unit(fits_data):
    f = fits_data.copy()
    k = units.Unit('K')
    f.add_local_unit(k)
    assert f.local_units == {'Jy': 1 * units.Unit('Jy'),
                             'K': 1 * units.Unit('K')}
    assert f.alternate_unit_names == {
        'Jy': 'Jy', 'Jansky': 'Jy', 'jansky': 'Jy', 'K': 'K', 'Kelvin': 'K'}

    f.add_local_unit(k, alternate_names=['foo', 'bar'])
    assert f.alternate_unit_names == {'Jy': 'Jy', 'Jansky': 'Jy',
                                      'jansky': 'Jy', 'K': 'K', 'Kelvin': 'K',
                                      'foo': 'K', 'bar': 'K'}


def test_add_alternate_unit_names():
    f = FitsData()
    assert f.alternate_unit_names == {'': ''}
    f.add_alternate_unit_names('foo', 'bar')
    assert f.alternate_unit_names == {'': '', 'bar': 'foo', 'foo': 'foo'}


def test_get_unit():
    f = FitsData()
    k = units.Unit('K')
    f.add_local_unit(k)
    assert f.get_unit(k) == 1 * k
    assert f.get_unit('Kelvin') == 1 * k
    assert f.get_unit('Jy') == 1 * units.Unit('Jy')
    assert f.get_unit(2 * k) == 1 * k


def test_set_unit():
    f = FitsData()
    k = units.Unit('K')
    f.set_unit('K')
    assert f.local_units == {'': 1 * ud, 'K': 1 * k}
    assert f.unit == 1 * k


def test_set_default_unit():
    f = FitsData()
    f.unit = 1 * units.Unit('K')
    f.set_default_unit()
    assert f.unit == 1 * ud


def test_clear_history(fits_data):
    f = fits_data.copy()
    assert len(f.history) == 1
    f.clear_history()
    assert len(f.history) == 0


def test_add_history():
    f = FitsData()
    f.verbose = True
    with log.log_to_list() as log_list:
        f.add_history('foo')
    assert len(log_list) == 1 and log_list[0].msg == 'foo'
    assert f.history[-1] == 'foo'
    f.verbose = False
    f.history = None
    f.add_history(['foo', 'bar'])
    assert f.history == ['foo', 'bar']


def test_set_history():
    f = FitsData()
    f.set_history(['foo', 'bar'])
    assert f.history == ['foo', 'bar']
    f.set_history('foo')
    assert f.history == ['foo']


def test_add_history_to_header():
    f = FitsData()
    f.history = None
    header = fits.Header()
    f.add_history_to_header(header)
    assert len(header) == 0
    f.history = ['foo', 'bar']
    f.add_history_to_header(header)
    assert list(header['HISTORY']) == ['foo', 'bar']


def test_record_new_data(fits_data):
    f = fits_data.copy()
    f.log_new_data = False
    assert not f.log_new_data
    f.record_new_data()
    assert len(f.history) == 1
    assert f.log_new_data
    f.record_new_data(detail='foobar')
    assert f.history == ['set new image 11x10 foobar']


def test_set_parallel():
    f = FitsData()
    f.set_parallel(10)
    assert f.parallelism == 10


def test_set_executor():
    f = FitsData()
    f.set_executor('foo')
    assert f.executor == 'foo'


def test_clear(fits_data):
    f = fits_data.copy()
    f.clear()
    assert f.data.shape == (10, 11) and np.allclose(f.data, 0)
    assert f.history == ['clear 11x10']
    f.data = np.ones(f.shape)
    mask = np.full(f.shape, False)
    mask[:4] = True
    f.clear(mask)
    assert np.allclose(f.data[:4], 0)
    assert np.allclose(f.data[4:], 1)
    f.data = np.ones(f.shape)
    indices = np.nonzero(mask)[::-1]  # x, y format
    f.clear(indices)
    assert np.allclose(f.data[:4], 0)
    assert np.allclose(f.data[4:], 1)


def test_destroy(fits_data):
    f = fits_data.copy()
    assert len(f.history) == 1
    f.destroy()
    assert len(f.history) == 0
    assert f.shape == (0, 0)


def test_fill(fits_data):
    f = fits_data.copy()
    f.data = np.zeros(f.shape)
    f.fill(1)
    assert f.shape == (10, 11) and np.allclose(f.data, 1)
    assert f.history == ['fill 11x10 with 1']

    mask = np.full(f.shape, False)
    mask[:, :2] = True
    f.fill(2, mask)
    assert np.allclose(f.data[mask], 2)
    assert np.allclose(f.data[~mask], 1)

    indices = np.nonzero(mask)[::-1]  # FITS (x, y) order
    f.fill(0)
    f.fill(1, indices=indices)
    assert np.allclose(f.data[~mask], 0)
    assert np.allclose(f.data[mask], 1)


def test_add(fits_data):
    f = fits_data.copy()
    f.clear()
    f.add(1)
    assert np.allclose(f.data, 1)
    assert f.history[-1] == 'add 1'
    mask = np.full(f.shape, False)
    mask[:2] = True
    f.add(1, indices=mask, factor=2)
    assert np.allclose(f.data[:2], 3)
    assert np.allclose(f.data[2:], 1)
    assert f.history[-1] == 'add 2'
    f.fill(1)
    f2 = f.copy()
    indices = np.asarray(np.nonzero(mask))[::-1]  # For FITS (x, y) order
    f.add(f2, indices=indices)
    assert np.allclose(f.data[:2], 2)
    assert np.allclose(f.data[2:], 1)
    assert f.history[-1] == 'added FitsData'
    f.fill(1)
    f.add(np.ones(f.shape), factor=3)
    assert f.history[-1].startswith(
        'add [[3. 3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]\n')
    f.fill(1)
    f.add(f2, indices=indices, factor=3)
    assert np.allclose(f.data[:2], 4)
    assert np.allclose(f.data[2:], 1)
    assert f.history[-1] == 'added scaled FitsData (3x)'


def test_scale(ones):
    f = ones.copy()
    mask = np.full(f.shape, False)
    mask[:2] = True
    indices = np.asarray(np.nonzero(mask))[::-1]  # For FITS (x, y) order
    f.scale(3, indices=indices)
    assert np.allclose(f.data[:2], 3)
    assert np.allclose(f.data[2:], 1)
    assert f.history[-1] == 'scale by 3'


def test_validate(ones):
    f = ones.copy()

    class Validator(object):
        def __call__(self, array):
            array.discard(array.data == 2)

    data = f.data.copy()
    data[1] = 2
    data[0, 0] = np.nan
    f.data = data
    assert np.allclose(f.flag, 0)
    f.validate()
    assert f.flag[0, 0] == 1
    assert np.allclose(f.flag[np.isfinite(data)], 0)
    assert f.history[-1] == 'validate'

    f.validate(validator=Validator())
    assert f.flag[0, 0] == 1
    assert np.allclose(f.flag[0, 1:], 0)
    assert np.allclose(f.flag[1], 1)
    assert np.allclose(f.flag[2:], 0)
    assert f.history[-1].startswith('validate via')


def test_paste(ones):
    f = ones.copy()
    f2 = f.copy()
    f2.scale(2)
    f.paste(f2)
    assert f == f2
    assert f.history[-1] == 'pasted new content: 11x10'


def test_smooth(ones, beam_map):
    f = ones.copy()
    data = np.zeros(f.shape)
    data[5, 5] = 1
    reference_index = np.array([1, 1])
    f.data = data
    f.smooth(beam_map, reference_index=reference_index)
    assert np.allclose(f.data[4:7, 4:7], beam_map / 4)
    assert f.history[-1] == 'smoothed'


def test_get_smoothed(ones, beam_map):
    f = ones.copy()
    reference_index = np.array([1, 1])
    smoothed, weights = f.get_smoothed(
        beam_map, reference_index=reference_index)
    assert np.allclose(smoothed, 1)
    assert np.allclose(weights[0], [2.25, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2.25])
    assert np.allclose(weights[0], weights[-1])
    assert np.allclose(weights[:, 0], [2.25, 3, 3, 3, 3, 3, 3, 3, 3, 2.25])
    assert np.allclose(weights[:, 0], weights[:, -1])
    assert np.allclose(weights[1:9, 1:10], 4)


def test_fast_smooth(ones, beam_map):
    f = ones.copy()
    f.fast_smooth(beam_map, np.ones(2, dtype=int))
    assert np.allclose(f.data, 1)
    assert f.history[-2] == 'pasted new content: 11x10'
    assert f.history[-1] == 'smoothed (fast method)'


def test_get_fast_smoothed(ones, beam_map):
    f = ones.copy()
    steps = np.ones(2, dtype=int)
    smoothed = f.get_fast_smoothed(beam_map, steps)
    assert np.allclose(smoothed, 1)
    smoothed, weights = f.get_fast_smoothed(beam_map, steps, get_weights=True)
    assert np.allclose(weights[0], [2.25, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2.25])
    assert np.allclose(weights[0], weights[-1])
    assert np.allclose(weights[:, 0], [2.25, 3, 3, 3, 3, 3, 3, 3, 3, 2.25])
    assert np.allclose(weights[:, 0], weights[:, -1])
    assert np.allclose(weights[1:9, 1:10], 4)


def test_create_fits(ones):
    f = ones.copy()
    hdul = f.create_fits()
    assert len(hdul) == 1 and np.allclose(hdul[0].data, 1)


def test_get_hdus(ones):
    f = ones.copy()
    hdus = f.get_hdus()
    assert isinstance(hdus, list) and len(hdus) == 1


def test_create_hdu(ones):
    f = ones.copy()
    hdu = f.create_hdu()
    assert isinstance(hdu, fits.ImageHDU)
    assert np.allclose(hdu.data, 1)
    assert hdu.header['BUNIT'] == 'Jy'


def test_get_fits_data(ones):
    f = ones.copy()
    assert f.get_fits_data() is f.data


def test_edit_header(ones):
    f = ones.copy()
    header = fits.Header()
    data = f.data
    data[5, 5] = 100
    f.data = data
    f.edit_header(header)
    assert header['DATAMIN'] == 1.0
    assert header['DATAMAX'] == 100.0
    assert header['BZERO'] == 0.0
    assert header['BSCALE'] == 1.0
    assert header['BUNIT'] == 'Jy'
    assert header['HISTORY'] == 'new size 11x10'
    f.unit = ud
    f.edit_header(header)
    assert header['BUNIT'] == 'ct'


def test_parse_header(ones):
    f = ones.copy()
    header = fits.Header()
    f.parse_header(header)
    assert f.unit == 1 * ud
    header['BUNIT'] = 'Jy'
    f.parse_header(header)
    assert f.unit == 1 * units.Unit('Jy')


def test_parse_history(ones):
    f = ones.copy()
    header = fits.Header()
    f.parse_history(header)
    assert f.history == []
    header['HISTORY'] = 'foo'
    f.parse_history(header)
    assert f.history == ['foo']


def test_get_indices(ones):
    with pytest.raises(NotImplementedError):
        ones.get_indices(np.arange(5))


def test_delete_indices(ones):
    with pytest.raises(NotImplementedError):
        ones.delete_indices(np.arange(5))


def test_insert_blanks(ones):
    with pytest.raises(NotImplementedError):
        ones.insert_blanks(np.arange(5))


def test_merge(ones):
    with pytest.raises(NotImplementedError):
        ones.merge(ones.copy())


def test_resample_from(ones, beam_map):
    image = ones.copy()
    image.data[5, 6] = 2.0
    kernel = beam_map.copy()
    to_indices = np.stack(
        [x.ravel() for x in np.indices(image.shape)])[::-1]  # xy not yx
    kernel_reference_index = np.array([1, 1])
    f = ones.copy()
    f.resample_from(image, to_indices, kernel=kernel,
                    kernel_reference_index=kernel_reference_index)

    mask = np.full(image.shape, False)
    mask[4:7, 5:8] = True
    assert np.allclose(
        f.data[mask],
        [1.0625, 1.125, 1.0625, 1.125, 1.25, 1.125, 1.0625, 1.125, 1.0625])
    assert np.allclose(f.data[~mask], 1)
    assert f.history[-1] == 'resampled 11x10 from 11x10'

    f.clear()
    f.resample_from(image.data, to_indices, kernel=kernel,
                    kernel_reference_index=kernel_reference_index)
    assert np.allclose(
        f.data[mask],
        [1.0625, 1.125, 1.0625, 1.125, 1.25, 1.125, 1.0625, 1.125, 1.0625])
    assert np.allclose(f.data[~mask], 1)
    assert f.history[-1] == 'resampled 11x10 from 11x10'


def test_despike(ones):
    f = ones.copy()
    f.despike(2.0)
    assert f.history[-1] == 'despiked at 2.000'


def test_get_index_range(ones):
    f = ones.copy()
    index_range = f.get_index_range()
    assert np.allclose(index_range, [[0, 11], [0, 10]])


def test_value_at(ones):
    f = ones.copy()
    data = f.data
    f.data[3, 5] = 2.0  # (x, y) = (5, 3)
    f.data = data
    assert np.isclose(f.value_at([5, 3]), 2)
    assert np.isclose(f.value_at([5, 4]), 1)
    assert np.isclose(f.value_at([4.9, 3.1]), 1.715, atol=1e-3)


def test_index_of_max(ones):
    f = ones.copy()
    data = f.data
    data[4, 5] = 2.0  # (x, y) = (5, 4)
    f.data = data
    value, index = f.index_of_max()
    assert value == 2
    assert np.allclose(index, [5, 4])


def test_get_refined_peak_index(ones):
    f = ones.copy()
    data = f.data
    data[5, 4] = 1.4
    data[5, 5] = 1.9
    data[5, 6] = 1.6
    peak_index = np.asarray([5, 5])
    f.data = data
    index = f.get_refined_peak_index(peak_index)
    assert np.allclose(index, [5.125, 5])


def test_crop(ones):
    f = ones.copy()
    f.crop(np.array([[4, 8], [2, 4]]))
    assert f.shape == (2, 4)
