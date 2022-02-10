# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.coordinates import SkyCoord, EarthLocation
import numpy as np
import pytest
from scipy.sparse.csr import csr_matrix

from sofia_redux.scan.flags.flagged_data import FlaggedData
from sofia_redux.scan.flags.flags import Flags
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.flags.channel_flags import ChannelFlags


class Dummy(FlaggedData):

    flagspace = ChannelFlags

    @property
    def referenced_attributes(self):
        attributes = super().referenced_attributes
        attributes.add('value')
        return attributes

    @property
    def special_fields(self):
        return {'value'}

    @property
    def default_field_types(self):
        defaults = super().default_field_types
        defaults.update({
            'empty_value': float,
            'quantity_value': 1 * units.Unit('degree'),
            'unit_value': units.Unit('meter'),
            'coordinate_quantity': (Coordinate2D, 1 * units.Unit('arcsec')),
            'coordinate_unit': (Coordinate2D, units.Unit('arcminute')),
            'coordinate_str': (Coordinate2D, 'Kelvin'),
            'coordinate_value': (Coordinate2D, 2.0),
            'coordinate_shaped': (Coordinate2D, 1.5 * units.Unit('s'), 2),
            'shaped_quantity': (1 * units.Unit('um'), 3),
            'shaped_unit': (units.Unit('count'), 2, 3),
            'shaped_type': (int, 5),
            'shaped_value': (4, 2, 2),
            'value': 3.0
        })
        return defaults


def set_dummy_data():
    d = Dummy()
    d.fixed_index = np.arange(10)
    d.set_default_values()
    d.flag[:5] = 1
    row = np.array([0, 1, 2, 0])
    col = np.array([0, 1, 1, 2])
    data = np.array([1, 2, 4, 8])
    m = csr_matrix((data, (row, col)), shape=(10, 10))
    d.matrix = m
    d.skycoord = SkyCoord(ra=np.full(10, 45) * units.Unit('degree'),
                          dec=np.arange(10) * units.Unit('degree'))
    d.earth = EarthLocation(lon=np.zeros(10), lat=np.zeros(10))
    d.none = None
    return d


@pytest.fixture
def dummy_data():
    return set_dummy_data()


def test_init():
    d = FlaggedData()
    assert d.flagspace == Flags
    assert d.fixed_index is None
    assert d.flag is None


def test_is_singular():
    d = FlaggedData()
    d.fixed_index = 1
    assert d.is_singular
    d.fixed_index = np.arange(3)
    assert not d.is_singular


def test_fixed_index():
    d = FlaggedData()
    d.fixed_index = 1
    assert isinstance(d.fixed_index, np.ndarray) and d.fixed_index == 1


def test_flag():
    d = FlaggedData()
    d.fixed_index = np.arange(5)
    d.flag = np.full(5, 1)
    assert np.allclose(d.flag, [1, 1, 1, 1, 1])


def test_default_field_types():
    d = FlaggedData()
    assert d.default_field_types == {'flag': 0}


def test_referenced_attributes():
    d = FlaggedData()
    assert d.referenced_attributes == set()


def test_internal_attributes():
    d = FlaggedData()
    assert d.internal_attributes == set()


def test_special_fields():
    d = FlaggedData()
    assert d.special_fields == set()


def test_fields():
    d = FlaggedData()
    assert d.fields == {'_fixed_index', '_flag'}


def test_size():
    d = FlaggedData()
    assert d.size == 0
    d.fixed_index = 1
    assert d.size == 1
    d.fixed_index = np.arange(5)
    assert d.size == 5


def test_shape():
    d = FlaggedData()
    assert d.shape == ()
    d.fixed_index = 1
    assert d.shape == ()
    d.fixed_index = np.arange(5)
    assert d.shape == (5,)


def test_getitem(dummy_data):
    d = dummy_data.copy()
    d2 = d[:5]
    assert d2.size == 5


def test_set_default_values():
    d = Dummy()
    d.fixed_index = np.arange(5)
    d.set_default_values()
    assert d.empty_value.shape == (5,) and d.empty_value.dtype == float

    assert d.quantity_value.shape == (5,)
    assert np.allclose(d.quantity_value.value, 1)
    assert d.quantity_value.unit == 'degree'

    assert d.unit_value.shape == (5,)
    assert d.unit_value.unit == 'meter'

    assert d.coordinate_quantity.shape == (5,)
    assert isinstance(d.coordinate_quantity, Coordinate2D)
    assert d.coordinate_quantity.unit == 'arcsec'
    assert np.allclose(d.coordinate_quantity.coordinates.value, 1)

    assert d.coordinate_unit.shape == (5,)
    assert isinstance(d.coordinate_unit, Coordinate2D)
    assert d.coordinate_unit.unit == 'arcminute'

    assert d.coordinate_shaped.shape == (5, 2)
    assert d.coordinate_shaped.unit == 'second'
    assert np.allclose(d.coordinate_shaped.coordinates.value, 1.5)

    assert d.shaped_quantity.shape == (5, 3)
    assert d.shaped_quantity.unit == 'um'
    assert np.allclose(d.shaped_quantity.value, 1)

    assert d.shaped_unit.shape == (5, 2, 3)
    assert d.shaped_unit.unit == 'count'

    assert d.shaped_type.shape == (5, 5)
    assert d.shaped_type.dtype == int

    assert d.shaped_value.shape == (5, 2, 2)
    assert d.shaped_value.dtype == int
    assert np.allclose(d.shaped_value, 4)

    assert d.value.shape == (5,)
    assert d.value.dtype == float
    assert np.allclose(d.value, 3)


def test_copy(dummy_data):
    d = dummy_data
    new = d.copy()
    assert d is not new

    for key in new.default_field_types.keys():
        v1 = getattr(d, key)
        v2 = getattr(new, key)
        if key == 'value':  # referenced attribute
            assert v1 is v2
        else:
            assert v1 is not v2

        if isinstance(v1, Coordinate2D):
            assert v1 == v2
        else:
            assert np.allclose(v1, v2)
            assert np.allclose(v1, v2)

    new.weird = 1
    new2 = new.copy()
    assert new2.weird == 1


def test_is_flagged(dummy_data):
    d = dummy_data
    assert np.allclose(d.is_flagged(), [True] * 5 + [False] * 5)
    inds = d.is_flagged(indices=True)
    assert np.allclose(inds, np.arange(5))
    inds = d.is_flagged(indices=True, flag=0)
    assert np.allclose(inds, np.arange(5) + 5)


def test_is_unflagged(dummy_data):
    d = dummy_data
    assert np.allclose(d.is_unflagged(), [False] * 5 + [True] * 5)
    inds = d.is_unflagged(indices=True)
    assert np.allclose(inds, np.arange(5) + 5)
    inds = d.is_unflagged(indices=True, flag=0)
    assert np.allclose(inds, np.arange(5))


def test_set_flags(dummy_data):
    d = dummy_data
    original = d.flag.copy()
    d2 = FlaggedData()
    d2.fixed_index = np.arange(10).reshape((2, 5))
    d2.set_default_values()

    d.set_flags(3, indices=np.empty(0))
    assert np.allclose(d.flag, original)

    d2.set_flags(3, indices=np.empty((2, 0)))
    assert np.allclose(d2.flag, 0)

    indices = np.nonzero(np.arange(d.size) % 2)[0]
    d.set_flags('DEAD', indices=indices)
    assert np.allclose(d.flag, [1, 1, 1, 1, 1, 1, 0, 1, 0, 1])


def test_unflag(dummy_data):
    d = dummy_data
    original = d.flag.copy()
    d2 = FlaggedData()
    d2.fixed_index = np.arange(10).reshape((2, 5))
    d2.set_default_values()

    d.unflag(1, indices=np.empty(0))
    assert np.allclose(d.flag, original)

    d2.flag.fill(1)
    d2.unflag(3, indices=np.empty((2, 0)))
    assert np.allclose(d2.flag, 1)

    indices = np.nonzero(np.arange(d.size) % 2)[0]
    d.unflag('DEAD', indices=indices)
    assert np.allclose(d.flag, [1, 0, 1, 0, 1, 0, 0, 0, 0, 0])


def test_discard_flag(dummy_data):
    d = dummy_data
    d.discard_flag('DEAD')
    assert d.size == 5
    assert np.allclose(d.fixed_index, np.arange(5) + 5)


def test_get_flagged_indices(dummy_data):
    d = dummy_data
    d.flag[0] = 3
    flagged = d.get_flagged_indices(indices=True)
    assert np.allclose(flagged, np.arange(10))  # always everything
    flagged = d.get_flagged_indices(keep_flag=0, indices=True)
    assert np.allclose(flagged, np.arange(5) + 5)
    flagged = d.get_flagged_indices(keep_flag=0, indices=False)
    assert np.allclose(flagged, [False] * 5 + [True] * 5)
    flagged = d.get_flagged_indices(keep_flag=1, indices=True)
    assert np.allclose(flagged, np.arange(5))
    flagged = d.get_flagged_indices(match_flag=1, indices=True)
    assert np.allclose(flagged, [1, 2, 3, 4])
    flagged = d.get_flagged_indices(discard_flag=1, indices=True, keep_flag=3)
    assert flagged.size == 0
    flagged = d.get_flagged_indices(discard_flag=1, indices=True)
    assert np.allclose(flagged, np.arange(5) + 5)


def test_find_fixed_indices():
    d = FlaggedData()
    fixed_indices = np.arange(0, 20, 2)
    d.fixed_index = fixed_indices
    assert d.find_fixed_indices(1, cull=False) == -1
    assert d.find_fixed_indices(2) == 1
    d.fixed_index = np.full(5, 1)
    assert np.allclose(d.find_fixed_indices(1), np.arange(5))

    d.fixed_index = fixed_indices
    assert np.allclose(d.find_fixed_indices([0, 1, 2], cull=True), [0, 1])
    assert np.allclose(d.find_fixed_indices([0, 1, 2], cull=False), [0, -1, 1])


def test_to_indices():
    d = FlaggedData()
    d.fixed_index = np.arange(10)
    assert d.to_indices(None) is None
    bool_array = np.array([True, False, True])
    assert np.allclose(d.to_indices(bool_array, discard=True), [1])
    assert np.allclose(d.to_indices(bool_array), [0, 2])

    int_array = np.arange(5)
    assert np.allclose(d.to_indices(int_array, discard=True), [5, 6, 7, 8, 9])
    assert np.allclose(d.to_indices(int_array), int_array)

    s = slice(0, 5)
    assert np.allclose(d.to_indices(s), [0, 1, 2, 3, 4])


def test_get_indices(dummy_data):
    empty = FlaggedData()
    indices = np.arange(5)
    assert empty.get_indices(indices).size == 0

    d = dummy_data
    new = d.get_indices(indices)
    assert new.coordinate_shaped.shape == (5, 2)
    assert np.allclose(new.value, [3] * 5)

    new = d.get_indices(np.asarray(1))
    assert new.value == 3
    assert new.coordinate_shaped.shape == (2,)


def test_get_attribute_indices():
    d = FlaggedData()
    indices = np.arange(2)
    assert d.get_attribute_indices({'foo'}, 'foo', d, indices) is d

    row = np.array([0, 1, 2, 0])
    col = np.array([0, 1, 1, 0])
    data = np.array([1, 2, 4, 8])
    m = csr_matrix((data, (row, col)), shape=(3, 3))
    m2 = d.get_attribute_indices({}, 'foo', m, indices)
    assert np.allclose(m2.toarray(), [[9, 0], [0, 2]])

    assert np.allclose(
        d.get_attribute_indices({}, 'foo', np.arange(4), indices), [0, 1])

    assert d.get_attribute_indices({}, 'foo', 1, indices) == 1


def test_delete_indices(dummy_data):
    o = dummy_data.copy()
    d = o.copy()
    original_size = d.size
    d.delete_indices(np.empty(0, dtype=int))
    assert d.size == original_size
    d.delete_indices(np.arange(5))
    assert d.size == 5
    assert d.value.size == 10  # in special_fields
    assert d.coordinate_shaped.shape == (5, 2)
    assert d.matrix.shape == (5, 5)


def test_insert_blanks(dummy_data):
    d = dummy_data.copy()
    d.insert_blanks(np.full(2, 1))
    assert d.size == 12
    assert d.value.size == 10  # special field
    assert d.matrix.shape == (12, 12)
    assert d.earth.size == 12
    assert d.skycoord.size == 12
    assert np.allclose(d.skycoord[1:3].ra, 0)
    assert np.allclose(d.fixed_index[1:3], -1)


def test_merge(dummy_data):
    d1 = dummy_data.copy()
    d2 = dummy_data.copy()
    d = d1.copy()
    d.merge(d2)
    assert d.size == 20
    assert np.allclose(d.value, d1.value)  # special field
    assert d.matrix.shape == (20, 20)
    assert d.coordinate_shaped.shape == (20, 2)
    assert d.earth.shape == (20,)
    assert d.skycoord.shape == (20,)
    assert d.empty_value.size == 20
    assert np.allclose(d.fixed_index, np.arange(20) % 10)

    d1.empty_value = None
    d2.quantity_value = None
    d1.merge(d2)
    assert d1.empty_value is None
    assert d1.quantity_value.size == 10


def test_get_index_size(dummy_data):
    d = dummy_data.copy()
    i, s = d.get_index_size()
    assert i == slice(None)
    assert s == 10
    i, s = d.get_index_size(slice(0, 5))
    assert i == slice(0, 5) and s == 5
    i, s = d.get_index_size(np.arange(2))
    assert np.allclose(i, [0, 1]) and s == 2
    i, s = d.get_index_size(11)
    assert i == 11 and s == 0
    with pytest.raises(ValueError) as err:
        _ = d.get_index_size('a')
    assert "Incorrect indices format" in str(err.value)
