# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.coordinates import SkyCoord, EarthLocation
import numpy as np
import pytest
from scipy.sparse.csr import csr_matrix

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.flags.flagged_data_group import FlaggedDataGroup
from sofia_redux.scan.flags.tests.test_flagged_data import set_dummy_data


@pytest.fixture
def dummy_data():
    return set_dummy_data()


@pytest.fixture
def dummy_group(dummy_data):
    """
    Return a test data group.

    Parameters
    ----------
    dummy_data : FlaggedData

    Returns
    -------
    FlaggedDataGroup
    """
    g = FlaggedDataGroup(dummy_data, indices=np.arange(5), name='test')
    return g


def test_init(dummy_data):
    g = FlaggedDataGroup(dummy_data, indices=np.arange(5), name='test')
    assert g.name == 'test'
    assert np.allclose(g.indices, np.arange(5))
    assert np.allclose(g.fixed_index, np.arange(5))
    assert np.allclose(g.fixed_indices, np.arange(5))


def test_copy(dummy_group):
    g = dummy_group
    g2 = g.copy()
    assert g.data is g2.data
    assert g is not g2
    g3 = g.copy(full=True)
    assert g.data is not g3.data
    assert np.allclose(g.matrix.toarray(), g2.matrix.toarray())
    assert np.allclose(g.matrix.toarray(), g3.matrix.toarray())


def test_getitem(dummy_group):
    g = dummy_group
    g2 = g[0:2]
    assert g2.size == 2


def test_getattr(dummy_group, dummy_data):
    g = dummy_group
    data = g.__getattr__('data')
    assert data is dummy_data

    with pytest.raises(AttributeError) as err:
        _ = g.__getattr__('foobar')
    assert "no attribute" in str(err.value)

    assert g.none is None

    g2 = g.copy(full=True)
    g2._indices = None
    assert g2.matrix.shape == (10, 10)
    assert g.matrix.shape == (5, 5)
    assert g.coordinate_shaped.shape == (5, 2)


def test_setattr(dummy_group):
    g = dummy_group.copy(full=True)
    data = g.data.copy()
    d1 = g.data
    assert g.data is d1
    g.data = data
    assert g.data is not d1

    g = dummy_group.copy(full=True)
    g.name = 'new_name'
    assert g.name == 'new_name'

    # Check does not alter the original data with None values
    g.value = None
    assert g.value is None
    assert g.data.value.size == 10

    g = dummy_group.copy(full=True)
    g._indices = None
    g.empty_value = 3
    assert g.empty_value.size == 10 and np.allclose(g.empty_value, 3)

    g = dummy_group.copy(full=True)
    g.matrix = 1
    m0 = g.data.matrix.toarray()
    expected = np.zeros((10, 10))
    expected[:5, :5] = 1
    assert np.allclose(m0, expected)

    row = np.array([0, 1, 2, 0])
    col = np.array([0, 1, 1, 2])
    data = np.array([1, 2, 3, 4])
    m = csr_matrix((data, (row, col)), shape=(5, 5))
    g.matrix = m
    assert np.allclose(g.data.matrix.data, [1, 4, 2, 3])

    sky = SkyCoord(ra=123 * units.Unit('degree'),
                   dec=33 * units.Unit('degree'))
    g.skycoord = sky
    assert np.allclose(g.data.skycoord.ra.to('degree').value,
                       [123] * 5 + [45] * 5)
    assert np.allclose(g.data.skycoord.dec.to('degree').value,
                       [33] * 5 + [5, 6, 7, 8, 9])

    earth = EarthLocation(lat=1 * units.Unit('degree'),
                          lon=2 * units.Unit('degree'))
    g.earth = earth
    assert np.allclose(g.data.earth.lat.to('degree').value,
                       [1] * 5 + [0] * 5)
    assert np.allclose(g.data.earth.lon.to('degree').value,
                       [2] * 5 + [0] * 5)

    c = Coordinate2D([np.arange(5), np.arange(5) + 5])
    g.coordinate_value = c
    assert np.allclose(g.data.coordinate_value.x,
                       [0, 1, 2, 3, 4, 2, 2, 2, 2, 2])
    assert np.allclose(g.data.coordinate_value.y,
                       [5, 6, 7, 8, 9, 2, 2, 2, 2, 2])


def test_protected_attributes(dummy_group):
    assert 'data' in dummy_group.protected_attributes


def test_indices(dummy_group):
    assert np.allclose(dummy_group.indices, np.arange(5))
    g = dummy_group.copy(full=True)
    g.indices = np.arange(5) + 5
    assert np.allclose(g.indices, np.arange(5) + 5)
    g.indices = None
    assert np.allclose(g.indices, np.arange(10))


def test_fixed_indices(dummy_group):
    g = dummy_group.copy(full=True)
    assert np.allclose(g.fixed_indices, np.arange(5))
    g.fixed_indices = None
    assert np.allclose(g.fixed_indices, np.arange(10))
    g.fixed_indices = np.arange(5) + 1
    assert np.allclose(g.fixed_indices, np.arange(5) + 1)


def test_fields(dummy_group):
    assert 'matrix' in dummy_group.fields
    g = dummy_group.copy(full=True)
    g.data = None
    assert len(g.fields) == 0


def test_size(dummy_group):
    g = dummy_group.copy(full=True)
    assert g.size == 5
    g._indices = None
    assert g.size == 10
    g = dummy_group.copy(full=True)
    g._fixed_index = None
    assert g.size == 0


def test_flagspace(dummy_group, dummy_data):
    assert dummy_group.flagspace == dummy_data.flagspace
    g = dummy_group.copy(full=True)
    g.data = None
    assert g.flagspace is None


def test_apply_data(dummy_group):
    g = dummy_group.copy(full=True)
    g.empty_value = 2
    data = g.data.copy()
    g2 = g.copy(full=True)
    g2.empty_value = 1
    g.apply_data(g2)
    g.apply_data(g2)
    assert np.allclose(g.empty_value, 1)
    data.empty_value[...] = 5
    g.apply_data(data)
    assert np.allclose(g.empty_value, 5)

    class GenericThing(object):
        def __init__(self, data_in):
            self.data = data_in
    thing = GenericThing(data)
    g = dummy_group.copy(full=True)
    g.apply_data(thing)
    assert np.allclose(g.empty_value, 5)

    with pytest.raises(ValueError) as err:
        g.apply_data(1)
    assert 'Flagged data must be' in str(err.value)


def test_create_group(dummy_group):
    g = dummy_group.copy(full=True)
    g2 = g.create_data_group()
    assert np.allclose(g.matrix.toarray(), g2.matrix.toarray())
    assert g2.name == g.name
    g2 = g.create_data_group(name='new')
    assert g2.name == 'new'

    g2 = g.create_data_group(indices=np.arange(3))
    assert g2.size == 3
    g.flag = np.arange(5)
    g2 = g.create_data_group(indices=np.arange(4), keep_flag=1)
    assert np.allclose(g2.indices, [1, 3])


def test_delete_indices(dummy_group):
    g = dummy_group.copy(full=True)
    g.delete_indices(4)
    assert np.allclose(g.indices, np.arange(4))


def test_discard_flag(dummy_group):
    g = dummy_group.copy(full=True)
    g.flag = np.arange(5)
    g.discard_flag(1)
    assert np.allclose(g.indices, [0, 2, 4])


def test_set_flags(dummy_group):
    g = dummy_group.copy(full=True)
    base = np.arange(5)
    g.flag = base.copy()
    g.set_flags(10, indices=np.zeros(0, dtype=int))
    assert np.allclose(g.flag, base)

    g.flag = base.copy()
    g.set_flags(10)
    assert np.allclose(g.flag, [10, 11, 10, 11, 14])

    g.flag = base.copy()
    g.set_flags('GAIN')
    assert np.allclose(g.flag, [8, 9, 10, 11, 12])


def test_unflag(dummy_group):
    g = dummy_group.copy(full=True)
    base = np.arange(5)
    g.flag = base.copy()
    g.unflag(1, indices=np.zeros(0, dtype=int))
    assert np.allclose(g.flag, base)
    g.unflag(1)
    assert np.allclose(g.flag, [0, 0, 2, 2, 4])

    g.flag = base.copy()
    g.unflag('BLIND')
    assert np.allclose(g.flag, [0, 1, 0, 1, 4])


def test_reindex(dummy_group):
    g = dummy_group.copy(full=True)
    g._fixed_indices = np.arange(4) + 1
    assert np.allclose(g.indices, np.arange(5))
    g.reindex()
    assert np.allclose(g.indices, np.arange(4) + 1)


def test_new_indices_in_old(dummy_group):
    g = dummy_group.copy(full=True)
    g._fixed_indices = np.arange(4) + 1
    assert np.allclose(g.new_indices_in_old(), np.arange(4))
