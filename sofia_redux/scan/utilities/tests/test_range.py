# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest
import regex

from sofia_redux.scan.utilities.range import Range


def test_init():
    # Test defaults
    r = Range()
    assert np.isinf(r.min) and r.min < 0
    assert np.isinf(r.max) and r.max > 0
    assert r.include_max
    assert r.include_min

    r = Range(1, 3, include_min=False)
    assert isinstance(r.min, float) and r.min == 1
    assert isinstance(r.max, float) and r.max == 3
    assert r.include_max
    assert not r.include_min

    r = Range(1, 3, include_max=False)
    assert r.include_min
    assert not r.include_max

    r = Range(1 * units.Unit('s'), 3 * units.Unit('s'))
    assert r.min.value == 1
    assert r.max.value == 3
    assert r.min.unit == units.Unit('s')
    assert r.max.unit == units.Unit('s')
    assert r.include_min and r.include_max

    r = Range(1 * units.Unit('s'))
    assert r.min.value == 1 and r.min.unit == units.Unit('s')
    assert np.isinf(r.max) and r.max > 0 and r.max.unit == units.Unit('s')

    r = Range(max_val=3 * units.Unit('s'))
    assert r.max.value == 3 and r.max.unit == units.Unit('s')
    assert np.isinf(r.min) and r.min < 0 and r.min.unit == units.Unit('s')

    # Test unit errors
    with pytest.raises(ValueError) as err:
        Range(1, 3 * units.Unit('s'))
    assert "Range units are incompatible" in str(err.value)

    with pytest.raises(ValueError) as err:
        Range(1 * units.Unit('s'), 3)
    assert "Range units are incompatible" in str(err.value)

    with pytest.raises(ValueError) as err:
        Range(1 * units.Unit('m'), 3 * units.Unit('s'))
    assert "Range units are incompatible" in str(err.value)

    with pytest.raises(ValueError) as err:
        Range(1 * units.Unit('s'), 3 * units.Unit('m'))
    assert "Range units are incompatible" in str(err.value)

    # Check compatible units
    Range(1 * units.Unit('second'), 1 * units.Unit('minute'))


def test_copy():
    r = Range(1 * units.Unit('second'), 2 * units.Unit('second'),
              include_min=True, include_max=False)
    r2 = r.copy()
    assert r.min == r2.min
    assert r.max == r2.max
    assert r.include_min is r2.include_min
    assert r.include_max is r2.include_max
    assert r is not r2


def test_midpoint():
    assert Range(1, 3).midpoint == 2
    r = Range(1 * units.Unit('minute'), 180 * units.Unit('second'))
    assert r.midpoint == 2 * units.Unit('minute')


def test_span():
    assert Range(1, 3).span == 2
    assert Range(
        1 * units.Unit('s'), 3 * units.Unit('s')).span == 2 * units.Unit('s')
    assert Range(3, 1).span == 0
    assert Range(
        3 * units.Unit('s'), 1 * units.Unit('s')).span == 0 * units.Unit('s')


def test_eq():
    r = Range(1, 3)
    assert r == Range(1, 3)
    assert r != 1
    assert r != Range(0, 3)
    assert r != Range(1, 4)
    assert r != Range(1, 3, include_min=False)
    assert r != Range(1, 3, include_max=False)
    assert r == r


def test_str():
    assert f'{Range(1, 3)}' == '(1.0 -> 3.0)'


def test_repr():
    r = Range(1, 3)
    assert regex.search(
        r'.*Range object at.*\(1.0 -> 3.0\)', repr(r)) is not None


def test_call():
    r = Range(1, 3)
    assert r(1) and r(2) and r(3)
    assert not r(0) and not r(4)
    assert np.allclose(r(np.arange(5)), [False, True, True, True, False])


def test_contains():
    r = Range(1, 3)
    for i in range(5):
        if i < 1 or i > 3:
            assert i not in r
        else:
            assert i in r

    r2 = Range(2, 3)
    assert r2 in r
    r2 = Range(2, 4)
    assert r2 not in r


def test_in_range():
    r = Range(1 * units.Unit('minute'), 3 * units.Unit('minute'))
    assert not r(30 * units.Unit('second'))
    assert not r(1 * units.Unit('hour'))
    assert r(120 * units.Unit('second'))
    assert r(1 * units.Unit('minute'))
    assert r(3 * units.Unit('minute'))

    r = Range(1 * units.Unit('minute'), 3 * units.Unit('minute'),
              include_min=False, include_max=False)
    assert not r(1 * units.Unit('minute'))
    assert not r(3 * units.Unit('minute'))


def test_from_spec():
    r = Range.from_spec('*')
    assert np.isinf(r.min) and r.min < 0
    assert np.isinf(r.max) and r.max > 0

    r = Range.from_spec('>=2')
    assert r.min == 2 and np.isinf(r.max) and r.max > 0
    assert r.include_min

    r = Range.from_spec('<=2')
    assert r.max == 2 and np.isinf(r.min) and r.min <= 0
    assert r.include_max

    r = Range.from_spec('>2')
    assert r.min == 2 and np.isinf(r.max) and r.max > 0
    assert not r.include_min

    r = Range.from_spec('<2')
    assert r.max == 2 and np.isinf(r.min) and r.min <= 0
    assert not r.include_max

    r = Range.from_spec('1:3')
    assert r.min == 1
    assert r.max == 3
    assert r.include_max and r.include_min

    r = Range.from_spec('*:3')
    assert np.isinf(r.min) and r.min < 0
    assert r.max == 3

    r = Range.from_spec('1:*')
    assert r.min == 1
    assert np.isinf(r.max) and r.max > 0

    assert Range.from_spec(None) is None

    r = Range.from_spec('1')
    assert r.min == 1 and r.max == 1

    with pytest.raises(ValueError) as err:
        Range.from_spec('1:2:3')
    assert 'Incorrect range spec' in str(err.value)

    # Test positive switch
    r = Range.from_spec('-1:3')
    assert r.min == -1 and r.max == 3
    with pytest.raises(ValueError) as err:
        Range.from_spec('-1:3', is_positive=True)
    assert 'Incorrect range spec' in str(err.value)

    r = Range.from_spec('1-3', is_positive=True)
    assert r.min == 1 and r.max == 3


def test_intersect_with():
    r = Range(2, 5)
    r.intersect_with(Range(1, 4))
    assert r.min == 2 and r.max == 4
    r.intersect_with(Range(max_val=3))
    assert r.min == 2 and r.max == 3

    r = Range(2, 5)
    r.intersect_with(1, 4)
    assert r.min == 2 and r.max == 4
    r.intersect_with(0, 3)
    assert r.min == 2 and r.max == 3

    r.intersect_with(3, 4)
    assert r.min == 3 and r.max == 3

    with pytest.raises(ValueError) as err:
        r.intersect_with(1, 2, 3)
    assert "Intersection requires two arguments" in str(err.value)


def test_include_value():
    r = Range(1, 3)
    r.include_value(2)
    assert r.min == 1 and r.max == 3
    r.include_value(0)
    assert r.min == 0 and r.max == 3
    r.include_value(4)
    assert r.min == 0 and r.max == 4
    r.include_value(np.nan)
    assert r.min < 0 and np.isinf(r.min)
    assert r.max > 0 and np.isinf(r.max)


def test_include():
    r = Range(1, 3)
    r.include(0)
    assert r.min == 0 and r.max == 3
    r.include(Range(1, 4))
    assert r.min == 0 and r.max == 4
    r.include(Range(-1, 5))
    assert r.min == -1 and r.max == 5


def test_scale():
    r = Range(1, 3)
    r.scale(units.Unit('km'))
    assert r.min.decompose().value == 1000
    assert r.max.decompose().value == 3000
    assert r.min.unit == units.Unit('km')
    assert r.max.unit == units.Unit('km')


def test_flip():
    r = Range(1, 3)
    r.flip()
    assert r.min == 3
    assert r.max == 1


def test_empty():
    r = Range(1, 3)
    r.empty()
    assert np.isinf(r.min) and r.min > 0
    assert np.isinf(r.max) and r.max < 0


def test_is_empty():
    r = Range(3, 1)
    assert r.is_empty()
    r = Range(1, 3)
    assert not r.is_empty()
    r.empty()
    assert r.is_empty()


def test_full():
    r = Range(1, 3)
    r.full()
    assert np.isinf(r.min) and r.min < 0
    assert np.isinf(r.max) and r.max > 0


def test_lower_bounded():
    r = Range(1, 3)
    assert r.lower_bounded
    r.min = -np.inf
    assert not r.lower_bounded


def test_upper_bounded():
    r = Range(1, 3)
    assert r.upper_bounded
    r.max = np.inf
    assert not r.upper_bounded


def test_bounded():
    r = Range(1, 3)
    assert r.bounded
    r.min = -np.inf
    assert not r.bounded
    r = Range(1, np.inf)
    assert not r.bounded


def test_is_intersecting():
    r = Range(1, 3)
    r2 = Range(0, 2)
    assert r.is_intersecting(r2)
    assert r2.is_intersecting(r)
    r2 = Range(2, 4)
    assert r.is_intersecting(r2)
    assert r2.is_intersecting(r)
    r2 = Range(0, 5)
    assert r.is_intersecting(r2)
    assert r2.is_intersecting(r)
    r2 = Range(6, 8)
    assert not r.is_intersecting(r2)
    assert not r2.is_intersecting(r)
    r2 = Range()
    r2.empty()
    assert not r.is_intersecting(r2)
    assert not r2.is_intersecting(r)


def test_grow():
    r = Range(1, 3)
    r.grow(1)
    assert r.min == 1
    assert r.max == 3
    r.grow(2)
    assert r.min == 0
    assert r.max == 4
    r = Range(1, np.inf)
    r0 = r.copy()
    r.grow(2)
    assert r == r0


def test_full_range():
    r = Range.full_range()
    assert r.min < 0 and np.isinf(r.min)
    assert r.max > 0 and np.isinf(r.max)


def test_positive_range():
    r = Range.positive_range()
    assert r.min == 0
    assert r.max > 0 and np.isinf(r.max)


def test_negative_range():
    r = Range.negative_range()
    assert r.max == 0
    assert r.min < 0 and np.isinf(r.min)
