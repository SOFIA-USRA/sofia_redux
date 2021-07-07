# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.fill import polyfillaa


def test_expected_single_output():
    px = [1.5, 1.5, 3.5, 3.5]
    py = [4.5, 6.5, 6.5, 4.5]
    result = polyfillaa(px, py)
    assert result.shape == (9, 2)
    assert np.allclose(np.unique(result[:, 0]), [4, 5, 6])
    assert np.allclose(np.unique(result[:, 1]), [1, 2, 3])
    result2, area = polyfillaa(px, py, area=True)
    assert np.allclose(result, result2)
    assert area.shape == (9,)
    assert np.allclose(np.unique(area), [0.25, 0.5, 1])


def test_expected_poly_output():
    px = np.array([1.5, 1.5, 2.5, 2.5] * 10)
    py = np.array([3.5, 4.5, 4.5, 3.5] * 10)
    start_indices = np.arange(10) * 4
    result, area = polyfillaa(px, py, start_indices=start_indices, area=True)
    assert len(result) == 10
    assert isinstance(result, dict)
    assert isinstance(area, dict)
    assert np.allclose(list(result.keys()), np.arange(10))
    assert np.allclose(list(area.keys()), np.arange(10))
    afull = np.array(list(area.values()))
    assert (afull == 0.25).all()
    for v in result.values():
        assert np.allclose(v, [[3, 1], [3, 2], [4, 1], [4, 2]])


def test_multi_shape():
    px = [[1.5, 0.5, 0.5], [2, 2, 3, 3]]  # a triangle and square
    py = [[0.5, 0.5, 1.5], [2, 3, 3, 2]]
    result, area = polyfillaa(px, py, area=True)
    assert np.allclose(result[0], [[0, 0], [0, 1], [1, 0]])
    assert np.allclose(result[1], [[2, 2]])
    assert np.allclose(area[0], [0.25, 0.125, 0.125])
    assert np.allclose(area[1], 1)


def test_grid_points_found():
    px = np.array([1.5, 0.5, 0.5])  # a triangle
    py = np.array([0.5, 0.5, 1.5])
    result, area = polyfillaa(px, py, area=True)
    # edge should cross exactly at (1, 1) but not enclosing the cell
    idx = np.where((result[:, 0] == 1) & (result[:, 1] == 1))
    assert len(idx) == 1
    assert result.shape == (3, 2)
    assert area.sum() == 0.5


def test_options():
    px = np.array([1.5, 0.5, 0.5])  # a triangle
    py = np.array([0.5, 0.5, 1.5])
    # cut off everything useful
    result, area = polyfillaa(px, py, xrange=[1, 2], yrange=[1, 2], area=True)
    assert result.shape == (0, 2)
    assert area.shape == (0,)


def test_errors():
    px = np.array([1.5, 0.5, 0.5])  # a triangle
    py = np.array([0.5, 0.5, 1.5])
    py_bad = py[:2]
    with pytest.raises(ValueError) as err:
        _ = polyfillaa(px, py_bad, xrange=[1, 2], yrange=[1, 2], area=True)
    assert "px and py must be the same shape" in str(err.value)

    px_bad = px[None]
    py_bad = py[None]
    with pytest.raises(ValueError) as err:
        _ = polyfillaa(px_bad, py_bad, xrange=[1, 2], yrange=[1, 2],
                       area=True, start_indices=[0])
    assert "polygons must be flat arrays" in str(err.value)


def test_parallel():
    px = np.array([1, 1, 2.5, 2.5])
    py = np.array([1, 2.5, 2.5, 1])
    result, area = polyfillaa(px, py, area=True)
    assert np.allclose(area, [1, 0.5, 0.5, 0.25])
    assert np.allclose(result, [[1, 1], [1, 2], [2, 1], [2, 2]])
