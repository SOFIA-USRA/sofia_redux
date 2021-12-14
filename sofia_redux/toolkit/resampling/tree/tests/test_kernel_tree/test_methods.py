# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.kernel_tree import KernelTree

import numpy as np
import pytest


@pytest.fixture
def kernel_2d():
    y, x = np.mgrid[:17, :18]
    y2 = (y - 8) ** 2
    x2 = (x - 8.5) ** 2
    a = -0.1
    b = -0.15
    data = np.exp(a * x2 + b * y2)
    return data


def test_set_kernel(kernel_2d):
    # kernel shape is (ny, nx) -> (17, 18)
    tree = KernelTree((5, 6))
    # Test all parameters are being passed
    tree.set_kernel(kernel_2d, kernel_spacing=0.5, degrees=[2, 3],
                    smoothing=0.0, imperfect=False)

    coords = tree.kernel_coordinates.copy()
    assert not tree.imperfect
    assert tree.smoothing == 0
    assert np.allclose(tree.degrees, [2, 3])
    assert np.allclose(tree.kernel_spacing, [0.5, 0.5])
    assert coords.shape == (2, 306)
    assert coords[0].min() == -4.25
    assert coords[0].max() == 4.25
    assert coords[1].min() == -4
    assert coords[1].max() == 4

    xy = coords.copy()
    tree.set_kernel(kernel_2d.ravel(), imperfect=True,
                    kernel_offsets=xy, degrees=3, smoothing=1.0)
    assert tree.smoothing == 1
    assert np.allclose(tree.kernel_coordinates, coords)
    assert tree.imperfect


def test_parse_kernel(kernel_2d):
    # kernel shape is (ny, nx) -> (17, 18)
    tree = KernelTree((5, 6))
    ny, nx = kernel_2d.shape
    # Test regular grid (scalar spacing)
    tree.parse_kernel(kernel_2d, kernel_spacing=1.0)
    coords = tree.kernel_coordinates.copy()
    ux, uy = np.unique(coords[0]), np.unique(coords[1])
    assert ux.size == nx
    assert uy.size == ny
    assert np.allclose([ux.min(), ux.max()], [-8.5, 8.5])
    assert np.allclose([uy.min(), uy.max()], [-8, 8])

    # Test regular grid (different spacing)
    tree.parse_kernel(kernel_2d, kernel_spacing=[0.5, 2])
    coords = tree.kernel_coordinates.copy()
    ux, uy = np.unique(coords[0]), np.unique(coords[1])
    assert np.allclose(tree.kernel_spacing, [0.5, 2])
    assert coords.shape == (2, 306)
    assert ux.size == nx
    assert uy.size == ny
    assert np.allclose([ux.min(), ux.max()], [-4.25, 4.25])
    assert np.allclose([uy.min(), uy.max()], [-16, 16])

    # Supply irregular kernel
    xy = tree.kernel_coordinates.copy()
    tree = KernelTree((5, 6))
    tree.parse_kernel(kernel_2d.ravel(), kernel_offsets=xy)
    assert np.allclose(xy, tree.kernel_coordinates)

    # Supply grid indices
    tree = KernelTree((5, 6))
    kernel = kernel_2d
    x = np.linspace(-4.25, 4.25, nx)
    y = np.linspace(-16, 16, ny)
    tree.parse_kernel(kernel, kernel_offsets=(x, y))
    assert np.allclose(tree.kernel_coordinates, xy)

    with pytest.raises(ValueError) as err:
        tree.parse_kernel(kernel, kernel_offsets=None, kernel_spacing=None)
    assert "Must supply either kernel_spacing or kernel_offsets" in str(
        err.value)

    with pytest.raises(ValueError) as err:
        tree.parse_kernel(np.empty((3, 3, 3)))
    assert "Kernel must have the same number of dimensions" in str(err.value)

    with pytest.raises(ValueError) as err:
        tree.parse_kernel(kernel_2d, kernel_spacing=[1, 2, 3])
    assert "Kernel spacing size does not equal" in str(err.value)

    with pytest.raises(ValueError) as err:
        tree.parse_kernel(kernel_2d.ravel(), kernel_offsets=(np.arange(3),
                                                             np.arange(5)))
    assert "Irregular kernel offsets should be supplied as" in str(err.value)

    with pytest.raises(ValueError) as err:
        tree.parse_kernel(kernel_2d.ravel(), kernel_offsets=np.empty((2, 2)))
    assert "Irregular kernel offsets do not match" in str(err.value)

    tree = KernelTree((5,))
    with pytest.raises(ValueError) as err:
        tree.parse_kernel(np.arange(10), kernel_offsets=np.arange(5))
    assert "1-D kernel coordinates do not match" in str(err.value)

    # Test the 1-D case
    tree.parse_kernel(np.arange(10), kernel_offsets=np.arange(10) - 5.0)
    assert tree.kernel_coordinates.shape == (1, 10)
    assert np.allclose(tree.kernel_coordinates,
                       [[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]])


def test_fit_spline(kernel_2d):
    tree = KernelTree((5, 5))
    tree.parse_kernel(kernel_2d, kernel_spacing=[0.5, 2])
    tree.fit_spline(degrees=2, smoothing=0.0)
    assert "interpolating spline" in tree.exit_message

    with pytest.raises(RuntimeError) as err:
        tree.fit_spline(degrees=3, smoothing=0.0)
    assert "Unsuccessful fit" in str(err.value)

    tree.fit_spline(degrees=3, smoothing=1)
    assert tree.exit_code == 0

    tree.fit_spline(degrees=3, smoothing=1, solve=False)
    assert tree.exit_code == 0


def test_init(kernel_2d):
    tree = KernelTree((4, 4))
    for attribute in ['kernel', 'kernel_spacing', 'kernel_coordinates',
                      'spline']:
        assert getattr(tree, attribute, -1) is None
    assert not tree.imperfect

    tree = KernelTree((4, 4), imperfect=True)
    assert tree.imperfect

    tree = KernelTree((4, 4), kernel=kernel_2d, degrees=3, smoothing=1)
    assert tree.exit_code == 0
