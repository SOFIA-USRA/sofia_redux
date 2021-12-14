# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.tree.kernel_tree import KernelTree

import numpy as np


def test_expected_1d():
    x = np.linspace(0, np.pi, 21)
    y = np.sin(x)
    tree = KernelTree((5,), kernel=y, kernel_offsets=x)
    result = tree.spline(x)
    assert np.allclose(result, y)

    tree = KernelTree((5,), kernel=y, kernel_spacing=np.pi / 20)
    x_offset = x - np.pi / 2
    assert np.allclose(tree.spline(x_offset), y)


def test_expected_2d():
    y, x = np.mgrid[:31, :33]
    y2 = (y - 15) ** 2
    x2 = (x - 16) ** 2
    a = -0.1
    b = -0.15
    kernel = np.exp(a * x2 + b * y2)
    tree = KernelTree((5, 5), kernel=kernel)
    spline = tree.spline
    ny, nx = kernel.shape
    x = np.linspace(-16, 16, nx)
    y = np.linspace(-15, 15, ny)
    result = spline(x, y)
    assert result.shape == kernel.shape
    assert np.allclose(result, kernel)


def test_expected_3d():
    z, y, x = np.mgrid[:9, :10, :11]
    kernel = -np.sin(10 * ((x ** 2) + (y ** 2) + (z ** 2)))
    tree = KernelTree((5, 5, 5), kernel=kernel, degrees=2)
    xg = np.arange(11) - 5.0
    yg = np.arange(10) - 4.5
    zg = np.arange(9) - 4.0
    assert np.allclose(tree.spline(xg, yg, zg), kernel)
