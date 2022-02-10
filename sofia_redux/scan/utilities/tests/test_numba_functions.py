# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.utilities import numba_functions as nf
from sofia_redux.toolkit.splines.spline import Spline
from sofia_redux.toolkit.splines.spline_utils import flat_index_mapping


def test_smart_median_1d():
    x = np.arange(10).astype(float)
    w = np.full(10, 1.0)
    y = nf.smart_median_1d(x)
    assert y == (4.5, 10)
    y = nf.smart_median_1d(x, weights=w)
    assert y == (4, 10)  # No longer use the standard mid-point definition

    y = nf.smart_median_1d(np.full(1, 2.0), weights=np.full(1, 3.0))
    assert y == (2, 3)
    y = nf.smart_median_1d(np.full(1, 2.0))
    assert y == (2, 1)

    w[5] = np.nan
    x[4] = np.nan
    y = nf.smart_median_1d(x, weights=w)
    assert y == (35 / 6, 9)

    w = np.zeros(x.size)
    w[np.isnan(x)] = 1.0
    y = nf.smart_median_1d(x, weights=w)
    assert y == (0, 0)

    x = np.arange(11).astype(float)
    y = nf.smart_median_1d(x)
    assert y == (5, 11)

    w = np.ones(x.size)
    w[5] = 1000
    w[4] = np.nan
    y = nf.smart_median_1d(x, weights=w, max_dependence=0.5)
    assert np.allclose(y, [5.00099108, 1009.0], atol=1e-7)


def test_smart_median_2d():
    x = np.arange(132).reshape(11, 12).astype(float).T
    y = nf.smart_median_2d(x)
    assert np.allclose(y[0], np.arange(60, 72))
    assert np.allclose(y[1], 11)
    weights = np.full(x.shape, 2.0)
    y = nf.smart_median_2d(x, weights=weights)
    assert np.allclose(y[0], np.arange(60, 72))
    assert np.allclose(y[1], 22)


def test_smart_median():
    x = np.arange(132).reshape(11, 12).astype(float).T
    y = nf.smart_median(x)
    assert y == (65, 132)
    y = nf.smart_median(x, axis=0)
    assert np.allclose(y[0], np.arange(5, 126, 12))
    assert np.allclose(y[1], 12)
    y = nf.smart_median(x, axis=1)
    assert np.allclose(y[0] * 11, np.arange(654, 776, 11))
    assert np.allclose(y[1], 11)

    x = np.arange(60).reshape((3, 4, 5))
    y = nf.smart_median(x, axis=0)
    assert np.allclose(y[0] * 3, np.arange(50, 108, 3).reshape(4, 5))
    assert np.allclose(y[1], 3)
    y = nf.smart_median(x, axis=2)
    assert np.allclose(y[0] * 5, np.arange(9.5, 285, 25).reshape(3, 4))
    assert np.allclose(y[1], 5) and y[1].shape == (3, 4)


def test_roundup_ratio():
    assert nf.roundup_ratio(4, 4) == 1
    assert nf.roundup_ratio(5, 4) == 2
    assert nf.roundup_ratio(8, 4) == 2
    assert nf.roundup_ratio(9, 4) == 3


def test_level():
    x = np.linspace(10, 20, 101)
    x0 = x.copy()
    avg = nf.level(x)
    assert avg == 15
    assert np.allclose(x + avg, x0)
    nan = np.full(10, np.nan)
    avg = nf.level(nan)
    assert avg == 0
    assert np.isnan(nan).all()

    # Check start and end indices
    x = x0.copy()
    avg = nf.level(x, start=10, end=21)
    assert avg == 11.5
    assert np.allclose(x[10:21] + avg, x0[10:21])
    assert np.allclose(x[21:], x0[21:])

    # Check resolution (defines resolution of start and end indices).
    x = x0.copy()
    avg = nf.level(x, start=40, end=81, resolution=4)
    assert avg == 11.5
    assert np.allclose(x[10:21] + avg, x0[10:21])
    assert np.allclose(x[21:], x0[21:])


def test_smooth_1d():
    x = np.zeros(100)
    x[50] = 2.0
    kernel = np.array([1, 2, 4, 2, 1]) / 10
    x[20:30] = np.nan
    nf.smooth_1d(x, kernel)
    assert np.allclose(x[48: 53], 2 * kernel)
    assert np.allclose(x[:48], 0)
    assert np.allclose(x[53:], 0)


def test_gaussian_kernel():
    g = nf.gaussian_kernel(11, 2)
    assert np.allclose(
        g,
        [0.04393693, 0.13533528, 0.32465247, 0.60653066, 0.8824969, 1,
         0.8824969, 0.60653066, 0.32465247, 0.13533528, 0.04393693])


def test_mean():
    x = np.linspace(10, 20, 101)
    weights = np.full(x.size, 2.0)

    y = nf.mean(x)
    assert y == (15, 101)

    y = nf.mean(x, weights=weights)
    assert y == (15, 202)

    weights[:50] = 0.0
    y = nf.mean(x, weights=weights)
    assert y == (17.5, 102)

    y = nf.mean(np.full(x.size, np.nan), weights=weights)
    assert y == (0, 0)


def test_box_smooth_along_zero_axis():
    x = np.arange(60).reshape(3, 20)
    result = nf.box_smooth_along_zero_axis(x, 1)
    assert np.allclose(result, x, equal_nan=True)

    result = nf.box_smooth_along_zero_axis(x, 2)
    expected = x + 0.5
    expected[:, -1] = np.nan
    assert np.allclose(result, expected, equal_nan=True)

    result = nf.box_smooth_along_zero_axis(x, 3)
    expected = x.astype(float)
    expected[:, 0] = np.nan
    expected[:, -1] = np.nan
    assert np.allclose(result, expected, equal_nan=True)

    # Test validity
    valid = np.full(x.shape[1], True)
    valid[10] = False
    result = nf.box_smooth_along_zero_axis(x, 4, valid=valid)

    expected = x + 0.5
    expected[:, 0] = np.nan
    expected[:, -2:] = np.nan
    expected[:, 8] = x[:, 8]
    expected[:, 9] = x[:, 9] + 1 / 3
    expected[:, 10] = x[:, 10] + 2 / 3
    expected[:, 11] = x[:, 11] + 1
    assert np.allclose(result, expected, equal_nan=True)


def test_log2round():
    assert nf.log2round(1024) == 10
    assert nf.log2round(1000) == 10
    assert nf.log2round(1200) == 10
    assert nf.log2round(700) == 9


def test_log2ceil():
    assert nf.log2ceil(1024) == 10
    assert nf.log2ceil(1025) == 11


def test_pow2round():
    assert nf.pow2round(7) == 8
    assert nf.pow2round(1025) == 1024


def test_pow2floor():
    assert nf.pow2floor(7) == 4
    assert nf.pow2floor(1025) == 1024


def test_pow2ceil():
    assert nf.pow2ceil(7) == 8
    assert nf.pow2ceil(1025) == 2048


def test_regular_kernel_convolve():
    # 2-dimensional mapping is mostly used by SOFSCAN...
    data = np.zeros((51, 51))
    data[30, 32] = 1.0
    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1
    kernel[2, 3] = 0.5
    kernel[2, 1] = 0.5
    kernel[1, 2] = 0.5
    kernel[3, 2] = 0.5
    c, w = nf.regular_kernel_convolve(data, kernel)
    c0, w0 = c.copy(), w.copy()
    # weights should either be 2, 2.5, or 3 depending on edges...
    assert set(np.unique(w)) == {2, 2.5, 3}
    cross_y = np.asarray([29, 30, 30, 30, 31])
    cross_x = np.asarray([32, 31, 32, 33, 32])
    assert np.allclose(c[cross_y, cross_x],
                       [1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6])
    mask = np.full(data.shape, True)
    mask[cross_y, cross_x] = False
    assert np.allclose(c[mask], 0)

    # Check reference index offset
    c, w = nf.regular_kernel_convolve(data, kernel,
                                      kernel_reference_index=np.zeros(2))
    oy, ox = np.nonzero(c)
    assert np.allclose(oy, cross_y - 2)
    assert np.allclose(ox, cross_x - 2)

    # Test weights
    weights = np.full(data.shape, 2.0)
    c, w = nf.regular_kernel_convolve(data, kernel, weight=weights)
    assert np.allclose(w, w0 * 2)
    assert np.allclose(c, c0)

    # Test validity
    valid = np.full(data.shape, True)
    valid[30, 31] = False
    c, w = nf.regular_kernel_convolve(data, kernel, valid=valid)
    assert np.allclose(c[cross_y, cross_x],
                       [1 / 6, 1 / 4, 2 / 5, 1 / 6, 1 / 6])
    assert np.allclose(w[cross_y, cross_x], [3, 2, 2.5, 3, 3])

    # Check 3-D
    data = np.zeros((11, 11, 11))
    data[5, 5, 5] = 1.0
    z, y, x = np.mgrid[:5, :5, :5]
    z, y, x = z - 2, y - 2, x - 2
    kernel = 3 - np.sqrt(x ** 2 + y ** 2 + z ** 2)
    kernel[kernel < 0] = 0
    c, w = nf.regular_kernel_convolve(data, kernel)
    # Check symmetrical about center
    assert np.allclose(c[5], c[:, 5])
    assert np.allclose(c[5], c[:, :, 5])
    assert np.isclose(c.sum(), 1)
    assert np.isclose(c.max() * kernel.sum(), 3)


def test_regular_coarse_kernel_convolve():
    data = np.zeros((41, 41))
    data[20, 20] = 1.0
    x, y = np.meshgrid(*([np.linspace(-3, 3, 7)] * 2))
    kernel = 3 - np.sqrt(x ** 2 + y ** 2)
    kernel[kernel < 0] = 0.0
    kernel /= 3
    steps = np.full(2, 2)
    c, w, shape = nf.regular_coarse_kernel_convolve(data, kernel, steps)
    assert np.allclose(shape, [21, 21])
    c = c.reshape(shape)
    w = w.reshape(shape)
    c0, w0 = c.copy(), w.copy()
    c_max = c.max()
    max_i = np.nonzero(c == c_max)
    assert max_i[0][0] == 10 and max_i[1][0] == 10
    assert np.isclose(c_max, 1 / kernel.sum())

    # Check kernel reference index
    c, w, shape = nf.regular_coarse_kernel_convolve(
        data, kernel, steps, kernel_reference_index=np.zeros(2))
    c = c.reshape(shape)
    max_i = np.nonzero(c == c.max())
    assert np.allclose(max_i[0], [8, 8, 9, 9])
    assert np.allclose(max_i[1], [8, 9, 8, 9])

    # Check weights
    weights = np.full(data.shape, 2.0)
    c, w, shape = nf.regular_coarse_kernel_convolve(
        data, kernel, steps, weight=weights)
    c = c.reshape(shape)
    w = w.reshape(shape)
    assert np.allclose(c, c0)
    assert np.allclose(w, w0 * 2)

    # Check validity
    valid = np.full(data.shape, True)
    valid[20, 21] = False
    c, w, shape = nf.regular_coarse_kernel_convolve(
        data, kernel, steps, valid=valid)
    c = c.reshape(shape)
    w = w.reshape(shape)
    c1 = c.copy()
    w1 = w.copy()
    diff = c - c0

    idx = np.nonzero(diff)
    assert np.allclose(idx[0], [9, 9, 10, 10, 11, 11])
    assert np.allclose(idx[1], [10, 11, 10, 11, 10, 11])
    assert np.allclose(diff[idx],
                       [0.00099159, 0.00017013, 0.0081563,
                        0.00271877, 0.00099159, 0.00017013])

    # Check the effect of a zero weight
    weights = np.ones(data.shape)
    weights[20, 21] = 0.0
    c, w, shape = nf.regular_coarse_kernel_convolve(
        data, kernel, steps, weight=weights)
    c, w = c.reshape(shape), w.reshape(shape)
    assert np.allclose(c, c1)
    assert np.allclose(w, w1)


def test_smooth_values_at():
    # This test checks that no transpose operation is performed due to
    # spline coordinates expressed as (x, y), but everything else as (y, x).
    data = np.zeros((51, 51))
    data[25, 25] = 1.0

    inds = np.indices((7, 9)) - np.array([3.0, 4.0])[:, None, None]
    dy, dx = abs(inds)
    inds[0] /= 3
    kernel = np.hypot(*inds)
    kernel = kernel.max() - kernel
    kernel /= kernel.max()
    sum_x = (dx * kernel).sum() / dx.sum()
    sum_y = (dy * kernel).sum() / dy.sum()
    skew_kernel = sum_y / sum_x
    assert skew_kernel > 1  # Test is vertical

    spline = Spline(kernel, exact=False, reduce_degrees=True)
    shaped_indices = np.indices(data.shape)
    syi, sxi = shaped_indices
    all_indices = np.stack([x.ravel() for x in np.indices(data.shape)])
    yi, xi = all_indices

    kernel_reference_index = (np.asarray(kernel.shape) - 1) / 2.0
    # Check no pixel offset
    assert np.allclose(kernel_reference_index % 1, 0)

    # This should be done using direct kernel resampling...
    s, w = nf.smooth_values_at(
        data=data,
        kernel=kernel,
        indices=all_indices,
        kernel_reference_index=kernel_reference_index,
        knots=spline.knots,
        coefficients=spline.coefficients,
        degrees=spline.degrees,
        panel_mapping=spline.panel_mapping,
        panel_steps=spline.panel_steps,
        knot_steps=spline.knot_steps,
        nk1=spline.nk1,
        spline_mapping=spline.spline_mapping,
        weight=None,
        valid=None
    )
    s_shaped = s.reshape(data.shape) * kernel.sum()
    ry, rx = abs(np.indices(data.shape) - 25)
    sum_rx = (rx * s_shaped).sum() / rx.sum()
    sum_ry = (ry * s_shaped).sum() / ry.sum()
    assert sum_ry / sum_rx > 1  # Still vertical

    # Now add a small offset to the kernel reference index, which should force
    # spline resampling on the kernel.
    offset_index = kernel_reference_index.copy()
    offset_index += 1e-4

    ss, ws = nf.smooth_values_at(
        data=data,
        kernel=kernel,
        indices=all_indices,
        kernel_reference_index=offset_index,
        knots=spline.knots,
        coefficients=spline.coefficients,
        degrees=spline.degrees,
        panel_mapping=spline.panel_mapping,
        panel_steps=spline.panel_steps,
        knot_steps=spline.knot_steps,
        nk1=spline.nk1,
        spline_mapping=spline.spline_mapping,
        weight=None,
        valid=None
    )
    # Not perfect because the kernel does not go to zero near edge, but...
    ss_shaped = ss.reshape(data.shape) * kernel.sum()
    ws_shaped = ws.reshape(data.shape)
    sum_rx = (rx * ss_shaped).sum() / rx.sum()
    sum_ry = (ry * ss_shaped).sum() / ry.sum()
    assert sum_ry / sum_rx > 1  # Still vertical

    # Check weighting
    sw, ww = nf.smooth_values_at(
        data=data,
        kernel=kernel,
        indices=all_indices,
        kernel_reference_index=offset_index,
        knots=spline.knots,
        coefficients=spline.coefficients,
        degrees=spline.degrees,
        panel_mapping=spline.panel_mapping,
        panel_steps=spline.panel_steps,
        knot_steps=spline.knot_steps,
        nk1=spline.nk1,
        spline_mapping=spline.spline_mapping,
        weight=np.full(data.shape, 2.0),
        valid=None
    )
    sw_shaped = sw.reshape(data.shape) * kernel.sum()
    ww_shaped = ww.reshape(data.shape)
    assert np.allclose(sw_shaped, ss_shaped)
    assert np.allclose(ww_shaped / ws_shaped, 2)

    # Check validity
    valid = np.full(data.shape, True)
    valid[25, 26] = False
    sv, wv = nf.smooth_values_at(
        data=data,
        kernel=kernel,
        indices=all_indices,
        kernel_reference_index=kernel_reference_index,
        knots=spline.knots,
        coefficients=spline.coefficients,
        degrees=spline.degrees,
        panel_mapping=spline.panel_mapping,
        panel_steps=spline.panel_steps,
        knot_steps=spline.knot_steps,
        nk1=spline.nk1,
        spline_mapping=spline.spline_mapping,
        weight=None,
        valid=valid
    )
    sv_shaped = sv.reshape(data.shape)
    wv_shaped = wv.reshape(data.shape)
    assert not np.allclose(sv_shaped * kernel.sum(), s_shaped)
    assert np.allclose(sv_shaped * wv_shaped, s_shaped)


def test_single_value_at():
    data = np.zeros((11, 11))
    data[5, 5] = 1.0
    kernel = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 0.25, 0.5, 0.25, 0],
         [0, 0.5, 1, 0.5, 0],
         [0, 0.25, 0.5, 0.25, 0],
         [0, 0, 0, 0, 0]])
    spline = Spline(kernel, exact=True, reduce_degrees=True)
    data_shape = np.asarray(data.shape)
    kernel_shape = np.asarray(kernel.shape)
    kernel_indices, _, kernel_steps = flat_index_mapping(kernel_shape)
    data_indices, _, data_steps = flat_index_mapping(data_shape)
    kernel_reference_index = np.full(2, 2.0)
    # Ignore weighting an validity for this test
    flat_weight = np.empty(0, dtype=float)
    flat_valid = np.empty(0, dtype=bool)

    # Check direct convolution
    for i in range(data_indices.shape[1]):
        s, w = nf.smooth_value_at(
            data=data,
            kernel=kernel,
            index=data_indices[:, i],
            kernel_indices=kernel_indices,
            kernel_reference_index=kernel_reference_index,
            knots=spline.knots,
            coefficients=spline.coefficients,
            degrees=spline.degrees,
            panel_mapping=spline.panel_mapping,
            panel_steps=spline.panel_steps,
            knot_steps=spline.knot_steps,
            nk1=spline.nk1,
            spline_mapping=spline.spline_mapping,
            data_shape=data_shape,
            kernel_shape=kernel_shape,
            data_steps=data_steps,
            flat_weight=flat_weight,
            flat_valid=flat_valid,
            weighted=False,
            validated=False)
        sw = s * w
        if i in [48, 50, 70, 72]:
            assert sw == 0.25
        elif i in [49, 59, 61, 71]:
            assert sw == 0.5
        elif i == 60:
            assert sw == 1
        else:
            assert sw == 0

    # Test interpolated convolution
    s, w = nf.smooth_value_at(
        data=data,
        kernel=kernel,
        index=np.array([5, 5.5]),
        kernel_indices=kernel_indices,
        kernel_reference_index=kernel_reference_index,
        knots=spline.knots,
        coefficients=spline.coefficients,
        degrees=spline.degrees,
        panel_mapping=spline.panel_mapping,
        panel_steps=spline.panel_steps,
        knot_steps=spline.knot_steps,
        nk1=spline.nk1,
        spline_mapping=spline.spline_mapping,
        data_shape=data_shape,
        kernel_shape=kernel_shape,
        data_steps=data_steps,
        flat_weight=flat_weight,
        flat_valid=flat_valid,
        weighted=False,
        validated=False)
    assert np.isclose(s * w, 0.75)


def test_point_aligned_smooth():
    data = np.zeros((13, 13))
    data[6, 6] = 1.0
    kernel = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 0.25, 0.5, 0.25, 0],
         [0, 0.5, 1, 0.5, 0],
         [0, 0.25, 0.5, 0.25, 0],
         [0, 0, 0, 0, 0]])
    flat_data = data.ravel()
    flat_kernel = kernel.ravel()
    weight = np.ones(data.shape)
    valid = np.ones(data.shape, dtype=bool)
    flat_weight = weight.ravel()
    flat_valid = valid.ravel()
    kernel_reference_index = np.full(2, 2.0)
    data_shape = np.asarray(data.shape)
    kernel_shape = np.asarray(kernel.shape)
    kernel_indices, _, kernel_steps = flat_index_mapping(kernel_shape)
    data_indices, _, data_steps = flat_index_mapping(data_shape)

    s, w = nf.point_aligned_smooth(
        flat_data=flat_data,
        flat_kernel=flat_kernel,
        flat_weight=flat_weight,
        flat_valid=flat_valid,
        data_index=np.asarray([6, 6]),
        kernel_indices=kernel_indices,
        kernel_reference_index=kernel_reference_index,
        data_shape=data_shape,
        data_steps=data_steps,
        validated=False,
        weighted=False
    )
    assert s == 0.25 and w == 4

    s, w = nf.point_aligned_smooth(
        flat_data=flat_data,
        flat_kernel=flat_kernel,
        flat_weight=flat_weight,
        flat_valid=flat_valid,
        data_index=np.asarray([6, 6]),
        kernel_indices=kernel_indices + 1,
        kernel_reference_index=kernel_reference_index,
        data_shape=data_shape,
        data_steps=data_steps,
        validated=True,
        weighted=True
    )
    assert s == 0.0625 and w == 4

    wt = flat_weight.copy()
    v = flat_valid.copy()
    wt[85] = 0.0
    v[85] = False

    s, w = nf.point_aligned_smooth(
        flat_data=flat_data,
        flat_kernel=flat_kernel,
        flat_weight=flat_weight,
        flat_valid=v,
        data_index=np.asarray([6, 6]),
        kernel_indices=kernel_indices,
        kernel_reference_index=kernel_reference_index,
        data_shape=data_shape,
        data_steps=data_steps,
        validated=True,
        weighted=False
    )
    assert np.isclose(s, 0.285714285) and w == 3.5

    s, w = nf.point_aligned_smooth(
        flat_data=flat_data,
        flat_kernel=flat_kernel,
        flat_weight=wt,
        flat_valid=flat_valid,
        data_index=np.asarray([6, 6]),
        kernel_indices=kernel_indices,
        kernel_reference_index=kernel_reference_index,
        data_shape=data_shape,
        data_steps=data_steps,
        validated=False,
        weighted=True
    )
    assert np.isclose(s, 0.285714285) and w == 3.5

    s, w = nf.point_aligned_smooth(
        flat_data=flat_data,
        flat_kernel=flat_kernel,
        flat_weight=flat_weight * 0,
        flat_valid=flat_valid,
        data_index=np.asarray([6, 6]),
        kernel_indices=kernel_indices,
        kernel_reference_index=kernel_reference_index,
        data_shape=data_shape,
        data_steps=data_steps,
        validated=False,
        weighted=True
    )
    assert s == 0 and w == 0

    # Invalid data indices
    for x in [-2, 1000]:
        s, w = nf.point_aligned_smooth(
            flat_data=flat_data,
            flat_kernel=flat_kernel,
            flat_weight=flat_weight * 0,
            flat_valid=flat_valid,
            data_index=np.asarray([6, x]),
            kernel_indices=kernel_indices,
            kernel_reference_index=kernel_reference_index,
            data_shape=data_shape,
            data_steps=data_steps,
            validated=False,
            weighted=True
        )
        assert s == 0 and w == 0


def test_point_smooth():
    data = np.zeros((9, 9))
    data[4, 4] = 1.0
    data_shape = np.asarray(data.shape)
    data_indices, _, data_steps = flat_index_mapping(data_shape)
    kernel = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 0.25, 0.5, 0.25, 0],
         [0, 0.5, 1, 0.5, 0],
         [0, 0.25, 0.5, 0.25, 0],
         [0, 0, 0, 0, 0]])
    kernel_reference_index = np.full(2, 2.0)
    kernel_shape = np.asarray(kernel.shape)
    kernel_indices, _, kernel_steps = flat_index_mapping(kernel_shape)
    spline = Spline(kernel, exact=True, reduce_degrees=True)
    weight = np.ones(data.shape)
    valid = np.ones(data.shape, dtype=bool)
    flat_data = data.ravel()
    flat_weight = weight.ravel()
    flat_valid = valid.ravel()

    kwargs = {
        'flat_data': flat_data,
        'kernel_indices': kernel_indices,
        'kernel_reference_index': kernel_reference_index,
        'knots': spline.knots,
        'coefficients': spline.coefficients,
        'degrees': spline.degrees,
        'panel_mapping': spline.panel_mapping,
        'panel_steps': spline.panel_steps,
        'knot_steps': spline.knot_steps,
        'nk1': spline.nk1,
        'spline_mapping': spline.spline_mapping,
        'data_shape': data_shape,
        'kernel_shape': kernel_shape,
        'data_steps': data_steps}

    s, w = nf.point_smooth(
        index=np.array([4, 4]),
        weighted=False,
        validated=False,
        flat_weight=flat_weight,
        flat_valid=flat_valid,
        **kwargs)
    assert np.isclose(s, 0.25) and w == 4

    s, w = nf.point_smooth(
        index=np.array([4.5, 4]),
        weighted=False,
        validated=False,
        flat_weight=flat_weight,
        flat_valid=flat_valid,
        **kwargs)
    assert np.isclose(s, 0.1875) and w == 4

    # Test points off the scale
    for y in [-10, 1000]:
        s, w = nf.point_smooth(
            index=np.array([y, 4]),
            weighted=False,
            validated=False,
            flat_weight=flat_weight,
            flat_valid=flat_valid,
            **kwargs)
        assert s == 0 and w == 0

    wt = flat_weight.copy()
    wt[41] = 0
    s, w = nf.point_smooth(
        index=np.array([4, 4]),
        weighted=True,
        validated=False,
        flat_weight=wt,
        flat_valid=flat_valid,
        **kwargs)
    assert np.isclose(s, 0.28571428571428) and w == 3.5

    v = flat_valid.copy()
    v[41] = False
    s, w = nf.point_smooth(
        index=np.array([4, 4]),
        weighted=False,
        validated=True,
        flat_weight=flat_weight,
        flat_valid=v,
        **kwargs)
    assert np.isclose(s, 0.28571428571428) and w == 3.5


def test_round_value():
    assert nf.round_value(1) == 1
    assert nf.round_value(2) == 2
    assert nf.round_value(1.5) == 2
    assert nf.round_value(2.5) == 3
    assert nf.round_value(1.4) == 1
    assert nf.round_value(1.6) == 2
    assert nf.round_value(2.4) == 2
    assert nf.round_value(2.6) == 3
    assert nf.round_value(-1.5) == -2
    assert nf.round_value(-2.5) == -3


def test_round_array():
    x = (np.arange(41) - 20) / 4
    r = nf.round_array(x)
    assert np.allclose(
        r,
        [-5, -5, -5, -4, -4, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2,
         -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
         4, 4, 4, 4, 5, 5, 5])


def test_sequential_array_add():
    x = np.zeros(10)
    add_values = np.ones(5)
    at_indices = np.arange(5)
    added = nf.sequential_array_add(x, add_values, at_indices)
    assert np.allclose(x[:5], 1) and np.allclose(x[5:], 0)
    assert np.all(added[:5]) and not np.any(added[5:])

    # Test all on multiple values
    x.fill(0)
    at_indices = np.array([0, 0, 3, 3, 3])
    added = nf.sequential_array_add(x, add_values, at_indices)
    assert added[0] and added[3]
    assert x[0] == 2 and x[3] == 3

    # Test valid_indices
    x.fill(0)
    vi = np.full(5, True)
    vi[-1] = False
    _ = nf.sequential_array_add(x, add_values, at_indices, valid_indices=vi)
    assert x[0] == 2 and x[3] == 2

    # Test valid_array
    x.fill(0)
    va = np.full(x.size, True)
    va[0] = False
    added = nf.sequential_array_add(x, add_values, at_indices, valid_array=va)
    assert not added[0] and x[0] == 0
    assert added[3] and x[3] == 3

    # Test invalid indices
    x.fill(0)
    indices = at_indices.copy()
    indices[2:] += 1000
    added = nf.sequential_array_add(x, add_values, indices)
    assert added[0] and not added[3] and x[0] == 2 and x[3] == 0

    # Test ND
    x = np.zeros((4, 5))
    indices = np.array([[0, 1, 3, 3, 3],
                        [1, 1, 1, 1, 1]])
    added = nf.sequential_array_add(x, add_values, indices)
    assert x[0, 1] == 1 and x[1, 1] == 1 and x[3, 1] == 3
    assert added[0, 1] and added[1, 1] and added[3, 1]

    x.fill(0)
    vi = np.full(5, True)
    vi[-1] = False
    _ = nf.sequential_array_add(x, add_values, indices, valid_indices=vi)
    assert x[0, 1] == 1 and x[1, 1] == 1 and x[3, 1] == 2


def test_index_of_max():
    x = np.zeros((5, 6))
    x[1, 2] = 1.0
    x[2, 3] = -2.0
    m, i = nf.index_of_max(x, sign=1)
    assert m == 1 and np.allclose(i, [1, 2])
    m, i = nf.index_of_max(x, sign=-1)
    assert m == -2 and np.allclose(i, [2, 3])
    m, i = nf.index_of_max(x, sign=0)
    assert m == -2 and np.allclose(i, [2, 3])

    x[3, 4] = 3.0
    assert nf.index_of_max(x)[0] == 3
    valid = np.full(x.shape, True)
    valid[3, 4] = False
    m, i = nf.index_of_max(x, valid=valid)
    assert m == 1 and np.allclose(i, [1, 2])

    nan = np.full(x.shape, np.nan)
    m, i = nf.index_of_max(nan)
    assert np.isnan(m) and np.allclose(i, -1)


def test_robust_mean():

    x = np.arange(100).astype(float)
    x[90:] += 1000

    m = nf.robust_mean(x)
    assert m == 149.5
    m = nf.robust_mean(x, tails=0.2)
    assert m == 49.5
    x[10] = np.nan
    m = nf.robust_mean(x, tails=0.2)
    assert m == 50

    x.fill(np.nan)
    m = nf.robust_mean(x, tails=0.2)
    assert np.isnan(m)
