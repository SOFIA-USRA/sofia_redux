from sofia_redux.toolkit.splines.spline import Spline
from sofia_redux.toolkit.splines import spline_utils as su

import itertools
import numpy as np
import pytest


@pytest.fixture
def knots2d4():
    degree = 4
    k1 = degree - 1
    line = np.arange(10, dtype=float)
    knot_line = np.concatenate([np.full(k1, 0), line, np.full(k1, 9)])
    knots = knot_line, knot_line
    valid_knot_start = np.full(2, degree)
    valid_knot_end = np.full(2, knot_line.size - degree - 1)
    return knots, valid_knot_start, valid_knot_end


@pytest.fixture
def uninitialized_spline_2d():
    coordinates = np.array([x.ravel() for x in np.mgrid[:10, :10]],
                           dtype=float)
    values = coordinates[0] + coordinates[1]
    spline = Spline(coordinates[0], coordinates[1], values, solve=False,
                    degrees=3)
    return spline


@pytest.fixture
def initialized_spline_2d(uninitialized_spline_2d):
    spline = uninitialized_spline_2d
    spline.initialize_iteration()
    amat, beta, splines, ssr = su.build_observation(
        coordinates=spline.coordinates,
        values=spline.values,
        weights=spline.weights,
        n_coefficients=spline.n_coefficients,
        bandwidth=spline.bandwidth,
        degrees=spline.degrees,
        knots=spline.knots,
        knot_steps=spline.knot_steps,
        start_indices=spline.start_indices,
        next_indices=spline.next_indices,
        panel_mapping=spline.panel_mapping,
        spline_mapping=spline.spline_mapping)
    return spline, amat, beta, splines, ssr


@pytest.fixture
def full_spline_2d():
    s_coordinates = np.array([x.ravel().copy() for x in np.mgrid[:10, :10]],
                             dtype=float)
    values = s_coordinates[0] + s_coordinates[1]
    spline = Spline(s_coordinates[0], s_coordinates[1], values, degrees=3)
    return spline


def test_find_knots(knots2d4):
    find_knots = su.find_knots
    knots, valid_knot_start, valid_knot_end = knots2d4
    knot_size = knots[0].size
    degree = 4

    coordinates = np.arange(4, dtype=float)
    coordinates = np.stack([coordinates, coordinates])
    coordinates[0] += 2.5
    coordinates[1] += 4

    # Test standard knot finding
    indices = find_knots(coordinates, knots, valid_knot_start, valid_knot_end)
    assert indices.shape == coordinates.shape
    for dimension in range(indices.shape[0]):
        for i in range(indices.shape[1]):
            index = indices[dimension, i]
            coord = coordinates[dimension, i]
            assert coord >= knots[dimension][index]
            assert coord < knots[dimension][index + 1]

    # Test coordinates outside the expected limits
    # Negative
    indices = find_knots(
        coordinates * -1, knots, valid_knot_start, valid_knot_end)
    assert np.allclose(indices, degree)
    indices = find_knots(
        coordinates + 100, knots, valid_knot_start, valid_knot_end)
    assert np.allclose(indices, knot_size - degree - 2)


def test_find_knot(knots2d4):
    find_knot = su.find_knot
    knots, valid_knot_start, valid_knot_end = knots2d4

    # Test a coordinate inside knot bounds
    coordinate = np.asarray([2.5, 4.5])
    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end)
    assert coordinate[0] >= knots[0][index[0]]
    assert coordinate[0] < knots[0][index[0] + 1]
    assert coordinate[1] >= knots[1][index[1]]
    assert coordinate[1] < knots[1][index[1] + 1]
    expected_index = index.copy()

    # Test coordinates outside knot bounds
    coordinate = np.asarray([-1.0, 100.0])
    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end)
    assert index[0] == valid_knot_start[0]
    assert index[1] == valid_knot_end[1] - 1

    # Test coordinate where nothing is allowed outside bounds
    coordinate = np.asarray([-1.0, 4.5])
    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end,
                      allow_outside=False)
    assert index[0] == -1

    coordinate = np.asarray([2.5, 100])
    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end,
                      allow_outside=False)
    assert index[0] == -1

    # Test coordinate bounds
    coordinate = np.asarray([2.5, 4.5])
    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end,
                      allow_outside=False, upper_bounds=np.asarray([5.0, 5]),
                      lower_bounds=np.asarray([0.0, 0]))
    assert np.allclose(index, expected_index)

    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end,
                      allow_outside=False, upper_bounds=np.asarray([5.0, 4]),
                      lower_bounds=np.asarray([0.0, 0]))
    assert index[0] == -1

    index = find_knot(coordinate, knots, valid_knot_start, valid_knot_end,
                      allow_outside=False, upper_bounds=np.asarray([5.0, 5]),
                      lower_bounds=np.asarray([3.0, 3.0]))
    assert index[0] == -1


def test_calculate_minimum_bandwidth():
    n_dimensions = 2
    n_knots = np.full(n_dimensions, 5)
    degrees = np.full(n_dimensions, 3)

    permutations = np.asarray(
        list(itertools.permutations(np.arange(n_dimensions))))

    min_bandwidth, perm_index, changed = su.calculate_minimum_bandwidth(
        degrees, n_knots, permutations)
    assert not changed
    assert np.allclose(perm_index, permutations[0])
    assert min_bandwidth == 7

    n_knots = np.array([9, 10])
    degrees = np.array([4, 2])
    min_bandwidth, perm_index, changed = su.calculate_minimum_bandwidth(
        degrees, n_knots, permutations)
    assert changed
    assert np.allclose(perm_index, permutations[1])
    assert min_bandwidth == 13


def test_flat_index_mapping():
    shape = np.asarray((2, 3, 4))
    n = np.prod(shape)
    flat = np.arange(n)
    shaped = flat.reshape(shape)
    map_indices, transpose_indices, step_size = su.flat_index_mapping(shape)

    assert np.allclose(shaped[tuple(map_indices)], flat)
    assert np.allclose(shaped.T.ravel()[transpose_indices], flat)
    assert np.allclose(step_size, [12, 4, 1])


def test_create_ordering():
    x = np.array([0, 0, 2, 1, 4, 2, 1, 2, 2, 1])
    start_indices, next_indices = su.create_ordering(x)

    for value in range(5):
        reverse_index = start_indices[value]
        while reverse_index != -1:
            assert x[reverse_index] == value
            reverse_index = next_indices[reverse_index]


def test_check_input_arrays():
    np.random.RandomState(0)
    shape = (3, 100)
    coordinates = np.random.random(shape)
    values = np.random.random(shape[1])
    weights = np.ones(shape[1])

    valid = su.check_input_arrays(values, coordinates, weights)
    assert valid.all() and valid.size == 100

    w = weights.copy()
    w[90:] = 0
    valid = su.check_input_arrays(values, coordinates, w)
    assert valid[:90].all() and not valid[90:].any()

    c = coordinates.copy()
    c[1, 99] = np.nan
    c[0, 98] = np.nan
    valid = su.check_input_arrays(values, c, weights)
    assert valid[:98].all() and not valid[98:].any()

    v = values.copy()
    v[:10] = np.nan
    valid = su.check_input_arrays(v, coordinates, weights)
    assert valid[10:].all() and not valid[:10].any()


def test_givens_parameters():
    y, c, s = su.givens_parameters(1, 2)
    sq5 = np.sqrt(5)
    assert np.isclose(y, sq5)
    assert np.isclose(c, 2 / sq5)
    assert np.isclose(s, 1 / sq5)

    y, c, s = su.givens_parameters(2, 1)
    sq5 = np.sqrt(5)
    assert np.isclose(y, sq5)
    assert np.isclose(c, 1 / sq5)
    assert np.isclose(s, 2 / sq5)


def test_givens_rotate():
    # Test no rotation
    cos = 1.0
    sin = 0.0
    x = 1.0
    y = 1.0
    xr, yr = su.givens_rotate(cos, sin, x, y)
    assert xr == 1 and yr == 1

    # 90 degree rotation (anti-clockwise)
    xr, yr = su.givens_rotate(0, 1, x, y)
    assert xr == -1 and yr == 1

    # 180 degree rotation
    xr, yr = su.givens_rotate(-1, 0, x, y)
    assert xr == -1 and yr == -1


def test_build_observation(initialized_spline_2d):
    spline, amat, beta, splines, ssr = initialized_spline_2d

    max_k1 = np.max(spline.degrees + 1)
    assert amat.shape == (spline.n_coefficients, spline.bandwidth)
    assert beta.shape == (spline.n_coefficients,)
    assert splines.shape == (spline.n_dimensions, spline.size, max_k1)
    assert isinstance(ssr, float) and np.isfinite(ssr)

    assert np.allclose(beta,
                       [11.97111626, 23.47169947, 23.89891411, 23.5216759,
                        23.47169947, 30.99807417, 28.46064834, 26.98636449,
                        23.89891411, 28.46064834, 25.18021535, 23.52742333,
                        23.5216759, 26.98636449, 23.52742333, 21.85059423])

    # Check zero elements where expected
    tri0, tri1 = np.tril_indices(amat.shape[0])
    tri0 += 1
    tri1 = amat.shape[0] - tri1
    keep = (tri0 < amat.shape[0]) & (tri1 < amat.shape[0])
    tri0, tri1 = tri0[keep], tri1[keep]
    assert np.allclose(amat[tri0, tri1], 0)

    # Check a single row of the A matrix
    assert np.allclose(
        amat[0],
        [1.84104162, 0.61542109, 0.25700689, 0.06430817,
         0.61542109, 0.20572219, 0.08591194, 0.02149686,
         0.25700689, 0.08591194, 0.03587781, 0.00897733,
         0.06430817, 0.02149686, 0.00897733, 0.00224631])

    assert np.allclose(splines[0, :10], [1, 0, 0, 0])
    assert np.allclose(splines[1, 0::10], [1, 0, 0, 0])
    # Just test one single value mid-way
    assert np.allclose(
        splines[0, 50], [0.0877915, 0.32921811, 0.41152263, 0.17146776])
    assert np.allclose(
        splines[1, 5], [0.0877915, 0.32921811, 0.41152263, 0.17146776])


def test_back_substitute(initialized_spline_2d):
    spline, amat, beta, splines, ssr = initialized_spline_2d
    coeffs = su.back_substitute(amat, beta, spline.n_coefficients,
                                spline.bandwidth)
    assert np.allclose(coeffs,
                       [0, 3, 6, 9, 3, 6, 9, 12, 6, 9, 12, 15, 9, 12, 15, 18])


def test_solve_rank_deficiency(initialized_spline_2d):
    spline, amat, beta, splines, ssr = initialized_spline_2d
    spline.tolerance = 1e-3
    coeffs, ssr, rank = su.solve_rank_deficiency(
        amat, beta, spline.n_coefficients, spline.bandwidth, spline.tolerance)

    assert rank == 16
    assert np.isclose(ssr, 0)
    assert np.allclose(coeffs,
                       [0, 3, 6, 9, 3, 6, 9, 12, 6, 9, 12, 15, 9, 12, 15, 18])

    amat[10, 0] = 0.0
    coeffs, ssr, rank = su.solve_rank_deficiency(
        amat, beta, spline.n_coefficients, spline.bandwidth, spline.tolerance)
    assert np.allclose(coeffs,
                       [0.27835415, 2.32138976, 5.54259091, 9.35340415,
                        1.46623901, 9.50498082, 12.13083546, 9.85458694,
                        7.59030955, 6.00505379, 7.08768604, 19.98938218,
                        9.35340415, 4.53128633, 17.03696335, 20.28368102])
    assert np.isclose(ssr, 6.5335220660550775)
    assert rank == 15


def test_solve_observation(initialized_spline_2d):
    spline, amat, beta, splines, ssr = initialized_spline_2d
    spline.eps = 1e-8
    coeffs, rank, ssr = su.solve_observation(
        amat=amat,
        beta=beta,
        n_coefficients=spline.n_coefficients,
        bandwidth=spline.bandwidth,
        eps=spline.eps)
    assert rank == 16
    assert ssr == 0
    assert np.allclose(coeffs,
                       [0, 3, 6, 9, 3, 6, 9, 12, 6, 9, 12, 15, 9, 12, 15, 18])

    amat[10, 0] = 0.0
    coeffs, rank, ssr = su.solve_observation(
        amat=amat,
        beta=beta,
        n_coefficients=spline.n_coefficients,
        bandwidth=spline.bandwidth,
        eps=spline.eps)
    assert rank == 15
    assert np.isclose(ssr, 6.5335220660550775)
    assert np.allclose(coeffs,
                       [0.27835415, 2.32138976, 5.54259091, 9.35340415,
                        1.46623901, 9.50498082, 12.13083546, 9.85458694,
                        7.59030955, 6.00505379, 7.08768604, 19.98938218,
                        9.35340415, 4.53128633, 17.03696335, 20.28368102])


def test_knot_fit(initialized_spline_2d):
    spline, amat, beta, splines, ssr = initialized_spline_2d

    spline.amat = amat
    spline.beta = beta
    spline.splines = splines
    spline.sum_square_residual = ssr

    spline.coefficients, spline.rank, ssr_solve = su.solve_observation(
        amat=spline.amat, beta=spline.beta,
        n_coefficients=spline.n_coefficients,
        bandwidth=spline.bandwidth, eps=spline.eps)

    spline.sum_square_residual += ssr_solve
    if spline.exit_code == -2:
        spline.initial_sum_square_residual = spline.sum_square_residual

    # Test whether the lsq spline is acceptable
    spline.smoothing_difference = (
        spline.sum_square_residual - spline.smoothing)

    fitted_knots, knot_weights, knot_coordinates = su.knot_fit(
        splines=spline.splines,
        coefficients=spline.coefficients,
        start_indices=spline.start_indices,
        next_indices=spline.next_indices,
        panel_mapping=spline.panel_mapping,
        spline_mapping=spline.spline_mapping,
        knot_steps=spline.knot_steps,
        panel_shape=spline.panel_shape,
        k1=spline.k1,
        weights=spline.weights,
        values=spline.values,
        coordinates=spline.coordinates)

    # Since this is a fit to a basic equation without noise,
    # we expect an almost exact fit.
    assert np.allclose(fitted_knots, spline.values)
    assert np.allclose(knot_weights, 0)
    # The knot coordinate should be at (0, 0)
    assert np.allclose(knot_coordinates, 0)
    ndim = 2
    assert knot_coordinates.shape == (ndim, 1)
    assert knot_weights.shape == (ndim, 1)


def test_add_knot():
    knot_line = np.asarray([0, 0, 0, 0, 9, 9, 9, 9, np.nan, np.nan])
    knots = tuple([x.copy() for x in [knot_line, knot_line]])
    knot_coords = np.array([0.4, 0.5])[:, None]
    knot_weights = np.array([0.1, 0.11])[:, None]
    panel_shape = np.ones(2, dtype=int)
    n_knots = np.full(2, 8)
    knot_estimate = np.full(2, 10)
    k1 = np.full(2, 4)

    # Test standard add knots
    code = su.add_knot(
        knot_weights=knot_weights,
        knot_coords=knot_coords,
        panel_shape=panel_shape,
        knots=knots,
        n_knots=n_knots,
        knot_estimate=knot_estimate,
        k1=k1)

    assert np.allclose(n_knots, [8, 9])
    assert np.allclose(knots[0], knot_line, equal_nan=True)
    assert code == 0
    expected = np.insert(knot_line, 4, 0.5 / 0.11)[:-1]
    assert np.allclose(knots[1], expected, equal_nan=True)

    # Test no more knots to add
    knots = tuple([x.copy() for x in [knot_line, knot_line]])
    n_knots = np.full(2, 8)
    code = su.add_knot(
        knot_weights=knot_weights,
        knot_coords=knot_coords,
        panel_shape=panel_shape,
        knots=knots,
        n_knots=n_knots,
        knot_estimate=np.zeros(2, dtype=int),
        k1=k1)
    assert code == 1

    # Test duplicate knots
    code = su.add_knot(
        knot_weights=np.ones(2, dtype=int)[:, None],
        knot_coords=knot_coords,
        panel_shape=panel_shape,
        knots=knots,
        n_knots=n_knots,
        knot_estimate=knot_estimate,
        k1=k1)
    assert code == 5


def test_evaluate_bspline():
    knot_line = np.arange(11, dtype=float)
    degree = 3
    assert np.allclose(su.evaluate_bspline(knot_line, degree, 4, 3),
                       [0, 1 / 6, 2 / 3, 1 / 6])
    assert np.allclose(su.evaluate_bspline(knot_line, degree, 4, 4),
                       [1 / 6, 2 / 3, 1 / 6, 0])
    assert np.allclose(su.evaluate_bspline(knot_line, degree, 4.5, 4),
                       [1 / 48, 23 / 48, 23 / 48, 1 / 48])
    assert np.allclose(su.evaluate_bspline(knot_line, degree, 5, 4),
                       [0, 1 / 6, 2 / 3, 1 / 6])


def test_work_array():
    knot_line = np.arange(11, dtype=float)
    degree = 4
    s = np.empty(5, dtype=float)
    s2 = su.evaluate_bspline(knot_line, degree, 5, 5, spline=s)
    assert s is s2
    assert np.allclose(s, [1 / 24, 11 / 24, 11 / 24, 1 / 24, 0])


def test_determine_smoothing_spline():

    coordinates = np.array([x.ravel() for x in np.mgrid[:10, :10]],
                           dtype=float)
    np.random.seed(0)
    values = np.random.random(coordinates.shape[1])

    spline = Spline(coordinates[0], coordinates[1], values, solve=False,
                    degrees=3, smoothing=0.001)
    spline.next_iteration()

    assert spline.exit_code == 0

    def run_smooth(s):
        return su.determine_smoothing_spline(
            knots=s.knots,
            n_knots=s.n_knots,
            knot_estimate=s.knot_estimate,
            degrees=s.degrees,
            initial_sum_square_residual=s.initial_sum_square_residual,
            smoothing=s.smoothing,
            smoothing_difference=s.smoothing_difference,
            n_coefficients=s.n_coefficients,
            bandwidth=s.bandwidth,
            amat=s.amat,
            beta=s.beta,
            max_iteration=s.max_iteration,
            knot_steps=s.knot_steps,
            knot_mapping=s.knot_mapping,
            eps=s.eps,
            splines=s.splines,
            start_indices=s.start_indices,
            next_indices=s.next_indices,
            panel_mapping=s.panel_mapping,
            spline_mapping=s.spline_mapping,
            coordinates=s.coordinates,
            values=s.values,
            weights=s.weights,
            panel_shape=s.panel_shape,
            accuracy=s.accuracy)

    # Check exit code 2 -
    coefficients, sq, exit_code, fitted_knots = run_smooth(spline)

    assert exit_code == 2

    # Check exit code 3 (reached max iterations)
    spline = Spline(coordinates[0], coordinates[1], values, solve=False,
                    degrees=3, smoothing=100)
    np.random.seed(0)
    spline.values += np.random.random(spline.size)
    spline.next_iteration()
    coefficients, sq, exit_code, fitted_knots = run_smooth(spline)
    assert exit_code == 3


def test_discontinuity_jumps():

    degrees = np.array([3, 3])
    knot_estimate = np.array([10, 10])
    k1 = degrees + 1
    k2 = k1 + 1
    n_knot = 8
    knot_line = np.arange(10)

    b_spline = np.zeros((knot_estimate.max(), k2.max()), dtype=float)

    degree = 3
    s0 = b_spline.copy()
    su.discontinuity_jumps(knot_line, n_knot, degree, b_spline)
    assert np.allclose(s0, 0)

    degree = 1
    su.discontinuity_jumps(knot_line, n_knot, degree, b_spline)
    assert np.allclose(b_spline[:4], [1, -2, 1, 0, 0])
    assert np.allclose(b_spline[4:], 0)
    b_spline.fill(0)

    degree = 2
    su.discontinuity_jumps(knot_line, n_knot, degree, b_spline)
    assert np.allclose(b_spline[:2], [0.5, -1.5, 1.5, -0.5, 0])
    assert np.allclose(b_spline[2:], 0)


def test_rational_interp_zero():
    p1, f1 = -1, -1
    p2, f2 = 0, 0
    p3, f3 = 1, 1
    result = su.rational_interp_zero(p1, f1, p2, f2, p3, f3)
    assert np.allclose(result, [0, 0, 0, 0, 0, 1, 1])

    p1, f1 = 1, 1
    p3, f3 = -1, -1
    result = su.rational_interp_zero(p1, f1, p2, f2, p3, f3)
    assert np.allclose(result, [0, 0, 0, 0, 0, -1, -1])

    x = np.arange(-1, 2, dtype=float)
    y = x + 0.5
    p1, p2, p3 = x
    f1, f2, f3 = y
    result = su.rational_interp_zero(p1, f1, p2, f2, p3, f3)
    assert np.allclose(result, [-0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 1.5])


def test_fit_point(full_spline_2d):
    spline = full_spline_2d

    # The b_spline at coordinate (5.5, 5.5)
    b_spline = np.array([[0.05881344, 0.27726337, 0.43569959, 0.22822359],
                         [0.05881344, 0.27726337, 0.43569959, 0.22822359]])

    result = su.fit_point(
        coefficients=spline.coefficients.copy(),
        spline=b_spline.copy(),
        spline_mapping=spline.spline_mapping.copy(),
        knot_steps=spline.knot_steps.copy(),
        j_rot=0,
        n_spline=16,
        n_dimensions=2)

    assert np.isclose(result, 11)


def test_single_fit(full_spline_2d):
    spline = full_spline_2d
    result = su.single_fit(
        coordinate=np.array([5.5, 5.5]),
        knots=spline.knots.copy(),
        coefficients=spline.coefficients.copy(),
        degrees=spline.degrees.copy(),
        panel_mapping=spline.panel_mapping.copy(),
        panel_steps=spline.panel_steps.copy(),
        knot_steps=spline.knot_steps.copy(),
        nk1=spline.nk1.copy(),
        spline_mapping=spline.spline_mapping.copy())

    assert np.isclose(result, 11)
