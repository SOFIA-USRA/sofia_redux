from sofia_redux.toolkit.splines.spline import Spline
import numpy as np
import pytest


@pytest.fixture
def tuple_args_2d():
    coordinates = np.array([x.ravel() for x in np.mgrid[:10, :10]],
                           dtype=float)
    values = coordinates[0] + coordinates[1]
    return coordinates[0], coordinates[1], values


@pytest.fixture
def splrep_2d(tuple_args_2d):
    return Spline(*tuple_args_2d)


@pytest.fixture
def blank_splrep_2d(tuple_args_2d):
    return Spline(*tuple_args_2d, solve=False)


@pytest.fixture
def complex_3d_spline():
    z, y, x = np.mgrid[:9, :10, :11]
    values = -np.sin(10 * ((x ** 2) + (y ** 2) + (z ** 2))) / 10
    return Spline(values, degrees=3, exact=True)


@pytest.fixture
def gaussian_2d_data():
    y, x = np.mgrid[:20, :20]
    y2 = (y - 10.0) ** 2
    x2 = (x - 10.0) ** 2
    a = -0.1
    b = -0.15
    data = np.exp(a * x2 + b * y2)
    return data


def test_init(tuple_args_2d):
    spline = Spline(*tuple_args_2d)
    assert spline.coefficients is not None
    spline = Spline(*tuple_args_2d, solve=False)
    assert spline.coefficients is None


def test_exit_message(splrep_2d):
    spline = splrep_2d
    spline.exit_code = 0
    assert "The spline has a residual sum of squares" in spline.exit_message
    spline.exit_code = -1
    assert "The spline is an interpolating spline" in spline.exit_message
    spline.exit_code = -2
    assert "The spline is a weighted least-squares" in spline.exit_message
    spline.exit_code = -3
    assert "Warning.  The coefficients of the spline" in spline.exit_message
    spline.exit_code = -4
    assert "full rank" in spline.exit_message
    spline.n_coefficients = spline.rank + 1
    assert "rank deficient" in spline.exit_message
    spline.exit_code = 1
    assert "storage space exceeds the available storage" in spline.exit_message
    spline.exit_code = 2
    assert "A theoretically impossible result" in spline.exit_message
    spline.exit_code = 3
    assert "The maximal number of iterations" in spline.exit_message
    spline.exit_code = 4
    assert "coefficients already exceeds the number" in spline.exit_message
    spline.exit_code = 5
    assert "would coincide with an old one" in spline.exit_message
    spline.exit_code = 6
    assert "An unknown error occurred" in spline.exit_message
    spline.exit_code = 20
    assert "Knots are not initialized" in spline.exit_message


def test_size(splrep_2d):
    assert splrep_2d.size == 100


def test_knot_size(splrep_2d):
    assert np.allclose(splrep_2d.knot_size, [8, 8])


def test_parse_inputs(blank_splrep_2d, tuple_args_2d):
    spline = blank_splrep_2d
    args = c0, c1, data = tuple_args_2d

    # Check coordinates are inferred from data shape
    data_2d = np.random.random((8, 10))
    spline.parse_inputs(data_2d)
    assert spline.n_dimensions == 2
    expected_coords = np.array([x.ravel() for x in np.mgrid[:8, :10][::-1]])
    assert np.allclose(spline.coordinates, expected_coords)
    assert spline.weights.shape == (80,) and np.allclose(spline.weights, 1)

    data_2d[4, 4] = np.nan
    spline.parse_inputs(data_2d)
    assert spline.size == 79
    assert spline.weights.size == 79

    # Check user coordinates
    spline.parse_inputs(c0, c1, data)
    assert spline.size == 100
    assert spline.n_dimensions == 2
    assert spline.weights.shape == (100,) and np.allclose(spline.weights, 1)

    # Check degrees
    spline.parse_inputs(*args, degrees=3)
    assert np.allclose(spline.degrees, [3, 3])
    assert np.allclose(spline.k1, [4, 4])

    # Check exact fit
    spline.parse_inputs(*args, degrees=[3, 4], exact=False)
    assert np.allclose(spline.degrees, [3, 4])
    assert np.allclose(spline.k1, [4, 5])
    assert not spline.exact
    assert np.allclose(spline.knots[0],
                       [0, 0, 0, 0, 9, 9, 9, 9, np.nan, np.nan, np.nan],
                       equal_nan=True)
    assert np.allclose(spline.knots[1],
                       [0, 0, 0, 0, 0, 9, 9, 9, 9, 9, np.nan],
                       equal_nan=True)
    assert np.allclose(spline.n_knots, [8, 10])

    spline.parse_inputs(*args, degrees=3, smoothing=None, exact=True)
    assert spline.smoothing == 0
    assert spline.exact
    assert np.allclose(spline.degrees, [3, 3])
    assert np.allclose(spline.knots, np.arange(10))
    assert np.allclose(spline.n_knots, [10, 10])

    with pytest.raises(ValueError) as err:
        spline.parse_inputs(*args, exact=True, knots=np.zeros((2, 10)))
    assert "Cannot use" in str(err.value)

    with pytest.raises(ValueError) as err:
        spline.parse_inputs(*args, degrees=5, exact=True)
    assert "There must be at least" in str(err.value)

    spline.parse_inputs(*args, degrees=5, exact=True, reduce_degrees=True)
    assert np.allclose(spline.degrees, [4, 4])

    # Check eps
    for eps in [-0.1, 0, 1, 1.1]:
        with pytest.raises(ValueError) as err:
            spline.parse_inputs(*args, eps=eps)
        assert "eps not in range" in str(err.value)
    spline.parse_inputs(*args, eps=0.12345)
    assert spline.eps == 0.12345

    # Check tolerance
    for tolerance in [-0.1, 0, 1, 1.1]:
        with pytest.raises(ValueError) as err:
            spline.parse_inputs(*args, tolerance=tolerance)
        assert "tolerance not in range" in str(err.value)
    spline.parse_inputs(*args, tolerance=0.12345)
    assert spline.tolerance == 0.12345

    # Check smoothing
    with pytest.raises(ValueError) as err:
        spline.parse_inputs(*args, smoothing=-1)
    assert "smoothing must be >= 0" in str(err.value)
    spline.parse_inputs(*args, smoothing=0.12345)
    assert spline.smoothing == 0.12345
    spline.parse_inputs(*args, smoothing=None)
    assert spline.smoothing == 100 - np.sqrt(200)

    # Check fix_knots
    spline.parse_inputs(*args, fix_knots=None, knots=None)
    assert not spline.fix_knots
    test_knots = np.stack([np.arange(10), np.arange(10)])
    spline.parse_inputs(*args, fix_knots=None, knots=test_knots)
    assert spline.fix_knots
    for knots in [None, test_knots]:
        spline.parse_inputs(*args, fix_knots=False, knots=knots)
        assert not spline.fix_knots
    spline.parse_inputs(*args, fix_knots=True, knots=test_knots)
    assert spline.fix_knots
    with pytest.raises(ValueError) as err:
        spline.parse_inputs(*args, fix_knots=True, knots=None)
    assert "Knots must be supplied" in str(err.value)

    # Check limits
    spline.parse_inputs(*args, fix_knots=False, limits=None)
    assert np.allclose(spline.limits, [[0, 9], [0, 9]])
    knots = np.stack([np.arange(11), np.arange(11)])
    spline.parse_inputs(*args, fix_knots=True, limits=None, knots=knots)
    assert np.allclose(spline.limits, [[0, 10], [0, 10]])
    spline.parse_inputs(*args, limits=[[0, 10], [0, 11]])
    assert np.allclose(spline.limits, [[0, 10], [0, 11]])
    with pytest.raises(ValueError) as err:
        spline.parse_inputs(*args, limits=[1, 2])
    assert "limits must be of shape" in str(err.value)

    # Check knots
    knots = [np.arange(10), np.arange(11)]
    spline.parse_inputs(*args, knots=knots, degrees=3)
    assert spline.knots.shape == (2, 11)
    assert np.allclose(spline.knots[0, :10], np.arange(10))
    assert np.isnan(spline.knots[0, 10])
    assert np.allclose(spline.knots[1], np.arange(11))

    spline.parse_inputs(*args, knots=None, degrees=[3, 4])
    assert np.allclose(spline.knots[0],
                       [0, 0, 0, 0, 9, 9, 9, 9, np.nan, np.nan, np.nan],
                       equal_nan=True)
    assert np.allclose(spline.knots[1],
                       [0, 0, 0, 0, 0, 9, 9, 9, 9, 9, np.nan],
                       equal_nan=True)

    # Check knot_estimate
    spline.parse_inputs(*args, knots=knots, fix_knots=True)
    assert np.allclose(spline.knot_estimate, [10, 11])
    spline.parse_inputs(*args, knots=knots, fix_knots=True,
                        knot_estimate=[5, 5])
    assert np.allclose(spline.knot_estimate, [10, 11])

    spline.parse_inputs(*args, knot_estimate=20, fix_knots=False)
    assert np.allclose(spline.knot_estimate, [20, 20])
    spline.parse_inputs(*args, knot_estimate=[20, 21], fix_knots=False)
    assert np.allclose(spline.knot_estimate, [20, 21])
    with pytest.raises(ValueError) as err:
        spline.parse_inputs(*args, knot_estimate=[20, 21, 22],
                            fix_knots=False)
    assert "Knot estimate" in str(err.value)

    spline.parse_inputs(*args, knot_estimate=None, smoothing=0,
                        fix_knots=False)
    expected = int(np.sqrt(spline.size * 3)) + spline.degrees
    assert np.allclose(spline.knot_estimate, expected)

    spline.parse_inputs(*args, knot_estimate=None, smoothing=10,
                        fix_knots=False)
    expected = int(np.sqrt(spline.size / 2)) + spline.degrees
    assert np.allclose(spline.knot_estimate, expected)

    spline.parse_inputs(*args, knot_estimate=2, fix_knots=False)
    expected = (2 * spline.degrees) + 3
    assert np.allclose(spline.knot_estimate, expected)

    # Test remaining parameters
    spline.parse_inputs(*args, max_iteration=123)
    assert spline.max_iteration == 123
    assert spline.exit_code == -2
    assert spline.sum_square_residual == 0
    max_knot = np.max(spline.knot_estimate + spline.k1)
    max_k1 = np.max(spline.k1)
    assert spline.knot_coordinates.shape == (2, max_knot)
    assert spline.knot_weights.shape == (2, max_knot)
    assert spline.splines.shape == (spline.n_dimensions, spline.size, max_k1)
    assert spline.accuracy == spline.tolerance * spline.smoothing
    assert np.allclose(spline.dimension_order, [0, 1])

    spline.parse_inputs(*args, weights=np.full_like(data, 0.1))
    assert spline.weights.shape == (100,)
    assert np.allclose(spline.weights, 0.1)

    small = np.random.random((2, 2))
    with pytest.raises(ValueError) as err:
        spline.parse_inputs(small)
    assert "Data size" in str(err.value)


def test_check_array_inputs(blank_splrep_2d):
    spline = blank_splrep_2d
    size0 = spline.size
    spline.check_array_inputs()
    assert spline.values.shape == (size0,)
    assert spline.weights.shape == (size0,)
    assert spline.coordinates.shape == (2, size0)

    spline.values[50] = np.nan
    spline.check_array_inputs()
    assert spline.values.shape == (size0 - 1,)
    assert spline.weights.shape == (size0 - 1,)
    assert spline.coordinates.shape == (2, size0 - 1)


def test_initialize_iteration(blank_splrep_2d):
    spline = blank_splrep_2d
    assert spline.panel_shape is None
    assert spline.n_panels == 0
    assert spline.n_intervals == 0
    assert spline.nk1 is None
    assert spline.n_coefficients == 0
    assert spline.bandwidth == 0
    assert spline.permutation is None
    assert not spline.change_order
    assert spline.panel_mapping is None
    assert spline.panel_steps is None
    assert spline.knot_mapping is None
    assert spline.knot_steps is None
    assert spline.spline_mapping is None
    assert spline.spline_steps is None

    spline.initialize_iteration()
    assert np.allclose(spline.panel_shape, [1, 1])
    assert spline.n_panels == 1
    assert spline.n_intervals == 2
    assert np.allclose(spline.nk1, [4, 4])
    assert spline.n_coefficients == 16
    assert spline.bandwidth == 16
    assert np.allclose(spline.permutation, [0, 1])
    assert not spline.change_order
    assert np.allclose(spline.panel_mapping, [[0], [0]])
    assert np.allclose(spline.panel_steps, [1, 1])
    assert spline.knot_mapping.shape == (2, 16)
    assert np.allclose(spline.knot_steps, [4, 1])
    assert spline.spline_mapping.shape == (2, 16)
    assert np.allclose(spline.spline_steps, [4, 1])

    spline = blank_splrep_2d
    spline.n_knots = np.array([9, 10])
    spline.degrees = np.array([4, 2])
    spline.initialize_iteration()
    assert spline.change_order


def test_reorder_dimensions(complex_3d_spline):

    # Exercise the 1-dimensional case
    spline = Spline(np.arange(10, dtype=float))
    spline.final_reorder()
    assert np.allclose(spline.dimension_order, [0])

    spline = complex_3d_spline

    basic_attributes = ['coordinates', 'limits', 'degrees', 'k1', 'nk1',
                        'knots', 'n_knots', 'knot_estimate',
                        'knot_coordinates', 'knot_weights', 'n_knots',
                        'panel_shape', 'splines']
    c_attributes = ['coefficients', 'amat', 'beta']
    old_results = {}
    all_attributes = basic_attributes + c_attributes
    for attribute in all_attributes:
        value = getattr(spline, attribute)
        old_results[attribute] = value.copy()

    assert np.allclose(spline.dimension_order, [0, 1, 2])

    # Test no change
    spline.reorder_dimensions(np.arange(spline.n_dimensions))
    for attribute in all_attributes:
        value = getattr(spline, attribute)
        assert np.allclose(value, old_results[attribute], equal_nan=True)

    # Test standard change
    order = np.array([2, 0, 1])
    spline.reorder_dimensions(order)
    for attribute in c_attributes:
        assert not np.allclose(getattr(spline, attribute),
                               old_results[attribute])
    for attribute in basic_attributes:
        value = getattr(spline, attribute)
        assert np.allclose(value, old_results[attribute][order],
                           equal_nan=True)

    # Test bad values
    spline = complex_3d_spline
    spline.limits = None
    spline.beta = None
    spline.knot_estimate = spline.knot_estimate[:2]
    spline.reorder_dimensions([2, 0, 1])
    assert spline.limits is None
    assert spline.beta is None
    assert spline.knot_estimate.shape != old_results['knot_estimate'].shape

    # Test padding
    spline = complex_3d_spline
    spline.n_coefficients += 1
    spline.reorder_dimensions([2, 0, 1])


def test_final_reorder(complex_3d_spline):
    spline = complex_3d_spline
    attributes = ['coordinates', 'limits', 'degrees', 'k1', 'nk1', 'knots',
                  'n_knots', 'knot_estimate', 'knot_coordinates',
                  'knot_weights', 'n_knots', 'panel_shape', 'splines',
                  'coefficients', 'amat', 'beta']
    old_results = {}
    for attribute in attributes:
        value = getattr(spline, attribute)
        old_results[attribute] = value.copy()

    spline.reorder_dimensions(np.array([2, 0, 1]))
    spline.final_reorder()
    for attribute in attributes:
        assert np.allclose(getattr(spline, attribute), old_results[attribute],
                           equal_nan=True)


def test_order_points(complex_3d_spline):
    spline = complex_3d_spline
    spline.knot_indices = None
    spline.panel_indices = None
    spline.n_panels = 0
    spline.start_indices = None
    spline.next_indices = None
    spline.order_points()
    assert spline.knot_indices.shape == (3, 990)
    assert spline.panel_indices.shape == (990,)
    assert spline.n_panels == 24
    assert not np.any(spline.start_indices[:24] == -1)

    d_index = np.arange(3)

    for panel in range(24):
        start_index = spline.start_indices[panel]
        data_indices = []
        index = spline.next_indices[start_index]
        while index != -1:
            data_indices.append(index)
            index = spline.next_indices[index]

        assert len(data_indices) != 0
        data_indices = np.array(data_indices)
        coordinates = spline.coordinates[:, data_indices]
        knot_indices = spline.knot_indices[:, data_indices]
        assert np.allclose(knot_indices, knot_indices[:, 0][:, None])
        knot_index = knot_indices[:, 0]

        lower_knot = spline.knots[d_index, knot_index][:, None]
        upper_knot = spline.knots[d_index, knot_index + 1][:, None]
        lower_invalid = lower_knot <= spline.degrees[:, None]
        assert np.all(lower_invalid | (coordinates >= lower_knot))
        upper_invalid = upper_knot >= spline.nk1[:, None]
        assert np.all(upper_invalid | (coordinates < upper_knot))


def test_knot_indices_to_panel_indices(complex_3d_spline):
    spline = complex_3d_spline
    panel_indices = spline.knot_indices_to_panel_indices(spline.knot_indices)
    assert np.allclose(panel_indices, spline.panel_indices)


def test_panel_indices_to_knot_indices(complex_3d_spline):
    spline = complex_3d_spline
    knot_indices = spline.panel_indices_to_knot_indices(spline.panel_indices)
    assert np.allclose(knot_indices, spline.knot_indices)


def test_iterate(gaussian_2d_data):
    data = gaussian_2d_data

    # Check fixed knots
    spline = Spline(data, exact=True, degrees=3)
    assert spline.fix_knots
    assert spline.exit_code == -256

    # Check iterations with smoothing spline
    spline = Spline(data, degrees=3, smoothing=0.1, tolerance=1e-3)
    assert spline.exit_code == 0
    assert spline.iteration > 0
    assert np.isclose(spline.sum_square_residual, 0.1, atol=1e-3)

    # Check reach maximum iterations on smoothing spline
    spline = Spline(data, degrees=3, smoothing=0.1, tolerance=1e-3,
                    max_iteration=1)
    assert spline.exit_code == 3

    # Check accuracy exit
    np.random.seed(0)
    noisy_data = data + np.random.random(data.shape) * 0.1
    spline = Spline(noisy_data, degrees=3, smoothing=0.1, tolerance=1 - 1e-5)
    assert abs(spline.smoothing_difference) <= spline.accuracy


def test_next_iteration(gaussian_2d_data):
    data = gaussian_2d_data
    spline = Spline(data, degrees=3, solve=False)
    assert not spline.next_iteration()
    assert spline.exit_code == -2

    spline = Spline(data, degrees=3, solve=False, knot_estimate=3,
                    smoothing=0.1)
    assert spline.next_iteration() and spline.exit_code == 0

    # Test no further knots
    spline = Spline(data, degrees=3, solve=False, knot_estimate=3,
                    smoothing=0.1, exact=True)
    assert not spline.next_iteration()
    assert spline.exit_code == -256

    # Check rank deficiency
    np.random.seed(0)
    noisy_data = data + np.random.random(data.shape)
    spline = Spline(noisy_data, degrees=3, solve=False,
                    smoothing=0.0, tolerance=1e-16)
    while spline.next_iteration():
        pass
    assert spline.exit_code == -400 and "rank deficient" in spline.exit_message
    assert spline.sum_square_residual == 0

    # Check interpolating spline
    np.random.seed(0)
    c0 = np.random.random(25)
    c1 = np.random.random(25)
    d = np.random.random(25)
    spline = Spline(c0, c1, d, degrees=3, solve=False, smoothing=0.0,
                    tolerance=1e-8)
    while spline.next_iteration():
        pass
    assert spline.exit_code == -1 and spline.sum_square_residual == 0

    # Check no further coefficients can be added
    spline = Spline(c0, c1, d, degrees=1, solve=False, smoothing=0.0,
                    tolerance=1e-8)
    while spline.next_iteration():
        pass
    assert spline.exit_code == 4


def test_determine_smoothing_spline(complex_3d_spline):
    spline = complex_3d_spline
    assert spline.smoothing == 0.0
    ssr0 = spline.sum_square_residual
    c0 = spline.coefficients.copy()
    assert 'full rank' in spline.exit_message

    spline.exit_code = -2
    spline.determine_smoothing_spline()
    assert spline.sum_square_residual == ssr0
    assert np.allclose(c0, spline.coefficients, equal_nan=True)
    assert spline.exit_code == -2

    # Create an inexact spline and smooth further...
    z, y, x = np.mgrid[:9, :10, :11]
    values = -np.sin(10 * ((x ** 2) + (y ** 2) + (z ** 2)))
    np.random.seed(1)
    noisy_values = values + np.random.random(values.shape) * 0.05
    spline = Spline(noisy_values, smoothing=510, degrees=1)
    assert spline.exit_code == 0
    assert np.isclose(spline.sum_square_residual, 510, atol=0.5)
    # This should only work for very small smoothing increments...
    spline.determine_smoothing_spline(smoothing=511)
    assert np.isclose(spline.sum_square_residual, 511, atol=0.5)
    spline.determine_smoothing_spline(smoothing=509)
    assert np.isclose(spline.sum_square_residual, 509, atol=0.5)

    # Check unable to solve
    spline.determine_smoothing_spline(smoothing=800)
    assert 'theoretically impossible' in spline.exit_message

    # Check bad smoothing
    with pytest.raises(ValueError) as err:
        spline.determine_smoothing_spline(smoothing=-1)
    assert 'smoothing must be >= 0' in str(err.value)


def test_call():
    z0, y0, x0 = np.mgrid[:9, :10, :11]
    values0 = -np.sin(10 * ((x0 ** 2) + (y0 ** 2) + (z0 ** 2))) / 10

    spline = Spline(x0.ravel(), y0.ravel(), z0.ravel(), values0.ravel(),
                    smoothing=0, degrees=4)
    # Check it's an interpolating spline
    assert spline.exit_code == -1

    # Check arbitrary coordinate fit
    coordinates = np.stack([x0.ravel(), y0.ravel(), z0.ravel()])
    results = spline(coordinates)
    assert np.allclose(results, values0.ravel())

    # Check grid fit
    #                 x               y            z
    coordinates = np.arange(11), np.arange(10), np.arange(9)
    results = spline(*coordinates)
    assert np.allclose(results, values0)

    # Check single fits
    x, y, z = 3, 4, 8
    result = spline(x, y, z)
    assert np.isclose(result, values0[z, y, x])

    result = spline(np.asarray([x, y, z]))
    assert np.isclose(result, values0[z, y, x])
    result = spline([x, y, z])
    assert np.isclose(result, values0[z, y, x])

    with pytest.raises(ValueError) as err:
        spline(1, 2, 3, 4)
    assert "Number of arguments does not match" in str(err.value)

    with pytest.raises(ValueError) as err:
        spline(np.random.random((4, 10)))
    assert "Coordinate shape[0] does not match" in str(err.value)
