import numba as nb
import numpy as np


__all__ = ['find_knots', 'find_knot', 'calculate_minimum_bandwidth',
           'flat_index_mapping', 'create_ordering', 'check_input_arrays',
           'givens_parameters', 'givens_rotate', 'build_observation',
           'back_substitute', 'solve_rank_deficiency', 'solve_observation',
           'knot_fit', 'add_knot', 'evaluate_bspline',
           'determine_smoothing_spline', 'discontinuity_jumps',
           'rational_interp_zero', 'fit_point', 'perform_fit', 'single_fit']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def find_knots(coordinates, knots, valid_knot_start, valid_knot_end
               ):  # pragma: no cover
    """
    Find the knot indices for an array of coordinates.

    The knot index (i) for a given coordinate (x) is defined such that
    knot[i] <= x < knot[i + 1].  Coordinates that are less than the minimum
    valid knot are set to the minimum valid knot index, and values that are
    greater than the maximum valid knot are set to the maximum valid knot
    index - 1.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The coordinates of shape (n_dimensions, m) where m are the number of
        coordinates.
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    valid_knot_start : numpy.ndarray (int)
        The start indices for the first valid knot in each dimension.  For a
        spline of degree k, this should be k.  The shape is (n_dimensions,).
    valid_knot_end : numpy.ndarray (int)
        The last valid knot index in each dimension of shape (n_dimensions,).
        for a spline of degree k, this should be n_knots - degree - 1.

    Returns
    -------
    knot_indices : numpy.ndarray (int)
        The knot index for each coordinate in each dimension of shape
        (n_dimensions, m).
    """
    dimensions, n_data = coordinates.shape
    knot_indices = np.empty((dimensions, n_data), dtype=nb.int64)
    for dimension in range(dimensions):
        x = coordinates[dimension]
        knot = knots[dimension]
        knot_end = valid_knot_end[dimension]
        knot_start = valid_knot_start[dimension]

        max_value = knot[knot_end]
        min_value = knot[knot_start]
        for i in range(n_data):
            value = x[i]
            if value < min_value:
                knot_indices[dimension, i] = knot_start
            elif value > max_value:
                knot_indices[dimension, i] = knot_end - 1
            else:
                for k in range(knot_start, knot_end):
                    if value < knot[k + 1]:
                        knot_indices[dimension, i] = k
                        break
                else:
                    knot_indices[dimension, i] = knot_end - 1

    return knot_indices


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def find_knot(coordinate, knots, valid_knot_start, valid_knot_end,
              allow_outside=True, lower_bounds=None, upper_bounds=None,
              ):  # pragma: no cover
    """
    Find the knot index for a single coordinate.

    Parameters
    ----------
    coordinate : numpy.ndarray (float)
        The coordinates of shape (n_dimensions, m).
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    valid_knot_start : numpy.ndarray (int)
        The start indices for the first valid knot in each dimension.  For a
        spline of degree k, this should be k.  The shape is (n_dimensions,).
    valid_knot_end : numpy.ndarray (int)
        The last valid knot index in each dimension of shape (n_dimensions,).
        for a spline of degree k, this should be n_knots - degree - 1.
    allow_outside : bool, optional
        If `True` (default), allow a fit outside the bounds of the knots.
    lower_bounds : numpy.ndarray (float)
        Specifies the lower range of valid coordinates for each dimension.
    upper_bounds : numpy.ndarray (float)
        Specifies the upper range of valid coordinates for each dimension.

    Returns
    -------
    knot_indices : numpy.ndarray (int)
        The knot index for each coordinate in each dimension of shape
        (n_dimensions, m).
    """
    n_dimensions = coordinate.size
    knot_index = np.empty(n_dimensions, dtype=nb.int64)

    for dimension in range(n_dimensions):
        knot_line = knots[dimension]
        knot_start = valid_knot_start[dimension]
        knot_end = valid_knot_end[dimension]

        if lower_bounds is None:
            min_knot = knot_line[knot_start]
        else:
            min_knot = lower_bounds[dimension]

        if upper_bounds is None:
            max_knot = knot_line[knot_end]
        else:
            max_knot = upper_bounds[dimension]

        value = coordinate[dimension]
        if value < min_knot:
            if not allow_outside:
                knot_index[0] = -1
                return knot_index
            knot_index[dimension] = knot_start
        elif value > max_knot:
            if not allow_outside:
                knot_index[0] = -1
                return knot_index
            knot_index[dimension] = knot_end - 1
        else:
            for k in range(knot_start, knot_end):
                if value < knot_line[k + 1]:
                    knot_index[dimension] = k
                    break
            else:
                knot_index[dimension] = knot_end - 1

    return knot_index


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calculate_minimum_bandwidth(degrees, n_knots, permutations
                                ):  # pragma: no cover
    """
    Calculate the minimum possible bandwidth given knots and spline degree.

    The bandwidth of a given observation matrix is given as:

    k0(d1*d2*d3*...*dn) + k1(d2*d3*...*dn) + k2(d3*...*dn) + ... + kn

    for the n-dimensional case where a number (1,2,3, etc) signifies the
    dimension number, k represents the degree, and dn is given as
    n_knots - k - 1.  This function loops through all available dimensional
    permutations in order to find the permutation that results in the smallest
    bandwidth.

    Finding the bandwidth is only relevant in more than one dimension.
    The bandwidth of the observation matrix may be minimized by switching the
    order of dimensions (coordinates, degrees, knot lines, etc.).  This can
    speed up evaluation of spline coefficients and knot locations.

    To save speed, permutations should be pre-calculated and supplied to this
    function, containing each possible ordering of the dimensions.  For
    example, for two dimensions, permutations would be [[0, 1], [1, 0]].
    For three dimensions permutations would be:

    [[0 1 2]
     [0 2 1]
     [1 0 2]
     [1 2 0]
     [2 0 1]
     [2 1 0]]

    Note that for 1-dimension the minimum bandwidth will be fixed.

    Dimensional permutations can be calculated via:

        permutations = np.array(list(itertools.permutations(range(n_dim))))

    where ndim is the number of dimensions.

    Parameters
    ----------
    degrees : numpy.ndarray (int)
        The degrees of the spline in each dimension (n_dimensions,).
    n_knots : numpy.ndarray (int)
        The number of knots in each dimension (n_dimensions,).
    permutations : numpy.ndarray (int)
        All possible dimensional ordering of shape
        (n_permutations, n_dimensions).

    Returns
    -------
    minimum_bandwidth, permutation_index, changed : int, int, bool
        The minimum bandwidth, the index of the permutation for the minimum
        bandwidth, and whether the permutation is different to the first
        available permutation.
    """
    n_permutations, n_dimensions = permutations.shape
    if n_dimensions == 1:
        return degrees[0] + 1, permutations[0], False

    difference = n_knots - degrees - 1
    min_bandwidth = -1
    min_permutation = permutations[0]
    min_i = -1

    for pi in range(n_permutations):
        bandwidth = 0
        step_size = 1
        dimensions = permutations[pi]

        for ri in range(n_dimensions - 1, -1, -1):
            dimension = dimensions[ri]
            bandwidth += degrees[dimension] * step_size
            step_size *= difference[dimension]

        if min_bandwidth == -1 or bandwidth < min_bandwidth:
            min_bandwidth = bandwidth
            min_permutation = permutations[pi]
            min_i = pi

    return min_bandwidth + 1, min_permutation, min_i != 0


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def flat_index_mapping(shape):  # pragma: no cover
    """
    Return index slices for Numba flattened arrays.

    Given the shape of an array, return a variety of useful parameters for
    indexing a flattened (x.ravel()) version of that array.

    For example, consider an array (x) of shape (3, 4):

    x = [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]

    which when flattened (x_flat) is equal to

    x_flat = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]

    `map_indices` returns the indices of x_flat on x, and is a 2-D array of
    shape (n_dimensions, x.size).  map_indices[:, 6] gives the N-D index of
    element 6 of x_flat in terms of x.  i.e, map_indices[:, 6] = [1, 2].

    `transpose_indices` contains the flat indices of an array that has been
    transposed.  i.e. x.T.ravel()[transpose_indices] == x_flat

    `step_size` contains indicates the flat index jump along the given
    dimension.  i.e., the step size for the above example is [4, 1], indicating
    that for every increment along the first dimension, the flat index
    increments by 4.  Likewise, for every increment along the second dimension,
    the flat index increments by 1.

    Parameters
    ----------
    shape : numpy.ndarray (int)

    Returns
    -------
    map_indices, transpose_indices, step_size
    """
    size = int(np.prod(shape))
    n_dimensions = shape.size
    step_size = np.empty(n_dimensions, dtype=nb.int64)
    flat_indices = np.empty((n_dimensions, size), dtype=nb.int64)
    transpose_indices = np.zeros(size, dtype=nb.int64)

    div = np.empty(n_dimensions, dtype=nb.int64)
    mod_t = np.empty(n_dimensions, dtype=nb.int64)
    c_t = np.empty(n_dimensions, dtype=nb.int64)

    div_factor = size
    c = 1
    step = 1
    for i in range(n_dimensions):
        ri = n_dimensions - i - 1
        step_size[ri] = step
        mod_t[i] = div_factor
        c_t[i] = c
        c *= shape[i]
        step *= shape[ri]
        div_factor /= shape[i]
        div[i] = div_factor

    for i in range(n_dimensions):
        for j in range(size):
            flat_indices[i, j] = (j // div[i]) % shape[i]
            transpose_indices[j] += c_t[i] * ((j % mod_t[i]) // step_size[i])

    # For shape (2, 3, 4):
    # 6 every 1 in chunks of 4   6 * ((j % 4) // 1)
    # 2 every 4 in chunks of     2 * ((j % 12) // 4)
    # 1 every 12 in              1 * ((j % 24) // 12)

    return flat_indices, transpose_indices, step_size


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def create_ordering(indices, size=-1):  # pragma: no cover
    """
    Given a list of indices, create an ordering structure for fast access.

    The purpose of this function is to create two arrays than can be used
    to quickly retrieve all reverse indices for a given index using numba.
    For example, consider the array x:

    >>> x = np.array([0, 0, 2, 1, 4, 2, 1, 2, 2, 1])
    >>> si, ni = create_ordering(x)

    We can now extract the indices on x for a given value.  E.g.,

    >>> ri = []
    >>> for value in [0, 1, 2, 3, 4]:
    ...     inds = []
    ...     ri.append(inds)
    ...     reverse_index = si[value]
    ...     while reverse_index != -1:
    ...         inds.append(reverse_index)
    ...         reverse_index = ni[reverse_index]
    >>> print(ri)
    [[1, 0], [9, 6, 3], [8, 7, 5, 2], [], [4]]

    i.e., the indices on x where x is equal to 0 are [1, 0], and [8, 7, 5, 2]
    where x is equal to 2.

    This is a very useful feature for N-D numba functions as reverse indices
    can be precalculated and represented as 1-dimensional arrays and we avoid
    functions such as np.nonzero, argwhere etc, that introduce additional
    overhead associated with array creation.

    Parameters
    ----------
    indices : numpy.ndarray (int)
        A list of input indices.
    size : int
        The size of the array to return.

    Returns
    -------
    start_indices, next_indices : numpy.ndarray (int), numpy.ndarray (int)
    """
    if size == -1:
        size = max(max(indices) + 1, indices.size)

    n_data = indices.size
    start_indices = np.full(size, -1)
    next_indices = np.full(size, -1)
    for i in range(n_data):
        index = indices[i]
        next_indices[i] = start_indices[index]
        start_indices[index] = i

    return start_indices, next_indices


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def check_input_arrays(values, coordinates, weights):  # pragma: no cover
    """
    Check all input arrays.

    For a data point to be valid, it must have a finite value, weight
    and coordinate.  The weight must greater than zero, and coordinates
    must be finite in each dimension.

    Parameters
    ----------
    values : numpy.ndarray (float)
        An array of values of shape (n_data,).
    coordinates : numpy.ndarray (float)
        The coordinate values of shape (n_dimensions, n_data).
    weights : numpy.ndarray (float)
        The point weights of shape (n_data,).

    Returns
    -------
    valid : numpy.ndarray (bool)
        A boolean mask of shape (n_data,) where `False` marks an invalid
        data coordinate.
    """
    n_dimensions, n_data = coordinates.shape
    valid = np.full(n_data, True)
    for i in range(n_data):
        if not np.isfinite(values[i]):
            valid[i] = False
        elif not np.isfinite(weights[i]) or weights[i] <= 0:
            valid[i] = False

    for dimension in range(n_dimensions):
        x = coordinates[dimension]
        for i in range(n_data):
            if not valid[i]:
                continue
            x_value = x[i]
            if not np.isfinite(x_value):
                valid[i] = False

    return valid


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def givens_parameters(x, y):  # pragma: no cover
    """
    Calculate the parameters of a Givens transformation.

    Parameters
    ----------
    x : float
        The x value.
    y : float
        The y value.

    Returns
    -------
    updated_y, cos, sin
    """
    z = np.abs(x)
    if z >= y:
        c = z * np.sqrt(1.0 + (y / x) ** 2)
    else:
        c = y * np.sqrt(1.0 + (x / y) ** 2)
    cos = y / c
    sin = x / c
    updated_y = c
    return updated_y, cos, sin


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def givens_rotate(cos, sin, x, y):  # pragma: no cover
    """
    Apply the Givens transformation to a value.

    Parameters
    ----------
    cos : float
    sin : float
    x : float
    y : float

    Returns
    -------
    x_rotated, y_rotated
    """
    temp = (cos * x) - (sin * y)
    y = (sin * x) + (cos * y)
    x = temp
    return x, y


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def build_observation(coordinates, values, weights, n_coefficients, bandwidth,
                      degrees, knots, knot_steps, start_indices,
                      next_indices, panel_mapping, spline_mapping
                      ):  # pragma: no cover
    """
    Build the spline observation matrices.

    The observation matrices are used to solve the linear system of equations
    Ax = B in row-echelon form.  A and B (amat and beta) may be used to solve
    for x using either :func:`back_substitute` if A is full rank, or
    :func:`solve_rank_deficiency` if A is rank deficient.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The m coordinates to fit (n_dimensions, m).
    values : numpy.ndarray (float)
        The values at each coordinate (m,).
    weights : numpy.ndarray (float)
        The associated weight value for each coordinate (m,).
    n_coefficients : int
        The number of coefficients to fit.
    bandwidth : int
        The bandwidth of the observation.
    degrees : numpy.ndarray (int)
        The degrees of the splines to fit (n_dimensions,).
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    start_indices : numpy.ndarray (int)
        The starting indices of shape (m,) as returned by
        :func:`create_ordering`.
    next_indices : numpy.ndarray (int)
        The next indices of shape (m,) as returned by :func:`create_ordering`.
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    spline_mapping : numpy.ndarray (int)
        An array containing the spline mapping (flat to n-D) indices.  This is
        created by passing the spline shape (degrees + 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_spline_coefficients).

    Returns
    -------
    amat, beta, knot_splines, ssr : array, array, array, float
        The observation matrix amat of shape (n_coefficients, bandwidth), the
        observation matrix beta of shape (n_coefficients,), the base spline
        coefficients for each knot (n_dimensions, n_knots, max(degree) + 1),
        and ssr (sum of the residuals squared)
    """
    k1 = degrees + 1
    b1 = bandwidth - 1
    max_k1 = np.max(k1)
    n_dimensions, m = coordinates.shape
    beta = np.zeros(n_coefficients, dtype=nb.float64)
    amat = np.zeros((n_coefficients, bandwidth), dtype=nb.float64)
    splines = np.zeros((n_dimensions, m, max_k1), dtype=nb.float64)
    work_spline = np.empty((n_dimensions, splines.shape[2]), dtype=nb.float64)
    row = np.empty(bandwidth, dtype=nb.float64)
    n_spline = int(np.prod(k1))

    sum_square_residual = 0.0
    for panel in range(start_indices.size):
        point = start_indices[panel]
        if point == -1:
            continue

        panel_index = panel_mapping[:, panel]
        knot_index = panel_index + degrees
        j_rot = 0
        for dimension in range(n_dimensions):
            j_rot += panel_mapping[dimension, panel] * knot_steps[dimension]

        while True:
            weight = weights[point]
            value = values[point] * weight

            for dimension in range(n_dimensions):
                evaluate_bspline(
                    knots[dimension], degrees[dimension],
                    coordinates[dimension, point], knot_index[dimension],
                    work_spline[dimension])
                for i in range(k1[dimension]):  # copy it over for later use
                    splines[dimension, point, i] = work_spline[dimension, i]

            # Initialize a new row of the observation matrix
            for i in range(bandwidth):
                row[i] = 0.0

            # Calculate the spline dimensional cross-products and store it in
            # the row.
            for i in range(n_spline):
                row_index = 0
                s = weight
                for dimension in range(n_dimensions):
                    spline_index = spline_mapping[dimension, i]
                    s *= work_spline[dimension, spline_index]
                    row_index += spline_index * knot_steps[dimension]
                row[row_index] = s

            # Rotate the row into a triangle by givens transformations
            i_rot = j_rot - 1
            for i in range(bandwidth):
                i_rot += 1
                pivot = row[i]
                if pivot == 0:
                    continue
                # Calculate parameters
                amat[i_rot, 0], cos, sin = givens_parameters(
                    pivot, amat[i_rot, 0])
                # Apply transformation to the RHS
                value, beta[i_rot] = givens_rotate(
                    cos, sin, value, beta[i_rot])
                if i == b1:
                    break
                i2 = 0
                i3 = i + 1
                # Apply transformation to the LHS
                for j in range(i3, bandwidth):
                    i2 += 1
                    row[j], amat[i_rot, i2] = givens_rotate(
                        cos, sin, row[j], amat[i_rot, i2])

            sum_square_residual += value ** 2
            point = next_indices[point]
            if point == -1:
                break

    return amat, beta, splines, sum_square_residual


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def back_substitute(amat, beta, n_coefficients, bandwidth):  # pragma: no cover
    """
    Use back-substitution to solve a reduced row-echelon form matrix.

    The amat matrix MUST be full rank.

    Parameters
    ----------
    amat : numpy.ndarray (float)
        The 'A' in the system Ax = B of shape (>=n_coefficients, >=bandwidth).
    beta : numpy.ndarray (float)
        The 'B' in the system Ax = B of shape (>=n_coefficients,).
    n_coefficients : int
        The number of coefficients to solve for.
    bandwidth : int
        The bandwidth of matrix A (amat).

    Returns
    -------
    coefficients : numpy.ndarray (float)
        The coefficients of shape (n_coefficients.).
    """
    n = n_coefficients
    n1 = n - 1
    b1 = bandwidth - 1

    coefficients = np.zeros(n_coefficients, dtype=nb.float64)
    coefficients[n1] = beta[n1] / amat[n1, 0]

    i = n - 2
    if i < 0:
        return coefficients

    for j in range(1, n):
        value = beta[i]
        i1 = b1
        if j <= b1:
            i1 = j
        m = i
        for j2 in range(1, i1 + 1):
            m += 1
            value -= coefficients[m] * amat[i, j2]
        coefficients[i] = value / amat[i, 0]
        i -= 1

    return coefficients


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def solve_rank_deficiency(amat, beta, n_coefficients, bandwidth, tolerance
                          ):  # pragma: no cover
    """
    Solve a rank-deficient row-echelon reduced form matrix.

    Parameters
    ----------
    amat : numpy.ndarray (float)
        The 'A' in the system Ax = B of shape (>=n_coefficients, >=bandwidth).
    beta : numpy.ndarray (float)
        The 'B' in the system Ax = B of shape (>=n_coefficients,).
    n_coefficients : int
        The number of coefficients to solve for.
    bandwidth : int
        The bandwidth of matrix A (amat).
    tolerance : float
        The value over which the zeroth element of `amat` will be considered
        rank deficient.  Deficient rows will be rotated into a new reduced
        rank matrix and solved accordingly.

    Returns
    -------
    coefficients, ssr, rank : numpy.ndarray (float), float, int
        The coefficients of shape (n_coefficients.), the sum of the squared
        residuals (ssr), and the rank.
    """
    coefficients = np.zeros(n_coefficients, dtype=nb.float64)
    b1 = bandwidth - 1
    nc1 = n_coefficients - 1
    deficiency = 0
    sum_squared_residuals = 0.0
    row = np.zeros(bandwidth, dtype=nb.float64)

    for i in range(n_coefficients):  # 90
        if amat[i, 0] > tolerance:
            continue
        deficiency += 1

        if i == nc1:
            continue

        yi = beta[i]

        for j in range(b1):
            row[j] = amat[i, j + 1]
        row[b1] = 0.0

        i1 = i + 1
        for ii in range(i1, n_coefficients):
            i2 = min(nc1 - ii, b1)
            piv = row[0]

            if piv != 0:  # givens_rotate
                amat[ii, 0], cos, sin = givens_parameters(
                    piv, amat[ii, 0])
                yi, beta[ii] = givens_rotate(cos, sin, yi, beta[ii])
                if i2 == 0:
                    break
                for j in range(i2):
                    j1 = j + 1
                    row[j1], amat[ii, j1] = givens_rotate(
                        cos, sin, row[j1], amat[ii, j1])
                    row[j] = row[j1]

            else:
                for j in range(i2):
                    row[j] = row[j + 1]

            row[i2] = 0.0

        sum_squared_residuals += yi ** 2  # 90

    rank = n_coefficients - deficiency
    amat2 = np.zeros((rank, bandwidth), dtype=nb.float64)
    beta2 = np.zeros(rank, dtype=nb.float64)

    ii = -1
    for i in range(n_coefficients):  # 120
        if amat[i, 0] <= tolerance:
            continue
        ii += 1
        beta2[ii] = beta[i]
        amat2[ii, 0] = amat[i, 0]

        j1 = min(i, b1)
        if j1 == 0:
            continue

        jj = ii
        kk = 0
        j = i
        for k in range(1, j1 + 1):  # 110
            if amat[j, 0] > tolerance:
                kk += 1
                jj -= 1
                amat2[jj, kk] = amat[j - 1, k]
            j -= 1

    ii = 0
    for i in range(n_coefficients):  # 200
        ii += 1
        if amat[i, 0] > tolerance:
            continue
        ii -= 1
        if ii == 0:
            continue
        jj = 0
        j = i
        j1 = min(j, b1)
        for k in range(1, j1 + 1):  # 130
            j -= 1
            if amat[j, 0] <= tolerance:
                continue
            row[jj] = amat[j, k]
            jj += 1

        for kk in range(jj, bandwidth):  # 140
            row[kk] = 0.0

        jj = ii - 1
        for i1 in range(ii):  # 190
            j1 = min(jj, b1)
            piv = row[0]
            j3 = 0  # irrelevant - just for numba compilation
            if piv == 0:
                if j1 == 0:
                    break
                for j2 in range(j1):  # 150
                    j3 = j2 + 1
                    row[j2] = row[j3]
                jj -= 1
                row[j3] = 0.0

            else:  # givens_rotate
                amat2[jj, 0], cos, sin = givens_parameters(piv, amat2[jj, 0])
                if j1 == 0:
                    break
                kk = jj
                for j2 in range(j1):  # 170
                    j3 = j2 + 1
                    kk -= 1
                    row[j3], amat2[kk, j3] = givens_rotate(
                        cos, sin, row[j3], amat2[kk, j3])
                    row[j2] = row[j3]
                jj -= 1
                row[j3] = 0.0

    i = rank - 1
    beta2[i] = beta2[i] / amat2[i, 0]

    if i != 0:
        for j in range(1, rank):  # 220
            i -= 1
            store = beta2[i]
            i1 = min(j, b1)
            k = i
            for ii in range(1, i1 + 1):  # 210
                k += 1
                stor1 = beta2[k]
                stor2 = amat2[i, ii]
                store -= stor1 * stor2

            stor1 = amat2[i, 0]
            beta2[i] = store / stor1

    beta2[0] /= amat2[0, 0]
    if rank != 1:
        for j in range(1, rank):  # 250
            store = beta2[j]
            i1 = min(j, b1)
            k = j
            for ii in range(1, i1 + 1):  # 240
                k -= 1
                stor1 = beta2[k]
                stor2 = amat2[k, ii]
                store -= stor1 * stor2
            stor1 = amat2[j, 0]
            beta2[j] = store / stor1

    k = -1
    for i in range(n_coefficients):  # 280
        store = 0.0
        if amat[i, 0] > tolerance:
            k += 1
        ij = i + 1
        j1 = min(ij, bandwidth)
        kk = k
        for j in range(j1):  # 270
            ij -= 1
            if amat[ij, 0] <= tolerance:
                continue

            stor1 = amat[ij, j]
            stor2 = beta2[kk]
            store += stor1 * stor2
            kk -= 1
        coefficients[i] = store

    stor3 = 0.0
    for i in range(n_coefficients):  # 310
        if amat[i, 0] > tolerance:
            continue
        store = beta[i]
        i1 = min(nc1 - i, b1)
        if i1 != 0:
            for j in range(i1):  # 290
                ij = i + j
                stor1 = coefficients[ij]
                stor2 = amat[i, j + 1]
                store -= stor1 * stor2

        stor1 = amat[i, 0]
        stor2 = coefficients[i]
        stor1 *= stor2
        stor3 += stor1 * (stor1 - store - store)

    fac = stor3
    sum_squared_residuals += fac
    return coefficients, sum_squared_residuals, rank


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def solve_observation(amat, beta, n_coefficients, bandwidth, eps
                      ):  # pragma: no cover
    """
    Solve a row-echelon reduced linear system of equations Ax=B.

    Returns the results of :func:`back_substitute` if A is full rank, or the
    results of :func:`solve_rank_deficiency` otherwise.

    Parameters
    ----------
    amat : numpy.ndarray (float)
        The array A in row-echelon form.  Should be of shape
        (>=n_coefficients, >=bandwidth).
    beta : numpy.ndarray (float)
        The array B accounting for row-echelon form.  Should be of shape
        (>=n_coefficients,).
    n_coefficients : int
        The number of coefficients to solve for.
    bandwidth : int
        The bandwidth of the observation matrix A (`amat`).
    eps : float
        The precision to determine singular values of A.  If any row of
        amat[:,0] < (eps * max(amat[:,0])) it will be considered singular.

    Returns
    -------
    coefficients, rank, ssr : numpy.ndarray (float), int, float
        The derived coefficients, the rank of A, and the sum of the squared
        residuals.
    """
    diagonals = amat[:n_coefficients, 0]
    max_diagonal = np.max(diagonals)
    sigma = eps * max_diagonal
    full_rank = True

    for diagonal in diagonals:
        if diagonal < sigma:
            full_rank = False
            break

    if full_rank:
        coefficients = back_substitute(amat, beta, n_coefficients, bandwidth)
        rank = n_coefficients
        sum_squared_residuals = 0.0
    else:
        amat2 = np.empty((n_coefficients, bandwidth), dtype=nb.float64)
        beta2 = np.empty(n_coefficients, dtype=nb.float64)
        for i in range(n_coefficients):  # copy over amat, beta
            beta2[i] = beta[i]
            for j in range(bandwidth):
                amat2[i, j] = amat[i, j]

        coefficients, sum_squared_residuals, rank = solve_rank_deficiency(
            amat2, beta2, n_coefficients, bandwidth, sigma)

    return coefficients, rank, sum_squared_residuals


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def knot_fit(splines, coefficients, start_indices, next_indices, panel_mapping,
             spline_mapping, knot_steps, panel_shape, k1, weights, values,
             coordinates):  # pragma: no cover
    """
    Calculate the spline fit at each knot location.

    Parameters
    ----------
    splines : numpy.ndarray (float)
        The splines of shape (n_dimensions, n_data, max(k1)).
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    start_indices : numpy.ndarray (int)
        The start indices of the reverse lookup array of shape (n_data,).  See
        :func:`create_ordering` for further details.
    next_indices : numpy.ndarray (int)
        The "next" indices of the reverse lookup array of shape (n_data,_.  See
        :func:`create_ordering` for further details.
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    spline_mapping : numpy.ndarray (int)
        An array containing the spline mapping (flat to n-D) indices.  This is
        created by passing the spline shape (degrees + 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_spline_coefficients).
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    panel_shape : numpy.ndarray (int)
        The panel shape will be defined as n_knots - (2 * k1) + 1 where k1 and
        n_knots are both of shape (n_dimensions,).
    k1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) where k1[dimension] =
        degree[dimension] + 1.
    weights : numpy.ndarray (float)
        The value weights of shape (n_data,).
    values : numpy.ndarray (float)
        The values to fit of shape (n_data,).
    coordinates : numpy.ndarray (float)
        The coordinates of each value in each dimension of shape
        (n_dimensions, n_data).

    Returns
    -------
    fit, knot_weights, knot_coordinates : 3-tuple of numpy.ndarray (float)
        The fitted value at each knot of shape (n_data,), the knot weights of
        shape (max_panels,), and the knot coordinates of shape (max_panels,)
        where max_panels is the maximum number of panels available in the
        spline fit.
    """

    n_dimensions, n_panels = panel_mapping.shape
    n_spline = int(np.prod(k1))
    m = start_indices.size
    fit = np.empty(m, dtype=nb.float64)
    max_panels = np.max(panel_shape)

    knot_weights = np.zeros((n_dimensions, max_panels), dtype=nb.float64)
    knot_coordinates = np.zeros((n_dimensions, max_panels), dtype=nb.float64)

    for panel in range(n_panels):
        point = start_indices[panel]
        if point == -1:
            continue

        p_map = panel_mapping[:, panel]
        j_rot = 0
        for dimension in range(n_dimensions):
            j_rot += p_map[dimension] * knot_steps[dimension]

        while point != -1:

            fit_value = fit_point(
                coefficients, splines[:, point], spline_mapping, knot_steps,
                j_rot, n_spline, n_dimensions)
            fit[point] = fit_value

            sq = weights[point] * (values[point] - fit_value)
            sq *= sq

            for dimension in range(n_dimensions):
                d_index = p_map[dimension]
                knot_weights[dimension, d_index] += sq
                knot_coordinates[dimension, d_index] += (
                    sq * coordinates[dimension, point])

            point = next_indices[point]

    return fit, knot_weights, knot_coordinates


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def add_knot(knot_weights, knot_coords, panel_shape, knots, n_knots,
             knot_estimate, k1):  # pragma: no cover
    """
    Add a knot to the spline fit.

    Adds a knot near the currently highest weighted knot.

    Parameters
    ----------
    knot_weights : numpy.ndarray (float)
        The knot weights of shape (n_dimensions, n_knots).
    knot_coords : numpy.ndarray (float)
        The knot coordinates of shape (n_dimensions, n_knots).
    panel_shape : numpy.ndarray (int)
        The number of panels in the spline fit of shape (n_dimensions,).
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    n_knots : numpy.ndarray (int)
        The number of knots in each dimension of shape (n_dimensions,).
    knot_estimate : numpy.ndarray (int)
        The maximum number of knots allowable of shape (n_dimensions,).
    k1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) where k1[dimension] =
        degree[dimension] + 1.

    Returns
    -------
    exit_code : int
        Returns an exit code of 1 if the maximum number of allowable knots has
        already been reached and no more should be added.  Returns an exit code
        of 5 if the new knot location coincides with an already existing knot.
        Returns 0 if a knot was successfully added.
    """
    n_dimensions = k1.size
    while True:
        for dimension in range(n_dimensions):
            if n_knots[dimension] < knot_estimate[dimension]:
                break
        else:
            exit_code = 1
            return exit_code

        max_weight = 0.0
        max_dimension = -1
        max_index = -1
        for dimension in range(n_dimensions):
            if n_knots[dimension] >= knot_estimate[dimension]:
                continue
            for index in range(panel_shape[dimension]):
                w = knot_weights[dimension, index]
                if w > max_weight:
                    max_dimension = dimension
                    max_index = index
                    max_weight = w

        if max_dimension == -1:
            exit_code = 5
            return exit_code

        x = knot_coords[max_dimension, max_index]
        x /= knot_weights[max_dimension, max_index]
        knot_weights[max_dimension, max_index] = 0.0  # don't use again

        knot_line = knots[max_dimension]
        knot_index = max_index + k1[max_dimension]
        fac1 = knot_line[knot_index] - x
        fac2 = x - knot_line[knot_index - 1]
        if fac1 > (10 * fac2) or fac2 > (10 * fac1):
            continue
        for i in range(n_knots[max_dimension] - 1, knot_index - 1, -1):
            knot_line[i + 1] = knot_line[i]
        knot_line[knot_index] = x
        n_knots[max_dimension] += 1
        exit_code = 0
        return exit_code


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def evaluate_bspline(knot_line, degree, x, knot_index, spline=None
                     ):  # pragma: no cover
    """
    Evaluate b-splines for given knots at a coordinate.

    Evaluates the (degree + 1) non-zero b-splines at t[i] <= x < t[i + 1]
    where t is the line of knots `knot_line` and i is the `knot_index`.
    This uses the stable recurrence relation of DeBoor and Cox (2007).

    Parameters
    ----------
    knot_line : numpy.ndarray (float)
        The line of monotonically increasing knots of shape (>=n_knots).
    degree : int
        The degree of spline to evaluate.
    x : float
        The coordinate at which the spline should be evaluated.
    knot_index : int
        The index (i) of the knot satisfying t[i] <= x < t[i + 1] where
        t is the `knot_line`.
    spline : numpy.ndarray (float), optional
        An optionally created array to hold the results of this function of
        shape (>=degree + 1).

    Returns
    -------
    spline : numpy.ndarray (float)
        The output spline of shape (degree + 1) if `spline` is not supplied as
        an input parameter, or (spline.size,) otherwise.
    """
    if spline is None:
        spline = np.empty(degree + 1, dtype=nb.float64)

    spline[0] = 1.0
    z = np.empty(spline.size, dtype=nb.float64)
    for j in range(degree):
        j1 = j + 1
        for i in range(j1):
            z[i] = spline[i]
        spline[0] = 0.0
        for i in range(j1):
            i1 = i + 1
            ku = knot_index + i1
            kl = ku - j1
            low = knot_line[kl]
            high = knot_line[ku]
            if low == high:
                spline[i1] = 0.0
            else:
                factor = z[i] / (high - low)
                spline[i] += factor * (high - x)
                spline[i1] = factor * (x - low)

    return spline


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def determine_smoothing_spline(knots, n_knots, knot_estimate, degrees,
                               initial_sum_square_residual, smoothing,
                               smoothing_difference,
                               n_coefficients, bandwidth,
                               amat, beta, max_iteration, knot_steps,
                               knot_mapping, eps, splines, start_indices,
                               next_indices, panel_mapping, spline_mapping,
                               coordinates, values, weights, panel_shape,
                               accuracy):  # pragma: no cover
    """
    Smooth the current solution to a specified level.

    Parameters
    ----------
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    n_knots : numpy.ndarray (int)
        The number of knots in each dimension of shape (n_dimensions,).
    knot_estimate : numpy.ndarray (int)
        The maximum number of knots allowable of shape (n_dimensions,).
    degrees : numpy.ndarray (int)
        The degrees of the spline in each dimension (n_dimensions,).
    initial_sum_square_residual : float
        The initial sum square of the residuals from the first knot fit.
    smoothing : float
        Used to specify the smoothing factor.
    smoothing_difference : float
        The sum of the square residuals minus the smoothing factor.
    n_coefficients : int
        The number of coefficients to fit.
    bandwidth : int
        The bandwidth of the observation.
    amat : numpy.ndarray (float)
        The 'A' in the system Ax = B of shape (>=n_coefficients, >=bandwidth).
    beta : numpy.ndarray (float)
        The 'B' in the system Ax = B of shape (>=n_coefficients,).
    max_iteration : int
        The maximum number of iterations used to determine the
        smoothing spline.
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    knot_mapping : numpy.ndarray (int)
        An array containing the knot mapping (flat to n-D) indices.  This is
        created by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    eps : float
        The precision to determine singular values of A.  If any row of
        amat[:,0] < (eps * max(amat[:,0])) it will be considered singular.
    splines : numpy.ndarray (float)
        The splines of shape (n_dimensions, n_data, max(k1)).
    start_indices : numpy.ndarray (int)
        The starting indices of shape (m,) as returned by
        :func:`create_ordering`.
    next_indices : numpy.ndarray (int)
        The next indices of shape (m,) as returned by :func:`create_ordering`.
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    spline_mapping : numpy.ndarray (int)
        The 1-D to N-D spline mapping array as returned by
        :func:`flat_index_mapping`.  Should be of shape
        (n_dimensions, n_spline).
    coordinates : numpy.ndarray (float)
        The coordinates at which to evaluate the spline of shape
        (n_dimensions, n).
    values : numpy.ndarray (float)
        An array of values of shape (n_data,).
    weights : numpy.ndarray (float)
        The value weights of shape (n_data,).
    panel_shape : numpy.ndarray (int)
        The number of panels in the spline fit of shape (n_dimensions,).
    accuracy : float
        The accuracy that is used to determine when a suitable smoothing fit
        has been achieved.  Iterations will stop when
        abs(smoothing - sum_square_residuals) < accuracy.

    Returns
    -------
    coefficients, fp, ier, fitted_values : ndarray, float, float, ndarray
        The new spline coefficients, sum of the residuals^2, exit code, and
        the fitted values.
    """
    n_dimensions = degrees.size
    k1 = degrees + 1
    k2 = k1 + 1
    b_splines = np.zeros((n_dimensions, knot_estimate.max(), k2.max()),
                         dtype=nb.float64)

    nk1 = n_knots - k1

    for dimension in range(n_dimensions):
        if nk1[dimension] == k1[dimension]:
            continue
        discontinuity_jumps(knots[dimension], n_knots[dimension],
                            degrees[dimension], b_splines[dimension])

    fp = np.nan
    fp0 = initial_sum_square_residual
    s = smoothing
    p1 = 0.0
    f1 = fp0 - s
    p3 = -1.0
    f3 = smoothing_difference
    p = 0.0

    for i in range(n_coefficients):
        p += amat[i, 0]

    rn = n_coefficients
    p = rn / p

    # Find the bandwidth of the extended observation matrix
    iband3 = k1[0]
    if n_dimensions > 1:
        for dimension in range(1, n_dimensions):
            iband3 *= nk1[dimension]
    iband4 = iband3 + 1

    ich1 = 0
    ich3 = 0
    row = np.empty(iband4, dtype=nb.float64)
    h = np.empty(iband4, dtype=nb.float64)
    q = np.empty((n_coefficients, iband4), dtype=nb.float64)
    ff = np.empty(n_coefficients, dtype=nb.float64)
    coefficients = np.zeros(n_coefficients, dtype=nb.float64)
    fitted_values = np.zeros(start_indices.size, dtype=nb.float64)

    for iteration in range(max_iteration):
        pinv = 1.0 / p
        for i in range(n_coefficients):
            ff[i] = beta[i]
            for j in range(bandwidth):
                q[i, j] = amat[i, j]

            if iband4 > bandwidth:
                for j in range(bandwidth, iband4):
                    q[i, j] = 0.0

        for dimension in range(n_dimensions):
            if nk1[dimension] == k1[dimension]:
                continue
            knot_step = knot_steps[dimension]
            # Extend the observation matrix with the rows of a matrix,
            # expressing that for x=cst. sp(x, y) must be a polynomial in y of
            # degree ky.

            for ll in range(iband4):
                row[ll] = 0.0

            for i in range(k2[dimension], nk1[dimension] + 1):
                ii = i - k2[dimension]

                for ll in range(k2[dimension]):
                    row[ll * knot_step] = b_splines[dimension, ii, ll] * pinv

                j_rots = np.nonzero(knot_mapping[dimension] == ii)[0]
                for j_rot in j_rots:

                    zi = 0.0

                    # Copy the row
                    for ll in range(iband4):
                        h[ll] = row[ll]

                    for i_rot in range(j_rot, n_coefficients):
                        pivot = row[0]
                        i2 = min(iband3, n_coefficients - i_rot)
                        if pivot != 0:
                            q[i_rot, 0], cos, sin = givens_parameters(
                                pivot, q[i_rot, 0])
                            zi, ff[i_rot] = givens_rotate(
                                cos, sin, zi, ff[i_rot])
                            if i2 <= 0:
                                break

                            for ll in range(i2):
                                l1 = ll + 1
                                h[l1], q[i_rot, l1] = givens_rotate(
                                    cos, sin, h[l1], q[i_rot, l1])

                        if i2 <= 0:
                            break
                        for ll in range(i2):
                            h[ll] = h[ll + 1]
                        h[i2] = 0.0
                    else:
                        continue
                    break

        dmax = 0.0
        for i in range(n_coefficients):
            if q[i, 0] <= dmax:
                continue
            dmax = q[i, 0]
        sigma = eps * dmax

        for i in range(n_coefficients):
            if q[i, 0] <= sigma:
                coefficients, _, rank = solve_rank_deficiency(
                    q, ff, n_coefficients, iband4, sigma)
                break
        else:
            coefficients = back_substitute(q, ff, n_coefficients, iband4)
            rank = n_coefficients

        for i in range(n_coefficients):
            q[i, 0] /= dmax

        fitted_values, knot_weights, knot_coordinates = knot_fit(
            splines=splines,
            coefficients=coefficients,
            start_indices=start_indices,
            next_indices=next_indices,
            panel_mapping=panel_mapping,
            spline_mapping=spline_mapping,
            knot_steps=knot_steps,
            panel_shape=panel_shape,
            k1=k1,
            weights=weights,
            values=values,
            coordinates=coordinates)

        fp = np.sum((weights * (values - fitted_values)) ** 2)
        fpms = fp - s

        if abs(fpms) <= accuracy:
            if rank != n_coefficients:
                # Rank deficient solution
                ier = -rank
                return coefficients, fp, ier, fitted_values
            else:
                # Good solution
                ier = 0
                return coefficients, fp, ier, fitted_values

        p2 = p
        f2 = fpms

        if ich3 == 0:
            if (f2 - f3) <= accuracy:
                # The initial choice of p is too large
                p3 = p2
                f3 = f2
                p *= 0.04
                if (p <= p1):
                    p = (0.9 * p1) + (0.1 * p2)
                continue  # next iteration
            elif f2 < 0:
                ich3 = 1

        if ich1 == 0:
            if (f1 - f2) <= accuracy:
                # The initial choice of p is too small
                p1 = p2
                f1 = f2
                p /= 0.04

                if p3 < 0:
                    continue  # next iteration
                if p >= p3:
                    p = (0.1 * p2) + (0.9 * p3)
                continue  # next iteration

            ich1 = 1

        if (f2 >= f1) or (f2 <= f3):
            # Can't determine p
            ier = 2
            return coefficients, fp, ier, fitted_values

        p, p1, f1, p2, f2, p3, f3 = rational_interp_zero(p1, f1, p2,
                                                         f2, p3, f3)

    else:
        # Iteration reached maximum number
        ier = 3
        return coefficients, fp, ier, fitted_values


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def discontinuity_jumps(knot_line, n_knot, degree, b_spline
                        ):  # pragma: no cover
    """
    Calculates the discontinuity jumps.

    Calculates the discontinuity jumps of the kth derivative of the b-splines
    of degree k at the knots k+2 -> n - k - 1.  The results are updated
    in-place in the `b_spline` array.

    Adapted from the Fortran function fpdisc in the fitpack library.

    Parameters
    ----------
    knot_line : numpy.ndarray (float)
        The knot locations in a single dimension of shape (>= n_knots,).
    n_knot : int
        The number of knots in the knot line.
    degree : int
        The degree of the spline.
    b_spline : numpy.ndarray (float)
        An array of shape (max(knot_estimate), k + 2) where k is the degree of
        the spline containing the spline coefficients.  Values will be updated
        in-place.

    Returns
    -------
    None
    """
    k1 = degree + 1
    k2 = k1 + 1
    nk1 = n_knot - k1
    fac = (nk1 - degree) / (knot_line[nk1] - knot_line[degree])
    h = np.zeros(2 * k1, dtype=nb.float64)

    for ll in range(k1, nk1):

        lmk = ll - k1
        for j in range(k1):
            ik = j + k1
            lj = ll + j + 1
            lk = lj - k2
            h[j] = knot_line[ll] - knot_line[lk]
            h[ik] = knot_line[ll] - knot_line[lj]

        lp = lmk
        for j in range(k2):
            jk = j
            prod = h[j]
            for i in range(degree):
                jk += 1
                prod *= h[jk] * fac

            lk = lp + k1
            b_spline[lmk, j] = (knot_line[lk] - knot_line[lp]) / prod
            lp += 1


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def rational_interp_zero(p1, f1, p2, f2, p3, f3):  # pragma: no cover
    """
    Determines p where (u + p + v)/(p + w) = 0.

    Given three points (p1, f1), (p2, f2), (p3, f3), rational_interp_zero gives
    the value of p such that the rational interpolating function of the form
    r(p) = (u*p+v)/(p+w) equals zero at p.

    Adapted from the Fortran function fprati in the fitpack library.

    Parameters
    ----------
    p1 : float
    f1 : float
    p2 : float
    f2 : float
    p3 : float
    f3 : float

    Returns
    -------
    p, p1, f1, p2, f2, p3, f3 : float
    """

    if p3 <= 0:
        # The value of p in case p3 = infinity
        p = (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3)
    else:
        h1 = f1 * (f2 - f3)
        h2 = f2 * (f3 - f1)
        h3 = f3 * (f1 - f2)
        p = -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (
            p1 * h1 + p2 * h2 + p3 * h3)

    # Adjust the value of p1, f1, p3, and f3 such that f1 > 0 and f3 < 0.
    if f2 >= 0:
        p1 = p2
        f1 = f2
    else:
        p3 = p2
        f3 = f2

    return p, p1, f1, p2, f2, p3, f3


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def fit_point(coefficients, spline, spline_mapping, knot_steps, j_rot,
              n_spline, n_dimensions):  # pragma: no cover
    """
    Evaluate fitted value given a spline and coefficients.

    Parameters
    ----------
    coefficients : numpy.ndarray (float)
        The coefficients of shape (n_coefficients,).
    spline : numpy.ndarray (float)
        The spline for the point of shape (n_dimensions, n_spline).
    spline_mapping : numpy.ndarray (int)
        The 1-D to N-D spline mapping array as returned by
        :func:`flat_index_mapping`.  Should be of shape
        (n_dimensions, n_spline).
    knot_steps : numpy.ndarray (int)
        The N-D to 1-D knot mapping steps as returned by
        :func:`flat_index_mapping`.  Should be of shape (n_dimensions,).
    j_rot : int
        The starting 1-D index on the coefficient array for the given spline.
    n_spline : int
        The total number of spline coefficients that will be used to perform
        the fit.
    n_dimensions : int
        The number of dimensions in the fit.

    Returns
    -------
    fitted_value : float
    """
    fit_value = 0.0
    for i in range(n_spline):
        i1 = j_rot
        s = 1.0
        for dimension in range(n_dimensions):
            spline_i = spline_mapping[dimension, i]
            s *= spline[dimension, spline_i]
            i1 += spline_i * knot_steps[dimension]
        s *= coefficients[i1]
        fit_value += s
    return fit_value


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def perform_fit(coordinates, knots, coefficients, degrees, panel_mapping,
                panel_steps, knot_steps, nk1, spline_mapping, n_knots
                ):  # pragma: no cover
    """
    Evaluate a given spline at multiple coordinates.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        The coordinates at which to evaluate the spline of shape
        (n_dimensions, n).
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    coefficients : numpy.ndarray (float)
        The spline coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The degrees of the spline in each dimension (n_dimensions,).
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    panel_steps : numpy.ndarray (int)
        The flat index mapping steps in panel-space of shape (n_dimensions,).
        These are returned by passing the shape `Spline.panel_shape` into
        :func:`flat_index_mapping`.
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    nk1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) containing the values n_knots - k1
        where n_knots are the number of knots in each dimension, and k1 are the
        spline degrees + 1 in each dimension.
    spline_mapping : numpy.ndarray (int)
        An array containing the spline mapping (flat to n-D) indices.  This is
        created by passing the spline shape (degrees + 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_spline_coefficients).
    n_knots : numpy.ndarray (int)
        The number of knots in each dimension (n_dimensions,).

    Returns
    -------
    fitted_values : numpy.ndarray (float)
        The spline evaluated at `coordinates` of shape (n,).
    """

    n_dimensions, m = coordinates.shape
    k1 = degrees + 1
    x = np.empty(n_dimensions, dtype=nb.float64)
    n_spline = int(np.prod(k1))
    spline = np.empty((n_dimensions, np.max(k1)), dtype=nb.float64)
    fitted_values = np.empty(m, dtype=nb.float64)
    upper_limits = np.empty(n_dimensions, dtype=nb.float64)
    lower_limits = np.empty(n_dimensions, dtype=nb.float64)
    # Find the panel to which each point belongs

    knot_indices = find_knots(
        coordinates=coordinates,
        knots=knots,
        valid_knot_start=degrees,
        valid_knot_end=nk1)

    panel_indices = np.zeros(m, dtype=nb.int64)
    for dimension in range(n_dimensions):
        degree = degrees[dimension]
        step = panel_steps[dimension]
        knot_line = knots[dimension]
        lower_limits[dimension] = knot_line[0]
        upper_limits[dimension] = knot_line[n_knots[dimension] - 1]
        for i in range(m):
            panel_indices[i] += (knot_indices[dimension, i] - degree) * step

    start_indices, next_indices = create_ordering(panel_indices)
    n_panels = np.nonzero(start_indices != -1)[0][-1] + 1

    for panel in range(n_panels):
        point = start_indices[panel]
        if point == -1:
            continue
        p_map = panel_mapping[:, panel]
        knot_index = p_map + degrees
        j_rot = 0
        for dimension in range(n_dimensions):
            j_rot += p_map[dimension] * knot_steps[dimension]

        while point != -1:
            for dimension in range(n_dimensions):
                x_val = coordinates[dimension, point]
                ll = lower_limits[dimension]
                ul = upper_limits[dimension]
                if x_val < ll:
                    x[dimension] = ll
                elif x_val > ul:
                    x[dimension] = ul
                else:
                    x[dimension] = x_val
                evaluate_bspline(
                    knots[dimension], degrees[dimension], x[dimension],
                    knot_index[dimension], spline[dimension])

            fitted_values[point] = fit_point(
                coefficients, spline, spline_mapping, knot_steps, j_rot,
                n_spline, n_dimensions)
            point = next_indices[point]

    return fitted_values


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def single_fit(coordinate, knots, coefficients, degrees, panel_mapping,
               panel_steps, knot_steps, nk1, spline_mapping,
               k1=None, n_spline=None, work_spline=None,
               lower_bounds=None, upper_bounds=None):  # pragma: no cover
    """
    Return a fitted value at the given coordinate.

    Parameters
    ----------
    coordinate : numpy.ndarray (float)
        The coordinate at which to return a fit of shape (n_dimensions,).
    knots : numpy.ndarray (float)
        The knots in each dimension of shape (n_dimensions, max_knot_estimate).
        Must be monotonically increasing for each dimension.
    coefficients : numpy.ndarray (float)
        The coefficients of shape (n_coefficients,).
    degrees : numpy.ndarray (int)
        The degrees of the spline in each dimension (n_dimensions,).
    panel_mapping : numpy.ndarray (int)
        An array containing the panel mapping (flat to n-D) indices.  This is
        created by passing the panel shape (n_knots - (2 * degrees) - 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_panels).
    panel_steps : numpy.ndarray (int)
        The flat index mapping steps in panel-space of shape (n_dimensions,).
        These are returned by passing the shape `Spline.panel_shape` into
        :func:`flat_index_mapping`.
    knot_steps : numpy.ndarray (int)
        The flat index mapping steps in knot-space of shape (n_dimensions,).
        These are returned by passing the shape (n_knots - degrees - 1) into
        :func:`flat_index_mapping`.
    nk1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) containing the values n_knots - k1
        where n_knots are the number of knots in each dimension, and k1 are the
        spline degrees + 1 in each dimension.
    spline_mapping : numpy.ndarray (int)
        An array containing the spline mapping (flat to n-D) indices.  This is
        created by passing the spline shape (degrees + 1) into
        :func:`flat_index_mapping`.  Should be an array of shape
        (n_dimensions, n_spline_coefficients).
    k1 : numpy.ndarray (int)
        An array of shape (n_dimensions,) where k1[dimension] =
        degree[dimension] + 1.
    n_spline : int
        The total number of spline coefficients that will be used to perform
        the fit.
    work_spline : numpy.ndarray (float)
        An optional work array of shape (n_dimensions, max(k1)) that can
        be supplied in order to skip the overhead involved with array creation.
    lower_bounds : numpy.ndarray (float)
        Specifies the lower range of valid coordinates for each dimension and
        is of shape (n_dimensions,).
    upper_bounds : numpy.ndarray (float)
        Specifies the upper range of valid coordinates for each dimension and
        is of shape (n_dimensions,).

    Returns
    -------
    fitted_value : float
    """

    knot_index = find_knot(
        coordinate=coordinate,
        knots=knots,
        valid_knot_start=degrees,
        valid_knot_end=nk1,
        allow_outside=False,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds)

    if knot_index[0] == -1:
        return np.nan

    n_dimensions = coordinate.size
    if k1 is None:
        k1 = degrees + 1

    if work_spline is None:
        work_spline = np.empty((n_dimensions, np.max(k1)), dtype=nb.float64)

    if n_spline is None:
        n_spline = int(np.prod(k1))

    panel = 0
    for dimension in range(n_dimensions):
        panel += (knot_index[dimension] - degrees[dimension]
                  ) * panel_steps[dimension]
        evaluate_bspline(
            knots[dimension], degrees[dimension], coordinate[dimension],
            knot_index[dimension], work_spline[dimension])

    panel_map = panel_mapping[:, panel]
    j_rot = 0
    for dimension in range(n_dimensions):
        j_rot += panel_map[dimension] * knot_steps[dimension]

    fitted_value = fit_point(
        coefficients, work_spline, spline_mapping, knot_steps, j_rot,
        n_spline, n_dimensions)

    return fitted_value


def perform_fit_slow(coordinates, knots, coefficients, degrees, panel_mapping,
                     panel_steps, knot_steps, nk1, spline_mapping, n_knots
                     ):  # pragma: no cover
    """
    This is for testing purposes only.

    Parameters
    ----------
    coordinates
    knots
    coefficients
    degrees
    panel_mapping
    panel_steps
    knot_steps
    nk1
    spline_mapping
    n_knots

    Returns
    -------

    """

    n_dimensions, m = coordinates.shape
    k1 = degrees + 1
    x = np.empty(n_dimensions, dtype=float)
    n_spline = int(np.prod(k1))
    spline = np.empty((n_dimensions, np.max(k1)), dtype=float)
    fitted_values = np.empty(m, dtype=float)
    upper_limits = np.empty(n_dimensions, dtype=float)
    lower_limits = np.empty(n_dimensions, dtype=float)
    # Find the panel to which each point belongs

    knot_indices = find_knots(
        coordinates=coordinates,
        knots=knots,
        valid_knot_start=degrees,
        valid_knot_end=nk1)

    panel_indices = np.zeros(m, dtype=int)
    for dimension in range(n_dimensions):
        degree = degrees[dimension]
        step = panel_steps[dimension]
        knot_line = knots[dimension]
        lower_limits[dimension] = knot_line[0]
        upper_limits[dimension] = knot_line[n_knots[dimension] - 1]
        for i in range(m):
            panel_indices[i] += (knot_indices[dimension, i] - degree) * step

    start_indices, next_indices = create_ordering(panel_indices)
    n_panels = np.nonzero(start_indices != -1)[0][-1] + 1

    for panel in range(n_panels):
        point = start_indices[panel]
        if point == -1:
            continue
        p_map = panel_mapping[:, panel]
        knot_index = p_map + degrees
        j_rot = 0
        for dimension in range(n_dimensions):
            j_rot += p_map[dimension] * knot_steps[dimension]

        while point != -1:
            for dimension in range(n_dimensions):
                x_val = coordinates[dimension, point]
                ll = lower_limits[dimension]
                ul = upper_limits[dimension]
                if x_val < ll:
                    x[dimension] = ll
                elif x_val > ul:
                    x[dimension] = ul
                else:
                    x[dimension] = x_val
                evaluate_bspline(
                    knots[dimension], degrees[dimension], x[dimension],
                    knot_index[dimension], spline[dimension])

            fitted_values[point] = fit_point(
                coefficients, spline, spline_mapping, knot_steps, j_rot,
                n_spline, n_dimensions)
            point = next_indices[point]

    return fitted_values
