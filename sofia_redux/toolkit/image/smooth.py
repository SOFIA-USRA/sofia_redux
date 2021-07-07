# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.fitting.polynomial import polysys, gaussj


__all__ = ['quadfit', 'bicubic_coefficients', 'bicubic_evaluate',
           'fitplane', 'fiterpolate']


# bi-cubic coefficients (bcucof in Numerical Recipes)
_bicubic_weights = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
    [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
    [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
    [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
    [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
    [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
    [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
    [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
    [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1]
]).astype(float)


def quadfit(image):
    """
    Quick and simple cubic polynomial fit to surface - no checks

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    coefficients : numpy.ndarray of numpy.float64 (6,)
        where:

            z = c[0] + c[1].x + c[2].y + c[3].x^2 + c[4].y^2 + c[5].xy
    """
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    exponents = np.array([[0, 0],  # c[0]
                          [1, 0],  # c[1].x
                          [0, 1],  # c[2].y
                          [2, 0],  # c[3].x**2
                          [0, 2],  # c[4].y**2
                          [1, 1]])  # c[5].x.y
    samples = np.empty((3, image.size))
    samples[0] = xx.ravel()
    samples[1] = yy.ravel()
    samples[2] = image.ravel()
    alpha, beta = polysys(samples, -1, exponents=exponents, ignorenans=False)
    return gaussj(alpha, beta)


def bicubic_coefficients(z, dx, dy, dxy, nx, ny):
    """
    Returns the coefficients necessary for bicubic interpolation.

    Parameters
    ----------
    z : numpy ndarray of float (N,)
        The grid points values starting at the lower left and moving
        counterclockwise.
    dx : numpy ndarray of float (N,)
        The gradient in dimension 1 evaluated at y.
    dy : numpy ndarray of float (N,)
        The gradient in dimension 2 evaluated at y.
    dxy : numpy ndarray of float (N,)
        The cross derivative evaluated at y.
    nx : float
        Number of grid points in the x direction.
    ny : float
        Number of grid points in the y direction.

    Returns
    -------

    """
    nxy = nx * ny
    x = np.hstack((z, dx * nx, dy * ny, dxy * nxy))
    return (_bicubic_weights @ x).reshape(4, 4)


def bicubic_evaluate(z, dx, dy, dxy, xrange, yrange, x, y):
    """

    Parameters
    ----------
    z : numpy.ndarray (4,)
        Functional values at grid points
    dx : numpy.ndarray (4,)
        Derivative in the x direction
    dy : numpy.ndarray (4,)
        Derivative in the y direction
    dxy : numpy.ndarray (4,)
        Cross derivative
    xrange : array_like (2,)
        The (lower, upper) coordinates of the grid in the x direction
    yrange : array_like of float (2,)
        The (lower, upper) coordinates of the grid in the y direction
    x : numpy.ndarray (shape)
        x coordinates at desired interpolation points
    y : numpy.ndarray (shape)
        y coordinates at desired interpolation points

    Returns
    -------
    z : numpy.ndarray (shape)
        The bicubic evaluation at `x` and `y`.
    """
    nx, ny = xrange[1] - xrange[0], yrange[1] - yrange[0]
    c = bicubic_coefficients(z, dx, dy, dxy, nx, ny)
    t = (x - xrange[0]) / nx
    u = (y - yrange[0]) / ny

    z = x * 0.0
    for i in range(3, -1, -1):
        z *= t
        z += c[i, 0] + u * (c[i, 1] + u * (c[i, 2] + u * c[i, 3]))
    return z


def fiterpolate(image, nx, ny):
    """
    Fits a smooth surface to data using J. Tonry's fiterpolate routine

    Breaks the image up into shape[0] rows and shape[1] columns.
    Determines values and derivatives at the boundary points by
    fitting quadratic surfaces to the subimages.  Uses bicubic
    interpolation to create a smoothed version of the image.

    Parameters
    ----------
    image : array_like of float (nrow, ncol)
        The 2D data to be fit
    nx : int
        Number of column grid cells.
    ny : int
        Number of row grid cells.

    Returns
    -------
    numpy.ndarray of float (shape)
        Smooth version of image
    """
    s1 = np.array(image.shape)
    s2 = np.array([ny, nx])
    ngrid = s2 + 1
    fitimage = np.zeros(s1)

    # Determine grid layout
    ratio = (s1 - 1) / s2
    gy = np.round(np.arange(ngrid[0], dtype=float) * ratio[0]).astype(int)
    gx = np.round(np.arange(ngrid[1], dtype=float) * ratio[1]).astype(int)

    xranges = np.array([gx[:-1], gx[1:] + 1])
    yranges = np.array([gy[:-1], gy[1:] + 1])
    xsizes = np.ptp(xranges, axis=0)
    ysizes = np.ptp(yranges, axis=0)

    # Determine the values z, dy1, dy2, and dy12, at the grid points
    x1 = np.mean([gx, np.roll(gx, 1)], axis=0).astype(int)
    x2 = np.roll(x1, -1)
    x1[0], x2[-1] = 0, gx[s2[1]]
    y1 = np.mean([gy, np.roll(gy, 1)], axis=0).astype(int)
    y2 = np.roll(y1, -1)
    y1[0], y2[-1] = 0, gy[s2[0]]
    x2 += 1
    y2 += 1

    # c[0] + c[1].x + c[2].y + c[3].x^2 + c[4].y^2 + c[5].x.y
    c = np.zeros((6, ngrid[0], ngrid[1]))
    for i in range(ngrid[0]):
        for j in range(ngrid[1]):
            c[:, i, j] = quadfit(image[y1[i]:y2[i], x1[j]: x2[j]])

    # determine z, dz1, dz2, and dz12 at the grid points
    # Get the value and derivatives of the function
    y, x = np.meshgrid(gy - y1, gx - x1, indexing='ij')
    z = c[0] + (c[1] * x) + (c[2] * y) + (c[3] * x ** 2) + \
        (c[4] * y ** 2) + (c[5] * x * y)
    dx = c[1] + (2 * c[3] * x) + (c[5] * y)
    dy = c[2] + (2 * c[4] * y) + (c[5] * x)
    dxy = c[5]

    for i in range(nx):
        for j in range(ny):
            y, x = np.mgrid[:ysizes[j], :xsizes[i]]
            # lower-left, lower-right, upper-right, upper-left
            xi = (i, i + 1, i + 1, i)
            yi = (j, j, j + 1, j + 1)
            fitimage[yranges[0, j]:yranges[1, j],
                     xranges[0, i]:xranges[1, i]] = \
                bicubic_evaluate(
                    z[yi, xi], dx[yi, xi], dy[yi, xi], dxy[yi, xi],
                    [0, xsizes[i]], [0, ysizes[j]], x, y)
    return fitimage


def fitplane(points):
    """
    Fit a plane to distribution of points.

    Parameters
    ----------
    points : array_like of float (2, npoints)
        Where points[0] are the coordinates in the first dimension and
        points[1] are the coordinates in the second dimension

    Returns
    -------
    A point on the plane (point cloud centroid) and the normal
    """
    points = np.reshape(points, (np.shape(points)[0], -1))
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    m = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, np.linalg.svd(m)[0][:, -1]
