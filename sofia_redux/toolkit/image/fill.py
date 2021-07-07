# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
import warnings

from astropy import log
import bottleneck
import numpy as np
from matplotlib import path
from matplotlib.transforms import Bbox
from scipy import interpolate
from scipy.sparse import csr_matrix

from sofia_redux.toolkit.fitting.polynomial import polyinterp2d


__all__ = ['clough_tocher_2dfunc', 'spline_interp_2dfunc', 'maskinterp',
           'image_naninterp', 'polyclip', 'polyfillaa', 'polygon_area',
           'polygon_weights']


def clough_tocher_2dfunc(d, cin, cout, **kwargs):
    interp = interpolate.CloughTocher2DInterpolator(cin, d, **kwargs)
    return interp(cout)


def spline_interp_2dfunc(d, cin, cout, **kwargs):
    return polyinterp2d(
        cin[:, 1], cin[:, 0], d, cout[:, 1], cout[:, 0], **kwargs)


def maskinterp(image, func=spline_interp_2dfunc,
               mask=None, apstep=1, maxap=9, cval=np.nan, minfrac=0.2,
               minpoints=4, cdis=1, statistical=False, creep=False,
               coplanar=None, **kwargs):
    """
    Interpolates over image using a mask.

    The `func` parameter defines the function used for interpolation
    within the aperture. This can be either a statistical or interpolation
    function depending on whether `statistical` is True or False:

    if statistical is True::

        data_out = func(data_in, **kwargs)

    if statistical is False::

        data_out = func(data_in, c_in, c_out, **kwargs)
        c_in, c_out : numpy.array of floats (npoints, (y, x))

    Parameters
    ----------
    image : numpy.ndarray
        Input data array (ncol, nrow)
    func : function
        The function to use for interpolating values within each
        aperture.  The format of the function is described above.
        Note the distinction between functions if statistical=True.
    mask : numpy.ndarray, optional
        True = Use as a data point for interpolation
        False = Interpolate values for these points
    apstep : float, optional
        The step by which to increase aperture radius between each
        iteration
    maxap : float, optional
        The maximum interpolation aperture radius in pixels. Iterations
        will be terminated after reaching this aperature size.
    cval : float, optional
        value with which to fill missing data points
    minfrac : float, optional
        The minimum fraction of valid points within an aperture in
        order for interpolation to be attempted (0 < minfrac < 1)
    minpoints: int, optional
        The minimum number of valid points within aperture radius
        in order for interpolation to be attempted.
    cdis : float, optional
        Maximum distance from the center of aperture to the "center of
        mass" of good pixels in the aperture.
    statistical : bool, optional
        Use statistical function rather than interpolation
    creep : bool, optional
        If True, allow interpolated values to be used in successive
        iterations.  Otherwise, only the original "good" mask=True data
        will be used in all iterations.
    coplanar : bool, optional
        If True, interpolation is only performed if interpolants are
        functions of both the x and y features.  If False, only
        one of the features may vary.  Use depends on the interpolating
        algorithm.  May be set to False for statistical / minimizing
        functions.

    Returns
    -------
    np.ndarray
        output image (nrow, ncol)
    """
    if mask is None:
        mask = ~np.isnan(image)
    if not mask.any():
        log.error("Mask is all False")
        return
    elif mask.all():
        return image.copy()
    mask = mask.astype(bool)

    if coplanar is None:
        coplanar = hasattr(func, '__name__') \
            and func.__name__ == 'clough_tocher_2dfunc'

    # geometry used to define whether interpolation should
    # be attempted
    width = (maxap * 2) + 1
    ay, ax = np.mgrid[:width, :width]
    ax -= maxap
    ay -= maxap
    dr = np.sqrt((ay ** 2) + (ax ** 2))
    ygrid, xgrid = np.mgrid[:image.shape[0], :image.shape[1]]
    basemask = mask.copy()
    found = basemask.copy()
    corrected = image.copy()
    imshape = image.shape
    radius = apstep
    while True:
        wdata = corrected.copy() if creep else image.copy()
        apmask = np.array(dr <= np.ceil(radius))
        nap = apmask.sum()
        find = ~found & ~mask
        xfind, yfind = xgrid[find], ygrid[find]
        nfind = len(xfind)

        xs = np.empty((nap, nfind), dtype=int)
        ys = np.empty((nap, nfind), dtype=int)
        dx, dy = np.empty_like(xs), np.empty_like(ys)

        for i, (offy, offx) in enumerate(zip(ay[apmask], ax[apmask])):
            xs[i], ys[i] = xgrid[find] + offx, ygrid[find] + offy
            dx[i], dy[i] = offx, offy

        # must exist inside features of image
        valid = (ys >= 0) & (ys < imshape[0]) & (xs >= 1) & (xs < imshape[1])

        # populate data and mask
        ds = np.full((nap, nfind), np.nan, dtype=image.dtype)
        ms = np.full((nap, nfind), False)
        ds[valid] = wdata.copy()[(ys[valid], xs[valid])]
        ms[valid] = basemask.copy()[(ys[valid], xs[valid])]

        # data must not be NaN
        valid[np.isnan(ds)] = False
        # do not use unmask data
        valid[~ms] = False

        # must be more than minpoints
        npts = np.sum(valid, axis=0)
        valid[:, npts < minpoints] = False

        # must be more than minfrac
        valid[:, (npts / nap) < minfrac] = False

        # coplanar?
        if coplanar:
            planevalid = np.any(valid, axis=0)
            tx, ty = dx.copy().astype(float), dy.copy().astype(float)
            tx[:, ~planevalid] = 0
            ty[:, ~planevalid] = 0
            tx = np.nanmax(tx, axis=0) - np.nanmin(tx, axis=0)
            ty = np.nanmax(ty, axis=0) - np.nanmin(ty, axis=0)
            valid[:, tx == 0] = False
            valid[:, ty == 0] = False

        # calculate center-of-mass
        cx, cy = dx.copy().astype(float), dy.copy().astype(float)
        w = np.sum(valid, axis=0)
        zi = w == 0
        cx[:, zi] = 0
        cy[:, zi] = 0
        cx, cy = np.nanmean(cx, axis=0), np.nanmean(cy, axis=0)
        cr = np.hypot(cx, cy)
        cr = cr > cdis
        valid[:, cr] = False

        pts = np.nonzero(np.any(valid, axis=0))[0]

        for pt in pts:
            idx = valid[:, pt]
            din = ds[idx, pt]
            xout = xfind[pt]
            yout = yfind[pt]
            if statistical:
                corrected[yout, xout] = func(din, **kwargs)
            else:
                cin = np.array([xs[idx, pt], ys[idx, pt]]).T
                cout = np.array([[xout], [yout]]).T
                corrected[yout, xout] = func(din, cin, cout, **kwargs)
            found[yout, xout] = True

        radius += apstep
        if radius > maxap or found.all():
            break

    if not found.all():
        corrected[~found] = cval

    return corrected


def image_naninterp(data):
    """
    Fills in NaN values in an image

    Uses the Clough-Tocher scheme to construct a piecewise cubic
    interpolating Bexier polynomial on Delaunay triangulation.

    Parameters
    ----------
    data : numpy.ndarray (nrow, ncol)
        image array with missing data represented by NaNs

    Returns
    -------
    numpy.ndarray
        output image
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        log.error("data must be a 2D %s" % np.ndarray)
        return
    mask = np.isnan(data)
    if not mask.any():
        return data
    if mask.all():
        log.error("data are all NaN")
        return

    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    points = np.array([yy[~mask], xx[~mask]]).T
    interp = interpolate.CloughTocher2DInterpolator(points, data[~mask])
    result = data.copy()
    result[mask] = interp(np.array([yy[mask], xx[mask]]).T)
    return result


def polyclip(i, j, pol_x, pol_y, area=False):
    """
    Clip a polygon to a square unit pixel

    Uses the Sutherland-Hodgman polygon clipping algorithm.  Pixel
    centers for pixel (i, j) is at (i + 0.5, j + 0.5).

    Parameters
    ----------
    i : float, int
        Pixel x-coordinate
    j : float, int
        Pixel y-coordinate
    pol_x : array_like of float
        Polygon x-coordinates (N,)
    pol_y : array_like of float
        Polygon y-coordinates (N,)
    area : bool, optional
        If True, return the area instead of the polygon

    Returns
    -------
    ((list of float), (list of float)) or float
        Clipped x and y coordinates of the polygon vertices.  If a pixel
        if fully outside the specified polygon, x=None, y=None is returned.
        If area is set to True, then the resulting area will be returned
        as a float.
    """
    n = len(pol_x)
    nout = n + 4
    px_out, py_out = [0] * nout, [0] * nout
    clip_vals = [i, i + 1, j + 1, j]

    for ctype in range(4):
        cv = clip_vals[ctype]
        if ctype == 0:
            inside = [px > i for px in pol_x]
        elif ctype == 1:
            inside = [(px < i + 1) for px in pol_x]
        elif ctype == 2:
            inside = [(py < j + 1) for py in pol_y]
        else:
            inside = [py > j for py in pol_y]
        if all(inside):
            continue

        shiftp1 = inside.copy()
        shiftp1.insert(0, shiftp1.pop(-1))
        crosses = [i1 != i2 for (i1, i2) in zip(inside, shiftp1)]
        pind = 0
        for k in range(n):
            px, py = pol_x[k], pol_y[k]
            if crosses[k]:  # out->in or in->out, add intersection
                ind = n - 1 if k == 0 else k - 1
                sx, sy = pol_x[ind], pol_y[ind]
                try:
                    if ctype <= 1:  # left or right
                        px_out[pind] = cv
                        py_out[pind] = sy + ((py - sy) / (px - sx)) * (cv - sx)
                    else:  # top or bottom
                        px_out[pind] = sx + ((px - sx) / (py - sy)) * (cv - sy)
                        py_out[pind] = cv
                except ZeroDivisionError:  # pragma: no cover
                    px_out[pind] = np.nan
                    py_out[pind] = np.nan
                pind += 1

            if inside[k]:  # out->in or in->in, add 2nd point
                px_out[pind] = px
                py_out[pind] = py
                pind += 1

            if pind >= nout - 2:
                nout *= 2
                px_out = px_out + [0] * nout
                py_out = py_out + [0] * nout
                nout *= 2

        if pind == 0:  # polygon is entirely outside this line
            return None, None
        n = pind
        pol_x = px_out[:n].copy()
        pol_y = py_out[:n].copy()

    if area:
        if pol_x is None:  # pragma: no cover
            return 0.0
        shiftx = pol_x.copy()
        shifty = pol_y.copy()
        shiftx.append(shiftx.pop(0))
        shifty.append(shifty.pop(0))
        a1 = [p[0] * p[1] for p in zip(pol_x, shifty)]
        a2 = [p[0] * p[1] for p in zip(pol_y, shiftx)]
        a = [p[0] - p[1] for p in zip(a1, a2)]
        return abs(sum(a)) / 2

    return pol_x, pol_y


def shift1(arr):
    s = arr.shape
    result = np.empty(s, dtype=arr.dtype)
    result[:, 1:] = arr[:, :-1]
    result[:, 0] = arr[:, -1]
    return result


def polyfillaa(px, py, xrange=None, yrange=None, start_indices=None,
               area=False):
    """
    Finds all pixels at least partially inside a specified polygon

    Find the number of vertices for each polygon and then loop through
    groups of polygons with an equal sides.  Then for each group of
    similar sided polygons:

    1. Create a shared pixel grid to be used by all polygons in the
       group.  The size of the grid is determined by the maximum
       range of polygon widths and heights in the group.
    2. For each polygon edge, calculate where it crosses the left
       and bottom edges of the each grid cell.
    3. Determine whether the vertices are inside the grid points.
       The crossings determined in step 2 can be used to determine
       whether the lower-left grid points are contained within a
       polygon.  In this case, a grid point is said to be in the
       polygon if there are an odd number of intersection points
       with the polygon along the y-coordinate of the grid point.
    4. Remember that we have only calculated polygon crossings on the
       left and lower edges of each cell and whether the lower-left
       corner of a cell is enclosed in the polygon.  To avoid
       duplicate calculations, take note of the following facts:

           a) If a lower-left grid point (gp) is enclosed in the polygon
              then the lower-right gp of the cell to the left, the
              upper-right gp of the cell to the lower-left, and the
              upper-left gp of the cell below must also be enclosed.
           b) If the polygon crosses the left edge of the cell, it
              must also cross the right edge of the cell to the left.
           c) If the polygon crosses the bottom edge of the cell, it
              must also cross the top edge of the cell below it.

    5. Given all of these points, it is clear that the maximum number
       of clipped polygon vertices will be equal to 2 times the number
       of polygon vertices of the input group of polygons since each
       edge can cross a maximum of 2 cell sides, or be clipped to
       where one vertex remains inside the cell and the other is
       located on the edge of the cell.  If both vertices are inside
       the cell then that edge remains unchanged.  Therefore the
       maximum number of clipped polygon vertices occurs when all
       polygon edges intersect 2 cell edges (imagine two superimposed
       squares and then rotating one by 45 degrees).  We create a
       3D output array containing the new vertices for each polygon
       and for each pixel and fill it in the following order:

           a) vertices that are inside
           b) grid points that are inside
           c) clipped vertices

    6. If we do not need to calculate area, then we can stop here
       since we have all the new vertices.  However, if we do need
       to calculate area, then these points need to be arranged in
       clockwise or anti-clockwise order.  This is done by ordering
       the points based on angle with respect to the center-of-mass
       of the points.  As a side note, this takes up a significant
       amount of processing time (the sorting, not the angle).  I
       attempted many alternate solutions to this by not changing
       the original order of the input vertices and clipping
       in-place.  One of the most promising methods was to encode
       where a vertex was in relation to the cell and then clip
       based on that, ordering by keeping points in the following
       manner and looping around the edges of a cell in the order
       bottom -> left -> top -> right:

           outside -> inside = keep the inside and clipped vertices
           inside -> outside = clipped vertex only
           inside -> inside = keep the second inside point
           outside -> outside = keep both clipped vertices

        The problem occurs with whether grid points (corners) should be
        included or not, and where they are located in the final order
        of points.  This can be achieved by encoding a vertex location
        relative to the cell as bits where 1 indicates outside and 0
        indicates inside in the order bottom-left-top-right. So for
        example, 0000 indicates a point is inside the cell and 0100
        indicates a vertex is to the left of the cell.  codes
        containing two 1's indicate corners, and a set of rules can then
        be established determining whether a gp is inside or outside
        based on the code combination of vertices for one edge.

        However, in order to achieve vectorization, this requires
        storing at least 16 times the amount of initial data in the
        cache (number of polygons * area of polygon * 16) which can
        be huge and clumsy.  We can get around this with loops, but
        this is not efficient with Python.  Therefore, it is quicker
        and safer to just use a sort on the angle and be done with it.
        If you think there's a better solution then please feel free
        to implement it (and tell me too for my own curiosity).

    7. Calculate area using the shoelace formula::

            A = 0.5 * sum( x_i * y_(j+1)) - sum(x_(i+1) * y(j))|

    8. Finally, organize the results by lumping together polygons
       based on the number of cells enclosed within.  This allows
       us to grab which cells belong to which polygons quickly.

    Parameters
    ----------
    px : array_like of (int or float)
        Contains the x-coordinates of the polygons.  May be provided as a flat
        array if `start_indices` is provided. Alternitvely, a 2-level nested
        list may be provided with each sublist containing vertices for a
        single polygon.
    py : array_like of (int or float)
        Contains the y-coordinates of the polygons.  May be provided as a flat
        array if `start_indices` is provided. Alternitvely, a 2-level nested
        list may be provided with each sublist containing vertices for a
        single polygon.
    xrange : array_like, optional
        (2,) [lower x-limit, upper x-limit] array defining the x range
        on which the polygon is superimposed (number of columns).
        size of the pixel grid on which the polygon is superimposed
        Supplying it will clip all x results to within
        xrange[0]->xrange[1].
    yrange : array_like, optional
        (2,) [lower y-limit, upper y-limit] array defining the y range
        on which the polygon is superimposed (number of rows).
        size of the pixel grid on which the polygon is superimposed
        Supplying it will clip all y results to within
        yrange[0]->yrange[1].
    start_indices : array_like of int, optional
        Multiple polygon shapes may be specified with the `polygons`
        parameter by specifying indices in the first dimension of
        `polygons` which should be considered a first vertex of a
        polygon.  For example, the nth polygon consists of vertices
        at polygons[n:(n+1)].  Note that `start_indices` will be sorted
        and that the last index (start_indices[-1]) gives the last point
        belonging to start_indices[-2].  The last polygons vertex is not
        automatically appended.  i.e. the last polygon is
        polygons[start_indices[-2]:start_indices[-1]], not
        polygons[start_indices[-1]:].
    area : bool, optional
        if True, return an additional dictionary containing the area

    Returns
    -------
    polygon, [areas] : (2-tuple of (dict or array)) or (dict or array)
        A dictionary containing the output:
            polygon index (int) -> tuples of (y, x) grid coordinates contained
            within each polygon
        areas (optional if `area` is set to True)
            polygon index (int) -> list of pixel areas in the same order as
                given by output indices.

    Notes
    -----
    I vectorized the crap out of this thing and removed redundancies
    so it could run faster than the IDL version.  The IDL version was
    loop driven which is generally a no no, but it this case it was
    very very well done and also able to use C compiled clipping code.
    Python looping using the same method could not even slightly keep up.
    For example, 50,000 polygons on a 256, 256 grid, where each polygon
    covered an area of 3x3 grid pixels took 20 seconds using the Python
    equivalent of the IDL code with all the speed saving tricks / data
    types and objects available.  In comparison, IDL took ~3 seconds
    while this method took ~1 second.  If you want to attempt speeding
    it up further then the main choking point is the sorting of angles
    when calculating area.  If you want to use the old method for
    calculating output polygons/area, replace the main body of code
    after normalization to a regular pixel grid to loop through all
    polygons, then all pixels covered by the polygon and calculate
    the area using `sofia_redux.toolkit.polygon.polyclip`.
    """
    if start_indices is None:
        if hasattr(px[0], '__len__'):
            single = False
            poly_ind = [0]
            count = 0
            ox, oy = px, py
            px, py = [], []
            for i in range(len(ox)):
                count += len(ox[i])
                poly_ind.append(count)
                px.extend(ox[i])
                py.extend(oy[i])
            poly_ind = np.array(poly_ind)
        else:
            single = True
            poly_ind = np.array([0, len(px)])
    else:
        poly_ind = np.array(start_indices, dtype=int)
        poly_ind = np.append(poly_ind, px.size)
        single = False

    if not isinstance(px, np.ndarray):
        px = np.array(px, dtype=float)
        py = np.array(py, dtype=float)

    if px.shape != py.shape:
        raise ValueError("px and py must be the same shape")
    elif px.ndim != 1:
        raise ValueError("polygons must be flat arrays")

    npoly = poly_ind[1:] - poly_ind[:-1]
    n = npoly.size
    minpoly = np.min(npoly)
    nbins = np.max(npoly) - minpoly + 1
    binned = (npoly - minpoly).astype(int)
    npoly_ind = np.arange(n)
    csr = csr_matrix(
        (npoly_ind, [binned, np.arange(n)]), shape=(nbins, n))

    areas = {} if area else None
    result = {}

    for i, put in enumerate(np.split(csr.data, csr.indptr[1:-1])):

        # number of vertices for each polygon in this group
        nvert = i + minpoly
        nshapes = put.size  # number of nvert sided shapes in polygon list

        # take holds indices of vertices in px and py for each polygon
        take = np.repeat([poly_ind[put]], nvert, axis=0).T
        take += np.arange(nvert)

        # store the left most and lowest pixel covered by each polygon
        left = np.floor(np.min(px[take], axis=1)).astype(int)
        bottom = np.floor(np.min(py[take], axis=1)).astype(int)

        # nx and ny are the span of pixels covered in x/y directions
        nx = np.floor(np.max(px[take], axis=1)).astype(int) - left + 1
        ny = np.floor(np.max(py[take], axis=1)).astype(int) - bottom + 1

        # create cell grids
        ngy, ngx = ny.max(), nx.max()
        gy, gx = np.mgrid[:ngy, :ngx]
        gy, gx = gy.ravel(), gx.ravel()
        ng = gx.size

        # indices for raveled arrays
        inds = tuple(ind.ravel() for ind in np.indices((nshapes, nvert, ng)))

        # polygon vertices minus the lowest left pixel so we can
        # use gx, gy to perform faster vector operations.
        vx = px[take] - left[:, None]
        vy = py[take] - bottom[:, None]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ux, uy = shift1(vx), shift1(vy)
            dx, dy = vx - ux, vy - uy
            mx, my = dy / dx, dx / dy

            minx = np.min([ux, vx], axis=0)[..., None]
            maxx = np.max([ux, vx], axis=0)[..., None]
            miny = np.min([uy, vy], axis=0)[..., None]
            maxy = np.max([uy, vy], axis=0)[..., None]

            # y coordinates at x grid lines (left edge of cell)
            cross_left_y = gx[None, None] - ux[..., None]
            cross_left_y *= mx[..., None]
            cross_left_y += uy[..., None]

            # x coordinates at y grid lines (bottom edge of cell)
            cross_bottom_x = gy[None, None] - uy[..., None]
            cross_bottom_x *= my[..., None]
            cross_bottom_x += ux[..., None]

            parallel_x = (dy == 0)[..., None] \
                & (uy[..., None] == gy[None, None])
            if parallel_x.any():
                parallel_x &= (minx <= gx[None, None])
                parallel_x &= (gx[None, None] <= maxx)
                cross_bottom_x[parallel_x] = gx[inds[2][parallel_x.ravel()]]

            parallel_y = (dx == 0)[..., None] \
                & (ux[..., None] == gx[None, None])
            if parallel_y.any():
                parallel_y &= (miny <= gy[None, None])
                parallel_y &= (gy[None, None] <= maxy)
                cross_left_y[parallel_y] = gy[inds[2][parallel_y.ravel()]]

            # Lines crossing bottom of cell (u -> v)
            valid_b_cross = gy[None, None] >= miny
            valid_b_cross &= gy[None, None] < maxy
            valid_x_cross = cross_bottom_x >= gx[None, None]
            valid_x_cross &= cross_bottom_x < gx[None, None] + 1
            valid_b_cross &= valid_x_cross

            # Lines crossing left of cell (u -> v)
            valid_l_cross = gx[None, None] >= minx
            valid_l_cross &= gx[None, None] < maxx
            valid_y_cross = cross_left_y >= gy[None, None]
            valid_y_cross &= cross_left_y < gy[None, None] + 1
            valid_l_cross &= valid_y_cross

            corner = cross_bottom_x == gx[None, None]
            corner &= cross_left_y == gy[None, None]
            corner |= ((gx[None, None] == ux[..., None])
                       & (gy[None, None] == uy[..., None]))

        # valid_b_cross |= corner
        # valid_l_cross |= corner

        # Add any grid points inside polygon, not intersected by lines
        xlines = valid_b_cross | corner
        xlines = xlines.reshape(nshapes, nvert, ngy, ngx)
        grid_points = np.sum(xlines, axis=1)
        grid_points = np.roll(np.cumsum(grid_points, axis=2), 1, axis=2)
        grid_points %= 2
        grid_points[:, :, 0] = 0
        grid_points = grid_points.astype(bool).reshape(nshapes, ng)
        grid_points |= corner.any(axis=1)

        # Now all grid points (in or on the polygon) have been determined,
        # they should be distinguished from edges to avoid duplication.
        # Inside grid points cannot coincide with intersections, so we only
        # need to examine corners.
        valid_b_cross &= ~corner
        valid_l_cross &= ~corner

        # Finally, vertices located inside cell
        vertex_inside = vx[..., None] > gx[None, None]
        vertex_inside &= vx[..., None] < (gx[None, None] + 1)
        vertex_inside &= vy[..., None] > gy[None, None]
        vertex_inside &= vy[..., None] < (gy[None, None] + 1)

        # okay, so we now have everything we need:
        #   - edges (bottom, left)
        #   - inside points
        #   - grid points

        # populate
        counter = np.zeros((nshapes, ng), dtype=int)
        sout = nshapes, (nvert * 4), ng
        polx = np.full(sout, np.nan)  # maximum size
        poly = np.full(sout, np.nan)  # maximum size

        # populate inside vertices
        if vertex_inside.any():
            ri = vertex_inside.ravel()
            itake = inds[0][ri], inds[1][ri], inds[2][ri]
            n_inside = np.cumsum(vertex_inside, axis=1) - 1
            vput = counter[itake[0], itake[2]]
            vput += n_inside[itake]
            polx[itake[0], vput, itake[2]] = vx[itake[0], itake[1]]
            poly[itake[0], vput, itake[2]] = vy[itake[0], itake[1]]
            counter[itake[0], itake[2]] += n_inside[itake] + 1

        # Grid points are so far calculated as the bottom-left of a cell.
        # This needs to be shared by neighbors to the west, south, and
        # south-west.
        if grid_points.any():
            # ri = np.repeat(gp_inside[:, None], nvert, axis=1).ravel()
            # itake = inds[0][ri], inds[1][ri], inds[2][ri]
            for dpx, dpy in itertools.product([0, 1], [0, 1]):
                if dpx == dpy == 0:
                    valid = grid_points
                else:
                    valid = grid_points & (gx[None] >= dpx) & (gy[None] >= dpy)
                if not valid.any():  # pragma: no cover
                    continue

                idx = np.nonzero(valid)
                gp_ind = idx[1] - (dpy * ngx + dpx)
                vput = counter[idx[0], gp_ind]

                polx[idx[0], vput, gp_ind] = gx[idx[1]]
                poly[idx[0], vput, gp_ind] = gy[idx[1]]
                vput += 1
                counter[idx[0], gp_ind] = vput

        # Left edge crossings: shared by neighbor to left on it's right edge
        if valid_l_cross.any():
            for dpx in [0, 1]:
                if dpx == 1:
                    valid = valid_l_cross & (gx[None, None] >= dpx)
                else:
                    valid = valid_l_cross
                ncross = valid.cumsum(axis=1) - 1
                ri = valid.ravel()
                itake = inds[0][ri], inds[1][ri], inds[2][ri]
                gp_ind = itake[2] - dpx
                vput = counter[itake[0], gp_ind] + ncross[itake]
                polx[itake[0], vput, gp_ind] = gx[itake[2]]
                poly[itake[0], vput, gp_ind] = cross_left_y[itake]
                vput += 1
                counter[itake[0], gp_ind] = vput

        # Bottom edge crossings: shared by neighbor below on it's top edge
        if valid_b_cross.any():
            for dpy in [0, 1]:
                if dpy == 1:
                    valid = valid_b_cross & (gy[None, None] >= dpy)
                else:
                    valid = valid_b_cross
                ncross = valid.cumsum(axis=1) - 1
                ri = valid.ravel()
                itake = inds[0][ri], inds[1][ri], inds[2][ri]
                gp_ind = itake[2] - (dpy * ngx)
                vput = counter[itake[0], gp_ind] + ncross[itake]
                polx[itake[0], vput, gp_ind] = cross_bottom_x[itake]
                poly[itake[0], vput, gp_ind] = gy[itake[2]]
                vput += 1
                counter[itake[0], gp_ind] = vput

        # print("populate: %f" % (t4 - t3))

        # Trim down the array as necessary and move coordinates off
        # the shared grid
        maxv = counter.max()
        polx, poly = polx[:, :maxv], poly[:, :maxv]
        polx += left[:, None, None]
        poly += bottom[:, None, None]
        gxout = left[..., None] + gx[None]
        gyout = bottom[..., None] + gy[None]

        keep = np.isfinite(polx)
        if xrange is not None:
            keep = np.logical_and(
                keep, np.greater_equal(gxout[:, None], xrange[0]), out=keep)
            keep = np.logical_and(
                keep, np.less(gxout[:, None], xrange[1]), out=keep)
        if yrange is not None:
            keep = np.logical_and(
                keep, np.greater_equal(gyout[:, None], yrange[0]), out=keep)
            keep = np.logical_and(
                keep, np.less(gyout[:, None], yrange[1]), out=keep)

        # print("normalize: %f" % (t5 - t4))

        # note that COM needs to be done before filling in NaNs
        # We also do this to kill any bad values (usually repeated), that
        # managed to find there way to this stage.
        comx = bottleneck.nanmean(polx, axis=1)
        comy = bottleneck.nanmean(poly, axis=1)
        polx = bottleneck.push(polx, axis=1)
        poly = bottleneck.push(poly, axis=1)
        np.subtract(polx, comx[:, None], out=polx)
        np.subtract(poly, comy[:, None], out=poly)
        angle = np.arctan2(poly, polx)
        sorti = np.argsort(angle, axis=1)
        og = np.ogrid[:nshapes, :maxv, :ng]
        polx = polx[og[0], sorti, og[2]]
        poly = poly[og[0], sorti, og[2]]

        pixareas = (0.5 * np.abs(bottleneck.nansum(
            (polx * np.roll(poly, -1, axis=1))
            - (poly * np.roll(polx, -1, axis=1)), axis=1)))

        keep &= pixareas[:, None] != 0

        # print("areas: %f, %f" % (t6 - t5, t6 - t1))

        mask = np.any(keep, axis=1)
        npixels = mask.sum(axis=1)
        minpix, maxpix = np.min(npixels), np.max(npixels) + 1
        npixbins = maxpix - minpix
        pixbins = (npixels - minpix).astype(int)
        pixind = np.arange(npixels.size)
        spix = csr_matrix((pixind, [pixbins, np.arange(npixels.size)]),
                          shape=(npixbins, npixels.size))

        for pixi, putpix in enumerate(np.split(spix.data, spix.indptr[1:-1])):
            npix = pixi + minpix
            if npix == 0 or len(putpix) == 0:  # pragma: no cover
                continue
            npolys = putpix.size
            takepix = mask[putpix]
            cellx = np.reshape(gxout[putpix][takepix], (npolys, npix))
            celly = np.reshape(gyout[putpix][takepix], (npolys, npix))
            cellxy = np.append(celly[:, :, None], cellx[:, :, None], axis=2)
            # this gives the cells overlapped by each polygon
            for polyind, cxy in zip(putpix, cellxy):
                result[put[polyind]] = cxy

            if area:
                aselect = np.reshape(pixareas[putpix][takepix], (npolys, npix))
                for polyind, pixarea in zip(putpix, aselect):
                    areas[put[polyind]] = pixarea

        # print("storing results: %f, %f" % (t7 - t6, t7 - t1))

    if single:
        if len(result) != 0:
            result = result[0]
            if area:
                areas = areas[0]
        else:
            result = np.empty((0, 2))
            if area:
                areas = np.empty(0)

    if not area:
        return result
    else:
        return result, areas


def polygon_area(ppath):  # pragma: no cover
    """
    Uses the shoelace method to calculate area of a polygon

    Goes as fast as I can, handling rounding errors as best I can.

    Parameters
    ----------
    ppath : matplotlib.path.Path

    Returns
    -------
    float
        Area of the polygon
    """
    v_ = ppath.vertices
    if len(v_) < 3:
        return 0.0
    x_ = v_[:, 1] - v_[:, 1].mean()
    y_ = v_[:, 0] - v_[:, 0].mean()
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def polygon_weights(polygon, xrange=None, yrange=None,
                    center=True):  # pragma: no cover
    """
    Get pixel weights - depreciated by polyfillaa

    Parameters
    ----------
    polygon : array_like of float
        (N, 2) where the last dimension is in the numpy (y, x) format,
        i.e. polygon[:, 0] = y coordinates, polygon[:, 1] = x coordinates
        Each point specifies a vertex of the polygon.  Must contain
        at least 3 vertices.
    xrange : array_like of float, optional
        (2,) Specifies the (minimum, maximum) allowable x values
    yrange : array_like of float, optional
        (2,) Specifies the (minimum, maximum) allowable y values
    center : bool, optional
        If True, integer (y,x) values define pixel centers, otherwise
        they define the lower-left corner of each pixel.

    Returns
    -------
    list of tuple
        Tuple of the form ((y, x), area) where y and x are the integer
        pixel locations and area are float fractions of pixel area
        within the polygon (0 -> 1).
    """
    poly = np.array(polygon)
    if poly.ndim != 2 or poly.shape[-1] != 2 or poly.shape[0] < 3:
        log.warning("invalid polygon shape")
        return []

    xlims = [poly[:, 1].min(), poly[:, 1].max()]
    ylims = [poly[:, 0].min(), poly[:, 0].max()]

    if xrange is not None:
        xlims[0] = np.nanmax((xlims[0], np.nanmin(xrange)))
        xlims[1] = np.nanmin((xlims[1], np.nanmax(xrange)))
    if yrange is not None:
        ylims[0] = np.nanmax((ylims[0], np.nanmin(yrange)))
        ylims[1] = np.nanmin((ylims[1], np.nanmax(yrange)))

    if xlims[0] >= xlims[1] or ylims[0] >= ylims[1]:
        log.debug("out of bounds")
        return []

    xlims = [int(np.floor(xlims[0])), int(np.ceil(xlims[1]))]
    ylims = [int(np.floor(ylims[0])), int(np.ceil(ylims[1]))]

    if center:
        dx = -0.5, 0.5
        dy = -0.5, 0.5
    else:
        dx = 0, 1
        dy = 0, 1

    gy, gx = np.mgrid[ylims[0]:ylims[1] + 1, xlims[0]:xlims[1] + 1]
    p = path.Path(poly)
    result = []
    for ycen, xcen in zip(gy.ravel(), gx.ravel()):
        bbox = Bbox([[ycen + dy[0], xcen + dx[0]],
                     [ycen + dy[1], xcen + dy[1]]])
        area = polygon_area(p.clip_to_bbox(bbox))
        if area != 0:
            result.append(((ycen, xcen), area))

    return result
