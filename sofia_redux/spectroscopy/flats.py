# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import inspect
import numpy as np
from sofia_redux.toolkit.utilities.fits import gethdul
from sofia_redux.toolkit.fitting.polynomial import poly1d
from sofia_redux.toolkit.image.adjust import rotate90

__all__ = ['FlatBase', 'FlatInfo', 'Flat']


class FlatBase:

    def __init__(self, filename):
        self.shape = None
        self._filename = None
        self._header = None
        self._data = None
        self._ishell = None
        self.guesspos = None
        self.omask = None
        self.orders = None
        self.norders = None
        self.edgedeg = None
        self.edgecoeffs = None
        self.xranges = None
        self.rotation = None
        self.ybuffer = None
        self._default_xranges = None
        self._default_guesspos = None
        self.filename = str(filename)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        value = value.strip()
        self._load_file(value)

    def __repr__(self):
        attributes = [x for x in dir(self) if not x.startswith('_')]
        s = ''
        for attrib in attributes:
            a = getattr(self, attrib)
            if not inspect.ismethod(a):
                if isinstance(a, np.ndarray):
                    a = '%s %s of %s' % (np.ndarray, a.shape, a.dtype)
                s += '%s: %s\n' % (attrib, repr(a))
        return s

    def _set_order_info(self):
        orders = np.array(
            [int(x) for x in self._header['ORDERS'].split(',')])
        norders = len(orders)
        edgedeg = int(self._header['EDGEDEG'])

        xranges = np.zeros((norders, 2), dtype=int)
        edgecoeffs = np.full((norders, 2, edgedeg + 1), np.nan)

        test = 'ODR%s_T*' % str(orders[0]).zfill(2)
        self._ishell = len(self._header.get(test)) == 0
        prefix, nz = ('OR', 3) if self._ishell else ('ODR', 2)

        for i, order in enumerate(orders):
            ostr = prefix + str(order).zfill(nz)
            bname = '%s_B*' % ostr
            tname = '%s_T*' % ostr
            xname = '%s_XR' % ostr
            edgecoeffs[i, 0] = list(self._header[bname].values())
            edgecoeffs[i, 1] = list(self._header[tname].values())
            xranges[i] = [int(x) for x in self._header[xname].split(',')]

        self.orders = orders
        self.norders = norders
        self.edgedeg = edgedeg
        self.edgecoeffs = edgecoeffs
        self.xranges = xranges
        self._default_xranges = xranges.copy()
        self.default_guess_position()

    def _load_file(self, filename):
        hdul = gethdul(filename, verbose=True)
        if hdul is None:
            raise ValueError("Could not load: %s" % filename)
        self._header = hdul[0].header.copy()
        if hdul[0].data is not None:
            self._data = hdul[0].data.copy()
        else:
            self._data = None
        self._filename = filename
        self._set_order_info()
        self.parse_info()

    def parse_info(self):  # pragma: no cover
        pass

    def generate_order_mask(self, offset=0):
        """
        Generate an order mask based on xranges and edge coefficients

        Notes
        -----
        No rotation is applied if we're generating the order mask.  The
        parameters required for order mask generation are already correct.

        Parameters
        ----------
        offset : int, optional
            If provided, will be added to the x coordinate before
            generating the edge coefficient polynomials.  This is intended
            to allow reusing edge coordinates for a shifted array in
            the x-direction.
        """
        nrow, ncol = self.shape
        yy, xx = np.mgrid[:nrow, :ncol]
        fillmask = np.empty(self.shape, dtype=bool)
        omask = np.zeros(self.shape, dtype=int)
        for i, order in enumerate(self.orders):
            fillmask[...] = False
            x = np.arange(self.xranges[i, 0], self.xranges[i, 1] + 1) + offset
            botedge = poly1d(x, self.edgecoeffs[i, 0])
            topedge = poly1d(x, self.edgecoeffs[i, 1])
            column_ok = topedge <= nrow - 0.5
            column_ok &= botedge >= -0.5
            botedge = np.clip(np.floor(botedge), 0, nrow).astype(int)
            topedge = np.clip(np.ceil(topedge), 0, nrow).astype(int)
            xl, xu = self.xranges[i]
            fillmask[:, xl:xu + 1] = ((yy[:, xl:xu + 1] <= topedge)
                                      & (yy[:, xl:xu + 1] >= botedge)
                                      & column_ok[None])
            omask[fillmask] = order
        return omask

    @staticmethod
    def create_order_mask(shape, orders, edgecoeffs, xranges):
        """
        Create an ordermask

        Parameters
        ----------
        shape : 2-tuple
            (nrow, ncol) shape of the ordermask
        orders : array_like of int
            (norders,) array of orders
        edgecoeffs : array_like of float
            (norders, 2)
        xranges : array_like of int
            (norders, 2)

        Returns
        -------
        ordermask : numpy.ndarray
            (nrow, ncol) of int
        """
        nrow, ncol = shape
        yy, xx = np.mgrid[:nrow, :ncol]
        fillmask = np.empty(shape, dtype=bool)
        omask = np.zeros(shape, dtype=int)
        for i, order in enumerate(orders):
            fillmask[...] = False
            x = np.arange(xranges[i, 0], xranges[i, 1] + 1)
            botedge = poly1d(x, edgecoeffs[i, 0])
            topedge = poly1d(x, edgecoeffs[i, 1])
            column_ok = topedge <= nrow - 0.5
            column_ok &= botedge >= -0.5
            botedge = np.clip(np.floor(botedge), 0, nrow).astype(int)
            topedge = np.clip(np.ceil(topedge), 0, nrow).astype(int)
            xl, xu = xranges[i]
            fillmask[:, xl:xu + 1] = ((yy[:, xl:xu + 1] <= topedge)
                                      & (yy[:, xl:xu + 1] >= botedge)
                                      & column_ok[None])
            omask[fillmask] = order
        return omask

    def default_guess_position(self):
        """Sets default guess positions for each order

        Notes
        -----
        xranges is also reset to default values
        """
        self.xranges = self._default_xranges.copy()
        guesspos = np.zeros((self.norders, 2), dtype=int)
        for i in range(self.norders):
            guesspos[i, 0] = self.xranges[i].mean()
            botedge = poly1d(guesspos[i, 0], self.edgecoeffs[i, 0])
            topedge = poly1d(guesspos[i, 0], self.edgecoeffs[i, 1])
            guesspos[i, 1] = (topedge + botedge) / 2
        self.guesspos = guesspos
        self._default_guesspos = guesspos.copy()

    def adjust_guess_position(self, image, order, ybuffer=3):
        """
        Updates guess positions and xranges via image correlation

        The correlation occurs only in the spatial dimension (y).

        Parameters
        ----------
        image : array_like of float
            (nrow, ncol) Image with which to perform the correlation.
        order : int
            The order from the ordermask with which to perform the
            correlation.
        ybuffer : int, optional
            The number of pixels to buffer form the top and bottom
            of the array.
        """
        image = np.asarray(image, dtype=float)

        omask = np.equal(self.omask, order)
        if not omask.any():
            raise ValueError("Order %i not present in ordermask" % order)
        oi = np.argwhere(self.orders == order)
        if oi.shape[0] == 0:
            raise ValueError("Order %i not present in orders" % order)
        oi = oi[0, 0]

        x = np.arange(self.xranges[oi, 0], self.xranges[oi, 1] + 1)
        botedge = poly1d(x, self.edgecoeffs[oi, 0])
        topedge = poly1d(x, self.edgecoeffs[oi, 1])
        slith_pix = int(np.ceil(max(topedge - botedge)))

        # Determine the top and bottom row of the subimage to clip out
        topidx = int(np.round(topedge).max()) + slith_pix + 1
        botidx = int(np.round(botedge).min()) - slith_pix
        topidx = np.clip(topidx, 0, self.shape[0])
        botidx = np.clip(botidx, 0, self.shape[0])

        subimg = image[botidx:topidx]
        subomask = omask[botidx:topidx]

        # shifts are in the y-direction only
        subshape = subimg.shape
        nshifts = slith_pix * 2 + 1
        shifts = np.arange(nshifts) - slith_pix
        cross_corr = np.zeros(nshifts)

        for i, s in enumerate(shifts):
            hbot = np.clip(-s, 0, subshape[0])
            htop = np.clip(subshape[0] - s, 0, subshape[0])
            mbot = np.clip(s, 0, subshape[0])
            mtop = np.clip(subshape[0] + s, 0, subshape[0])
            cross_corr[i] = np.sum(subimg[hbot:htop] * subomask[mbot:mtop])
        offset = shifts[np.argmax(cross_corr)]

        self.guesspos = self._default_guesspos.copy()
        self.guesspos[:, 1] -= offset

        # If we shift by this amount then xranges may need to be updated
        self.xranges = self._default_xranges.copy()
        dy = ybuffer - 1
        for oi, order in enumerate(self.orders):
            xrange = self.xranges[oi]
            x = np.arange(xrange[1] - xrange[0] + 1) + xrange[0]
            botedge = poly1d(x, self.edgecoeffs[oi, 0]) - offset
            topedge = poly1d(x, self.edgecoeffs[oi, 1]) - offset
            idx = (botedge > dy) & (topedge < (image.shape[0] - dy))
            if not idx.any():
                log.warning("Order %i shifted past y-buffer" % order)
                xrange[:] = -1, -1
            else:
                xrange[:] = np.min(x[idx]), np.max(x[idx])


class FlatInfo(FlatBase):

    def __init__(self, filename):
        super().__init__(filename)
        self.modename = None
        self.slith_arc = None
        self.slith_pix = None
        self.slith_range = None
        self.rppix = None
        self.ps = None
        self.fixed = None
        self.step = None
        self.flatfrac = None
        self.comwin = None
        self.norm_nxg = None
        self.norm_nyg = None
        self.oversamp = None
        self.ycororder = None

    def parse_info(self):
        self.omask = self._data.astype(int)
        self.shape = self.omask.shape
        h = self._header
        # Standard keywords without logic (name, [rename], type)
        get = [
            ('modename', str),
            ('rotation', int),  # IDL ROTATE value
            ('slth_arc', 'slith_arc', float),  # Slit length in arcseconds
            ('slth_pix', 'slith_pix', float),  # Slit length in pixels
            ('rppix', float),  # Resolving power per pixel
            ('pltscale', 'ps', float),  # Plate scale (arcseconds per pixel)
            ('fixed', bool),
            ('step', int),  # Edge find step size
            ('flatfrac', float),  # Edge find fraction
            ('comwin', int),  # Center-Of-Mass window
            ('norm_nxg', int),  # fiterpolate grid definition
            ('norm_nyg', int),  # fiterpolate grid definition
            ('oversamp', float),  # normspecflat (not actually used)
            ('ybuffer', int),  # buffer pixels (rows)
            ('ycorordr', 'ycororder', int)]

        defaults = {str: '', float: 0.0, int: 0, bool: False}
        for x in get:
            name = x[0] if len(x) == 2 else x[1]
            dtype = type(defaults[x[-1]])
            value = dtype(h.get(x[0], defaults[dtype]))
            setattr(self, name, value)

        if self.rotation is not None:
            # When reading the ordermask from the flatinfo file, rotation
            # should be applied.
            self.omask = rotate90(self.omask, self.rotation)

        # Begin extracting those that require a little parsing
        self.slith_range = np.array(
            [int(x) for x in h.get('SLTH_RNG', '').split(',')])


class Flat(FlatBase):

    def __init__(self, filename):
        self.image = None
        self.variance = None
        self.flags = None
        self.modename = None
        self.slith_arc = None
        self.slith_pix = None
        self.slitw_arc = None
        self.slitw_pix = None
        self.rp = None
        self.ps = None
        self.rms = None
        super().__init__(filename)

    def parse_info(self):
        # do not copy data
        if self._data.ndim == 3:
            ndata = self._data.shape[0]
            self.image = self._data[0]
            self.shape = self.image.shape
            self.variance = self._data[1] if ndata > 1 else None
            self.flags = self._data[2] if ndata > 2 else None
        else:
            self.image = self._data
            self.shape = self.image.shape
            self.variance = self.flags = None

        # Standard keywords without logic (name, [rename], type)
        get = [
            ('modename', str),
            ('rotation', int),  # IDL ROTATE value
            ('slth_arc', 'slith_arc', float),  # Slit length in arcseconds
            ('slth_pix', 'slith_pix', float),  # Slit length in pixels
            ('sltw_arc', 'slitw_arc', float),  # Slit length in arcseconds
            ('sltw_pix', 'slitw_pix', float),  # Slit length in pixels
            ('rp', float),  # Resolving power
            ('pltscale', 'ps', float)]  # Plate scale (arcseconds per pixel)]

        h = self._header
        defaults = {str: '', float: 0.0, int: 0, bool: False}
        for x in get:
            name = x[0] if len(x) == 2 else x[1]
            dtype = type(defaults[x[-1]])
            value = dtype(h.get(x[0], defaults[dtype]))
            setattr(self, name, value)

        prefix = 'OR' if self._ishell else 'ODR'
        self.rms = np.full(self.norders, np.nan)
        for i, order in enumerate(self.orders):
            self.rms[i] = h.get('%s%i' % (prefix, order), np.nan)
        self.omask = self.create_order_mask(
            self.shape, self.orders, self.edgecoeffs, self.xranges)
