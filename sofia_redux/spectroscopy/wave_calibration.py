# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pandas
from sofia_redux.toolkit.utilities.fits import gethdul
from sofia_redux.toolkit.utilities.func import goodfile
from scipy.interpolate import interp1d

__all__ = ['BaseWavecal', 'Wavecal', 'Wavecal1D', 'Wavecal1DXD',
           'Wavecal2D', 'Wavecal2DXD', 'readwavecalinfo', 'LineInfo']


class BaseWavecal:

    registry = {}

    @classmethod
    def register(cls, caltype):
        def decorator(subclass):
            cls.registry[caltype] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, caltype, data, header):
        if caltype not in cls.registry:
            raise ValueError("Invalid wave calibration type: %s" % caltype)
        return cls.registry[caltype](data, header)


class Wavecal(BaseWavecal):

    def __init__(self, data, header):
        """
        Initializes basic class

        Parameters
        ----------
        data : numpy.ndarray of float
        header : astropy.io.fits.header.Header
        """
        self.data = None
        self.header = None
        self.norders = None
        self.naps = None
        self.orders = None
        self.xranges = None
        self.extap = None
        self.linelist = None
        self.xcorordr = None
        self.xcorspec = None
        self.dispdeg = None
        self.p2wcoeffs = None
        self.wcaltype = None
        self.lines = LineInfo()
        self.xd = None
        self._load_data(data, header)
        self._get_spectrum()

    def _load_data(self, data, header):
        """
        Loads common attributes common to all classes

        Parameters
        ----------
        data : numpy.ndarray of float
        header : astropy.io.fits.header.Header
        """
        naps = int(header['NAPS'])
        norders = int(header['NORDERS'])
        orders = np.asarray(
            [int(x) for x in str(header['ORDERS']).split(',')])
        # Get extraction ranges
        xranges = np.zeros((norders, 2), dtype=np.int64)
        for orderi, order in enumerate(orders[:norders]):
            name = 'OR%s_XR' % str(order).zfill(3)
            xranges[orderi] = [
                int(xr) for xr in header[name].split(',')]
        wrange = np.array(
            [np.nanmin(data[:, 0]), np.nanmax(data[:, 0])])
        extap = float(header['extap'])
        self.data = data.copy()
        self.header = header.copy()
        self.naps = naps
        self.norders = norders
        self.orders = orders
        self.xranges = xranges
        self.wrange = wrange
        self.extap = extap
        self.dispdeg = int(self.header['DISPDEG'])
        ncoeffs = (self.dispdeg + 1)
        p2wcoeffs = np.empty(ncoeffs)
        for i in range(ncoeffs):
            name = 'P2W_C%s' % str(i).zfill(2)
            p2wcoeffs[i] = self.header[name]
        self.p2wcoeffs = p2wcoeffs
        self.wcaltype = header['WCALTYPE'].strip().upper()
        self.linelist = header['LINELIST'].strip()

    def _get_spectrum(self):
        """Get the spectrum in relation to the x-correlation order"""
        self.xcorordr = int(self.header['XCORORDR'])
        xcor_idx = self.orders == self.xcorordr
        if not xcor_idx.any():
            raise ValueError('X-correlation order %i not found in orders' %
                             self.xcorordr)
        xcor_idx = np.argwhere(xcor_idx)[0, 0]
        xcorspec = self.data[xcor_idx].copy()
        npix = int(np.ptp(self.xranges[xcor_idx])) + 1
        xcorspec[0] = np.arange(npix, dtype=float) + self.xranges[xcor_idx, 0]
        self.xcorspec = xcorspec

    def load_lines(self, filename):
        self.lines.readlinelist(filename)

    def guess_lines(self):
        if self.lines.table is None:
            raise ValueError("Line list not loaded")
        self.lines.xguess(self.data[:, 0], self.xranges, self.orders)


@BaseWavecal.register('1D')
class Wavecal1D(Wavecal):
    def __init__(self, data, header):
        super().__init__(data, header)
        self.rms = header['FITRMS']


@BaseWavecal.register('1DXD')
class Wavecal1DXD(Wavecal):
    def __init__(self, data, header):
        super().__init__(data, header)


@BaseWavecal.register('2D')
class Wavecal2D(Wavecal):
    def __init__(self, data, header):
        super().__init__(data, header)
        self.rms = header['FITRMS']
        self.linedeg = int(header['LINEDEG'])
        self.fndystep = int(header['FNDYSTEP'])
        self.fndysum = int(header['FNDYSUM'])
        self.genystep = int(header['GENYSTEP'])
        self.c1xdeg = int(header['C1XDEG'])
        if self.linedeg == 2:
            self.c2xdeg = int(header['C2XDEG'])


@BaseWavecal.register('2DXD')
class Wavecal2DXD(Wavecal):
    def __init__(self, data, header):
        super().__init__(data, header)
        self.linedeg = int(header['LINEDEG'])
        self.fndystep = int(header['FNDYSTEP'])
        self.fndysum = int(header['FNDYSUM'])
        self.genystep = int(header['GENYSTEP'])
        self.c1xdeg = int(header['C1XDEG'])
        self.c1ydeg = int(header['C1XDEG'])
        if self.linedeg == 2:
            self.c2xdeg = int(header['C2XDEG'])
            self.c2ydeg = int(header['C2YDEG'])


def readwavecalinfo(filename):
    """
    Retrieve wave calibration

    Parameters
    ----------
    filename : str

    Returns
    -------
    Wavecal
        A subclass of `Wavecal` depending on the data contents.
        one of {`Wavecal1D`, `Wavecal1DXD`, `Wavecal2D`, `Wavecal2DXD`}.
    """
    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        raise ValueError("Could not load file: %s" % filename)
    data = hdul[0].data
    if data is None:
        raise ValueError("No data present in primary HDU of %s" % filename)
    header = hdul[0].header
    wcaltype = header['WCALTYPE'].strip().upper()
    return BaseWavecal.create(wcaltype, data, header)


class LineInfo:

    def __init__(self):
        self.table = None
        self.nlines = None
        self._offset = 0.0

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        xcolumns = ['xguess']
        if self.table is None or 'xwindow' not in self.table:
            return
        value = float(value)
        previous = self._offset
        if previous != 0 or value != 0:
            for column in xcolumns:
                if column in self.table:
                    self.table[column] = self.table[column].apply(
                        lambda x: x - previous + value)
            self._offset = value

    def readlinelist(self, filename):
        """
        Read a Spextool line list

        Parameters
        ----------
        filename : str
            A Spextool line nlist with | delimited columns of the order
            number, wavelengths, ids, fit windows, fit types, and number
            of fit terms.

        Returns
        -------
        pandas.DataFrame
            Columns are:
                order : int
                    Order number
                wavelength : float
                    Wavelength in microns
                species : str
                    Line IDs
                window : float
                    Fit window in microns
                type : str
                    Fitting model. 'G' = Gaussian, 'L' = Lorentzian
                nterms : int
                    Number of terms in the fit.
                    3=basic, 4=basic+constant, 5=basic+line
        """
        if not goodfile(filename, verbose=True):
            raise ValueError("Unable to load: %s" % filename)
        self.table = pandas.read_csv(
            filename, comment='#', delimiter='|',
            names=['order', 'wavelength', 'species',
                   'window', 'type', 'nterms'],
            converters={'order': int,
                        'wavelength': float,
                        'species': str.strip,
                        'window': lambda w: float(w) / 1e4,
                        'type': str.strip,
                        'nterms': int})

    def xguess(self, wavelength, xranges, orders):
        """
        Determine guess positions for lines of a given order

        Updates the table with:
            xguess -> x positions of lines in xspec (float)
            xwindow -> The fit window in units of pixels (int)

        Parameters
        ----------
        wavelength : numpy.ndarray
            (norders, nwave) of float array of wavelengths in microns for
            each order.
        xranges : numpy.ndarray
            (norders, 2) of int array of column numbers where the orders
            are completely on the array.  xranges[:, 0] = lower limit,
            and xranges[:, 1] = upper limit.
        orders : numpy.ndarray
            (norders,) array of int giving the order numbers.
        """
        if self.table is None:
            raise ValueError('Line list not loaded.')

        self.table['xguess'] = np.nan
        self.table['xwindow'] = 0

        if orders.shape[0] != wavelength.shape[0]:
            raise ValueError("Orders and wavelength axis 0 size mismatch")
        if orders.shape[0] != xranges.shape[0]:
            raise ValueError("Orders and xranges axis 0 size mismatch")
        xx = np.arange(wavelength.shape[1])

        for wave, xrange, order in zip(wavelength, xranges, orders):
            mask = np.isfinite(wave)
            w, x = wave[mask], xx[mask] + xrange[0]
            fx = interp1d(w, x, kind='linear', fill_value='extrapolate')
            idx = self.table.order == order
            xout = self.table[idx].wavelength.values
            dw = self.table[idx].window.values / 2
            self.table.loc[idx, 'xguess'] = fx(xout)
            self.table.loc[idx, 'xwindow'] = np.round(
                np.ptp(fx([xout - dw, xout + dw]), axis=0)).astype(int)
