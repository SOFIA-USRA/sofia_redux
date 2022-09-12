# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from abc import ABC
from astropy import log, units
from astropy.modeling import functional_models
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from copy import deepcopy
import numpy as np
from sofia_redux.scan.utilities.utils import (
    get_header_quantity, to_header_float)

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D

__all__ = ['Gaussian1D']


ud = units.dimensionless_unscaled


class Gaussian1D(ABC):

    fwhm_to_size = np.sqrt(2 * np.pi) * gaussian_fwhm_to_sigma

    def __init__(self, peak=1.0, mean=0.0, fwhm=0.0,
                 peak_unit=None, position_unit=None):
        """
        Initializes a 2-D Gaussian beam model.

        The Gaussian2D class is a wrapper around the
        :class:`functional_models.Gaussian2D` class with additional
        functionality including convolution/deconvolution with another beam,
        and header parsing/editing.

        Parameters
        ----------
        peak : float or units.Quantity, optional
            The peak amplitude of the Gaussian.
        mean : float or units.Quantity, optional
            The position of the peak along the x-axis.
        fwhm : float or units.Quantity, optional
            The Full-Width-Half-Max beam width in the x-direction.
        peak_unit : units.Unit or units.Quantity or str, optional
            The physical units for the peak amplitude.  The default is
            dimensionless.
        position_unit : units.Unit or units.Quantity or str, optional
            The physical units of all position based parameters (`fwhm`,
            `mean`).
        """
        if position_unit is None:
            position_unit = units.dimensionless_unscaled
        else:
            position_unit = units.Unit(position_unit)
        if peak_unit is None:
            if isinstance(peak, units.Quantity):
                peak_unit = peak.unit
            else:
                peak_unit = units.dimensionless_unscaled
        else:
            peak_unit = units.Unit(peak_unit)

        if not isinstance(fwhm, units.Quantity):
            fwhm = fwhm * position_unit
        if not isinstance(mean, units.Quantity):
            mean = mean * position_unit

        if not isinstance(peak, units.Quantity):
            peak = peak * peak_unit

        self.unit = peak_unit
        self.model = functional_models.Gaussian1D(
            stddev=fwhm * gaussian_fwhm_to_sigma,
            mean=mean, amplitude=peak.value)
        self.validate()

    def copy(self):
        """
        Return a copy of the Gaussian 2D model.

        Returns
        -------
        Gaussian1D
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            new = self.__class__()
            for key, value in self.__dict__.items():
                if key in self.referenced_attributes:  # pragma: no cover
                    setattr(new, key, value)
                elif hasattr(value, 'copy'):
                    setattr(new, key, value.copy())
                else:
                    setattr(new, key, deepcopy(value))
        return new

    @property
    def referenced_attributes(self):
        """
        Return the names of attributes to be referenced rather than copied.

        Returns
        -------
        set of str
        """
        return set([])

    @property
    def fwhm(self):
        """
        Return the FWHM.

        Returns
        -------
        units.Quantity
        """
        return self.model.stddev * gaussian_sigma_to_fwhm

    @fwhm.setter
    def fwhm(self, value):
        """
        Set the FWHM.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if not isinstance(value, units.Quantity):
            if self.position_unit is None:  # pragma: no cover
                value = value * units.dimensionless_unscaled
            else:
                value = value * self.position_unit
        self.model.stddev = gaussian_fwhm_to_sigma * value
        self.validate()

    @property
    def stddev(self):
        """
        Return the standard deviation.

        Returns
        -------
        value : units.Quantity or float
        """
        return self.model.stddev

    @stddev.setter
    def stddev(self, value):
        """
        Set the standard deviation.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        if not isinstance(value, units.Quantity):
            value = value * self.position_unit
        self.model.stddev = value
        self.validate()

    @property
    def mean(self):
        """
        Return the center of the Gaussian.

        Returns
        -------
        units.Quantity or float
        """
        value = self.model.mean.value
        unit = self.model.mean.unit
        if unit is None:  # pragma: no cover
            return value
        else:
            return value * unit

    @mean.setter
    def mean(self, value):
        """
        Set the value of the center of the Gaussian in.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        if not isinstance(value, units.Quantity):
            self.model.mean = value * self.position_unit
        else:
            self.model.mean = value

    @property
    def peak(self):
        """
        Return the peak amplitude of the Gaussian.

        Note that the result is returned as a float, not a Quantity.

        Returns
        -------
        float
        """
        return self.model.amplitude * 1.0

    @peak.setter
    def peak(self, value):
        """
        Set the peak amplitude of the Gaussian.

        Parameters
        ----------
        value : float or units.Quantity

        Returns
        -------
        None
        """
        if isinstance(value, units.Quantity):
            value = value.to(self.unit).value
        self.model.amplitude = value

    @property
    def position_unit(self):
        """
        Return the units of the positional coordinates.

        Returns
        -------
        units.Unit
        """
        return self.stddev.unit

    def __str__(self):
        """
        Return string information on the Gaussian2D model.

        Returns
        -------
        str
        """
        if self.unit in [None, ud]:
            peak_str = f'{self.peak}'
        else:
            peak_str = f'{self.peak} {self.unit}'

        return f"fwhm={self.fwhm}, mean={self.mean}, peak={peak_str}"

    def __repr__(self):
        """
        Return a string representation of the Gaussian2D model.

        Returns
        -------
        str
        """
        return f'{object.__repr__(self)} {str(self)}'

    def __eq__(self, other):
        """
        Test if this Gaussian1D instance is functionally equivalent to another.

        Parameters
        ----------
        other : Gaussian1D

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        try:
            if not np.isclose(self.fwhm, other.fwhm):
                return False
            if not np.isclose(self.mean, other.mean):
                return False
        except (TypeError, units.UnitConversionError):  # pragma: no cover
            # Just in case these managed to get set strangely (hard to do)
            return False
        return (self.peak * self.unit) == (other.peak * other.unit)

    def validate(self):
        """
        Validates the 1-D Gaussian parameters.

        Returns
        -------
        None
        """
        pass

    def combine_with(self, psf, deconvolve=False):
        """
        Combine with another beam.

        Combination consists of either convolution (default) or deconvolution
        of this beam by another.

        Parameters
        ----------
        psf : Gaussian1D or None
            The beam to combine.
        deconvolve : bool, optional
            If `True`, indicates a deconvolution rather than a convolution.

        Returns
        -------
        None
        """
        if psf is None:
            return

        a2 = self.fwhm ** 2
        b2 = psf.fwhm ** 2
        if deconvolve:
            d2 = a2 - b2
        else:
            d2 = a2 + b2
        if d2 < 0:
            d2 *= 0
        d = np.sqrt(d2)
        self.fwhm = d

    def convolve_with(self, psf):
        """
        Convolve with a given PSF.

        Parameters
        ----------
        psf : Gaussian1D
           The Point Spread Function to convolve with.

        Returns
        -------
        None
        """
        self.combine_with(psf, deconvolve=False)

    def deconvolve_with(self, psf):
        """
        Deconvolve with a given PSF.

        Parameters
        ----------
        psf : Gaussian1D
           The Point Spread Function to deconvolve with.

        Returns
        -------
        None
        """
        self.combine_with(psf, deconvolve=True)

    def encompass(self, psf):
        """
        Encompass with another beam.

        Parameters
        ----------
        psf : Gaussian1D or units.Quantity
            Another Gaussian PSF or the FWHM of another Gaussian PSF.

        Returns
        -------
        None
        """
        if isinstance(psf, Gaussian1D):
            fwhm = psf.fwhm
        else:
            fwhm = psf

        if self.fwhm < fwhm:
            self.fwhm = fwhm

    def is_encompassing(self, psf):
        """
        Check if this psf is encompassing another.

        Parameters
        ----------
        psf : Gaussian1D or units.Quantity

        Returns
        -------
        bool
        """
        fwhm = psf.fwhm if isinstance(psf, Gaussian1D) else psf
        return fwhm <= self.fwhm

    def scale(self, factor):
        """
        Scale the FWHM by a given amount.

        Parameters
        ----------
        factor : float

        Returns
        -------
        None
        """
        self.model.stddev *= factor

    def extent(self):
        """
        Return the extent in x and y.

        The extent is the maximum (x, y) deviation away from the center of the
        psf in the native coordinate frame, accounting for rotation by the
        position angle.

        Returns
        -------
        extent : Coordinate1D
            The extent of the FWHM
        """
        return Coordinate1D(self.fwhm)

    def get_integral(self):
        """
        Return the integral of the Gaussian.

        Returns
        -------
        integral : units.Quantity or float
        """
        return self.fwhm * self.fwhm_to_size

    def set_integral(self, integral):
        """
        Set the FWHM based on a supplied integral.

        Parameters
        ----------
        integral : units.Quantity or float

        Returns
        -------
        None
        """
        self.fwhm = integral / self.fwhm_to_size

    def value_at(self, x):
        """
        Return the value of the Gaussian evaluated at `x`.

        Parameters
        ----------
        x : int of float or numpy.ndarray or units.Quantity

        Returns
        -------
        value : float or numpy.ndarray or units.Quantity
        """
        y = self.model(x)
        if isinstance(y, units.Quantity) and y.unit == ud:
            y = y.value
        return y

    def get_beam_unit(self):
        """
        Return the value of the Gaussian over the beam.

        Returns
        -------
        units.Quantity
        """
        return self.get_integral() / units.Unit('beam')

    def get_beam_map(self, grid, sigmas=3.0):
        """
        Return a beam map given an output grid.

        Parameters
        ----------
        grid : sofia_redux.scan.coordinate_systems.grid.grid_1d.Grid1D
            The output grid on which the beam map should be projected.
        sigmas : float, optional
            The number of Gaussian sigmas to be encapsulated within the map
            (determines the size of the map).

        Returns
        -------
        beam_map : numpy.ndarray or units.Quantity
            A 1-D beam map image.  A Quantity if the Gaussian amplitude has
            units.
        """
        resolution = grid.resolution
        if isinstance(resolution, Coordinate1D):
            resolution = resolution.x
        fwhm = self.fwhm
        if isinstance(fwhm, units.Quantity) and fwhm.unit == ud:
            fwhm = fwhm.value
        if isinstance(resolution, units.Quantity
                      ) and resolution.unit == ud:  # pragma: no cover
            resolution = resolution.value

        w = sigmas * np.abs(fwhm) / resolution
        if isinstance(w, units.Quantity):
            w = w.decompose().value

        center = int(np.ceil(w))
        n = (2 * center) + 1
        x = (np.arange(n) - center) * resolution

        x0 = self.mean
        if isinstance(x0, units.Quantity) and x0.unit == ud:
            x0 = x0.value

        x += x0
        if self.position_unit not in [None, ud]:
            x = x.to(self.position_unit)

        return self.model(x)

    @staticmethod
    def get_equivalent(beam_map, pixel_size):
        """
        Return a 1-D Gaussian for a beam map and given pixel size.

        The equivalent psf if circular with a FWHM equal to that determined
        from the beam map.

        Parameters
        ----------
        beam_map : numpy.ndarray or units.Quantity
            A map of the beam of shape (n,).
        pixel_size : float or units.Quantity or Coordinate1D

        Returns
        -------
        Gaussian1D
        """
        psf = Gaussian1D()
        psf.set_equivalent(beam_map, pixel_size)
        return psf

    def set_equivalent(self, beam_map, pixel_size):
        """
        Set Gaussian parameters from a given beam map and pixel size.

        The integral and thus FWHM are determined from the input parameters.
        The mean will always be set to zero.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            A map of the beam of shape (ky, kx).
        pixel_size : Coordinate1D or float or units.Quantity
            The pixel size.

        Returns
        -------
        None
        """
        if isinstance(pixel_size, Coordinate1D):
            px = pixel_size.coordinates
        else:
            px = pixel_size

        map_sum = np.sum(np.abs(beam_map))
        integral = map_sum * px
        self.set_integral(integral)

    def parse_header(self, header, size_unit=None, fits_id=''):
        """
        Set the beam from a FITS header.

        Reads a FITS header to determine::

            - The FWHM (B1D)

        By default, an attempt will be made to determine the size unit for the
        FWHMs from the header comments, although a fixed unit may be supplied
        by the `size_unit` parameter.  The same process will occur for the
        position angle, but no overriding unit may be supplied.  In case no
        unit is determined, all parameters will default to 'degree'.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
        size_unit : units.Unit or units.Quantity or str
            The size unit to apply to the header values.  If `None`, an attempt
            will be made to determine the size unit from the header comments
            for the appropriate keywords, and if not found, will default to
            'degree'.  Any other supplied value will take priority.
        fits_id : str, optional
            The name prefixing the FITS header B1D keyword. For example, IB1D.

        Returns
        -------
        None
        """
        key = f'{fits_id}B1D'
        if key not in header:
            log.error(f"FITS header contains no 1D beam description "
                      f"for type '{fits_id}'.")
            return

        if size_unit is None:
            fwhm = get_header_quantity(header, key, default_unit='Hz')
        else:
            size_unit = units.Unit(size_unit)
            fwhm = header[key] * size_unit

        self.fwhm = fwhm

    def edit_header(self, header, fits_id='', beam_name=None, size_unit=None):
        """
        Edit a FITS header with the beam parameters.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        fits_id : str, optional
            The name prefixing the FITS header B1D keyword. For example, IB1D.
        beam_name : str, optional
            The name of the beam.
        size_unit : units.Unit or units.Quantity or str, optional
            If set, convert the major/minor beam values to this unit before
            setting in the header.

        Returns
        -------
        None
        """
        if beam_name not in ['', None]:
            header[f'{fits_id}BNAM'] = beam_name, 'Beam name.'
        fwhm = self.fwhm

        if size_unit is not None:
            size_unit = units.Unit(size_unit)
        elif isinstance(fwhm, units.Quantity):
            if fwhm.unit != ud:
                size_unit = fwhm.unit

        fwhm = to_header_float(fwhm, unit=size_unit)

        if size_unit is None:
            unit_str = ''
        else:
            unit_str = f' ({size_unit})'

        header[f'{fits_id}B1D'] = fwhm, f'Beam 1D axis{unit_str}.'
