# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from abc import ABC
from astropy import log, units
from astropy.modeling import functional_models
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from copy import deepcopy
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.utilities.utils import (
    get_header_quantity, to_header_float)

__all__ = ['Gaussian2D']


class Gaussian2D(ABC):

    # The constant to convert from FWHM^2 to beam integral
    AREA_FACTOR = 2.0 * np.pi / (gaussian_sigma_to_fwhm ** 2)

    # The equivalent area lateral square size to Gaussian beam with FWHM
    FWHM_TO_SIZE = np.sqrt(AREA_FACTOR)
    QUARTER = np.pi / 2 * units.Unit('radian')

    def __init__(self, peak=1.0, x_mean=0.0, y_mean=0.0,
                 x_fwhm=0.0, y_fwhm=0.0, theta=0.0 * units.Unit('deg'),
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
        x_mean : float or units.Quantity, optional
            The position of the peak along the x-axis.
        y_mean : float or units.Quantity, optional
            The position of the peak along the y-axis.
        x_fwhm : float or units.Quantity, optional
            The Full-Width-Half-Max beam width in the x-direction.
        y_fwhm : float or units.Quantity, optional
            The Full-Width-Half-Max beam width in the y-direction.
        theta : float or units.Quantity, optional
            The rotation of the beam pertaining to `x_fwhm` and `y_fwhm` in
            relation to the actual (x, y) coordinate axis.  If a float value is
            supplied, it is assumed to be in degrees.
        peak_unit : units.Unit or units.Quantity or str, optional
            The physical units for the peak amplitude.  The default is
            dimensionless.
        position_unit : units.Unit or units.Quantity or str, optional
            The physical units of all position based parameters (`x_mean`,
            `y_mean`, `x_fwhm`, `y_fwhm`)
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

        if not isinstance(x_fwhm, units.Quantity):
            x_fwhm = x_fwhm * position_unit
        if not isinstance(y_fwhm, units.Quantity):
            y_fwhm = y_fwhm * position_unit
        if not isinstance(x_mean, units.Quantity):
            x_mean = x_mean * position_unit
        if not isinstance(y_mean, units.Quantity):
            y_mean = y_mean * position_unit
        if not isinstance(theta, units.Quantity):
            theta = theta * units.Unit('degree')

        if not isinstance(peak, units.Quantity):
            peak = peak * peak_unit

        self.unit = peak_unit
        self.model = functional_models.Gaussian2D(
            x_stddev=x_fwhm * gaussian_fwhm_to_sigma,
            y_stddev=y_fwhm * gaussian_fwhm_to_sigma,
            x_mean=x_mean, y_mean=y_mean,
            amplitude=peak.value, theta=theta)
        self.validate()

    def copy(self):
        """
        Return a copy of the Gaussian 2D model.

        Returns
        -------
        Gaussian2D
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
    def x_fwhm(self):
        """
        Return the FWHM in the x-direction.

        Returns
        -------
        units.Quantity
        """
        return self.model.x_stddev * gaussian_sigma_to_fwhm

    @x_fwhm.setter
    def x_fwhm(self, value):
        """
        Set the FWHM in the x-direction.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.model.x_stddev = gaussian_fwhm_to_sigma * value
        self.validate()

    @property
    def y_fwhm(self):
        """
        Return the FWHM in the y-direction.

        Returns
        -------
        units.Quantity
        """
        return self.model.y_stddev * gaussian_sigma_to_fwhm

    @y_fwhm.setter
    def y_fwhm(self, value):
        """
        Set the FWHM in the y-direction.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.model.y_stddev = gaussian_fwhm_to_sigma * value
        self.validate()

    @property
    def x_stddev(self):
        """
        Return the standard deviation in x.

        Returns
        -------
        value : units.Quantity or float
        """
        return self.model.x_stddev

    @x_stddev.setter
    def x_stddev(self, value):
        """
        Set the standard deviation in x.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        self.model.x_stddev = value
        self.validate()

    @property
    def y_stddev(self):
        """
        Return the standard deviation in y.

        Returns
        -------
        value : units.Quantity or float
        """
        return self.model.y_stddev

    @y_stddev.setter
    def y_stddev(self, value):
        """
        Set the standard deviation in y.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        self.model.y_stddev = value
        self.validate()

    @property
    def position_angle(self):
        """
        Return the position angle of the Gaussian.

        Returns
        -------
        units.Quantity
        """
        return self.model.theta * 1.0

    @position_angle.setter
    def position_angle(self, value):
        """
        Set the position angle of the Gaussian.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.model.theta = value % (np.pi * units.Unit('radian'))

    @property
    def theta(self):
        """
        Return the position angle of the Gaussian.

        Returns
        -------
        units.Quantity
        """
        return self.position_angle

    @theta.setter
    def theta(self, value):
        """
        Set the position angle of the Gaussian.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.position_angle = value

    @property
    def major_fwhm(self):
        """
        Return the major (maximum) FWHM value.

        Returns
        -------
        units.Quantity or float
        """
        return max(self.x_fwhm, self.y_fwhm)

    @property
    def minor_fwhm(self):
        """
        Return the minor (minimum) FWHM value.

        Returns
        -------
        units.Quantity
        """
        return min(self.x_fwhm, self.y_fwhm)

    @property
    def fwhm(self):
        """
        Return the average FWHM

        Returns
        -------
        units.Quantity
        """
        return self.get_circular_equivalent_fwhm()

    @fwhm.setter
    def fwhm(self, value):
        """
        Set the x and y FWHM to the given value.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.set_xy_fwhm(value, value)

    @property
    def area(self):
        """
        Return the area of the PSF.

        Returns
        -------
        units.Quantity
        """
        return abs(self.AREA_FACTOR * self.x_fwhm * self.y_fwhm)

    @area.setter
    def area(self, value):
        """
        Set the FWHM of the beam so that the area is that which is given.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.fwhm = np.sqrt(value / self.AREA_FACTOR)

    @property
    def x_mean(self):
        """
        Return the center of the Gaussian in x.

        Returns
        -------
        units.Quantity or float
        """
        value = self.model.x_mean.value
        unit = self.model.x_mean.unit
        if unit is None:
            return value
        else:
            return value * unit

    @x_mean.setter
    def x_mean(self, value):
        """
        Set the value of the center of the Gaussian in x.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        self.model.x_mean = value

    @property
    def y_mean(self):
        """
        Return the center of the Gaussian in y.

        Returns
        -------
        units.Quantity or float
        """
        value = self.model.y_mean.value
        unit = self.model.y_mean.unit
        if unit is None:
            return value
        else:
            return value * unit

    @y_mean.setter
    def y_mean(self, value):
        """
        Set the value of the center of the Gaussian in y.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        self.model.y_mean = value

    @property
    def peak(self):
        """
        Return the peak amplitude of the Gaussian.

        Returns
        -------
        float or units.Quantity
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
        self.model.amplitude = value

    def __str__(self):
        """
        Return string information on the Gaussian2D model.

        Returns
        -------
        str
        """
        s = f"x_fwhm={self.x_fwhm}, y_fwhm={self.y_fwhm}, "
        s += f"x_mean={self.x_mean}, y_mean={self.y_mean}, "
        s += f"theta={self.theta}, peak={self.peak} {self.unit}"
        return s

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
        Test if this Gaussian2D instance is functionally equivalent to another.

        Parameters
        ----------
        other : Gaussian2D

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        try:
            if not np.isclose(self.theta, other.theta):
                return False
            if not np.isclose(self.x_fwhm, other.x_fwhm):
                return False
            if not np.isclose(self.y_fwhm, other.y_fwhm):
                return False
            if not np.isclose(self.x_mean, other.x_mean):
                return False
            if not np.isclose(self.y_mean, other.y_mean):
                return False
        except (TypeError, units.UnitConversionError):  # pragma: no cover
            # Just in case these managed to get set strangely (hard to do)
            return False
        return (self.peak * self.unit) == (other.peak * other.unit)

    def set_xy_fwhm(self, x_fwhm, y_fwhm):
        """
        Set and validate both new x and y FWHM values at once.

        The correct position angle must be set for the process to occur
        correctly.  This is so both FWHMs get set correctly, as doing it
        sequentially may result in one of the values getting overwritten by
        the other.

        Parameters
        ----------
        x_fwhm : units.Quantity
        y_fwhm : units.Quantity

        Returns
        -------
        None
        """
        x_stddev = x_fwhm * gaussian_fwhm_to_sigma
        y_stddev = y_fwhm * gaussian_fwhm_to_sigma
        self.model.x_stddev = x_stddev
        self.model.y_stddev = y_stddev
        self.validate()

    def validate(self):
        """
        Set the (x, y) FWHM and position angle so that x >= y.

        For consistency, the major and minor axes of the Gaussian FWHM are
        set so they are equal to the (x, y) axes.  This will result in a 90
        degree offset to the position angle if y > x on input.

        Returns
        -------
        None
        """
        x = self.model.x_stddev
        y = self.model.y_stddev
        if x >= y:
            return

        x = x.value if x.unit is None else x.value * x.unit
        y = y.value if y.unit is None else y.value * y.unit

        self.model.x_stddev = y
        self.model.y_stddev = x
        self.position_angle += self.QUARTER

    def get_circular_equivalent_fwhm(self):
        """
        Return the FWHM of a circular Gaussian with equivalent area.

        Returns
        -------
        units.Quantity
        """
        return np.sqrt(self.x_fwhm * self.y_fwhm).to(
            self.x_fwhm.unit)

    def combine_with(self, psf, deconvolve=False):
        """
        Combine with another beam.

        Combination consists of either convolution (default) or deconvolution
        of this beam by another.

        Parameters
        ----------
        psf : Gaussian2D
            The beam to combine.
        deconvolve : bool, optional
            If `True`, indicates a deconvolution rather than a convolution.

        Returns
        -------
        None
        """
        if psf is None:
            return

        a2x = self.major_fwhm ** 2
        a2y = self.minor_fwhm ** 2
        b2x = psf.major_fwhm ** 2
        b2y = psf.minor_fwhm ** 2
        direction = -1 if deconvolve else 1
        a = a2x - a2y
        b = b2x - b2y
        angle_a = self.position_angle
        angle_b = psf.position_angle
        delta = (2 * (angle_b - angle_a)) % (np.pi * units.Unit('radian'))
        c = (a ** 2) + (b ** 2) + (2 * a * b * np.cos(delta))
        c = np.sqrt(c) if c >= 0 else np.nan
        bb = a2x + a2y + (direction * (b2x + b2y))

        major_fwhm = bb + c
        minor_fwhm = bb - c
        if np.isnan(c) or major_fwhm < 0:
            major_fwhm = self.major_fwhm * 0
        else:
            major_fwhm = np.sqrt(0.5 * major_fwhm)

        if np.isnan(c) or minor_fwhm < 0:
            minor_fwhm = self.minor_fwhm * 0
        else:
            minor_fwhm = np.sqrt(0.5 * minor_fwhm)

        if np.isnan(c) or c == 0:
            position_angle = 0.0
        else:
            sin_beta = direction * np.sin(delta) * b / c
            position_angle = self.position_angle + (0.5 * np.arcsin(sin_beta))

        self.position_angle = position_angle
        self.set_xy_fwhm(major_fwhm, minor_fwhm)

        # if minor_fwhm > major_fwhm:  # pragma: no cover
        #     self.x_fwhm = minor_fwhm
        #     self.y_fwhm = major_fwhm
        #     position_angle += (np.pi / 2) * units.Unit('radian')
        # else:
        #     self.x_fwhm = major_fwhm
        #     self.y_fwhm = minor_fwhm

    def convolve_with(self, psf):
        """
        Convolve with a given PSF.

        Parameters
        ----------
        psf : Gaussian2D
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
        psf : Gaussian2D
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
        psf : Gaussian2D or units.Quantity
            Another Gaussian PSF or the FWHM of another Gaussian PSF.

        Returns
        -------
        None
        """
        x_fwhm = self.major_fwhm
        y_fwhm = self.minor_fwhm
        if isinstance(psf, units.Quantity):
            if x_fwhm < psf:
                x_fwhm = psf
            if y_fwhm < psf:
                y_fwhm = psf
            self.set_xy_fwhm(x_fwhm, y_fwhm)
            return

        delta_angle = psf.position_angle - self.position_angle
        c = np.cos(delta_angle)
        s = np.sin(delta_angle)

        a = psf.major_fwhm
        b = psf.minor_fwhm

        min_x = np.sqrt(((a * c) ** 2) + (b * s) ** 2)
        min_y = np.sqrt(((a * s) ** 2) + (b * c) ** 2)

        new_x = min_x if x_fwhm < min_x else x_fwhm
        new_y = min_y if y_fwhm < min_y else y_fwhm
        self.set_xy_fwhm(new_x, new_y)

    def rotate(self, angle):
        """
        Rotate the beam by a given angle.

        Parameters
        ----------
        angle : units.Quantity

        Returns
        -------
        None
        """
        self.position_angle = self.position_angle + angle

    def scale(self, factor):
        """
        Scale the FWHM in both axes by a given amount.

        Parameters
        ----------
        factor : float

        Returns
        -------
        None
        """
        self.model.x_stddev *= factor
        self.model.y_stddev *= factor

    def parse_header(self, header, size_unit=None, fits_id=''):
        """
        Set the beam from a FITS header.

        Reads a FITS header to determine::

            - The major FWHM (BMAJ)
            - The minor FWHM (BMIN)
            - The position angle (BPA)

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
            The name prefixing the FITS header BMAJ/BMIN keywords.
            For example, IBMAJ.

        Returns
        -------
        None
        """
        major_key = f'{fits_id}BMAJ'
        minor_key = f'{fits_id}BMIN'
        angle_key = f'{fits_id}BPA'
        if major_key not in header:
            log.error(f"FITS header contains no beam description "
                      f"for type '{fits_id}'.")
            return
        if minor_key not in header:
            minor_key = major_key

        if size_unit is None:
            x_fwhm = get_header_quantity(header, major_key,
                                         default_unit='degree')
            y_fwhm = get_header_quantity(
                header, minor_key, default_unit='degree').to(x_fwhm.unit)
        else:
            size_unit = units.Unit(size_unit)
            x_fwhm = header[major_key] * size_unit
            y_fwhm = header[minor_key] * size_unit

        self.position_angle = get_header_quantity(
            header, angle_key, default=0.0, default_unit='degree')
        self.set_xy_fwhm(x_fwhm, y_fwhm)

    def edit_header(self, header, fits_id='', beam_name=None, size_unit=None):
        """
        Edit a FITS header with the beam parameters.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        fits_id : str, optional
            The name prefixing the FITS header BMAJ/BMIN keywords.
            For example, IBMAJ.
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
        major_fwhm = self.major_fwhm
        minor_fwhm = self.minor_fwhm

        if size_unit is not None:
            size_unit = units.Unit(size_unit)
        elif isinstance(major_fwhm, units.Quantity):
            if major_fwhm.unit != units.dimensionless_unscaled:
                size_unit = major_fwhm.unit

        major_fwhm = to_header_float(major_fwhm, unit=size_unit)
        minor_fwhm = to_header_float(minor_fwhm, unit=size_unit)
        angle = to_header_float(self.position_angle, unit='degree')

        if size_unit is None:
            unit_str = ''
        else:
            unit_str = f' ({size_unit})'

        header[f'{fits_id}BMAJ'] = major_fwhm, f'Beam major axis{unit_str}.'
        header[f'{fits_id}BMIN'] = minor_fwhm, f'Beam minor axis{unit_str}.'
        header[f'{fits_id}BPA'] = angle, 'Beam position angle (deg).'

    def is_circular(self):
        """
        Return whether the PSF is circular.

        Returns
        -------
        bool
        """
        return np.isclose(self.x_fwhm, self.y_fwhm, rtol=1e-6)

    def is_encompassing(self, psf):
        """
        Check if the psf is encompassing the beam.

        Parameters
        ----------
        psf : Gaussian2D

        Returns
        -------
        bool
        """
        delta_angle = psf.position_angle - self.position_angle
        cos = np.cos(delta_angle)
        sin = np.sin(delta_angle)
        a = psf.major_fwhm
        b = psf.minor_fwhm

        min_major = np.hypot(a * cos, b * sin)
        min_minor = np.hypot(a * sin, b * cos)
        if self.major_fwhm < min_major:
            return False
        if self.minor_fwhm < min_minor:
            return False
        return True

    def extent(self):
        """
        Return the extent in x and y.

        The extent is the maximum (x, y) deviation away from the center of the
        psf in the native coordinate frame, accounting for rotation by the
        position angle.

        Returns
        -------
        extent : Coordinate2D
            The extent in x and y.
        """
        cos = np.cos(self.position_angle)
        sin = np.sin(self.position_angle)
        x = np.hypot(cos * self.x_fwhm, sin * self.y_fwhm)
        y = np.hypot(cos * self.y_fwhm, sin * self.x_fwhm)
        return Coordinate2D(coordinates=[x, y])

    def get_beam_map(self, grid, sigmas=3.0):
        """
        Return a beam map given an output grid.

        Parameters
        ----------
        grid : Grid2D
            The output grid on which the beam map should be projected.
        sigmas : float, optional
            The number of Gaussian sigmas to be encapsulated within the map
            (determines the size of the map).

        Returns
        -------
        beam_map : numpy.ndarray (float)
            A 2-D beam map image.
        """
        extent = self.extent()
        pixel_sizes = grid.get_pixel_size()
        pixels_per_beam = extent.coordinates / pixel_sizes.coordinates
        if isinstance(pixels_per_beam, units.Quantity):
            pixels_per_beam = pixels_per_beam.decompose().value
        coord = (2 * np.ceil(sigmas * pixels_per_beam).astype(int)) + 1
        map_size = Index2D(coordinates=coord)

        sigma = Coordinate2D(coordinates=[self.major_fwhm, self.minor_fwhm])
        sigma.scale(gaussian_fwhm_to_sigma)
        ax = -0.5 * (pixel_sizes.x / sigma.x) ** 2
        ay = -0.5 * (pixel_sizes.y / sigma.y) ** 2
        if isinstance(ax, units.Quantity):
            ax = ax.decompose().value
            ay = ay.decompose().value

        center = Coordinate2D(coordinates=(map_size.coordinates - 1) / 2.0)

        y, x = np.mgrid[:map_size.y, :map_size.x]
        dx = x - center.x
        dy = y - center.y
        v = Coordinate2D(coordinates=[dx, dy])
        v.rotate(-self.position_angle)
        image = np.exp((ax * (v.x ** 2)) + (ay * (v.y ** 2)))
        return image

    @staticmethod
    def get_equivalent(beam_map, pixel_size):
        """
        Return a 2-D Gaussian for a beam map and given pixel size.

        The equivalent psf if circular with a FWHM equal to that determined
        from the beam map.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            A map of the beam of shape (ky, kx).
        pixel_size : Coordinate2D or numpy.ndarray or float or units.Quantity
            The pixel size.

        Returns
        -------
        Gaussian2D
        """
        psf = Gaussian2D()
        psf.set_equivalent(beam_map, pixel_size)
        return psf

    def set_equivalent(self, beam_map, pixel_size):
        """
        Set Gaussian parameters from a given beam map and pixel size.

        The area and thus FWHM are determined from the input parameters.
        Therefore, only a circular Gaussian will be set with zero position
        angle.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            A map of the beam of shape (ky, kx).
        pixel_size : Coordinate2D or numpy.ndarray or float or units.Quantity
            The pixel size.

        Returns
        -------
        None
        """
        if isinstance(pixel_size, Coordinate2D):
            px, py = pixel_size.coordinates
        elif isinstance(pixel_size, np.ndarray):
            if pixel_size.size == 2:
                px, py = pixel_size.ravel()
            else:
                px = py = np.atleast_1d(pixel_size).ravel()[0]
        else:
            px = py = pixel_size

        area = np.sum(np.abs(beam_map)) * px * py
        self.set_area(area)

    def set_area(self, area):
        """
        Set the Gaussian parameters based on the overall area of the beam.

        Parameters
        ----------
        area : float or units.Quantity

        Returns
        -------
        None
        """
        fwhm = np.sqrt(area / self.AREA_FACTOR)
        if not isinstance(fwhm, units.Quantity):
            fwhm = fwhm * units.dimensionless_unscaled
        self.fwhm = fwhm
