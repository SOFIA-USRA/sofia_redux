# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
from abc import ABC
from astropy import log, units
from astropy.modeling import functional_models
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from copy import deepcopy
import numpy as np

from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.index_2d import Index2D

__all__ = ['Gaussian2D']


class Gaussian2D(ABC):

    # The constant to convert from FWHM^2 to beam integral
    AREA_FACTOR = 2.0 * np.pi / (gaussian_sigma_to_fwhm ** 2)

    # The equivalent area lateral square size to Gaussian beam with FWHM
    FWHM_TO_SIZE = np.sqrt(AREA_FACTOR)

    def __init__(self, peak=1.0, x_mean=0.0, y_mean=0.0, x_fwhm=0.0, y_fwhm=0.0,
                 theta=0.0 * units.Unit('deg'), peak_unit=None,
                 position_unit=None):

        if position_unit is None:
            position_unit = units.dimensionless_unscaled
        if peak_unit is None:
            if isinstance(peak, units.Quantity):
                peak_unit = peak.unit
            else:
                peak_unit = units.dimensionless_unscaled

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

    def __repr__(self):
        s = f"x_fwhm={self.x_fwhm}, y_fwhm={self.y_fwhm}, "
        s += f"x_mean={self.x_mean}, y_mean={self.y_mean}, "
        s += f"theta={self.theta}, peak={self.peak} {self.unit}"
        return s

    @property
    def referenced_attributes(self):
        """
        Return the names of attributes to be referenced rather than copied.

        Returns
        -------
        set of str
        """
        return set([])

    def copy(self):
        """
        Return a copy of the data.

        Returns
        -------
        FlaggedData
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            new = self.__class__()
            for key, value in self.__dict__.items():
                if key in self.referenced_attributes:
                    setattr(new, key, value)
                elif hasattr(value, 'copy'):
                    setattr(new, key, value.copy())
                else:
                    setattr(new, key, deepcopy(value))
        return new

    @property
    def x_fwhm(self):
        """
        Return the FWHM in the x-direction.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.model.x_stddev * gaussian_sigma_to_fwhm

    @x_fwhm.setter
    def x_fwhm(self, value):
        """
        Set the FWHM in the x-direction.

        Parameters
        ----------
        value : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.model.x_stddev = gaussian_fwhm_to_sigma * value

    @property
    def y_fwhm(self):
        """
        Return the FWHM in the y-direction.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.model.y_stddev * gaussian_sigma_to_fwhm

    @y_fwhm.setter
    def y_fwhm(self, value):
        """
        Set the FWHM in the y-direction.

        Parameters
        ----------
        value : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.model.y_stddev = gaussian_fwhm_to_sigma * value

    @property
    def x_stddev(self):
        """
        Return the standard deviation in x.

        Returns
        -------
        value : astropy.units.Quantity or float
        """
        return self.model.x_stddev

    @x_stddev.setter
    def x_stddev(self, value):
        """
        Set the standard deviation in x.

        Parameters
        ----------
        value : astropy.units.Quantity or float

        Returns
        -------
        None
        """
        self.model.x_stddev = value

    @property
    def y_stddev(self):
        """
        Return the standard deviation in y.

        Returns
        -------
        value : astropy.units.Quantity or float
        """
        return self.model.y_stddev

    @y_stddev.setter
    def y_stddev(self, value):
        """
        Set the standard deviation in y.

        Parameters
        ----------
        value : astropy.units.Quantity or float

        Returns
        -------
        None
        """
        self.model.y_stddev = value

    @property
    def position_angle(self):
        """
        Return the angle of the Gaussian.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.model.theta * 1.0

    @position_angle.setter
    def position_angle(self, value):
        """
        Set the position angle of the Gaussian.

        Parameters
        ----------
        value : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.model.theta = value % (np.pi * units.Unit('radian'))

    @property
    def theta(self):
        return self.position_angle

    @theta.setter
    def theta(self, value):
        self.position_angle = value

    @property
    def major_fwhm(self):
        """
        Return the major (maximum) FWHM value.

        Returns
        -------
        astropy.units.Quantity or float
        """
        return max(self.x_fwhm, self.y_fwhm)

    @property
    def minor_fwhm(self):
        """
        Return the minor (minimum) FWHM value.

        Returns
        -------
        astropy.units.Quantity
        """
        return min(self.x_fwhm, self.y_fwhm)

    @property
    def fwhm(self):
        """
        Return the average FWHM

        Returns
        -------
        astropy.units.Quantity
        """
        return self.get_circular_equivalent_fwhm()

    @fwhm.setter
    def fwhm(self, value):
        """
        Set the x and y FWHM to the given value.

        Parameters
        ----------
        value : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.x_fwhm = self.y_fwhm = value

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
        astropy.units.Quantity or float
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
        value : astropy.units.Quantity or float

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
        astropy.units.Quantity or float
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
        value : astropy.units.Quantity or float

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
        float or astropy.units.Quantity
        """
        return self.model.amplitude * 1.0

    @peak.setter
    def peak(self, value):
        """
        Set the peak amplitude of the Gaussian.

        Parameters
        ----------
        value : float or astropy.units.Quantity

        Returns
        -------
        None
        """
        self.model.amplitude = value

    def get_circular_equivalent_fwhm(self):
        """
        Return the FWHM of a circular Gaussian with equivalent area.

        Returns
        -------
        astropy.units.Quantity
        """
        return np.sqrt(self.x_fwhm * self.y_fwhm).to(
            self.x_fwhm.unit)

    def combine_with(self, psf, deconvolve=False):
        """
        Combine with another beam.

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

        if minor_fwhm > major_fwhm:
            self.x_fwhm = minor_fwhm
            self.y_fwhm = major_fwhm
            position_angle += (np.pi / 2) * units.Unit('radian')
        else:
            self.x_fwhm = major_fwhm
            self.y_fwhm = minor_fwhm

        self.position_angle = position_angle % (np.pi * units.Unit('radian'))

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
        psf : Gaussian2D or astropy.units.Quantity
            Another Gaussian PSF or the FWHM of another Gaussian PSF.

        Returns
        -------
        None
        """
        x_fwhm = self.major_fwhm
        y_fwhm = self.minor_fwhm
        if isinstance(psf, units.Quantity):
            if x_fwhm < psf:
                self.x_fwhm = psf
            if y_fwhm < psf:
                self.y_fwhm = psf
            return

        delta_angle = psf.position_angle - self.position_angle
        c = np.cos(delta_angle)
        s = np.sin(delta_angle)

        min_x = np.sqrt(((x_fwhm * c) ** 2) + (psf.y_fwhm * s) ** 2)
        min_y = np.sqrt(((x_fwhm * s) ** 2) + (psf.y_fwhm * c) ** 2)
        if x_fwhm < min_x:
            self.x_fwhm = min_x
        if y_fwhm < min_y:
            self.y_fwhm = min_y

    def rotate(self, angle):
        """
        Rotate the beam by a given angle.

        Parameters
        ----------
        angle : astropy.units.Quantity

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

    def parse_header(self, header, size_unit='deg', fits_id=''):
        """
        Set the beam from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
        size_unit : astropy.units.Unit or astropy.units.Quantity or str
            The size unit to apply to the header values
        fits_id : str, optional
            The name prefixing the FITS header BMAJ/BMIN keywords.  For example,
            IBMAJ.

        Returns
        -------
        None
        """
        if isinstance(size_unit, str):
            size_unit = units.Unit(size_unit)
        major_key = f'{fits_id}BMAJ'
        if major_key not in header:
            log.error(f"FITS header contains no beam description "
                      f"for type '{fits_id}'.")
            return
        minor_key = f'{fits_id}BMIN'
        angle_key = f'{fits_id}BPA'
        self.x_fwhm = float(header.get(major_key, np.nan)) * size_unit
        self.y_fwhm = float(header.get(minor_key, np.nan)) * size_unit
        if np.isnan(self.y_fwhm):
            self.y_fwhm = self.x_fwhm
        self.position_angle = header.get(angle_key, 0.0) * units.Unit('deg')

    def edit_header(self, header, fits_id='', beam_name=None, size_unit=None):
        """
        Edit a FITS header with the beam parameters.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        fits_id : str, optional
            The name prefixing the FITS header BMAJ/BMIN keywords.  For example,
            IBMAJ.
        beam_name : str, optional
            The name of the beam.
        size_unit : astropy.units.Unit or Quantity or str, optional
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
        if isinstance(size_unit, str):
            size_unit = units.Unit(size_unit)
        elif isinstance(size_unit, units.Quantity):
            size_unit = size_unit.unit
        major_fwhm = major_fwhm.to(size_unit)
        minor_fwhm = minor_fwhm.to(size_unit)
        major_string_unit = major_fwhm.unit.name
        minor_string_unit = minor_fwhm.unit.name
        angle = self.position_angle.to('deg').value
        header[f'{fits_id}BMAJ'] = (
            major_fwhm.value, f'Beam major axis ({major_string_unit}).')
        header[f'{fits_id}BMIN'] = (
            minor_fwhm.value, f'Beam minor axis ({minor_string_unit}).')
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
        min_major = np.hypot(psf.x_fwhm * cos, psf.y_fwhm * sin)
        min_minor = np.hypot(psf.x_fwhm * sin, psf.y_fwhm * cos)
        if self.x_fwhm < min_major:
            return False
        if self.y_fwhm < min_minor:
            return False
        return True

    def extent(self):
        """
        Return the extent in x and y.

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
        sigmas : float, optional

        Returns
        -------
        beam_map : numpy.ndarray (float)
            A 2-D beam map image.
        """
        v0 = Coordinate2D(coordinates=[self.major_fwhm, self.minor_fwhm])
        v = v0.copy()
        v.rotate(self.position_angle)

        pixel_sizes = grid.get_pixel_size()
        pixels_per_beam = v.coordinates / pixel_sizes.coordinates
        if isinstance(pixels_per_beam, units.Quantity):
            pixels_per_beam = pixels_per_beam.decompose().value

        map_size = Index2D(
            coordinates=(2 * np.ceil(sigmas * pixels_per_beam).astype(int)) + 1)

        sigma = v0.copy()
        sigma.scale(gaussian_fwhm_to_sigma)
        ax = -0.5 * (pixel_sizes.x / sigma.x) ** 2
        ay = -0.5 * (pixel_sizes.y / sigma.y) ** 2
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
                px, py = np.atleast_1d(pixel_size).ravel()[0]
        else:
            px, py = pixel_size

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
        self.fwhm = np.sqrt(area / self.AREA_FACTOR).decompose().to('degree')
