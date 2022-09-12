# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from astropy import units
from astropy.stats import gaussian_fwhm_to_sigma
from copy import deepcopy
import numpy as np

from sofia_redux.scan.source_models.beams.gaussian_1d import Gaussian1D
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.coordinate_systems.coordinate_3d import Coordinate3D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.index_3d import Index3D

__all__ = ['Gaussian2D1']


class Gaussian2D1(Gaussian2D):

    fwhm_to_size = np.sqrt(2 * np.pi) * gaussian_fwhm_to_sigma

    def __init__(self, peak=1.0, x_mean=0.0, y_mean=0.0, z_mean=0.0,
                 x_fwhm=0.0, y_fwhm=0.0, z_fwhm=0,
                 theta=0.0 * units.Unit('deg'),
                 peak_unit=None, position_unit=None, z_unit=None):
        """
        Initializes a 2D + 1D Gaussian beam model.

        The Gaussian2D1 is used to represent a model using an (x, y) plane
        replicated along the z-axis.

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
        self.z = Gaussian1D(peak=peak, mean=z_mean, fwhm=z_fwhm,
                            peak_unit=peak_unit, position_unit=z_unit)
        super().__init__(peak=peak, x_mean=x_mean, y_mean=y_mean,
                         x_fwhm=x_fwhm, y_fwhm=y_fwhm, theta=theta,
                         peak_unit=peak_unit, position_unit=position_unit)

    def copy(self):
        """
        Return a copy of the Gaussian 2D+1 model.

        Returns
        -------
        Gaussian2D1
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
    def z_fwhm(self):
        """
        Return the FWHM in the z-direction.

        Returns
        -------
        units.Quantity
        """
        return self.z.fwhm

    @z_fwhm.setter
    def z_fwhm(self, value):
        """
        Set the FWHM in the z-direction.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.z.fwhm = value
        self.validate()

    @property
    def z_stddev(self):
        """
        Return the standard deviation in z.

        Returns
        -------
        value : units.Quantity or float
        """
        return self.z.stddev

    @z_stddev.setter
    def z_stddev(self, value):
        """
        Set the standard deviation in z.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        self.z.stddev = value
        self.validate()

    @property
    def volume(self):
        """
        Return the volume of the PSF.

        Returns
        -------
        float or units.Quantity
        """
        return self.area * self.z.get_integral()

    @property
    def z_integral(self):
        """
        Return the integral of the PSF in the z-direction.

        Returns
        -------
        integral : float or units.Quantity
        """
        return self.z.get_integral()

    @z_integral.setter
    def z_integral(self, value):
        """
        Set the FWHM of the PSF in the z-direction given an integral.

        Parameters
        ----------
        value : float or units.Quantity

        Returns
        -------
        None
        """
        self.z.set_integral(value)

    @property
    def z_mean(self):
        """
        Return the center of the Gaussian in z.

        Returns
        -------
        units.Quantity or float
        """
        value = self.z.model.mean.value
        unit = self.z.model.mean.unit
        if unit is None:  # pragma: no cover
            return value
        else:
            return value * unit

    @z_mean.setter
    def z_mean(self, value):
        """
        Set the value of the center of the Gaussian in z.

        Parameters
        ----------
        value : units.Quantity or float

        Returns
        -------
        None
        """
        self.z.mean = value

    def __str__(self):
        """
        Return string information on the Gaussian2D model.

        Returns
        -------
        str
        """
        if self.unit in [None, units.dimensionless_unscaled]:
            unit_str = ''
        else:
            unit_str = f' {self.unit}'
        s = f"x_fwhm={self.x_fwhm}, y_fwhm={self.y_fwhm}, "
        s += f"z_fwhm={self.z_fwhm}, x_mean={self.x_mean}, "
        s += f"y_mean={self.y_mean}, z_mean={self.z_mean}, "
        s += f"theta={self.theta}, peak={self.peak}{unit_str}"
        return s

    def __eq__(self, other):
        """
        Test if this Gaussian2D1 instance is equivalent to another.

        Parameters
        ----------
        other : Gaussian2D1

        Returns
        -------
        equal : bool
        """
        if not super().__eq__(other):
            return False
        if self.z != other.z:
            return False
        return True

    def set_xyz_fwhm(self, x_fwhm, y_fwhm, z_fwhm):
        """
        Set and validate both new x and y FWHM values at once.

        The correct position angle must be set for the process to occur
        correctly.  This is so both FWHMs get set correctly, as doing it
        sequentially may result in one of the values getting overwritten by
        the other.

        Parameters
        ----------
        x_fwhm : units.Quantity or float
        y_fwhm : units.Quantity or float
        z_fwhm : units.Quantity or float

        Returns
        -------
        None
        """
        xy_unit = self.x_stddev.unit
        z_unit = self.z_stddev.unit
        if not isinstance(x_fwhm, units.Quantity) and xy_unit is not None:
            x_fwhm = x_fwhm * xy_unit
        if not isinstance(y_fwhm, units.Quantity) and xy_unit is not None:
            y_fwhm = y_fwhm * xy_unit
        if not isinstance(z_fwhm, units.Quantity) and z_unit is not None:
            z_fwhm = z_fwhm * z_unit

        x_stddev = x_fwhm * gaussian_fwhm_to_sigma
        y_stddev = y_fwhm * gaussian_fwhm_to_sigma
        self.model.x_stddev = x_stddev
        self.model.y_stddev = y_stddev
        self.z.fwhm = z_fwhm
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
        super().validate()
        self.z.validate()

    def combine_with(self, psf, deconvolve=False):
        """
        Combine with another beam.

        Combination consists of either convolution (default) or deconvolution
        of this beam by another.

        Parameters
        ----------
        psf : Gaussian2D1 or None
            The beam to combine.
        deconvolve : bool, optional
            If `True`, indicates a deconvolution rather than a convolution.

        Returns
        -------
        None
        """
        if psf is None:
            return
        super().combine_with(psf, deconvolve=deconvolve)
        self.z.combine_with(psf.z, deconvolve=deconvolve)

    def convolve_with(self, psf):
        """
        Convolve with a given PSF.

        Parameters
        ----------
        psf : Gaussian2D1
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
        psf : Gaussian2D1
           The Point Spread Function to deconvolve with.

        Returns
        -------
        None
        """
        self.combine_with(psf, deconvolve=True)

    def encompass(self, psf, z_psf=None):
        """
        Encompass with another beam.

        Parameters
        ----------
        psf : Gaussian2D1 or Gaussian2D or units.Quantity
            Another Gaussian PSF or the FWHM of another Gaussian PSF.
        z_psf : Gaussian1D or units.Quantity, optional
            The Gaussian to encompass the z-psf.

        Returns
        -------
        None
        """
        super().encompass(psf)
        if z_psf is not None:
            self.z.encompass(z_psf)
        elif isinstance(psf, Gaussian2D1):
            self.z.encompass(psf.z)

    def scale_z(self, factor):
        """
        Scale the FWHM in z by a given amount.

        Parameters
        ----------
        factor : float

        Returns
        -------
        None
        """
        self.z.scale(factor)

    def parse_header(self, header, size_unit=None, z_unit=None, fits_id=''):
        """
        Set the beam from a FITS header.

        Reads a FITS header to determine::

            - The major FWHM (BMAJ)
            - The minor FWHM (BMIN)
            - The position angle (BPA)
            - The z FWHM (B1D)

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
        z_unit : units.Unit or units.Quantity or str
            The size unit to apply to the header values in the z-axis.  If
            `None`, an attempt will be made to determine the size unit from
            the header comments for the appropriate keywords, and if not
            found, will default to 'degree'.  Any other supplied value will
            take priority.
        fits_id : str, optional
            The name prefixing the FITS header BMAJ/BMIN keywords.
            For example, IBMAJ.

        Returns
        -------
        None
        """
        super().parse_header(header, size_unit=size_unit, fits_id=fits_id)
        self.z.parse_header(header, size_unit=z_unit, fits_id=fits_id)

    def edit_header(self, header, fits_id='', beam_name=None,
                    size_unit=None, z_unit=None):
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
        z_unit : units.Unit or units.Quantity or str, optional
            If set, convert the 1D beam values to this unit before setting
            in the header.

        Returns
        -------
        None
        """
        super().edit_header(header, fits_id=fits_id, beam_name=beam_name,
                            size_unit=size_unit)
        self.z.edit_header(header, fits_id=fits_id, beam_name=beam_name,
                           size_unit=z_unit)

    def is_encompassing(self, psf):
        """
        Check if the psf is encompassing the beam.

        Parameters
        ----------
        psf : Gaussian2D1 or Gaussian2D

        Returns
        -------
        bool
        """
        if not super().is_encompassing(psf):
            return False

        if not isinstance(psf, Gaussian2D1):
            return True

        return self.z.is_encompassing(psf.z)

    def extent(self):
        """
        Return the extent in x and y.

        The extent is the maximum (x, y) deviation away from the center of the
        psf in the native coordinate frame, accounting for rotation by the
        position angle.

        Returns
        -------
        extent : Coordinate2D1
            The extent in x and y.
        """
        xy_extent = super().extent()
        z_extent = self.z.extent()
        return Coordinate2D1(xy=xy_extent, z=z_extent)

    def get_beam_map(self, grid, sigmas=3.0):
        """
        Return a beam map given an output grid.

        Parameters
        ----------
        grid : Grid2D1
            The output grid on which the beam map should be projected.
        sigmas : float or Coordinate2D1 or iterable, optional
            The number of Gaussian sigmas to be encapsulated within the map
            (determines the size of the map).

        Returns
        -------
        beam_map : numpy.ndarray (float)
            A 3-D beam map image.
        """
        if isinstance(sigmas, Coordinate2D1):
            sigmas = np.asarray([sigmas.x, sigmas.y, sigmas.z])
        elif hasattr(sigmas, '__len__'):
            if isinstance(sigmas, np.ndarray) and sigmas.shape == ():
                sigmas = np.full(3, sigmas)
            elif len(sigmas) <= 1:
                sigmas = np.full(3, sigmas[0])
            elif len(sigmas) == 2:
                sigmas = np.asarray([sigmas[0], sigmas[0], sigmas[1]])
            else:
                sigmas = np.asarray(sigmas[:3])
        else:
            sigmas = np.full(3, sigmas)

        extent = self.extent()
        pixel_sizes = grid.get_pixel_size()

        pixels_per_beam = [extent.x / pixel_sizes.x,
                           extent.y / pixel_sizes.y,
                           extent.z / pixel_sizes.z]

        for i, px in enumerate(pixels_per_beam):
            if isinstance(px, units.Quantity):
                pixels_per_beam[i] = px.decompose().value

        pixels_per_beam = np.asarray(pixels_per_beam)
        coord = (2 * np.ceil(sigmas * pixels_per_beam).astype(int)) + 1
        map_size = Index3D(coordinates=coord)

        sigma = Coordinate2D1(xy=[self.major_fwhm, self.minor_fwhm],
                              z=self.z_fwhm)
        sigma.scale(gaussian_fwhm_to_sigma)

        ax = -0.5 * (pixel_sizes.x / sigma.x) ** 2 if sigma.x != 0 else 0.0
        ay = -0.5 * (pixel_sizes.y / sigma.y) ** 2 if sigma.y != 0 else 0.0
        az = -0.5 * (pixel_sizes.z / sigma.z) ** 2 if sigma.z != 0 else 0.0
        if isinstance(ax, units.Quantity):
            ax = ax.decompose().value
            ay = ay.decompose().value
        if isinstance(az, units.Quantity):
            az = az.decompose().value

        center = Coordinate3D(coordinates=(map_size.coordinates - 1) / 2.0)

        nx, ny, nz = map_size.coordinates
        z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                              indexing='ij')
        dx = x - center.x
        dy = y - center.y
        dz = z - center.z
        v = Coordinate3D(coordinates=[dx, dy, dz])
        if self.position_angle != 0:
            v.rotate(-self.position_angle)
        cube = np.exp((ax * (v.x ** 2)) +
                      (ay * (v.y ** 2)) +
                      (az * (v.z ** 2)))
        return cube

    @staticmethod
    def get_equivalent(beam_map, pixel_size):
        """
        Return a 2D+1 Gaussian for a beam map and given pixel size.

        The equivalent psf if circular with a FWHM equal to that determined
        from the beam map.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            A map of the beam of shape (ky, kx).
        pixel_size : Coordinate2D1
            The pixel size.

        Returns
        -------
        Gaussian2D1
        """
        psf = Gaussian2D1()
        psf.set_equivalent(beam_map, pixel_size)
        return psf

    def set_equivalent(self, beam_map, pixel_size):
        """
        Set Gaussian parameters from a given beam map and pixel size.

        The area and thus FWHM are determined from the input parameters.
        Therefore, only a circular Gaussian will be set with zero position
        angle.  The peak of the equivalent will always be set to 1.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            A map of the beam of shape (kz, ky, kx).
        pixel_size : Coordinate2D1

        Returns
        -------
        None
        """
        px, py, pz = pixel_size.x, pixel_size.y, pixel_size.z
        nz, ny, nx = beam_map.shape
        nxy = nx * ny

        # Need to normalize for xy
        xy_norm = beam_map.copy()
        xy_sum = np.nansum(abs(xy_norm.reshape(nz, nxy)), axis=-1)
        i_sum = np.zeros(nz, dtype=float)
        nzi = xy_sum != 0
        i_sum[nzi] = 1 / xy_sum[nzi]
        xy_norm *= i_sum[:, None, None]
        xy_map = np.mean(xy_norm, axis=0)
        xy_map /= np.max(xy_map)

        area = xy_map.sum() * px * py
        self.set_area(area)

        z_map = xy_sum / xy_sum.max()
        self.z.set_equivalent(z_map, pixel_size.z_coordinates)
