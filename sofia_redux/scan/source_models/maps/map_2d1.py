# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.stats import gaussian_fwhm_to_sigma
from copy import deepcopy
import numpy as np
from scipy.signal import fftconvolve
import warnings

from sofia_redux.scan.source_models.maps.image_2d1 import Image2D1
from sofia_redux.scan.source_models.beams.gaussian_1d import Gaussian1D
from sofia_redux.scan.coordinate_systems.coordinate import Coordinate
from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.source_models.beams.gaussian_2d1 import Gaussian2D1
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.grid.grid_2d1 import Grid2D1
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d1 import FlatGrid2D1
from sofia_redux.scan.coordinate_systems.projector.projector_2d import \
    Projector2D
from sofia_redux.scan.flags.flagged_array import FlaggedArray
from sofia_redux.scan.source_models.maps.map_2d import Map2D
from sofia_redux.scan.utilities import utils, numba_functions


__all__ = ['Map2D1']


class Map2D1(Map2D):

    def __init__(self, data=None, blanking_value=np.nan, dtype=float,
                 shape=None, unit=None):
        """
        Initialize a Map2D1 instance.

        The 2D1 map is an extension of the :class:`Map2D` with the ability to
        handle one extra orthogonal dimension, typically spectral in nature
        and referred to as z.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to initialize the flagged array with.  If supplied, sets the
            shape of the array.  Note that the data type will be set to that
            defined by the `dtype` parameter.
        blanking_value : int or float, optional
            The blanking value defines invalid values in the data array.  This
            is the equivalent of defining a NaN value.
        dtype : type, optional
            The data type of the data array.
        shape : tuple (int), optional
            The shape of the data array.  This will only be relevant if
            `data` is not defined.
        unit : str or units.Unit or units.Quantity, optional
            The data unit.
        """
        super().__init__(blanking_value=blanking_value, dtype=dtype,
                         data=data, shape=shape, unit=unit)
        self.grid = FlatGrid2D1()
        self.reuse_index = Coordinate2D1()
        self.filter_fwhm = Coordinate2D1(xy_unit='degree')
        self.correcting_fwhm = Coordinate2D1(xy_unit='degree')
        self.filter_fwhm.set_singular()
        self.filter_fwhm.nan()
        self.correcting_fwhm.set_singular()
        self.correcting_fwhm.nan()
        self.z_display_grid_unit = None

    def copy(self, with_contents=True):
        """
        Return a copy of the map.

        Returns
        -------
        Map2D1
        """
        return super().copy(with_contents=with_contents)

    def size_x(self):
        """
        Return the size of the image in the x-direction.

        Returns
        -------
        int
        """
        return int(self.shape[2])

    def size_y(self):
        """
        Return the size of the image in the y-direction.

        Returns
        -------
        int
        """
        return int(self.shape[1])

    def size_z(self):
        """
        Return the size of the image in the z-direction.

        Returns
        -------
        z : int
        """
        return int(self.shape[0])

    @classmethod
    def default_beam(cls):
        """
        Return the default beam for this map.

        Returns
        -------
        Gaussian2D1
        """
        return Gaussian2D1

    @classmethod
    def numpy_to_fits(cls, coordinates):
        """
        Convert numpy based (z, y, x) coordinates/indices to FITS coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or Iterable or Coordinate2D1

        Returns
        -------
        Coordinate2D1
        """

        xy_coordinates = super().numpy_to_fits(coordinates)
        new = Coordinate2D1(xy=xy_coordinates)
        if len(coordinates) == 3:
            new.z = coordinates[0]
        return new

    @classmethod
    def fits_to_numpy(cls, coordinates):
        """
        Convert FITS based (x, y, z) coordinates/indices to numpy (z, y, x).

        Reverses the dimensional ordering so that (x, y, z) coordinates are
        returned as (z, y, x) coordinates.  Note that boolean arrays remain
        unaltered, since these usually indicate masking arrays.

        Parameters
        ----------
        coordinates : numpy.ndarray or Coordinate or iterable

        Returns
        -------
        numpy_coordinates : numpy.ndarray (int or float) or list
            A list will be returned if a Coordinate2D1 is passed into
            `coordinates`.
        """
        if isinstance(coordinates, Coordinate2D1):
            coordinates = [coordinates.x, coordinates.y, coordinates.z]
        return super().fits_to_numpy(coordinates)

    @property
    def pixel_volume(self):
        """
        Return the pixel volume over the 2 and 1-D grids.

        Returns
        -------
        volume : units.Quantity or float
        """
        z_width = self.grid.z.resolution
        if isinstance(z_width, Coordinate1D):
            z_width = z_width.x

        return self.pixel_area * z_width

    @property
    def reference(self):
        """
        Return the reference position of the grid.

        Returns
        -------
        Coordinate2D1
        """
        return self.grid.reference

    @reference.setter
    def reference(self, value):
        """
        Set the reference position of the grid.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        self.grid.set_reference(value)

    @property
    def reference_index(self):
        """
        Return the reference index of the grids.

        Returns
        -------
        index : Coordinate2D1
        """
        return self.grid.reference_index

    @reference_index.setter
    def reference_index(self, value):
        """
        Set the reference index of the grid.

        Parameters
        ----------
        value : Coordinate2D1

        Returns
        -------
        None
        """
        self.grid.set_reference_index(value)

    def reset_filtering(self):
        """
        Reset the map filtering.

        Returns
        -------
        None
        """
        self.correcting_fwhm.nan()
        self.filter_fwhm.nan()
        self.set_filter_blanking(np.nan)

    def is_filtered(self):
        """
        Return whether the map has been filtered.

        Returns
        -------
        filtered : bool
        """
        return not np.all(self.filter_fwhm.is_nan())

    def is_corrected(self):
        """
        Return whether the map has been corrected.

        Returns
        -------
        bool
        """
        return not np.all(self.correcting_fwhm.is_nan())

    def set_correcting_fwhm(self, fwhm):
        """
        Set the correcting FWHM.

        Parameters
        ----------
        fwhm : Coordinate2D1

        Returns
        -------
        None
        """
        self.correcting_fwhm = fwhm.copy()

    def set_filtering(self, fwhm):
        """
        Set the filtering FWHM.

        Parameters
        ----------
        fwhm : Coordinate2D1

        Returns
        -------
        None
        """
        self.filter_fwhm = fwhm.copy()

    def set_grid(self, grid):
        """
        Set the map grid.

        Parameters
        ----------
        grid : SphericalGrid2D1 or FlatGrid2D1

        Returns
        -------
        None
        """
        if self.smoothing_beam is not None and self.grid is not None:
            # Undo prior pixel smoothing, if any
            old_pixel_beam = self.get_pixel_smoothing().copy()
            if isinstance(self.smoothing_beam, Gaussian2D1):
                self.smoothing_beam.deconvolve_with(old_pixel_beam)

        self.grid = grid
        # Apply new pixel smoothing
        pixel_beam = self.get_pixel_smoothing().copy()

        if self.smoothing_beam is None or self.smoothing_beam.volume == 0:
            self.smoothing_beam = pixel_beam
        else:
            self.smoothing_beam.encompass(pixel_beam)

    def set_resolution(self, resolution, redo=False):
        """
        Set the resolution of the grid.

        Sets a new resolution for the map grid.  If the smoothing beam has
        already been determined and `redo` is `True`, it is first deconvolved
        by the original grid resolution before being encompassed by a smoothing
        beam determined from the new grid resolution.

        Parameters
        ----------
        resolution : Coordinate2D1
            An array of shape (2,) giving the (x, y) grid resolution.
        redo : bool, optional
            If `True` deconvolve with the smoothing beam, and convolve
            after the resolution is set.

        Returns
        -------
        None
        """
        if (isinstance(self.smoothing_beam, Gaussian2D1) and
                self.grid is not None and redo):
            self.smoothing_beam.deconvolve_with(self.get_pixel_smoothing())

        self.grid.set_resolution(resolution)
        if self.smoothing_beam is None:
            self.smoothing_beam = self.get_pixel_smoothing()
        else:
            self.smoothing_beam.encompass(self.get_pixel_smoothing())

    def set_underlying_beam(self, psf):
        """
        Set the underlying beam.

        Parameters
        ----------
        psf : Gaussian2D1 or Coordinate2D1
            A Gaussian PSF model, or a FWHM from which to create the model.

        Returns
        -------
        None
        """
        if isinstance(psf, Gaussian2D1):
            self.underlying_beam = psf.copy()
        else:
            self.underlying_beam = Gaussian2D1(
                x_fwhm=psf.x, y_fwhm=psf.y, z_fwhm=psf.z)

    def set_smoothing(self, psf):
        """
        Set the smoothing beam.

        Parameters
        ----------
        psf : Gaussian2D1 or Coordinate2D1
            A Gaussian PSF model, or a FWHM from which to create the model.

        Returns
        -------
        None
        """
        if isinstance(psf, Gaussian2D1):
            self.smoothing_beam = psf.copy()
        else:
            self.smoothing_beam = Gaussian2D1(
                x_fwhm=psf.x, y_fwhm=psf.y, z_fwhm=psf.z)

    def set_image(self, image):
        """
        Set the basis image.

        Parameters
        ----------
        image : Image2D1 or numpy.ndarray

        Returns
        -------
        None
        """
        if isinstance(image, np.ndarray):
            image = Image2D1(data=image,
                             blanking_value=self.blanking_value,
                             dtype=self.dtype)

        if image is not self.basis:
            self.set_basis(image)
        self.claim_image(image)

    def set_z_display_grid_unit(self, unit):
        """
        Set the grid display unit.

        The display grid unit defines the spatial units of the map.

        Parameters
        ----------
        unit : str or units.Unit or units.Quantity or None

        Returns
        -------
        None
        """
        if isinstance(unit, str):
            self.z_display_grid_unit = units.Unit(unit) * 1.0
        elif isinstance(unit, units.Unit):
            self.z_display_grid_unit = unit * 1.0
        elif isinstance(unit, units.Quantity):
            self.z_display_grid_unit = unit
        elif unit is None:
            pass
        else:
            raise ValueError(f"Unit must be {str}, {units.Unit}, "
                             f"{units.Quantity}, or {None}.")

    def get_z_display_grid_unit(self):
        """
        Return the display grid unit for the z-axis.

        Returns
        -------
        units.Quantity
        """
        if self.z_display_grid_unit is not None:
            return self.z_display_grid_unit
        return self.get_z_default_grid_unit()

    def get_z_default_grid_unit(self):
        """
        Return the default grid unit for the z-axis.

        Returns
        -------
        units.Quantity
        """
        if self.grid is None:
            return units.Unit(self.get_unit('pixel'))
        return self.grid.z.get_default_unit()

    def get_volume(self):
        """
        Return the total volume of the map.

        The total volume is the number of pixels multiplied by the pixel
        volume.

        Returns
        -------
        area : units.Quantity or float
        """
        return self.count_valid_points() * self.grid.get_pixel_volume()

    def get_image_beam(self):
        """
        Return the image beam.

        The image beam is the underlying beam convolved with the smoothing
        beam.  If one is not set, the other is returned.  If neither is set,
        `None` is returned.

        Returns
        -------
        Gaussian2D1 or None
        """
        return super().get_image_beam()

    def get_image_beam_volume(self):
        """
        Get the beam volume of the image beam.

        Returns
        -------
        float or units.Quantity
        """
        if self.underlying_beam is None:
            underlying_volume = 0.0
        else:
            underlying_volume = self.underlying_beam.volume

        if not isinstance(self.smoothing_beam, Gaussian2D1):
            smoothing_volume = 0.0
        else:
            smoothing_volume = self.smoothing_beam.volume

        return underlying_volume + smoothing_volume

    def get_filter_area(self):
        """
        Get the filtering beam area.

        Returns
        -------
        area : units.Quantity
        """
        if self.filter_fwhm is None:
            return 0.0 * units.Unit('degree') ** 2

        return Gaussian2D1.AREA_FACTOR * (
                self.filter_fwhm.x * self.filter_fwhm.y)

    def get_filter_volume(self):
        """
        Get the filtering beam area.

        Returns
        -------
        volume : units.Quantity
        """
        area = self.get_filter_area()
        size_1d = Gaussian1D.fwhm_to_size * self.filter_fwhm.z
        return area * size_1d

    def get_filter_correction_factor(self, underlying_fwhm=None):
        """
        Return the filter correction factor.

        The filtering correction factor is given as::

            factor = 1 / (1 - ((v1 - v2) / (v1 + v3)))

        where v1 is the underlying beam volume, v2 is the smoothing beam
        volume, and v3 is the filtering beam volume.

        Parameters
        ----------
        underlying_fwhm : Coordinate2D1, optional
             The underlying FWHM for (x, y, z).  If not supplied, defaults to
             the map underlying beam FWHM.

        Returns
        -------
        correction_factor : float
        """
        if np.any(self.filter_fwhm.is_nan()):
            return 1.0

        if underlying_fwhm is None:
            underlying_beam = self.underlying_beam
        else:
            underlying_beam = Gaussian2D1(x_fwhm=underlying_fwhm.x,
                                          y_fwhm=underlying_fwhm.y,
                                          z_fwhm=underlying_fwhm.z)

        # Get the appropriate unit for unavailable beam FWHMs
        xy_unit = z_unit = None
        for beam in [underlying_beam, self.smoothing_beam]:
            if beam is None:
                continue
            if xy_unit is None and isinstance(beam.x_fwhm, units.Quantity):
                xy_unit = beam.x_fwhm.unit
            if z_unit is None and isinstance(beam.z_fwhm, units.Quantity):
                z_unit = beam.z_fwhm.unit
            if z_unit is not None and xy_unit is not None:
                break

        if xy_unit is None:
            xy_unit = units.dimensionless_unscaled
        if z_unit is None:
            z_unit = units.dimensionless_unscaled

        v_unit = z_unit * (xy_unit ** 2)

        if underlying_beam is None:
            underlying_beam_volume = 0.0 * v_unit
        else:
            underlying_beam_volume = underlying_beam.volume

        if not isinstance(self.smoothing_beam, Gaussian2D1):
            smoothing_beam_volume = 0.0 * v_unit
        else:
            smoothing_beam_volume = self.smoothing_beam.volume

        if underlying_beam_volume == 0:
            denom = self.get_filter_volume()
        else:
            denom = underlying_beam_volume + self.get_filter_volume()
        if denom == 0:
            return 1.0
        factor = ((underlying_beam_volume - smoothing_beam_volume) / denom
                  ).decompose().value
        return 1.0 / (1.0 - factor)

    def get_pixel_smoothing(self):
        """
        Return a Gaussian model representing pixel smoothing.

        Returns
        -------
        Gaussian2D1
        """
        xy_factor = Gaussian2D1.FWHM_TO_SIZE
        z_factor = Gaussian1D.fwhm_to_size

        return Gaussian2D1(x_fwhm=self.grid.resolution.x / xy_factor,
                           y_fwhm=self.grid.resolution.y / xy_factor,
                           z_fwhm=self.grid.resolution.z / z_factor,
                           theta=0.0 * units.Unit('deg'))

    def get_resolution(self):
        """
        Return the grid resolution in (x, y, z).

        Returns
        -------
        resolution : Coordinate2D1
        """
        return self.grid.resolution

    def get_anti_aliasing_beam_image_for(self, map2d1):
        """
        Return an antialiasing beam image.

        The anti-aliasing beam is the beam representing the pixel smoothing for
        this map deconvolved by the smoothing beam of the input map.  A
        representation of this beam is returned by projecting it onto the grid
        of this map.  If there is no valid anti-aliasing beam, `None` is
        returned.

        Parameters
        ----------
        map2d1 : Map2D1

        Returns
        -------
        anti_aliasing_beam_image : numpy.ndarray (float) or None
        """
        return super().get_anti_aliasing_beam_image_for(map2d1)

    def get_anti_aliasing_beam_for(self, map2d1):
        """
        Return the anti-aliasing beam for a given Map2D1.

        The anti-aliasing beam is the beam representing the pixel smoothing for
        this map deconvolved by the input map smoothing beam.  If the smoothing
        beam of the input map encompasses the pixel smoothing of this map,
        `None` will be returned.

        Parameters
        ----------
        map2d1 : Map2D1

        Returns
        -------
        anti_aliasing_beam : Gaussian2D1 or None
        """
        return super().get_anti_aliasing_beam_for(map2d1)

    def get_index_transform_to(self, map2d1):
        """
        Return the grid indices of this map projected onto another.

        Parameters
        ----------
        map2d1 : Map2D1

        Returns
        -------
        indices : Coordinate2D1
            The indices of this map projected onto the grid indices of the
            input `map2d1`.  Note that these indices are flat and correspond to
            the flattened map data.
        """
        # Just the (y, x) coordinates in a single plane
        indices = self.numpy_to_fits(
            np.stack([x.ravel() for x in np.indices(self.shape[1:])]))

        offsets = self.grid.get_offset(indices.xy_coordinates)
        from_projector = Projector2D(self.grid.projection)
        to_projector = Projector2D(map2d1.grid.projection)
        from_projector.offset.copy_coordinates(offsets)
        from_projector.deproject()  # Convert to coordinates
        to_projector.coordinates.convert_from(from_projector.coordinates)
        to_projector.project()  # Convert to offsets, then onto indices

        xy_coordinates = to_projector.offset
        z1_indices = Coordinate1D(np.arange(self.size_z()))
        z1_coordinates = self.grid.z.coordinates_at(z1_indices)
        new_indices = Coordinate2D1(xy=xy_coordinates, z=z1_coordinates)
        map2d1.grid.offset_to_index(new_indices, in_place=True)
        return new_indices

    def get_info(self):
        """
        Return strings describing the map.

        Returns
        -------
        list of str
        """
        xy_unit = self.get_display_grid_unit()
        z_unit = self.get_z_display_grid_unit()

        if isinstance(xy_unit, units.Quantity):
            xy_scale = xy_unit.value
            xy_unit = xy_unit.unit
        else:
            xy_scale = 1.0
        if isinstance(z_unit, units.Quantity):
            z_scale = z_unit.value
            z_unit = z_unit.unit
        else:
            z_scale = 1.0

        pixel_size = self.grid.get_pixel_size()

        px = pixel_size.x / xy_scale
        py = pixel_size.y / xy_scale
        pz = pixel_size.z / z_scale
        if xy_unit is not None and isinstance(xy_unit, units.Unit):
            px = px.to(xy_unit).value
            py = py.to(xy_unit).value
            xy_unit_str = f' {xy_unit}'
        else:
            xy_unit_str = ''
        if z_unit is not None and isinstance(z_unit, units.Unit):
            pz = pz.to(z_unit).value
            z_unit_str = f' {z_unit}'
        else:
            z_unit_str = ''

        u_beam = self.underlying_beam
        u_fwhm = (0.0, 0.0) if u_beam is None else (u_beam.fwhm, u_beam.z_fwhm)
        i_beam = self.get_image_beam()
        i_fwhm = (0.0, 0.0) if i_beam is None else (i_beam.fwhm, u_beam.z_fwhm)

        info = ["Map information:",
                f"Image Size: {self.get_size_string()} pixels "
                f"({px:.3f} x {py:.3f}{xy_unit_str}) x {pz:.5f}{z_unit_str}.",
                self.grid.to_string(),
                f'Instrument PSF: {u_fwhm[0]:.5f}, {u_fwhm[1]:.5f} '
                f'(includes pixelization)',
                f'Image resolution: {i_fwhm[0]:.5f}, {i_fwhm[1]:.5f} '
                f'(includes smoothing)']
        return info

    def get_points_per_smoothing_beam(self):
        """
        Get the number of pixels per smoothing beam.

        Returns
        -------
        float
        """
        if self.smoothing_beam is None:
            return 1.0
        points = self.smoothing_beam.volume / self.grid.get_pixel_volume()
        points = points.decompose().value
        return np.clip(points, 1.0, None)

    def copy_properties_from(self, other):
        """
        Copy the properties from another map.

        The properties copied include anything under
        :func:`Map2D.copy_processing_from` along with the map grid and FITS
        properties.

        Parameters
        ----------
        other : Map2D1

        Returns
        -------
        None
        """
        super().copy_properties_from(other)
        self.z_display_grid_unit = deepcopy(other.z_display_grid_unit)

    def merge_properties_from(self, other):
        """
        Merge properties from another map.

        Merging the properties results in the encompassed smoothing beam of
        this map by the other, and the minimum filtering FWHM of both maps.

        Parameters
        ----------
        other : Map2D1

        Returns
        -------
        None
        """
        if other.smoothing_beam is not None:
            if self.smoothing_beam is not None:
                self.smoothing_beam.encompass(other.smoothing_beam)
            else:
                self.smoothing_beam = deepcopy(other.smoothing_beam)

        x = np.empty(2, dtype=float)
        y = x.copy()
        z = x.copy()
        if isinstance(self.filter_fwhm.x, units.Quantity):
            x = x * self.filter_fwhm.x.unit
            y = y * self.filter_fwhm.y.unit
        if isinstance(self.filter_fwhm.z, units.Quantity):
            z = z * self.filter_fwhm.z.unit
        x[...] = self.filter_fwhm.x, other.filter_fwhm.x
        y[...] = self.filter_fwhm.y, other.filter_fwhm.y
        z[...] = self.filter_fwhm.z, other.filter_fwhm.z
        c = Coordinate2D1([x, y, z])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.filter_fwhm.copy_coordinates(c.min)

    def add_smoothing(self, psf):
        """
        Convolve the smoothing beam with a PSF (Point Spread Function).

        Parameters
        ----------
        psf : Gaussian2D1 or Coordinate2D1
            A Gaussian model or an FWHM from which to create the model.

        Returns
        -------
        None
        """
        if isinstance(psf, Coordinate2D1):
            psf = Gaussian2D1(x_fwhm=psf.x, y_fwhm=psf.y, z_fwhm=psf.z)

        if self.smoothing_beam is None:
            self.smoothing_beam = psf.copy()
        else:
            self.smoothing_beam.convolve_with(psf)

    def filter_beam_correct(self):
        """
        Scale the map data by the filter correction factor.

        The filter correction factor is determined by applying
        :func:`Map2D.get_filter_correction_factor` to the underlying beam of
        the map.  All data are scaled by this factor.

        Returns
        -------
        None
        """
        beam = self.underlying_beam
        if not isinstance(beam, Gaussian2D1):
            beam = Coordinate2D1(xy_unit=self.get_display_grid_unit(),
                                 z_unit=self.get_z_display_grid_unit())
            beam.set_singular()
        else:
            beam = Coordinate2D1([beam.x_fwhm, beam.y_fwhm, beam.z_fwhm])

        self.filter_correct(beam)

    def undo_filter_correct(self, reference=None, valid=None):
        """
        Undo the last filter correction.

        Performs the reverse of :func:`Map2D.filter_correct` with the last
        used underlying FWHM.

        Parameters
        ----------
        reference : FlaggedArray or numpy.ndarray (float), optional
            The data set to determine valid data within the blanking range.
            Defaults to self.data.
        valid : numpy.ndarray (bool), optional
            `True` indicates a data element that may have the filter correction
            factor un-applied.

        Returns
        -------
        None
        """
        if not self.is_corrected():
            return

        if valid is None:
            if reference is None:
                reference = self
            if isinstance(reference, FlaggedArray):
                ref_data = reference.data
                ref_valid = reference.valid
            else:
                ref_data = reference
                ref_valid = True

            blanking_value = self.filter_blanking
            valid = ref_data <= blanking_value
            valid &= ref_data >= -blanking_value
            valid &= ref_valid

        last_correction_factor = self.get_filter_correction_factor(
            self.correcting_fwhm)
        self.data[valid] /= last_correction_factor
        self.correcting_fwhm.nan()

    def update_filtering(self, fwhm):
        """
        Update the filtering.

        The filtering is only updated when it is NaN or the provided filtering
        FWHM is less than that which is currently set.

        Parameters
        ----------
        fwhm : Coordinate2D1

        Returns
        -------
        None
        """
        if np.any(self.filter_fwhm.is_nan()):
            self.filter_fwhm.copy_coordinates(fwhm)
        else:
            min_x = min(self.filter_fwhm.x, fwhm.x)
            min_y = min(self.filter_fwhm.y, fwhm.y)
            min_z = min(self.filter_fwhm.z, fwhm.z)
            self.filter_fwhm.set([min_x, min_y, min_z])

    def parse_coordinate_info(self, header, alt=''):
        """
        Parse and apply the WCS information from a FITS header.

        This process sets the grid for the map based on the contents of a given
        FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
        alt : str, optional
            The alternate WCS transform to use.  This replaces the "a" part of
            the CTYPEia cards.

        Returns
        -------
        None
        """
        self.set_grid(Grid2D1.from_header(header, alt=alt))

    def parse_corrected_beam(self, header):
        """
        Parse the corrected beam from a FITS header.

        The correcting beam FWHM is taken from the CBMAJ, CBMIN,
        and CB1D keywords, or the CORRECTN keyword in the FITS header.  If
        neither is found, the default correcting FWHM defaults to NaN.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.CORRECTED_BEAM_FITS_ID}BMAJ'
        if major_fwhm_key in header:
            beam = Gaussian2D1()
            beam.parse_header(
                header=header,
                size_unit=self.get_default_grid_unit(),
                z_unit=self.get_z_default_grid_unit(),
                fits_id=self.CORRECTED_BEAM_FITS_ID)
            self.correcting_fwhm = Coordinate2D1(
                xy=[beam.x_fwhm, beam.y_fwhm, beam.z_fwhm])
        else:
            self.correcting_fwhm = Coordinate2D1(
                xy_unit=self.get_default_grid_unit(),
                z_unit=self.get_z_default_grid_unit())
            self.correcting_fwhm.set_singular()
            self.correcting_fwhm.nan()

    def parse_smoothing_beam(self, header):
        """
        Parse the smoothing beam from a FITS header.

        The smoothing beam is taken from the SBMAJ/SBMIN/SB1D keywords, or the
        SMOOTH/SMOOTHZ keyword in the FITS header.  The no keywords are
        found, the smoothing beam FWHM defaults the pixel smoothing.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.SMOOTHING_BEAM_FITS_ID}BMAJ'
        if major_fwhm_key in header:
            self.smoothing_beam = Gaussian2D1()
            self.smoothing_beam.parse_header(
                header=header,
                size_unit=self.get_default_grid_unit(),
                z_unit=self.get_z_default_grid_unit(),
                fits_id=self.SMOOTHING_BEAM_FITS_ID)
        else:
            xy_unit = self.get_display_grid_unit()
            z_unit = self.get_z_display_grid_unit()
            if xy_unit is None:  # pragma: no cover
                xy_unit = 'degree'
            if z_unit is None:  # pragma: no cover
                z_unit = units.dimensionless_unscaled

            xy_fwhm = utils.get_header_quantity(
                header, 'SMOOTH',
                default=0.0,
                default_unit=xy_unit).to(xy_unit)
            z_fwhm = utils.get_header_quantity(
                header, 'SMOOTHZ',
                default=0.0,
                default_unit=z_unit).to(z_unit)

            self.smoothing_beam = Gaussian2D1(
                x_fwhm=xy_fwhm, y_fwhm=xy_fwhm, z_fwhm=z_fwhm)

        xy_pixel_smoothing = np.sqrt(self.grid.get_pixel_area()
                                     / Gaussian2D1.AREA_FACTOR)
        if not isinstance(xy_pixel_smoothing, units.Quantity):
            xy_pixel_smoothing = (xy_pixel_smoothing *
                                  self.smoothing_beam.x_fwhm.unit)

        pix_z_size = self.grid.get_pixel_size_z()
        if isinstance(pix_z_size, Coordinate1D):
            pix_z_size = pix_z_size.x

        z_pixel_smoothing = (pix_z_size / Gaussian2D1.fwhm_to_size)
        if not isinstance(z_pixel_smoothing, units.Quantity):
            z_pixel_smoothing = z_pixel_smoothing * z_unit

        pixel_smoothing = Gaussian2D1(x_fwhm=xy_pixel_smoothing,
                                      y_fwhm=xy_pixel_smoothing,
                                      z_fwhm=z_pixel_smoothing)
        self.smoothing_beam.encompass(pixel_smoothing)

    def parse_filter_beam(self, header):
        """
        Parse the filtering beam from a FITS header.

        The correcting beam FWHM is taken from the XBMAJ, XBMIN, XB1D keywords,
        or the EXTFLTR/EXTFLTRZ keyword in the FITS header  If neither is
        found, the default filtering FWHM defaults to NaN.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.FILTER_BEAM_FITS_ID}BMAJ'
        if major_fwhm_key in header:
            beam = Gaussian2D1()
            beam.parse_header(
                header,
                size_unit=self.get_default_grid_unit(),
                z_unit=self.get_z_default_grid_unit(),
                fits_id=self.FILTER_BEAM_FITS_ID)
            self.filter_fwhm = Coordinate2D1(
                [beam.x_fwhm, beam.y_fwhm, beam.z_fwhm])
        else:
            xy_unit = self.get_display_grid_unit()
            z_unit = self.get_z_display_grid_unit()
            if xy_unit is None:  # pragma: no cover
                xy_unit = self.get_default_grid_unit()
                if xy_unit is None:
                    xy_unit = 'degree'
            elif isinstance(xy_unit, units.Quantity):
                xy_unit = xy_unit.unit

            if z_unit is None:  # pragma: no cover
                z_unit = self.get_z_default_grid_unit()
                if z_unit is None:
                    z_unit = units.dimensionless_unscaled
            elif isinstance(z_unit, units.Quantity):
                z_unit = z_unit.unit

            xy_fwhm = utils.get_header_quantity(
                header, 'EXTFLTR', default=np.nan,
                default_unit=xy_unit).to(xy_unit)
            z_fwhm = utils.get_header_quantity(
                header, 'EXTFLTRZ', default=np.nan,
                default_unit=z_unit).to(z_unit)

            self.filter_fwhm = Coordinate2D1([xy_fwhm, xy_fwhm, z_fwhm])

    def parse_underlying_beam(self, header):
        """
        Parse the underlying beam from a FITS header.

        Uses IBMAJ/IBMIN/IB1D if available, will then look to BMAJ/BMIN,B1D,
        will then
        look for BEAM/ZBEAM, and finally the old RESOLUTN/RESOLUTZ.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.UNDERLYING_BEAM_FITS_ID}BMAJ'
        self.underlying_beam = Gaussian2D1()

        xy_unit = self.get_display_grid_unit()
        z_unit = self.get_z_display_grid_unit()
        xy_fits_unit = self.get_default_grid_unit()
        z_fits_unit = self.get_z_default_grid_unit()

        if xy_unit is None:  # pragma: no cover
            xy_unit = units.dimensionless_unscaled
        elif isinstance(xy_unit, units.Quantity):
            xy_unit = xy_unit.unit
        if xy_fits_unit is None:  # pragma: no cover
            xy_fits_unit = xy_unit
        elif isinstance(xy_fits_unit, units.Quantity):  # pragma: no cover
            xy_fits_unit = xy_fits_unit.unit

        if z_unit is None:  # pragma: no cover
            z_unit = units.dimensionless_unscaled
        elif isinstance(z_unit, units.Quantity):
            z_unit = z_unit.unit
        if z_fits_unit is None:  # pragma: no cover
            z_fits_unit = z_unit
        elif isinstance(z_fits_unit, units.Quantity):  # pragma: no cover
            z_fits_unit = z_fits_unit.unit

        if major_fwhm_key in header:
            self.underlying_beam.parse_header(
                header, size_unit=xy_fits_unit, z_unit=z_fits_unit,
                fits_id=self.UNDERLYING_BEAM_FITS_ID)

        elif 'BEAM' in header:
            xy_fwhm = utils.get_header_quantity(
                header, 'BEAM', default=np.nan,
                default_unit=xy_fits_unit).to(xy_unit)
            z_fwhm = utils.get_header_quantity(
                header, 'BEAMZ', default=np.nan,
                default_unit=z_fits_unit).to(z_unit)
            self.underlying_beam = Gaussian2D1(
                x_fwhm=xy_fwhm, y_fwhm=xy_fwhm, z_fwhm=z_fwhm)

        elif 'BMAJ' in header:
            self.underlying_beam.parse_header(
                header, size_unit=xy_fits_unit, z_unit=z_fits_unit, fits_id='')
            self.underlying_beam.deconvolve_with(self.smoothing_beam)

        elif 'RESOLUTN' in header:
            xy_fwhm = utils.get_header_quantity(
                header, 'RESOLUTN', default=np.nan,
                default_unit=xy_fits_unit).to(xy_unit)
            z_fwhm = utils.get_header_quantity(
                header, 'RESOLUTZ', default=np.nan,
                default_unit=z_fits_unit).to(z_unit)

            if self.smoothing_beam is None:
                self.underlying_beam.set_xyz_fwhm(xy_fwhm, xy_fwhm, z_fwhm)

            elif xy_fwhm > self.smoothing_beam.major_fwhm:
                xy_fwhm = np.sqrt((xy_fwhm ** 2) - (
                    self.smoothing_beam.x_fwhm * self.smoothing_beam.y_fwhm))
                if z_fwhm > self.smoothing_beam.z_fwhm:
                    z_fwhm -= self.smoothing_beam.z_fwhm
                self.underlying_beam.set_xyz_fwhm(xy_fwhm, xy_fwhm, z_fwhm)

            else:
                xy_fwhm = 0.0 * xy_unit
                z_fwhm = 0.0 * z_unit
                self.underlying_beam.set_xyz_fwhm(xy_fwhm, xy_fwhm, z_fwhm)

        else:
            xy_fwhm = 0.0 * xy_unit
            z_fwhm = 0.0 * z_unit
            self.underlying_beam.set_xyz_fwhm(xy_fwhm, xy_fwhm, z_fwhm)

    def edit_header(self, header):
        """
        Edit a FITS header using information in the current map.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        self.edit_coordinate_info(header)
        psf = self.get_image_beam()
        xy_fits_unit = self.get_default_grid_unit()
        z_fits_unit = self.get_z_default_grid_unit()
        xy_unit = self.get_display_grid_unit()
        z_unit = self.get_z_display_grid_unit()
        if isinstance(xy_unit, units.Quantity):
            xy_unit = xy_unit.unit
        if isinstance(z_unit, units.Quantity):
            z_unit = z_unit.unit

        if psf is not None:
            psf.edit_header(header, fits_id='', beam_name='image',
                            size_unit=xy_fits_unit, z_unit=z_fits_unit)
            if psf.is_circular():
                fwhm = psf.fwhm.to(xy_unit).value
                header['RESOLUTN'] = (
                    fwhm, f'{{Deprecated}} Effective image '
                          f'FWHM ({xy_unit}).')
                z_fwhm = psf.z_fwhm.to(z_unit).value
                header['RESOLUTZ'] = (
                    z_fwhm,
                    f'{{Deprecated}} Effective Z axis FWHM ({z_unit}).')

        if self.underlying_beam is not None:
            self.underlying_beam.edit_header(
                header, fits_id=self.UNDERLYING_BEAM_FITS_ID,
                beam_name='instrument', size_unit=xy_fits_unit,
                z_unit=z_fits_unit)

        if self.smoothing_beam is not None:
            self.smoothing_beam.edit_header(
                header, fits_id=self.SMOOTHING_BEAM_FITS_ID,
                beam_name='smoothing', size_unit=xy_fits_unit,
                z_unit=z_fits_unit)
            if self.smoothing_beam.is_circular():
                fwhm = self.smoothing_beam.fwhm.to(xy_unit).value
                header['SMOOTH'] = (
                    fwhm, f'{{Deprecated}} FWHM ({xy_unit}) smoothing.')
                z_fwhm = self.smoothing_beam.z_fwhm.to(z_unit).value
                header['SMOOTHZ'] = (
                    z_fwhm,
                    f'{{Deprecated}} FWHM ({z_unit}) z axis smoothing.')

        if not np.any(self.filter_fwhm.is_nan()):
            filter_beam = Gaussian2D1(x_fwhm=self.filter_fwhm.x,
                                      y_fwhm=self.filter_fwhm.y,
                                      z_fwhm=self.filter_fwhm.z)
            filter_beam.edit_header(header, fits_id='X',
                                    beam_name='Extended Structure Filter',
                                    size_unit=xy_fits_unit,
                                    z_unit=z_fits_unit)

        if not np.any(self.correcting_fwhm.is_nan()):
            correction_beam = Gaussian2D1(x_fwhm=self.correcting_fwhm.x,
                                          y_fwhm=self.correcting_fwhm.y,
                                          z_fwhm=self.correcting_fwhm.z)
            correction_beam.edit_header(header, fits_id='C',
                                        beam_name='Peak Corrected',
                                        size_unit=xy_fits_unit,
                                        z_unit=z_fits_unit)

        header['SMTHRMS'] = True, 'Is the Noise (RMS) image smoothed?'
        self.skip_model_edit_header = True

        super().edit_header(header)

    def count_beams(self):
        """
        Return the number of beams in the map.

        Returns
        -------
        float
        """
        return (self.get_volume() / self.get_image_beam_volume()
                ).decompose().value

    def count_independent_points(self, volume):
        """
        Find the number of independent points in a given area.

        In 1-D at least 3 points per beam are needed to separate a positioned
        point source from an extended source.  Similarly, 27 points per beam
        are necessary for 2D+1.

        Parameters
        ----------
        volume : astropy.Quantity
            The area to consider.

        Returns
        -------
        points : int
        """
        if self.smoothing_beam is None:
            smoothing_volume = 0.0
        else:
            smoothing_volume = self.smoothing_beam.volume

        if self.is_filtered():
            volume_factor = (np.sqrt(2 * np.pi) * gaussian_fwhm_to_sigma) ** 3
            filter_volume = volume_factor * (
                self.filter_fwhm.x * self.filter_fwhm.y * self.filter_fwhm.z)

            eta = 1.0 - (smoothing_volume / filter_volume).decompose().value
        else:
            eta = 1.0

        beam_volume = self.get_image_beam_volume()
        pixel_volume = self.grid.get_pixel_volume()
        if smoothing_volume == 0:
            inverse_points_per_beam = 0.0
        else:
            inverse_points_per_beam = eta * min(
                27, smoothing_volume / pixel_volume)

        return int(np.ceil((1.0 + volume / beam_volume).decompose().value
                           * inverse_points_per_beam))

    def nearest_to_offset(self, offset):
        """
        Return the nearest map index for a given offset.

        Parameters
        ----------
        offset : Coordinate2D1 or tuple or numpy.ndarray or list.
            The spatial offset given as a coordinate or tuple of
            (x, y, z) offsets.

        Returns
        -------
        x_index, y_index, z_index : 3-tuple (int or numpy.ndarray)
            The x and y map indices.
        """
        if not isinstance(offset, Coordinate2D1):
            self.reuse_index.set(offset)
            offset = self.reuse_index

        ix, iy = utils.round_values(offset.xy_coordinates.coordinates)
        iz = utils.round_values(offset.z)
        return ix, iy, iz

    def convert_range_value_to_index(self, ranges):
        """
        Calculate the index range for a given value range.

        Converts a range of dimensional offsets to an appropriate range of
        map indices for use in cropping.

        Parameters
        ----------
        ranges : Coordinate2D1 or iterable
            A coordinate of shape (2,) containing the minimum and maximum
            ranges in each dimension.  If an iterable is provided, it should
            be of length 3 and be ordered as (x, y, z).

        Returns
        -------
        index_range : numpy.ndarray (int)
            The ranges as indices (integer values) on the grid.
        """
        ranges = Coordinate2D1(ranges, xy_unit=self.get_display_grid_unit(),
                               z_unit=self.get_z_display_grid_unit())

        index_range = np.asarray(self.nearest_to_offset(
            self.grid.offset_to_index(ranges)))

        if self.verbose:
            span = ranges.span
            if span.xy_unit is not None:
                distance_str = f'({span.x.value}x{span.y})x'
            else:  # pragma: no cover
                distance_str = f'({span.x}x{span.y})x'
            distance_str += f'{span.z}'

            log.debug(f'Will crop to {distance_str}.')

        return index_range

    def crop(self, ranges):
        """
        Crop the image data.

        Parameters
        ----------
        ranges : numpy.ndarray (int,) or Coordinate2D1
            The ranges to set crop the data to.  Should be of shape
            (n_dimensions, 2) where ranges[0, 0] would give the minimum crop
            limit for the first dimension and ranges[0, 1] would give the
            maximum crop limit for the first dimension.  In this case, the
            'first' dimension is in FITS format.  i.e., (x, y) for a 2-D image.
            If a Quantity is supplied this should contain the min and max
            grid values to clip to in each dimension.

        Returns
        -------
        None
        """
        if self.data is None or self.size == 0:
            return
        if isinstance(ranges, Coordinate2D1):
            ranges = self.convert_range_value_to_index(ranges)
        else:
            ranges = np.asarray(self.nearest_to_offset(ranges))

        self.get_image().crop(ranges)
        self.grid.reference_index.subtract(Coordinate2D1(ranges[:, 0]))

    def smooth_to(self, psf):
        """
        Smooth to a given PSF or FWHM.

        Parameters
        ----------
        psf : Gaussian2D1 or Coordinate2D1
            The beam or FWHM of the beam.

        Returns
        -------
        None
        """
        if isinstance(psf, Coordinate2D1):
            psf = Gaussian2D1(x_fwhm=psf.x, y_fwhm=psf.y, z_fwhm=psf.z)

        if self.smoothing_beam is not None:
            if self.smoothing_beam.is_encompassing(psf):
                return
            psf.deconvolve_with(self.smoothing_beam)

        self.smooth_with_psf(psf)

    def smooth_with_psf(self, psf):
        """
        Smooth with a given PSF or FWHM.

        Parameters
        ----------
        psf : Gaussian2D1 or Coordinate2D1

        Returns
        -------
        None
        """
        if isinstance(psf, Coordinate2D1):
            psf = Gaussian2D1(x_fwhm=psf.x, y_fwhm=psf.y, z_fwhm=psf.z)

        extent = psf.extent()
        pixels = self.grid.get_pixel_size().copy()
        pixels.scale(5)

        steps = [extent.x / pixels.x, extent.y / pixels.y, extent.z / pixels.z]
        for i, step in enumerate(steps):
            if isinstance(step, units.Quantity):
                steps[i] = step.decompose().value
        steps = utils.round_values(np.asarray(steps))
        beam_map = psf.get_beam_map(self.grid)
        self.fast_smooth(beam_map, steps)

    def smooth(self, beam_map, reference_index=None, weights=None):
        """
        Smooth the data with a given beam map kernel.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The beam map image kernel with which to smooth the map of shape
            (kz, ky, kx).
        reference_index : Coordinate2D1
            The reference pixel index of the kernel in (x, y, z).
        weights : numpy.ndarray (float)
            The map weights of shape (nz, ny, nx).

        Returns
        -------
        None
        """
        # Convert to an array
        if isinstance(reference_index, Coordinate2D1):
            reference_index = np.asarray([
                reference_index.x, reference_index.y, reference_index.z])
        elif isinstance(reference_index, Coordinate):
            reference_index = reference_index.coordinates.copy()

        super().smooth(beam_map, reference_index=reference_index,
                       weights=weights)

    def fast_smooth(self, beam_map, steps, reference_index=None, weights=None):
        """
        Smooth the data with a given beam map kernel using fast method.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The beam map image kernel with which to smooth the map of shape
            (kz, ky, kx).
        steps : numpy.ndarray (int)
            The fast step skips in (x, y, z).
        reference_index : Coordinate2D1 or numpy.ndarray, optional
            The reference pixel index of the kernel in (x, y, z).  The
            default is the beam map center ((kx-1)/2, (ky-1)/2, (kz-1)/2).
        weights : numpy.ndarray (float), optional
            The map weights of shape (nz, ny, nx).  The default is no
            weighting.

        Returns
        -------
        None
        """
        if isinstance(reference_index, Coordinate2D1):
            reference_index = np.asarray([
                reference_index.x, reference_index.y, reference_index.z])
        elif isinstance(reference_index, Coordinate):
            reference_index = reference_index.coordinates.copy()

        if isinstance(steps, Coordinate2D1):
            steps = np.asarray([steps.x, steps.y, steps.z])
        elif isinstance(steps, Coordinate):
            steps = steps.coordinates.copy()

        super().fast_smooth(beam_map, steps, reference_index=reference_index,
                            weights=weights)
        self.add_smoothing(
            Gaussian2D1.get_equivalent(beam_map, self.grid.resolution))

    def fft_filter_above(self, fwhm, valid=None, weight=None):
        """
        Filter the image with the supplied FWHM using the FFT method.

        Parameters
        ----------
        fwhm : Coordinate2D1
            The FWHM of the Gaussian with which to filter the image.
        valid : numpy.ndarray (bool), optional
            An optional mask where `False` excludes a map element from
            inclusion in the convolution and subtraction.
        weight : numpy.ndarray (float), optional
            An optional weighting array with the same shape as the map data.
            These should be the inverse variance values.

        Returns
        -------
        None
        """
        nz = numba_functions.pow2ceil(self.shape[0] * 2)
        ny = numba_functions.pow2ceil(self.shape[1] * 2)
        nx = numba_functions.pow2ceil(self.shape[2] * 2)
        if valid is None:
            valid = self.valid
        else:
            valid = valid & self.valid

        if weight is None:
            weight = np.ones(self.shape, dtype=float)
        elif isinstance(weight, FlaggedArray):
            weight = weight.data
        weight = weight * valid

        sum_weight = (weight[valid] ** 2).sum()
        n = valid.sum()
        rmsw = np.sqrt(sum_weight / n)
        if rmsw <= 0:
            return

        values = np.zeros(valid.shape, dtype=float)
        values[valid] = weight[valid] * self.data[valid]
        transformer = np.zeros((nz, ny, nx), dtype=float)
        transformer[:valid.shape[0], :valid.shape[1], :valid.shape[2]] = values

        z, y, x = np.indices(transformer.shape)
        x -= nx // 2 - 1
        y -= ny // 2 - 1
        z -= nz // 2 - 1

        sigma = fwhm.copy()
        sigma.scale(gaussian_fwhm_to_sigma)

        resolution = self.grid.resolution

        dx2 = (resolution.x / sigma.x) ** 2
        dy2 = (resolution.y / sigma.y) ** 2
        dz2 = (resolution.z / sigma.z) ** 2
        if isinstance(dx2, units.Quantity):
            dx2 = dx2.decompose().value
        if isinstance(dy2, units.Quantity):
            dy2 = dy2.decompose().value
        if isinstance(dz2, units.Quantity):
            dz2 = dz2.decompose().value

        a = -0.5 * np.asarray([dx2, dy2, dz2])
        g = np.exp(a[0] * x ** 2 +
                   a[1] * y ** 2 +
                   a[2] * z ** 2)

        c = fftconvolve(transformer, g / g.sum(), mode='same')
        c = c[:self.shape[0], :self.shape[1], :self.shape[2]]
        image = self.get_image()
        image.add(c, factor=-1 / rmsw)  # Subtract re-weighted
        self.update_filtering(fwhm)

    def resample(self, resolution):
        """
        Resample the map to a new resolution.

        Parameters
        ----------
        resolution : Coordinate2D1 or units.Quantity or iterable
            The new resolution for the map.

        Returns
        -------
        None
        """
        if not isinstance(resolution, Coordinate2D1):
            resolution = Coordinate2D1(resolution)

        original = self.copy()
        original.fits_properties = self.fits_properties.copy()

        r1 = self.grid.resolution
        scale_x = r1.x / resolution.x
        if isinstance(scale_x, units.Quantity):
            scale_x = scale_x.decompose().value
        scale_y = r1.y / resolution.y
        if isinstance(scale_y, units.Quantity):
            scale_y = scale_y.decompose().value
        scale_z = r1.z / resolution.z
        if isinstance(scale_z, units.Quantity):
            scale_z = scale_z.decompose().value

        scaling = np.asarray([scale_z, scale_y, scale_x])
        new_shape = np.ceil(np.asarray(self.shape) * scaling).astype(int)
        self.set_data_shape(new_shape)
        self.set_grid(original.grid.for_resolution(resolution))
        self.resample_from_map(original)
