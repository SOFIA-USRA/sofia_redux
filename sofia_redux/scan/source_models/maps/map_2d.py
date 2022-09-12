# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.stats import gaussian_fwhm_to_sigma
from copy import deepcopy
import numpy as np
from scipy.signal import fftconvolve
import warnings

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.coordinate_systems.grid.grid_2d import Grid2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.coordinate_systems.projector.projector_2d import \
    Projector2D
from sofia_redux.scan.source_models.maps.overlay import Overlay
from sofia_redux.scan.utilities import utils, numba_functions
from sofia_redux.scan.source_models.fits_properties.fits_properties import (
    FitsProperties)
from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.flags.flagged_array import FlaggedArray


__all__ = ['Map2D']


class Map2D(Overlay):

    UNDERLYING_BEAM_FITS_ID = 'I'
    SMOOTHING_BEAM_FITS_ID = 'S'
    CORRECTED_BEAM_FITS_ID = 'C'
    FILTER_BEAM_FITS_ID = 'X'

    def __init__(self, data=None, blanking_value=np.nan, dtype=float,
                 shape=None, unit=None):
        """
        Initialize a Map2D instance.

        The 2D map is an extension of the :class:`Image` and :class:`Overlay`
        classes.  In addition to the standard image handling methods, a grid
        is also included to allow for projections and deprojections of pixels
        to/from real coordinates.

        An underlying beam representing the observation PSF may also be
        modeled as well as a beam to represent any smoothing operations that
        may be applied.

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
        self.fits_properties = FitsProperties()
        self.grid = FlatGrid2D()
        self.display_grid_unit = None
        self.underlying_beam = None  # Gaussian2D
        self.smoothing_beam = None  # Gaussian2D
        self.filter_fwhm = np.nan * units.Unit('deg')
        self.correcting_fwhm = np.nan * units.Unit('deg')
        self.filter_blanking = np.inf
        self.reuse_index = Coordinate2D()
        self.skip_model_edit_header = False

        super().__init__(blanking_value=blanking_value, dtype=dtype,
                         data=data, shape=shape, unit=unit)
        self.set_image(self.basis)

    def copy(self, with_contents=True):
        """
        Return a copy of the map.

        Returns
        -------
        Map2D
        """
        return super().copy(with_contents=with_contents)

    def __eq__(self, other):
        """
        Check if this Map2D instance is functionally equivalent to another.

        Parameters
        ----------
        other : Map2D

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if other.__class__ != self.__class__:
            return False
        if self.fits_properties != other.fits_properties:
            return False

        if isinstance(self.correcting_fwhm, (Coordinate2D, Coordinate2D1)):
            if self.correcting_fwhm != other.correcting_fwhm:
                return False
        elif not np.isclose(self.correcting_fwhm, other.correcting_fwhm,
                            equal_nan=True):
            return False
        if isinstance(self.filter_fwhm, (Coordinate2D, Coordinate2D1)):
            if self.filter_fwhm != other.filter_fwhm:
                return False
        elif not np.isclose(self.filter_fwhm, other.filter_fwhm,
                            equal_nan=True):
            return False
        if self.grid != other.grid:
            return False
        if self.display_grid_unit != other.display_grid_unit:
            return False
        if self.underlying_beam != other.underlying_beam:
            return False
        if self.smoothing_beam != other.smoothing_beam:
            return False
        return super().__eq__(other)

    @classmethod
    def default_beam(cls):
        """
        Return the default beam for this map.

        Returns
        -------
        Gaussian2D
        """
        return Gaussian2D

    @classmethod
    def numpy_to_fits(cls, coordinates):
        """
        Convert numpy based (x, y) coordinates/indices to FITS coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray

        Returns
        -------
        Coordinate2D
        """
        coordinates = super().numpy_to_fits(coordinates)
        return Coordinate2D(coordinates)

    @property
    def pixel_area(self):
        """
        Return the pixel area of the grid.

        Returns
        -------
        area : units.Quantity or float
        """
        return self.grid.get_pixel_area()

    @property
    def reference(self):
        """
        Return the reference position of the grid.

        Returns
        -------
        Coordinate2D
        """
        return self.grid.reference

    @reference.setter
    def reference(self, value):
        """
        Set the reference position of the grid.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self.grid.set_reference(value)

    @property
    def reference_index(self):
        """
        Return the reference index of the grid.

        Returns
        -------
        index : Coordinate2D
        """
        return self.grid.reference_index

    @reference_index.setter
    def reference_index(self, value):
        """
        Set the reference index of the grid.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self.grid.set_reference_index(value)

    @property
    def projection(self):
        """
        Return the grid projection.

        The grid projection transforms coordinates to map offsets.

        Returns
        -------
        Projection2D
        """
        return self.grid.projection

    @projection.setter
    def projection(self, value):
        """
        Set the grid projection.

        Parameters
        ----------
        value : Projection2D

        Returns
        -------
        None
        """
        self.grid.set_projection(value)

    def reset_processing(self):
        """
        Reset the processing status.

        Sets the smoothing beam to be consistent with the current grid, and
        removes all filtering and correcting FWHM parameters.

        Returns
        -------
        None
        """
        if self.fits_properties is not None:
            self.fits_properties.reset_processing()
        self.reset_smoothing()
        self.reset_filtering()

    def reset_smoothing(self):
        """
        Reset the map smoothing.

        Returns
        -------
        None
        """
        self.set_pixel_smoothing()

    def reset_filtering(self):
        """
        Reset the map filtering.

        Returns
        -------
        None
        """
        deg = units.Unit('deg')
        self.set_filtering(np.nan * deg)
        self.set_correcting_fwhm(np.nan * deg)
        self.set_filter_blanking(np.nan)

    def renew(self):
        """
        Renew the map by clearing all processing and data.

        Returns
        -------
        None
        """
        self.reset_processing()
        self.clear()

    def add_proprietary_units(self):
        """
        Add proprietary units to the local units.

        Returns
        -------
        None
        """
        self.add_local_unit(np.nan * units.Unit('beam'),
                            alternate_names=['beam', 'BEAM', 'Beam', 'bm',
                                             'BM', 'Bm'])
        self.add_local_unit(self.grid.get_pixel_area() * units.Unit('pixel'),
                            alternate_names=['pixel', 'PIXEL', 'Pixel',
                                             'PIXELS', 'Pixels', 'pxl', 'PXL',
                                             'Pxl'])

    def no_data(self):
        """
        Discard all data.

        Returns
        -------
        None
        """
        self.discard()

    def is_filtered(self):
        """
        Return whether the map has been filtered.

        Returns
        -------
        filtered : bool
        """
        return not np.isnan(self.filter_fwhm)

    def is_corrected(self):
        """
        Return whether the map has been corrected.

        Returns
        -------
        bool
        """
        return not np.isnan(self.correcting_fwhm)

    def is_filter_blanked(self):
        """
        Return whether the map is filter blanked.

        Returns
        -------
        bool
        """
        return np.isfinite(self.filter_blanking)

    def set_correcting_fwhm(self, fwhm):
        """
        Set the correcting FWHM.

        Parameters
        ----------
        fwhm : units.Quantity

        Returns
        -------
        None
        """
        self.correcting_fwhm = fwhm

    def set_default_unit(self):
        """
        Set the default unit for the map data.

        Returns
        -------
        None
        """
        self.add_proprietary_units()
        super().set_default_unit()

    def set_unit(self, unit):
        """
        Set the map data unit.

        Parameters
        ----------
        unit : astropy.units.Quantity or str or astropy.units.Unit
            The unit to set as the map data unit.  Should be a quantity
            (value and unit type).  If a string or Unit is supplied, the
            map data unit will be set to the value located in the local_units
            dictionary.  If no such value exists, a KeyError will be raised.

        Returns
        -------
        None
        """
        super().set_unit(unit)
        image = self.get_image()
        if image is not None:
            image.set_unit(unit)

    def set_grid(self, grid):
        """
        Set the map grid.

        Parameters
        ----------
        grid : SphericalGrid

        Returns
        -------
        None
        """
        if self.smoothing_beam is not None and self.grid is not None:
            # Undo prior pixel smoothing, if any
            old_pixel_beam = self.get_pixel_smoothing().copy()
            self.smoothing_beam = utils.deconvolve_beam(
                self.smoothing_beam, old_pixel_beam)

        self.grid = grid
        # Apply new pixel smoothing
        pixel_beam = self.get_pixel_smoothing().copy()

        if self.smoothing_beam is None or self.smoothing_beam.area == 0:
            self.smoothing_beam = pixel_beam
        else:
            self.smoothing_beam = utils.encompass_beam(
                self.smoothing_beam, pixel_beam)

    def set_resolution(self, resolution, redo=False):
        """
        Set the resolution of the grid.

        Sets a new resolution for the map grid.  If the smoothing beam has
        already been determined and `redo` is `True`, it is first deconvolved
        by the original grid resolution before being encompassed by a smoothing
        beam determined from the new grid resolution.

        Parameters
        ----------
        resolution : float or numpy.ndarray or units.Quantity or Coordinate2D
            An array of shape (2,) giving the (x, y) grid resolution.
        redo : bool, optional
            If `True` deconvolve with the smoothing beam, and convolve
            after the resolution is set.

        Returns
        -------
        None
        """
        if self.smoothing_beam is not None and self.grid is not None and redo:
            self.smoothing_beam = utils.deconvolve_beam(
                self.smoothing_beam, self.get_pixel_smoothing())

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
        psf : Gaussian2D or units.Quantity
            A Gaussian PSF model, or a FWHM from which to create the model.

        Returns
        -------
        None
        """
        if isinstance(psf, Gaussian2D):
            self.underlying_beam = psf.copy()
        else:
            self.underlying_beam = Gaussian2D(x_fwhm=psf, y_fwhm=psf)

    def set_pixel_smoothing(self):
        """
        Set the smoothing beam to the pixel smoothing beam.

        Returns
        -------
        None
        """
        self.smoothing_beam = self.get_pixel_smoothing().copy()

    def set_smoothing(self, psf):
        """
        Set the smoothing beam.

        Parameters
        ----------
        psf : Gaussian2D or units.Quantity
            A Gaussian PSF model, or a FWHM from which to create the model.

        Returns
        -------
        None
        """
        if isinstance(psf, Gaussian2D):
            self.smoothing_beam = psf.copy()
        else:
            self.smoothing_beam = Gaussian2D(x_fwhm=psf, y_fwhm=psf)

    def set_filtering(self, fwhm):
        """
        Set the filtering FWHM.

        Parameters
        ----------
        fwhm : .units.Quantity

        Returns
        -------
        None
        """
        self.filter_fwhm = fwhm

    def set_filter_blanking(self, value):
        """
        Set the filter blanking.

        Parameters
        ----------
        value : float or None
            The value to set for filter blanking.  If `None` is supplied, the
            filter blanking value is set to NaN (disabled).

        Returns
        -------
        None
        """
        if value is None:
            self.filter_blanking = np.nan
        else:
            self.filter_blanking = float(value)

    def set_image(self, image):
        """
        Set the basis image.

        Parameters
        ----------
        image : Image2D or numpy.ndarray

        Returns
        -------
        None
        """
        if isinstance(image, np.ndarray):
            image = Image2D(data=image,
                            blanking_value=self.blanking_value,
                            dtype=self.dtype)

        if image is not self.basis:
            self.set_basis(image)
        self.claim_image(image)

    def set_display_grid_unit(self, unit):
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
            self.display_grid_unit = units.Unit(unit) * 1.0
        elif isinstance(unit, units.Unit):
            self.display_grid_unit = unit * 1.0
        elif isinstance(unit, units.Quantity):
            self.display_grid_unit = unit
        elif unit is None:
            pass
        else:
            raise ValueError(f"Unit must be {str}, {units.Unit}, "
                             f"{units.Quantity}, or {None}.")

    def get_area(self):
        """
        Return the total area of the map.

        The total area is the number of pixels multiplied by the pixel area.

        Returns
        -------
        area : units.Quantity or float
        """
        return self.count_valid_points() * self.grid.get_pixel_area()

    def get_display_grid_unit(self):
        """
        Return the display grid unit.

        Returns
        -------
        units.Quantity
        """
        if self.display_grid_unit is not None:
            return self.display_grid_unit
        return self.get_default_grid_unit()

    def get_default_grid_unit(self):
        """
        Return the default grid unit.

        Returns
        -------
        units.Quantity
        """
        if self.grid is None:
            return self.get_unit('pixel')
        return self.grid.get_default_unit()

    def get_image(self, dtype=None, blanking_value=None):
        """
        Return the basis image.

        Parameters
        ----------
        dtype : type, optional
            The image data type.
        blanking_value : int or float, optional
            The new image blanking value.

        Returns
        -------
        Image2D
        """
        if dtype is not None or blanking_value is not None:
            log.warning("Cannot change base image type or blanking value from "
                        "Map2D.")
        return self.basis

    def get_image_beam(self):
        """
        Return the image beam.

        The image beam is the underlying beam convolved with the smoothing
        beam.  If one is not set, the other is returned.  If neither is set,
        `None` is returned.

        Returns
        -------
        Gaussian2D or None
        """
        if self.underlying_beam is None and self.smoothing_beam is None:
            return None
        elif self.underlying_beam is None and self.smoothing_beam is not None:
            return self.smoothing_beam.copy()
        elif self.smoothing_beam is None and self.underlying_beam is not None:
            return self.underlying_beam.copy()

        beam = self.underlying_beam.copy()
        beam.convolve_with(self.smoothing_beam)
        return beam

    def get_image_beam_area(self):
        """
        Get the beam area of the image beam.

        Returns
        -------
        float or units.Quantity
        """
        if self.underlying_beam is None:
            underlying_area = 0.0
        else:
            underlying_area = self.underlying_beam.area

        if self.smoothing_beam is None:
            smoothing_area = 0.0
        else:
            smoothing_area = self.smoothing_beam.area

        return underlying_area + smoothing_area

    def get_filter_area(self):
        """
        Get the filtering beam area.

        Returns
        -------
        area : units.Quantity
        """
        if self.filter_fwhm is None:
            return 0.0 * units.Unit('degree') ** 2

        return Gaussian2D.AREA_FACTOR * (self.filter_fwhm ** 2)

    def get_filter_correction_factor(self, underlying_fwhm=None):
        """
        Return the filter correction factor.

        The filtering correction factor is given as::

            factor = 1 / (1 - ((a1 - a2) / (a1 + a3)))

        where a1 is the underlying beam area, a2 is the smoothing beam area,
        and a3 is the filtering beam area.

        Parameters
        ----------
        underlying_fwhm : astropy.units.Quantity or float, optional
             The underlying FWHM.  If not supplied, defaults to the map
             underlying beam FWHM.

        Returns
        -------
        correction_factor : float
        """
        if np.isnan(self.filter_fwhm):
            return 1.0

        if underlying_fwhm is None:
            underlying_beam = self.underlying_beam
        else:
            underlying_beam = Gaussian2D(x_fwhm=underlying_fwhm,
                                         y_fwhm=underlying_fwhm)

        # Get the appropriate unit for unavailable beam FWHMs
        for beam in [underlying_beam, self.smoothing_beam]:
            if beam is not None and isinstance(beam.x_fwhm, units.Quantity):
                unit = beam.x_fwhm.unit
                break
        else:
            if isinstance(self.filter_fwhm, units.Quantity):
                unit = self.filter_fwhm.unit
            else:
                unit = units.dimensionless_unscaled
        u2 = unit ** 2

        if underlying_beam is None:
            underlying_beam_area = 0.0 * u2
        else:
            underlying_beam_area = underlying_beam.area

        if self.smoothing_beam is None:
            smoothing_beam_area = 0.0 * u2
        else:
            smoothing_beam_area = self.smoothing_beam.area

        denominator = underlying_beam_area + self.get_filter_area()
        if denominator == 0:
            return 1.0
        factor = ((underlying_beam_area - smoothing_beam_area) / denominator
                  ).decompose().value
        return 1.0 / (1.0 - factor)

    def get_pixel_smoothing(self):
        """
        Return a Gaussian model representing pixel smoothing.

        Returns
        -------
        Gaussian2D
        """
        factor = Gaussian2D.FWHM_TO_SIZE

        return Gaussian2D(x_fwhm=self.grid.resolution.x / factor,
                          y_fwhm=self.grid.resolution.y / factor,
                          theta=0.0 * units.Unit('deg'))

    def get_resolution(self):
        """
        Return the grid resolution in (x, y).

        Returns
        -------
        resolution : Coordinate2D
        """
        return self.grid.resolution

    def get_anti_aliasing_beam_for(self, map2d):
        """
        Return the anti-aliasing beam for a given Map2D.

        The anti-aliasing beam is the beam representing the pixel smoothing for
        this map deconvolved by the input map smoothing beam.  If the smoothing
        beam of the input map encompasses the pixel smoothing of this map,
        `None` will be returned.

        Parameters
        ----------
        map2d : Map2D

        Returns
        -------
        anti_aliasing_beam : Gaussian2D or None
        """
        map_smoothing = map2d.smoothing_beam
        pixelization = self.get_pixel_smoothing()
        if map_smoothing is None:
            return pixelization
        elif map_smoothing.is_encompassing(pixelization):
            return None
        antialias = pixelization.copy()
        antialias.deconvolve_with(map_smoothing)
        return antialias

    def get_anti_aliasing_beam_image_for(self, map2d):
        """
        Return an antialiasing beam image.

        The anti-aliasing beam is the beam representing the pixel smoothing for
        this map deconvolved by the smoothing beam of the input map.  A
        representation of this beam is returned by projecting it onto the grid
        of this map.  If there is no valid anti-aliasing beam, `None` is
        returned.

        Parameters
        ----------
        map2d : Map2D

        Returns
        -------
        anti_aliasing_beam_image : numpy.ndarray (float) or None
        """
        antialias = self.get_anti_aliasing_beam_for(map2d)
        if antialias is None:
            return None
        return antialias.get_beam_map(self.grid)

    def get_index_transform_to(self, map2d):
        """
        Return the grid indices of this map projected onto another.

        Parameters
        ----------
        map2d : Map2D

        Returns
        -------
        indices : Coordinate2D
            The indices of this map projected onto the grid indices of the
            input `map2d`.  Note that these indices are flat and correspond to
            the flattened map data.
        """
        indices = self.numpy_to_fits(np.stack([x.ravel() for x in np.indices(
            self.shape)]))
        offsets = self.grid.get_offset(indices)
        from_projector = Projector2D(self.grid.projection)
        to_projector = Projector2D(map2d.grid.projection)
        from_projector.offset.copy_coordinates(offsets)
        from_projector.deproject()  # Convert to coordinates
        to_projector.coordinates.convert_from(from_projector.coordinates)
        to_projector.project()  # Convert to offsets, then onto indices
        new_indices = map2d.grid.offset_to_index(to_projector.offset)
        return new_indices

    def get_table_entry(self, name):
        """
        Return a parameter value for a given name.

        Parameters
        ----------
        name : str, optional

        Returns
        -------
        value
        """
        if name == 'beams':
            return self.count_beams()
        elif name == 'min':
            return self.min() / self.unit.value
        elif name == 'max':
            return self.max() / self.unit.value
        elif name == 'unit':
            return str(self.unit.unit)
        elif name == 'mean':
            return self.mean()[0] / self.unit.value
        elif name == 'median':
            return self.median()[0] / self.unit.value
        elif name == 'rms':
            return self.rms(robust=True) / self.unit.value
        else:
            return self.fits_properties.get_table_entry(name)

    def get_info(self):
        """
        Return strings describing the map.

        Returns
        -------
        list of str
        """
        size_unit = self.get_display_grid_unit()
        pxy = self.grid.get_pixel_size().coordinates
        if size_unit is not None:
            if isinstance(size_unit, units.UnitBase):
                size_value = 1.0
            else:
                size_value, size_unit = size_unit.value, size_unit.unit
            px, py = (pxy.to(size_unit) / size_value).value
            unit_str = f' {size_unit}'
        elif (isinstance(pxy, units.Quantity) and
              pxy.unit != units.dimensionless_unscaled):  # pragma: no cover
            unit_str = f' {pxy.unit}'
            px, py = pxy.value
        else:  # pragma: no cover
            px, py = pxy
            unit_str = ''

        u_beam = self.underlying_beam
        u_fwhm = 0.0 if u_beam is None else u_beam.fwhm
        i_beam = self.get_image_beam()
        i_fwhm = 0.0 if i_beam is None else i_beam.fwhm

        info = ["Map information:",
                f"Image Size: {self.get_size_string()} pixels "
                f"({px} x {py}{unit_str}).",
                self.grid.to_string(),
                f'Instrument PSF: {u_fwhm:.5f} '
                f'(includes pixelization)',
                f'Image resolution: {i_fwhm:.5f} '
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
        points = self.smoothing_beam.area / self.grid.get_pixel_area()
        points = points.decompose().value
        return np.clip(points, 1.0, None)

    def copy_processing_from(self, other):
        """
        Copy the processing from another map.

        Processing includes the underlying and smoothing beams, the filtering
        FWHM, filtering blanking value and correcting FWHM.

        Parameters
        ----------
        other : Map2D

        Returns
        -------
        None
        """
        self.underlying_beam = deepcopy(other.underlying_beam)
        self.smoothing_beam = deepcopy(other.smoothing_beam)
        self.filter_fwhm = other.filter_fwhm
        self.filter_blanking = other.filter_blanking
        self.correcting_fwhm = other.correcting_fwhm

    def copy_properties_from(self, other):
        """
        Copy the properties from another map.

        The properties copied include anything under
        :func:`Map2D.copy_processing_from` along with the map grid and FITS
        properties.

        Parameters
        ----------
        other : Map2D

        Returns
        -------
        None
        """
        if other.fits_properties is None:
            self.fits_properties = None
        else:
            self.fits_properties = other.fits_properties.copy()

        self.copy_processing_from(other)
        self.filter_fwhm = other.filter_fwhm
        self.correcting_fwhm = other.correcting_fwhm
        self.filter_blanking = other.filter_blanking
        if other.grid is not None:
            self.grid = other.grid.copy()
        self.display_grid_unit = deepcopy(other.display_grid_unit)
        self.underlying_beam = deepcopy(other.underlying_beam)
        self.smoothing_beam = deepcopy(other.smoothing_beam)

    def merge_properties_from(self, other):
        """
        Merge properties from another map.

        Merging the properties results in the encompassed smoothing beam of
        this map by the other, and the minimum filtering FWHM of both maps.

        Parameters
        ----------
        other : Map2D

        Returns
        -------
        None
        """
        if other.smoothing_beam is not None:
            if self.smoothing_beam is not None:
                self.smoothing_beam = utils.encompass_beam(
                    self.smoothing_beam, other.smoothing_beam)
            else:
                self.smoothing_beam = deepcopy(other.smoothing_beam)

        unit = self.filter_fwhm.unit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.filter_fwhm = np.nanmin([self.filter_fwhm.to(unit).value,
                                          other.filter_fwhm.to(unit).value]
                                         ) * unit

    def add_smoothing(self, psf):
        """
        Convolve the smoothing beam with a PSF (Point Spread Function).

        Parameters
        ----------
        psf : Gaussian2D or astropy.units.Quantity
            A Gaussian model or an FWHM from which to create the model.

        Returns
        -------
        None
        """
        if isinstance(psf, units.Quantity):
            psf = Gaussian2D(x_fwhm=psf, y_fwhm=psf)

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
        if isinstance(beam, Gaussian2D):
            fwhm = beam.get_circular_equivalent_fwhm()
        else:
            fwhm = 0.0 * units.Unit('degree')

        self.filter_correct(fwhm)

    def filter_correct(self, underlying_fwhm, reference=None, valid=None):
        """
        Apply a filtering correction factor to the map data.

        The filtering correction factor is derived from the supplied
        `underlying_fwhm` using :func:`Map2D.get_filter_correction_factor`, and
        is only applied to valid data elements.  The validity of the elements
        may be supplied or determined from the map itself or a supplied
        `reference`.

        If a correction factor was previously applied, it will be removed
        before applying the new factor.

        Parameters
        ----------
        underlying_fwhm : units.Quantity or Coordinate
            The underlying FWHM of the beam used to derive the filtering
            correction factor to apply to the data.
        reference : FlaggedArray or numpy.ndarray, optional
            If `valid` is not supplied, calculate the validity mask from this
            reference array and whether those values are within the blanking
            value range, and are themselves valid (if the reference is a
            FlaggedArray).  Only valid data will be scaled by the filtering
            correction factor.
        valid : numpy.ndarray (bool), optional
            A boolean mask with the same shape as the map data.  Elements
            flagged as `False` will not be scaled by the filtering correction
            factor.  If not supplied, the validity mask will be derived from
            this map, or a supplied `reference` array.

        Returns
        -------
        None
        """
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

        if self.is_corrected():
            if underlying_fwhm == self.correcting_fwhm:
                return
            self.undo_filter_correct(valid=valid)

        self.data[valid] *= self.get_filter_correction_factor(underlying_fwhm)
        self.set_correcting_fwhm(underlying_fwhm)

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
        self.set_correcting_fwhm(np.nan * units.Unit('degree'))

    def update_filtering(self, fwhm):
        """
        Update the filtering.

        The filtering is only updated when it is NaN or the provided filtering
        FWHM is less than that which is currently set.

        Parameters
        ----------
        fwhm : units.Quantity

        Returns
        -------
        None
        """
        if np.isnan(self.filter_fwhm):
            self.filter_fwhm = fwhm
        else:
            self.filter_fwhm = min(self.filter_fwhm, fwhm)

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
        self.set_grid(Grid2D.from_header(header, alt=alt))

    def parse_corrected_beam(self, header):
        """
        Parse the corrected beam from a FITS header.

        The correcting beam FWHM is taken from the CBMAJ and CBMIN keywords,
        or the CORRECTN keyword in the FITS header.  If neither is found, the
        default correcting FWHM defaults to NaN.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.CORRECTED_BEAM_FITS_ID}BMAJ'
        if major_fwhm_key in header:
            beam = Gaussian2D()
            beam.parse_header(
                header=header,
                size_unit=self.get_default_grid_unit(),
                fits_id=self.CORRECTED_BEAM_FITS_ID)
            self.correcting_fwhm = beam.fwhm
        else:
            unit = self.get_display_grid_unit()
            if unit is None:  # pragma: no cover
                unit = self.get_default_grid_unit()
                if unit is None:
                    unit = 'degree'
            else:
                if isinstance(unit, units.Quantity):
                    unit = unit.unit

            self.correcting_fwhm = utils.get_header_quantity(
                header, 'CORRECTN',
                default=np.nan,
                default_unit=unit).to(unit)

    def parse_smoothing_beam(self, header):
        """
        Parse the smoothing beam from a FITS header.

        The smoothing beam is taken from the SBMAJ and SBMIN keywords, or the
        SMOOTH keyword in the FITS header.  The no keywords are found, the
        smoothing beam FWHM defaults the pixel smoothing.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.SMOOTHING_BEAM_FITS_ID}BMAJ'
        if major_fwhm_key in header:
            self.smoothing_beam = Gaussian2D()
            self.smoothing_beam.parse_header(
                header=header,
                size_unit=self.get_default_grid_unit(),
                fits_id=self.SMOOTHING_BEAM_FITS_ID)
        else:
            unit = self.get_display_grid_unit()
            if unit is None:  # pragma: no cover
                unit = self.get_default_grid_unit()
                if unit is None:
                    unit = 'degree'
            elif isinstance(unit, units.Quantity):
                unit = unit.unit

            fwhm = utils.get_header_quantity(
                header, 'SMOOTH',
                default=0.0,
                default_unit=unit).to(unit)
            self.smoothing_beam = Gaussian2D(x_fwhm=fwhm, y_fwhm=fwhm)

        pixel_smoothing = np.sqrt(self.grid.get_pixel_area()
                                  / Gaussian2D.AREA_FACTOR)
        if not isinstance(pixel_smoothing, units.Quantity):
            pixel_smoothing = pixel_smoothing * self.smoothing_beam.x_fwhm.unit

        self.smoothing_beam.encompass(pixel_smoothing)

    def parse_filter_beam(self, header):
        """
        Parse the filtering beam from a FITS header.

        The correcting beam FWHM is taken from the XBMAJ and XBMIN keywords,
        or the EXTFLTR keyword in the FITS header.  If neither is found, the
        default filtering FWHM defaults to NaN.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.FILTER_BEAM_FITS_ID}BMAJ'
        if major_fwhm_key in header:
            beam = Gaussian2D()
            beam.parse_header(
                header,
                size_unit=self.get_default_grid_unit(),
                fits_id=self.FILTER_BEAM_FITS_ID)
            self.filter_fwhm = beam.fwhm
        else:
            unit = self.get_display_grid_unit()
            if unit is None:  # pragma: no cover
                unit = self.get_default_grid_unit()
                if unit is None:
                    unit = 'degree'
            elif isinstance(unit, units.Quantity):
                unit = unit.unit

            self.filter_fwhm = utils.get_header_quantity(
                header, 'EXTFLTR', default=np.nan,
                default_unit=unit).to(unit)

    def parse_underlying_beam(self, header):
        """
        Parse the underlying beam from a FITS header.

        Uses IBMAJ/IBMIN if available, will then look to BMAJ/MIN, will then
        look for BEAM, and finally the old RESOLUTN.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        major_fwhm_key = f'{self.UNDERLYING_BEAM_FITS_ID}BMAJ'
        self.underlying_beam = Gaussian2D()
        display_unit = self.get_display_grid_unit()
        fits_unit = self.get_default_grid_unit()

        if display_unit is None:  # pragma: no cover
            display_unit = 'degree'
        elif isinstance(display_unit, units.Quantity):
            display_unit = display_unit.unit

        if fits_unit is None:  # pragma: no cover
            fits_unit = 'degree'
        elif isinstance(fits_unit, units.Quantity):  # pragma: no cover
            fits_unit = fits_unit.unit

        if major_fwhm_key in header:
            self.underlying_beam.parse_header(
                header, size_unit=fits_unit,
                fits_id=self.UNDERLYING_BEAM_FITS_ID)

        elif 'BEAM' in header:
            self.underlying_beam.fwhm = utils.get_header_quantity(
                header, 'BEAM', default=np.nan,
                default_unit=display_unit).to(display_unit)

        elif 'BMAJ' in header:
            self.underlying_beam.parse_header(
                header, size_unit=fits_unit, fits_id='')
            self.underlying_beam.deconvolve_with(self.smoothing_beam)

        elif 'RESOLUTN' in header:
            resolution = utils.get_header_quantity(
                header, 'RESOLUTN', default=np.nan,
                default_unit=display_unit).to(display_unit)

            if self.smoothing_beam is None:
                self.underlying_beam.fwhm = resolution
            elif resolution > self.smoothing_beam.major_fwhm:
                self.underlying_beam.fwhm = np.sqrt(
                    (resolution ** 2) - (self.smoothing_beam.x_fwhm
                                         * self.smoothing_beam.y_fwhm))
            else:
                self.underlying_beam.fwhm = 0.0 * units.Unit(display_unit)

        else:
            self.underlying_beam.fwhm = 0.0 * units.Unit(display_unit)

    def parse_header(self, header):
        """
        Parse and apply information from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        try:
            self.parse_coordinate_info(header)
        except Exception as err:  # pragma: no cover
            log.error(f"Could not parse coordinate info in header: {err}")

        self.parse_corrected_beam(header)
        self.parse_smoothing_beam(header)
        self.parse_filter_beam(header)

        # The underlying beam must be parsed after the smoothing because it
        # may rely on the smoothing value in some cases.
        self.parse_underlying_beam(header)

        # The image data unit must be parsed after the instrument beam
        # (underlying + smoothing) and the coordinate grid are established as
        # it may contain 'beam' or 'pixel' type units.
        super().parse_header(header)

    def edit_coordinate_info(self, header):
        """
        Update a header with the current WCS information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        if self.grid is not None:
            self.grid.edit_header(header)

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
        if self.skip_model_edit_header:
            self.skip_model_edit_header = False
            super().edit_header(header)
            return

        self.edit_coordinate_info(header)
        psf = self.get_image_beam()
        fits_unit = self.get_default_grid_unit()
        display_unit = self.get_display_grid_unit()

        if psf is not None:
            psf.edit_header(header, fits_id='', beam_name='image',
                            size_unit=fits_unit)
            if psf.is_circular():
                fwhm = psf.fwhm.to(display_unit.unit).value
                unit_name = display_unit.unit.name
                header['RESOLUTN'] = (
                    fwhm, f'{{Deprecated}} Effective image '
                          f'FWHM ({unit_name}).')

        if self.underlying_beam is not None:
            self.underlying_beam.edit_header(
                header, fits_id=self.UNDERLYING_BEAM_FITS_ID,
                beam_name='instrument', size_unit=fits_unit)

        if self.smoothing_beam is not None:
            self.smoothing_beam.edit_header(
                header, fits_id=self.SMOOTHING_BEAM_FITS_ID,
                beam_name='smoothing', size_unit=fits_unit)
            if self.smoothing_beam.is_circular():
                fwhm = self.smoothing_beam.fwhm.to(display_unit.unit).value
                unit_name = display_unit.unit.name
                header['SMOOTH'] = (
                    fwhm, f'{{Deprecated}} FWHM ({unit_name}) smoothing.')

        if isinstance(self.filter_fwhm, units.Quantity):
            if not np.isnan(self.filter_fwhm):
                filter_beam = Gaussian2D(x_fwhm=self.filter_fwhm,
                                         y_fwhm=self.filter_fwhm)
                filter_beam.edit_header(header, fits_id='X',
                                        beam_name='Extended Structure Filter',
                                        size_unit=fits_unit)

        if isinstance(self.correcting_fwhm, units.Quantity):
            if not np.isnan(self.correcting_fwhm):
                correction_beam = Gaussian2D(x_fwhm=self.correcting_fwhm,
                                             y_fwhm=self.correcting_fwhm)
                correction_beam.edit_header(header, fits_id='C',
                                            beam_name='Peak Corrected',
                                            size_unit=fits_unit)

        header['SMTHRMS'] = True, 'Is the Noise (RMS) image smoothed?'

        super().edit_header(header)

    def claim_image(self, image):
        """
        Claim an image.

        Parameters
        ----------
        image : Image2D

        Returns
        -------
        None
        """
        image.set_unit(self.unit)
        image.set_parallel(self.parallelism)
        image.set_executor(self.executor)

    def count_beams(self):
        """
        Return the number of beams in the map.

        Returns
        -------
        float
        """
        return (self.get_area() / self.get_image_beam_area()).decompose().value

    def count_independent_points(self, area):
        """
        Find the number of independent points in a given area.

        In 1-D at least 3 points per beam are needed to separate a positioned
        point source from an extended source.  Similarly, 9 points per beam
        are necessary for 2-D.

        Parameters
        ----------
        area : astropy.Quantity
            The area to consider.

        Returns
        -------
        points : int
        """

        beam_area = self.get_image_beam_area()
        if self.smoothing_beam is None:
            smoothing_area = 0.0
        else:
            smoothing_area = self.smoothing_beam.area

        if self.is_filtered():
            area_factor = 2 * np.pi * (gaussian_fwhm_to_sigma ** 2)
            filter_area = area_factor * (self.filter_fwhm ** 2)
            eta = 1.0 - (smoothing_area / filter_area).decompose().value
        else:
            eta = 1.0

        pixel_area = self.grid.get_pixel_area()
        if smoothing_area == 0:
            inverse_points_per_beam = 0.0
        else:
            inverse_points_per_beam = eta * min(
                9, smoothing_area / pixel_area)

        return int(np.ceil((1.0 + area / beam_area).decompose().value
                           * inverse_points_per_beam))

    def nearest_to_offset(self, offset):
        """
        Return the nearest map index for a given offset.

        Parameters
        ----------
        offset : Coordinate2D or tuple or numpy.ndarray or list.
            The spatial offset given as a coordinate or tuple of
            (x, y) offsets.

        Returns
        -------
        x_index, y_index : 2-tuple (int or numpy.ndarray)
            The x and y map indices.
        """
        if not isinstance(offset, Coordinate2D):
            self.reuse_index.set(offset)
            offset = self.reuse_index

        ix, iy = utils.round_values(offset.coordinates)
        return ix, iy

    def convert_range_value_to_index(self, ranges):
        """
        Calculate the index range for a given value range.

        Converts a range of dimensional offsets to an appropriate range of
        map indices for use in cropping.

        Parameters
        ----------
        ranges : units.Quantity or Coordinate2D
            An array of shape (n_dimensions, 2) containing the minimum and
            maximum ranges in each dimension.  Dimension ordering is FITS based
            (x, y).

        Returns
        -------
        index_range : numpy.ndarray (int)
            The ranges as indices (integer values) on the grid.
        """
        ranges = Coordinate2D(ranges, unit=self.get_display_grid_unit())
        index_range = np.asarray(self.nearest_to_offset(
            self.grid.offset_to_index(ranges)))

        if self.verbose:
            span = ranges.span
            if (isinstance(span.unit, units.Unit)
                    and span.unit != units.dimensionless_unscaled):
                unit_str = f' {span.unit}'
                distance = span.coordinates.value
            else:  # pragma: no cover
                unit_str = ''
                distance = span.coordinates

            distance_string = 'x'.join(str(x) for x in distance) + unit_str
            log.debug(f"Will crop to {distance_string}.")

        return index_range

    def crop(self, ranges):
        """
        Crop the image data.

        Parameters
        ----------
        ranges : numpy.ndarray (int,) or units.Quantity
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
        if isinstance(ranges, units.Quantity):
            ranges = self.convert_range_value_to_index(ranges)
        else:
            ranges = np.asarray(self.nearest_to_offset(ranges))

        self.get_image().crop(ranges)
        self.grid.reference_index.subtract(Coordinate2D(ranges[:, 0]))

    def auto_crop(self):
        """
        Auto crop the image data.

        The data is cropped to the extent of valid data point indices.

        Returns
        -------
        ranges : numpy.ndarray (int)
            The cropping range for each dimension of shape (n_dimensions, 2)
            where [..., 0] is the minimum range and [..., 1] is the maximum
            crop value (inclusive).
        """
        if self.data is None:
            return
        ranges = self.get_index_range()  # This is a Coordinate2D (float)
        ranges = ranges.coordinates.astype(int)
        if len(ranges) != 2 or None in ranges:  # pragma: no cover
            return
        if (np.allclose(ranges[:, 0], 0)
                and np.allclose(ranges[:, 1], self.shape[::-1])):
            return  # No cropping required

        if self.verbose:
            distance = ranges[:, 1] - ranges[:, 0]
            distance_string = 'x'.join([str(x) for x in distance])
            log.debug(f"Auto-cropping: {distance_string}")
        self.crop(ranges)
        return ranges

    def smooth_to(self, psf):
        """
        Smooth to a given PSF or FWHM.

        Parameters
        ----------
        psf : float or Gaussian2D or Quantity

        Returns
        -------
        None
        """
        if not isinstance(psf, Gaussian2D):
            if not isinstance(psf, units.Quantity):
                psf = psf * self.get_display_grid_unit()
            psf = Gaussian2D(x_fwhm=psf, y_fwhm=psf)

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
        psf : float or Gaussian2D or Quantity

        Returns
        -------
        None
        """
        if not isinstance(psf, Gaussian2D):
            if not isinstance(psf, units.Quantity):
                psf = psf * self.get_display_grid_unit()
            psf = Gaussian2D(x_fwhm=psf, y_fwhm=psf)

        extent = psf.extent()
        steps = Index2D(
            extent.coordinates / (5 * self.grid.get_pixel_size().coordinates))

        beam_map = psf.get_beam_map(self.grid)
        self.fast_smooth(beam_map, steps)

    def smooth(self, beam_map, reference_index=None, weights=None):
        """
        Smooth the data with a given beam map kernel.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The beam map image kernel with which to smooth the map of shape
            (ky, kx).
        reference_index : Coordinate2D or numpy.ndarray
            The reference pixel index of the kernel in (x, y).
        weights : numpy.ndarray (float)
            The map weights of shape (ny, nx).

        Returns
        -------
        None
        """
        super().smooth(beam_map, reference_index=reference_index,
                       weights=weights)
        self.add_smoothing(
            self.default_beam().get_equivalent(beam_map, self.grid.resolution))

    def fast_smooth(self, beam_map, steps, reference_index=None, weights=None):
        """
        Smooth the data with a given beam map kernel using fast method.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The beam map image kernel with which to smooth the map of shape
            (ky, kx).
        steps : Index2D
            The fast step skips in (x, y).
        reference_index : Coordinate2D, optional
            The reference pixel index of the kernel in (x, y).  The default is
            the beam map center ((kx-1)/2, (ky-1)/2).
        weights : numpy.ndarray (float), optional
            The map weights of shape (ny, nx).  The default is no weighting.

        Returns
        -------
        None
        """
        super().fast_smooth(beam_map, steps, reference_index=reference_index,
                            weights=weights)
        self.add_smoothing(
            self.default_beam().get_equivalent(beam_map, self.grid.resolution))

    def filter_above(self, fwhm, valid=None):
        """
        Filter the image using the supplied filtering FWHM.

        The image data is modified via::

            filtered = data - convolve(data, Gaussian(fwhm))

        Parameters
        ----------
        fwhm : units.Quantity or Coordinate2D1
            The FWHM of the Gaussian with which to filter the image.
        valid : numpy.ndarray (bool), optional
            An optional mask where `False` excludes a map element from
            inclusion in the convolution and subtraction.

        Returns
        -------
        None
        """
        extended = self.copy()

        if (hasattr(extended, 'is_zero_weight_valid')  # pragma: no cover
                and hasattr(extended, 'validate')):  # pragma: no cover
            # Make sure zero weights are flagged
            extended.validate()
            extended.is_zero_weight_valid = True

        # Null out the points that are to be skipped over by the validator
        if valid is not None:
            valid = valid & self.valid
            invalid = np.logical_not(valid)
            extended.clear(indices=invalid)
            extended.unflag(indices=invalid)
        else:
            valid = None

        extended.smooth_to(fwhm)
        image = self.get_image()
        image.add(extended, indices=valid, factor=-1)
        self.update_filtering(fwhm)

    def fft_filter_above(self, fwhm, valid=None, weight=None):
        """
        Filter the image with the supplied FWHM using the FFT method.

        Parameters
        ----------
        fwhm : units.Quantity
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
        ny = numba_functions.pow2ceil(self.shape[0] * 2)
        nx = numba_functions.pow2ceil(self.shape[1] * 2)
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
        transformer = np.zeros((ny, nx), dtype=float)
        transformer[:valid.shape[0], :valid.shape[1]] = values

        y, x = np.indices(transformer.shape)
        x -= nx // 2 - 1
        y -= ny // 2 - 1
        sigma = fwhm * gaussian_fwhm_to_sigma

        resolution = self.grid.resolution.coordinates
        a = -0.5 * ((resolution / sigma) ** 2).decompose().value
        g = np.exp(a[0] * x ** 2 + a[1] * y ** 2)
        c = fftconvolve(transformer, g / g.sum(), mode='same')
        c = c[:self.shape[0], :self.shape[1]]
        image = self.get_image()
        image.add(c, factor=-1 / rmsw)  # Subtract re-weighted
        self.update_filtering(fwhm)

    def resample_from_map(self, map2d, weights=None):
        """
        Resample from one map to another.

        Parameters
        ----------
        map2d : Map2D
            The map to resample from.
        weights : numpy.ndarray (float), optional
            Optional weights to include during the resampling.  Weights should
            be the same shape as the `map2d` and represent inverse variance.

        Returns
        -------
        None
        """
        beam = self.get_anti_aliasing_beam_image_for(map2d)
        map_indices = self.get_index_transform_to(map2d)
        if isinstance(map_indices, Coordinate2D1):
            map_indices = map_indices.to_coordinate_3d()
        self.resample_from(map2d, map_indices, kernel=beam, weights=weights)
        self.copy_processing_from(map2d)

    def resample(self, resolution):
        """
        Resample the map to a new resolution.

        Parameters
        ----------
        resolution : units.Quantity
            A resolution for all dimensions if scalar, or each dimension if
            supplied as an array of shape (n_dimensions,) using the (x, y)
            FITS convention.

        Returns
        -------
        None
        """
        original = self.copy()
        original.fits_properties = self.fits_properties.copy()
        if resolution.shape == ():
            resolution = np.full(self.ndim, resolution.value) * resolution.unit
        scaling = (self.grid.resolution.coordinates / resolution
                   ).decompose().value

        new_shape = np.ceil(np.asarray(self.shape) * scaling).astype(int)
        self.set_data_shape(new_shape)
        self.set_grid(original.grid.for_resolution(resolution))
        self.resample_from_map(original)
