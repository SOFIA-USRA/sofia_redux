# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
from scipy.optimize import curve_fit
import warnings

from sofia_redux.scan.source_models.beams.gaussian_2d import Gaussian2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.utilities.utils import (
    to_header_quantity, UNKNOWN_FLOAT_VALUE)

__all__ = ['GaussianSource']


class GaussianSource(Gaussian2D):
    def __init__(self, peak=1.0, x_mean=0.0, y_mean=0.0,
                 x_fwhm=0.0, y_fwhm=0.0, theta=0.0 * units.Unit('deg'),
                 peak_unit=None, position_unit=None, gaussian_model=None):
        """
        Initialize a GaussianSource model.

        The GaussianSource is an extension of the Gaussian2D model that allows
        the model to be fit to (or from) a source map.

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
        gaussian_model : Gaussian2D, optional
            If supplied, extracts all the above parameters from the supplied
            model.
        """
        self.positioning_method = 'position'
        self.coordinates = None
        self.source_mask = None
        self.source_radius = None
        self.source_sum = 0.0
        self.grid = None
        self.center_index = None
        self.peak_weight = 1.0
        self.fwhm_weight = 1.0
        self.is_corrected = False

        if isinstance(gaussian_model, Gaussian2D):
            super().__init__(peak=gaussian_model.peak,
                             x_mean=gaussian_model.x_mean,
                             y_mean=gaussian_model.y_mean,
                             x_fwhm=gaussian_model.x_fwhm,
                             y_fwhm=gaussian_model.y_fwhm,
                             theta=gaussian_model.theta,
                             peak_unit=gaussian_model.unit)
        else:
            super().__init__(peak=peak, x_mean=x_mean, y_mean=y_mean,
                             x_fwhm=x_fwhm, y_fwhm=y_fwhm, theta=theta,
                             peak_unit=peak_unit, position_unit=position_unit)

    def copy(self):
        """
        Return a copy of the Gaussian source.

        Returns
        -------
        GaussianSource
        """
        return super().copy()

    @property
    def referenced_attributes(self):
        """
        Return the names of attributes to be referenced rather than copied.

        Returns
        -------
        set of str
        """
        attrs = super().referenced_attributes
        attrs.add('grid')
        return attrs

    @property
    def position(self):
        """
        Return the (x, y) position coordinate.

        Returns
        -------
        astropy.units.Quantity
        """
        return Coordinate2D([self.x_mean, self.y_mean])

    @position.setter
    def position(self, value):
        """
        Set the peak position.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self.set_peak_position(value)

    @property
    def peak_significance(self):
        """
        Return the peak significance.

        Returns
        -------
        float
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.abs(self.peak) * np.sqrt(self.peak_weight)

    @property
    def peak_rms(self):
        """
        Return the peak rms.

        Returns
        -------
        float
        """
        weight = self.peak_weight
        if weight > 0:
            return 1.0 / np.sqrt(weight)
        else:
            return weight * 0

    @property
    def fwhm_significance(self):
        """
        Return the FWHM significance.

        Returns
        -------
        float
        """
        return (np.abs(self.fwhm) * np.sqrt(self.fwhm_weight)).value

    @property
    def fwhm_rms(self):
        """
        Return the FWHM RMS.

        Returns
        -------
        float or astropy.units.Quantity
        """
        weight = self.fwhm_weight

        if isinstance(self.fwhm, units.Quantity):
            rms_unit = self.fwhm.unit
        else:  # pragma: no cover
            rms_unit = None

        rms = 0.0 if weight <= 0 else 1.0 / np.sqrt(self.fwhm_weight)
        if not isinstance(rms, units.Quantity) and rms_unit is not None:
            rms = rms * rms_unit  # pragma: no cover
        return rms

    def __eq__(self, other):
        """
        Test if this GaussianSource is functionally equivalent to another.

        Note that only the model is tested for equality.  No factors such as
        grid or coordinates will be examined.

        Parameters
        ----------
        other : GaussianSource

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if not super().__eq__(other):
            return False
        if self.is_corrected != other.is_corrected:
            return False
        return True

    def set_positioning_method(self, method):
        """
        Set the peak positioning method.

        Parameters
        ----------
        method : str
            May be one of {'position', 'peak', 'centroid'}.

        Returns
        -------
        None
        """
        methods = ['position', 'centroid', 'peak']
        s = str(method).lower().strip()
        if s not in methods:
            raise ValueError(f"Available positioning methods are {methods}. "
                             f"Received {method}.")
        if s == 'peak':
            s = 'position'
        self.positioning_method = s

    def set_peak_position(self, peak_position):
        """
        Set the peak position.

        Parameters
        ----------
        peak_position : Coordinate2D or Quantity or numpy.ndarray
            The (x, y) peak position.

        Returns
        -------
        None
        """
        if isinstance(peak_position, Coordinate2D):
            self.x_mean = peak_position.x
            self.y_mean = peak_position.y
            self.coordinates = peak_position.copy()
        elif isinstance(peak_position, np.ndarray):
            if peak_position.shape != ():
                self.x_mean = peak_position[0]
                self.y_mean = peak_position[1]
                self.coordinates = Coordinate2D(peak_position)
        else:  # pragma: no cover
            self.x_mean = peak_position
            self.y_mean = peak_position
            self.coordinates = Coordinate2D([peak_position, peak_position])

    def fit_map_least_squares(self, map2d):
        """
        Fit the Gaussian to a given map using LSQ method (adaptTo).

        Parameters
        ----------
        map2d : Map2D or Observation2D

        Returns
        -------
        data_sum : float
            The sum of the source withing the source radius.
        """
        if hasattr(map2d, 'get_significance'):
            # Use significance if an Observation2D object
            image = map2d.get_significance()
        else:
            image = map2d

        # Get peak in (x, y) indexing
        self.grid = map2d.grid
        data = image.data.copy()
        func, p0, bounds = self.get_lsq_fit_parameters(image)

        y, x = np.indices(data.shape)
        d, x, y = data.ravel(), x.ravel(), y.ravel()
        p_opt, p_cov = curve_fit(func, (x, y), d, p0=p0, bounds=bounds)

        amplitude, x0, y0, pixel_stddev = p_opt
        self.set_center_index(Coordinate2D([x0, y0]))

        # fwhm = sqrt(i * a / peak_value) / fwhm2size
        # fwhm * fwhm2size = sqrt(i * a / peak_value)
        # (fwhm * fwhm2size)^2 = i * a / peak_value
        # (peak_value) / a * (fwhm * fwhm2size)^2 = i

        fwhm_pix = pixel_stddev / gaussian_fwhm_to_sigma
        fwhm = fwhm_pix * np.sqrt(self.grid.get_pixel_area())
        self.fwhm = fwhm

        if image is not map2d:
            self.set_peak_from(map2d)
            fwhm_rms = np.sqrt(2) * fwhm / self.peak_significance
            self.fwhm_weight = 1 / (fwhm_rms ** 2)

        integrated_value = self.peak * (
            (fwhm_pix * self.FWHM_TO_SIZE) ** 2)

        return integrated_value

    def get_lsq_fit_parameters(self, image):
        """
        Return the LSQ fit parameters for curve_fit.

        Determines the entry parameters for a non-linear least squares fit to
        the Gaussian source.  These are the initial guess parameters for the
        peak amplitude, (x, y) mean, and standard deviation.  The bounds for
        these parameters are also returned.  For the GaussianSource, the
        amplitude and centroid location are unbounded, and a minimum bound on
        the standard deviation is set, based on the currently set FWHM.

        Parameters
        ----------
        image : Observation2D or SignificanceMap

        Returns
        -------
        function, initial_values, bounds
        """
        peak_coordinates = self.find_peak(image, sign=0)
        x0, y0 = peak_coordinates.coordinates
        amplitude = image.data[int(np.round(y0)), int(np.round(x0))]
        pixel_size = self.grid.get_pixel_size().coordinates
        fwhm_pix = (self.fwhm / pixel_size.min()).decompose().value
        sigma_pix = gaussian_fwhm_to_sigma * fwhm_pix
        sigma_pix = np.max([1, sigma_pix])

        p0 = (amplitude, x0, y0, sigma_pix)
        bounds = ((-np.inf, -np.inf, -np.inf, sigma_pix),
                  (np.inf, np.inf, np.inf, np.inf))
        return self.gaussian_2d_fit, p0, bounds

    @staticmethod
    def gaussian_2d_fit(coordinates, amplitude, x0, y0, sigma):
        """
        A simple 2-dimensional Gaussian function.

        The return value is:

            z = A.exp((-(x - x0)^2) / (2 sigma^2)) +
                      (-(y - y0)^2) / (2 sigma^2)))

        Parameters
        ----------
        coordinates : 2-tuple (numpy.ndarray)
            The (x, y) coordinates to evaluate.
        amplitude : float
            The scaling factor.
        x0 : float
            The center of the Gaussian in x.
        y0 : float
            The center of the Gaussian in y.
        sigma : float
            The Gaussian standard deviation.

        Returns
        -------
        z : numpy.ndarray (float)
            The derived function value.  Will be the same shape as x or y
            in the `coordinates`.
        """
        dx = coordinates[0] - x0
        dy = coordinates[1] - y0
        a = 2 * (sigma ** 2)
        x1 = -(dx ** 2) / a
        x2 = -(dy ** 2) / a
        return amplitude * np.exp(x1 + x2)

    def fit_map(self, map2d, max_iterations=40, radius_increment=1.1,
                tolerance=0.05):
        """
        Fit the Gaussian source to a given map.

        An Observation2D object or Map2D overlap object may be supplied
        and fit with a Gaussian source.  If an Observation2D is supplied, the
        Gaussian is fit to the significance image, and thus the FWHM rms may
        also be determined.  Otherwise, all other parameters aside from the
        FWHM weight will be fit for.

        Parameters
        ----------
        map2d : Map2D or Observation2D
        max_iterations : int, optional
            The maximum number of iterations by which to increase the radius.
        radius_increment : float, optional
            The factor by which to increase the radius for each iteration.  The
            initial radius size is set to min(grid.pixel_size).
        tolerance : float, optional
            Stop the iterations if (1-tolerance) * last_sum <= sum <=
            (1+tolerance).

        Returns
        -------
        data_sum : float
            The sum of the source withing the source radius.
        """
        if hasattr(map2d, 'get_significance'):
            # Use significance if an Observation2D object
            image = map2d.get_significance()
        else:
            image = map2d

        # Get peak in (x, y) indexing
        self.grid = map2d.grid
        self.set_center_index(self.find_peak(image, sign=0))

        self.find_source_extent(image, max_iterations=max_iterations,
                                radius_increment=radius_increment,
                                tolerance=tolerance)

        # Calculate the peak value and fwhm from the image (maybe significance)
        self.set_peak_from(image)
        fwhm2 = np.abs(self.source_sum
                       * self.grid.get_pixel_area()
                       / self.peak)
        self.fwhm = np.sqrt(fwhm2) / self.FWHM_TO_SIZE
        fwhm_unit = self.fwhm.unit
        self.fwhm_weight = 1.0 / (fwhm_unit ** 2)

        # If the image is not the map, calculate the actual peak value.
        if image is not map2d:
            self.set_peak_from(map2d)  # Sets the peak, peak weight, and unit.
            fwhm_rms = np.sqrt(2.0) * self.fwhm / self.peak_significance
            if fwhm_rms != 0 and not np.isnan(fwhm_rms):
                self.fwhm_weight = 1.0 / (fwhm_rms ** 2)
            else:
                self.fwhm_weight = np.inf / (fwhm_unit ** 2)

        fwhm_pix = self.fwhm / np.sqrt(self.grid.get_pixel_area())
        integrated_value = self.peak * (
            (fwhm_pix * self.FWHM_TO_SIZE) ** 2)

        return integrated_value

    def set_center_index(self, center_index):
        """
        Set the center pixel index on the image for the source.

        Parameters
        ----------
        center_index : Coordinate2D
            The (x, y) center index.

        Returns
        -------
        None
        """
        self.position = self.get_grid_coordinates(center_index)
        self.center_index = center_index

    def get_grid_coordinates(self, index):
        """
        Return the grid coordinates for the given map indices.

        Parameters
        ----------
        index : Coordinate2D
            The dimensionless map pixel coordinates.

        Returns
        -------
        coordinates : Coordinate2D
            The physical map coordinates for the given `index`.
        """
        offset = self.grid.index_to_offset(index)
        grid_coordinates = self.grid.reference.copy()
        self.grid.projection.deproject(offset, coordinates=grid_coordinates)
        return grid_coordinates

    def find_source_extent(self, image, max_iterations=40,
                           radius_increment=0.1, tolerance=0.05):
        """
        Find the extent of the source and shape.

        Calculates the integral of the source, a circular source radius, and
        a source mask.  The algorithm for determining the source iterates using
        an expanding circular aperture centered on the currently defined
        `center_index` attribute.  The iteration will be halted, and parameters
        updated once either the `max_iteration` number of iterations has been
        reached, or::

            data_sum <= last_data_sum * (1 + tolerance)

        The data_sum is the sum of data values inside the circular aperture,
        and the last_data_sum is that same sum from the previous iteration.
        Each iteration the radius of the aperture is increased by::

            min(pixel_size, radius_increment * current_radius)

        The starting radius is equal to the pixel size in the data.

        Parameters
        ----------
        image : FlaggedArray
        max_iterations : int, optional
            The maximum number of iterations, each of which increases the
            search radius by `radius_increment`.
        radius_increment : float, optional
            The factor by which to increase the search radius between
            iterations.
        tolerance : float, optional
            Halt iterations if the change in data sum is less than
            1 + `tolerance` between iterations.

        Returns
        -------
        None
        """
        offset = Coordinate2D(np.indices(image.shape)[::-1])  # numpy to FITS
        offset.subtract(self.center_index)
        pixel_size = self.grid.get_pixel_size()
        offset.scale(pixel_size)
        grow = np.min(pixel_size.coordinates)

        radius = offset.length
        search_radius = 1.0 * grow

        sort_index = np.argsort(radius.ravel())
        radius_sort = radius.ravel()[sort_index]
        data_sort = image.data.ravel()[sort_index]
        search_index = np.nonzero(radius_sort <= search_radius)[0]
        if len(search_index) == 0:
            search_index = 0
        else:
            search_index = search_index[-1]

        tolerance_check = 1 + tolerance

        last_data_sum = np.sum(data_sort[:search_index + 1])
        for iteration in range(max_iterations):

            break_limit = tolerance_check * last_data_sum

            search_radius += min([grow, radius_increment * search_radius])
            search_index = np.nonzero(radius_sort <= search_radius)[0]
            if len(search_index) == 0:
                search_index = 0
            else:
                search_index = search_index[-1]
            data_sum = np.sum(data_sort[:search_index + 1])

            if data_sum == 0:  # may be the case for simulated data/pixel maps
                last_data_sum = 0.0
                continue

            if data_sum >= 0:
                if data_sum <= break_limit:
                    break
            else:
                if data_sum >= break_limit:
                    break
            last_data_sum = data_sum

        self.source_radius = search_radius
        self.source_mask = radius <= search_radius
        self.source_sum = last_data_sum

    def get_center_offset(self, offset=None):
        """
        Find the offset of the source position from the grid reference.

        Parameters
        ----------
        offset : Coordinate2D, optional
            An optional offset into which to place the results.

        Returns
        -------
        offset : Coordinate2D
        """
        center_offset = self.grid.index_to_offset(self.center_index)
        if offset is None:
            return center_offset
        offset.copy_coordinates(center_offset)
        return offset

    def find_peak(self, image, grid=None, sign=0):
        """
        Fit the peak coordinates from a given map. (moveTo)

        Parameters
        ----------
        image : FlaggedArray (float)
            The image to fit.
        grid : SphericalGrid
            The grid to fit onto.
        sign : int or float, optional
            If positive, fit to positive sources in the map.  If negative,
            fit to negative sources in the map.  If zero, fit to source
            magnitudes in the map.  Note, this only takes affect if `position`
            is set as the positioning_method.

        Returns
        -------
        peak : Coordinate2D
            The (x, y) peak position.
        """
        # Note that the peak position is in terms of pixels (y, x)
        if self.positioning_method == 'position':
            peak = self.find_local_peak(image, sign=sign)
        elif self.positioning_method == 'centroid':
            peak = self.find_local_centroid(image)
        else:
            raise ValueError(
                f"Unknown positioning method: {self.positioning_method}.")

        # peak is in pixel positions (y, x) format
        if grid is None:
            return peak
        else:
            return grid.projection.deproject(peak)

    @staticmethod
    def find_local_peak(image, sign=0):
        """
        Find the local peak in a map using the 'position' method.

        Parameters
        ----------
        image : FlaggedArray
            The image to fit.
        sign : int or float, optional
            If positive, fit to positive sources in the map.  If negative,
            fit to negative sources in the map.  If zero, fit to source
            magnitudes in the map.  Note, this only takes affect if `position`
            is set as the positioning_method.

        Returns
        -------
        peak : Coordinate2D
            The (x, y) pixel peak position.
        """
        peak_value, peak_index = image.index_of_max(sign=sign)
        if isinstance(peak_index, Coordinate2D):
            peak_index = Index2D(peak_index.coordinates)
        else:  # pragma: no cover
            peak_index = Index2D(peak_index)
        return Coordinate2D(image.get_refined_peak_index(peak_index))

    @staticmethod
    def find_local_centroid(image):
        """
        Find the local peak in a map using the 'centroid' method.

        Parameters
        ----------
        image : FlaggedArray

        Returns
        -------
        peak : Coordinate2D
            The (x, y) peak position.
        """
        indices = np.indices(image.shape)[::-1]  # to (x, y) FITS format
        data = image.data

        w = np.abs(data) * image.valid
        w[np.isnan(w)] = 0.0
        w_sum = w.sum()
        wd_sum = np.empty(image.ndim, dtype=float)
        for dimension in range(image.ndim):
            wd_sum[dimension] = np.sum(indices[dimension] * w)

        return Coordinate2D(wd_sum / w_sum)

    def set_peak_from(self, image, degree=3):
        """
        Set the peak value from a given image.

        The peak value is determined from spline interpolation on the image
        at `self.center_index`.  This will also set the weight of the peak
        as exact (infinity) or will interpolate from a weight map if available.

        Parameters
        ----------
        image : FlaggedArray or Map2D
        degree : int, optional
            The spline degree to fit.

        Returns
        -------
        None
        """
        peak_value = image.value_at(self.center_index, degree=degree)
        unit = getattr(image, 'unit', 1.0 * units.dimensionless_unscaled)
        self.peak = peak_value
        if hasattr(image, 'weight'):
            self.peak_weight = image.get_weights().value_at(self.center_index,
                                                            degree=degree)
        else:
            self.set_exact()  # peak weight set to infinity

        self.set_unit(unit)

    def set_unit(self, unit):
        """
        Set the unit of the Gaussian peak.

        Parameters
        ----------
        unit : units.Quantity or units.Unit or float or str or None

        Returns
        -------
        None
        """
        if unit is None:
            unit_value = 1.0
            unit_type = units.dimensionless_unscaled
        elif isinstance(unit, units.Quantity):
            unit_value = unit.value
            unit_type = unit.unit
        elif isinstance(unit, (units.Unit, str)):
            unit = units.Unit(unit)
            unit_value = 1.0
            unit_type = unit
        else:  # assume a number
            unit_value = float(unit)
            unit_type = units.dimensionless_unscaled

        self.unit = unit_type
        self.scale_peak(unit_value)

    def scale_peak(self, factor):
        """
        Scale the peak by a given amount.

        Parameters
        ----------
        factor : float

        Returns
        -------
        None
        """
        if factor == 1:
            return
        self.peak = self.peak * factor
        self.peak_weight /= factor ** 2

    def scale_fwhm(self, factor):
        """
        Scale the fwhm by a given amount

        Parameters
        ----------
        factor : float

        Returns
        -------
        None
        """
        if factor == 1:
            return
        self.fwhm *= factor
        self.fwhm_weight /= factor ** 2

    def set_exact(self):
        """
        Set the peak to be 'exact'.

        Returns
        -------
        None
        """
        self.peak_weight = np.inf

    def get_correction_factor(self, map2d):
        """
        Get the correction factor for a given map.

        Notes
        -----
        The original CRUSH would always return a value of 1, since the logic
        based on the map2d.filter_fwhm returns 1 if it is not NaN, and sets the
        filtering factor to 0 if it is NaN.  This has been remedied here, but
        does not appear to actually be used during the reduction.

        Parameters
        ----------
        map2d : Map2D

        Returns
        -------
        correction_factor : float
        """
        if not map2d.is_filtered():  # map2d.filter_fwhm = NaN
            return 1.0

        if map2d.is_filter_blanked():
            filter_fraction = min(
                map2d.filter_blanking / self.peak_significance, 1.0)
        else:
            filter_fraction = 1.0

        filtering = 1.0 - (1.0 / map2d.get_filter_correction_factor())
        correction = 1.0 / (1.0 - filtering * filter_fraction)
        return correction

    def correct(self, map2d):
        """
        Apply peak value correction from a map.

        Parameters
        ----------
        map2d : Map2d

        Returns
        -------
        None
        """
        if self.is_corrected:
            log.warning("Source is already corrected.")
            return
        self.scale_peak(self.get_correction_factor(map2d))
        self.is_corrected = True

    def uncorrect(self, map2d):
        """
        Uncorrect the peak value for a given map.

        Parameters
        ----------
        map2d : Map2d

        Returns
        -------
        None
        """
        if not self.is_corrected:
            log.warning("Source is already uncorrected")
            return
        self.scale_peak(1.0 / self.get_correction_factor(map2d))
        self.is_corrected = False

    def get_gaussian_2d(self):
        """
        Return a representation of the beam as a Gaussian2D object.

        Returns
        -------
        Gaussian2D
        """
        return Gaussian2D(peak=self.peak,
                          x_fwhm=self.x_fwhm,
                          y_fwhm=self.y_fwhm,
                          x_mean=self.x_mean,
                          y_mean=self.y_mean,
                          theta=self.theta)

    def deconvolve_with(self, psf):
        """
        Deconvolve with a given psf.

        Note that this process will result in a circular beam.  I.e. one in
        which the FWHM in the x and y-directions are equivalent.

        Parameters
        ----------
        psf : Gaussian2D

        Returns
        -------
        None
        """
        factor = self.fwhm
        beam = self.get_gaussian_2d()
        beam.deconvolve_with(psf)
        self.fwhm = beam.get_circular_equivalent_fwhm()
        new_fwhm = self.fwhm

        if isinstance(self.fwhm, units.Quantity):
            weight_unit = (1 / self.fwhm.unit) ** 2
        else:  # pragma: no cover
            weight_unit = None

        if new_fwhm == 0:
            self.fwhm_weight = 0.0
        else:
            factor /= new_fwhm
            self.fwhm_weight /= factor ** 2

        if weight_unit is not None:
            if not isinstance(self.fwhm_weight, units.Quantity) or (
                    self.fwhm_weight.unit == units.dimensionless_unscaled):
                self.fwhm_weight = self.fwhm_weight * weight_unit

    def convolve_with(self, psf):
        """
        Convolve with a given PSF.

        Convolve the current beam with a given PSF.  The shape of the beam
        will be accounted for correctly in both (x, y) dimensions unlike
        the `deconvolve_with` method.

        Parameters
        ----------
        psf : Gaussian2D

        Returns
        -------
        None
        """
        factor = self.fwhm
        beam = self.get_gaussian_2d()
        beam.convolve_with(psf)
        self.set_xy_fwhm(beam.x_fwhm, beam.y_fwhm)
        new_fwhm = self.fwhm
        if new_fwhm == 0:
            self.fwhm_weight = 0.0
        else:
            factor /= new_fwhm
            self.fwhm_weight /= factor ** 2

        if isinstance(self.fwhm, units.Quantity):
            weight_unit = (1 / self.fwhm.unit) ** 2
            if not isinstance(self.fwhm_weight, units.Quantity) or (
                    self.fwhm_weight.unit == units.dimensionless_unscaled):
                self.fwhm_weight = self.fwhm_weight * weight_unit

    def edit_header(self, header, fits_id='', beam_name=None, size_unit=None):
        """
        Edit a FITS header with the beam parameters.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        fits_id : str, optional
            Not used.
        beam_name : str, optional
            The name of the beam.  Not implemented for the GaussianSource.
        size_unit : astropy.units.Unit or Quantity or str, optional
            If set, convert the major/minor beam values to this unit before
            setting in the header.

        Returns
        -------
        None
        """
        fwhm = to_header_quantity(self.fwhm, unit=size_unit)
        if size_unit is None and isinstance(fwhm, units.Quantity):
            size_unit = fwhm.unit
        fwhm_rms = to_header_quantity(self.fwhm_rms, unit=size_unit)
        if size_unit is None and isinstance(
                fwhm_rms, units.Quantity):  # pragma: no cover
            # Cannot reach in normal operation
            size_unit = fwhm_rms.unit
            if not isinstance(fwhm, units.Quantity):
                fwhm = to_header_quantity(fwhm, unit=size_unit)

        if isinstance(fwhm, units.Quantity):
            size_comment = f'({size_unit}) '
            fwhm = fwhm.value
            fwhm_rms = fwhm_rms.value
        else:
            size_comment = ''

        peak_unit = self.unit
        peak = to_header_quantity(self.peak, unit=peak_unit)
        if isinstance(peak, units.Quantity):
            peak_unit = peak.unit
        else:
            peak_unit = None

        peak_rms = to_header_quantity(self.peak_rms, unit=peak_unit)
        if peak_unit is None and isinstance(
                peak_rms, units.Quantity):  # pragma: no cover
            # Cannot reach in normal operation
            peak_unit = peak_rms.unit
            peak = to_header_quantity(peak, unit=peak_unit)

        if isinstance(peak, units.Quantity):
            unit_comment = f'({peak_unit}) '
            peak = peak.value
            peak_rms = peak_rms.value
        else:
            unit_comment = ''

        header['SRCPEAK'] = peak, unit_comment + 'source peak flux.'
        header['SRCPKERR'] = peak_rms, unit_comment + 'peak flux error.'

        if np.isfinite(fwhm) and fwhm != UNKNOWN_FLOAT_VALUE:
            header['SRCFWHM'] = fwhm, size_comment + 'source FWHM.'
        if (self.fwhm_weight > 0 and np.isfinite(fwhm_rms) and
                fwhm_rms != UNKNOWN_FLOAT_VALUE):
            header['SRCWERR'] = fwhm_rms, size_comment + 'FWHM error.'

    def get_integral(self, psf_area):
        """
        Return the integral over a given area.

        Parameters
        ----------
        psf_area : astropy.units.Quantity or float
            The area over which to evaluate the integral.

        Returns
        -------
        integral, weight : (float, float) or (Quantity, Quantity)
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            factor = self.area / psf_area
            if isinstance(factor, units.Quantity):
                factor = factor.decompose().value
            integral = self.peak * factor
            weight = self.peak_weight / (factor ** 2)
        return integral, weight

    def pointing_info(self, map2d):
        """
        Return a list of strings with pointing information.

        Parameters
        ----------
        map2d : Map2d

        Returns
        -------
        info : list (str)
        """
        peak = to_header_quantity(self.peak, unit=self.unit)
        peak_value = peak.value if isinstance(peak, units.Quantity) else peak
        if peak_value == UNKNOWN_FLOAT_VALUE:
            peak_value = self.peak

        unit_str = f' {peak.unit}' if isinstance(peak, units.Quantity) else ''
        info = [f'Peak: {peak_value:.5f}{unit_str} '
                f'(S/N ~ {self.peak_significance:.5f})']
        size_unit = map2d.display_grid_unit
        integral, integral_weight = self.get_integral(
            map2d.underlying_beam.area)

        if integral_weight > 0:
            integral_rms = 1.0 / np.sqrt(integral_weight)
        else:
            integral_rms = integral_weight * 0
        if integral_rms != 0:
            info.append(
                f'Integral: {integral:.4f} +- {integral_rms:.4f}{unit_str}')
        else:
            info.append(f'Integral: {integral:.4f}{unit_str}')

        fwhm = to_header_quantity(self.fwhm, unit=size_unit)
        if size_unit is None and isinstance(fwhm, units.Quantity):
            size_unit = fwhm.unit
        fwhm_rms = to_header_quantity(self.fwhm_rms, unit=size_unit)
        if size_unit is None and isinstance(
                fwhm_rms, units.Quantity):  # pragma: no cover
            # Cannot reach during normal operation
            size_unit = fwhm_rms.unit
            fwhm = to_header_quantity(fwhm, unit=size_unit)

        if isinstance(fwhm, units.Quantity):
            unit_str = f' ({fwhm.unit})'
            fwhm = fwhm.value
            fwhm_rms = fwhm_rms.value
        else:
            unit_str = ''

        if fwhm_rms == 0:
            info.append(f'FWHM: {fwhm:.4f}{unit_str}')
        else:
            info.append(f'FWHM: {fwhm:.4f} +- {fwhm_rms:.4f}{unit_str}')
        return info

    def get_asymmetry_2d(self, image, angle, radial_range):
        """
        Return the asymmetry of the GaussianSource.

        Parameters
        ----------
        image : Image2D
            The image from which to determine the asymmetry.
        angle : units.Quantity
            The rotation of the image with respect to the GaussianSource.
        radial_range : Range
            The range over which the asymmetry should be calculated.

        Returns
        -------
        asymmetry : Asymmetry2D
        """
        return image.get_asymmetry_2d(self.grid, self.center_index,
                                      angle, radial_range)

    def get_representation(self, grid):
        """
        Return a representation of the Gaussian source on a new grid.

        Parameters
        ----------
        grid : Grid2D
            The new grid to apply to a copy of this GaussianSource.

        Returns
        -------
        GaussianSource
            A copy of the Gaussian source on the new grid.
        """
        center_offset = self.grid.index_to_offset(self.center_index)
        new = self.copy()
        new.grid = grid.copy()
        new.set_center_index(new.grid.offset_to_index(center_offset))
        return new

    def get_data(self, map2d, size_unit=None):
        """
        Return a dictionary of properties for to the source model on a map.

        The key values returned are:

          - peak: The fitted peak value
          - dpeak: The fitted peak value RMS
          - peakS2N: The peak signal-to-noise ratio
          - int: The integral of the peak on the map
          - dint: The integral rms of the peak on the map
          - intS2N: The significance of the peak on the map
          - FWHM: The full-width-half maximum of the peak
          - dFWHM: The full-width-half-maximum RMS of the peak

        Parameters
        ----------
        map2d : Map2D
            The map for which to calculate an integral.
        size_unit : units.Unit or str, optional
            If set, converts FWHM and dFWHM to `size_unit`.

        Returns
        -------
        data : dict
        """
        data = {}
        if map2d.underlying_beam is not None:
            i, iw = self.get_integral(map2d.underlying_beam.area)
        else:
            i, iw = 0.0, np.inf

        fwhm = self.fwhm
        fwhm_rms = self.fwhm_rms

        data['peak'] = self.peak * self.unit
        data['dpeak'] = self.peak_rms * self.unit
        data['peakS2N'] = self.peak_significance
        data['int'] = i * self.unit
        data['dint'] = 1.0 / np.sqrt(iw) * self.unit
        if i == 0 and iw == np.inf:
            data['intS2N'] = 0.0
        else:
            data['intS2N'] = np.abs(i) * np.sqrt(iw)

        fwhm = to_header_quantity(fwhm, unit=size_unit, keep=True)
        fwhm_rms = to_header_quantity(fwhm_rms, unit=size_unit, keep=True)

        data['FWHM'] = fwhm
        data['dFWHM'] = fwhm_rms
        return data
