# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.source_models.beams.gaussian_source import GaussianSource
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import (
    to_header_quantity, UNKNOWN_FLOAT_VALUE)

__all__ = ['EllipticalSource']


class EllipticalSource(GaussianSource):

    def __init__(self, peak=1.0, x_mean=0.0, y_mean=0.0, x_fwhm=0.0,
                 y_fwhm=0.0, theta=0.0 * units.Unit('deg'), peak_unit=None,
                 position_unit=None, gaussian_model=None):
        """
        Initialize an EllipticalSource model.

        The elliptical source is used to represent a Gaussian source using
        elliptical parameters such as major/minor FWHM, elongation and
        rotation.

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
        self.elongation = None
        self.elongation_weight = None
        self.angle_weight = None
        super().__init__(peak=peak, x_mean=x_mean, y_mean=y_mean,
                         x_fwhm=x_fwhm, y_fwhm=y_fwhm,
                         theta=theta, peak_unit=peak_unit,
                         position_unit=position_unit,
                         gaussian_model=gaussian_model)

        self.set_elongation()

    @property
    def major_fwhm(self):
        """
        Return the major FWHM of the Gaussian ellipse.

        Returns
        -------
        major : units.Quantity
        """
        fwhm = self.fwhm
        major = fwhm + (fwhm * self.elongation)
        major /= np.sqrt(1.0 - (self.elongation ** 2))
        return major

    @property
    def minor_fwhm(self):
        """
        Return the minor FWHM of the Gaussian ellipse.

        Returns
        -------
        minor : units.Quantity
        """
        fwhm = self.fwhm
        minor = fwhm - (self.elongation * fwhm)
        minor /= np.sqrt(1 - (self.elongation ** 2))
        return minor

    @property
    def major_fwhm_weight(self):
        """
        Return the major FWHM weight (1/variance) of the Gaussian ellipse.

        Returns
        -------
        major_weight : float or units.Quantity
        """
        a = self.fwhm
        aw = self.fwhm_weight

        if (isinstance(a, units.Quantity) and
                not isinstance(aw, units.Quantity)):
            aw = aw / (a.unit ** 2)
        elif (isinstance(aw, units.Quantity) and
              not isinstance(a, units.Quantity)):  # pragma: no cover
            a = a / (aw.unit ** 2)

        b = self.elongation
        bw = self.elongation_weight
        w = aw * bw
        if not np.isfinite(w):
            return w

        if w > 0:
            # The product operation
            w /= (((a ** 2) * aw) + ((b ** 2) * bw))
            bw = w
            # The sum operation
            w = (aw * bw) / (aw + bw)
            factor = 1.0 / (1 - (b ** 2))  # The actual factor squared
            w /= factor
        elif isinstance(aw, units.Quantity):
            w = 0.0 * aw.unit
        else:  # pragma: no cover
            w = 0.0

        return w

    @property
    def minor_fwhm_weight(self):
        """
        Return the minor FWHM weight (1/variance) of the Gaussian ellipse.

        Returns
        -------
        minor_weight : float or units.Quantity
        """
        return self.major_fwhm_weight

    @property
    def major_fwhm_rms(self):
        """
        Return the major FWHM RMS.

        Returns
        -------
        major_rms : float or units.Quantity
        """
        return np.sqrt(1.0 / self.major_fwhm_weight)

    @property
    def minor_fwhm_rms(self):
        """
        Return the minor FWHM RMS.

        Returns
        -------
        minor_rms : float or units.Quantity
        """
        return np.sqrt(1.0 / self.minor_fwhm_weight)

    @property
    def angle(self):
        """
        Return the angle of the ellipse.

        Returns
        -------
        angle : units.Quantity
        """
        return self.position_angle

    @angle.setter
    def angle(self, value):
        """
        Set the angle of the ellipse.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        self.position_angle = value % (units.Unit('radian') * np.pi)

    @property
    def angle_rms(self):
        """
        Return the rms of the angle.

        Returns
        -------
        angle_rms : units.Quantity
        """
        if self.angle_weight is None:
            return 0.0 * units.Unit('degree')
        return np.sqrt(1.0 / self.angle_weight)

    @property
    def elongation_rms(self):
        """
        Return the RMS of the elongation.

        Returns
        -------
        elongation_rms : float
        """
        if self.elongation_weight is None or self.elongation_weight <= 0:
            return 0.0
        return np.sqrt(1.0 / self.elongation_weight)

    def set_elongation(self, major=None, minor=None, weight=np.inf,
                       angle=None):
        """
        Set the elongation parameter.

        Parameters
        ----------
        major : float or astropy.units.Quantity, optional
            The major axis of the ellipse.  If not supplied defaults to the
            fwhm of the Gaussian in the x-direction.
        minor : float or astropy.units.Quantity, optional
            The minor axis of the ellipse.  If not supplied defaults to the
            fwhm of the Gaussian in the y-direction.
        weight : float or astropy.units.Quantity, optional
            The weight (inverse variance) of the elongation.  If not supplied
            defaults to infinity (exact).
        angle : float or astropy.units.Quantity, optional
            The angle of the principle axis.  If not supplied defaults to the
            position angle of the Gaussian.  If a float value is supplied, is
            assumed to be in radians.

        Returns
        -------
        None
        """
        if major is None:
            if self.elongation is None:
                major = super().major_fwhm  # The x FWHM
            else:
                major = self.major_fwhm
        if minor is None:
            if self.elongation is None:
                minor = super().minor_fwhm  # The y FWHM
            else:
                minor = self.minor_fwhm
        if angle is None:
            angle = self.position_angle
        if not isinstance(angle, units.Quantity):  # pragma: no cover
            angle = angle * units.Unit('radian')
        if weight is None:
            weight = np.inf  # Exact

        if minor > major:
            temp = minor
            minor = major
            major = temp
            angle += np.pi * units.Unit('radian') / 2

        self.set_xy_fwhm(major, minor)
        self.fwhm = self.fwhm  # Set circular x and y fwhm
        self.position_angle = angle
        self.elongation_weight = weight

        axis_sum = major + minor
        if axis_sum == 0:
            self.elongation = 0.0
        else:
            self.elongation = (major - minor) / axis_sum
        if isinstance(self.elongation, units.Quantity):
            self.elongation = self.elongation.decompose().value

    def edit_header(self, header, fits_id='', beam_name=None, size_unit=None):
        """
        Edit a FITS header with the elliptical beam parameters.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        fits_id : str, optional
            Not used.
        beam_name : str, optional
            The name of the beam.
        size_unit : astropy.units.Unit or Quantity or str, optional
            If set, convert the major/minor beam values to this unit before
            setting in the header.

        Returns
        -------
        None
        """
        super().edit_header(header, fits_id=fits_id, beam_name=beam_name,
                            size_unit=size_unit)

        unknown = UNKNOWN_FLOAT_VALUE
        major, major_rms = self.major_fwhm, self.major_fwhm_rms
        minor, minor_rms = self.minor_fwhm, self.minor_fwhm_rms

        major = to_header_quantity(major, unit=size_unit)
        minor = to_header_quantity(minor, unit=size_unit)
        if size_unit is None and isinstance(major, units.Quantity):
            size_unit = major.unit

        major_rms = to_header_quantity(major_rms, unit=size_unit)
        minor_rms = to_header_quantity(minor_rms, unit=size_unit)
        if size_unit is None and isinstance(
                major_rms, units.Quantity):  # pragma: no cover
            # Cannot reach during normal operation
            size_unit = major_rms.unit
            major = to_header_quantity(major, unit=size_unit)
            minor = to_header_quantity(minor, unit=size_unit)

        if isinstance(major, units.Quantity):
            major = major.value
            minor = minor.value
            major_rms = major_rms.value
            minor_rms = minor_rms.value
            size_comment = f'({size_unit}) '
        else:
            size_comment = ''

        angle = to_header_quantity(self.position_angle, unit='degree').value
        angle_rms = to_header_quantity(self.angle_rms, unit='degree').value
        has_error = self.fwhm_weight != 0 and np.isfinite(self.fwhm_weight)

        if np.isfinite(major) and major != unknown:
            header['SRCMAJ'] = major, size_comment + 'source major axis.'
            if has_error and np.isfinite(major_rms) and major != unknown:
                header['SRCMAJER'] = (
                    major_rms, size_comment + 'major axis error.')

        if np.isfinite(minor) and minor != unknown:
            header['SRCMIN'] = minor, size_comment + 'source minor axis.'
            if has_error and np.isfinite(minor_rms) and minor_rms != unknown:
                header['SRCMINER'] = (
                    minor_rms, size_comment + 'minor axis error.')

        if np.isfinite(angle) and angle != unknown:
            header['SRCPA'] = angle, '(deg) source position angle.'
            if np.isfinite(angle_rms) and angle != unknown:
                header['SRCPAERR'] = angle_rms, '(deg) source angle error.'

    def pointing_info(self, map2d):
        """
        Return a list of strings with pointing information.

        Parameters
        ----------
        map2d : Map2d

        Returns
        -------
        list (str)
        """
        info = super().pointing_info(map2d)
        size_unit = map2d.display_grid_unit  # A quantity, not unit

        major = to_header_quantity(self.major_fwhm, unit=size_unit)
        minor = to_header_quantity(self.minor_fwhm, unit=size_unit)

        if size_unit is None and isinstance(major, units.Quantity):
            size_unit = major.unit

        major_rms = to_header_quantity(self.major_fwhm_rms, unit=size_unit)
        minor_rms = to_header_quantity(self.minor_fwhm_rms, unit=size_unit)
        if size_unit is None and isinstance(
                major_rms, units.Quantity):  # pragma: no cover
            # Cannot reach during normal operation
            size_unit = major_rms.unit
            major = to_header_quantity(self.major_fwhm, unit=size_unit)
            minor = to_header_quantity(self.minor_fwhm, unit=size_unit)

        if isinstance(major, units.Quantity):
            unit_str = f' {major.unit}'
            major = major.value
            minor = minor.value
            major_rms = major_rms.value
            minor_rms = minor_rms.value
        else:
            unit_str = ''

        angle = to_header_quantity(self.angle, unit='degree').value
        values = [major, minor, major_rms, minor_rms, angle]
        for i, value in enumerate(values):
            if value == UNKNOWN_FLOAT_VALUE:
                values[i] = np.nan
        major, minor, major_rms, minor_rms, angle = values

        info.append(f'(a={major:.6f}+-{major_rms:.6f}{unit_str}, '
                    f'b={minor:.6f}+-{minor_rms:.6f}{unit_str}, '
                    f'angle={angle:.6f} deg)')
        return info

    def find_source_extent(self, image, max_iterations=40,
                           radius_increment=1.1, tolerance=0.05):
        """
        Find the extent of the source and shape.

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
        super().find_source_extent(image, max_iterations=max_iterations,
                                   radius_increment=radius_increment,
                                   tolerance=tolerance)
        self.measure_shape(image, min_radius_scale=0.0, max_radius_scale=1.5)

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
        data_sum = super().fit_map_least_squares(map2d)
        self.measure_shape(map2d, min_radius_scale=0.0, max_radius_scale=1.5)
        return data_sum

    def measure_shape(self, image, min_radius_scale=0.0, max_radius_scale=1.5):
        """
        Measure the shape of an elliptical source on an image.

        Parameters
        ----------
        image : FlaggedArray
        min_radius_scale : float, optional
        max_radius_scale : float, optional

        Returns
        -------
        None
        """
        fwhm = self.fwhm
        min_radius = min_radius_scale * fwhm
        max_radius = max_radius_scale * fwhm

        # The offset of the source from the grid reference position
        center = self.get_center_offset()
        grid_offsets = Coordinate2D(np.indices(image.shape)[::-1])
        self.grid.index_to_offset(grid_offsets, in_place=True)
        grid_offsets.subtract(center)
        distance = grid_offsets.length

        keep = (image.valid
                & (distance >= min_radius)
                & (distance <= max_radius))

        data_values = image.data[keep]
        angle = 2 * np.arctan2(grid_offsets.y[keep], grid_offsets.x[keep])

        m2c = np.nansum(np.cos(angle) * data_values)
        m2s = np.nansum(np.sin(angle) * data_values)
        sum_w = np.nansum(np.abs(data_values))

        if sum_w > 0:
            m2c /= sum_w
            m2s /= sum_w
            self.elongation = 2 * np.hypot(m2s, m2c).value
            self.elongation_weight = sum_w
            self.angle = (0.5 * np.arctan2(m2s, m2c)).to('degree')
            angle_rms = ((self.elongation_rms / self.elongation
                          ) * units.Unit('radian')).to('degree')
            self.angle_weight = 1.0 / (angle_rms ** 2)
        else:
            self.angle = 0.0 * units.Unit('degree')
            self.angle_weight = 0.0 / units.Unit('deg2')
            self.elongation = 0.0
            self.elongation_weight = 0.0

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
          - a: The major FWHM
          - b: The minor FWHM
          - da: The major FWHM RMS
          - db: The minor FWHM RMS
          - angle: The rotation of the major axis
          - dangle: The RMS of the rotation of the major axis

        Parameters
        ----------
        map2d : Map2D
            The map for which to calculate an integral.
        size_unit : units.Unit or str, optional
            If set, converts FWHM and dFWHM to `size_unit`.

        Returns
        -------
        dict
        """
        data = super().get_data(map2d, size_unit=size_unit)
        a = to_header_quantity(self.major_fwhm, unit=size_unit, keep=True)
        b = to_header_quantity(self.minor_fwhm, unit=size_unit, keep=True)
        da = to_header_quantity(self.major_fwhm_rms, unit=size_unit, keep=True)
        db = to_header_quantity(self.minor_fwhm_rms, unit=size_unit, keep=True)
        angle = self.angle.to('degree')
        angle_rms = self.angle_rms.to('degree')
        data['a'] = a
        data['b'] = b
        data['da'] = da
        data['db'] = db
        data['angle'] = angle
        data['dangle'] = angle_rms
        return data
