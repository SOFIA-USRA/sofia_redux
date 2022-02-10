# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units
import numpy as np
import warnings

from sofia_redux.scan.coordinate_systems.projection.projection_2d import \
    Projection2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection import \
    projection_numba_functions as pnf
from sofia_redux.scan.utilities.class_provider import get_projection_class

__all__ = ['SphericalProjection']


class SphericalProjection(Projection2D):

    right_angle = 90 * units.Unit('degree')
    full_circle = 360 * units.Unit('degree')
    pi = np.pi * units.Unit('radian')
    registry = None

    def __init__(self):
        """
        Initialize a spherical projection.

        A forward spherical projection takes a set of spherical coordinates,
        converting them from a celestial pole to a native pole, and usually
        returns offsets from a given reference wrt a native reference.  The
        exact details will vary by projection model.  A deprojection is the
        reverse operation, where a set of offsets is converted back to
        spherical coordinates.
        """
        super().__init__()
        self._native_reference = SphericalCoordinates(
            [0.0, 0.0], unit='degree')
        self._native_pole = SphericalCoordinates(
            [0.0, 90.0], unit='degree')
        self._celestial_pole = SphericalCoordinates()
        self.user_pole = False  # The pole or native pole was set by the user
        self.user_reference = False  # The reference was set via header parsing
        self.inverted_fits_axes = False  # LON/LAT header axes are reversed
        self.select_solution = 'nearest'

    def copy(self):
        """
        Return a copy of the spherical projection.

        Returns
        -------
        SphericalProjection
        """
        return super().copy()

    @property
    def reference(self):
        """
        Return the reference position of the spherical projection.

        Returns
        -------
        SphericalCoordinates
        """
        return self.get_reference()

    @reference.setter
    def reference(self, value):
        """
        Set the reference position of the spherical projection.

        Parameters
        ----------
        value : SphericalCoordinates

        Returns
        -------
        None
        """
        self.set_reference(value)

    @property
    def native_pole(self):
        """
        Return the native pole of the spherical projection.

        Returns
        -------
        SphericalCoordinates
        """
        return self._native_pole

    @native_pole.setter
    def native_pole(self, value):
        """
        Set the native pole of the spherical projection.

        Parameters
        ----------
        value : SphericalCoordinates

        Returns
        -------
        None
        """
        self.set_native_pole(value)

    @property
    def celestial_pole(self):
        """
        Return the celestial pole of the spherical projection.

        Returns
        -------
        SphericalCoordinates
        """
        return self._celestial_pole

    @celestial_pole.setter
    def celestial_pole(self, value):
        """
        Set the celestial pole of the spherical projection.

        Parameters
        ----------
        value : SphericalCoordinates

        Returns
        -------
        None
        """
        self.set_celestial_pole(value)

    @property
    def native_reference(self):
        """
        Return the native reference position of the spherical projection.

        Returns
        -------
        SphericalCoordinates
        """
        return self._native_reference

    @native_reference.setter
    def native_reference(self, value):
        """
        Set the native reference of the spherical projection.

        Parameters
        ----------
        value : SphericalCoordinates

        Returns
        -------
        None
        """
        self.set_native_reference(value)

    def __eq__(self, other):
        """
        Check whether this projection is equal to another.

        Parameters
        ----------
        other : SphericalProjection

        Returns
        -------
        equal : bool
        """
        if other is self:
            return True
        elif not isinstance(other, SphericalProjection):
            return False
        if not super().__eq__(other):
            return False
        if other.user_pole != self.user_pole:
            return False
        elif other.native_reference != self.native_reference:
            return False
        elif other.native_pole != self.native_pole:
            return False
        elif self.is_right_angle_pole():
            if other.celestial_pole != self.celestial_pole:
                return False
        return True

    @classmethod
    def create_registry(cls):
        """
        Create the lookup registry for spherical projections.

        The SphericalProjection registry is a dictionary of the form
        {name: class} where "name" may be the FITS CTYPE projection designation
        (uppercase), the explicit name of the projection (lowercase), or
        the lowercase name of the class.  For example, the Gnomonic projection
        class will be returned as a value for the keys "TAN", "gnomonic", or
        "gnomonicprojection".  Spaces may be used to specify a compound
        projection name.  For example, "cylindrical equal area".

        Returns
        -------
        None
        """
        cls.registry = {}
        for projection in ['slant_orthographic',  # SIN
                           'gnomonic',  # TAN
                           'zenithal_equal_area',  # ZEA
                           'sanson_flamsteed',  # SFL
                           'mercator',  # MER
                           'plate_carree',  # CAR
                           'hammer_aitoff',  # AIT
                           'global_sinusoidal',  # GLS
                           'stereographic',  # STG
                           'zenithal_equidistant',  # ARC
                           'polyconic',  # PCO
                           'bonnes',  # BON
                           'cylindrical_perspective',  # CYP
                           'cylindrical_equal_area',  # CEA
                           'parabolic']:  # PAR

            projection_class = get_projection_class(projection)
            if projection_class is not None:
                cls.register(projection_class())

    @classmethod
    def register(cls, projection):
        """
        Register a projection into the registry.

        Any given spherical projection will be registered using the uppercase
        FITS ID, the lowercase projection name, and the lowercase class name.

        Parameters
        ----------
        projection : SphericalProjection
           A SphericalProjection subclass instance.

        Returns
        -------
        None
        """
        if cls.registry is None:
            cls.registry = {}
        if not isinstance(projection, SphericalProjection):
            return
        projection_class = projection.__class__
        cls.registry[projection.get_fits_id()] = projection_class
        cls.registry[projection.get_full_name().lower()] = projection_class
        cls.registry[projection.__class__.__name__.lower()] = projection_class

    @classmethod
    def for_name(cls, name):
        """
        Retrieve a projection instance for a given name.

        Parameters
        ----------
        name : str
            The name of the projection to retrieve.

        Returns
        -------
        SphericalProjection
            An initialized spherical projection of the requested type.
        """
        if cls.registry is None:
            cls.create_registry()
        projection = cls.registry.get(name)
        if projection is None:
            raise ValueError(f"No projection {name} in registry.")
        return projection()

    @classmethod
    def asin(cls, value):
        """
        Return the inverse sine of a given value.

        This a wrapper around the :func:`np.arcsin` function to convert values
        from :class:`units.Quantity` to floats if applicable, and then
        bound all values between -1 <= x <= 1 before performing the inverse
        sine operation.  The result will always be return as a
        :class:`units.Quantity` in radians.

        Parameters
        ----------
        value : units.Quantity or float or numpy.ndarray

        Returns
        -------
        angle : units.Quantity
            The angle in radians.
        """
        if isinstance(value, units.Quantity):
            value = value.decompose().value  # converts to radians if angle
        if not isinstance(value, np.ndarray):
            return pnf.asin(value) * units.Unit('radian')
        else:
            return pnf.asin_array(value) * units.Unit('radian')

    @classmethod
    def acos(cls, value):
        """
        Return the inverse cosine of a given value.

        This a wrapper around the :func:`np.arccos` function to convert values
        from :class:`units.Quantity` to floats if applicable, and then
        bound all values between -1 <= x <= 1 before performing the inverse
        cosine operation.  The result will always be return as a
        :class:`units.Quantity` in radians.

        Parameters
        ----------
        value : units.Quantity or float or numpy.ndarray

        Returns
        -------
        angle : units.Quantity
            The angle in radians.
        """
        if isinstance(value, units.Quantity):
            value = value.decompose().value
        if not isinstance(value, np.ndarray):
            return pnf.acos(value) * units.Unit('radian')
        else:
            return pnf.acos_array(value) * units.Unit('radian')

    @classmethod
    def phi_theta_to_radians(cls, phi, theta):
        """
        Convert phi and theta to radian quantities.

        Parameters
        ----------
        phi : float or units.Quantity
            The phi (longitude) parameter.
        theta : float or units.Quantity
            the theta (latitude) parameter.

        Returns
        -------
        phi, theta : units.Quantity, units.Quantity
            phi and theta as radian quantities.
        """
        rad = units.Unit('radian')

        result = []
        for parameter in [phi, theta]:
            if not isinstance(parameter, units.Quantity):
                result.append(parameter * rad)
            elif parameter.unit == units.dimensionless_unscaled:
                result.append(parameter * rad)
            else:
                result.append(parameter.to(rad))

        return result[0], result[1]

    @classmethod
    def offset_to_xy_radians(cls, offset):
        """
        Return the x, y coordinates of an offset as radian quantities.

        Parameters
        ----------
        offset : Coordinate2D

        Returns
        -------
        x, y : units.Quantity, units.Quantity
        """
        rad = units.Unit('radian')
        if offset.unit is None or offset.unit == units.dimensionless_unscaled:
            x, y = offset.x * rad, offset.y * rad
        else:
            x, y = offset.x.to(rad), offset.y.to(rad)
        return x, y

    def get_coordinate_instance(self):
        """
        Return a coordinate instance relevant to the projection.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        return SphericalCoordinates()

    def get_longitude_parameter_prefix(self):
        """
        Return the longitude parameter string prefix.

        Returns
        -------
        str
        """
        return f'PV{2 if self.inverted_fits_axes else 1}_'

    def get_latitude_parameter_prefix(self):
        """
        Return the latitude parameter string prefix.

        Returns
        -------
        str
        """
        return f'PV{1 if self.inverted_fits_axes else 2}_'

    def get_reference(self):
        """
        Return the reference position.

        Returns
        -------
        reference : SphericalCoordinates
        """
        return super().get_reference()

    def is_right_angle_pole(self):
        """
        Return whether the pole is a right angle pole.

        A right-angle pole is when the native latitude of the celestial pole
        is equal to 90 degrees.

        Returns
        -------
        bool
        """
        y = self.celestial_pole.y
        if y is None:
            return False
        return SphericalCoordinates.equal_angles(y, self.right_angle)

    def project(self, coordinates, projected=None):
        """
        Convert coordinates into projection offsets.

        Forward projection consists of first converting coordinates from their
        celestial pole to the native pole of the spherical projection.  The
        resultant output from the projection are generally the offsets of these
        transformed coordinates from a given reference position, although the
        exact definition of an offset may depend on the type of projection.

        Note that if no celestial pole has been defined, it will default to
        (0, 0) LON/LAT coordinates.

        Parameters
        ----------
        coordinates : SphericalCoordinates
            The coordinates to project.
        projected : Coordinate2D, optional
            The output coordinates.  Will be created if not supplied.

        Returns
        -------
        offsets : Coordinate2D
            The projected coordinate offsets using the native spherical pole
            from a native reference.
        """
        array_like = (coordinates.size > 1 or self.celestial_pole.size > 1)
        radian = units.Unit('radian')

        if self.celestial_pole.size == 0:
            celestial_pole = SphericalCoordinates([0, 0])
        else:
            celestial_pole = self.celestial_pole

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if array_like:
                arrays = [coordinates, self.celestial_pole]
                shape = arrays[int(np.argmax([c.size for c in arrays]))].shape
                theta, phi = pnf.spherical_project_array(
                    x=coordinates.x.to('radian').value.ravel(),
                    y=coordinates.y.to('radian').value.ravel(),
                    cos_lat=coordinates.cos_lat.ravel(),
                    sin_lat=coordinates.sin_lat.ravel(),
                    celestial_pole_x=celestial_pole.x.to(
                        'radian').value.ravel(),
                    celestial_pole_y=celestial_pole.y.to(
                        'radian').value.ravel(),
                    celestial_cos_lat=celestial_pole.cos_lat.ravel(),
                    celestial_sin_lat=celestial_pole.sin_lat.ravel(),
                    native_pole_x=self.native_pole.x.to('radian').value)
                theta = theta.reshape(shape) * radian
                phi = phi.reshape(shape) * radian
            else:
                theta, phi = pnf.spherical_project(
                    x=coordinates.x.to('radian').value,
                    y=coordinates.y.to('radian').value,
                    cos_lat=coordinates.cos_lat,
                    sin_lat=coordinates.sin_lat,
                    celestial_pole_x=celestial_pole.x.to('radian').value,
                    celestial_pole_y=celestial_pole.y.to('radian').value,
                    celestial_cos_lat=celestial_pole.cos_lat,
                    celestial_sin_lat=celestial_pole.sin_lat,
                    native_pole_x=self.native_pole.x.to('radian').value)
                theta, phi = theta * radian, phi * radian

            phi = np.fmod(phi, self.full_circle)

        return self.get_offsets(theta, phi, offsets=projected)

    def deproject(self, offsets, coordinates=None):
        """
        Convert projection offsets to spherical coordinates.

        Reverse projection (deprojection) is the process of converting
        projection offsets to coordinates about a native pole, proceeded by a
        transformation to the celestial pole.  The exact details on how native
        coordinates are derived vary by projection type.

        Note, if no celestial pole has been defined, it will default to LON/LAT
        coordinates of (0, 0).

        Parameters
        ----------
        offsets : Coordinate2D
            The projected offsets to deproject to coordinates.
        coordinates : SphericalCoordinates, optional
            The output deprojected coordinates.  Will default to a fresh set
            of SphericalCoordinates if not provided.

        Returns
        -------
        coordinates : SphericalCoordinates or Coordinates2D
            The deprojected coordinates.
        """
        phi_theta = self.get_phi_theta(offsets)
        phi = phi_theta.x
        theta = phi_theta.y

        if coordinates is None:
            coordinates = self.reference.empty_copy()

        array_like = (phi_theta.size > 1 or self.celestial_pole.size > 1)

        if self.celestial_pole.size == 0:
            celestial_pole = SphericalCoordinates([0, 0])
        else:
            celestial_pole = self.celestial_pole

        if array_like:
            arrays = [phi_theta, self.celestial_pole]
            shape = arrays[int(np.argmax([c.size for c in arrays]))].shape
            x, y = pnf.spherical_deproject_array(
                phi=phi.to('radian').value.ravel(),
                theta=theta.to('radian').value.ravel(),
                celestial_pole_x=celestial_pole.x.to(
                    'radian').value.ravel(),
                celestial_pole_y=celestial_pole.y.to(
                    'radian').value.ravel(),
                celestial_cos_lat=celestial_pole.cos_lat.ravel(),
                celestial_sin_lat=celestial_pole.sin_lat.ravel(),
                native_pole_x=self.native_pole.x.to('radian').value)
            x = x.reshape(shape) * units.Unit('radian')
            y = y.reshape(shape) * units.Unit('radian')
        else:
            x, y = pnf.spherical_deproject(
                phi=phi.to('radian').value,
                theta=theta.to('radian').value,
                celestial_pole_x=celestial_pole.x.to('radian').value,
                celestial_pole_y=celestial_pole.y.to('radian').value,
                celestial_cos_lat=celestial_pole.cos_lat,
                celestial_sin_lat=celestial_pole.sin_lat,
                native_pole_x=self.native_pole.x.to('radian').value)
            x, y = x * units.Unit('radian'), y * units.Unit('radian')

        coordinates.set_x(x)
        coordinates.set_y(y)
        coordinates.standardize()
        return coordinates

    def set_reference(self, value):
        """
        Set the reference position.

        Setting a reference position will also result in calculation of a
        celestial pole for the projection.  If a native pole has not been
        defined by the user, a default will be set at this point.

        Parameters
        ----------
        value : SphericalCoordinates

        Returns
        -------
        None
        """
        super().set_reference(value)
        if not self.user_pole:
            self.set_default_native_pole()
        self.calculate_celestial_pole()

    def calculate_celestial_pole(self):
        """
        Calculate the celestial pole.

        Calculates the celestial pole based on the projection reference
        position coordinate.  There are cases when either a northern or
        southern celestial pole may exist as solutions, and the
        `select_solution` attribute can be set to "northern" or "southern"
        to always return this pole.  If `select_solution` is defined as
        something else, such as "nearest" (default), then the solution
        closest to the native pole will be set.

        Returns
        -------
        None
        """
        reference = self.get_reference()
        if reference is None or reference.size == 0:
            return

        self.celestial_pole = SphericalCoordinates()
        array_like = (reference.size > 1
                      or self.native_reference.size > 1
                      or self.native_pole.size > 1)
        if array_like:
            numba_func = pnf.calculate_celestial_pole_array
        else:
            numba_func = pnf.calculate_celestial_pole

        if self.select_solution == 'southern':
            select_solution = -1
        elif self.select_solution == 'northern':
            select_solution = 1
        else:
            select_solution = 0

        x, y = numba_func(
            native_reference_x=self.native_reference.x.to('radian').value,
            native_reference_cos_lat=self.native_reference.cos_lat,
            native_reference_sin_lat=self.native_reference.sin_lat,
            reference_x=reference.x.to('radian').value,
            reference_y=reference.y.to('radian').value,
            reference_cos_lat=reference.cos_lat,
            reference_sin_lat=reference.sin_lat,
            native_pole_x=self.native_pole.x.to('radian').value,
            native_pole_y=self.native_pole.y.to('radian').value,
            select_solution=select_solution)

        self.celestial_pole.set_x(x * units.Unit('radian'))
        self.celestial_pole.set_y(y * units.Unit('radian'))
        self.celestial_pole.standardize()

    def set_native_pole(self, native_pole):
        """
        Set the native pole of the spherical projection.

        If set, the native pole will persist each time a new reference position
        is given.  Otherwise, the native pole will default to either (0, 90) or
        (180, 90) LON/LAT degrees based on whether the reference latitude is
        >= or < the native reference latitude.

        Parameters
        ----------
        native_pole : SphericalCoordinates

        Returns
        -------
        None
        """
        self.user_pole = True
        self._native_pole = native_pole

    def set_default_native_pole(self):
        """
        Set the native pole to the default value.

        The default native pole longitude will be set to 0 degrees if the
        projection reference latitude is greater than or equal to the
        projection native reference latitude (0 by default).  Otherwise, it
        will be set to 180 degrees.  Any user native pole will be overwritten
        during this process.

        Returns
        -------
        None
        """
        rad = units.Unit('radian')
        reference = self.reference

        self.user_pole = False

        if reference is not None and reference.size != 0:
            if reference.singular and self.native_reference.singular:
                if reference.y >= self.native_reference.y:
                    x = 0.0 * rad
                else:
                    x = np.pi * rad
            else:
                less = np.asarray(reference.y < self.native_reference.y)
                # Set the native pole so that most are satisfied
                less_sum = less.sum()
                if less_sum > (less.size - less_sum):
                    x = np.pi * rad
                else:
                    x = 0 * rad
            self.native_pole.set_x(x)

    def set_celestial_pole(self, celestial_pole):
        """
        Set the celestial pole of the spherical projection.

        Parameters
        ----------
        celestial_pole : SphericalCoordinates

        Returns
        -------
        None
        """
        self._celestial_pole = celestial_pole

    def set_native_reference(self, reference):
        """
        Set the native reference for the spherical projection.

        Parameters
        ----------
        reference : SphericalCoordinates

        Returns
        -------
        None
        """
        self._native_reference = reference

    def set_default_pole(self):
        """
        Set the default pole of the spherical projection.

        Returns
        -------
        None
        """
        self.user_pole = False
        self.native_pole = SphericalCoordinates([0, 90], unit='degree')
        self.user_pole = False
        self.set_reference(self.reference)

    def set_native_pole_latitude(self, native_pole_latitude):
        """
        Set the native pole latitude for the spherical projection.

        The native pole latitude will only be set if it is in the range
        -90 <= latitude <= 90 degrees.  Otherwise, it will remain fixed, but
        the selected solution for calculating the celestial pole will change
        to "northern" if greater than zero, or "southern" if less than zero.

        Parameters
        ----------
        native_pole_latitude : units.Quantity

        Returns
        -------
        None
        """
        if np.abs(native_pole_latitude) <= self.right_angle:
            self.native_pole.set_y(native_pole_latitude)
            self.select_solution = 'nearest'
        elif native_pole_latitude > 0:
            self.select_solution = 'northern'
        else:
            self.select_solution = 'southern'

    def parse_header(self, header, alt=''):
        """
        Parse and apply a FITS header to the projection.

        Parameters
        ----------
        header : fits.Header
            The FITS header to parse.
        alt : str, optional
            The alternate FITS system.

        Returns
        -------
        None
        """
        self.inverted_fits_axes = False

        axis_1 = header.get(f'CTYPE1{alt}', '').lower()
        if axis_1.startswith('dec') or axis_1.startswith('lat'):
            self.inverted_fits_axes = True
        elif 'lat' in axis_1 and axis_1[1:].startswith('lat'):
            self.inverted_fits_axes = True

        deg = units.Unit('degree')
        lon_prefix = self.get_longitude_parameter_prefix()
        lat_prefix = self.get_latitude_parameter_prefix()

        for native_pole_lon_key in [f'{lon_prefix}3{alt}', f'LONPOLE{alt}']:
            if native_pole_lon_key in header:
                self.user_pole = True
                self.native_pole.set_x(header[native_pole_lon_key] * deg)
                break

        for native_pole_lat_key in [f'{lat_prefix}4{alt}', f'LATPOLE{alt}']:
            if native_pole_lat_key in header:
                self.user_pole = True
                self.set_native_pole_latitude(
                    header[native_pole_lat_key] * deg)

        reference_lon_key = f'{lon_prefix}1{alt}'
        if reference_lon_key in header:
            self.user_reference = True
            self.native_reference.set_x(header[reference_lon_key] * deg)

        reference_lat_key = f'{lat_prefix}2{alt}'
        if reference_lat_key in header:
            self.user_reference = True
            self.native_reference.set_y(header[reference_lat_key] * deg)

    def edit_header(self, header, alt=''):
        """
        Edit a FITS header with the projection information.

        Parameters
        ----------
        header : fits.Header
            The FITS header to edit.
        alt : str, optional
            The alternate FITS system.

        Returns
        -------
        None
        """
        reference = self.reference
        axes = reference.coordinate_system.axes
        header[f'CTYPE1{alt}'] = (
            f'{reference.fits_longitude_stem}-{self.get_fits_id()}',
            f'{axes[0].short_label} in {self.get_full_name()} projection')
        header[f'CTYPE2{alt}'] = (
            f'{reference.fits_latitude_stem}-{self.get_fits_id()}',
            f'{axes[1].short_label} in {self.get_full_name()} projection')

        if self.user_pole:
            header[f'LONPOLE{alt}'] = (
                self.native_pole.x.to('degree').value,
                'The longitude (deg) of the native pole.')
            header[f'LATPOLE{alt}'] = (
                self.native_pole.y.to('degree').value,
                'The latitude (deg) of the native pole.')

        if self.user_reference:
            lon_prefix = self.get_longitude_parameter_prefix()
            header[f'{lon_prefix}1{alt}'] = (
                self.native_reference.x.to('degree').value,
                'The longitude (deg) of the native reference.')
            lat_prefix = self.get_latitude_parameter_prefix()
            header[f'{lat_prefix}2{alt}'] = (
                self.native_reference.y.to('degree').value,
                'The latitude (deg) of the native reference.')

    @abstractmethod
    def get_phi_theta(self, offset, phi_theta=None):  # pragma: no cover
        """
        Return the phi_theta coordinates.

        The phi and theta coordinates refer to the inverse projection
        (deprojection) of projected offsets about the native pole.  phi is
        the deprojected longitude, and theta is the deprojected latitude of
        the offsets.

        Parameters
        ----------
        offset : Coordinate2D
        phi_theta : SphericalCoordinates, optional
            An optional output coordinate system in which to place the results.

        Returns
        -------
        coordinates : SphericalCoordinates
        """
        pass

    @abstractmethod
    def get_offsets(self, theta, phi, offsets=None):  # pragma: no cover
        """
        Get the offsets given theta and phi.

        Takes the theta (latitude) and phi (longitude) coordinates about the
        celestial pole and converts them to offsets from a reference position.

        Parameters
        ----------
        theta : units.Quantity
            The theta angle.
        phi : units.Quantity
            The phi angle.
        offsets : Coordinate2D, optional
            An optional coordinate system in which to place the results.

        Returns
        -------
        offsets : Coordinate2D
        """
        pass
