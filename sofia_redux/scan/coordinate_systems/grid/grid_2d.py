# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log, units
import numpy as np
import re

from sofia_redux.scan.coordinate_systems.grid.grid import Grid
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.ecliptic_coordinates import \
    EclipticCoordinates
from sofia_redux.scan.coordinate_systems.galactic_coordinates import \
    GalacticCoordinates
from sofia_redux.scan.coordinate_systems.super_galactic_coordinates import \
    SuperGalacticCoordinates

__all__ = ['Grid2D']


class Grid2D(Grid):

    def __init__(self):
        """
        Initialize a 2-dimensional abstract grid.

        The grid is used to convert from 2D coordinates to offsets in relation
        to a specified reference onto a regular grid, and the reverse
        operation.

        Forward transform: grid projection -> offsets -> coordinates
        Reverse transform: coordinates -> offsets -> grid projection
        """
        super().__init__()
        self._projection = None
        self._reference_index = Coordinate2D(np.zeros(2, dtype=float))
        self.m = None
        self.i = None
        self.set_defaults()

    def copy(self):
        """
        Return a copy of the Grid2D.

        Returns
        -------
        Grid2D
        """
        return super().copy()

    def get_dimensions(self):
        """
        Return the number of grid dimensions.

        Returns
        -------
        n_dimensions : int
        """
        return 2

    def set_defaults(self):
        """
        Set the default values for the grid.

        Returns
        -------
        None
        """
        self.m = np.eye(2, dtype=float)
        self.i = np.eye(2, dtype=float)
        self.reference_index.zero()

    @property
    def reference_index(self):
        """
        Return the reference index for the grid.

        Returns
        -------
        Coordinate2D
        """
        return self.get_reference_index()

    @reference_index.setter
    def reference_index(self, value):
        """
        Set the reference index for the grid.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self.set_reference_index(value)

    @property
    def projection(self):
        """
        Return the grid projection.

        Returns
        -------
        Projection2D
        """
        return self.get_projection()

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
        self.set_projection(value)

    @property
    def transform(self):
        """
        Return the transform matrix for the grid.

        Returns
        -------
        m : numpy.ndarray or units.Quantity
        """
        return self.get_transform()

    @property
    def inverse_transform(self):
        """
        Return the inverse transform matrix for the grid.

        Returns
        -------
        i : numpy.ndarray or units.Quantity
        """
        return self.get_inverse_transform()

    @property
    def rectilinear(self):
        """
        Return whether the grid is rectilinear.

        Returns
        -------
        bool
        """
        return self.m[0, 1] == 0 and self.m[1, 0] == 0

    @property
    def x_axis(self):
        """
        Return the x-axis of the grid.

        Returns
        -------
        CoordinateAxis
        """
        return self.coordinate_system.axes[0]

    @property
    def y_axis(self):
        """
        Return the y-axis of the grid.

        Returns
        -------
        CoordinateAxis
        """
        return self.coordinate_system.axes[1]

    @property
    def fits_x_unit(self):
        """
        Return the unit for the X axis data.

        Returns
        -------
        units.Unit
        """
        return self.x_axis.unit

    @property
    def fits_y_unit(self):
        """
        Return the unit for the Y axis data.

        Returns
        -------
        units.Unit
        """
        return self.y_axis.unit

    def __eq__(self, other):
        """
        Check if this grid is equal to another.

        Parameters
        ----------
        other : Grid2D

        Returns
        -------
        equal : bool
        """
        if other is self:
            return True
        elif not isinstance(other, Grid2D):
            return False

        if self.projection != other.projection:
            return False
        elif self.reference_index != other.reference_index:
            return False
        elif not np.allclose(self.m, other.m):
            return False
        elif not np.allclose(self.i, other.i):
            return False
        return True

    def __str__(self):
        """
        Return a string representation of the grid.

        Returns
        -------
        str
        """
        return self.to_string()

    def __repr__(self):
        """
        Return a representation of the grid and underlying object.

        Returns
        -------
        str
        """
        return f'{object.__repr__(self)}\n{self.to_string()}'

    @staticmethod
    def to_coordinate2d(value):
        """
        Convert a value to a Coordinate2D object.

        Parameters
        ----------
        value : int or float or numpy.ndarray or units.Quantity or Coordinate2D

        Returns
        -------
        Coordinate2D
        """
        if not isinstance(value, Coordinate2D):
            if isinstance(value, (units.Quantity, np.ndarray)):
                value = np.atleast_1d(value)
                if value.size == 1:
                    if isinstance(value, units.Quantity):
                        value = np.full(
                            2, value[0].value, dtype=float) * value.unit
                    else:
                        value = np.full(2, value[0])
            else:
                value = np.full(2, value)

            value = Coordinate2D(value)
        return value

    @classmethod
    def get_coordinate_instance_for(cls, name):
        """
        Return a coordinate instance for the given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        Coordinate2D
        """
        return Coordinate2D.get_instance(name)

    @classmethod
    def get_default_unit(cls):
        """
        Return the default unit for the grid.

        Returns
        -------
        None or units.Unit
        """
        return None

    @classmethod
    def from_header(cls, header, alt=''):
        """
        Create a grid instance from a header.

        Parameters
        ----------
        header : fits.Header
            The FITS header from which to create the grid.
        alt : str, optional
            The alternate coordinate system designation.

        Returns
        -------
        Grid2D
        """
        ctype_1 = header.get(f'CTYPE1{alt}')
        ctype_2 = header.get(f'CTYPE2{alt}')

        grid = cls.get_grid_2d_instance_for(ctype_1, ctype_2)
        grid.parse_header(header)
        return grid

    @classmethod
    def get_grid_2d_instance_for(cls, ctype_1, ctype_2):
        """
        Return a Grid2D instance for the given FITS coordinate types.

        A suitable grid instance will be returned depending on the provided
        CTYPE values.  If any CTYPE value is not recognized, a FlatGrid2D
        instance will be returned.  In cases where a spherical system is
        recognized, a SphericalGrid will be returned.

        CTYPE values should be 8 characters in length, divided into 2 segments
        of 4 characters each.  The first set should give the type of world
        coordinates, and the second set should provide the type of projection
        geometry.  The first set is left justified, with hyphens filling in any
        blank characters to the right, and the second set should be right
        justified with hyphens filling in any blank characters to the left.
        For example, 'RA---TAN' or 'GLON-AIT'.

        Note that in addition to the standard FITS CTYPE standards, the LON and
        LAT coordinate systems are also recognized.  E.g., 'LAT--TAN'.
        xLON, xyLN, xLAT, xyLT coordinate systems are also recognized, where
        x and y indicate arbitrary (but matching across longitude and latitude)
        characters.

        Parameters
        ----------
        ctype_1 : str or None
            The type of the first world coordinate axis.
        ctype_2 : str or None
            The type of the second world coordinate axis.

        Returns
        -------
        Grid2D
        """
        if ctype_1 is None or ctype_2 is None:
            return cls.get_grid_instance('flat_grid_2d')
        elif len(ctype_1) < 6 or len(ctype_2) < 6:
            return cls.get_grid_instance('flat_grid_2d')

        tokens_x = re.split(r'[-]+', ctype_1.lower())
        tokens_y = re.split(r'[-]+', ctype_2.lower())
        if len(tokens_x) != 2 or len(tokens_y) != 2:
            return cls.get_grid_instance('flat_grid_2d')

        x_type, proj_x = tokens_x
        y_type, proj_y = tokens_y

        if (x_type, y_type) in [('ra', 'dec'), ('dec', 'ra'),
                                ('lon', 'lat'), ('lat', 'lon')]:
            return cls.get_grid_instance('spherical_grid')

        if x_type[0] != y_type[0]:
            return cls.get_grid_instance('flat_grid_2d')

        # Check for things like GLON, GLAT
        x_type, y_type = x_type[1:], y_type[1:]
        if (x_type, y_type) in [('lat', 'lon'), ('lon', 'lat')]:
            return cls.get_grid_instance('spherical_grid')

        if x_type[0] != y_type[0]:
            return cls.get_grid_instance('flat_grid_2d')

        # Check for things like ..LN, ..LT
        x_type, y_type = x_type[1:], y_type[1:]
        if (x_type, y_type) in [('lt', 'ln'), ('ln', 'lt')]:
            return cls.get_grid_instance('spherical_grid')

        return cls.get_grid_instance('flat_grid_2d')

    def to_string(self):
        """
        Return a string representation of the grid.

        Returns
        -------
        str
        """
        projection_name = self.reference.__class__.__name__.split(
            'Coordinates')[0]
        reference_str = f'{projection_name}: {self.reference}'
        projection_str = (f'{self.projection.get_full_name()} '
                          f'({self.projection.get_fits_id()})')
        dx, dy = self.resolution.coordinates
        if isinstance(dx, units.Quantity):
            dx = dx.value
        spacing_str = f'{dx} x {dy}'
        crpix_str = f'{self.reference_index} C-style, 0-based'
        return (f'{reference_str}\n'
                f'Projection: {projection_str}\n'
                f'Grid Spacing: {spacing_str}\n'
                f'Reference Pixel: {crpix_str}')

    def for_resolution(self, resolution):
        """
        Return a Grid2D for a given resolution.

        Parameters
        ----------
        resolution : astropy.units.Quantity or numpy.ndarray or Coordinate2D

        Returns
        -------
        grid : Grid2D
        """
        resolution = self.to_coordinate2d(resolution)
        grid = self.copy()
        grid.set_resolution(resolution)
        x_factor = self.resolution.x / resolution.x
        y_factor = self.resolution.y / resolution.y
        if isinstance(x_factor, units.Quantity):
            x_factor = x_factor.decompose().value
            y_factor = y_factor.decompose().value

        grid.reference_index.scale_x(x_factor)
        grid.reference_index.scale_y(y_factor)
        return grid

    def get_pixel_area(self):
        """
        Return the area of one pixel on the grid.

        Returns
        -------
        area : float or units.Quantity
        """
        return np.abs(np.linalg.det(self.m))

    def set_resolution(self, resolution):
        """
        Set the grid resolution.

        Parameters
        ----------
        resolution : Coordinate2D or float or numpy.ndarray or units.Quantity
            The resolution to set.

        Returns
        -------
        None
        """
        resolution = self.to_coordinate2d(resolution)
        dx, dy = resolution.x, resolution.y
        if isinstance(dx, units.Quantity):
            unit = dx.unit
            dx, dy = dx.value, dy.value
        else:
            unit = None
        self.m = np.array([[dx, 0.0], [0.0, dy]])
        if unit is not None:
            self.m = self.m * unit
        self.calculate_inverse_transform()

    def calculate_inverse_transform(self):
        """
        Calculate the inverse transform for the grid (calculate i from m).

        Returns
        -------
        None
        """
        self.i = np.linalg.pinv(self.m)

    def get_transform(self):
        """
        Return the transform matrix for the grid.

        Returns
        -------
        m : numpy.ndarray or units.Quantity
        """
        return self.m.copy()

    def set_transform(self, m):
        """
        Set the transform matrix for the grid.

        Parameters
        ----------
        m : numpy.ndarray or units.Quantity

        Returns
        -------
        None
        """
        if m.shape != (2, 2):
            raise ValueError("Coordinate transform should have shape (2, 2).")
        self.m = m.copy()
        self.calculate_inverse_transform()

    def get_inverse_transform(self):
        """
        Return the inverse transform matrix for the grid.

        Returns
        -------
        i : numpy.ndarray or units.Quantity
        """
        return self.i.copy()

    def is_horizontal(self):
        """
        Return whether the reference coordinate is a horizontal coordinate.

        Returns
        -------
        horizontal : bool
        """
        return isinstance(self.reference, HorizontalCoordinates)

    def is_equatorial(self):
        """
        Return whether the reference coordinate is an equatorial coordinate.

        Returns
        -------
        equatorial : bool
        """
        return isinstance(self.reference, EquatorialCoordinates)

    def is_ecliptic(self):
        """
        Return whether the reference coordinate is an ecliptic coordinate.

        Returns
        -------
        ecliptic : bool
        """
        return isinstance(self.reference, EclipticCoordinates)

    def is_galactic(self):
        """
        Return whether the reference coordinate is a Galactic coordinate.

        Returns
        -------
        galactic : bool
        """
        return isinstance(self.reference, GalacticCoordinates)

    def is_super_galactic(self):
        """
        Return whether the reference coordinate is a super Galactic coordinate.

        Returns
        -------
        super_galactic : bool
        """
        return isinstance(self.reference, SuperGalacticCoordinates)

    def local_affine_transform(self, grid_indices):
        """
        Perform a local affine transform using the forward m matrix.

        The transformed coordinates are of the form:

        new = (m @ coordinates) - reference_index

        Parameters
        ----------
        grid_indices: Coordinate2D

        Returns
        -------
        transformed_indices : numpy.ndarray or units.Quantity
        """
        if isinstance(self.m, units.Quantity):
            if grid_indices.unit is None:
                raise ValueError("Grid indices should be quantities.")
            m = self.m.to(grid_indices.unit).value
        else:
            m = self.m

        if grid_indices.singular:
            coordinates = grid_indices.coordinates[..., None]
        else:
            coordinates = grid_indices.coordinates

        transformed = (m @ coordinates)
        transformed[0] -= self.reference_index.x
        transformed[1] -= self.reference_index.y
        if grid_indices.singular:
            tx, ty = transformed[:, 0]
        else:
            tx, ty = transformed

        t_indices = grid_indices.empty_copy()
        t_indices.set_x(tx)
        t_indices.set_y(ty)
        return t_indices

    def get_resolution(self):
        """
        Return the grid resolution in (x, y).

        Returns
        -------
        resolution : Coordinate2D
        """
        return Coordinate2D(np.diag(self.m))

    def get_pixel_size(self):
        """
        Return the pixel size in (x, y).

        Returns
        -------
        resolution : Coordinate2D
        """
        return self.get_resolution()

    def get_pixel_size_x(self):
        """
        Return the pixel size in the x-direction.

        Returns
        -------
        size : float or units.Quantity
        """
        return self.m[0, 0]

    def get_pixel_size_y(self):
        """
        Return the pixel size in the y-direction.

        Returns
        -------
        size : float or units.Quantity
        """
        return self.m[1, 1]

    def rotate(self, angle):
        """
        Rotate the grid by a given angle.

        This effectively rotates the transformation matrices (m and i).

        Parameters
        ----------
        angle : units.Quantity
            The angle of rotation.

        Returns
        -------
        None
        """
        c = np.cos(angle)
        s = np.sin(angle)
        if isinstance(angle, units.Quantity):
            c, s = c.value, s.value

        a = self.m.copy()
        self.m[0, 0] = (c * a[0, 0]) - (s * a[1, 0])
        self.m[0, 1] = (c * a[0, 1]) - (s * a[1, 1])
        self.m[1, 0] = (s * a[0, 0]) + (c * a[1, 0])
        self.m[1, 1] = (s * a[0, 1]) + (c * a[1, 1])
        self.calculate_inverse_transform()

    def is_reverse_x(self):
        """
        Returns if the x-axis is reversed.

        Returns
        -------
        bool
        """
        return self.x_axis.reverse

    def is_reverse_y(self):
        """
        Returns if the y-axis is reversed.

        Returns
        -------
        bool
        """
        return self.y_axis.reverse

    def parse_header(self, header):
        """
        Parse a FITS header and apply to the grid.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        alt = self.fits_id
        c_type = header.get(f'CTYPE1{alt}')
        try:
            self.parse_projection(header)
        except Exception as err:  # pragma: no cover
            log.warning(f'Unknown projection {c_type}')
            log.error(err)

        default_unit = self.get_default_unit()
        x_unit = header.get(f'CUNIT1{alt}')
        y_unit = header.get(f'CUNIT2{alt}')
        if x_unit is None:
            x_unit = 1.0 if default_unit is None else default_unit
        else:
            x_unit = units.Unit(x_unit)
        if y_unit is None:
            y_unit = 1.0 if default_unit is None else default_unit
        else:
            y_unit = units.Unit(y_unit)

        if isinstance(x_unit, units.Unit):
            self.x_axis.unit = x_unit
        if isinstance(y_unit, units.Unit):
            self.y_axis.unit = y_unit

        cd_keys = [f'CD1_1{alt}', f'CD1_2{alt}', f'CD2_1{alt}', f'CD2_2{alt}']
        for key in cd_keys:
            if key not in header:
                dx = float(header.get(f'CDELT1{alt}', 1.0)) * x_unit
                dy = float(header.get(f'CDELT2{alt}', 1.0)) * y_unit
                m11 = float(header.get(f'PC1_1{alt}', 1.0)) * dx
                m12 = float(header.get(f'PC1_2{alt}', 0.0)) * dx
                m21 = float(header.get(f'PC2_1{alt}', 0.0)) * dy
                m22 = float(header.get(f'PC2_2{alt}', 1.0)) * dy
                break
        else:
            m11 = float(header.get(f'CD1_1{alt}')) * x_unit
            m12 = float(header.get(f'CD1_2{alt}')) * x_unit
            m21 = float(header.get(f'CD2_1{alt}')) * y_unit
            m22 = float(header.get(f'CD2_2{alt}')) * y_unit

        self.m = np.empty((2, 2), dtype=float) * x_unit
        self.m[0, 0] = m11
        self.m[0, 1] = m12
        self.m[1, 0] = m21
        self.m[1, 1] = m22

        # The rotation of the latitude axis.
        if f'CROTA2{alt}' in header:
            self.rotate(header[f'CROTA2{alt}'] * units.Unit('degree'))

        if self.is_reverse_x():
            self.m[:, 0] *= -1
        if self.is_reverse_y():
            self.m[0, 0] *= -1
            self.m[1, 1] *= -1

        one = Coordinate2D([1.0, 1.0])  # FITS origin is at (1, 1)
        self.reference_index.parse_header(
            header, key_stem='CRPIX', alt='', default=one)
        self.reference_index.subtract(one)

        reference = self.get_coordinate_instance_for(c_type)
        reference.parse_header(header, key_stem='CRVAL', alt=alt,
                               default=Coordinate2D([0.0, 0.0]))
        self.set_reference(reference)
        self.calculate_inverse_transform()

    def edit_header(self, header):
        """
        Edit a FITS header with the grid information.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        alt = self.get_fits_id()
        self.projection.edit_header(header, alt=alt)
        header[f'CRPIX1{alt}'] = (self.reference_index.x + 1,
                                  "Reference grid index")
        header[f'CRPIX2{alt}'] = (self.reference_index.y + 1,
                                  "Reference grid index")
        self.projection.reference.edit_header(header,
                                              key_stem='CRVAL', alt=alt)

        # Change from native to apparent for reversed axes.
        a = self.m.copy()
        if self.is_reverse_x():
            a[0] *= -1
        if self.is_reverse_y():
            a[1] *= -1

        nd = units.dimensionless_unscaled

        fits_x_unit = self.fits_x_unit
        fits_y_unit = self.fits_y_unit
        unit = a.unit if isinstance(a, units.Quantity) else None
        if unit == nd:
            a = a.value
            unit = None

        if fits_x_unit == nd or fits_x_unit is None:
            fits_x_unit = unit
        if fits_y_unit == nd or fits_y_unit is None:
            fits_y_unit = unit

        if unit is None and fits_x_unit is not None:
            a = a * fits_x_unit
            unit = fits_x_unit

        x_unit = f' ({fits_x_unit})' if fits_x_unit is not None else ''
        y_unit = f' ({fits_y_unit})' if fits_y_unit is not None else ''

        if unit is not None:
            a00, a01 = a[0].to(fits_x_unit).value
            a10, a11 = a[1].to(fits_y_unit).value
        else:
            a00, a01, a10, a11 = a.ravel()

        if self.m[0, 1] == 0 and self.m[1, 0] == 0:
            header[f'CDELT1{alt}'] = a00, f'Grid spacing{x_unit}'
            header[f'CDELT2{alt}'] = a11, f'Grid spacing{y_unit}'
        else:
            header[f'CD1_1{alt}'] = a00, 'Transformation matrix element'
            header[f'CD1_2{alt}'] = a01, 'Transformation matrix element'
            header[f'CD2_1{alt}'] = a10, 'Transformation matrix element'
            header[f'CD2_2{alt}'] = a11, 'Transformation matrix element'

        if x_unit != '':
            header[f'CUNIT1{alt}'] = (fits_x_unit.name, 'Coordinate axis unit')
            header[f'CUNIT2{alt}'] = (fits_y_unit.name, 'Coordinate axis unit')

    def index_of(self, coordinates, grid_indices=None):
        """
        Return the index of a coordinate on the grid.

        Parameters
        ----------
        coordinates : Coordinate2D
            The "true" coordinates.
        grid_indices : Coordinate2D, optional
            The output coordinates that will hold the result.

        Returns
        -------
        grid_indices : Coordinate2D
        """
        if grid_indices is None:
            grid_indices = Coordinate2D()
        # Convert to offsets
        grid_indices = self.projection.project(
            coordinates, projected=grid_indices)
        self.offset_to_index(grid_indices, in_place=True)
        return grid_indices

    def offset_to_index(self, offset, in_place=False):
        """
        Convert an offset to a grid location.

        Parameters
        ----------
        offset : Coordinate2D
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        index : Coordinate2D
        """
        if isinstance(self.i, units.Quantity):
            if offset.unit is None:
                raise ValueError("Offsets should be quantities.")
            im = self.i.to(1 / offset.unit).value
        else:
            im = self.i

        if offset.singular:
            coordinates = offset.coordinates[..., None]
            singular = True
        else:
            coordinates = offset.coordinates
            singular = False
        singular &= self.reference_index.singular
        if isinstance(self.i, units.Quantity):
            coordinates = coordinates.value

        i00, i01, i10, i11 = im.ravel()
        rx, ry = self.reference_index.coordinates
        ix = (i00 * coordinates[0]) + (i01 * coordinates[1]) + rx
        iy = (i10 * coordinates[0]) + (i11 * coordinates[1]) + ry

        if singular:
            ix, iy = ix[0], iy[0]

        if not in_place:
            indices = Coordinate2D()
        else:
            indices = offset
            indices.unit = None
            indices.coordinates = None

        indices.set([ix, iy])
        return indices

    def index_to_offset(self, grid_indices, in_place=False):
        """
        Convert grid indices to offset positions.

        Parameters
        ----------
        grid_indices : Coordinate2D
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        offset : Coordinate2D
        """
        m = self.m

        if grid_indices.singular:
            indices = grid_indices.coordinates[..., None]
        else:
            indices = grid_indices.coordinates

        di = indices[0] - self.reference_index.x
        dj = indices[1] - self.reference_index.y
        x = (m[0, 0] * di) + (m[0, 1] * dj)
        y = (m[1, 0] * di) + (m[1, 1] * dj)
        if grid_indices.singular and self.reference_index.singular:
            x, y = x[0], y[0]

        if in_place:
            offset = grid_indices
            offset.coordinates = None
            offset.unit = None
        else:
            offset = Coordinate2D()
        offset.set([x, y])
        return offset

    def coordinates_at(self, grid_indices, coordinates=None):
        """
        Return the coordinates at given grid indices.

        Parameters
        ----------
        grid_indices : Coordinate2D
            The grid indices for which to find coordinates.
        coordinates : Coordinate2D, optional
            Optional output coordinates to hold the result.

        Returns
        -------
        Coordinate2D
        """
        if coordinates is None:
            coordinates = self.projection.get_coordinate_instance()
        offset = self.index_to_offset(grid_indices, in_place=False)
        return self.projection.deproject(offset, coordinates=coordinates)

    def get_reference(self):
        """
        Return the reference position of the grid.

        Returns
        -------
        reference : Coordinate2D
        """
        if self.projection is None:
            return None
        return self.projection.get_reference()

    def set_reference(self, value):
        """
        Set the reference position of the grid.

        Parameters
        ----------
        value : Coordinate2D
            The reference coordinate to set.

        Returns
        -------
        None
        """
        self.projection.set_reference(value)

    def get_reference_index(self):
        """
        Return the reference index of the reference position on the grid.

        Returns
        -------
        index : Coordinate2D
        """
        return self._reference_index

    def set_reference_index(self, value):
        """
        Set the reference index of the reference position on the grid.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self._reference_index = value

    def get_projection(self):
        """
        Return the grid projection.

        Returns
        -------
        projection : Projection2D
        """
        return self._projection

    def set_projection(self, value):
        """
        Set the grid projection.

        Parameters
        ----------
        value : Projection2D

        Returns
        -------
        None
        """
        self._projection = value

    def get_coordinate_index(self, coordinates, indices=None):
        """
        Get the indices of the given coordinates.

        Parameters
        ----------
        coordinates : Coordinate2D
            The coordinates for which to find grid indices.
        indices : Coordinate2D, optional
            Optional output coordinates to return.  If not created, a fresh
            Coordinate2D instance will be used.

        Returns
        -------
        indices : Coordinate2D
            The grid indices.
        """
        indices = self.projection.project(coordinates, projected=indices)
        return self.offset_to_index(indices, in_place=True)

    def get_offset_index(self, offsets, indices=None):
        """
        Get the grid indices of the given offsets.

        Parameters
        ----------
        offsets : Coordinate2D
            The offsets for which to find indices.
        indices : Coordinate2D
            Optional coordinates in which to place the results.

        Returns
        -------
        indices : Coordinate2D
            The grid indices.
        """
        offset_indices = self.offset_to_index(offsets, in_place=False)
        i_xy = np.stack([np.round(offset_indices.x).astype(int),
                         np.round(offset_indices.y).astype(int)])
        if indices is None:
            indices = Coordinate2D()
        indices.set(i_xy)
        return indices

    def get_coordinates(self, indices, coordinates=None):
        """
        Get the coordinates at the given grid indices.

        Parameters
        ----------
        indices : Coordinate2D
        coordinates : Coordinate2D, optional
            Optional coordinates in which to store the output coordinates.

        Returns
        -------
        coordinates : Coordinates2D
        """
        if coordinates is None:
            coordinates = self.projection.get_coordinate_instance()
        index_offsets = self.index_to_offset(indices, in_place=False)
        return self.projection.deproject(index_offsets,
                                         coordinates=coordinates)

    def get_offset(self, indices, offset=None):
        """
        Get coordinate offsets from the given grid indices.

        Parameters
        ----------
        indices : Coordinate2D
            The grid indices.
        offset : Coordinate2D, optional
            Optional coordinates in which to store the results.

        Returns
        -------
        offset : Coordinate2D
        """
        if offset is None:
            offset = Coordinate2D()
        offset.copy_coordinates(indices)
        return self.index_to_offset(offset, in_place=True)

    def toggle_native(self, offset, in_place=True):
        """
        Reverse the x and y axis of native offsets if axes are reversed.

        Scales reversed axis by -1.

        Parameters
        ----------
        offset : Coordinate2D
        in_place : bool, optional
            If `True`, toggle in-place, otherwise return a copy.

        Returns
        -------
        Coordinate2D
        """
        if not in_place:
            offset = offset.copy()
        if self.is_reverse_x():
            offset.scale_x(-1)
        if self.is_reverse_y():
            offset.scale_y(-1)
        return offset

    @abstractmethod
    def parse_projection(self, header):  # pragma: no cover
        """
        Parse the projection from the header.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        pass
