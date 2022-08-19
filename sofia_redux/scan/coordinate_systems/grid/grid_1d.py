# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.grid.cartesian_grid import \
    CartesianGrid


class Grid1D(CartesianGrid):

    def __init__(self, first_axis=1):
        """
        Initialize a Cartesian grid.

        The CartesianGrid abstract class is a base for representation of
        Cartesian coordinates on a regular grid.
        """
        super().__init__(first_axis=first_axis, dimensions=1)
        name = self.coordinate_system.dimension_name(first_axis-1)
        self.axis.label = name
        self.axis.short_label = name

    def copy(self):
        """
        Return a copy of the Grid1D.

        Returns
        -------
        Grid1D
        """
        return super().copy()

    @property
    def axis(self):
        """
        Return the grid coordinate axis.

        Returns
        -------
        sofia_redux.scan.coordinate_systems.coordinate_axis.CoordinateAxis
        """
        return self.coordinate_system.axes[0]

    @property
    def resolution(self):
        """
        Return the grid resolution.

        Returns
        -------
        Coordinate1D
        """
        return super().resolution

    @resolution.setter
    def resolution(self, value):
        """
        Set the grid resolution.

        Parameters
        ----------
        value : Coordinate1D or int or float or units.Quantity

        Returns
        -------
        None
        """
        self.set_resolution(value)

    @property
    def reference(self):
        """
        Return the grid reference value.

        Returns
        -------
        Coordinate1D
        """
        return super().reference

    @reference.setter
    def reference(self, value):
        """
        Set the grid reference value.

        Parameters
        ----------
        value : int or float or units.Quantity or Coordinate1

        Returns
        -------
        None
        """
        self.set_reference(value)

    def get_default_unit(self):
        """
        Return the unit for the grid.

        Returns
        -------
        units.Unit
        """
        unit = None
        if isinstance(self.resolution, Coordinate1D):
            unit = self.resolution.unit
        elif isinstance(self.resolution, units.Quantity):  # pragma: no cover
            unit = self.resolution.unit

        if unit is None:
            return units.Unit('pixel')
        else:
            return unit

    def get_default_coordinate_instance(self):
        """
        Return a coordinate for the dimensions of the grid.

        Returns
        -------
        Coordinate1D
        """
        return Coordinate1D()

    def get_dimensions(self):
        """
        Return the number of dimensions for the grid.

        Returns
        -------
        dimensions : int
        """
        return 1

    def set_resolution(self, resolution):
        """
        Set the resolution for the grid.

        Parameters
        ----------
        resolution : int or float or astropy.units.Quantity or Coordinate1D

        Returns
        -------
        None
        """
        if isinstance(resolution, Coordinate1D):
            self.axis_resolution.set_x(resolution.x)
        else:
            self.axis_resolution.set_x(resolution)

    def get_resolution(self):
        """
        Return the resolution for the grid.

        Returns
        -------
        resolution : int or float or astropy.units.Quantity
        """
        return self.axis_resolution

    def set_reference(self, value):
        """
        Set the reference value for the grid.

        Parameters
        ----------
        value : int or float or astropy.units.Quantity or Coordinate1D

        Returns
        -------
        None
        """
        if isinstance(value, Coordinate1D):
            self.reference_value.set_x(value.x)
        else:
            self.reference_value.set_x(value)

    def set_reference_index(self, index):
        """
        Set the resolution for the grid.

        Parameters
        ----------
        index : int or float or Coordinate1D

        Returns
        -------
        None
        """
        if isinstance(index, Coordinate1D):
            self.reference_index.set_x(index.x)
        else:
            self.reference_index.set_x(index)

    def coordinates_at(self, grid_indices, coordinates=None):
        """
        Return the coordinates at a given index.

        Parameters
        ----------
        grid_indices : Coordinate1D
            The grid indices for which to find coordinates.
        coordinates : Coordinate1D, optional
            Optional output coordinates to hold the result.

        Returns
        -------
        coordinates : Coordinate1D
        """
        if coordinates is None:
            coordinates = Coordinate1D()
        return super().coordinates_at(grid_indices, coordinates=coordinates)

    def index_of(self, coordinates, grid_indices=None):
        """
        Return the index of a coordinate on the grid.

        Parameters
        ----------
        coordinates : Coordinate1D
            The "true" coordinates.
        grid_indices : Coordinate1D, optional
            The output coordinates that will hold the result.

        Returns
        -------
        grid_indices : Coordinate1D
        """
        if grid_indices is None:
            grid_indices = Coordinate1D()
        return super().index_of(coordinates, grid_indices=grid_indices)

    def index_to_offset(self, index, in_place=False):
        """
        Convert grid indices to offset positions.

        Parameters
        ----------
        index : Coordinate1D
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        offset : Coordinate1D
        """
        return super().index_to_offset(index, in_place=in_place)

    def offset_to_index(self, offset, in_place=False):
        """
        Convert an offset to a grid location.

        Parameters
        ----------
        offset : Coordinate1D
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        index : Coordinate1D
        """
        return super().offset_to_index(offset, in_place=in_place)

    def parse_header(self, header):
        """
        Parse and apply a FITS header.

        Parses the CTYPE, CUNIT, CRVAL, and CDELT values for the axes.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The FITS header to parse.

        Returns
        -------
        None
        """
        alt = self.variant_id
        axis = self.axis
        index = self.first_axis
        axis_id = f'{index}{alt}'

        ctype = header.get(f'CTYPE{axis_id}')
        if ctype is not None:
            axis.short_label = ctype
        cunit = header.get(f'CUNIT{axis_id}')
        if cunit is not None:
            axis.unit = units.Unit(cunit)
        else:
            axis.unit = units.dimensionless_unscaled

        # FITS is one based
        self.reference_index = Coordinate1D(header.get(f'CRPIX{axis_id}', 1))
        self.reference_index.subtract(1)
        self.reference_value = Coordinate1D(header.get(
            f'CRVAL{axis_id}', 0.0) * axis.unit)
        self.resolution = Coordinate1D(header.get(
            f'CDELT{axis_id}', 1.0) * axis.unit)

    def edit_header(self, header):
        """
        Edit a FITS header with the grid information.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The FITS header to edit.

        Returns
        -------
        None
        """
        alt = self.variant_id
        axis = self.axis
        index = self.first_axis
        axis_id = f'{index}{alt}'
        name = f'Axis-{index}'
        header[f'CTYPE{axis_id}'] = axis.short_label, f'{name} name'
        u = axis.unit
        if (u is None or u == units.dimensionless_unscaled and
                self.resolution.unit is not None):
            u = self.resolution.unit
        if u is not None:
            header[f'CUNIT{axis_id}'] = str(u), f'{name} unit'

        reference_index = self.reference_index
        if isinstance(reference_index, Coordinate1D):
            reference_index = reference_index.x

        header[f'CRPIX{axis_id}'] = (
            reference_index + 1, f'{name} reference grid index (1-based)')
        ref = self.reference_value.x
        if isinstance(ref, units.Quantity):
            ref = ref.to(u).value
        header[f'CRVAL{axis_id}'] = ref, f'{name} value at reference index'
        delta = self.resolution.x
        if isinstance(delta, units.Quantity):
            delta = delta.to(u).value
        header[f'CDELT{axis_id}'] = delta, f'{name} spacing'
