# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate import Coordinate
from sofia_redux.scan.coordinate_systems.cartesian_system import \
    CartesianSystem
from sofia_redux.scan.coordinate_systems.grid.grid import Grid


class CartesianGrid(Grid):

    def __init__(self, first_axis=1, dimensions=1):
        """
        Initialize a Cartesian grid.

        The CartesianGrid abstract class is a base for representation of
        Cartesian coordinates on a regular grid.
        """
        super().__init__()
        self.first_axis = 1
        self.set_first_axis_index(first_axis)
        self.set_coordinate_system(CartesianSystem(dimensions))
        self.reference_index = self.get_default_coordinate_instance()
        self.reference_value = self.get_default_coordinate_instance()
        self.axis_resolution = self.get_default_coordinate_instance()
        self.reference_index.set_singular()
        self.reference_value.set_singular()
        self.axis_resolution.set_singular()
        self.axis_resolution.fill(1.0)

    def __eq__(self, other):
        """
        See if this Grid1D is equal to another.

        Parameters
        ----------
        other : Grid1D

        Returns
        -------
        equal : bool
        """
        if not super().__eq__(other):
            return False
        if self.first_axis != other.first_axis:
            return False
        if self.reference_index != other.reference_index:
            return False
        if self.reference_value != other.reference_value:
            return False
        if self.axis_resolution != other.axis_resolution:
            return False
        return True

    @property
    def variant_id(self):
        """
        Return the alternate FITS variant ID for the grid.

        Returns
        -------
        str
        """
        if self.variant == 0:
            return ''
        return chr(ord('A') + self.variant)

    @property
    def unit(self):
        """
        Return the unit for the grid resolution.

        Returns
        -------
        units.Unit
        """
        return self.resolution.unit

    def get_dimensions(self):
        """
        Return the number of dimensions for the grid.

        Returns
        -------
        dimensions : int
        """
        if self.coordinate_system is None:
            return None
        return len(self.coordinate_system)

    def set_first_axis_index(self, counted_from_one):
        """
        Set the index of the first axis.

        Parameters
        ----------
        counted_from_one : int

        Returns
        -------
        None
        """
        self.first_axis = counted_from_one

    def set_resolution(self, resolution):
        """
        Set the resolution for the grid.

        Parameters
        ----------
        resolution : int or float or array or Quantity or Coordinate

        Returns
        -------
        None
        """
        if isinstance(resolution, Coordinate):
            self.axis_resolution.copy_coordinates(resolution)
            return

        if not hasattr(resolution, '__len__'):
            self.axis_resolution.fill(resolution)
        else:
            self.axis_resolution.set(resolution, copy=True)

    def get_resolution(self):
        """
        Get the resolution for the grid.

        Returns
        -------
        resolutions : list
        """
        return self.axis_resolution

    def set_reference(self, value):
        """
        Set the reference value for the grid.

        Parameters
        ----------
        value : int or float or array or Quantity or Coordinate

        Returns
        -------
        None
        """
        if isinstance(value, Coordinate):
            self.reference_value.copy_coordinates(value)
            return

        if not hasattr(value, '__len__'):
            self.reference_value.fill(value)
        else:
            self.reference_value.set(value, copy=True)

    def get_reference(self):
        """
        Get the reference value for the grid.

        Returns
        -------
        reference_values : Coordinate
        """
        return self.reference_value

    def set_reference_index(self, index):
        """
        Set the resolution for the grid.

        Parameters
        ----------
        index : int or float or Iterable or numpy.ndarray or Coordinate

        Returns
        -------
        None
        """
        if isinstance(index, Coordinate):
            self.reference_index.copy_coordinates(index)
        elif not hasattr(index, '__len__'):
            self.reference_index.fill(index)
        else:
            self.reference_index.set(index, copy=True)

        if self.reference_index.unit is not None:
            self.reference_index.coordinates = (
                self.reference_index.coordinates.decompose().value)
            self.reference_index.unit = None

    def get_reference_index(self):
        """
        Get the reference value for the grid.

        Returns
        -------
        reference_index : Coordinate
        """
        return self.reference_index

    def coordinates_at(self, grid_indices, coordinates=None):
        """
        Return the coordinates at a given index.

        Parameters
        ----------
        grid_indices : Coordinate
            The grid indices for which to find coordinates.
        coordinates : Coordinate, optional
            Optional output coordinates to hold the result.

        Returns
        -------
        coordinates : Coordinate
        """
        if coordinates is None:
            coordinates = self.get_default_coordinate_instance()

        offset = self.index_to_offset(grid_indices, in_place=False)
        offset.add(self.reference_value)

        coordinates.copy_coordinates(offset)
        return coordinates

    def index_of(self, coordinates, grid_indices=None):
        """
        Return the index of a coordinate on the grid.

        Parameters
        ----------
        coordinates : Coordinate
            The "true" coordinates.
        grid_indices : Coordinate, optional
            The output coordinates that will hold the result.

        Returns
        -------
        grid_indices : Coordinate
        """
        if grid_indices is None:
            grid_indices = self.get_default_coordinate_instance()

        offset = coordinates.copy()
        offset.subtract(self.reference_value)
        indices = self.offset_to_index(offset, in_place=True)
        grid_indices.copy_coordinates(indices)
        return grid_indices

    def index_to_offset(self, index, in_place=False):
        """
        Convert grid indices to offset positions.

        Parameters
        ----------
        index : Coordinate
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        offset : Coordinate
        """
        if not in_place:
            offset = index.copy()
        else:
            offset = index

        offset.subtract(self.reference_index)
        offset.scale(self.resolution)
        return offset

    def offset_to_index(self, offset, in_place=False):
        """
        Convert an offset to a grid location.

        Parameters
        ----------
        offset : Coordinate or int or float or units.Quantity
        in_place : bool, optional
            If `True`, updates `offset` in-place.  Otherwise, a new instance is
            returned.

        Returns
        -------
        index : Coordinate
        """
        if not in_place:
            index = offset.copy()
        else:
            index = offset

        inverse = self.resolution.coordinates
        if isinstance(inverse, units.Quantity):
            unit = inverse.unit
            value = inverse.value.copy()
        else:
            unit = None
            value = np.asarray(inverse).copy()

        if isinstance(inverse, np.ndarray) and inverse.shape != ():
            inverse = value.copy()
            nzi = inverse != 0
            inverse[nzi] = 1 / value[nzi]
        else:
            inverse = 1 / value

        if unit is not None:
            inverse = inverse / unit

        indices = index.coordinates * inverse
        if isinstance(indices, units.Quantity):
            indices = indices.decompose().value
        if index.unit is not None:
            index.unit = None
        index.coordinates = indices

        index.add(self.reference_index)
        return index

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
        ud = units.dimensionless_unscaled
        for i in range(self.ndim):
            axis = self.coordinate_system.axes[i]
            index = self.first_axis + i
            axis_id = f'{index}{alt}'
            ctype = header.get(f'CTYPE{axis_id}')
            if ctype is not None:
                axis.short_label = ctype
            cunit = header.get(f'CUNIT{axis_id}')
            if cunit is not None:
                axis.unit = units.Unit(cunit)
            else:
                axis.unit = units.dimensionless_unscaled

            self.reference_index.coordinates[i] = header.get(
                f'CRPIX{axis_id}', 1) - 1

            if self.reference_value.unit in [None, ud]:
                self.reference_value.change_unit(axis.unit)
            self.reference_value.coordinates[i] = header.get(
                f'CRVAL{axis_id}', 0.0) * axis.unit

            if self.resolution.unit in [None, ud]:
                self.resolution.change_unit(axis.unit)
            self.resolution.coordinates[i] = header.get(
                f'CDELT{axis_id}', 1.0) * axis.unit

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
        for i in range(self.ndim):
            axis = self.coordinate_system.axes[i]
            index = i + self.first_axis
            axis_id = f'{index}{alt}'
            name = f'Axis-{index}'
            header[f'CTYPE{axis_id}'] = axis.short_label, f'{name} name'
            u = axis.unit
            if u == units.dimensionless_unscaled:
                u = None
            if u is not None:
                header[f'CUNIT{axis_id}'] = str(u), f'{name} unit'
            header[f'CRPIX{axis_id}'] = (
                self.reference_index.coordinates[i] + 1,
                f'{name} reference grid index (1-based)')
            ref = self.reference_value.coordinates[i]
            if isinstance(ref, units.Quantity):
                ref = ref.to(u).value
            header[f'CRVAL{axis_id}'] = ref, f'{name} value at reference index'
            delta = self.resolution.coordinates[i]
            if isinstance(delta, units.Quantity):
                delta = delta.to(u).value
            header[f'CDELT{axis_id}'] = delta, f'{name} spacing'
