# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod
from copy import deepcopy

from sofia_redux.scan.utilities.class_provider import get_grid_class

__all__ = ['Grid']


class Grid(ABC):

    def __init__(self):
        """
        Initialize a Grid.

        The Grid abstract class is a base for representation of astronomical
        data on a regular grid.  The exact coordinate system used is not
        specified, and should be defined by the user.  At the simplest level,
        a grid consists of a coordinate system (set of axes), a reference
        position, and the grid resolution.

        The grid is used to convert from coordinates to offsets in relation
        to a specified reference onto a regular grid, and the reverse
        operation.

        Forward transform: grid projection -> offsets -> coordinates
        Reverse transform: coordinates -> offsets -> grid projection

        """
        self._coordinate_system = None
        self.variant = 0

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced rather than copied.

        Returns
        -------
        set (str)
        """
        return set([])

    def copy(self):
        """
        Create and return a copy.

        Note that the copy scan is a reference.  The configuration is unlinked
        (i.e. is no longer a reference).

        Returns
        -------
        Channels
        """
        new = self.__class__()
        for attribute, value in self.__dict__.items():
            if attribute in self.referenced_attributes:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))
        return new

    @property
    def coordinate_system(self):
        """
        Return the coordinate system.

        Returns
        -------
        CoordinateSystem
        """
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, system):
        """
        Set the coordinate system.

        Parameters
        ----------
        system : CoordinateSystem

        Returns
        -------
        None
        """
        self.set_coordinate_system(system)

    @property
    def ndim(self):
        """
        Return the number of grid dimensions.

        Returns
        -------
        n_dimensions : int
        """
        return self.get_dimensions()

    @property
    def resolution(self):
        """
        Return the grid resolution.

        Returns
        -------
        Coordinate
        """
        return self.get_resolution()

    @resolution.setter
    def resolution(self, value):
        """
        Set the grid resolution.

        Parameters
        ----------
        value : Coordinate

        Returns
        -------
        None
        """
        self.set_resolution(value)

    @property
    def reference(self):
        """
        Return the grid reference position.

        Returns
        -------
        Coordinate
        """
        return self.get_reference()

    @reference.setter
    def reference(self, value):
        """
        Set the grid reference position.

        Parameters
        ----------
        value : Coordinate

        Returns
        -------
        None
        """
        self.set_reference(value)

    @property
    def fits_id(self):
        """
        Return the FITS ID for the grid.

        Returns
        -------
        str
        """
        return self.get_fits_id()

    @staticmethod
    def get_grid_class(name):
        """
        Return a grid class of the given name.

        Parameters
        ----------
        name : str
            The name of the grid.

        Returns
        -------
        grid : class
        """
        return get_grid_class(name)

    @staticmethod
    def get_grid_instance(name):
        """
        Return a grid instance for the given name.

        Parameters
        ----------
        name : str
            The name of the grid.

        Returns
        -------
        grid : Grid
        """
        return get_grid_class(name)()

    def set_coordinate_system(self, system):
        """
        Set the coordinate system for the grid.

        Parameters
        ----------
        system : CoordinateSystem

        Returns
        -------
        None
        """
        if self.ndim != system.size:
            raise ValueError(
                f"Number of coordinate system axes ({system.size}) "
                f"does not equal the grid dimensions ({self.ndim}).")
        self._coordinate_system = system

    def get_fits_id(self):
        """
        Return the FITS variant ID for the grid.

        Returns
        -------
        str
        """
        if self.variant == 0:
            return ''
        else:
            return chr(ord('A') + self.variant)

    @abstractmethod
    def get_dimensions(self):  # pragma: no cover
        """
        Return the number of grid dimensions.

        Returns
        -------
        n_dimensions : int
        """
        pass

    @abstractmethod
    def get_reference(self):  # pragma: no cover
        """
        Return the reference position of the grid.

        Returns
        -------
        reference : Coordinate
        """
        pass

    @abstractmethod
    def set_reference(self, value):  # pragma: no cover
        """
        Set the reference position of the grid.

        Parameters
        ----------
        value : Coordinate
            The reference coordinate to set.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_reference_index(self):  # pragma: no cover
        """
        Return the reference index of the reference position on the grid.

        Returns
        -------
        index : Offset
        """
        pass

    @abstractmethod
    def set_reference_index(self, value):  # pragma: no cover
        """
        Set the reference index of the reference position on the grid.

        Parameters
        ----------
        value : Offset

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_resolution(self):  # pragma: no cover
        """
        Return the grid resolution.

        Returns
        -------
        resolution : Offset
        """
        pass

    @abstractmethod
    def set_resolution(self, value):  # pragma: no cover
        """
        Set the grid resolution.

        Parameters
        ----------
        value : Offset

        Returns
        -------
        None
        """
        pass
