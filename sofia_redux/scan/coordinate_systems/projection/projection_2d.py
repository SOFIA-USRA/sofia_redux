# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod
from copy import deepcopy

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['Projection2D']


class Projection2D(ABC):

    def __init__(self):
        """
        Initialize an abstract 2D projection.

        The 2D projection is used to convert between real coordinates and a
        projection of those coordinates with respect to a reference position.
        """
        self._reference = self.get_coordinate_instance()

    def copy(self):
        """
        Return a copy of the projection.

        Returns
        -------
        Projection2D
        """
        return deepcopy(self)

    @property
    def reference(self):
        """
        Return the reference position.

        Returns
        -------
        Coordinate2D
        """
        return self.get_reference()

    @reference.setter
    def reference(self, value):
        """
        Set the reference position.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self.set_reference(value)

    def __eq__(self, other):
        """
        Check if this projection is equal to another.

        Parameters
        ----------
        other : Projection2D

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if not isinstance(other, Projection2D):
            return False
        if self.reference != other.reference:
            return False
        return True

    def get_reference(self):
        """
        Return the reference position.

        Returns
        -------
        reference : Coordinate2D
        """
        return self._reference

    def set_reference(self, value):
        """
        Set the reference position.

        Parameters
        ----------
        value : Coordinate2D

        Returns
        -------
        None
        """
        self._reference = value

    def get_projected(self, coordinates):
        """
        Return the projected coordinates.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        projected : Coordinate2D
        """
        return self.project(coordinates, projected=Coordinate2D())

    def get_deprojected(self, projected):
        """
        Return the de-projected coordinates.

        Parameters
        ----------
        projected : Coordinate2D

        Returns
        -------
        coordinates : Coordinate2D
        """
        return self.deproject(projected, coordinates=Coordinate2D())

    @abstractmethod
    def get_coordinate_instance(self):  # pragma: no cover
        """
        Return a coordinate instance relevant to the projection.

        Returns
        -------
        coordinates : Coordinate2D
        """
        pass

    @abstractmethod
    def project(self, coordinates, projected=None):  # pragma: no cover
        """
        Project the coordinates.

        Converts coordinates to offsets w.r.t. a reference position.

        Parameters
        ----------
        coordinates : Coordinate2D
            The coordinates to project.
        projected : Coordinate2D, optional
            The output coordinates.  Will be created if not supplied.

        Returns
        -------
        projected : Coordinate2D
            The projected coordinates.
        """
        pass

    @abstractmethod
    def deproject(self, projected, coordinates=None):  # pragma: no cover
        """
        Deproject a projection onto coordinates.

        Converts offsets w.r.t a reference position to coordinates.

        Parameters
        ----------
        projected : Coordinate2D
            The projected coordinates to deproject.
        coordinates : Coordinate2D, optional
            The output deprojected coordinates.  Will be created if not
            supplied.

        Returns
        -------
        coordinates : Coordinate2D
            The deprojected coordinates.
        """
        pass

    @abstractmethod
    def parse_header(self, header, alt=''):  # pragma: no cover
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
        pass

    @abstractmethod
    def edit_header(self, header, alt=''):  # pragma: no cover
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
        pass

    @abstractmethod
    def get_fits_id(self):  # pragma: no cover
        """
        Return the FITS ID for the projection.

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def get_full_name(self):  # pragma: no cover
        """
        Return the full name for the projection.

        Returns
        -------
        str
        """
        pass
