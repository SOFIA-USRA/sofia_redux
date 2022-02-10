# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['Offset2D']


class Offset2D(Coordinate2D):

    def __init__(self, reference, coordinates=None, unit=None, copy=True):
        """

        Parameters
        ----------
        reference : Coordinate2D or numpy.ndarray or units.Quantity
            The reference (x, y) coordinate(s).
        coordinates : Coordinate2D or units.Quantity or numpy.ndarray, optional
            Offset coordinates from the reference position.
        unit
        copy
        """
        if unit is None:
            if isinstance(reference, (Coordinate2D, units.Quantity)):
                unit = reference.unit
            elif isinstance(coordinates, (Coordinate2D, units.Quantity)):
                unit = coordinates.unit

        if not isinstance(reference, Coordinate2D):
            self.reference = Coordinate2D(reference, unit=unit, copy=copy)
        else:
            if copy:
                self.reference = reference.copy()
            else:
                self.reference = reference

        coordinates = Coordinate2D(coordinates, unit=unit, copy=False)
        super().__init__(coordinates=None, unit=unit)
        self.copy_coordinates(coordinates)

    def __eq__(self, other):
        """
        Check whether these offsets are equal to another.

        Parameters
        ----------
        other : Offset2D

        Returns
        -------
        equal : bool
        """
        if other is self:
            return True
        if not isinstance(other, Offset2D):
            return False
        if other.reference != self.reference:
            return False
        return super().__eq__(other)

    def get_coordinate_class(self):
        """
        Return the coordinate class of the reference position.

        Returns
        -------
        class
        """
        return self.reference.__class__
