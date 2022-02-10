# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from abc import ABC
from astropy import units

__all__ = ['BracketedValues']


class BracketedValues(ABC):

    def __init__(self, start=np.nan, end=np.nan, unit=None):
        """
        Initialize a BracketedValues object.

        The bracketed values class is used to represent a range, and
        may perform operations with other BracketedValues objects.

        Parameters
        ----------
        start : int or float or units.Quantity, optional
            The starting value of the range.
        end : int or float or units.Quantity, optional
            The ending value of the range.
        unit : units.Unit or str, optional
            The unit of the range.
        """
        if unit is None:
            if isinstance(start, units.Quantity):
                unit = start.unit
            elif isinstance(end, units.Quantity):
                unit = end.unit
        else:
            unit = units.Unit(unit)

        self.unit = unit
        if self.unit is not None:
            if not isinstance(start, units.Quantity):
                start = start * self.unit
            else:
                start = start.to(self.unit)
            if not isinstance(end, units.Quantity):
                end = end * self.unit
            else:
                end = end.to(self.unit)

        self.start = start
        self.end = end

    def __str__(self):
        """
        Return a string representation of the bracket.

        Returns
        -------
        str
        """
        if self.unit is None:
            return f"({self.start} --> {self.end})"
        return f"({self.start.value} --> {self.end.value} {self.unit})"

    def __repr__(self):
        """
        Return the canonical string representation of the object, and values.

        Returns
        -------
        str
        """
        return f'{object.__repr__(self)} {str(self)}'

    @property
    def midpoint(self):
        """
        Return the midpoints of the bracket.

        The midpoint is the mean of the start and end points of the bracket.

        Returns
        -------
        float or units.Quantity
        """
        if np.isnan(self.start):
            return self.end
        elif np.isnan(self.end):
            return self.start
        return 0.5 * (self.start + self.end)

    def copy(self):
        """
        Return a copy of the bracket.

        Returns
        -------
        BracketedValues
        """
        return BracketedValues(self.start, self.end)

    def merge(self, other):
        """
        Combine with another bracket.

        The resulting (start, end) points of the bracket will be
        (min[self.start, other.start], max[self.end, other.end]).

        Parameters
        ----------
        other : BracketedValues

        Returns
        -------
        None
        """
        if other.start < self.start:
            if self.unit is None:
                self.start = other.start
            else:
                self.start = other.start.to(self.unit)
        if other.end > self.end:
            if self.unit is None:
                self.end = other.end
            else:
                self.end = other.end.to(self.unit)
