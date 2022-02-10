# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units

__all__ = ['CoordinateAxis']


class CoordinateAxis(ABC):

    def __init__(self, label='unspecified axis', short_label=None,
                 unit=units.dimensionless_unscaled):
        """
        Initialize a coordinate axis used for a coordinate system.

        One or more coordinate axes are used to define various coordinate
        systems.

        Parameters
        ----------
        label : str, optional
            The "long" name for the axis such as "Right Ascension".
        short_label : str, optional
            The "short" name for the axis such as "RA".
        unit : units.Unit or str, optional
            The units for the coordinate system.  The default is dimensionless.
        """
        self.label = label
        if short_label is None:
            self.short_label = label
        else:
            self.short_label = short_label
        self.reverse = False
        self.reverse_from = 0.0
        if isinstance(unit, str):
            unit = units.Unit(unit)
        self.unit = unit

    def __str__(self):
        """
        Return a string representation of the axis.

        Returns
        -------
        str
        """
        s = str(self.label)
        if self.short_label is not None:
            s += f' ({self.short_label})'
        return s
