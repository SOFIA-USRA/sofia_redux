# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
from copy import deepcopy
import numpy as np

__all__ = ['InstantFocus']


class InstantFocus(ABC):

    def __init__(self):
        """
        Initialize an InstantFocus object.

        The instant focus is designed to store focus measurements for later
        access.  Focus measurements may be derived from the configuration,
        asymmetry, and elliptical elongation parameters.
        """
        self.x = None
        self.x_weight = None
        self.y = None
        self.y_weight = None
        self.z = None
        self.z_weight = None

    def copy(self):
        """
        Return a copy of the focus.

        Returns
        -------
        InstantFocus
        """
        return deepcopy(self)

    def copy_from(self, focus):
        """
        Copy the focus from another focus object.

        Parameters
        ----------
        focus : InstantFocus

        Returns
        -------
        None
        """
        self.x = focus.x
        self.y = focus.y
        self.z = focus.z
        self.x_weight = focus.x_weight
        self.y_weight = focus.y_weight
        self.z_weight = focus.z_weight

    @property
    def is_valid(self):
        """
        Return whether the focus is valid.

        For the focus to be valid, it must contain at least one value in the
        x, y, or z attribute (not `None`).

        Returns
        -------
        bool
        """
        if self.x is not None or self.y is not None or self.z is not None:
            return True
        return False

    @property
    def is_complete(self):
        """
        Return whether all focus parameters are available.

        Returns
        -------
        bool
        """
        if self.x is None or self.y is None or self.z is None:
            return False
        return True

    def __str__(self):
        """
        Return a string representation of the focus.

        Returns
        -------
        str
        """
        if not self.is_valid:
            return "No focus results"
        info = []
        for param in ['x', 'y', 'z']:
            v, w = getattr(self, param), getattr(self,  f'{param}_weight')
            if v is None:
                continue
            unit = None
            if isinstance(v, units.Quantity):
                unit = v.unit
                v = v.value
            if isinstance(w, units.Quantity):
                w = w.value
            if w is None or np.isnan(w) or w == 0:
                rms = None
            else:
                rms = 1 / np.sqrt(w)
            s = f'{param}={v:.6f}'
            if rms is not None:
                s += f'+-{rms:.6f}'
            if unit is not None:
                s += f' {unit}'
            info.append(s)
        return f'Focus results: {" ".join(info)}'

    def __repr__(self):
        """
        Return a string representation of the focus instance.

        Returns
        -------
        str
        """
        return f'{object.__repr__(self)} {self}'

    def __eq__(self, other):
        """
        Check if this InstantFocus is equal to another.

        Parameters
        ----------
        other : InstantFocus

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if self.x != other.x or self.x_weight != other.x_weight:
            return False
        if self.y != other.y or self.y_weight != other.y_weight:
            return False
        if self.z != other.z or self.z_weight != other.z_weight:
            return False
        return True

    def derive_from(self, configuration, asymmetry=None, elongation=None,
                    elongation_weight=None):
        """
        Derive the focus from the asymmetry and elongation measurements.

        The focus results are derived from the (x, y) asymmetry measurement,
        and from the elongation in the z-direction.

        Parameters
        ----------
        configuration : Configuration
            The SOFSCAN configuration options.
        asymmetry : Asymmetry2D, optional
            The source asymmetry.
        elongation : float, optional
            The source elongation.
        elongation_weight : float, optional
            The source elongation weight.

        Returns
        -------
        None
        """
        self.x = self.y = self.z = None
        self.x_weight = self.y_weight = self.z_weight = None

        mm = units.Unit('mm')
        s2n = configuration.get_float('focus.significance', default=2.0)

        if elongation is not None:
            elongation -= configuration.get_float(
                'focus.elong0', default=0.0) * 0.01

        for direction in ['x', 'y', 'z']:
            coeff = configuration.get_float(f'focus.{direction}coeff',
                                            default=np.nan)
            if np.isnan(coeff) or coeff == 0:
                continue

            if direction == 'z':
                v = elongation
                w = elongation_weight
            elif asymmetry is None:
                continue
            else:
                v = getattr(asymmetry, direction)
                w = getattr(asymmetry, f'{direction}_weight')

            if v is None or w is None:
                continue

            significance = np.abs(v) * np.sqrt(w)
            if significance <= s2n:
                continue

            scale = -mm / coeff
            v *= scale
            w /= scale * scale

            scatter = configuration.get_float(f'focus.{direction}scatter',
                                              default=np.nan) * mm
            if not np.isnan(scatter):
                variance = (1.0 / w) + (scatter ** 2)
                w = 1.0 / variance

            setattr(self, direction, v)
            setattr(self, f'{direction}_weight', w)
