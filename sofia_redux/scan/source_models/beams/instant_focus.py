# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
from copy import deepcopy
import numpy as np

__all__ = ['InstantFocus']


class InstantFocus(ABC):

    def __init__(self):
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

    def derive_from(self, configuration, asymmetry=None, elongation=None,
                    elongation_weight=None):
        """
        Derive the focus.

        Parameters
        ----------
        configuration : Configuration
            The SOFSCAN configuration options.
        asymmetry : Asymmetry2D, optional
            The source asymmetry.
        elongation : float or units.Quantity, optional
            The source elongation.
        elongation_weight : float or units.Quantity, optional
            The source elongation weight.

        Returns
        -------
        None
        """
        self.x = self.y = self.z = None
        self.x_weight = self.y_weight = self.z_weight = None
        if asymmetry is None:
            return

        mm = units.Unit('mm')
        imm2 = 1 / (mm ** 2)
        s2n = configuration.get_float('focus.significance', default=2.0)

        if asymmetry is not None:
            for direction in ['x', 'y']:
                if configuration.is_configured(f'focus.{direction}coeff'):
                    v = getattr(asymmetry, direction)
                    w = getattr(asymmetry, f'{direction}_weight')
                    if v is None or w is None:
                        continue

                    if not isinstance(v, units.Quantity):
                        v = v * mm
                        w = w * imm2

                    setattr(self, direction, v)
                    setattr(self, f'{direction}_weight', w)
                    significance = np.abs(v) * np.sqrt(w)
                    if significance <= s2n:
                        continue
                    scale = configuration.get_float(f'focus.{direction}coeff')
                    v = v * scale * mm
                    w = w / (scale ** 2) * imm2
                    setattr(self, direction, v)
                    setattr(self, f'{direction}_weight', w)

                    if configuration.is_configured(
                            f'focus.{direction}scatter'):
                        scatter = configuration.get_float(
                            f'focus.{direction}scatter') * mm

                        variance = (1 / w) + (scatter ** 2)
                        w = 1 / variance
                        setattr(self, f'{direction}_weight', w)

        if elongation is not None and elongation_weight is not None:
            if configuration.is_configured('focus.elong0'):
                elongation -= 0.01 * configuration.get_float('focus.elong0')
            significance = np.abs(elongation) * np.sqrt(elongation_weight)
            if significance > s2n:
                if configuration.is_configured('focus.zcoeff'):
                    if not isinstance(elongation, units.Quantity):
                        elongation = elongation * mm
                        elongation_weight = elongation_weight * imm2
                    scale = configuration.get_float('focus.zcoeff')
                    z = elongation * scale
                    w = elongation_weight / (scale ** 2)
                    self.z = z
                    self.z_weight = w
                    if configuration.is_configured('focus.zscatter'):
                        scatter = configuration.get_float(
                            'focus.zscatter') * mm
                        variance = (1 / w) + (scatter ** 2)
                        w = 1 / variance
                        self.z_weight = w
