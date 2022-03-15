# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from copy import deepcopy
import numpy as np

__all__ = ['Asymmetry2D']


class Asymmetry2D(ABC):

    def __init__(self, x=None, y=None, x_weight=0.0, y_weight=0.0):
        """
        Initialize an Asymmetry2D object.

        The 2-dimensional asymmetry is a simple container for asymmetry
        measurements of (x, y) data.  It is used to provide informational
        strings and significance/RMS values when requested.

        Parameters
        ----------
        x : float
            The asymmetry in the x-direction.
        y : float
            The asymmetry in the y-direction.
        x_weight : float
            The asymmetry weight (1/variance) in the x-direction.
        y_weight : float
            The asymmetry weight (1/variance) in the y-direction.
        """
        self.x = x
        self.x_weight = x_weight
        self.y = y
        self.y_weight = y_weight

    def copy(self):
        """
        Return a copy of the Asymmetry2D object.

        Returns
        -------
        Asymmetry2D
        """
        return deepcopy(self)

    def __eq__(self, other):
        """
        Return whether this asymmetry is equal to another.

        Parameters
        ----------
        other : Asymmetry2D

        Returns
        -------
        bool
        """
        if self is other:
            return True
        if not isinstance(other, Asymmetry2D):
            return False
        if self.x != other.x or self.y != other.y:
            return False
        return True

    def __str__(self):
        """
        Return a string representation of the asymmetry.

        Returns
        -------
        str
        """
        if self.x is None and self.y is None:
            return "Asymmetry: empty"

        result = 'Asymmetry: '
        if self.x is not None:
            result += f'x = {self.x * 100:.3f}% +- {self.x_rms * 100:.3f}%'
        if self.y is not None:
            if self.x is not None:
                result += ', '
            result += f'y = {self.y * 100:.3f}% +- {self.y_rms * 100:.3f}%'
        return result

    def __repr__(self):
        """
        Return a string representation of the Asymmetry instance.

        Returns
        -------
        str
        """
        return f'{object.__repr__(self)} {self}'

    @property
    def x_significance(self):
        """
        Return the significance of the asymmetry in x.

        Returns
        -------
        significance : float
        """
        if self.x_weight is None:
            return np.inf
        return np.abs(self.x) * np.sqrt(self.x_weight)

    @property
    def y_significance(self):
        """
        Return the significance of the asymmetry in x.

        Returns
        -------
        significance : float
        """
        if self.y_weight is None:
            return np.inf
        return np.abs(self.y) * np.sqrt(self.y_weight)

    @property
    def x_rms(self):
        """
        Return the root-mean-square of the asymmetry in the x-direction.

        Returns
        -------
        rms : float
        """
        if self.x_weight is not None:
            x_rms = np.sqrt(1 / self.x_weight)
        else:
            x_rms = 0.0
        return x_rms

    @property
    def y_rms(self):
        """
        Return the root-mean-square of the asymmetry in the y-direction.

        Returns
        -------
        rms : float
        """
        if self.y_weight is not None:
            y_rms = np.sqrt(1 / self.y_weight)
        else:
            y_rms = 0.0
        return y_rms
