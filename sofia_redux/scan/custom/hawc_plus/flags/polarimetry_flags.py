# Licensed under a 3-clause BSD style license - see LICENSE.rst

import enum
import numpy as np

__all__ = ['HawcPlusPolarModulation']


class HawcPlusPolarModulation(enum.Enum):
    """
    The polar modulation class contains Stokes parameter and gain scaling.
    """
    N = enum.auto()
    Q = enum.auto()
    U = enum.auto()
    V = enum.auto()

    @classmethod
    def get_n_gain(cls):
        """
        Returns the scaling parameter for Stokes N.

        Returns
        -------
        gain_factor : float
        """
        return 1.0

    @classmethod
    def get_qu_gain(cls):
        """
        Returns the scaling parameter for Stokes Q and U.

        Returns
        -------
        gain_factor : float
        """
        return 1.0

    @classmethod
    def get_gain_factor(cls, signal_mode):
        """
        Return the gain scaling factor for the given signal mode

        Parameters
        ----------
        signal_mode : enum.Enum

        Returns
        -------
        gain_factor : float
        """
        if signal_mode == cls.N or signal_mode == cls.V:
            return cls.get_n_gain()
        elif signal_mode == cls.Q or signal_mode == cls.U:
            return cls.get_qu_gain()
