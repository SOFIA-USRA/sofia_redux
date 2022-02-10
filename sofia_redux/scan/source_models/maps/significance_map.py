# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np


from sofia_redux.scan.source_models.maps.overlay import Overlay
from sofia_redux.scan.source_models.maps.noise_map import NoiseMap
from sofia_redux.scan.flags.flagged_array import FlaggedArray

__all__ = ['SignificanceMap']


class SignificanceMap(Overlay):

    def __init__(self, observation=None):
        """
        Create a significance map overlay of an observation.

        Parameters
        ----------
        observation : Observation2D, optional
            The observation map from which to overlay significance.
        """
        super().__init__(data=observation)

    def set_default_unit(self):
        """
        Set the default unit for the map significance.

        Returns
        -------
        None
        """
        super().set_unit(1 * units.dimensionless_unscaled)

    def set_unit(self, unit):
        """
        Set the unit for the map (Not available for S2N map).

        Parameters
        ----------
        unit : astropy.units.Unit or astropy.units.Quantity

        Returns
        -------
        None
        """
        super().set_unit(1 * units.dimensionless_unscaled)

    @property
    def data(self):
        """
        Return the data values of the significance map.

        Returns
        -------
        numpy.ndarray
        """
        weight = self.basis.weight
        if weight is None:
            return None
        data = self.basis.data
        if data is None:
            return None

        return data * np.sqrt(weight.data)

    @data.setter
    def data(self, significance):
        """
        Set the significance data values.

        Parameters
        ----------
        significance : numpy.ndarray or FlaggedArray

        Returns
        -------
        None
        """
        noise = NoiseMap(self.basis).data
        if noise is None:
            return
        if isinstance(significance, FlaggedArray):
            data = noise * significance.data
            valid = significance.valid
            data[~valid] = self.blanking_value
        else:
            data = noise * significance
        self.basis.data = data
