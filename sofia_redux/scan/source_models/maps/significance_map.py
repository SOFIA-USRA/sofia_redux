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

        The exposure map overlay returns and operates on the data and weight
        images of the Observation2D basis.

        Parameters
        ----------
        observation : Observation2D, optional
            The observation map from which to overlay significance.
        """
        super().__init__(data=observation)

    def set_default_unit(self):
        """
        Set the default unit for the map significance.

        The default unit will always be dimensionless (a number).

        Returns
        -------
        None
        """
        super().set_unit(1 * units.dimensionless_unscaled)

    def set_unit(self, unit):
        """
        Set the unit for the map (Not available for S2N map).

        The signal-to-noise ratio will always be expressed in dimensionless
        units (as a number).

        Parameters
        ----------
        unit : str or units.Unit or units.Quantity

        Returns
        -------
        None
        """
        super().set_unit(1 * units.dimensionless_unscaled)

    @property
    def data(self):
        """
        Return the data values of the significance map.

        The significance (S2N or signal-to-noise ratio) is given by::

            s2n = data * sqrt(weight)

        where weight should be equivalent to 1/variance.

        Returns
        -------
        numpy.ndarray
        """

        weight = self.basis.weight
        if weight is None or weight.data is None:
            return None
        data = self.basis.data
        if data is None:
            return None

        return data * np.sqrt(weight.data)

    @data.setter
    def data(self, significance):
        """
        Set the significance data values.

        This results in the basis image data being set to::

            data = noise * significance

        where noise is calculated as the 1/sqrt(weight) values of the
        basis observation.

        Parameters
        ----------
        significance : numpy.ndarray or FlaggedArray

        Returns
        -------
        None
        """
        noise = NoiseMap(self.basis).data
        if noise is None:  # pragma: no cover
            return
        if isinstance(significance, FlaggedArray):
            data = noise * significance.data
            valid = significance.valid
            data[~valid] = self.blanking_value
        else:
            data = noise * significance
        self.basis.data = data
