# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.source_models.maps.overlay import Overlay
from sofia_redux.scan.flags.flagged_array import FlaggedArray

__all__ = ['NoiseMap']


class NoiseMap(Overlay):

    def __init__(self, observation=None):
        """
        Create a noise map overlay of an observation.

        The exposure map overlay returns and operates on the weight image of
        the Observation2D basis in terms of noise.

        Parameters
        ----------
        observation : Observation2D, optional
            The observation map from which to generate a noise map.
        """
        super().__init__(data=observation)

    @property
    def data(self):
        """
        Return the noise values as calculated from the basis weight.

        The noise values are calculated as::

            noise = 1 / sqrt(weight)

        Returns
        -------
        noise_values : numpy.ndarray
        """
        weight = self.basis.weight
        if weight is None:
            return None
        weight = weight.data
        noise = np.empty(self.shape, dtype=float)
        valid = self.basis.valid
        noise[valid] = 1.0 / np.sqrt(weight[valid])
        noise[~valid] = 0.0
        return noise

    @data.setter
    def data(self, values):
        """
        Set the weight values in the basis map from noise values.

        The weight values in the basis map are set via::

            weight = 1 / noise^2

        Parameters
        ----------
        values : numpy.ndarray or FlaggedArray

        Returns
        -------
        None
        """
        if isinstance(values, FlaggedArray):
            variance = values.data ** 2
            valid = values.valid
            variance[~valid] = 0
        else:
            variance = values * values
        nzi = variance > 0
        weight = np.empty(self.shape, dtype=self.basis.weight_dtype)
        weight[nzi] = 1.0 / variance[nzi]
        weight[~nzi] = 0.0
        self.basis.weight.set_data(weight)

    def set_default_unit(self):
        """
        Set the default unit for the map data.

        Returns
        -------
        None
        """
        self.set_unit(self.basis.unit)
