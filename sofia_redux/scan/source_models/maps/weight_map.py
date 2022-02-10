# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.source_models.maps.overlay import Overlay

__all__ = ['WeightMap']


class WeightMap(Overlay):

    def __init__(self, observation):
        """
        Create a weight map overlay of an observation.

        Parameters
        ----------
        observation : Observation2D
        """
        super().__init__(data=observation)

    @property
    def data(self):
        """
        Return the data values of the weight map.

        Returns
        -------
        data_values : numpy.ndarray
        """
        weight = self.basis.weight
        if weight is None:
            return None
        return weight.data

    @data.setter
    def data(self, values):
        """
        Set the weight data.

        Parameters
        ----------
        values : numpy.ndarray or FlaggedArray

        Returns
        -------
        None
        """
        self.basis.weight.data = values

    def discard(self, indices=None):
        """
        Set the flags for discarded indices to DISCARD and data to zero.

        Parameters
        ----------
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        self.basis.discard(indices=indices)
