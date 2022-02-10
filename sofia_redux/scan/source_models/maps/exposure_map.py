# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.source_models.maps.overlay import Overlay

__all__ = ['ExposureMap']


class ExposureMap(Overlay):

    def __init__(self, observation=None):
        """
        Create a weight map overlay of an observation.

        Parameters
        ----------
        observation : Observation2D, optional
            The observation map from which to generate an exposure map.
        """
        super().__init__(data=observation)

    @property
    def data(self):
        """
        Return the data values of the exposure map.

        Returns
        -------
        numpy.ndarray
        """
        exposure = self.basis.exposure
        if exposure is None:
            return None
        return exposure.data

    @data.setter
    def data(self, values):
        """
        Set the exposure data.

        Parameters
        ----------
        values : numpy.ndarray or FlaggedArray

        Returns
        -------
        None
        """
        self.basis.exposure.data = values

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
