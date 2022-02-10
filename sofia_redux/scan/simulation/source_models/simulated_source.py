# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod

from sofia_redux.scan.utilities.class_provider \
    import get_simulated_source_class

__all__ = ['SimulatedSource']


class SimulatedSource(ABC):

    def __init__(self):
        self.name = 'base'

    @abstractmethod
    def apply_to_offsets(self, offsets):
        """
        Apply the model to a set of offset coordinates.

        Parameters
        ----------
        offsets : Coordinate2D

        Returns
        -------
        generated_model : numpy.ndarray
        """
        pass

    def __call__(self, offsets):
        """
        Generate a source model for the applied offsets.

        Parameters
        ----------
        offsets : Coordinate2D

        Returns
        -------
        generated_model : numpy.ndarray
        """
        return self.apply_to_offsets(offsets)

    @staticmethod
    def get_source_model(name, **kwargs):
        """
        Return an initialize source model of the given name.

        Parameters
        ----------
        name : str
            The name of the synthetic source model.
        kwargs : dict, optional
            Optional keyword arguments during source initialization.

        Returns
        -------
        source_model : SimulatedSource
        """
        source_class = get_simulated_source_class(name)
        return source_class(**kwargs)
