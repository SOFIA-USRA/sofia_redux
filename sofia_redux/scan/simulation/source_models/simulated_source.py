# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod
from copy import deepcopy

from sofia_redux.scan.utilities.class_provider \
    import get_simulated_source_class
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates

__all__ = ['SimulatedSource']


class SimulatedSource(ABC):

    def __init__(self):
        """
        Initialize a simulated source.

        This is an abstract class base for simulating source data given input
        positions.  Currently equatorial offsets and horizontal coordinates
        may be converted to timestream data.
        """
        self.name = 'base'

    def __call__(self, coordinates):
        """
        Generate a source model for the applied offsets.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        generated_model : numpy.ndarray
        """
        if isinstance(coordinates, HorizontalCoordinates):
            return self.apply_to_horizontal(coordinates)
        else:
            return self.apply_to_offsets(coordinates)

    def copy(self):
        """
        Return a copy of the simulated source.

        Returns
        -------
        SimulatedSource
        """
        return deepcopy(self)

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

    @abstractmethod
    def apply_to_offsets(self, offsets):  # pragma: no cover
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

    @abstractmethod
    def apply_to_horizontal(self, offsets):  # pragma: no cover
        """
        Apply the model to a set of offset coordinates.

        Parameters
        ----------
        offsets : HorizontalCoordinates

        Returns
        -------
        generated_model : numpy.ndarray
        """
        pass
