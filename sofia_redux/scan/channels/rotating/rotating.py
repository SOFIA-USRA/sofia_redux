# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod

__all__ = ['Rotating']


class Rotating(ABC):

    @abstractmethod
    def get_rotation(self):
        """
        Return the channel rotation.

        Returns
        -------
        angle : units.Quantity
        """
        pass

    @abstractmethod
    def rotate(self, angle):
        """
        Rotate the channels.

        Parameters
        ----------
        angle : units.Quantity
            The angle to rotate by.

        Returns
        -------
        None
        """
        pass
