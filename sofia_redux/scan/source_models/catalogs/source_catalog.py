# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
import inspect

__all__ = ['SourceCatalog']


class SourceCatalog(ABC):

    def __init__(self, coordinate_class):
        """
        The source catalog contains an array of sources modelled as
        2-dimensional Gaussians.

        Parameters
        ----------
        coordinate_class : object or class
            The coordinate class.  Generally, an astropy coordinate scheme
            such as `astropy.coordinates.sky_coordinate.SkyCoord`.
        """
        if inspect.isclass(coordinate_class):
            self.coordinate_class = coordinate_class
        else:
            self.coordinate_class = coordinate_class.__class__

        self.sources = []

    def insert(self, image):
        """
        Insert the sources onto an image.

        Parameters
        ----------
        image : Map2D

        Returns
        -------
        None
        """
