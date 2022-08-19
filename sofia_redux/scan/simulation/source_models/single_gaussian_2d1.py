# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.modeling.functional_models import Gaussian1D
from astropy.stats import gaussian_fwhm_to_sigma
from copy import deepcopy

from sofia_redux.scan.simulation.source_models.single_gaussian import (
    SingleGaussian)
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['SingleGaussian2d1']


class SingleGaussian2d1(SingleGaussian):

    def __init__(self, **kwargs):
        """
        Initialize a SingleGaussian simulated source.

        Models the source as a single Gaussian that may be applied to any
        2-dimensional coordinates with the exception of horizontal type.

        Parameters
        ----------
        kwargs : dict
            Optional keyword arguments to pass into :class:`Gaussian2D`.  Note
            that width settings MUST be provided.  The width parameters can
            take the values {x_stddev, y_stddev, fwhm, x_fwhm, y_fwhm}.
        """
        self.z_model = None
        super().__init__(**kwargs)
        self.name = 'single_gaussian_2d1'

    def copy(self):
        """
        Return a copy of the SingleGaussian2d1.

        Returns
        -------
        SingleGaussian2d1
        """
        return super().copy()

    def initialize_model(self, **kwargs):
        """
        Initialize the model with the provided options.

        Parameters
        ----------
        kwargs : dict, optional

        Returns
        -------
        None
        """
        super().initialize_model(**kwargs)
        options = deepcopy(kwargs)
        if 'z_mean' not in options:
            raise ValueError('Spectral center not specified')
        if 'z_fwhm' not in options and 'z_stddev' not in options:
            raise ValueError('Spectral width not specified')
        if 'z_fwhm' in options:
            z_sigma = options['z_fwhm'] * gaussian_fwhm_to_sigma
        else:
            z_sigma = options['z_stddev']

        final_options = {'amplitude': 1.0,
                         'mean': options['z_mean'],
                         'stddev': z_sigma}

        self.z_model = Gaussian1D(**final_options)

    def apply_to_offsets(self, offsets):
        """
        Apply the source model to a set of 3-D offsets.

        Parameters
        ----------
        offsets : Coordinate2D1
            Equatorial native offsets and wavelength.

        Returns
        -------
        data : numpy.ndarray
            The modelled data of the source given the offsets.
        """
        data = super().apply_to_offsets(offsets.xy_coordinates)
        data *= self.z_model(offsets.z_coordinates.coordinates)
        return data

    def apply_to_horizontal(self, horizontal):
        """
        Apply the source model to a set of 2-D offsets.

        Parameters
        ----------
        horizontal : HorizontalCoordinates
            HorizontalCoordinates.

        Returns
        -------
        data : numpy.ndarray
            The modelled data of the source given the offsets.
        """
        raise NotImplementedError("Cannot determine model from horizontal "
                                  "coordinates.")
