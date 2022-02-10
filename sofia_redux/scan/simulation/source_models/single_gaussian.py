# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.modeling.functional_models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma
from copy import deepcopy

from sofia_redux.scan.simulation.source_models.simulated_source import (
    SimulatedSource)

__all__ = ['SingleGaussian']


class SingleGaussian(SimulatedSource):

    def __init__(self, **kwargs):
        """
        Initialize a SingleGaussian simulated source.

        Parameters
        ----------
        kwargs : dict
            Optional keyword arguments to pass into :class:`Gaussian2D`.  Note
            that width settings MUST be provided.  The width parameters can
            take the values {x_stddev, y_stddev, fwhm, x_fwhm, y_fwhm}.
        """
        super().__init__()
        self.model = None
        self.initialize_model(**kwargs)

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
        options = deepcopy(kwargs)
        arcsec = units.Unit('arcsec')
        if 'amplitude' not in options:
            options['amplitude'] = 1.0
        if 'x_mean' not in options:
            options['x_mean'] = 0.0 * arcsec
        if 'y_mean' not in options:
            options['y_mean'] = 0.0 * arcsec

        if 'fwhm' in options:
            fwhm = options['fwhm']
            sigma = fwhm * gaussian_fwhm_to_sigma
            options['x_stddev'] = sigma
            options['y_stddev'] = sigma
        elif 'stddev' in options:
            options['x_stddev'] = options['stddev']
            options['y_stddev'] = options['stddev']

        if 'x_fwhm' in options:
            options['x_stddev'] = gaussian_fwhm_to_sigma * options['x_fwhm']
        if 'y_fwhm' in options:
            options['y_stddev'] = gaussian_fwhm_to_sigma * options['y_fwhm']

        if 'info' in options and ('x_stddev' not in options
                                  or 'y_stddev' not in options):
            info = options['info']
            options['x_stddev'] = info.instrument.resolution
            options['y_stddev'] = info.instrument.resolution

        final_options = {}
        valid_options = ['amplitude', 'x_mean', 'y_mean', 'x_stddev',
                         'y_stddev', 'theta', 'cov_matrix']

        for key, value in options.items():
            if key in valid_options:
                final_options[key] = value

        if 'x_stddev' not in final_options or 'y_stddev' not in final_options:
            raise ValueError("Gaussian width parameter has not been supplied.")

        self.model = Gaussian2D(**final_options)

    def apply_to_offsets(self, offsets):
        """
        Apply the source model to a set of 2-D offsets.

        Parameters
        ----------
        offsets : Coordinate2D.

        Returns
        -------
        data : numpy.ndarray
            The modelled data of the source given the offsets.
        """
        return self.model(offsets.x, offsets.y)
