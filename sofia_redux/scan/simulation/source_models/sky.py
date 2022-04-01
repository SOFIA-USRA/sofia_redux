# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from copy import deepcopy

from sofia_redux.scan.simulation.source_models.simulated_source import (
    SimulatedSource)
from sofia_redux.scan.source_models.sky_dip_model import SkyDipModel

__all__ = ['Sky']


class Sky(SimulatedSource):

    def __init__(self, **kwargs):
        """
        Initialize a simulated sky source.

        The sky source model may only operate on horizontal type coordinate
        systems, since only elevation is of concern.  The sky is modelled
        as::

          data = offset - ((exp(-tau / sin(elevation)) - 1) * tsky * scaling)

        Where offset is a DC temperature offset in Kelvins, tsky is the
        sky temperature in Kelvins, tau is the atmospheric opacity, and scaling
        gives the appropriate temperature scaling for the instrument.  If not
        supplied, the defaults are tau = 0.1, offset = 0K, tsky = 273K, and
        scaling = 1.

        Parameters
        ----------
        kwargs : dict, optional
        """
        super().__init__()
        self.name = 'sky'
        self.model = SkyDipModel()
        self.tau = 0.1
        self.t_offset = 0.0 * units.Unit('Kelvin')
        self.scaling = 1.0
        self.t_sky = 273 * units.Unit('Kelvin')
        self.initialize_model(**kwargs)

    def initialize_model(self, **kwargs):
        """
        Initialize the model with the provided options.

        Parameters
        ----------
        kwargs : dict, optional
            The available keys are 'tau', 'scaling', 'tsky', and 'offset'.
            tsky and offset will be converted to Kelvins if not supplied as
            `units.Quantity` values.

        Returns
        -------
        None
        """
        options = deepcopy(kwargs)
        self.tau = float(options.get('tau', 0.1))
        self.scaling = float(options.get('scaling', 1.0))
        self.t_sky = options.get('tsky', 273)
        if not isinstance(self.t_sky, units.Quantity):
            self.t_sky = self.t_sky * units.Unit('Kelvin')
        self.t_offset = options.get('offset', 0.0)
        if not isinstance(self.t_offset, units.Quantity):
            self.t_offset = self.t_offset * units.Unit('Kelvin')

    def apply_to_offsets(self, offsets):
        """
        Apply the source model to a set of 2-D offsets.

        Parameters
        ----------
        offsets : Coordinate2D
            Equatorial native offsets.

        Returns
        -------
        data : numpy.ndarray
            The modelled data of the source given the offsets.
        """
        raise NotImplementedError("Can only determine sky from horizontal "
                                  "coordinates.")

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
        elevation = horizontal.el.to('radian').value
        tau = self.tau
        offset = self.t_offset.to('Kelvin').value
        kelvin = self.scaling
        tsky = self.t_sky.to('Kelvin').value
        return self.model.value_at(elevation, tau, offset, kelvin, tsky)
