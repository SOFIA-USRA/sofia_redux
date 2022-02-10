# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units
import numpy as np

from sofia_redux.scan.channels.channel_data.color_arrangement_data import (
    ColorArrangementData)

__all__ = ['SofiaChannelData']


class SofiaChannelData(ColorArrangementData):

    def __init__(self, channels=None):
        super().__init__(channels=channels)

    @property
    def info(self):
        """
        Return the instrument information object.

        Returns
        -------
        SofiaInfo
        """
        return super().info

    def apply_info(self, info):
        """
        Initialize data from available information.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        super().apply_info(info)
        angular_resolution = info.instrument.angular_resolution
        if isinstance(angular_resolution, units.Quantity):
            angular_unit = angular_resolution.unit
            if angular_unit != units.dimensionless_unscaled:
                angular_unit = units.Unit('radian')
            angular_resolution = angular_resolution.value
        else:
            angular_unit = units.Unit('radian')
        self.angular_resolution = np.full(self.size, angular_resolution,
                                          dtype=float) * angular_unit

        frequency = info.instrument.frequency
        if isinstance(frequency, units.Quantity):
            frequency_unit = frequency.unit
            if frequency_unit != units.dimensionless_unscaled:
                frequency_unit = units.Unit('Hz')
            frequency = frequency.value
        else:
            frequency_unit = units.Unit('Hz')
        self.frequency = np.full(self.size, frequency,
                                 dtype=float) * frequency_unit

    @abstractmethod
    def read_channel_data_file(self, filename):
        """
        Read a channel data file and return the information within.

        Parameters
        ----------
        filename : str
            The path to a channel data file.

        Returns
        -------
        channel_info : pandas.DataFrame
        """
        pass
