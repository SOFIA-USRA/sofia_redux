# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log, units
import numpy as np

from sofia_redux.scan.channels.channel_data.channel_data import ChannelData

__all__ = ['ColorArrangementData']


class ColorArrangementData(ChannelData):

    def set_beam_size(self, beam_size):
        """
        Sets the resolution of the channels to a specified beam size.

        Parameters
        ----------
        beam_size : astropy.units.Quantity

        Returns
        -------
        None
        """
        if not isinstance(beam_size, units.Quantity):
            raise ValueError(f"Beam size must be {units.Quantity}.")
        self.resolution = np.full(self.size, beam_size.value) * beam_size.unit

    def apply_info(self, info):
        """
        Apply information to the channel data.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        super().apply_info(info)
        if self.configuration.has_option('beam'):
            resolution = self.configuration.get_float('beam')
            if np.isnan(resolution):
                alias = self.configuration.get_string('beam')
                if self.configuration.has_option(alias):
                    resolution = self.configuration.get_float(alias)
                else:
                    log.warning(f"Could not parse configuration beam keyword "
                                f"value ({alias})")

            self.info.instrument.resolution = (
                resolution * self.info.instrument.get_size_unit())

    @abstractmethod
    def get_pixel_count(self):
        """
        Return the number of pixels in the channels.

        Returns
        -------
        pixels : int
        """
        pass

    @abstractmethod
    def get_pixels(self):
        """
        Return the pixels in the arrangement.

        Returns
        -------
        ChannelData
        """
        pass

    @abstractmethod
    def get_mapping_pixels(self, indices=None, name=None, keep_flag=None,
                           discard_flag=None, match_flag=None):
        """
        Creates and returns mapping pixels.

        Parameters
        ----------
        indices : numpy.ndarray (int), optional
            The indices to reference.  If not supplied, defaults to all
            channels.
        name : str, optional
            The name of the channel group.  If not supplied, defaults to the
            name of the channel data.
        discard_flag : int or str or ChannelFlagTypes, optional
            Flags to discard_flag from the new group.
        keep_flag : int or str or ChannelFlagTypes, optional
            Keep channels with these matching flags.
        match_flag : int or str or ChannelFlagTypes, optional
            Keep only channels with a flag exactly matching this flag.

        Returns
        -------
        ChannelGroup
            A newly created channel group.
        """
        pass
