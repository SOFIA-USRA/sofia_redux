# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log
import numpy as np

from sofia_redux.scan.channels.camera.camera import Camera

__all__ = ['ColorArrangement']


class ColorArrangement(Camera):

    def apply_configuration(self):
        """
        Apply the configuration options to the instrument channels.

        Returns
        -------
        None
        """
        super().apply_configuration()
        beam = self.configuration.get_float('beam', default=np.nan)
        if np.isfinite(beam):
            self.info.resolution = beam * self.info.instrument.get_size_unit()
            self.data.set_beam_size(self.info.resolution)

        elif self.configuration.has_option('beam'):
            alias = self.configuration.get_string('beam')
            if not self.configuration.has_option('beam'):
                log.warning(f"Could not parse 'beam' configuration value "
                            f"({alias}).")
                return
            beam = self.configuration.get_float(alias, default=np.nan)
            if np.isfinite(beam):
                self.info.resolution = (
                    beam * self.info.instrument.get_size_unit())
                self.data.set_beam_size(self.info.resolution)

    def get_min_beam_fwhm(self):
        """
        Return the minimum FWHM of the beam.

        Returns
        -------
        astropy.units.Quantity
            The unit type depends on the instrument size unit.
        """
        return np.nanmin(self.get_pixels().resolution)

    def get_max_beam_fwhm(self):
        """
        Return the maximum FWHM of the beam.

        Returns
        -------
        astropy.units.Quantity
            The unit type depends on the instrument size unit.
        """
        return np.nanmax(self.get_pixels().resolution)

    def get_average_beam_fwhm(self):
        """
        Return the average FWHM of the beam.

        Returns
        -------
        astropy.units.Quantity
            The unit type depends on the instrument size unit.
        """
        return np.nanmean(self.get_pixels().resolution)

    @abstractmethod
    def get_pixel_count(self):
        """
        Return the number of pixels.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_pixels(self):
        """
        Return the pixel data.

        Returns
        -------
        ChannelData
        """
        pass

    @abstractmethod
    def get_mapping_pixels(self, discard_flag=None, keep_flag=None,
                           match_flag=None):
        """
        Return the mapping pixels.

        Parameters
        ----------
        discard_flag : int or str or ChannelFlagTypes
        keep_flag : int or str or ChannelFlagTypes
        match_flag : int or str or ChannelFlagTypes

        Returns
        -------
        ChannelGroup
        """
        pass
