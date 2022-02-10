# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.channels.camera.color_arrangement import ColorArrangement

__all__ = ['SingleColorArrangement']


class SingleColorArrangement(ColorArrangement):

    def get_pixel_count(self):
        """
        Return the number of pixels.

        Returns
        -------
        int
        """
        return self.size

    def get_pixels(self):
        """
        Return the pixel data.

        Returns
        -------
        ChannelData
        """
        return self.data

    def get_mapping_pixels(self, discard_flag=None, keep_flag=None,
                           match_flag=None, name='mapping-pixels'):
        """
        Return the mapping pixels.

        Parameters
        ----------
        discard_flag : int or str or ChannelFlagTypes
        keep_flag : int or str or ChannelFlagTypes
        match_flag : int or str or ChannelFlagTypes
        name : str, optional
            The name of the returned channel group.  The default is
            'mapping-pixels'.

        Returns
        -------
        ChannelGroup
            The mapping channel group.
        """
        obs_channels = self.get_observing_channels()
        return obs_channels.create_data_group(discard_flag=discard_flag,
                                              keep_flag=keep_flag,
                                              match_flag=match_flag,
                                              name=name)
