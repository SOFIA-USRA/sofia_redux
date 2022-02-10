# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from sofia_redux.scan.channels.camera.single_color_arrangement import \
    SingleColorArrangement
from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)

__all__ = ['ExampleChannels']


class ExampleChannels(SingleColorArrangement):

    def __init__(self, parent=None, info=None, size=0, name='example'):
        super().__init__(name=name, parent=parent, info=info, size=size)

    def init_divisions(self):
        """
        Initializes channel divisions.

        Divisions contain sets of channel groups.

        Each divisions is composed of channel groups where all channels in
        a group contain a unique value of a certain data field.  For example,
        the "rows" division contains a group for row 1, a group for row 2, etc.

        Returns
        -------
        None
        """
        super().init_divisions()
        dead_blind = self.flagspace.flags.DEAD | self.flagspace.flags.BLIND

        mux_division = self.get_division(name='mux', field='mux',
                                         discard_flag=dead_blind)
        bias_division = self.get_division(name='bias', field='bias_line',
                                          discard_flag=dead_blind)
        self.add_division(mux_division)
        self.add_division(bias_division)

    def init_modalities(self):
        """
        Initializes channel modalities.

        A modality is based of a channel division and contains a mode for each
        channel group in the channel division.

        Here, we add mux and bias lines to the correlated modalities.

        Returns
        -------
        None
        """
        super().init_modalities()

        mux_modality = CorrelatedModality(
            name='mux', identity='m',
            channel_division=self.divisions.get('mux'),
            gain_provider='mux_gain')
        mux_modality.set_gain_flag(self.flagspace.flags.MUX)
        self.add_modality(mux_modality)

        bias_modality = CorrelatedModality(
            name='bias', identity='b',
            channel_division=self.divisions.get('bias'),
            gain_provider='bias_gain')
        bias_modality.set_gain_flag(self.flagspace.flags.BIAS)
        self.add_modality(bias_modality)

    def read_data(self, hdul):
        """
        Read and apply channel data from an HDU list.

        Parameters
        ----------
        hdul : fits.HDUList

        Returns
        -------
        None
        """
        pass

    def load_channel_data(self):
        """
        Load the channel data.

        The pixel data and wiring data files should be defined in the
        configuration.

        Returns
        -------
        None
        """
        pixel_data_file = self.configuration.get_string(
            'pixeldata', default='auto')
        if pixel_data_file.lower().strip() == 'auto':
            log.info("Initializing channel data from detector array info.")
            self.info.detector_array.initialize_channel_data(self.data)
        else:
            pixel_data_file = self.configuration.priority_file('pixeldata')
            if pixel_data_file is None:
                log.error(f"Pixel data file "
                          f"{self.configuration.get('pixeldata')} not found.")
                log.warning("Cannot read pixel data. "
                            "Using default gains and flags.")
                self.info.detector_array.initialize_channel_data(self.data)
            else:
                self.read_pixel_data(pixel_data_file)

        if 'wiring' in self.configuration:
            wiring_data_file = self.configuration.priority_file('wiring')
            if wiring_data_file is None:
                log.warning(f"Wiring data file "
                            f"{self.configuration.get('wiring')} not found.")
                log.warning("Cannot read wiring data. "
                            "Specific channel divisions not established.")
            else:
                self.read_wiring_data(wiring_data_file)

    def get_si_pixel_size(self):
        """
        Return the science instrument pixel size

        Returns
        -------
        x, y : Coordinate2D
            The (x, y) pixel sizes
        """
        return self.info.detector_array.pixel_sizes
