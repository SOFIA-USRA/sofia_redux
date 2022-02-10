# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod

from sofia_redux.scan.channels.camera.single_color_arrangement import \
    SingleColorArrangement

__all__ = ['SofiaCamera']


class SofiaCamera(SingleColorArrangement):

    def read_pixel_data(self, filename):
        """
        Read the pixel data file.

        If the instrument integration time is greater than zero, will set
        weighting accordingly.  Otherwise, standard weights are used.

        Parameters
        ----------
        filename : str
            Path to the pixel data file.

        Returns
        -------
        None
        """
        super().read_pixel_data(filename)
        self.info.register_config_file(filename)

    def read_rcp(self, filename):
        """
        Read and apply the RCP file information to channels (pixels).

        The RCP information is read and applied from a given file.  The RCP
        file should contain comma-separated values in one of following column
        formats:

        CHANNEL_INDEX, X_POSITION(arcsec), Y_POSITION(arcsec)
        CHANNEL_INDEX, GAIN, X_POSITION(arcsec), Y_POSITION(arcsec)
        CHANNEL_INDEX, GAIN, COUPLING, X_POSITION(arcsec), Y_POSITION(arcsec)

        All pixels not contained in the RCP file are flagged as BLIND, and will
        only be unflagged if a GAIN column is available in the file.  The
        channel coupling will be set to GAIN/COUPLING or GAIN/channel.gain
        depending on the column format, or ignored if not available.  X and Y
        positions are also set at this stage.

        If no RCP information is available (no file), these attributes should
        be set via other methods.

        Parameters
        ----------
        filename : str
            Path to the RCP file.

        Returns
        -------
        None
        """
        super().read_rcp(filename)
        self.info.register_config_file(filename)

    @abstractmethod
    def read_data(self, hdul):
        """
        Read a FITS HDU list to populate channel data.

        Parameters
        ----------
        hdul : fits.HDUList

        Returns
        -------
        None
        """
        pass
