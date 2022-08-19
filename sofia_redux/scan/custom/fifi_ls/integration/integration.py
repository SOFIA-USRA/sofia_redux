# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import numpy as np

from sofia_redux.scan.custom.fifi_ls.integration import (
    fifi_ls_integration_numba_functions)
from sofia_redux.scan.custom.sofia.integration.integration import (
    SofiaIntegration)

__all__ = ['FifiLsIntegration']


class FifiLsIntegration(SofiaIntegration):

    def __init__(self, scan=None):
        """
        Initialize a FIFI-LS integration.

        Parameters
        ----------
        scan : sofia_redux.scan.custom.hawc_plus.scan.scan.FifiLsScan
            The scan to which this integration belongs (optional).
        """
        super().__init__(scan=scan)

    @property
    def scan_astrometry(self):
        """
        Return the scan astrometry.

        Returns
        -------
        FifiLsAstrometryInfo
        """
        return super().scan_astrometry

    def apply_configuration(self):
        """
        Apply configuration options to an integration.

        Returns
        -------
        None
        """
        pass

    def read(self, hdul):
        """
        Read integration information from a FITS HDU List.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            A list of data HDUs containing "timestream" data.

        Returns
        -------
        None
        """
        log.info("Processing scan data:")
        records = hdul['FLUX'].data.shape[0]
        log.debug(f"Reading {records} frames from HDU list.")
        sampling = (1.0 / self.info.instrument.integration_time).to(
            units.Unit('Hz'))
        minutes = (self.info.instrument.sampling_interval * records).to(
            units.Unit('min'))
        log.info(f"Sampling at {sampling:.3f} ---> {minutes:.2f}.")

        self.frames.initialize(self, records)
        self.frames.read_hdul(hdul)

    def validate(self):
        """
        Validate the integration after a read.

        Returns
        -------
        None
        """
        self.flag_zeroed_channels()
        super().validate()

    def flag_zeroed_channels(self):
        """
        Flags all channels with completely zeroed frame data as DISCARD/DEAD.

        Returns
        -------
        None
        """
        log.debug("Flagging zeroed channels.")

        fifi_ls_integration_numba_functions.flag_zeroed_channels(
            frame_data=self.frames.data,
            frame_valid=self.frames.valid,
            channel_indices=np.arange(self.channels.size),
            channel_flags=self.channels.data.flag,
            discard_flag=self.channel_flagspace.convert_flag('DISCARD').value)

        # Flag discarded channels as DEAD
        self.channels.data.set_flags(
            'DEAD', indices=self.channels.data.is_flagged('DISCARD'))

    def get_full_id(self, separator='|'):
        """
        Return the full integration ID.

        Parameters
        ----------
        separator : str, optional
            The separator character/phase between the scan and integration ID.

        Returns
        -------
        str
        """
        return self.scan.get_id()

    def get_first_frame(self, reference=0):
        """
        Return the first valid frame.

        Parameters
        ----------
        reference : int, optional
            The first actual frame index after which to return the first valid
            frame.  The default is the first (0).

        Returns
        -------
        FifiLsFrames
        """
        return super().get_first_frame(reference=reference)

    def get_last_frame(self, reference=None):
        """
        Return the first valid frame.

        Parameters
        ----------
        reference : int, optional
            The last actual frame index before which to return the last valid
            frame.  The default is the last.

        Returns
        -------
        FifiLsFrames
        """
        return super().get_last_frame(reference=reference)

    def get_crossing_time(self, source_size=None):
        """
        Return the crossing time for a given source size.

        Parameters
        ----------
        source_size : Coordinate2D1, optional
            The size of the source.  If not supplied, defaults to (in order
            of priority) the source size in the scan model, or the instrument
            source size.

        Returns
        -------
        time : astropy.units.Quantity
            The crossing time in time units.
        """
        if source_size is None:
            if self.scan.source_model is None:
                source_size = self.info.instrument.get_source_size()
            else:
                source_size = self.scan.source_model.get_source_size()

        return super().get_crossing_time(source_size=source_size.x)
