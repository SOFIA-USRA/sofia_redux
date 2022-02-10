# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import numpy as np

from sofia_redux.scan.custom.hawc_plus.integration import (
    hawc_integration_numba_functions)
from sofia_redux.scan.custom.sofia.integration.integration import (
    SofiaIntegration)

__all__ = ['HawcPlusIntegration']


class HawcPlusIntegration(SofiaIntegration):

    def __init__(self, scan=None):
        """
        Initialize a HAWC+ integration.
        """
        self.fix_jumps = False
        self.min_jump_level_frames = 0
        self.fix_subarray = None
        self.drift_dependents = None
        super().__init__(scan=scan)

    @property
    def scan_astrometry(self):
        """
        Return the scan astrometry.

        Returns
        -------
        HawcPlusAstrometryInfo
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

    def read(self, hdus):
        """
        Read integration information from a list of Data HDUs.  All HDUs should
        consist of "timestream" data.

        Parameters
        ----------
        hdus : list (astropy.io.fits.hdu.table.BinTableHDU)
            A list of data HDUs containing "timestream" data.

        Returns
        -------
        None
        """
        log.info("Processing scan data:")
        records = 0
        for hdu in hdus:
            records += int(hdu.header.get('NAXIS2', 0))

        log.debug(f"Reading {records} frames from {len(hdus)} HDU(s)")
        sampling = (1.0 / self.info.instrument.integration_time).to(
            units.Unit('Hz'))
        minutes = (self.info.instrument.sampling_interval * records).to(
            units.Unit('min'))
        log.debug(f"Sampling at {sampling:.3f} ---> {minutes:.2f}.")

        self.frames.initialize(self, records)
        self.frames.read_hdus(hdus)

    def validate(self):
        """
        Validate the integration after a read.

        Returns
        -------
        None
        """
        if self.configuration.is_configured('chopper.shift'):
            self.shift_chopper(self.configuration.get_int('chopper.shift'))

        self.flag_zeroed_channels()
        self.check_jumps()
        if self.configuration.is_configured('jumpdata'):
            self.correct_jumps()
        if self.configuration.is_configured('gyrocorrect'):
            self.info.gyro_drifts.correct(integration=self)
        super().validate()

    def get_table_entry(self, name):
        """
        Return a parameter value for the given name.

        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.

        Returns
        -------
        value
        """
        if name == 'hwp':
            return self.get_mean_hwp_angle().decompose().value
        if name == 'pwv':
            return self.get_mean_pwv().to('um').value
        return super().get_table_entry(name)

    def shift_chopper(self, n_frames):
        """
        Shift the chopper position by a given number of frames

        Parameters
        ----------
        n_frames : int
            The number of frames to shift the chopper signal.

        Returns
        -------
        None
        """
        if n_frames == 0:
            return
        log.debug(f"Shifting chopper signal by {n_frames} frames.")
        self.frames.chopper_position.shift(n_frames, fill_value=np.nan)
        if n_frames > 0:
            self.frames.valid[:n_frames] = False
        elif n_frames < 0:
            self.frames.valid[n_frames:] = False

    def flag_zeroed_channels(self):
        """
        Flags all channels with completely zeroed frame data as DISCARD/DEAD.

        Returns
        -------
        None
        """
        log.debug("Flagging zeroed channels.")

        hawc_integration_numba_functions.flag_zeroed_channels(
            frame_data=self.frames.data,
            frame_valid=self.frames.valid,
            channel_indices=np.arange(self.channels.size),
            channel_flags=self.channels.data.flag,
            discard_flag=self.channel_flagspace.convert_flag('DISCARD').value)

        # Flag discarded channels as DEAD
        self.channels.data.set_flags(
            'DEAD', indices=self.channels.data.is_flagged('DISCARD'))

    def set_tau(self, spec=None, value=None):
        """
        Set the tau values for the integration.

        If a value is explicitly provided without a specification, will be used
        to set the zenith tau if ground based, or transmission.  If a
        specification and value is provided, will set the zenith tau as:

        ((band_a / t_a) * (value - t_b)) + band_b

        where band_a/b are retrieved from the configuration as tau.<spec>.a/b,
        and t_a/b are retrieved from the configuration as tau.<instrument>.a/b.

        Parameters
        ----------
        spec : str, optional
            The tau specification to read from the configuration.  If not
            supplied, will be read from the configuration 'tau' setting.
        value : float, optional
            The tau value to set.  If not supplied, will be retrieved from the
            configuration as tau.<spec>.

        Returns
        -------
        None
        """
        super().set_tau(spec=spec, value=value)
        self.print_equivalent_taus(self.zenith_tau)

    def print_equivalent_taus(self, tau):
        """
        Write a log message for the given tau value.

        Parameters
        ----------
        tau : float

        Returns
        -------
        None
        """
        pwv = (self.get_tau('pwv', tau) * units.Unit('um')).round(1)
        los = np.round(tau / self.scan_astrometry.horizontal.sin_lat, 3)
        wave = self.info.instrument.wavelength.round(0).astype(int)
        msg = f'---> tau({wave}):{np.round(tau, 3)}, tau(LOS):{los}, PWV:{pwv}'
        log.info(msg)

    def check_jumps(self):
        """
        Checks for jumps in the jump counter.

        Returns
        -------
        has_jumps : bool
            `True` if jumps were detected.
        """
        log.debug("Checking for flux jumps.")

        if (not hasattr(self.frames, 'jump_counter')
                or self.frames.jump_counter is None):
            log.warning("Scan has no jump counter data.")
            return

        n_jumps = hawc_integration_numba_functions.check_jumps(
            start_counter=self.get_first_frame().jump_counter,
            jump_counter=self.frames.jump_counter,
            frame_valid=self.frames.valid,
            has_jumps=self.channels.data.has_jumps,
            channel_indices=np.arange(self.channels.size))

        if n_jumps < 0:
            log.warning("No valid frames available")
        elif n_jumps == 0:
            log.debug("---> All good!")
        else:
            log.debug(f"---> found jump(s) in {n_jumps} pixels.")

    def correct_jumps(self):
        """
        Correct jumps in the data.

        The data are corrected by:

        data -= jump_counter * channel_jumps

        where jump_counter is created for each frame and channel, and jumps
        are per channel.  Since jump counter is a byte valued, wrap around
        values are accounted for by jump_range (power of 2 value).

        Returns
        -------
        None
        """
        log.debug("Correcting flux jumps.")
        hawc_integration_numba_functions.correct_jumps(
            frame_data=self.frames.data,
            frame_valid=self.frames.valid,
            jump_counter=self.frames.jump_counter,
            channel_indices=np.arange(self.channels.size),
            channel_jumps=self.channels.data.jump,
            jump_range=self.info.detector_array.JUMP_RANGE)

    def remove_drifts(self, target_frame_resolution=None, robust=False):
        """
        Remove drifts in frame data given a target frame resolution.

        Will also set the filter time scale based on the target frame
        resolution.

        Sets additional attributes based on jumps.

        Parameters
        ----------
        target_frame_resolution : int
            The number of frames for the target resolution.
        robust : bool, optional
            If `True` use the robust (median) method to determine means.

        Returns
        -------
        None
        """
        self.fix_jumps = self.configuration.get_bool('fixjumps')
        det = self.info.detector_array
        self.fix_subarray = np.full(det.subarrays, False)

        self.fix_subarray[det.R0] = self.configuration.is_configured(
            'fixjumps.r0')
        self.fix_subarray[det.R1] = self.configuration.is_configured(
            'fixjumps.r1')
        self.fix_subarray[det.T0] = self.configuration.is_configured(
            'fixjumps.t0')
        self.fix_subarray[det.T1] = self.configuration.is_configured(
            'fixjumps.t1')

        self.min_jump_level_frames = self.frames_for(
            10 * self.get_point_crossing_time())

        self.drift_dependents = self.get_dependents('drifts')

        super().remove_drifts(target_frame_resolution=target_frame_resolution,
                              robust=robust)

    def get_mean_hwp_angle(self):
        """
        Return the mean Half Wave Plate angle.

        The mean HWP angle is given as the mean of the first and last valid
        frame HWP angle values.

        Returns
        -------
        astropy.units.Quantity
            The mean HWP angle.
        """
        hwp_1 = self.frames.get_first_frame_value('hwp_angle')
        hwp_2 = self.frames.get_last_frame_value('hwp_angle')
        return 0.5 * (hwp_1 + hwp_2)

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

    def check_consistency(self, channels, frame_dependents, start_frame=None,
                          stop_frame=None):
        """
        Check consistency of frame dependents and channels.

        In addition to the standard consistency checks, will also fix jumps
        in the frame data if configuration settings allow, and jumps are
        present.

        Parameters
        ----------
        channels : ChannelGroup
        frame_dependents : numpy.ndarray (float)
        start_frame : int, optional
            The starting frame (inclusive).  Defaults to the first (0) frame.
        stop_frame : int, optional
            The end frame (exclusive).  Defaults to the last (self.size) frame.

        Returns
        -------
        consistent : numpy.ndarray (bool)
            An array of size self.size where `True` indicates a consistent
            frame.
        """
        is_ok = super().check_consistency(channels, frame_dependents,
                                          start_frame=start_frame,
                                          stop_frame=stop_frame)

        no_jumps = self.level_jumps(channels, frame_dependents,
                                    start_frame=start_frame,
                                    stop_frame=stop_frame)

        return is_ok & no_jumps

    def get_jump_blank_range(self):
        """
        Return the number of frames to flag before and after a jump.

        Returns
        -------
        blank_frames : numpy.ndarray (int)
            The [flag_before, flag_after] number of frames to flag before and
            after each jump.
        """
        blank_time = self.configuration.get_float_list('fixjumps.blank',
                                                       default=None)
        if blank_time is None:
            blank_frames = np.zeros(2, dtype=int)
        else:
            if len(blank_time) == 1:
                blank_time = np.full(2, blank_time[0]) * units.Unit('second')
            elif len(blank_time) != 2:
                log.warning("Jump blanking time in configuration is "
                            "not a 1 or 2 element array.  "
                            "Will not apply blank flags.")
                blank_time = np.full(2, blank_time[0]) * units.Unit('second')
            else:
                blank_time = np.asarray(blank_time) * units.Unit('second')

            blank_frames = np.asarray(list(map(self.frames_for, blank_time)))
            blank_frames[blank_time == 0] = 0
        return blank_frames

    def level_jumps(self, channels, frame_dependents, start_frame=None,
                    stop_frame=None):
        """
        Levels frame data based on jump locations.

        Parameters
        ----------
        channels : ChannelGroup
        frame_dependents : numpy.ndarray (float)
        start_frame : int, optional
            The starting frame (inclusive).  Defaults to the first (0) frame.
        stop_frame : int, optional
            The end frame (exclusive).  Defaults to the last (self.size) frame.

        Returns
        -------
        consistent : numpy.ndarray (bool)
            An array of size self.size where `True` indicates a consistent
            frame.
        """
        self.drift_dependents = self.get_dependents('drifts')

        exclude_sample_flag = ~(
            self.flagspace.convert_flag('SAMPLE_SOURCE_BLANK').value)
        jump_flag = self.flagspace.convert_flag('SAMPLE_PHI0_JUMP').value
        blank_frames = self.get_jump_blank_range()

        no_jumps = hawc_integration_numba_functions.fix_jumps(
            frame_valid=self.frames.valid,
            frame_data=self.frames.data,
            frame_weights=self.frames.relative_weight,
            modeling_frames=self.frames.is_flagged('MODELING_FLAGS'),
            frame_parms=frame_dependents,
            sample_flags=self.frames.sample_flag,
            exclude_sample_flag=exclude_sample_flag,
            channel_indices=channels.indices,
            channel_parms=self.drift_dependents.for_channel,
            min_jump_level_frames=self.min_jump_level_frames,
            jump_flag=jump_flag,
            fix_each=self.fix_jumps,
            fix_subarray=self.fix_subarray,
            has_jumps=channels.has_jumps,
            subarray=channels.sub,
            jump_counter=self.frames.jump_counter,
            start_frame=start_frame,
            end_frame=stop_frame,
            flag_before=blank_frames[0],
            flag_after=blank_frames[1])

        return no_jumps

    def update_inconsistencies(self, channels, frame_dependents, drift_size):
        """
        Check consistency of frame dependents and channels.

        Looks for inconsistencies in the channel and frame data post levelling
        and updates the `inconsistencies` attribute of the channel data.

        Parameters
        ----------
        frame_dependents : numpy.ndarray (float)
        channels : ChannelGroup
        drift_size : int
            The size of the drift removal block size in frames.

        Returns
        -------
        None
        """
        super().update_inconsistencies(channels, frame_dependents, drift_size)
        drift_parms = self.get_dependents('drifts')
        exclude_sample_flag = ~(
            self.flagspace.convert_flag('SAMPLE_SOURCE_BLANK').value)
        jump_flag = self.flagspace.convert_flag('SAMPLE_PHI0_JUMP').value
        blank_frames = self.get_jump_blank_range()

        inconsistencies = \
            hawc_integration_numba_functions.find_inconsistencies(
                frame_valid=self.frames.valid,
                frame_data=self.frames.data,
                frame_weights=self.frames.relative_weight,
                modeling_frames=self.frames.is_flagged('MODELING_FLAGS'),
                frame_parms=frame_dependents,
                sample_flags=self.frames.sample_flag,
                exclude_sample_flag=exclude_sample_flag,
                channel_indices=channels.indices,
                channel_parms=drift_parms.for_channel,
                min_jump_level_frames=self.min_jump_level_frames,
                jump_flag=jump_flag,
                fix_each=self.fix_jumps,
                fix_subarray=self.fix_subarray,
                has_jumps=channels.has_jumps,
                subarray=channels.sub,
                jump_counter=self.frames.jump_counter,
                drift_size=drift_size,
                flag_before=blank_frames[0],
                flag_after=blank_frames[1])

        channels.inconsistencies += inconsistencies

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
        HawcPlusFrames
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
        HawcPlusFrames
        """
        return super().get_last_frame(reference=reference)
