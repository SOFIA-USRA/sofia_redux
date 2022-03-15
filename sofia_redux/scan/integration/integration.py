# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log, units
from astropy.io import fits
from astropy.table import Column, Table
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from scipy.signal import windows as scipy_windows
from scipy.signal import welch
import warnings

from sofia_redux.scan.frames.frames import Frames
from sofia_redux.scan.flags.motion_flags import MotionFlags
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.signal.signal import Signal
from sofia_redux.scan.integration.dependents.dependents import Dependents
from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.filters.multi_filter import MultiFilter
from sofia_redux.scan.filters.motion_filter import MotionFilter
from sofia_redux.scan.filters.kill_filter import KillFilter
from sofia_redux.scan.filters.whitening_filter import WhiteningFilter
from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)
from sofia_redux.scan.utilities.class_provider import get_integration_class
from sofia_redux.scan.integration import integration_numba_functions as int_nf
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.reduction.version import ReductionVersion
from sofia_redux.scan.signal.correlated_signal import CorrelatedSignal
__all__ = ['Integration']


class Integration(ABC):

    def __init__(self, scan=None):
        self.scan = None
        self.channels = None
        self.frames = None
        self.comments = []
        self.integration_number = 0
        self.gain = 1.0
        self.nefd = np.nan
        self.source_generation = 0
        self.source_sync_gain = None
        self.filter_time_scale = np.inf * units.Unit('second')
        # speed and weight (1/variance)
        self.average_scan_speed = (np.nan * units.Unit('deg/second'),
                                   np.nan * units.Unit('s2/deg2'))
        self.min_speed = np.nan * units.Unit('deg/second')
        self.max_speed = np.nan * units.Unit('deg/second')
        self.dependents = None
        self.signals = None
        self.filter = None
        self.is_detector_stage = False
        self.is_valid = False
        self.zenith_tau = 0.0
        self.parallelism = None
        if scan is not None:
            self.set_scan(scan)

    @property
    def flagspace(self):
        """
        Return the frame flagspace.

        Returns
        -------
        FrameFlags : class
        """
        if self.frames is None:
            return None
        else:
            return self.frames.flagspace

    @property
    def channel_flagspace(self):
        """
        Return the channel flagspace.

        Returns
        -------
        ChannelFlags : class
        """
        if self.channels is None:
            return None
        else:
            return self.channels.flagspace

    @property
    def motion_flagspace(self):
        """
        Return the motion flagspace.

        Returns
        -------
        MotionFlags : class
        """
        return MotionFlags

    @property
    def info(self):
        """
        Return the Info object for the integration.

        Returns
        -------
        Info
        """
        if self.channels is None:
            return None
        return self.channels.info

    @property
    def configuration(self):
        """
        Return the integration configuration (from channel configuration).

        Returns
        -------
        Configuration
        """
        if self.info is None:
            return None
        return self.info.configuration

    @property
    def instrument_name(self):
        """
        Return the instrument name.

        Returns
        -------
        str
        """
        if self.info is None:
            return None
        return self.info.instrument.name

    @property
    def size(self):
        """
        Return the number of frames in the integration.

        Returns
        -------
        int
        """
        if self.frames is None:
            return 0
        else:
            return self.frames.size

    @property
    def n_channels(self):
        """
        Return the number of channels in the integration.

        Returns
        -------
        int
        """
        if self.channels is None:
            return 0
        else:
            return self.channels.size

    @property
    def reference_attributes(self):
        """
        Return attributes that should be referenced rather than copied.

        Returns
        -------
        set
        """
        return {'scan'}

    @classmethod
    def get_integration_class(cls, name):
        """
        Return an Integration instance given an instrument name.

        Parameters
        ----------
        name : str
            The name of the instrument.

        Returns
        -------
        Integration : class
        """
        return get_integration_class(name)

    @property
    def scan_astrometry(self):
        """
        Return the scan astrometry.

        Returns
        -------
        AstrometryInfo
        """
        if self.scan is None:
            return None
        return self.scan.astrometry

    def __getitem__(self, indices):
        """
        Return a selection of the frame data.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)

        Returns
        -------
        None
        """
        return self.frames[indices]

    def copy(self):
        """
        Return a new copy of the integration.

        Returns
        -------

        """
        new = self.__class__()
        referenced = self.reference_attributes
        for attribute, value in self.__dict__.items():
            if attribute in referenced:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))

        for attribute, value in new.__dict__.items():
            if hasattr(value, 'integration'):
                value.integration = new

        return new

    def clone(self):
        """
        Return a copy of the integration without frame type data.

        Returns
        -------
        Integration
        """
        new = self.__class__()
        new.scan = self.scan
        new.channels = self.channels.copy()
        new.channels.set_parent(new)
        return new

    def has_option(self, option):
        """
        Return whether the configuration option is configured.

        In order to be considered "configured", the option must exist
        and also have a value.

        Parameters
        ----------
        option : str

        Returns
        -------
        configured : bool
        """
        if self.configuration is None:
            return False
        return self.configuration.is_configured(option)

    def set_scan(self, scan):
        """
        Set the parent scan of the integration.

        The info is copied from the scan, and the configuration is unlinked
        (separate copy from the scan configuration).

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        if scan.hdul is None:
            raise ValueError("Scan does not contain FITS HDUL.")
        if not scan.info.configuration.enabled:
            raise ValueError("Scan has not been configured.")

        # A separate configuration/channels but scan is a direct reference.
        self.scan = scan
        self.channels = scan.channels.copy()
        self.channels.set_parent(self)
        self.info.unlink_configuration()
        self.channels.info = self.channels.info.copy()
        self.frames = Frames.instance_from_instrument_name(
            self.instrument_name)

    def validate(self):
        """
        Validate an integration following the initial read.

        Returns
        -------
        None
        """
        if self.is_valid or self.configuration is None:
            return

        if self.configuration.is_configured('shift'):
            time_shift = self.configuration.get_float(
                'shift') * units.Unit('second')
            self.shift_frames(time_shift)

        self.frames.validate()

        if not self.frames.valid.any():
            self.is_valid = False
            return

        if self.configuration.get_bool('fillgaps'):
            if self.has_gaps(tolerance=1):
                try:
                    self.fill_gaps()
                except ValueError as err:
                    log.warning(f"Could not fill gaps: {err}")
                    self.is_valid = False
                    return

        if self.configuration.get_bool('notch'):
            self.notch_filter()

        if self.configuration.is_configured('frames'):
            self.select_frames()

        if self.size == 0 or not self.frames.valid.any():
            log.warning("No valid frames.")
            self.is_valid = False
            return

        if not self.configuration.get_bool('lab'):
            if self.configuration.is_configured('vclip'):
                self.velocity_clip()
            if self.configuration.is_configured('aclip'):
                self.acceleration_clip()
            self.calculate_scan_speed_stats()
        elif self.configuration.is_configured('lab.scanspeed'):
            speed_unit = units.Unit('arcsec/second')
            speed = self.configuration.get_float('lab.scanspeed') * speed_unit
            rms = 0.0 * (speed_unit ** -2)
            self.average_scan_speed = (speed, rms)
        else:
            speed = self.info.instrument.resolution / units.Unit('second')
            self.average_scan_speed = (speed, 0.0 / (speed ** 2))

        # check again for too few valid frames after clipping
        if self.size == 0 or not self.frames.valid.any():
            log.warning("No valid frames.")
            self.is_valid = False
            return

        if self.configuration.is_configured('filter.kill'):
            log.debug("FFT Filtering specified sub-bands...")
            self.remove_offsets(robust=False)
            kill_filter = KillFilter(self)
            kill_filter.update_config()
            kill_filter.apply()

        # Flag out-of-range data
        if self.configuration.is_configured('range'):
            self.check_range()

        # Continue only if enough valid channels remain
        min_channels = self.configuration.get_int('mappingpixels', default=2)
        if self.channels.n_mapping_channels < min_channels:
            log.warning(f"Too few valid channels "
                        f"({self.channels.n_mapping_channels}).")
            self.is_valid = False
            return

        # Automatic downsampling after velocity clipping
        if self.configuration.is_configured('downsample'):
            self.downsample()

        self.trim()

        # Continue only if integration is long enough to be processed
        if self.configuration.is_configured('subscan.minlength'):
            min_frames = self.configuration.get_float('subscan.minlength')
            dt = self.info.instrument.sampling_interval.decompose().value
            min_frames = int(min_frames / dt)
            mapping_frames = self.get_frame_count(discard_flag='SOURCE_FLAGS')
            if mapping_frames < min_frames:
                integration_time = mapping_frames * dt * units.Unit('second')
                log.warning(f"Integration is too short ({integration_time}).")
                self.is_valid = False
                return

        if self.configuration.is_configured('filter.ordering'):
            self.setup_filters()

        self.detector_stage()

        # Remove the DC offsets, either if explicitly requested or to allow
        # bootstrapping pixel weights when pixeldata is not defined.
        # Must do this before direct tau estimates.
        if (self.configuration.is_configured('level')
                or not self.configuration.is_configured('pixeldata')):
            robust = self.configuration.get_string('estimator') == 'median'
            log.debug(f"Removing DC offsets{' (robust)' if robust else ''}.")
            self.remove_offsets(robust=robust)

        if self.configuration.is_configured('tau'):
            self.set_tau()

        if self.configuration.is_configured('scale'):
            self.set_scaling()

        if self.configuration.get_bool('invert'):
            self.gain *= -1

        if self.configuration.get_bool('noslim'):
            self.channels.reindex()
        else:
            self.slim()

        if self.configuration.is_configured('jackknife'):
            self.jackknife()

        self.bootstrap_weights()
        self.channels.calculate_overlaps(point_size=self.scan.get_point_size())
        # self.reindex()
        self.is_valid = True

    def apply_configuration(self):
        """
        Apply configuration options to an integration.

        Returns
        -------
        None
        """
        pass  # pragma: no cover

    def get_first_frame_index(self, reference=0):
        """
        Return the first valid frame index of the integration.

        Parameters
        ----------
        reference : int, optional
            If supplied, finds the first frame from `reference`, rather than
            the first index (0).  May take negative values to indicate an
            index relative to the last.

        Returns
        -------
        first_frame : int

        """
        return self.frames.get_first_frame_index(reference=reference)

    def get_last_frame_index(self, reference=None):
        """
        Return the last valid frame index of the integration.

        Parameters
        ----------
        reference : int, optional
            If supplied, finds the last frame before `reference`, rather than
            the last index (self.size).  May take negative values to indicate
            an index relative to the last index.

        Returns
        -------
        last_frame : int
        """
        return self.frames.get_last_frame_index(reference=reference)

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
        Frames
        """
        return self.frames[self.get_first_frame_index(reference=reference)]

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
        Frames
        """
        return self.frames[self.get_last_frame_index(reference=reference)]

    def get_frame_count(self, keep_flag=None, discard_flag=None,
                        match_flag=None):
        """
        Return the number of frames in an integration.

        A number of flags may also be supplied to return the number of a
        certain type of frame.

        Parameters
        ----------
        keep_flag : int or ChannelFlagTypes, optional
            Flag values to keep in the calculation.
        discard_flag : int or ChannelFlagTypes, optional
            Flag values to discard_flag from the calculation.
        match_flag : int or ChannelFlagTypes, optional
            Only matching flag values will be used in the calculation.

        Returns
        -------
        n_frames : int
            The number of matching frames in the integration.
        """
        return self.frames.get_frame_count(keep_flag=keep_flag,
                                           discard_flag=discard_flag,
                                           match_flag=match_flag)

    def select_frames(self):
        """
        Delete frames not inside the frame range defined in the configuration.

        The "frames" configuration setting is responsible for defining a valid
        frame range.  All fixed frame indices outside of this range will be
        removed from the integration.  Note that the "fixed" index of the
        frames will be used - the index as initially read from the data file.

        Returns
        -------
        None
        """
        if 'frames' not in self.configuration:
            return
        frame_range = self.configuration.get_range('frames')
        if frame_range.max < 0:
            frame_range.max += self.frames.fixed_index[-1]
        delete_mask = ~frame_range.in_range(self.frames.fixed_index)
        if np.any(delete_mask):
            log.debug(f'Removing {np.sum(delete_mask)} frames '
                      f'outside range {frame_range}')
        self.frames.delete_indices(delete_mask)
        self.reindex()

    def check_range(self):
        """
        Checks the range of frame data values, flagging as necessary.

        In addition to flagging frames, channels may also be flagged as having
        bad DAC ranges and dead if a certain fraction of frames for a given
        channel is above a threshold.

        The configuration must contain a "range" branch for range checking to
        occur.  If a range value is assigned to "range", it is explicitly used
        to set the range of valid DAC values.  If not, only NaN values will
        be flagged as out-of-range.  The "range.flagfraction" configuration
        option is used to specify what fraction of frames flagged as
        out-of-range for a channel is required to flag that channel as
        bad.

        Returns
        -------
        None
        """
        if not self.configuration.is_configured('range'):
            return
        data_range = self.configuration.get_range('range')
        if np.isfinite(data_range.min) or np.isfinite(data_range.max):
            out_of_range = ~data_range.in_range(self.frames.data)
        else:
            out_of_range = np.isnan(self.frames.data)

        check_fraction = self.configuration.is_configured('range.flagfraction')
        if not out_of_range.any():
            if check_fraction:
                log.debug("Flagging out-of-range data. "
                          "0 channel(s) discarded.")
            return
        else:
            range_flag = self.flagspace.flags.SAMPLE_SKIP.value
            self.frames.sample_flag[out_of_range] |= range_flag

        if not check_fraction:
            return

        n_valid_frames = self.get_frame_count()
        n_invalid_channels = np.sum(out_of_range, axis=0)
        bad_channel_fraction = n_invalid_channels / n_valid_frames
        critical = self.configuration.get_float('range.flagfraction')
        bad_channels = bad_channel_fraction > critical

        n_flagged_channels = np.sum(bad_channels)
        if n_flagged_channels > 0:
            bad_channel_flag = (self.channel_flagspace.flags.DAC_RANGE
                                | self.channel_flagspace.flags.DEAD)
            self.channels.data.set_flags(bad_channel_flag,
                                         indices=np.nonzero(bad_channels)[0])

        log.debug(f"Flagging out-of-range data. "
                  f"{n_flagged_channels} channel(s) discarded.")

        self.channels.census()

    def trim(self, start=True, end=True):
        """
        Remove invalid frames from the start and end of the integration.

        Parameters
        ----------
        start : bool, optional
            If `True` (default), delete all frames before the first valid frame
            in the integration.
        end : bool, optional
            If `True` (default), delete all frames after the last valid frame.

        Returns
        -------
        None
        """
        if not self.frames.valid.any():
            return
        start_index = self.get_first_frame_index() if start else 0
        end_index = self.get_last_frame_index() if end else (self.size - 1)
        delete = np.full(self.size, True)
        delete[start_index:end_index + 1] = False

        if delete.any():
            self.frames.delete_indices(delete)
            if self.dependents is not None:
                for dependent in self.dependents.values():
                    dependent.for_frame = dependent.for_frame[~delete]

        self.reindex()
        log.debug(f"Trimmed to {self.size} frames.")

    def get_pa(self):
        """
        Return the position angle of the integration.

        The position angle is derived from the average position angles of the
        first and last valid frames.

        Returns
        -------
        position_angle : astropy.units.Quantity
        """
        pa = np.arctan2(self.frames.get_first_frame_value('sin_pa'),
                        self.frames.get_first_frame_value('cos_pa'))
        pa += np.arctan2(self.frames.get_last_frame_value('sin_pa'),
                         self.frames.get_last_frame_value('cos_pa'))
        return pa / 2.0 * units.Unit('radian')

    def scale(self, factor):
        """
        Scale all data (in `data`) by `factor`.

        Parameters
        ----------
        factor : int or float, optional

        Returns
        -------
        None
        """
        self.frames.scale(factor)

    def frames_for(self, time=None):
        """
        Return the number of frames for a given time interval.

        Parameters
        ----------
        time : astropy.units.Quantity, optional
            The time interval.  If not supplied, defaults to the sampling
            interval.

        Returns
        -------
        n_frames : int
            The number of frames in the time interval.
        """
        if time is None or time > self.filter_time_scale:
            time = self.filter_time_scale

        n_frames = (time / self.info.instrument.sampling_interval)
        n_frames = np.round(n_frames.decompose().value)
        n_frames = int(np.clip(n_frames, 1, self.size))
        return n_frames

    def power2_frames_for(self, time=None):
        """
        Return the number of frames for a given time interval using pow2ceil.

        pow2ceil raises the number of frames to the nearest power of 2.  E.g.,
        5 -> 8, 12 -> 16 etc.

        Parameters
        ----------
        time : astropy.units.Quantity, optional
            The time interval.  If not supplied, defaults to the sampling
            interval.

        Returns
        -------
        n_frames : int
            The number of frames in the time interval.
        """
        return numba_functions.pow2ceil(self.frames_for(time=time))

    def filter_frames_for(self, spec=None, default_time=None):
        """
        Return the number of filtering frames for a given specification.

        The available specifications are 'auto', 'max', or a float value in
        seconds.  'auto' derives the time from the crossing time, `max`
        returns the maximum number of frames in the integration.

        Note that the return value will be rounded up to the nearest power of
        2.

        Parameters
        ----------
        spec : float or str, optional
            The specification.  May take values of 'auto', 'max', or a float
            value (in seconds).
        default_time : astropy.units.Quantity
            If a specification is not supplied or cannot be parsed, use this
            default time instead.

        Returns
        -------
        frames : int

        Raises
        ------
        ValueError
           If no spec or default value is provided.
        """
        if spec == 'auto':
            stability = self.info.instrument.get_stability()
            if self.configuration.is_configured('photometry'):
                drift_time = stability
            else:
                drift_time = max(stability, self.get_crossing_time())
            n_frames = self.frames_for(drift_time)
        elif spec == 'max':
            n_frames = self.size
        elif isinstance(spec, units.Quantity):
            n_frames = self.frames_for(spec)
        else:
            try:
                drift_time = float(spec) * units.Unit('second')
                n_frames = self.frames_for(drift_time)
            except (ValueError, TypeError):
                if default_time is None:
                    raise ValueError("No spec or default value provided.")
                n_frames = self.frames_for(default_time)

        return utils.pow2ceil(n_frames)

    def get_positions(self, motion_flag):
        """
        Return positions based on a type of motion.

        Available motion types are TELESCOPE, SCANNING, or CHOPPING.
        Combinations of TELESCOPE|CHOPPING|PROJECT_GLS and SCANNING|CHOPPING
        are also permitted.  i.e., TELESCOPE and CHOPPING should be supplied
        with or without CHOPPING, and PROJECT_GLS is available for the
        TELESCOPE flag.  If these flags are not present, an array containing
        zero values will be returned.

        Parameters
        ----------
        motion_flag : MotionFlagTypes or str or int
            The motion type for which to extract positions.

        Returns
        -------
        position : astropy.units.Quantity (numpy.ndarray)
            An array of shape (N, 2) containing the (x, y) positions.
        """
        motion_flag = self.motion_flagspace.convert_flag(motion_flag)
        position = Coordinate2D()

        if motion_flag & self.motion_flagspace.flags.TELESCOPE:
            coordinates = SphericalCoordinates()
            coordinates.copy_coordinates(
                self.frames.get_absolute_native_coordinates())

            if not (motion_flag & self.motion_flagspace.flags.CHOPPER):
                # Subtract the chopper motion if it is not requested
                coordinates.subtract_native_offset(
                    self.frames.chopper_position)

            position.copy_coordinates(coordinates)
            if motion_flag & self.motion_flagspace.flags.PROJECT_GLS:
                position.scale_x(coordinates.cos_lat)

        # Scanning includes the chopper mode
        elif motion_flag & self.motion_flagspace.flags.SCANNING:
            position.copy_coordinates(
                self.frames.get_absolute_native_offsets())

            if not (motion_flag & self.motion_flagspace.flags.CHOPPER):
                # Subtract the chopper motion if it is not requested.
                position.subtract(self.frames.chopper_position)

        # The chopper position only
        elif motion_flag & self.motion_flagspace.flags.CHOPPER:
            position.copy_coordinates(self.frames.chopper_position)

        # Nada
        else:
            position.set(np.zeros((2, self.size), dtype=float
                                  ) * units.Unit('arcsec'))

        position.nan(~self.frames.valid)

        return position

    def get_smooth_positions(self, motion_flag):
        """
        Return the positions, smoothed according to the configuration.

        The frames are smoothed using a box kernel with width specified in time
        (seconds), and should be set using "positions.smooth".

        Parameters
        ----------
        motion_flag : MotionFlagTypes or str or int
            The motion type for which to extract positions.  See
            `get_positions` for details on motion flag types.

        Returns
        -------
        position : astropy.units.Quantity (numpy.ndarray)
            An array of shape (N, 2) containing the (x, y) positions.
        """
        smooth_time = self.configuration.get_float(
            'positions.smooth', default=np.nan) * units.Unit('second')

        if np.isnan(smooth_time):
            smooth_frames = 1
        else:
            smooth_frames = self.frames_for(smooth_time)

        position = self.get_positions(motion_flag)

        if smooth_frames <= 1:
            return position

        position.set(int_nf.smooth_positions(
            coordinates=position.coordinates.value,
            bin_size=smooth_frames,
            fill_value=np.nan))

        return position

    def get_velocities(self, motion_flag, return_position=False):
        """
        Return the scanning velocity (including chopping motion).

        Parameters
        ----------
        motion_flag : MotionFlagTypes or str or int
            The motion type for which to extract positions.  See
            `get_positions` for details on motion flag types.
        return_position : bool, optional
            If `True`, return the position in addition to velocity.

        Returns
        -------
        velocity, [position] : Coordinate2D, [Coordinate2D]
           The position will be returned in addition to velocity if
           `return_position` is `True`.  Both have (x, y) coordinates.
        """
        position = self.get_smooth_positions(motion_flag)
        dt = self.info.instrument.sampling_interval
        velocity = Coordinate2D(unit=position.x.unit / dt.unit)

        velocity.set(int_nf.calculate_2d_velocities(
            position.coordinates.value, dt.value))

        return (velocity, position) if return_position else velocity

    def get_scanning_velocities(self, return_position=False):
        """
        Return the scanning velocity (including chopping motion).

        Parameters
        ----------
        return_position : bool, optional
            If `True`, return the position in addition to velocity.

        Returns
        -------
        velocity, [position] : Coordinate2D, [Quantity]
           The position will be returned in addition to velocity if
           `return_position` is `True`.   Both contain (x, y) coordinates.
        """
        return self.get_velocities(self.motion_flagspace.flags.SCANNING
                                   | self.motion_flagspace.flags.CHOPPER,
                                   return_position=return_position)

    def get_typical_scanning_speed(self):
        """
        Get the mean (robust) scanning speed and weight (inverse variance).

        Returns
        -------
        speed, weight : Quantity, Quantity
            The speed in distance/time units, and weight in time^2/distance^2
            units.

        """
        velocities = self.get_scanning_velocities()
        speed_unit = (self.info.instrument.get_size_unit()
                      / units.Unit('second'))
        speeds = velocities.length.to(speed_unit)

        self.min_speed = np.nanmin(speeds)
        self.max_speed = np.nanmax(speeds)
        n_valid = np.sum(np.isfinite(speeds))
        if n_valid > 10:
            average_speed = numba_functions.robust_mean(
                speeds.value, tails=0.1) * speed_unit
        else:
            average_speed = np.nanmedian(speeds)

        dev = (speeds - average_speed) ** 2
        if n_valid > 10:
            weight = 1.0 / numba_functions.robust_mean(
                dev.value, tails=0.1) * (1 / (speed_unit ** 2))
        else:
            weight = 0.454937 / np.nanmedian(dev)

        return average_speed, weight

    def get_speed_clip_range(self):
        """
        Determine the range of permissible scanning speeds.

        The permissible scanning speeds are determined from the configuration
        'vclip' (velocity clipping) branch.  If vclip = 'auto', the minimum
        speed is 5 FWHMs over the stability timescale, and the maximum speed
        is 1/2.5 beams per sample to avoid smearing.  Otherwise, the 'vclip'
        configuration value should be a range in units of arcseconds per
        second.

        If the 'chopped' configuration option is set, the minimum speed is set
        to zero.

        Returns
        -------
        speed_range : Range
            The minimum and maximum speeds are astropy.unit.Quantity values
            in units of the default size unit per second.
        """
        speed_unit = (self.info.instrument.get_size_unit()
                      / units.Unit('second'))
        if not self.configuration.is_configured('vclip'):
            return Range(-np.inf * speed_unit, np.inf * speed_unit)

        if self.configuration.get_string('vclip') == 'auto':
            min_speed = 5 * (self.info.instrument.get_source_size()
                             / self.info.instrument.get_stability()
                             ).to(speed_unit)
            max_speed = 0.4 * (self.scan.get_point_size()
                               / self.info.instrument.sampling_interval
                               ).to(speed_unit)
            speed_range = Range(min_speed, max_speed)
        else:
            read_unit = units.Unit('arcsec/second')
            speed_range = self.configuration.get_range('vclip')
            speed_range.min = (speed_range.min * read_unit).to(speed_unit)
            speed_range.max = (speed_range.max * read_unit).to(speed_unit)

        if self.configuration.is_configured('chopped'):
            speed_range.min *= 0

        return speed_range

    def calculate_scan_speed_stats(self):
        """
        Calculates and reports statistics on the scanning speed.

        Notes
        -----
        This should be run before and frame clipping/invalidation, since
        smoothing positions (which is used prior to velocity calculations)
        results in identical position values for frames near the edge of
        a block of invalid frames.

        Returns
        -------
        None
        """
        speed, weight = self.get_typical_scanning_speed()
        rms = np.sqrt(1.0 / weight)

        self.average_scan_speed = speed, weight
        log.debug(f"Typical scanning speeds are {speed:.3f} +- {rms:.3f}.")
        log.debug(f"Min speed = {self.min_speed:.3f}, "
                  f"Max speed = {self.max_speed:.3f}")

    def velocity_clip(self, speed_range=None, strict=None, sigma_clip=None):
        """
        Clip frames that are outside of the permissible scanning speed range.

        Parameters
        ----------
        speed_range : Range, optional
            The minimum and maximum speeds are astropy.unit.Quantity values
            in units of the default size unit per second.  If not supplied,
            will be determined from the configuration via the
            `get_speed_clip_range` method, and the 'vclip' configuration value.
        strict : bool, optional
            Strict (`True`) velocity clipping completely invalidates all
            rejected frames.  The alternative flags rejected frames with the
            SKIP_SOURCE_MODELING flag.  The default is taken from the
            'vclip.strict' configuration setting.
        sigma_clip : float, optional
            Invalidates frame speeds using an iterative sigma clipping to
            sequentially remove speeds that are `sigma_clip` times the
            standard deviation away from the median speed value.

        Returns
        -------
        None
        """
        if speed_range is None:
            speed_range = self.get_speed_clip_range()

        if strict is None:
            strict = self.configuration.get_bool('vclip.strict')

        log.debug(f"Velocity clipping frames (strict={strict}) to "
                  f"range {speed_range.min.value:.3f} -> "
                  f"{speed_range.max:.3f}")

        speed_unit = (self.info.instrument.get_size_unit()
                      / units.Unit('second'))
        velocities, position = self.get_scanning_velocities(
            return_position=True)
        speed = velocities.length.to(speed_unit)

        keep, cut, flag = int_nf.classify_scanning_speeds(
            speeds=speed.value,
            min_speed=speed_range.min.to(speed_unit).value,
            max_speed=speed_range.max.to(speed_unit).value,
            valid=self.frames.valid,
            strict=strict)

        self.frames.set_flags('SKIP_SOURCE_MODELING', indices=flag)

        if sigma_clip is None:
            sigma_clip = self.configuration.get_string('sigmaclip')

        if sigma_clip is not None:
            try:
                sigma_multiplier = float(sigma_clip)
            except ValueError:
                sigma_multiplier = 5.0
            log.debug(f"Sigma clipping speeds (sigma={sigma_multiplier}).")
            valid = self.frames.valid & self.frames.is_unflagged(
                'SKIP_SOURCE_MODELING')

            clipped_mask = utils.robust_sigma_clip_mask(
                speed, mask=valid, sigma=sigma_multiplier, verbose=True)
            clipped_indices = np.nonzero(valid & ~clipped_mask)[0]
            keep_mask = np.full(self.size, False)
            keep_mask[keep] = True
            keep_mask[clipped_indices] = False
            keep = np.nonzero(keep_mask)[0]

            if strict:
                self.frames.valid[clipped_indices] = False
                cut = np.unique(np.concatenate([cut, clipped_indices]))
            else:
                self.frames.set_flags(
                    'SKIP_SOURCE_MODELING', indices=clipped_indices)
                flag = np.unique(np.concatenate([flag, clipped_indices]))

        flagged_percent = 100 * flag.size / self.size
        cut_percent = 100 * cut.size / self.size
        log.debug(f"Discarding unsuitable mapping speeds. "
                  f"[{flagged_percent}% flagged, {cut_percent}% clipped]")

        if not self.configuration.is_configured('vcsv'):
            return

        df = pd.DataFrame({
            'X pos': position.x,
            'Y pos': position.y,
            'Velocity': speed,
            'Range Min': speed_range.min,
            'Range Max': speed_range.max})

        speed_string = str(np.round(speed_range.min.to(speed_unit).value, 2))
        base_used = os.path.join(
            self.configuration.work_path, f'used{speed_string}')

        used_file = base_used + '.csv'
        i = 0
        while os.path.isfile(used_file):
            i += 1
            used_file = base_used + f'({i}).csv'
        with open(used_file, 'w') as f:
            df.iloc[keep].to_csv(f, index=False)
        log.info(f"Wrote valid speeds to {used_file}")

        base_cleared = os.path.join(
            self.configuration.work_path, f'cleared{speed_string}')
        cleared_file = base_cleared + '.csv'
        i = 0
        while os.path.isfile(cleared_file):
            i += 1
            cleared_file = base_cleared + f'({i}).csv'

        cleared = cut if strict else flag
        with open(cleared_file, 'w') as f:
            df.iloc[cleared].to_csv(f, index=False)
        log.info(f"Wrote invalid speeds to {cleared_file}")

    def get_accelerations(self):
        """
        Return the telescope accelerations.

        Returns
        -------
        astropy.units.Quantity (numpy.ndarray)
            An array of accelerations of shape (N, 2) containing the (az, alt)
            telescope accelerations in the default size units/second/second.
        """
        position = self.get_smooth_positions(
            self.motion_flagspace.flags.TELESCOPE)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            acceleration_coordinates = int_nf.calculate_2d_accelerations(
                position.coordinates.to('radian').value,
                self.info.instrument.sampling_interval.to('second').value
            ) * units.Unit('radian/second2')

        acceleration = Coordinate2D(unit=self.info.instrument.get_size_unit()
                                    / units.Unit('s2'))
        acceleration.set(acceleration_coordinates)
        return acceleration

    def acceleration_cut(self, max_acceleration):
        """
        Invalidate frames with excessive acceleration.

        Parameters
        ----------
        max_acceleration : astropy.units.Quantity
            The maximum allowable acceleration.  Should be in size/time/time
            units.

        Returns
        -------
        None
        """
        acc_unit = (self.info.instrument.get_size_unit()
                    / units.Unit('second')
                    / units.Unit('second'))
        acceleration_magnitude = self.get_accelerations().length.to(acc_unit)

        keep, cut, flag = int_nf.classify_scanning_speeds(
            speeds=acceleration_magnitude.value,
            min_speed=0,
            max_speed=max_acceleration.to(acc_unit).value,
            valid=self.frames.valid,
            strict=True)

        flagged_percent = 100 * flag.size / self.size
        cut_percent = 100 * cut.size / self.size
        log.debug(f"Discarding excessive telescope accelerations "
                  f"(> {max_acceleration:.3f}). "
                  f"[{flagged_percent}% flagged, {cut_percent}% clipped]")
        average_acc = np.nanmedian(
            acceleration_magnitude.value[keep]) * acc_unit
        std_acc = np.nanstd(
            acceleration_magnitude.value[keep]) * acc_unit
        log.debug(f"Median acceleration: {average_acc:.3f} +- {std_acc:.3f}")

    def acceleration_clip(self, max_acceleration=None):
        """
        Invalidate frames with excessive acceleration.

        Parameters
        ----------
        max_acceleration : astropy.units.Quantity
            The maximum allowable acceleration.  Should be in size/time/time
            units.  If not supplied, is read from the 'aclip' configuration
            option in units of arcseconds/second/second.  If neither value is
            available, no acceleration clipping will occur.

        Returns
        -------
        None
        """
        if max_acceleration is None:
            max_acceleration = self.configuration.get_float(
                'aclip', default=None)

            if max_acceleration is None:
                return
            max_acceleration *= units.Unit('arcsec/s2')

        self.acceleration_cut(max_acceleration)

    def downsample(self, factor=None):
        """
        Downsample frame data by a given, or derived factor.

        Parameters
        ----------
        factor : int, optional
            The factor by which to downsample frames.  If not supplied, will
            be determined using the `get_downsample_factor` method.

        Returns
        -------

        """
        if factor is None:
            factor = self.get_downsample_factor()
        if factor <= 1:
            return

        window = self.downsampling_window(factor)
        n = utils.roundup_ratio(self.size - window.size, factor)

        if n <= 0:
            log.warning(f"Time stream (n={self.size}) is too short to "
                        f"downsample by specified amount ({factor}).")
            return

        log.debug(f"Downsampling by {factor} to {n} frames.")

        center_offset = (window.size // 2) + 1
        start_indices = np.arange(n) * factor
        to_indices = start_indices + window.size
        central_indices = to_indices - center_offset

        downsampled_valid = int_nf.get_valid_downsampling_frames(
            valid_frames=self.frames.valid,
            start_indices=start_indices,
            window_size=window.size)

        downsampled_frames = self.frames[central_indices]
        downsampled_frames.set_from_downsampled(
            self.frames, start_indices, downsampled_valid, window)
        self.frames = downsampled_frames

        self.info.instrument.sampling_interval *= factor
        self.info.instrument.integration_time *= factor

        # Clear dependents and signals
        self.dependents = None
        self.signals = None  # A dictionary by mode
        self.reindex()

    @staticmethod
    def downsampling_window(factor):
        """
        Create a normalized Hann window and the new indices.

        Parameters
        ----------
        factor : int
            The downsampling factor.

        Returns
        -------
        window, indices : numpy.ndarray (float), numpy.ndarray (int)
            The Hann kernel, and the indices over which the center of the
            kernel should be placed to generate the downsampled data.
        """
        # This is 1.5 (Hann factor) * 1.82 = 2.73
        window_size_float = 2.73 * factor
        if (window_size_float % 1) == 0.5:  # pragma: no cover
            # rare but possible to hit exactly 0.5
            window_size = int(np.ceil(window_size_float))
        else:
            window_size = int(np.round(window_size_float))

        window = scipy_windows.hann(window_size, sym=True)
        window /= np.sum(np.abs(window))
        return window

    def get_downsample_factor(self):
        """
        Return the factor by which to downsample frames.

        The downsampling factor is retrieved from the configuration
        'downsample' option which may either be valued as 'auto', or an
        integer value. Any value derived at this stage may be corrected by the
        'downsample.autofactor' configuration setting valued as an integer or
        float. i.e, final factor = downsample * autofactor.

        Returns
        -------
        factor : int
           A factor greater than 1 indicates downsampling should occur.
        """
        # Keep to the rule of thumb of at least 2.5 samples per beam
        downsample = self.configuration.get('downsample', default=None)
        if downsample is None:
            return 1

        if str(downsample).lower().strip() == 'auto':
            if np.isnan(self.average_scan_speed[0]):
                self.calculate_scan_speed_stats()
            speed, weight = self.average_scan_speed
            speed_rms = np.sqrt(1 / weight)
            # Choose downsampling to accommodate at ~90% of scanning speeds
            maximum_speed = speed + (1.25 * speed_rms)
            if maximum_speed == 0:
                log.warning("No automatic downsampling for zero scan speed.")
                return 1
            elif np.isnan(maximum_speed):
                log.warning("No automatic downsampling for unknown scanning "
                            "speed.")
                return 1

            point_size = self.scan.get_point_size()
            if np.isnan(point_size):
                log.warning("No automatic downsampling for unknown "
                            "point size.")
                return 1

            max_time = 0.4 * self.scan.get_point_size() / maximum_speed
            dt = self.info.instrument.sampling_interval
            if np.isnan(dt):
                log.warning("No automatic downsampling for unknown sampling "
                            "interval.")
                return 1

            factor = int(np.floor(max_time / dt).decompose().value)
            if factor >= 1e15:
                log.warning("No automatic downsampling for negligible scan "
                            "speed.")
                return 1

            auto_factor = self.configuration.get_float('downsample.autofactor',
                                                       default=1.0)
            factor = int(np.floor(auto_factor * factor))
            return np.clip(factor, 1, None)
        else:
            return int(np.clip(int(downsample), 1, None))

    def get_frame_gaps(self):
        """
        Return the gaps between frames.

        Returns
        -------
        frame_gaps, time_gaps : numpy.ndarray (int), units.Quantity
        """
        first_index = self.frames.get_first_frame_value('fixed_index')
        first_mjd = self.frames.get_first_frame_value('mjd')

        dt = self.info.sampling_interval
        measured_time = (self.frames.mjd - first_mjd) * units.Unit('day')
        expected_time = (self.frames.fixed_index - first_index) * dt
        gap_time = (measured_time - expected_time).decompose().to(dt.unit)
        frame_gaps = np.round((gap_time / dt).decompose().value).astype(int)
        frame_gaps[~self.frames.valid] = 0
        gap_time[~self.frames.valid] = np.nan
        return frame_gaps, gap_time

    def has_gaps(self, tolerance=1):
        """
        Check if there are gaps in the time stream data.

        This assumes a uniform sampling interval between frame measurements
        and calculates gaps based on deviation from the expected MJD of a
        frame vs. it's recorded value.

        Parameters
        ----------
        tolerance : int or float
            The maximum number of frame missing in a gap.

        Returns
        -------
        bool
        """
        log.debug("Checking for gaps:")
        frame_gaps, gap_time = self.get_frame_gaps()

        if frame_gaps.max() > tolerance:
            index = np.nanargmax(gap_time)
            frame = self.frames.fixed_index[index]
            max_gap = gap_time[index].to('ms')
            n = frame_gaps[index]
            mjd = self.frames.mjd[index]
            log.warning(f'Gap(s) found: max = {max_gap} ({n} frames) at '
                        f'frame {frame} (mjd={mjd})')
            return True

        log.debug("No gaps.")
        return False

    def fill_gaps(self):
        """
        Add invalid frames in cases where there are timestream gaps.

        The MJD of the invalid frames are set to NaN.

        Returns
        -------
        None
        """
        frame_gaps, time_gaps = self.get_frame_gaps()
        max_skip_index = int(np.nanargmax(time_gaps))
        n = frame_gaps[max_skip_index]
        if n == 0:
            return
        if n > 10:
            raise ValueError(
                f"Large gap of {n} frames at "
                f"index {self.frames.fixed_index[max_skip_index]}, "
                f"MJD: {self.frames.mjd[max_skip_index]}")

        add_frames = np.clip(frame_gaps, 0, None)
        log.debug(f"Padding with {add_frames.sum()} empty frames.")

        insert_at = np.nonzero(add_frames)[0]
        insert_indices = []
        for ii in insert_at:
            insert_indices.extend([ii] * add_frames[ii])

        insert_indices = np.asarray(insert_indices, dtype=int)
        self.frames.insert_blanks(insert_indices)

        # Add bad MJDs so no further blanks are inserted
        inserted_indices = insert_indices + np.arange(insert_indices.size)
        self.frames.mjd[inserted_indices] = np.nan
        self.reindex()

    def notch_filter(self):
        """
        Apply notch filtering based on configuration options.

        The frequencies to filter are taken from the configuration
        'frequencies' option, in the notch section.  Other options of
        interest are 'width' (width of the filter), 'harmonics' (number
        of harmonics to include), and `bands`.  If "bands"
        is available, extra frequencies ranging from min(band):max(band) are
        added with separation of 'width'.

        Returns
        -------
        None
        """
        if not self.configuration.is_configured('notch.frequencies'):
            return
        frequencies = np.asarray(
            self.configuration.get_float_list('notch.frequencies'))
        width = self.configuration.get_float('notch.width', default=0.1)

        harmonics = self.configuration.get_int('notch.harmonics', default=1)
        if harmonics > 1:
            frequencies = ((np.arange(harmonics - 1) + 1)[:, None]
                           * frequencies[None]).flatten()

        bands = self.configuration.get_list('notch.bands')
        if len(bands) > 0:
            for band in bands:
                band_range = Range.from_spec(band, is_positive=True)

                frequencies = np.concatenate(
                    (frequencies,
                     np.arange(band_range.min, band_range.max, width)))
        frequencies = np.unique(frequencies)
        self.apply_notch_filter(frequencies, width)

    def apply_notch_filter(self, frequencies, width):
        """
        Apply notch filtering to frame data for given frequencies and width.

        Parameters
        ----------
        frequencies : numpy.ndarray (float)
            The frequencies to filter.
        width : float
            The width of the filter.

        Returns
        -------
        None
        """
        sampling = self.info.instrument.sampling_interval.decompose().value
        window_size = utils.pow2ceil(np.ceil(1 / (width * sampling)))
        nf = window_size >> 1

        log.debug(f"Notching {frequencies.size} bands.")

        fft_freq = np.fft.rfftfreq(window_size, sampling)
        bins = np.interp(frequencies, fft_freq, np.arange(fft_freq.size),
                         left=np.nan, right=np.nan)
        bins = bins[np.isfinite(bins)]
        bins = np.floor(bins).astype(int)
        # add bins + 1
        bins = np.append(bins, bins + 1)
        zero_phase = (bins == nf).any()
        bins = bins[(bins > 0) & (bins < nf)]
        bins = np.unique(bins)

        for i in range(0, self.frames.size, window_size):
            self.process_notch_filter_block(
                i, i + window_size, window_size, bins, zero_phase)

    def process_notch_filter_block(self, start, end, window_size, bins,
                                   zero_phase):
        if end > self.frames.size:
            real_end = self.frames.size
            missing = True
        else:
            missing = False
            real_end = end

        log.debug(f"Notch filtering indices {start} to {real_end}")

        n_frames = real_end - start
        data = np.empty((window_size, self.frames.data.shape[1]), dtype=float)
        valid = np.empty(window_size, dtype=bool)
        data[:n_frames] = self.frames.data[start:real_end].copy()
        valid[:n_frames] = self.frames.valid[start:real_end]
        if missing:
            data[n_frames:] = 0.0
            valid[n_frames:] = False

        data[~valid] = 0
        n = valid.sum()
        if n > 0:
            channel_average = data.sum(axis=0) / n
        else:
            channel_average = np.zeros(data.shape[1], dtype=float)

        data[valid] -= channel_average[None]
        data = np.fft.rfft(data, axis=0)
        if zero_phase:
            data[0].imag = 0

        data[bins] = 0
        data = np.fft.irfft(data, axis=0)
        self.frames.data[start:real_end][valid[:n_frames]] = (
            data[valid] + channel_average[None])

    def offset(self, value):
        """
        Add an offset value to frame data for all unflagged channels.

        Parameters
        ----------
        value : float

        Returns
        -------
        None
        """
        unflagged_channels = self.channels.data.is_unflagged()
        self.frames.data[:, unflagged_channels] += value

    def get_position_signal(self, motion_flag, direction, mode=None):
        """
        Return a position signal.

        Parameters
        ----------
        motion_flag : MotionFlagTypes or str or int
            The motion type for which to extract positions.
        direction : str or MotionFlag
            The direction of motion.
        mode : Mode, optional

        Returns
        -------
        Signal
        """
        positions = self.get_positions(motion_flag)
        if not isinstance(direction, MotionFlags):
            direction = self.motion_flagspace(direction)

        values = direction(positions)
        values[~self.frames.valid] = np.nan
        return Signal(self, mode=mode, values=values, is_floating=True)

    def get_acceleration_signal(self, direction, mode=None):
        """
        Return an acceleration signal.

        Parameters
        ----------
        direction : str or MotionFlag
            The direction of motion.
        mode : Mode, optional

        Returns
        -------
        Signal
        """
        accelerations = self.get_accelerations()
        if not isinstance(direction, self.motion_flagspace):
            direction = self.motion_flagspace(direction)
        acceleration = direction(accelerations)
        acceleration[~self.frames.valid] = np.nan
        return Signal(self, mode=mode, values=acceleration, is_floating=False)

    def get_spectra(self, window_function='hamming', window_size=None):
        """
        Return the power spectrum of the data.

        The returned power spectrum is computed using Welch's method
        of dividing the data into periodograms and averaging the result.

        Parameters
        ----------
        window_function : str or function, optional
            The window filter name.  The default is "hamming".  Must be an
            available function in `scipy.signal.windows`.
        window_size : int, optional
            The size of the filter in frames.  The default is twice the number
            of frames in the filtering time scale.

        Returns
        -------
        frequency, spectrum : Quantity, Quantity
            The frequency spectrum is of shape (nf2,) in units of Hertz.  The
            spectrum is of shape (nf2, n_channels) in units of Janskys.
        """
        if window_size is None:
            window_size = 2 * self.frames_for(self.filter_time_scale)

        sampling_frequency = (1.0 / self.info.instrument.sampling_interval
                              ).decompose().value

        invalid = ~self.frames.valid
        invalid |= self.frames.is_flagged(self.flagspace.flags.MODELING_FLAGS)
        invalid = invalid[:, None] | (self.frames.sample_flag != 0)
        data = self.frames.data.copy()
        if invalid.any():
            data[invalid] = 0.0

        window_size = int(np.clip(window_size, 1, data.shape[0]))
        frequency, power = welch(data,
                                 fs=sampling_frequency,
                                 window=window_function,
                                 nperseg=window_size,
                                 scaling='density',
                                 average='mean',
                                 axis=0)
        janskys = self.gain * self.info.instrument.jansky_per_beam()
        spectrum = (np.sqrt(power) / janskys) * units.Unit('Jy')
        frequency = frequency * units.Unit('Hz')
        return frequency, spectrum

    def write_products(self):
        """
        Write the integration products to an output file.

        Returns
        -------
        None
        """
        if self.has_option('write.pattern'):
            try:
                self.write_scan_pattern()
            except Exception as err:
                log.warning(f"Could not write scan pattern: {err}")

        if self.configuration.get_bool('write.pixeldata'):
            out_file = os.path.join(self.configuration.work_path,
                                    f'pixel-{self.get_file_id()}.dat')
            try:
                self.channels.write_channel_data(
                    out_file, header=self.get_ascii_header())
            except Exception as err:
                log.warning(f"Could not write pixel data: {err}")

        if self.configuration.get_bool('write.flatfield'):
            if self.has_option('write.flatfield.name'):
                out_name = self.configuration.get_string(
                    'write.flatfield.name')
            else:
                out_name = f'flat-{self.get_file_id()}.fits'
            out_file = os.path.join(self.configuration.work_path, out_name)
            try:
                self.channels.write_flat_field(out_file)
            except Exception as err:
                log.warning(f"Could not write flat field: {err}")

        if self.has_option('write.covar'):
            try:
                self.write_covariances()
            except Exception as err:
                log.warning(f"Could not write covariances: {err}")

        if self.configuration.get_bool('write.ascii'):
            try:
                self.write_ascii_time_stream()
            except Exception as err:
                log.warning(f'Could not write time stream data: {err}')

        if self.configuration.get_bool('write.signals'):
            for name, signal in self.signals.items():
                try:
                    out_file = os.path.join(
                        self.configuration.work_path,
                        f'{signal.mode.name}-{self.get_file_id()}.tms')
                    signal.write_signal_values(out_file)
                    log.info(f"Written signal data to {out_file}")
                except Exception as err:
                    log.warning(f"Could not write signal data: {err}")

        if self.has_option('write.spectrum'):
            window_name = self.configuration.get('write.spectrum',
                                                 default='Hamming')
            window_size = self.configuration.get(
                'write.spectrum.size',
                default=2 * self.frames_for(self.filter_time_scale))
            try:
                self.write_spectra(window_name=window_name,
                                   window_size=window_size)
            except Exception as err:
                log.warning(f"Could not write spectra: {err}")

        if self.has_option('write.coupling'):
            try:
                self.write_coupling_gains(
                    self.configuration.get_list('write.coupling'))
            except Exception as err:
                log.warning(f"Could not write coupling gains: {err}")

    def write_spectra(self, window_name=None, window_size=None):
        """
        Write the spectra to file.

        Parameters
        ----------
        window_name : str, optional
            The name of the filtering window.  If not supplied, will be
            extracted from the configuration option "write.spectrum" or default
            to 'Hamming' if not found.
        window_size : int, optional
            The size of the filtering window.  Will be extracted from the
            configuration option "write.spectrum.size" if not supplied.  If
            not found in the configuration, the default will be 2 times the
            number of frames in the filtering time scale.

        Returns
        -------
        None
        """
        file_name = os.path.join(self.configuration.work_path,
                                 self.get_file_id()) + '.spec'

        if window_name is None:
            window_name = self.configuration.get('write.spectrum',
                                                 default='Hamming')
        if window_size is None:
            window_size = self.configuration.get(
                'write.spectrum.size',
                default=2 * self.frames_for(self.filter_time_scale))

        freq, power = self.get_spectra(
            window_function=window_name.lower().strip(),
            window_size=window_size)

        header = ['# SOFSCAN Residual Detector Power Spectra',
                  '',
                  self.get_ascii_header(),
                  f'# Window Function: {window_name}',
                  f'# Window Size: {window_size} samples',
                  '# PSF unit: Jy/sqrt(Hz)',
                  '',
                  '# f(Hz),PSD(ch=0),PSD(ch=1),...',
                  '']

        with open(file_name, 'w') as f:
            f.write('\n'.join(header))
            for ff, pp in zip(freq.value, power.value):
                line = f"{'%.3e' % ff},{','.join(['%.3e' % p for p in pp])}"
                print(line, file=f)

        log.info(f"Written power spectra to {file_name}")

    def write_covariances(self):
        """
        Write the covariances to file.

        Returns
        -------
        None
        """
        covariance = self.get_covariance()
        specs = self.configuration.get_list('write.covar', default=[])
        if len(specs) == 0:
            specs.append('full')

        prefix = os.path.join(self.configuration.work_path, 'covar')
        prefix += f'-{self.get_file_id()}'

        for name in specs:
            name = name.lower().strip()
            if name == 'full':
                filename = f'{prefix}.fits'
                self.write_covariance_to_file(
                    filename, self.get_full_covariance(covariance))

            elif name == 'reduced':
                filename = f'{prefix}.reduced.fits'
                self.write_covariance_to_file(filename, covariance)

            else:
                division = self.channels.divisions.get(name)
                if division is None:
                    log.warning(f"Cannot write covariance for {name}. "
                                f"Undefined grouping.")
                    return
                filename = f'{prefix}.{name}.fits'
                self.write_covariance_to_file(
                    filename, self.get_group_covariance(division, covariance))

    def write_covariance_to_file(self, filename, covariance):
        """
        Write a covariance matrix to file.

        Parameters
        ----------
        filename : str
            The path of the file to write to.
        covariance : numpy.ndarray (float)
            The covariance matrix.

        Returns
        -------
        None
        """
        if self.configuration.get_bool('write.covar.condensed'):
            covariance = self.condense_covariance(covariance)

        if not filename.endswith('.fits'):
            filename += f'-{self.scan.get_id()}-{self.get_file_id()}.fits'

        if covariance is None:
            return

        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data=covariance))
        hdul.writeto(filename, overwrite=True)
        hdul.close()
        log.info(f"Written covariance to {filename}")

    @staticmethod
    def condense_covariance(covariance):
        """
        Strips a covariance matrix of all zero and NaN column/row pairs.

        Parameters
        ----------
        covariance : numpy.ndarray (float)
            The covariance matrix of shape (n_channels, n_channels).

        Returns
        -------
        condensed_covariance : numpy.ndarray (float)
            The condensed covariance matrix without zero or NaN values of shape
            (n_valid, n_valid).
        """
        valid = np.isfinite(covariance) & (covariance != 0)
        keep = np.nonzero(np.any(valid, axis=0))[0]
        if keep.size == covariance.shape[0]:
            return covariance
        return covariance[keep[:, None], keep[None]]

    def get_full_covariance(self, covariance):
        """
        Return the full covariance array for all initial channels.

        Parameters
        ----------
        covariance : numpy.ndarray (float)
            The covariance array for all channels (not including the initial
            channels removed).

        Returns
        -------
        full_covariance : numpy.ndarray (float)
            The full covariance array of shape (store_channels,
            store_channels).
        """
        return int_nf.get_full_covariance_matrix(
            covariance=covariance,
            fixed_indices=self.channels.data.fixed_index)

    @staticmethod
    def get_group_covariance(division, covariance):
        """
        Return the covariance for all groups in a channel division.

        Parameters
        ----------
        division : ChannelDivision
            The channel division for which to extract covariances.
        covariance : numpy.ndarray (float)
            The covariance matrix of shape (all_channels, all_channels) for
            all available channels.

        Returns
        -------
        group_covariance : numpy.ndarray (float)
            The covariance for channels in the division.
        """
        channel_indices = np.empty(0, dtype=int)
        for group in division:
            channel_indices = np.concatenate((channel_indices, group.indices))

        return int_nf.get_partial_covariance_matrix(
            covariance=covariance,
            indices=channel_indices)

    def get_covariance(self):
        """
        Return the channel covariance.

        Returns
        -------
        covariance : numpy.ndarray (float)
            The covariance matrix of shape (n_channels, n_channels).
        """
        log.info("Calculating covariance matrix (this may take a while...)")
        return int_nf.get_covariance(
            frame_data=self.frames.data,
            frame_valid=self.frames.valid,
            frame_weight=self.frames.relative_weight,
            channel_flags=self.channels.data.flag,
            channel_weight=self.channels.data.weight,
            sample_flags=self.frames.sample_flag,
            frame_flags=self.frames.flag,
            source_flags=self.flagspace.convert_flag('SOURCE_FLAGS').value)

    def write_ascii_time_stream(self, filename=None):
        """
        Write the frame data to a text file.

        Parameters
        ----------
        filename : str, optional
            The name of the file to output the time stream results.

        Returns
        -------
        None
        """
        if filename is None:
            filename = os.path.join(
                self.configuration.work_path,
                f'{self.scan.get_id()}-{self.get_file_id()}.tms')

        with open(filename, 'w') as f:
            print(f'# {1 / self.info.sampling_interval.decompose().value:.3e}',
                  file=f)

        data = self.frames.data.copy()
        valid_frames = self.frames.valid & self.frames.is_unflagged('BAD_DATA')
        spike = self.flagspace.convert_flag('SAMPLE_SPIKE').value
        valid_samples = (self.frames.sample_flag & spike) == 0
        valid_samples &= valid_frames[:, None]
        data[~valid_samples] = np.nan

        log.info(f"Writing time stream data to {filename}")

        with open(filename, 'a') as f:
            for frame in range(data.shape[0]):
                if (frame % 1000) == 0:
                    print(frame)
                line = ','.join([f'{x:.5e}' for x in data[frame]])
                print(line, file=f)

        # # This is slow
        # df = pd.DataFrame(data=data)
        # df.to_csv(filename, index=False, mode='a', header=False,
        #           float_format='%.5e')

    def write_scan_pattern(self, filename=None):
        """
        Write the scan pattern to file.

        Parameters
        ----------
        filename : str, optional
            The path to the output file.

        Returns
        -------
        None
        """
        if filename is None:
            filename = os.path.join(self.configuration.work_path,
                                    f'pattern-{self.get_file_id()}.dat')

        offsets = self.frames.get_base_native_offset()
        x = offsets.x.to('arcsec').value
        y = offsets.y.to('arcsec').value
        invalid = ~self.frames.valid
        flagged = self.frames.is_flagged()
        invalid_line = '---,---'
        flagged_line = '...,...'

        log.info(f"Writing scan pattern to {filename}")
        with open(filename, 'w') as f:
            for frame in range(x.size):
                if invalid[frame]:
                    print(invalid_line, file=f)
                elif flagged[frame]:
                    print(flagged_line, file=f)
                else:
                    print(f'{x[frame]:.3f},{y[frame]:.3f}', file=f)

    def write_coupling_gains(self, signal_names):
        """
        Write the coupling gains to file.

        Parameters
        ----------
        signal_names : list (str)
            The signals to write.

        Returns
        -------
        None
        """
        if signal_names is None or len(signal_names) == 0:
            return
        for signal_name in signal_names:
            modality = self.channels.modalities.get(signal_name)
            if modality is None:
                continue
            modality.update_all_gains(integration=self, robust=False)
            gains = np.full(self.channels.size, np.nan)
            for mode in modality.modes:
                signal = self.get_signal(mode)
                if signal is not None:
                    signal_gains = signal.mode.get_gains()
                    try:
                        gains[mode.channel_group.indices] = signal_gains
                    except(TypeError, IndexError, ValueError):
                        pass

            filename = os.path.join(
                self.configuration.work_path,
                f'{self.get_file_id()}.{signal_name}-coupling.dat')
            channel_ids = self.channels.data.channel_id
            keep = gains != 0
            channel_ids = channel_ids[keep]
            gains = gains[keep]
            log.info(f"Writing coupling gains to {filename}")
            with open(filename, 'w') as f:
                print(self.get_ascii_header(), file=f)
                print('#', file=f)
                print('# ch\tgain', file=f)
                for channel, gain in zip(channel_ids, gains):
                    print(f'{channel}\t{gain:.3f}', file=f)

    def get_crossing_time(self, source_size=None):
        """
        Return the crossing time for a given source size.

        Parameters
        ----------
        source_size : astropy.units.Quantity, optional
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

        if np.isnan(self.average_scan_speed[0]):
            self.calculate_scan_speed_stats()
        scan_speed = self.average_scan_speed[0]

        chopper = getattr(self, 'chopper', None)
        if chopper is not None:
            return min(chopper.stare_duration, source_size / scan_speed)

        freq = self.get_modulation_frequency(self.flagspace.flags.TOTAL_POWER)
        if freq == 0:
            modulation_time = 0.0 * units.Unit('second')
        else:
            modulation_time = (1.0 / freq).decompose()

        return modulation_time + (source_size / scan_speed)

    def get_point_crossing_time(self):
        """
        Return the time required to cross the point size of the scan.

        Returns
        -------
        time : astropy.units.Quantity
            The point crossing time in time units.
        """
        return self.get_crossing_time(source_size=self.scan.get_point_size())

    def get_modulation_frequency(self, signal_flag):
        """
        Return the modulation frequency.

        The modulation frequency is taken from the chopper frequency if
        available, or set to 0 Hz otherwise.

        Parameters
        ----------
        signal_flag : FrameFlagTypes or str or int
            The signal flag (not relevant for this implementation).

        Returns
        -------
        frequency : astropy.units.Quantity.
            The modulation frequency in Hz.
        """
        chopper = getattr(self, 'chopper', None)
        if chopper is None:
            return 0.0 * units.Unit('Hz')
        else:
            return chopper.frequency

    def get_mjd(self):
        """
        Return the midpoint MJD of the integration.

        The midpoint MJD is defined as the average of the first and last
        valid frame MJD values.

        Returns
        -------
        mjd : float
        """
        mjd_1 = self.frames.get_first_frame_value('mjd')
        mjd_2 = self.frames.get_last_frame_value('mjd')
        return (mjd_1 + mjd_2) / 2.0

    def get_exposure_time(self):
        """
        Return the total exposure time.

        Returns
        -------
        time : astropy.units.Quantity
            The exposure time in time units.
        """
        n_frames = self.get_frame_count(
            discard_flag=self.flagspace.flags.SKIP_SOURCE_MODELING)
        return n_frames * self.info.instrument.sampling_interval

    def get_duration(self):
        """
        Return the total integration duration.

        Returns
        -------
        time : astropy.units.Quantity
            The integration duration in time units.
        """
        return self.size * self.info.instrument.sampling_interval

    def get_ascii_header(self):
        """
        Return a string header for output text files.

        Returns
        -------
        header : str
        """
        timestamp = self.scan.info.astrometry.time_stamp
        mjd = self.scan.mjd
        exposure = self.get_frame_count(discard_flag='SOURCE_FLAGS'
                                        ) * self.info.integration_time

        header = [f'# Instrument: {self.info.instrument.name}',
                  f'# Scan: {self.scan.get_id()}']
        if self.scan.size > 1:
            header.append(f'# Integration: {self.integration_number + 1}')
        header.extend([
            f'# Object: {self.scan.source_name}',
            f'# Date: {timestamp} (MJD: {mjd})',
            f'# Project: {self.scan.info.observation.project}',
            f'# Exposure: {exposure}',
            f'# Equatorial: {self.scan.equatorial}'])
        if self.info.astrometry.ground_based:
            header.append(f"# Horizontal: {self.scan.horizontal}")
        header.append(
            f'# SOFSCAN version: {ReductionVersion.get_full_version()}')
        return '\n'.join(header)

    def get_id(self):
        """
        Return the simple integration ID.

        Returns
        -------
        str
        """
        if self.integration_number is None:
            return '1'
        else:
            return str(self.integration_number + 1)

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
        return f'{self.scan.get_id()}{separator}{self.get_id()}'

    def get_standard_id(self, separator='|'):
        """
        Get the standard integration ID.

        Parameters
        ----------
        separator : str, optional
            The separator character/phase between the scan and integration ID.

        Returns
        -------
        str
        """
        if self.scan is not None and (self.size > 1 or self.scan.is_split):
            return self.get_full_id(separator=separator)
        else:
            return self.get_id()

    def get_display_id(self):
        """
        Get the display integration ID.

        Returns
        -------
        str
        """
        return self.get_standard_id(separator='|')

    def get_file_id(self):
        """
        Get the file integration ID.

        Returns
        -------
        str
        """
        return self.get_standard_id(separator='-')

    def add_dependents(self, dependents):
        """
        Add dependents to the integration.

        Parameters
        ----------
        dependents : Dependents
            The dependents to add.

        Returns
        -------
        None
        """
        if self.dependents is None:
            self.dependents = {dependents.name: dependents}
        else:
            self.dependents[dependents.name] = dependents

    def get_dependents(self, name):
        """
        Return or create and return dependents of the given name.

        Parameters
        ----------
        name : str
            The name of the dependents.

        Returns
        -------
        Dependents
        """
        if self.dependents is None or name not in self.dependents:
            return Dependents(self, name)
        else:
            return self.dependents[name]

    def get_channel_weights(self, method=None):
        """
        Derive and set channel weights.

        Parameters
        ----------
        method : str
            The method to derive channel weights.  Must be one of {robust,
            differential, rms}.  If not supplied, will be determined from the
            configuration.  Defaults to 'rms' if no valid method can be found.

        Returns
        -------
        None
        """
        if method is None:
            method = self.configuration.get_string('weighting.method',
                                                   default='rms')
        method = str(method).strip().lower()

        if method == 'robust':
            self.get_robust_channel_weights()
        elif method == 'differential':
            self.get_differential_channel_weights()
        else:
            self.get_rms_channel_weights()
        self.flag_weights()

    def get_robust_channel_weights(self):
        """
        Derive and set robustly derived channel weights for live channels.

        The variance and variance weight for a channel `i` is given as::

           var = median(relative_weight * frame_data[:, i] ** 2)
           weight = sum(relative_weight)

        taken over all valid frames for the channel `i` and zero flagged
        samples for the channel `i`.  These values are then passed to
        `set_weights_from_var_stats` to set the channel degrees of
        freedom, weights, and variances for channels.

        Returns
        -------
        None
        """
        self.comments.append('[W]')
        live_channels = self.channels.get_live_channels()
        valid_frames = self.frames.is_unflagged(
            self.flagspace.flags.CHANNEL_WEIGHTING_FLAGS) & self.frames.valid

        var_sum, var_weight = int_nf.robust_channel_weights(
            frame_data=self.frames.data,
            relative_weights=self.frames.relative_weight,
            sample_flags=self.frames.sample_flag,
            valid_frames=valid_frames,
            channel_indices=live_channels.indices)

        self.set_weights_from_var_stats(
            live_channels, var_sum, var_weight)

    def get_differential_channel_weights(self):
        """
        Derive and set differentially derived weights for live channels.

        The variance and variance weight for a channel `i` is given as::

           var = sum(w * (data[:, i] - data[:, i+delta])^2)
           weight = sum(w)

        taken over all valid frames for the channel `i` and `i+delta`, and zero
        flagged samples for the channel `i` and `i+delta`.  These values are
        then passed to `set_weights_from_var_stats` to set the channel
        degrees of freedom, weights, and variances for channels.

        In this case delta is defined as 10 times the number of frames
        required to cross the point response of the instrument (based on
        scanning speed).

        Returns
        -------
        None
        """
        self.comments.append('w')
        frame_delta = self.frames_for(10 * self.get_point_crossing_time())
        live_channels = self.channels.get_live_channels()
        valid_frames = self.frames.is_unflagged(
            self.flagspace.flags.CHANNEL_WEIGHTING_FLAGS) & self.frames.valid

        var, var_weight = int_nf.differential_channel_weights(
            frame_data=self.frames.data,
            relative_weights=self.frames.relative_weight,
            sample_flags=self.frames.sample_flag,
            valid_frames=valid_frames,
            channel_indices=live_channels.indices,
            frame_delta=frame_delta)

        self.set_weights_from_var_stats(live_channels, var, var_weight)

    def get_rms_channel_weights(self):
        """
        Derive and set channel weights for live channels based on data values.

        The variance and variance weight for a channel `i` is given as:

            var = sum(w * data^2)
            weight = sum(w)

        taken over all valid frames for the channel `i`, and zero flagged
        samples for the channel `i`.  These values are then passed to
        `set_weights_from_var_stats` to set the channel degrees of
        freedom, weights, and variances for channels.

        Returns
        -------
        None
        """
        self.comments.append('W')
        valid_frames = self.frames.is_unflagged('CHANNEL_WEIGHTING_FLAGS')
        valid_frames &= self.frames.valid

        live_channels = self.channels.get_live_channels()
        var_sum, var_weight = int_nf.rms_channel_weights(
            frame_data=self.frames.data,
            frame_weight=self.frames.relative_weight,
            valid_frames=valid_frames,
            sample_flags=self.frames.sample_flag,
            channel_indices=live_channels.indices)
        self.set_weights_from_var_stats(live_channels, var_sum, var_weight)

    @staticmethod
    def set_weights_from_var_stats(channel_group, var_sum, var_weight):
        """
        Set channel weights from the calculated variance.

        In addition to channel weights, the degrees of freedom (DOF) and
        channel variances are also set for a given channel group where:

            DOF = 1 - (dependents / var_weight)
            channel_variance = variance
            weight = DOF / variance

        Parameters
        ----------
        channel_group : ChannelGroup or ChannelData
            The channel group or data for which to set weights.  Should be of
            size n_channels.
        var_sum: numpy.ndarray (float)
            The sum of weight * variance over all valid frames (n_channels,).
        var_weight : numpy.ndarray (float)
            The variance weights of shape (n_channels,).

        Returns
        -------
        None
        """
        base_data = channel_group.data
        int_nf.set_weights_from_var_stats(
            channel_indices=channel_group.indices,
            var_sum=var_sum,
            var_weight=var_weight,
            base_dependents=base_data.dependents,
            base_dof=base_data.dof,
            base_variance=base_data.variance,
            base_weight=base_data.weight)

    def flag_weights(self):
        """
        Check if channel weights should be flagged and update source NEFD.

        Returns
        -------
        None
        """
        self.channels.flag_weights()
        self.calculate_source_nefd()
        self.comments.append(str(self.channels.n_mapping_channels))

    def calculate_source_nefd(self):
        """
        Set the source Noise-Equivalent-Flux-Density (NEFD).

        Returns
        -------
        None
        """
        self.nefd = self.channels.get_source_nefd(self.gain)
        if self.configuration.get_bool('nefd.map'):
            scan_weight = self.scan.weight
            if scan_weight > 0:
                self.nefd /= np.sqrt(scan_weight)
            else:
                self.nefd = np.inf

        nefd = self.nefd / self.info.instrument.jansky_per_beam()
        self.comments.append(f'{nefd:.2e}')

    def get_time_weights(self, channels=None, block_size=None, flag=True):
        """
        Derive and set frame weights.

        Relative weight values are derived for chunks of frames in turn.
        The `block_size` parameter determines how many frames are to be
        included in each chunk.  If not supplied it is either retrieved
        from the 'weighting.frames.resolution` option (in seconds), or
        defaults to the number of frames in 10 seconds.

        This process sets the both the frame relative weights, the frame
        degrees of freedom, flags frames that do not have sufficient degrees
        of freedom, and optionally flags frames that do not have weights
        (noise values) within the acceptable range (determined by the
        'weighting.frames.noiserange` configuration option).

        Parameters
        ----------
        channels : Channels or ChannelData or ChannelGroup, optional
            The channels from which to derive weights.
        block_size : int, optional
            Processing occurs serially in blocks, and values are derived for
            each block independently.  The block size determines the number of
            frames in each block.
        flag : bool, optional
            If `True`, flag frames that are outside of the weight range
            determined by the 'weighting.frames.noiserange' configuration
            option.  Such flags will be marked with the FLAG_WEIGHT flag.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.channels

        if block_size is None:
            spec = self.configuration.get_string('weighting.frames.resolution')
            if spec is not None:
                block_size = self.filter_frames_for(
                    spec, default_time=10 * units.Unit('second'))
                block_size = utils.pow2ceil(block_size)
            else:
                block_size = 1

        self.comments.append('tW')
        if block_size > 1:
            self.comments.append(f'({block_size})')

        if hasattr(channels, 'create_channel_group'):
            channel_group = channels.create_channel_group()
        else:
            # Assume a channel group has already been passed in
            channel_group = channels

        frame_weight_flag = self.flagspace.convert_flag('FLAG_WEIGHT').value
        frame_dof_flag = self.flagspace.convert_flag('FLAG_DOF').value

        # TODO: I changed channel time weighting flags to frame time weighting
        #       flags.
        # time_weight_flag = self.channel_flagspace.convert_flag(
        #     'TIME_WEIGHTING').value
        time_weight_flag = self.flagspace.convert_flag(
            'TIME_WEIGHTING_FLAGS').value

        # The following unflags frame FLAG_WEIGHT flag, flags or unflags the
        # frame FLAG_DOF flag, and sets the frame relative_weight and dof.
        int_nf.determine_time_weights(
            block_size=block_size,
            frame_data=self.frames.data,
            frame_dof=self.frames.dof,
            frame_weight=self.frames.relative_weight,
            frame_valid=self.frames.valid,
            frame_dependents=self.frames.dependents,
            frame_flags=self.frames.flag,
            frame_weight_flag=frame_weight_flag,
            frame_dof_flag=frame_dof_flag,
            channel_weights=channel_group.weight,
            channel_indices=channel_group.indices,
            channel_flags=channel_group.flag,
            time_weight_flag=time_weight_flag,
            sample_flags=self.frames.sample_flag
        )

        if not flag:
            return
        if not self.configuration.is_configured('weighting.frames.noiserange'):
            return

        # Here we re-flag those frames outside the acceptable noise range
        # with the FLAG_WEIGHT frame flag.
        weight_range = self.configuration.get_range(
            'weighting.frames.noiserange', is_positive=True)
        out_of_range = ~weight_range.in_range(self.frames.relative_weight)
        out_of_range &= self.frames.valid
        self.frames.set_flags(self.flagspace.flags.FLAG_WEIGHT,
                              indices=out_of_range)

    def dejump_frames(self):
        """
        Remove jumps in the frame data.

        This will also calculate frame relative weights if necessary.

        "Jumps" are determined from the noise (weights) of the frame data.
        At every transition between high and low weights (determined by
        the 'dejump.level' configuration option), the average channel level
        data is subtracted, and dependencies are updated.  Integration signals
        will also be updated.

        Returns
        -------
        None
        """
        log.debug(f"Dejumping frames for integration {self.get_full_id()}.")

        if self.configuration.is_configured('dejump.resolution'):
            resolution = self.configuration.get_float(
                'dejump.resolution', default=np.nan) * units.Unit('second')
            resolution = utils.pow2round(self.frames_for(resolution))
        else:
            resolution = 1

        level_rms = self.configuration.get_float('dejump.level', default=2.0)
        level_weight = 1.0 / (level_rms ** 2)

        # Make sure the level is significant at the 3-sigma level
        level = 1.0 - (9.0 / (resolution * self.channels.n_mapping_channels))
        level = min(level, level_weight)

        robust = self.configuration.get_string('estimator') == 'median'
        self.comments.append('[J]' if robust else 'J')

        if self.configuration.is_configured('dejump.minlength'):
            min_level_time = self.configuration.get_float(
                'dejump.minlength') * units.Unit('second')
        else:
            min_level_time = 5 * self.get_point_crossing_time()

        min_frames = int(
            np.round(min_level_time / self.info.instrument.sampling_interval))
        if min_frames < 2:
            min_frames = np.inf

        # Save the old time weights
        self.frames.temp_c = self.frames.relative_weight.copy()

        parms = self.get_dependents('jumps')

        # Derive new time weights temporarily (frame.relative_weight)
        self.get_time_weights(
            channels=self.channels, block_size=resolution, flag=False)

        levelled = removed = 0
        levelled_frames = removed_frames = 0
        start_frame = self.next_weight_transit(level, start=0, above=False)

        while start_frame >= 0:

            end_frame = self.next_weight_transit(
                level, start=start_frame, above=True)
            if end_frame == -1:
                end_frame = self.size

            if (end_frame - start_frame) > min_frames:
                # Remove the offsets, and update dependencies
                self.local_level(parms, start=start_frame, end=end_frame,
                                 robust=robust)

                # Set default frame weights (for now, might be overwritten if
                # re-weighting below).
                valid = self.frames.valid[start_frame:end_frame]
                weight = self.frames.temp_c[start_frame:end_frame]
                weight[valid] = 1.0
                self.frames.temp_c[start_frame:end_frame] = valid
                levelled_frames += end_frame - start_frame
                levelled += 1
            else:
                self.frames.set_flags(
                    self.flagspace.flags.FLAG_JUMP,
                    indices=np.arange(start_frame, end_frame))
                removed += 1
                removed_frames += end_frame - start_frame

            start_frame = self.next_weight_transit(
                level, start=end_frame, above=False)

        # Recalculate the frame weights as necessary.
        # TODO: Triple check - currently removed frames are marked with the
        #       FLAG_JUMP frame (MODELING_FLAG parent) but are STILL included
        #       in the frame weight calculation.  It looks like this is a bug
        #       but I can't be sure.  Channels having the TIME_WEIGHTING_FLAG
        #       are not included, but shouldn't this be a frame flag?
        #       For now, assuming this is a lucky error and disallowing flagged
        #       channels from the weight calculation.
        if (levelled > 0) or (removed > 0):
            if self.configuration.is_configured('weighting.frames'):
                self.get_time_weights()
        else:  # Otherwise, just reinstate the old weights.
            self.frames.relative_weight = self.frames.temp_c.copy()

        log.debug(f"Jumps: levelled {levelled} ({levelled_frames} frames), "
                  f"removed {removed} ({removed_frames} frames)")
        self.comments.append(f"{levelled}:{removed}")

    def next_weight_transit(self, level, start=0, above=True):
        """
        Find the next frame index with relative weight above or below a level.

        Parameters
        ----------
        level : float
            The weight threshold reference value.
        start : int, optional
            The starting frame.  The default is the first (0).
        above : bool, optional
            If `True`, return the first frame above level.  Otherwise, return
            the first frame below level.

        Returns
        -------
        frame_index : int
            The first frame index above or below the given level threshold.
            If no frame is found, returns -1.
        """
        weight_flag_value = self.flagspace.flags.TIME_WEIGHTING_FLAGS.value
        return int_nf.next_weight_transit(
            frame_weights=self.frames.relative_weight,
            level=level,
            frame_valid=self.frames.valid,
            frame_flags=self.frames.flag,
            time_weighting_flags=weight_flag_value,
            start_frame=start,
            above=above)

    def local_level(self, parms, start=0, end=None, robust=True):
        """
        Level frame data between in a certain frame range.

        Parameters
        ----------
        parms : Dependents
        start : int, optional
            The starting frame (inclusive).  The default is the
            first (0) frame.
        end : int, optional
            The end frame (exclusive).  The default is the last frame
            (self.size).
        robust : bool, optional
            Level using a robust mean determination (median).

        Returns
        -------
        None
        """
        # Clear dependencies of any prior leveling.  Will use new dependencies
        # on the currently obtained level for the interval.
        parms.clear(channels=self.channels, start=start, end=end)

        frame_dependents = np.zeros(self.size, dtype=float)
        self.level(channels=self.channels, start=start, end=end,
                   frame_dependents=frame_dependents, robust=robust)

        # Apply local-level dependencies
        parms.add_async(self.channels, 1.0)
        parms.add_for_frames(frame_dependents)
        parms.apply(self.channels, start=start, end=end)

        # Remove the drifts from all signals to match bandpass.
        if self.signals is not None:
            for signal in self.signals.values():
                signal.level(start_frame=start, end_frame=end, robust=False)

    def get_mean_level(self, channels=None, start=None, end=None,
                       robust=False):
        """
        Return the median data values and associated weights for channels.

        Parameters
        ----------
        channels : Channels or ChannelData, optional
            The channels for which to calculate median levels.
        start : int, optional
            Calculate the median after (and including) this starting frame.
        end : int, optional
            Calculate the median from before (not including) this end frame.
        robust : bool, optional
            If `True`, use :func:`smart_median` to determine the mean values.
            Otherwise, use a simple mean.

        Returns
        -------
        values, weights : numpy.ndarray (float), numpy.ndarray (float)
            The median channel values and associated weight of the calculation,
            both if shape (channels.size,).
        """
        if channels is None:
            channel_indices = np.arange(self.channels.size)
        else:
            channel_indices = getattr(
                channels, 'indices', np.arange(channels.size))

        return int_nf.get_mean_frame_level(
            frame_data=self.frames.data,
            frame_weights=self.frames.relative_weight,
            frame_valid=self.frames.valid,
            modeling_frames=self.frames.is_flagged('MODELING_FLAGS'),
            sample_flags=self.frames.sample_flag,
            channel_indices=channel_indices,
            start_frame=start,
            stop_frame=end,
            robust=robust)

    def level(self, channel_means=None, channel_weights=None, channels=None,
              start=0, end=None, frame_dependents=None, robust=True):
        """
        Remove the mean frame data from each channel and update dependents.

        Parameters
        ----------
        channel_means : numpy.ndarray (float), optional
            The mean for each channel.  Will be calculated if necessary.  An
            array of shape (n_channels,).
        channel_weights : numpy.ndarray (float), optional
            The weight derived for each channel during the mean operation of
            shape (n_channels,)
        channels : Channels, optional
            The channels on which to perform the levelling.
        start : int, optional
            The starting frame from which to derive means and consistencies.
        end : int, optional
            The ending frame from which to derive means and consistencies.
        frame_dependents : numpy.ndarray (float), optional
            The frame dependents to update of shape (n_frames,).  If not
            supplied, defaults to an array of zeros.
        robust : bool, optional
            If `True`, use the robust method (median) to determine the channel
            means.

        Returns
        -------
        consistent : numpy.ndarray (bool)
            An array of shape (n_frames,) where `True` indicates a consistent
            frame.
        """
        if channels is None:
            channels = self.channels
        if end is None:
            end = self.size
        frame_indices = slice(start, end)
        if frame_dependents is None:
            frame_dependents = np.zeros(self.size, dtype=float)

        if channel_means is None or channel_weights is None:
            channel_means, channel_weights = self.get_mean_level(
                channels=channels, start=start, end=end, robust=robust)

        # Remove offsets from data
        data = self.frames.data[frame_indices]
        valid = self.frames.valid[frame_indices]
        data[valid] -= channel_means[None]
        self.frames.data[frame_indices] = data

        # Account for frame dependence.
        p_norm = np.zeros(channels.size, dtype=float)
        nzi = channel_weights > 0
        p_norm[nzi] = channels.get_filtering(self)[nzi] / channel_weights[nzi]

        flags = self.frames.sample_flag[frame_indices]
        weight = self.frames.relative_weight[frame_indices] * valid
        weight = weight[:, None] * (flags == 0)

        frame_dependents[frame_indices] += np.sum(weight * p_norm[None],
                                                  axis=1)
        return self.check_consistency(
            channels.create_channel_group(), frame_dependents,
            start_frame=start, stop_frame=end)

    def check_consistency(self, channels, frame_dependents,
                          start_frame=None, stop_frame=None):
        """
        Check consistency of frame dependents and channels.

        This is intended to be overridden for specific instruments
        as needed.  The generic implementation just returns True.

        Parameters
        ----------
        frame_dependents : numpy.ndarray (float)
        channels : ChannelGroup
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
        return np.full(channels.size, True)

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
        pass

    def get_time_stream(self, channels=None, weighted=False):
        """
        Return the time stream data.

        Returns the frame time stream data for a given set of channels (default
        is the integration channels).   Samples flagged as MODELING flags, have
        nonzero sample flags, or marked as invalid are returned as NaN if
        unweighted or zero otherwise.

        Parameters
        ----------
        channels : Channels or ChannelData
            The channels for which to extract time streams.  If not supplied,
            defaults to the integration channels.
        weighted : bool, optional
            If `True` return the frame data multiplied by relative weight.
            In addition, return the weights.

        Returns
        -------
        data, [weight] : numpy.ndarray (float), [numpy.ndarray (float)]
            An array of shape (n_frames, n_channels).  Invalid values are
            returned as NaN values if unweighted, or zero otherwise.
        """
        if channels is None:
            channels = self.channels
        indices = getattr(channels, 'indices', slice(0, channels.size))

        data = self.frames.data.copy()
        valid_samples = self.frames.sample_flag == 0
        valid_frames = self.frames.is_unflagged(
            self.flagspace.flags.MODELING_FLAGS) & self.frames.valid
        valid_samples &= valid_frames[:, None]

        fill_value = 0.0 if weighted else np.nan
        data[~valid_samples] = fill_value
        if not weighted:
            return data[:, indices]

        weight = self.frames.relative_weight[:, None] * valid_samples
        data *= self.frames.relative_weight[:, None]
        return data[:, indices], weight[:, indices]

    def despike(self, level=None):
        """
        Despike frame data.

        Parameters
        ----------
        level : float, optional
            The despiking level.  If not supplied, will be extracted from the
            configuration 'despike.level' setting.

        Returns
        -------
        None
        """
        method = self.configuration.get(
            'despike.method', default='absolute').strip().lower()

        if level is None:
            level = self.configuration.get_float('despike.level', default=10.0)

        flag_fraction = self.configuration.get_float(
            'despike.flagfraction', default=1.0)
        flag_count = self.configuration.get_int(
            'despike.flagcount', default=np.inf)
        frame_spikes = self.configuration.get_int(
            'despike.framespikes', default=self.channels.size)

        if method in ['neighbours', 'neighbors']:
            delta = self.frames_for(0.2 * self.get_point_crossing_time())
            self.despike_neighbouring(level, delta)

        elif method == 'absolute':
            self.despike_absolute(level)

        elif method == 'gradual':
            self.despike_gradual(level, depth=0.1)

        elif method in ['multires', 'features']:
            self.despike_multi_resolution(level)

        if method != 'features':
            self.flag_spiky_frames(frame_spikes=frame_spikes)
            self.flag_spiky_channels(flag_fraction=flag_fraction,
                                     flag_count=flag_count)
        else:
            feature_width = self.frames_for(self.filter_time_scale) // 2
            feature_fraction = 1.0 - np.exp(-feature_width * flag_fraction)
            feature_count = feature_width * flag_count
            self.flag_spiky_channels(flag_fraction=feature_fraction,
                                     flag_count=feature_count)

        if self.configuration.get_bool('despike.blocks'):
            self.flag_spiky_blocks()

    @staticmethod
    def set_temp_despike_levels(channel_data, level):
        """
        Set a temporary despiking level reference in the channel data.

        The despiking reference level is set to sqrt(variance) * level, and
        stored in the `temp` attribute of the channel data.

        Parameters
        ----------
        channel_data : ChannelData
            The channel data for which to set the despike reference level.
        level : float
            The relative reference level.

        Returns
        -------
        None
        """
        channel_data.temp = level * np.sqrt(channel_data.variance)

    def despike_neighbouring(self, level, delta):
        r"""
        Identifies and flags spikes in frame data using neighbours method.

        Identify spikes in the data using the relation::

           |x[i + delta] - x[i]| > threshold

        where::

           threshold = level * channel_noise * sqrt(1/w[i] + 1/w[i + delta])

        level and delta are supplied by the user, x is the frame data, and
        w is the relative weight of a given frame.

        Samples that obey the above inequality are flagged with the
        SAMPLE_SPIKE flag while all others will be unflagged.  Note that this
        is only applicable to the "live channels", and those frames that
        are not flagged with the SAMPLE_SKIP or SAMPLE_SOURCE_BLANK frame
        flags. Invalid frames are also ignored.

        Parameters
        ----------
        level : float
            The sigma level used to identify spikes.
        delta : int
            The frame difference between two "neighbours" used for the
            spike comparison.

        Returns
        -------
        None
        """
        self.comments.append('dN')
        log.debug(f"Despiking using neighbouring method level={level}, "
                  f"delta={delta}.")

        if self.size < delta:
            log.warning(f'delta ({delta}) too large. '
                        f'Number of frames = {self.size}')
            return

        live_channels = self.channels.get_live_channels()
        self.set_temp_despike_levels(live_channels, level)

        n_flagged = int_nf.despike_neighbouring(
            frame_data=self.frames.data,
            sample_flags=self.frames.sample_flag,
            channel_indices=live_channels.indices,
            frame_weight=self.frames.relative_weight,
            frame_valid=self.frames.valid,
            channel_level=live_channels.temp,
            delta=delta,
            spike_flag=self.flagspace.convert_flag('SAMPLE_SPIKE').value,
            exclude_flag=self.flagspace.convert_flag(
                'SAMPLE_SKIP | SAMPLE_SOURCE_BLANK').value
        )
        percent_flagged = 100 * n_flagged / (live_channels.size * self.size)
        log.debug(f"{n_flagged} live channel frames ({percent_flagged:.2f}%) "
                  f"flagged as spikes.")

    def despike_absolute(self, level):
        r"""
        Identifies and flags spikes in frame data using absolute method.

        Identify spikes in the data using the relation::

           |x| > threshold

        where::

           threshold = level * channel_noise / sqrt(w)

        level is supplied by the user, x is the frame data, and w is the
        relative weight of a given frame.

        Samples that obey the above inequality are flagged with the
        SAMPLE_SPIKE flag while all others will be unflagged.  Note that
        this is only applicable to the "live channels", and those frames
        that are not flagged with the SAMPLE_SKIP or SAMPLE_SOURCE_BLANK
        frame flags. Invalid frames are also ignored.

        Parameters
        ----------
        level : float
            The sigma level used to identify spikes.

        Returns
        -------
        None
        """
        self.comments.append('dA')
        log.debug(f"Despiking using absolute method level={level}.")
        live_channels = self.channels.get_live_channels()
        self.set_temp_despike_levels(live_channels, level)

        n_flagged = int_nf.despike_absolute(
            frame_data=self.frames.data,
            sample_flags=self.frames.sample_flag,
            channel_indices=live_channels.indices,
            frame_weight=self.frames.relative_weight,
            frame_valid=self.frames.valid,
            channel_level=live_channels.temp,
            spike_flag=self.flagspace.flags.SAMPLE_SPIKE.value,
            exclude_flag=self.flagspace.convert_flag(
                'SAMPLE_SKIP | SAMPLE_SOURCE_BLANK').value
        )
        percent_flagged = 100 * n_flagged / (live_channels.size * self.size)
        log.debug(f"{n_flagged} live channel frames ({percent_flagged:.2f}%) "
                  f"flagged as spikes.")

    def despike_gradual(self, level, depth=0.1):
        r"""
        Identifies and flags spikes in frame data using gradual method.

        Identify spikes in the data using the relation::

           |x| > threshold

        where::

           threshold = max(level * channel_noise / sqrt(w),
                           gain * depth * max(x/gain <for all channels>)).

        level and depth are supplied by the user, x is the frame data, and
        w is the relative weight of a given frame.

        Samples that obey the above inequality are flagged with the
        SAMPLE_SPIKE flag while all others will be unflagged.  Note that
        this is only applicable to the "live channels", and those frames
        that are not flagged with the SAMPLE_SKIP or SAMPLE_SOURCE_BLANK
        frame flags. Invalid frames are also ignored.

        Parameters
        ----------
        level : float
            The sigma level used to identify spikes.
        depth : float, optional
            The maximum allowable data value as a factor of the maximum channel
            data value.

        Returns
        -------
        None
        """
        self.comments.append('dG')
        log.debug(f"Despiking using gradual method level={level}.")

        valid_frames = self.frames.is_unflagged(
            self.flagspace.flags.MODELING_FLAGS) & self.frames.valid

        live_channels = self.channels.get_live_channels()
        self.set_temp_despike_levels(live_channels, level)

        blank_flag = self.flagspace.convert_flag('SAMPLE_SOURCE_BLANK')
        exclude_flag = self.flagspace.convert_flag('SAMPLE_SKIP') | blank_flag

        n_flagged = int_nf.despike_gradual(
            frame_data=self.frames.data,
            sample_flags=self.frames.sample_flag,
            channel_indices=live_channels.indices,
            frame_weight=self.frames.relative_weight,
            frame_valid=valid_frames,
            channel_level=live_channels.temp,
            spike_flag=self.flagspace.convert_flag('SAMPLE_SPIKE').value,
            source_blank_flag=blank_flag.value,
            exclude_flag=exclude_flag.value,
            channel_gain=live_channels.gain,
            depth=depth
        )

        percent_flagged = 100 * n_flagged / (live_channels.size * self.size)
        log.debug(f"{n_flagged} live channel frames ({percent_flagged:.2f}%) "
                  f"flagged as spikes.")

    def despike_multi_resolution(self, level):
        r"""
        Identifies and flag spikes in frame data using multi-resolution.

        Flags are identified in the frame data at a series of decreasing
        resolutions (power of 2).  The first resolution (1) determines the
        significance as::

           s = |d[i] - d[i-1]| * sqrt(w_sum)

        where::

           w_sum = (w[i] * w[i-1]) / (w[i] + w[i-1])

        and flagged frame i as a spike if s > level where d is the
        timestream data (frame_data * frame_relative_weight) and w is the
        frame_relative_weight.

        Timestream data is then set to half the resolution by setting

        d[i] -> d[i] + d[i-1]
        w[i] = w_sum

        The process is repeated at decreasing resolution up to a maximum block
        size defined as half the number of frames in the filter time scale.

        Parameters
        ----------
        level : float
            The significance above which data are flagged as spikes.

        Returns
        -------
        None
        """
        self.comments.append('dM')
        log.debug(f"Despiking using multi-resolution method level={level}.")
        max_block_size = self.frames_for(self.filter_time_scale) // 2
        max_block_size = int(np.clip(max_block_size, 1, self.size // 2))
        live_channels = self.channels.get_live_channels()
        valid_frames = self.frames.valid & self.frames.is_unflagged(
            'MODELING_FLAGS')

        timestream_data, timestream_weight = int_nf.get_weighted_timestream(
            self.frames.data, self.frames.sample_flag,
            valid_frames, self.frames.relative_weight,
            channel_indices=live_channels.indices)

        n_flagged = int_nf.despike_multi_resolution(
            timestream_data=timestream_data,
            timestream_weight=timestream_weight,
            sample_flags=self.frames.sample_flag,
            channel_indices=live_channels.indices,
            frame_valid=self.frames.valid,
            level=level,
            spike_flag=self.flagspace.flags.SAMPLE_SPIKE.value,
            max_block_size=max_block_size)

        percent_flagged = 100 * n_flagged / (live_channels.size * self.size)
        log.debug(f"{n_flagged} live channel frames ({percent_flagged:.2f}%) "
                  f"flagged as spikes.")

    def flag_spiky_frames(self, frame_spikes=None):
        """
        Flag frames with excessive spiky channels as FLAG_SPIKY.

        Parameters
        ----------
        frame_spikes : int, optional
            Frames with a number of spiky channels above this limit will be
            flagged.  The default is all channels.

        Returns
        -------
        None
        """
        if frame_spikes is None:
            frame_spikes = self.channels.size

        channel_indices = self.channels.data.is_unflagged(indices=True)
        flagged_channels = int_nf.flagged_channels_per_frame(
            sample_flags=self.frames.sample_flag,
            flag=self.flagspace.convert_flag('SAMPLE_SPIKE').value,
            valid_frames=self.frames.valid,
            channel_indices=channel_indices)

        spiky_frames = flagged_channels > frame_spikes
        self.frames.set_flags('FLAG_SPIKY', indices=spiky_frames)
        self.frames.unflag('FLAG_SPIKY', indices=~spiky_frames)
        flagged = int(np.sum(spiky_frames))
        percent_flagged = 100 * flagged / self.frames.valid.sum()
        log.debug(f"{flagged} valid frames flagged as spiky "
                  f"({percent_flagged:.2f}%).")

    def flag_spiky_channels(self, flag_fraction=1.0, flag_count=np.inf):
        """
        Flag channels with excessive spiky frames as SPIKY.

        The number of frames required to flag a channel as spiky is given as
        max(`flag_fraction` * n_frames, flag_count).  This procedure also
        set the channel data spikes attribute and performs a census of all
        channels.

        Parameters
        ----------
        flag_fraction : float, optional
            The minimum fraction of frames flagged as spiky required to flag
            a channel as spiky.
        flag_count : int or float, optional
            The number of frames flagged as spiky required to flag a channel
            as spiky.

        Returns
        -------
        None
        """
        max_spikes = max(flag_fraction * self.size, flag_count)
        if np.isfinite(max_spikes):
            max_spikes = round(max_spikes)
        else:
            max_spikes = np.inf

        # Only flag spiky channels if spikes are not in spiky frames.
        valid_frames = self.frames.is_unflagged(
            'MODELING_FLAGS') * self.frames.valid
        channel_indices = np.arange(self.frames.sample_flag.shape[1])
        channel_spikes = int_nf.flagged_frames_per_channel(
            sample_flags=self.frames.sample_flag,
            flag=self.flagspace.flags.SAMPLE_SPIKE.value,
            valid_frames=valid_frames,
            channel_indices=channel_indices
        )

        spiky_channels = channel_spikes > max_spikes
        self.channels.data.set_flags('SPIKY', indices=spiky_channels)
        self.channels.data.spikes = channel_spikes
        flagged = int(np.sum(spiky_channels))
        percent_flagged = 100 * flagged / self.frames.valid.sum()
        log.debug(f"{flagged} channels flagged as spiky "
                  f"({percent_flagged:.2f}%).")
        self.channels.census()

    def flag_spiky_blocks(self):
        """
        Flag all frames in a block if one or more frames is flagged as spiky.

        The number of frames in a "block" is defined as the number of frames
        within the filter time scale.

        Returns
        -------
        None
        """
        int_nf.frame_block_expand_flag(
            sample_flags=self.frames.sample_flag,
            valid_frames=self.frames.valid,
            flag=self.flagspace.convert_flag('SAMPLE_SPIKE').value,
            block_size=self.frames_for(self.filter_time_scale),
            channel_indices=np.arange(self.channels.size))

    def detector_stage(self):
        """
        Divide the frame data by the channel hardware gains.

        Will not be performed if this operation has already been completed.
        i.e. the detector is already staged, and the readout is not staged.

        Returns
        -------
        None
        """
        if self.is_detector_stage:
            return
        log.debug("Staging detector: removing hardware gains.")
        self.channels.load_temporary_hardware_gains()
        int_nf.detector_stage(
            frame_data=self.frames.data,
            frame_valid=self.frames.valid,
            channel_indices=np.arange(self.channels.size),
            channel_hardware_gain=self.channels.data.hardware_gain)
        self.is_detector_stage = True

    def readout_stage(self):
        """
        Multiply the frame data by the channel hardware gains.

        Will not be performed the detector is not staged (i.e. the readout
        is staged, and the detector is not staged)

        Returns
        -------
        None
        """
        if not self.is_detector_stage:
            return
        log.debug("Unstaging detector: applying hardware gains.")
        self.channels.load_temporary_hardware_gains()
        int_nf.readout_stage(
            frame_data=self.frames.data,
            frame_valid=self.frames.valid,
            channel_indices=np.arange(self.channels.size),
            channel_hardware_gain=self.channels.data.hardware_gain)
        self.is_detector_stage = False

    def add_signal(self, signal):
        """
        Add a signal to the integration signals.

        The integration signals is a dictionary of the form {mode: signal).

        Parameters
        ----------
        signal : Signal

        Returns
        -------
        None
        """
        if self.signals is None:
            self.signals = {signal.mode: signal}
        else:
            self.signals[signal.mode] = signal

    def get_signal(self, mode):
        """
        Return the signal for a given mode.

        If the signal exists in the current integration signals dictionary, it
        will be returned as is.  Otherwise, if a response mode is supplied,
        a new signal will be created, added to the dictionary, and returned.
        If the signal does not exist, and the mode is not a response mode, no
        signal will be returned.

        Parameters
        ----------
        mode : Mode

        Returns
        -------
        signal : Signal or None

        """
        if self.signals is None:
            signal = None
        else:
            signal = self.signals.get(mode, None)

        if signal is None and isinstance(mode, Response):
            signal = mode.get_signal(self)
            if signal.is_floating:
                signal.level(robust=False)
            signal.remove_drifts()

        return signal

    def get_coupling_gains(self, signal):
        """
        Return the coupling gains for a given channel.

        Parameters
        ----------
        signal : Signal
            The signal for which to extract channel gains.

        Returns
        -------
        coupling_gains : numpy.ndarray (float)
            The coupling gains as an array of shape (self.channels.size).
            Values for channels not included in the signal will be set to zero.
        """

        gain = np.zeros(self.channels.size, dtype=float)
        gain[signal.mode.channel_group.indices] = signal.mode.get_gains()
        return gain

    def shift_frames(self, time_or_frames):
        """
        Shift the frame data.

        Parameters
        ----------
        time_or_frames : astropy.units.Quantity or int
            The time or number of frames by which to shift frame data.

        Returns
        -------
        None
        """
        if not np.isfinite(time_or_frames):
            return
        if isinstance(time_or_frames, units.Quantity):
            frame_shift = (time_or_frames
                           / self.info.sampling_interval).decompose().value
            frame_shift = int(np.round(frame_shift))
        else:
            frame_shift = int(time_or_frames)

        log.debug(f"Shifting data by {frame_shift} frames.")
        self.frames.shift_frames(frame_shift)

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
        if name == 'scale':
            return self.gain
        if name == 'NEFD':
            return self.nefd
        if name == 'zenithtau':
            return self.zenith_tau
        if name.startswith('tau.'):
            return self.get_tau(name[4:].lower())
        if name == 'scanspeed':
            return self.average_scan_speed[0].to('arcsec/second').value
        if name == 'rmsspeed':
            return (np.sqrt(1.0 / self.average_scan_speed[1])).to(
                'arcsec/second').value
        if name == 'hipass':
            return self.filter_time_scale.to('second').value
        if name.startswith('chop'):
            chopper = getattr(self, 'chopper', None)
            if chopper is None:
                return None
            else:
                return chopper.get_chop_table_entry(name)
        return self.channels.get_table_entry(name)

    def setup_filters(self):
        """
        Set up the FFT frame data filters.

        Returns
        -------
        None
        """
        log.debug("Configuring filters.")
        filter_ordering = self.configuration.get_list('filter.ordering')
        self.filter = MultiFilter(integration=self)
        for filter_name in filter_ordering:
            sub_filter = self.get_filter(filter_name)
            if sub_filter is None:
                log.warning(f"No filter for {filter_name}.")
            else:
                self.filter.add_filter(sub_filter)

    def get_filter(self, filter_name):
        """
        Return a filter of the given name.

        Parameters
        ----------
        filter_name : str
            The name of the filter.

        Returns
        -------
        Filter
        """
        if filter_name == 'motion':
            return MotionFilter(integration=self)
        elif filter_name == 'kill':
            return KillFilter(integration=self)
        elif filter_name == 'whiten':
            return WhiteningFilter(integration=self)
        else:
            return None

    def remove_dc_offsets(self):
        """
        Remove the DC offsets, either if explicitly requested or to allow
        bootstrapping pixel weights when pixeldata is not defined.  This
        must be done before direct tau estimates

        Returns
        -------
        None
        """
        if not (self.configuration.is_configured('level')
                or not self.configuration.is_configured('pixeldata')):
            return

        robust = self.configuration.get_string('estimator').lower() == 'median'
        log.debug(f"Removing DC offsets{' (robust)' if robust else ''}.")
        self.remove_offsets(robust=robust)

    def remove_offsets(self, robust=None):
        """
        Remove the DC offsets from the frame data.

        Parameters
        ----------
        robust : bool
            If `True`, use the robust (median) method to determine means.

        Returns
        -------
        None
        """
        self.remove_drifts(target_frame_resolution=self.size, robust=robust)

    def remove_drifts(self, target_frame_resolution=None, robust=None):
        """
        Remove drifts in frame data given a target frame resolution.

        Will also set the filter time scale based on the target frame
        resolution.

        Parameters
        ----------
        target_frame_resolution : int, optional
            The number of frames for the target resolution.  If not supplied,
            extracted from the configuration 'drifts' option in seconds and
            then converted to frames.
        robust : bool, optional
            If `True` use the robust (median) method to determine means.  If
            not supplied determined from the 'estimator' configuration option
            (robust=median).

        Returns
        -------
        None
        """
        if target_frame_resolution is None:
            spec = self.configuration.get_string(
                'drifts', default=10 * units.Unit('second'))
            target_frame_resolution = self.filter_frames_for(spec)

        if robust is None:
            robust = self.configuration.get_string('estimator') == 'median'

        drift_n = min(self.size, utils.pow2ceil(target_frame_resolution))
        parms = self.get_dependents('drifts')
        dt = self.info.instrument.sampling_interval
        self.filter_time_scale = min(self.filter_time_scale, drift_n * dt)
        log.debug(f"Removing channel drifts{' (robust):' if robust else ':'} "
                  f"resolution = {drift_n}")

        if drift_n < self.size:
            msg = '[D]' if robust else 'D'
            msg += f'({drift_n})'
        else:
            msg = '[O]' if robust else 'O'
        self.comments.append(msg)
        self.comments.append(' ')

        # Remove the 1/f drifts from all channels
        self.remove_channel_drifts(self.channels.create_channel_group(),
                                   parms, drift_n, robust=robust)

        # Remove the drifts from all signals also to match bandpass
        if self.signals is not None:
            for signal in self.signals.values():
                signal.remove_drifts(n_frames=drift_n, is_reconstructable=True)

        return True

    def remove_channel_drifts(
            self, channel_group, parms, drift_n, robust=False):
        """
        Remove drifts from channels.

        Removes the average values from frames in blocks of `drift_n` frames.
        Dependents are updated as necessary.  If there is more than one
        block, then channel filtering time scales and source filtering are
        updated to account for the new length of the drift.

        Parameters
        ----------
        channel_group : ChannelGroup
            The channels from which to remove drifts.
        parms : Dependents
            The dependents to update.
        drift_n : int
            The number of frames in a "drift" (resolution).
        robust : bool, optional
            If `True`, use the robust (median) method to calculate means.
            Otherwise, use a simple mean.

        Returns
        -------
        None
        """
        parms.clear(channel_group, start=0, end=self.size)
        modeling_frames = self.frames.is_flagged('MODELING_FLAGS')

        average_drifts, average_drift_weights = int_nf.remove_channel_drifts(
            frame_data=self.frames.data,
            frame_weights=self.frames.relative_weight,
            frame_valid=self.frames.valid,
            modeling_frames=modeling_frames,
            sample_flags=self.frames.sample_flag,
            drift_frame_size=drift_n,
            channel_filtering=channel_group.get_filtering(self),
            frame_dependents=parms.for_frame,
            channel_dependents=parms.for_channel,
            channel_indices=channel_group.indices,
            robust=robust)

        self.update_inconsistencies(channel_group, parms.for_frame, drift_n)

        # Base data allows us to update the data in-place
        time_unit = channel_group.filter_time_scale.unit
        integration_time_scale = self.filter_time_scale.to(time_unit).value
        crossing_time = self.get_point_crossing_time().to(time_unit).value

        base_data = channel_group.data
        inconsistent_channels, tot_inc = int_nf.apply_drifts_to_channel_data(
            channel_indices=channel_group.indices,
            offsets=base_data.offset,
            average_drifts=average_drifts,
            inconsistencies=base_data.inconsistencies,
            hardware_gain=base_data.hardware_gain,
            filter_time_scale=base_data.filter_time_scale.value,
            source_filtering=base_data.source_filtering,
            integration_filter_time_scale=integration_time_scale,
            crossing_time=crossing_time,
            is_detector_stage=self.is_detector_stage,
            update_filtering=(drift_n < self.size))

        log.debug(f"Total drift inconsistencies = {tot_inc} in "
                  f"{inconsistent_channels} channels.")
        if tot_inc > 0:
            self.comments.append(f"!{inconsistent_channels}:{tot_inc}")

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
        if spec is None and value is not None:  # directly set tau value
            self.set_tau_value(value)
            return

        elif spec is not None and value is None:
            try:
                value = float(spec)
                self.set_tau_value(value)
                return
            except (ValueError, TypeError):
                value = self.configuration.get_float(
                    f'tau.{spec.lower()}', default=np.nan)
            if np.isnan(value):
                raise ValueError(f"Configuration does not contain a tau value "
                                 f"for tau specification {spec}.")
        elif spec is None and value is None:
            spec = self.configuration.get_string('tau')
            if spec is None:
                raise ValueError("Configuration does not contain a tau "
                                 "specification.")
            self.set_tau(spec=spec)
            return

        # At this point we have a specification and value
        t_a, t_b = self.get_tau_coefficients(spec)
        band_a, band_b = self.get_tau_coefficients(
            self.info.instrument.name)
        tau = ((band_a / t_a) * (value - t_b)) + band_b
        self.set_zenith_tau(tau)

    def set_tau_value(self, value):
        """
        Set the zenith tau if ground based, or the transmission.

        Parameters
        ----------
        value : float
            The tau value.

        Returns
        -------
        None
        """
        if self.info.astrometry.ground_based:
            self.set_zenith_tau(value)
        else:
            self.frames.set_transmission(np.exp(-value))

    def set_zenith_tau(self, tau_value):
        """
        Set the zenith tau value.

        Parameters
        ----------
        tau_value : float

        Returns
        -------
        None
        """
        if not self.info.astrometry.ground_based:
            raise NotImplementedError(
                "Only ground based observations can set a zenith tau.")

        log.info(f"Setting zenith tau to {tau_value:.3f}")
        self.zenith_tau = float(tau_value)
        self.frames.set_zenith_tau(self.zenith_tau)

    def get_tau(self, spec, value=None):
        """
        Get the tau value based on configuration settings.

        The returned value is given as:

        t_b + t_a * (value - band_b) / band_a

        where band_a/b are retrieved from the configuration as tau.<spec>.a/b,
        and t_a/b are retrieved from the configuration as tau.<instrument>.a/b.

        Parameters
        ----------
        spec : str
            The name of the tau specification.
        value : float, optional
            If not provided, defaults to the zenith tau.

        Returns
        -------
        float
        """
        if value is None:
            value = self.zenith_tau
        t_a, t_b = self.get_tau_coefficients(spec)
        band_a, band_b = self.get_tau_coefficients(
            self.info.instrument.name)
        return (t_a * (value - band_b) / band_a) + t_b

    def get_tau_coefficients(self, spec):
        """
        Return the tau coefficients for the tau specification.

        The tau a and b coefficients are retrieved from the configuration as
        tau.<spec>.a and tau.<spec>.b.

        Parameters
        ----------
        spec : str

        Returns
        -------
        t_a, t_b : float, float
           The tau a and b coefficients.
        """
        key = f"tau.{str(spec).lower().strip()}"
        a = self.configuration.get_float(f'{key}.a', default=np.nan)
        if np.isnan(a):
            raise ValueError(f"Tau {key} has no scaling relation.")
        b = self.configuration.get_float(f'{key}.b', default=np.nan)
        if np.isnan(b):
            raise ValueError(f"Tau {key} has no offset relation.")
        return a, b

    def set_scaling(self, factor=None):
        """
        Set the gain scaling for the integration.

        The factor is applied relatively.

        Parameters
        ----------
        factor : float, optional
            The factor by which to scale the integration gain.  Retrieved from
            the configuration 'scale' option if not supplied.

        Returns
        -------
        None
        """
        if factor is None:
            factor = self.get_default_scaling_factor()

        factor = float(factor)
        if np.isnan(factor) or factor == 1:
            return
        log.debug(f"Applying scaling factor {factor:.3f}")
        self.gain /= factor

    def get_default_scaling_factor(self):
        """
        Return the default scaling factor from the configuration.

        Returns
        -------
        scale : float
        """
        scale = self.configuration.get_float('scale', default=np.nan)
        if np.isnan(scale):
            return 1.0

        grid_scaling = self.configuration.get_float('scale.grid',
                                                    default=np.nan)
        if np.isnan(grid_scaling):
            return scale

        grid = self.configuration.get_float('grid', default=np.nan)
        if np.isnan(grid) or grid == grid_scaling:
            return scale

        return scale * ((grid / grid_scaling) ** 2)

    def slim(self):
        """
        Slim frame/channel type data to those channels still present.

        Returns
        -------
        slimmed : bool
            `True` if any channels were removed, `False` otherwise.
        """
        old_fixed_indices = self.channels.data.fixed_index.copy()
        slimmed = self.channels.slim(reindex=False)
        if not slimmed:
            return slimmed

        self.frames.slim(channels=self.channels)
        self.reindex_channels()

        if self.dependents is not None:
            new_fixed_indices = self.channels.data.fixed_index
            mapping = np.nonzero(
                new_fixed_indices[:, None] == old_fixed_indices)[1]
            for dependent in self.dependents.values():
                dependent.for_channel = dependent.for_channel[mapping]

        return slimmed

    def reindex_frames(self):
        """
        Reindex the time dependent data.

        Returns
        -------
        None
        """
        self.frames.fixed_index = np.arange(self.size)

    def reindex_channels(self):
        """
        Reindex the channels such that all data are self-consistent.

        Returns
        -------
        None
        """
        if self.channels is not None:
            self.channels.reindex()

        if self.signals is not None:
            for signal in self.signals.values():
                if signal.mode is not None:
                    if signal.mode.channel_group is not None:
                        signal.mode.channel_group.reindex()
                        signal.mode.channel_group.data = self.channels.data

        if self.filter is not None:
            self.filter.reindex()

    def reindex(self):
        """
        Reset the frame fixed indices to range sequentially from 0 -> size.

        Channels are also reset to ensure everything is self-consistent.

        Returns
        -------
        None
        """
        self.reindex_frames()
        self.reindex_channels()

    def jackknife(self):
        """
        Randomly invert integration and channel gains.

        The integration gain will be inverted if the configuration contains
        a 'jackknife' option.  The frame signs will be inverted if the
        configuration contains a 'jackknife.frames' options.

        Returns
        -------
        None
        """
        if self.configuration.get_bool('jackknife'):
            if np.random.random(1)[0] < 0.5:
                log.debug("JACKKNIFE: This integration will produce an "
                          "inverted source.")
                self.gain *= -1

        if self.configuration.get_bool('jackknife.frames'):
            log.debug("JACKKNIFE: Randomly inverted frames in source.")
            self.frames.jackknife()

    def bootstrap_weights(self):
        """
        Calculates channel weights from frame data if required.

        If weighting is required but were not extracted from the `pixeldata`
        configuration option, or set to 'uniform', will derive values using
        the `get_differential_channel_weights` method.

        Returns
        -------
        None
        """
        if not self.configuration.get_bool('weighting'):
            return
        if self.configuration.is_configured('pixeldata'):
            return
        if self.configuration.get_bool('uniform'):
            return

        self.get_differential_channel_weights()
        self.channels.census()
        log.debug(f"Bootstrapping pixel weights "
                  f"({self.scan.channels.n_mapping_channels} "
                  f"active channels).")

    def decorrelate(self, modality_name, robust=False):
        """
        Decorrelate a modality.

        Parameters
        ----------
        modality_name : str
            The name of the modality to decorrelate.
        robust : bool, optional
            If `True`, use the robust (median) method to derive means.
            Otherwise, use a standard mean.

        Returns
        -------
        decorrelated : bool
            `True` if the modality was decorrelated and gains were updated,
            `False` otherwise.
        """
        if not self.decorrelate_signals(modality_name, robust=robust):
            return False
        return self.update_gains(modality_name, robust=robust)

    def decorrelate_signals(self, modality_name, robust=False):
        """
        Decorrelate a modality signal.

        Parameters
        ----------
        modality_name : str
            The name of the modality to decorrelate.
        robust : bool, optional
            If `True`, use the robust (median) method to derive means.
            Otherwise, use a standard mean.

        Returns
        -------
        decorrelated : bool
            `True` if the modality was decorrelated.  `False` otherwise.
        """
        if self.channels.modalities is None:
            return False
        modality = self.channels.modalities.get(modality_name)
        if modality is None:
            return False
        modality.set_options(
            self.configuration, branch=f'correlated.{modality.name}')

        # Check for trigger
        if modality.trigger is not None:
            if not self.configuration.check_trigger(modality.trigger):
                return False

        if robust:
            self.comments.append(f'[{modality.id}]')
            robust_str = ' (robust)'
        else:
            self.comments.append(modality.id)
            robust_str = ''

        log.debug(f"Decorrelating {modality.name} modality{robust_str}.")
        frame_resolution = self.power2_frames_for(modality.resolution)
        if frame_resolution > 1:
            self.comments.append(f'({frame_resolution})')
            log.debug(f"Modality frame resolution = {frame_resolution}")

        if isinstance(modality, CorrelatedModality) and modality.solve_signal:
            modality.update_signals(integration=self, robust=robust)

        return True

    def update_gains(self, modality_name, robust=False):
        """
        Update gains in a modality following decorrelation.

        Parameters
        ----------
        modality_name : str
            The name of the modality to update.
        robust : bool, optional
            If `True`, use the robust (median) method to derive means.
            Otherwise, use a standard mean.

        Returns
        -------
        updated : bool
            `True` if the modality gains were updated.  `False` otherwise.
        """
        if self.channels.modalities is None:
            return False
        modality = self.channels.modalities.get(modality_name)
        if modality is None:
            return False
        modality.set_options(
            self.configuration, branch=f'correlated.{modality.name}')
        solve_gains = modality.solve_gains and self.configuration.get_bool(
            'gains')
        if not solve_gains:
            return True

        # Check for trigger
        if modality.trigger is not None:
            if not self.configuration.check_trigger(modality.trigger):
                return False

        if robust is None:
            robust = self.configuration.get_string(
                'gains.estimator') == 'median'

        if modality.update_all_gains(integration=self, robust=robust):
            self.channels.census(report=True)
            self.comments.append(f'{self.channels.n_mapping_channels}')

        return True

    def merge(self, integration):
        """
        Merge frames from another integration onto this integration.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        self.frames.merge(integration.frames)

    def pointing_at(self, offset, indices=None):
        """
        Applies pointing correction to coordinates via subtraction.

        Parameters
        ----------
        offset : astropy.units.Quantity (numpy.ndarray)
            An array of
        indices : numpy.ndarray (int), optional
            The frame indices that apply.  The default is all indices.

        Returns
        -------
        None
        """
        self.frames.pointing_at(offset, indices=indices)

    def next_iteration(self):
        """
        Perform certain steps prior to an iteration.

        Returns
        -------
        None
        """
        self.comments = []

    def set_thread_count(self, threads):
        """
        Set the number of parallel tasks for the integration.

        Parameters
        ----------
        threads : int
            The number of parallel tasks to perform.

        Returns
        -------
        None
        """
        self.parallelism = threads
        self.info.parallelism = threads

    def get_thread_count(self):
        """
        Return the number of parallel tasks for the integration.

        Returns
        -------
        threads : int
        """
        return self.parallelism

    def perform(self, task):
        """
        Perform a reduction task on the integration.

        Parameters
        ----------
        task : str
            The name of the task to perform.

        Returns
        -------
        performed : bool
            Indicates whether the task was performed.
        """
        if self.comments is None:
            self.comments = []
        robust = self.configuration.get_string('estimator') == 'median'
        if task == 'dejump':
            self.dejump_frames()
        elif task == 'offsets':
            self.remove_offsets(robust=robust)
        elif task == 'drifts':
            ok = self.remove_drifts(robust=robust)
            if not ok:
                return False  # If phase modulated - cannot perform
        elif task.startswith('correlated.'):
            modality_name = task.split('.')[1]
            if not self.decorrelate(modality_name, robust=robust):
                return False
        elif task == 'weighting':
            self.get_channel_weights()
        elif task == 'weighting.frames':
            self.get_time_weights()
        elif task == 'despike':
            self.despike()
        elif task == 'filter':
            if self.filter is None:
                return False
            if not self.filter.apply():
                return False
        else:
            return False
        self.comments.append(' ')
        return True

    def get_fits_data(self):
        """
        Return integration data as an astropy table.

        Returns
        -------
        astropy.table.Table
        """
        integration_time = self.size * self.info.instrument.integration_time
        if np.isnan(self.nefd):
            self.calculate_source_nefd()
        columns = [
            Column(name='Obs',
                   data=np.atleast_1d(self.integration_number),
                   dtype='int32'),
            Column(name='Integration_Time',
                   data=np.atleast_1d(integration_time.to('second').value),
                   dtype='float64'),
            Column(name='Frames', data=np.atleast_1d(self.size),
                   dtype='int32'),
            Column(name='Relative_Gain', data=np.atleast_1d(self.gain),
                   dtype='float64'),
            Column(name='NEFD', data=np.atleast_1d(self.nefd),
                   dtype='float64'),
            Column(name='Hipass_Timescale',
                   data=np.atleast_1d(
                       self.filter_time_scale.to('second').value),
                   dtype='float64'),
            Column(name='Filter_Resolution',
                   data=np.atleast_1d(0.5 / numba_functions.pow2ceil(
                       self.frames_for(self.filter_time_scale))),
                   dtype='float64')
        ]
        return Table(columns)

    def add_details(self, table=None):
        """
        Add integration details to an astropy table.

        Parameters
        ----------
        table : astropy.table.Table, optional
            If supplied, new columns will be added.  Otherwise, a new table
            will be created.

        Returns
        -------
        Table
        """
        if table is None:
            table = Table()
        columns = [table[column_name] for column_name in table.keys()]

        filter_profile = None
        if self.filter is not None:
            for sub_filter in self.filter.filters:
                if isinstance(sub_filter, WhiteningFilter):
                    whitener = sub_filter
                    filter_profile = whitener.get_valid_profiles()

        columns.append(Column(name='Channel_Index',
                              data=self.channels.data.fixed_index,
                              dtype='int32'))
        columns.append(Column(name='Channel_Gain',
                              data=self.channels.data.gain, dtype='float32'))
        columns.append(Column(name='Channel_Offset',
                              data=self.channels.data.offset, dtype='float32'))
        columns.append(Column(name='Channel_Weight',
                              data=self.channels.data.weight, dtype='float32'))
        columns.append(Column(name='Channel_Flags',
                              data=self.channels.data.flag, dtype='int32'))
        columns.append(Column(name='Channel_Spikes',
                              data=self.channels.data.spikes, dtype='int16'))
        if (filter_profile is not None
                and filter_profile.size > 0):  # pragma: no cover
            columns.append(Column(name='Whitening_Profile',
                                  data=filter_profile.T, dtype='float32'))

        columns.append(Column(name='Noise_Spectrum',
                              data=self.get_spectra()[1].T, dtype='float32'))

        max_size = max([column.shape[0] for column in columns])
        for i, column in enumerate(columns):
            if column.shape[0] != max_size and column.shape[0] == 1:
                shape = column.data.shape
                new_shape = (max_size,) + shape[1:]
                data = np.empty(new_shape, dtype=column.data.dtype)
                data[:] = column.data[0]
                columns[i] = Column(name=column.name, data=data,
                                    dtype=data.dtype)
        return Table(columns)

    def search_corners(self, perimeter_pixels, projection):
        """
        Find the corners of the mapping range of the integration.

        Parameters
        ----------
        perimeter_pixels : ChannelGroup
            A channel group ideally containing pixels near the edge of the
            array (Although this is not strictly necessary - just speeds things
            up).
        projection : Projection2D
            A grid projection used to convert pixel positions onto a map
            grid.

        Returns
        -------
        map_range : Coordinate2D
            The minimum (x, y) and maximum (x, y) map offsets.
        """
        log.debug(f"Search pixels: {perimeter_pixels.size} : "
                  f"{self.channels.size}")
        projector = AstroProjector(projection)
        offsets = self.frames.project(perimeter_pixels.position, projector)

        map_range = int_nf.search_corners(
            sample_coordinates=offsets.coordinates.value,
            valid_frames=self.frames.valid,
            channel_indices=perimeter_pixels.indices,
            sample_flags=self.frames.sample_flag,
            skip_flag=self.flagspace.flags.SAMPLE_SKIP.value) * offsets.unit

        return Coordinate2D(coordinates=map_range, unit='arcsec')

    def set_iteration(self, iteration, rounds=None):
        r"""
        Set the configuration for a given iteration

        Parameters
        ----------
        iteration : float or int or str
            The iteration to set.  A positive integer defines the exact
            iteration number.  A negative integer defines the iteration
            relative to the last (e.g. -1 is the last iteration). A float
            represents a fraction (0->1) of the number of rounds, and
            a string may be parsed in many ways such as last, first, a float,
            integer, or percentage (if suffixed with a %).
        rounds : int, optional
            The maximum number of iterations.

        Returns
        -------
        None
        """
        if rounds is not None:
            self.configuration.iterations.max_iteration = rounds
        self.configuration.set_iteration(iteration)
        self.channels.calculate_overlaps(self.scan.get_point_size())

    def get_floats(self):
        """
        Return a float array suitable for FFT.

        Returns
        -------
        numpy.ndarray (float)
        """
        return np.empty(numba_functions.pow2ceil(self.size), dtype=float)

    @staticmethod
    def get_correlated_signal_class():
        """
        Return the correlated signal class.

        Returns
        -------
        CorrelatedSignal
        """
        return CorrelatedSignal
