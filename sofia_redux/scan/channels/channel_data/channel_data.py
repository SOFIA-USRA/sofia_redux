# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log, units
import numpy as np
import pandas as pd
import re

from sofia_redux.scan.flags.flagged_data import FlaggedData
from sofia_redux.scan.flags.channel_flags import ChannelFlags
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.utilities.class_provider import (
    channel_data_class_for)
from sofia_redux.scan.channels import channel_numba_functions

__all__ = ['ChannelData']


class ChannelData(FlaggedData):

    flagspace = ChannelFlags

    def __init__(self, channels=None):
        super().__init__()
        # These are all arrays of equal size
        self.channel_id = None
        self.overlaps = None
        self.source_phase = None
        self.offset = None
        self.hardware_gain = None
        self.gain = None
        self.nonlinearity = None
        self.coupling = None
        self.weight = None
        self.variance = None
        self.dof = None
        self.dependents = None
        self.source_filtering = None
        self.direct_filtering = None
        self.filter_time_scale = None
        self.one_over_f_stat = None
        self.spikes = None
        self.inconsistencies = None
        self.nonlinearity = None
        self.temp = None
        self.temp_g = None
        self.temp_wg = None
        self.temp_wg2 = None
        self.resolution = None
        self.angular_resolution = None
        self.frequency = None

        # Special reference
        self.channels = None
        self.set_parent(channels)

    @property
    def info(self):
        """
        Return the instrument information object.

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
        Return the configuration for the channel data.

        Returns
        -------
        Configuration
        """
        if self.info is None:
            return None
        return self.info.configuration

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced rather than copied.

        Returns
        -------
        set (str)
        """
        attributes = super().referenced_attributes
        attributes.add('channels')
        return attributes

    @property
    def default_field_types(self):
        defaults = super().default_field_types
        defaults.update({
            'source_phase': 0,
            'offset': 0.0,
            'hardware_gain': 1.0,
            'coupling': 1.0,
            'gain': 1.0,
            'weight': 1.0,
            'variance': 1.0,
            'dof': 1.0,
            'dependents': 0.0,
            'source_filtering': 1.0,
            'direct_filtering': 1.0,
            'filter_time_scale': np.inf * units.Unit('s'),
            'one_over_f_stat': np.nan,
            'spikes': 0,
            'inconsistencies': 0,
            'nonlinearity': 0.0,
            'temp': float,
            'temp_g': float,
            'temp_wg': float,
            'temp_wg2': float,
            'channel_id': str,
            'resolution': np.nan * units.Unit('arcsec'),
            'frequency': np.nan * units.Unit('Hz'),
            'angular_resolution': np.nan * units.Unit('radian')
        })
        return defaults

    @classmethod
    def instance_from_instrument_name(cls, name):
        """
        Returns a ChannelData instance for a given instrument.

        Parameters
        ----------
        name : str
            The name of the instrument.

        Returns
        -------
        ChannelData
        """
        return channel_data_class_for(name)()

    def set_parent(self, channels):
        """
        Set the parent channels of the channel data.

        Parameters
        ----------
        channels : Channels

        Returns
        -------
        None
        """
        self.channels = channels

    def read_pixel_data(self, filename):
        """
        Read a pixel data file and apply the results.

        Parameters
        ----------
        filename : str
            File path to the pixel data file.

        Returns
        -------
        None
        """
        pixel_info = self.read_channel_data_file(filename)
        self.set_flags('DEAD')
        for index, channel_id in enumerate(self.channel_id):
            self.set_channel_data(index, pixel_info.get(channel_id))
        self.validate_pixel_data()

    def set_channel_data(self, index, channel_info):
        """
        Set the channel info for a selected index.

        Parameters
        ----------
        index : int
            The channel index for which to set new data.
        channel_info : dict
            A dictionary of the form {field: value} where.  The attribute
            field at 'index' will be set to value.

        Returns
        -------
        None
        """
        if channel_info is None:
            return
        self.gain[index] = channel_info['gain']
        self.weight[index] = channel_info['weight']
        self.flag[index] = channel_info['flag']

    def validate_pixel_data(self):
        """
        Validates data read from the pixel data file.

        Returns
        -------
        None
        """
        self.set_flags('BLIND', indices=self.gain == 0)

        flags = self.configuration.get_list('pixels.criticalflags',
                                            default=None)
        if flags is None:
            critical_flags = self.flagspace.critical_flags()
            if critical_flags is None:
                return
        else:
            critical_flags = self.flagspace.flags(0)
            for flag in flags:
                critical_flags |= self.flagspace.convert_flag(flag)

        self.flag = self.flagspace.and_operation(self.flag, critical_flags)
        self.validate_weights()

    def validate_weights(self):
        """
        Validates weight data.

        Returns
        -------
        None
        """
        inverse_weight = np.zeros(self.size, dtype=float)
        idx = self.weight > 0
        inverse_weight[idx] = 1.0 / self.weight[idx]
        idx = np.isnan(self.variance)

        # The following lines are required since we don't know if this is
        # operating on ChannelData or a ChannelGroup
        variance = self.variance
        variance[idx] = inverse_weight[idx]
        self.variance = variance

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

    def set_hardware_gain(self, info):
        """
        Set the hardware gain attribute from the supplied info object.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        self.hardware_gain = np.full(self.size, info.instrument.gain)

    def read_wiring_data(self, filename):
        pass

    def kill_channels(self, flag=None):
        """
        Given a flag, sets all matching elements to DEAD only flag status.

        Parameters
        ----------
        flag : int or ChannelFlagTypes

        Returns
        -------
        None
        """
        indices = np.nonzero(self.is_flagged(flag))[0]
        if len(indices) == 0:
            return
        self.unflag(flag, indices=indices)
        self.set_flags('DEAD', indices=indices)

    def flag_field(self, field, specs):
        """
        Given a data field name and list of specifications, flag as DEAD.

        The specifications may define a single value or range of values.  Any
        data value within that range or equal to a specified value will be
        flagged as dead.

        Parameters
        ----------
        field : str
            The data field name.
        specs : list of str
            Each element may contain a single value or range of values
            (marked by lower-upper or lower:upper) that should be flagged as
            dead.

        Returns
        -------
        None
        """
        if not hasattr(self, field):
            log.warning(f"flag_field: {self.__class__.__name__} "
                        f"does not have {field} attribute.")
            return

        log.debug(f"Flagging channels by {field} values")
        values = getattr(self, field)

        if isinstance(specs, str):
            specs = [specs]

        delimited = []
        for spec in specs:
            spec = ''.join(spec.split())  # remove all whitespace
            delimited.extend(spec.split(','))  # delimit commas
        specs = delimited

        for flag_range in specs:

            if ':' in flag_range:
                # This allows for negative numbers
                value_range = re.split(r'[:]', flag_range)
            else:
                value_range = re.split(r'[-]', flag_range)

            if len(value_range) == 1:
                mask = values == float(value_range[0])
            elif len(value_range) == 2:
                if value_range[0] == '*':
                    value_range[0] = -np.inf
                if value_range[1] == '*':
                    value_range[1] = np.inf
                mask = values >= float(value_range[0])
                mask &= values <= float(value_range[1])
            else:
                log.warning(f"Could not parse flag: {field} ({flag_range})")
                continue
            self.set_flags('DEAD', indices=np.nonzero(mask)[0])

    def flag_fields(self, fields):
        """
        Flags elements in various data fields as dead based on data values.

        Parameters
        ----------
        fields : dict
            A dictionary where keys should define a data field name, and
            values are lists of str where each element defines a value or
            range of values (min:max, or min-max) that should be flagged as
            dead for the given field.

        Returns
        -------
        None
        """
        if not isinstance(fields, dict):
            return
        for field, specs in fields.items():
            self.flag_field(field, specs)

    def set_flag_defaults(self):
        """
        Sets data values based on currently set flags.

        Returns
        -------
        None
        """
        mask = self.is_flagged('DEAD|DISCARD')
        self.coupling[mask] = 0
        self.gain[mask] = 0
        self.weight[mask] = 0
        self.variance[mask] = 0
        self.coupling[self.is_flagged(self.flagspace.flags.BLIND)] = 0

    def set_blind_channels(self, fixed_indices):
        """
        Set BLIND flag for elements based on fixed indices.

        Will kill (set flag to only DEAD) any previously defined BLIND
        channels. All new blinded channels will only have the BLIND flag.

        Parameters
        ----------
        fixed_indices : numpy.ndarray of int

        Returns
        -------
        None
        """
        self.kill_channels(self.flagspace.flags.BLIND)
        fixed_indices = np.asarray(fixed_indices, dtype=int)
        log.debug(f"Defining {fixed_indices.size} blind channels")
        indices = self.find_fixed_indices(fixed_indices, cull=True)
        self.unflag(indices=indices)
        self.set_flags(self.flagspace.flags.BLIND, indices=indices)

    def flag_channel_list(self, channel_list):
        """
        Flag channels as DEAD from a list of channel ranges/fixed indices.

        Parameters
        ----------
        channel_list : list or str
            If provided as a string, elements should be comma delimited.  Each
            element can be an int or string specifying fixed indices to flag.
            Note that ranges can be specified via 'min:max' or 'min-max'.

        Returns
        -------
        None
        """
        indices = self.find_fixed_indices(utils.to_int_list(channel_list),
                                          cull=True)
        log.debug(f"Flagging {indices.size} channels.")
        self.set_flags(self.flagspace.flags.DEAD, indices=indices)

    def set_uniform_gains(self, field=None):
        """
        Sets the gain and coupling to 1.0 for all channels.

        Parameters
        ----------
        field : str, optional
            If supplied, sets all values of the requested field to unity.

        Returns
        -------
        None
        """
        if field is None:
            self.gain = np.full(self.size, 1.0)
            self.coupling = np.full(self.size, 1.0)
        else:
            setattr(self, field, np.full(self.size, 1.0))

    def flatten_weights(self):
        """
        Flattens weight according to gain^2.

        Does not include hardware gains.

        Returns
        -------
        None
        """
        keep_mask = self.is_unflagged(self.flagspace.hardware_flags())
        g2 = self.gain[keep_mask] ** 2
        sum_wg2 = np.sum(g2 * self.weight[keep_mask])
        sum_g2 = np.sum(g2)

        if sum_g2 == 0:
            w = 1.0
        else:
            w = sum_wg2 / sum_g2

        self.weight = np.full(self.size, w)

    def get_filtering(self, integration):
        """
        Return the filtering for a given integration.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        filtering : numpy.ndarray (float)
        """
        n_drifts = np.ceil(integration.get_duration() / self.filter_time_scale)
        n_drifts = n_drifts.decompose().value.astype(float)
        return self.direct_filtering * (1.0 - (n_drifts / integration.size))

    def apply_info(self, info):
        """
        Apply scan/instrument information to the channels.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        self.set_hardware_gain(info)

    def get_typical_gain_magnitude(self, gains, keep_flag=None,
                                   discard_flag=None, match_flag=None):
        r"""
        Return the mean gain value of data.

        The mean value may be calculated given a number of flag criteria.  The
        outer 10% of log(1 + \|gain\|) are excluded from the mean calculation.

        Parameters
        ----------
        gains : numpy.ndarray (float)
        keep_flag : int or ChannelFlagTypes, optional
            Flag values to keep in the calculation.
        discard_flag : int or ChannelFlagTypes, optional
            Flag values to discard_flag from the calculation.
        match_flag : int or ChannelFlagTypes, optional
            Only matching flag values will be used in the calculation.

        Returns
        -------
        mean : float
            The mean gain value.
        """
        if isinstance(gains, units.Quantity):
            unit = gains.unit
            if unit == units.dimensionless_unscaled:
                unit = None
            gains = gains.value
        else:
            unit = None

        values = gains[self.get_flagged_indices(
            keep_flag=keep_flag, discard_flag=discard_flag,
            match_flag=match_flag)]

        if values.size == 0:
            return 1.0 if unit is None else 1.0 * unit

        gain_magnitude = channel_numba_functions.get_typical_gain_magnitude(
            values)
        return gain_magnitude if unit is None else gain_magnitude * unit

    def clear_overlaps(self):
        """
        Remove all overlap values.

        Returns
        -------
        None
        """
        if self.overlaps is None:
            return

        self.overlaps.data[...] = 0.0
        self.overlaps.eliminate_zeros()

    def calculate_overlaps(self, point_size, maximum_radius=2.0):
        """
        Calculates the overlaps between channels.

        The overlap array (in the `overlaps` attribute) is a csr_sparse array
        of shape (n_channels, n_channels) where overlaps[i, j] givens the
        overlap value of channel j from the channel i.

        Parameters
        ----------
        point_size : astropy.units.Quantity
            The point size for calculating the overlaps.  Typically, the beam
            FWHM.
        maximum_radius : float, optional
            The maximum radius in units of `point_size` to search for channel
            overlaps.

        Returns
        -------
        None
        """
        radius = point_size * maximum_radius
        overlap_indices = self.get_overlap_indices(radius)
        overlap_distances, distance_unit = self.get_overlap_distances(
            overlap_indices)

        self.calculate_overlap_values(overlap_distances,
                                      point_size.to(distance_unit))

    def add_dependents(self, dependents):
        """
        Add dependents to channel data.

        Parameters
        ----------
        dependents : numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.dependents += dependents

    def remove_dependents(self, dependents):
        """
        Remove dependents from channel data.

        Parameters
        ----------
        dependents : numpy.ndarray (float)

        Returns
        -------
        None
        """
        self.dependents -= dependents

    @abstractmethod
    def get_overlap_indices(self, radius):
        """
        Return a cross-array indicating overlapping indices.

        Parameters
        ----------
        radius : astropy.units.Quantity
            The maximum radius about which to include overlaps.
        pixel_xy_size : astropy.units.Quantity (numpy.ndarray)
            The pixel (x, y) size.

        Returns
        -------
        overlap_indices : scipy.sparse.csr.csr_matrix (bool)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where a `True` value for overlap_indices[i, j] signals that
            channel `i` overlaps with the channel `j`.
        """
        pass

    @abstractmethod
    def get_overlap_distances(self, overlap_indices):
        """
        Calculates the overlap distances.

        The overlap distances are stored in the `overlaps` attribute values.
        This should be a csr_sparse matrix of shape (n_channels, n_channels)
        where overlaps[i, j] gives the distance between channel i and
        channel j.

        Parameters
        ----------
        overlap_indices : scipy.sparse.csr.csr_matrix (bool)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where a `True` value for overlap_indices[i, j] signals that
            channel `i` overlaps with the channel `j`.

        Returns
        -------
        distances, unit : scipy.sparse.csr.csr_matrix, astropy.units.Unit
            `distances` is a Compressed Sparse Row (CSR) matrix of shape
            (channels, channels) and of float type where distances[i, j] gives
            the distance between channels i and j.  `unit` gives the distance
            unit.
        """
        pass

    def calculate_overlap_values(self, overlap_distances, point_size):
        """
        Calculates the overlap values based on overlap distances.

        The overlap values are stored in the `overlaps` attribute values.
        This should be a csr_sparse matrix of shape (n_channels, n_channels)
        where overlaps[i, j] gives the overlap value between channels i and j.

        Parameters
        ----------
        overlap_distances : scipy.sparse.csr.csr_matrix (float)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where distances[i, j] gives the distance between channels i and j.
            These float values should be converted to units of `point_size`.
        point_size : astropy.units.Quantity
            The point size for calculating the overlaps.  Typically, the beam
            FWHM.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_pixel_count(self):
        """
        Return the number of pixels in the arrangement.

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
        Channels
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
        Group
            A newly created channel group.
        """
        pass

    def to_string(self, indices=None, frame=False):
        """
        Return a string representation of channels.

        Parameters
        ----------
        indices : numpy.ndarray or slice, optional
            The channel indices (not fixed) to return.  The default is all
            channels.
        frame : bool, optional
            If `True`, returns a :class:`pd.DataFrame` instead of a string
            representation.

        Returns
        -------
        str or pd.DataFrame
        """
        if indices is None:
            indices = slice(None)
        df = pd.DataFrame(
            {'ch': self.channel_id[indices],
             'gain': list(map(lambda x: "%.3f" % x, self.gain[indices])),
             'weight': list(map(lambda x: "%.3e" % x, self.weight[indices])),
             'flag': self.flagspace.to_letters(self.flag[indices])})
        if frame:
            return df
        else:
            return df.to_csv(sep='\t', index=False)

    def get_rcp_string(self, indices=None):
        """
        Return a string representation for the RCP of ALL channels.

        Parameters
        ----------
        indices : numpy.ndarray or slice, optional
            The channel indices (not fixed) to return.  The default is all
            channels.

        Returns
        -------
        str
        """
        return None
