# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
import numpy as np

from sofia_redux.scan.flags.instrument_flags import InstrumentFlags
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.channels.gain_provider.field_gain_provider import (
    FieldGainProvider)
from sofia_redux.scan.channels.gain_provider.gain_provider import GainProvider

__all__ = ['Mode']


class Mode(ABC):

    instrument_flagspace = InstrumentFlags

    def __init__(self, channel_group=None, gain_provider=None, name=None):
        """
        Create a mode operating on a given channel group.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        Parameters
        ----------
        channel_group : ChannelGroup, optional
            The channel group owned by the mode.
        gain_provider : str or GainProvider, optional
            If a string is provided a `FieldGainProvider` will be set to
            operate on the given field of the channel group.
        name : str, optional
            The name of the mode.  If not provided, will be determined from the
            channel group name (if available).
        """
        self.name = name
        self.channel_group = None
        self.gain_provider = None
        self.coupled_modes = None

        self.fixed_gains = False
        self.phase_gains = False
        self.resolution = 0.0 * units.s
        self.gain_range = Range()
        self.gain_flag = None
        self.gain_type = self.instrument_flagspace.flags.GAINS_BIDIRECTIONAL

        self.counter = 0
        self.gain = None
        if channel_group is not None:
            self.set_channel_group(channel_group)
        if gain_provider is not None:
            self.set_gain_provider(gain_provider)

    @property
    def size(self):
        """
        Return the size of the channel group.

        Returns
        -------
        int
            The number of channels owned by the channel group.
        """
        if self.channel_group is None:
            return 0
        else:
            return self.channel_group.size

    @property
    def flagspace(self):
        """
        Return the flagspace for the mode channel group.

        Returns
        -------
        Flags
        """
        if self.channel_group is None:
            return None
        else:
            return self.channel_group.flagspace

    def __str__(self):
        """
        Return a string representation of the mode.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__} ({self.name}): {self.size} channels"

    def to_string(self):
        """
        Returns a long description of channels in the mode.

        Returns
        -------
        description : str
        """
        description = str(self.name) + ":"
        channel_ids = self.channel_group.channel_id
        if channel_ids is None:
            return description
        description += " " + " ".join(channel_ids)
        return description

    def set_channel_group(self, channel_group):
        """
        Apply a channel group to the mode.

        During this phase the gain flag type is determined as the zeroth flag
        of the channel group flagspace if previously undefined.  In addition,
        the name of the mode will be set to the name of the channel group
        if not previously defined.

        Parameters
        ----------
        channel_group : ChannelGroup

        Returns
        -------
        None
        """
        self.channel_group = channel_group
        if self.gain_flag is None:
            self.gain_flag = self.flagspace.flags(0)
        if self.coupled_modes is not None:
            for mode in self.coupled_modes:
                mode.set_channel_group(channel_group)

    def set_name(self, name=None):
        """
        Set the name of the mode.

        If not provided, the name will be determined from the channel group.
        If no channel group is available, the name will not be set.

        Parameters
        ----------
        name : str, optional
            The new name of the mode.

        Returns
        -------
        None
        """
        if name is not None:
            self.name = name
        else:
            if self.name is None:
                if self.channel_group is not None:
                    self.name = self.channel_group.name

    def set_gain_provider(self, gain_provider):
        """
        Set the gain provider that operates on the channel group.

        The gain provider determines what attributes or fields of a channel
        group will be defined as the gain.

        Parameters
        ----------
        gain_provider : str or GainProvider
            If a string is provided, a FieldGainProvider will be created that
            defines the gain as that field of the channel group data.
            If a gain provider is explicitly provided, it will be used instead.

        Returns
        -------
        None
        """
        if isinstance(gain_provider, str):
            self.gain_provider = FieldGainProvider(gain_provider)
        elif isinstance(gain_provider, GainProvider):
            self.gain_provider = gain_provider
        elif gain_provider is None:
            self.gain_provider = None
        else:
            raise ValueError(f"gain must be a {str} or {GainProvider}. "
                             f"Received {type(gain_provider)}.")

    def add_coupled_mode(self, coupled_mode):
        """
        Add a coupled mode to the available coupled modes of the mode.

        Parameters
        ----------
        coupled_mode : CoupledMode
            A coupled mode to append to the contained coupled modes of mode.

        Returns
        -------
        None
        """
        if self.coupled_modes is None:
            self.coupled_modes = []
        self.coupled_modes.append(coupled_mode)

    def get_gains(self, validate=True):
        """
        Return the gain values of the mode.

        If no gains are available and no gain provider is available, will
        return an array of ones.

        Parameters
        ----------
        validate : bool, optional
            If `True` (default), will cause the gain provider to "validate"
            the mode itself.  This could mean anything and is dependent on the
            gain provider.

        Returns
        -------
        gains : numpy.ndarray (float)
            The gain values.
        """
        if self.gain is None:
            self.gain = np.ones(self.channel_group.size, dtype=np.float64)
        elif self.gain.size != self.size:
            raise ValueError("Gain array size differs from mode channels.")

        if self.gain_provider is not None:
            self.apply_provider_gains(validate)

        if isinstance(self.gain, units.Quantity):
            if self.gain.unit == units.dimensionless_unscaled:
                self.gain = self.gain.value

        return self.gain

    def apply_provider_gains(self, validate):
        """
        Sets the internal gain values as returned by the gain provider.

        Any NaN gain values will be replaced by zeros.

        Parameters
        ----------
        validate : bool
            If `True`, causes the gain provider to "validate" the mode.  This
            could mean anything, but is usually used to update internal
            gain provider settings.

        Returns
        -------
        None
        """
        if validate:
            self.gain_provider.validate(self)

        gain = self.gain_provider.get_gain(self.channel_group)
        if (isinstance(gain, units.Quantity)
                and gain.unit == units.dimensionless_unscaled):
            gain = gain.value
        self.gain = gain
        self.gain[np.isnan(self.gain)] = 0.0

    def set_gains(self, gain, flag_normalized=True):
        """
        Set the gain values of the mode.

        If a gain provider is available, it will be used to update the gain
        values, which could also update values in the channel group.  Gains
        may be flagged depending on whether a gain range has been set
        (in the `gain_range` attribute).  Note that any flagging will back
        propagate to the channel group and therefore, the channels themselves.

        Parameters
        ----------
        gain : numpy.ndarray (float)
            The new gain values to apply.
        flag_normalized : bool, optional
            If `True`, will flag gain values outside the gain range after
            normalizing to the average gain value of those previously flagged.

        Returns
        -------
        flagging : bool
            If gain flagging was performed.  This does not necessarily mean
            any channels were flagged, just that it was attempted.
        """
        if self.gain_provider is None:
            if isinstance(gain, units.Quantity):
                if gain.unit == units.dimensionless_unscaled:
                    gain = gain.value
            self.gain = np.asarray(gain, dtype=float)
        else:
            # Updates values in the channel group via the gain provider.
            self.gain_provider.set_gain(self.channel_group, gain)

        return self.flag_gains(flag_normalized)

    def flag_gains(self, normalize):
        """
        Flag gain values and propagate back to channel group.

        Gains that are outside of the `gain_range` attribute range are flagged
        as the specified gain flag in the `gain_flag` attribute.  Note that
        previously flagged channels may be unflagged if in-range.

        Parameters
        ----------
        normalize : bool
            If `True`, before checking if the gain values are within the
            allowable gain range, normalize with respect to the average
            gain of those channels previously flagged.

        Returns
        -------
        flagging : bool
            `True` if gains were checked for flagging and `False` otherwise.
            Note that this does not necessarily mean any flags were actually
            updated.
        """
        if self.gain_flag.value == 0:
            # No flagging required/available
            return False

        signed = self.instrument_flagspace.flags.GAINS_SIGNED
        bi_directional = self.instrument_flagspace.flags.GAINS_BIDIRECTIONAL

        gain = self.get_gains().copy()
        if self.gain_type not in [signed, bi_directional]:
            gain.fill(np.nan)
        elif normalize:
            average_gain = self.channel_group.get_typical_gain_magnitude(
                gain, discard_flag=~self.gain_flag)
            gain /= average_gain

        if self.gain_type == bi_directional:
            gain = np.abs(gain)

        in_range = self.gain_range.in_range(gain)
        not_in_range = np.nonzero(~in_range)[0]
        in_range = np.nonzero(in_range)[0]

        if not_in_range.size > 0:
            self.channel_group.set_flags(self.gain_flag, indices=not_in_range)
        if in_range.size > 0:
            self.channel_group.unflag(flag=self.gain_flag, indices=in_range)

        return True

    def uniform_gains(self):
        """
        Sets all gains to unity.

        Returns
        -------
        None
        """
        self.set_gains(np.ones(self.channel_group.size, dtype=float),
                       flag_normalized=False)

    def derive_gains(self, integration, robust=True):
        """
        Return gains and weights derived from an integration.

        The returned values are the integration gains plus the mode gains.
        Weights are determined from only the integration.

        Parameters
        ----------
        integration : Integration
        robust : bool, optional
            If `True`, derives the gain increment from the integration using
            the "robust" definition.  This is only applicable if the
            integration is not phase modulated.

        Returns
        -------
        gains, weights : numpy.ndarray (float), numpy.ndarray (float)
            The gains and weights derived from the integration and mode.  Note
            that all non-finite values are reset to zero weight and zero value.
        """
        if self.fixed_gains:
            raise ValueError("Cannot solve gains for fixed gain modes.")
        g0 = self.get_gains()
        signal = integration.get_signal(mode=self)
        gains, weights = signal.get_gain_increment(robust)

        invalid = ~np.isfinite(gains)
        gains[invalid] = g0[invalid]
        weights[invalid] = 0.0
        gains[~invalid] += g0[~invalid]

        return gains, weights

    def sync_all_gains(self, integration, sum_wc2, is_temp_ready=False):
        """
        Synchronize all gains in the mode.

        Parameters
        ----------
        integration : Integration
        sum_wc2 : numpy.ndarray (float)
            An array of channel gains of shape (n_channels,)
        is_temp_ready : bool, optional
            Indicates whether the frame temporary values have already been
            calculated.  These should contain::

                temp_c = signal_value
                temp_wc = relative_weight * signal_value
                temp_wc2 = relative_weight * signal_value^2

        Returns
        -------
        None
        """
        integration.get_signal(self).synchronize_gains(
            sum_wc2=sum_wc2, is_temp_ready=is_temp_ready)

        # sync the gains to all the dependent modes too
        if self.coupled_modes is not None:
            for mode in self.coupled_modes:
                mode.resync_gains(integration)

    def get_frame_resolution(self, integration):
        """
        Returns the integration frame resolution.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        resolution : int
            The integration frame resolution.
        """
        return integration.power2_frames_for(self.resolution / np.sqrt(2.0))

    def signal_length(self, integration):
        """
        Return the length of signal in terms of integration frame resolution.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        length : int
             The length of the signal in terms of integration frame resolution.
        """
        return utils.roundup_ratio(integration.size,
                                   self.get_frame_resolution(integration))
