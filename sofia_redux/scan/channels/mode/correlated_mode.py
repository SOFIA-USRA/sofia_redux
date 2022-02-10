# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.channels.mode.mode import Mode
from sofia_redux.scan.signal.correlated_signal import CorrelatedSignal

__all__ = ['CorrelatedMode']


class CorrelatedMode(Mode):

    def __init__(self, channel_group=None, gain_provider=None, name=None):
        """
        Create a correlated mode operating on a given channel group.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        The correlated mode normalizes gains by the typical gain magnitude.
        It also provides methods to update signals in an integration.

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
        self.skip_flags = None
        super().__init__(channel_group=channel_group,
                         gain_provider=gain_provider,
                         name=name)

    def set_channel_group(self, channel_group):
        """
        Apply a channel group to the correlated mode.

        All channel flags except for the zero flag will be marked as flags to
        ignore by the correlated mode.

        Parameters
        ----------
        channel_group : ChannelGroup

        Returns
        -------
        None
        """
        super().set_channel_group(channel_group)
        self.skip_flags = self.flagspace.all_flags()  # everything but 0

    def get_gains(self, validate=True):
        """
        Return the normalized gain values of the correlated mode.

        If no gains are available and no gain provider is available, will
        return an array of ones.  The gains returned will always be normalized
        with respect to the typical gain values of the gain flagged channels.

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
        gains = super().get_gains(validate=validate)
        self.normalize_gains(gains)
        if isinstance(self.gain, units.Quantity):
            if self.gain.unit == units.dimensionless_unscaled:
                self.gain = self.gain.value

        if isinstance(gains, units.Quantity):
            if gains.unit == units.dimensionless_unscaled:
                return gains.value

        return gains

    def set_gains(self, gain, flag_normalized=True):
        """
        Set the gain values of the mode.

        If a gain provider is available, it will be used to update the gain
        values, which could also update values in the channel group.  Gains
        may be flagged depending on whether a gain range has been set
        (in the `gain_range` attribute).  Note that any flagging will back
        propagate to the channel group and therefore, the channels themselves.

        Correlated mode

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
        self.normalize_gains(gain)
        return super().set_gains(gain, flag_normalized=False)

    def normalize_gains(self, gain=None):
        """
        Normalizes the supplied (or contained) gains and returns the average.

        If no gains are supplied, they are retrieved from the gain provider and
        normalized by the typical gain magnitude of the gain flagged channels.
        Note that if this is the case, the gain provider will store the
        normalized values.  The average (normalization factor) is returned to
        the caller.

        Note that unlike the parent Mode class, average gain values are those
        derived from channels only flagged by the gain type flag, not those
        that include the gain flag.  However, no flagging of gain values will
        occur.

        Parameters
        ----------
        gain : numpy.ndarray (float), optional
            The gain values to normalize.

        Returns
        -------
        average_gain : float
            The average gain prior to normalization.
        """
        if gain is None:
            gain = super().get_gains(validate=True)
            average_gain = self.normalize_gains(gain)
            super().set_gains(gain, flag_normalized=False)
            return average_gain

        discard_flags = self.skip_flags & ~self.gain_flag
        average_gain = self.channel_group.get_typical_gain_magnitude(
            gain, discard_flag=discard_flags)

        if average_gain == 1:
            return 1.0

        gain /= average_gain  # Gain updated in-place
        return average_gain

    def get_valid_channels(self):
        """
        Return a channel group containing channels not flagged by `skip_flags`.

        Returns
        -------
        ChannelGroup
        """
        return self.channel_group.create_data_group(
            indices=self.channel_group.is_unflagged(self.skip_flags),
            name=self.name + '-valid')

    def update_signals(self, integration, robust=False):
        """
        Update signals in an integration.

        If the integration does not contain the required signal, it will be
        added to the integration.

        Parameters
        ----------
        integration : Integration
        robust : bool, optional
            If `True`, update the signals using the "robust" (median) method.
            Otherwise, use a weighted mean.

        Returns
        -------
        None
        """
        signal = integration.get_signal(self)
        if signal is None:
            signal = CorrelatedSignal(integration, self)

        signal.update(robust)
