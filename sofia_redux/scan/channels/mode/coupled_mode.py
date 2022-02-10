# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.channels.mode.correlated_mode import CorrelatedMode

__all__ = ['CoupledMode']


class CoupledMode(CorrelatedMode):

    def __init__(self, mode, gain_provider=None):
        """
        Creates and appends a coupled mode from a base mode.

        Parameters
        ----------
        mode : Mode
            The mode from which to base the coupling.  Is added to the modes
            coupled modes.
        gain_provider : str or GainProvider or numpy.ndarray (float)
            If a string is supplied, a FieldGainProvider will be used.  The
            gains may be set by supplying an array, or a GainProvider may be
            explicitly supplied.
        """
        name = self.__class__.__name__ + '-' + mode.name
        super().__init__(channel_group=mode.channel_group, name=name)
        self.parent_mode = mode
        self.parent_mode.add_coupled_mode(self)
        self.fixed_gains = True

        if isinstance(gain_provider, np.ndarray):
            super().set_gains(gain_provider)
        else:
            self.set_gain_provider(gain_provider)

    def get_gains(self, validate=True):
        """
        Return the gain values of the mode.

        If no gains are available and no gain provider is available, will
        return an array of ones.  The final output gains are the product of
        the parent mode gains and the gains of the coupled mode.

        Note that this is a correlated mode, so the gains retrieved from the
        gain provider will be normalized w.r.t the typical gain magnitude of
        the gain flagged channels before being multiplied by the parent gains.

        Parameters
        ----------
        validate : bool, optional
            If `True` (default), will cause the gain provider to "validate"
            the mode itself, and the parent mode to which it is coupled.  This
            could mean anything and is dependent on each gain provider.

        Returns
        -------
        gains : numpy.ndarray (float)
            The gain values returned are the product of the parent gains, and
            the coupled mode gains.
        """
        parent_gains = self.parent_mode.get_gains(validate=validate)
        gains = super().get_gains(validate=validate)
        gains *= parent_gains
        if (isinstance(gains, units.Quantity)
                and gains.units == units.dimensionless_unscaled):
            gains = gains.value
        return gains

    def set_gains(self, gain, flag_normalized=True):
        """
        Not implemented for coupled modes.

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
        raise NotImplementedError(
            f"Cannot adjust gains for {self.__class__.__name__}.")

    def resync_gains(self, integration):
        """
        Resynchronizes all gains in an integration and coupled modes.

        Aside from resynchronizing all dependent gains in an integration, the
        coupled mode may have dependent coupled modes.  These will also be
        resynchronized.

        Parameters
        ----------
        integration : Integration
            The integration to resynchronize.

        Returns
        -------
        None
        """
        # recursively resync all dependent modes.
        signal = integration.get_signal(self)
        if signal is not None:
            signal.resync_gains()

        # sync gains to all dependent modes too
        if self.coupled_modes is not None:
            for mode in self.coupled_modes:
                mode.resync_gains(integration)
