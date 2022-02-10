# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.channels.mode.response import Response
from sofia_redux.scan.signal.signal import Signal

__all__ = ['NonLinearResponse']


class NonLinearResponse(Response):

    def __init__(self, mode, gain_provider=None, name=None):
        """
        Create a non-linear response mode.

        A mode is an object that is applied to a given channel group, defining
        what constitutes its "gain" and how to operate thereon.  This is
        also dependent on a gain provider.

        The non-linear response mode contains the additional `get_signal`
        method to extract a Signal object from an integration.  In this case,
        the final signal is dependent on a parent mode.

        Parameters
        ----------
        mode : Mode, optional
            The channel group owned by the mode.
        gain_provider : str or GainProvider, optional
            If a string is provided a `FieldGainProvider` will be set to
            operate on the given field of the channel group.
        name : str, optional
            The name of the mode.  If not provided, will be determined from the
            channel group name (if available).
        """
        name = self.__class__.__name__ + '-' + mode.name
        super().__init__(channel_group=mode.channel_group,
                         gain_provider=gain_provider,
                         name=name)
        self.parent_mode = mode

    def get_signal(self, integration):
        """
        Get a signal from an integration.

        The signal values are initially set to zero before being updated by
        the parent signal with drifts removed.  Note that this may appear
        confusing at first: the signal belongs to the integration and is
        created for the integration if necessary.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        signal : Signal
        """
        parent_signal = integration.get_signal(self.parent_mode)
        signal = Signal(integration,
                        mode=self,
                        values=np.zeros(parent_signal.size, dtype=float),
                        is_floating=False)
        self.update_signal(integration)
        return signal

    def update_signal(self, integration):
        """
        Update the signal based on the parent signal values.

        The signal values are set to the square of the parent signal values
        plus parent drifts.  i.e. (parent values + parent drifts)^2.  The
        drifts are then moved from this squared parent signal values.

        In other words, values = (parent_values + parent_drifts)^2 - f(drifts).

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        parent_signal = integration.get_signal(self.parent_mode)
        signal = integration.get_signal(self)

        parent_drifts = parent_signal.drifts
        parent_values = parent_signal.value

        if parent_drifts is not None:
            drift_indices = (np.arange(parent_values.size)
                             // parent_signal.drift_n)
            parent_drifts = parent_drifts[drift_indices]
            signal.value[...] = (parent_values + parent_drifts) ** 2
        else:
            signal.value[...] = parent_values ** 2

        # Remove drifts from the squared signal
        if parent_drifts is not None:
            n_frames = parent_signal.drift_n * parent_signal.resolution
            signal.remove_drifts(n_frames=n_frames, is_reconstructable=False)

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
        self.set_gains(np.zeros(self.channel_group.size, dtype=float))
        self.sync_all_gains(integration, sum_wc2=None, is_temp_ready=True)

        # Calculate the new nonlinearity signal
        self.update_signal(integration)

        # Calculate new gains
        return super().derive_gains(integration, robust=robust)
