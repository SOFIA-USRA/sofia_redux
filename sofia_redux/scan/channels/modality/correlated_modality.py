# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.utilities.range import Range

__all__ = ['CorrelatedModality']


class CorrelatedModality(Modality):

    def __init__(self, name=None, identity=None, channel_division=None,
                 gain_provider=None, mode_class=None):
        """
        Create a correlated modality.

        A Modality is a collection of channel modes.  A channel mode
        extracts/sets/operates-on gains from a channel group (collection
        of channels).  Modes are created by the modality from a channel
        division which is a collection of channel groups.  The type of
        mode may be explicitly defined or will be set to a default mode
        class for the modality.

        The name of the modality should be set, as it is used by the
        modality to apply specific configuration options.

        The correlated modality is similar to the standard Modality but allows
        additional functionality to operate on correlated modes such as the
        ability to update signals in an integration.

        Parameters
        ----------
        name : str, optional
            The name of the modality.
        identity : str, optional
            A shorthand abbreviation for the modality.
        channel_division : ChannelDivision, optional
            A ChannelDivision object containing groups from which to create
            channel modes.
        gain_provider : GainProvider or str, optional
            If a string is provided, a FieldGainProvider will be created that
            defined the gain as that field of the channel group data.
            If a gain provider is explicitly provided, it will be used instead.
        mode_class : Mode, optional
            The mode class to be used when creating modes from the channel
            groups inside the channel division.  If not supplied, the default
            mode class for a modality will be used.
        """
        super().__init__(name=name,
                         identity=identity,
                         channel_division=channel_division,
                         gain_provider=gain_provider,
                         mode_class=mode_class)
        self.gain_range = Range()
        self.solve_signal = True

    def set_options(self, configuration, branch=None):
        """
        Apply a configuration to the modality and modes therein.

        The following information is extracted and applied:

        1. resolution (in seconds)
        2. triggers
        3. whether to solve for gains
        4. whether phase gains are used
        5. the gain range
        6. whether SIGNED or BIDIRECTIONAL gain flagging should be applied
        7. whether no gain fields exist (gain provider is disabled)
        8. whether to solve signals.

        Parameters
        ----------
        configuration : Configuration or dict
            Either a configuration object, or a specific subsection of
            a configuration relevant to the modality.
        branch : str, optional
            If a configuration object was provided, specifies the branch
            that applies to this modality.  If not supplied, the branch
            name is defined as the modality name.

        Returns
        -------
        None
        """
        if branch is None:
            branch = f'correlated.{self.name}'

        super().set_options(configuration, branch=branch)

        if isinstance(configuration, Configuration):
            self.solve_signal = not configuration.get_bool(
                f'{branch}.nosignals')
        elif isinstance(configuration, dict):
            self.solve_signal = ~utils.get_bool(configuration.get('nosignals'))
        else:
            return

    def set_skip_flags(self, gain_skip_flag):
        """
        Sets the gain skip flags in each mode of the modality.

        Parameters
        ----------
        gain_skip_flag : ChannelFlagTypes or str or int
            The gain flag marking channel types that should not be considered
            valid channels.  A string can be supplied such as 'GAIN', 'DEAD',
            etc., the flag type itself, or an integer.  Integers should be used
            with care since meaning may vary between various flag types.

        Returns
        -------
        None
        """
        gain_skip_flag = self.flagspace.convert_flag(gain_skip_flag)
        for mode in self.modes:
            mode.skip_flags = gain_skip_flag

    def update_signals(self, integration, robust=False):
        """
        Update integration signals for each mode in the modality.

        Parameters
        ----------
        integration : Integration
        robust : bool, optional
             If `True`, use the "robust" method to update signals.

        Returns
        -------
        None
        """
        for mode in self.modes[::-1]:  # TODO: reset to forward order (not -1)
            if not np.isnan(self.resolution):
                mode.resolution = self.resolution
                try:
                    mode.update_signals(integration, robust=robust)
                except Exception as err:
                    log.error(err)
                    raise err  # TODO: remove this line
