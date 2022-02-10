# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log, units
import numpy as np

from sofia_redux.scan.channels.mode.mode import Mode
from sofia_redux.scan.flags.instrument_flags import InstrumentFlags
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.utilities.class_provider import default_modality_mode

__all__ = ['Modality']


class Modality(ABC):

    def __init__(self, name=None, identity=None, channel_division=None,
                 gain_provider=None, mode_class=None):
        """
        Create a Modality.

        A Modality is a collection of channel modes.  A channel mode
        extracts/sets/operates-on gains from a channel group (collection
        of channels).  Modes are created by the modality from a channel
        division which is a collection of channel groups.  The type of
        mode may be explicitly defined or will be set to a default mode
        class for the modality.

        The name of the modality should be set, as it is used by the
        modality to apply specific configuration options.

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
        mode_class : class, optional
            The mode class to be used when creating modes from the channel
            groups inside the channel division.  If not supplied, the default
            mode class for a modality will be used.
        """
        self.name = name
        self.id = identity
        self.trigger = None
        self.solve_gains = True
        self.phase_gains = False
        self.resolution = np.nan * units.Unit('s')
        self.mode_class = None
        self.modes = None

        self.set_mode_class(mode_class)
        self.set_channel_division(channel_division)
        self.set_gain_provider(gain_provider)

    @property
    def size(self):
        """
        Returns the number of modes in the modality.

        Returns
        -------
        n_modes : int
        """
        if self.modes is None:
            return 0
        else:
            return len(self.modes)

    @property
    def flagspace(self):
        """
        Retrieves the channel flagspace from the first valid mode.

        Returns
        -------
        ChannelFlags
        """
        if self.modes is None:
            return None
        for mode in self.modes:
            if isinstance(mode, Mode):
                return mode.flagspace
        else:
            return None

    @property
    def fields(self):
        if self.size == 0:
            return set([])
        else:
            return self[0].channel_group.fields

    @classmethod
    def default_class_mode(cls):
        """
        Return a default mode class based on modality class name.

        For example, A CoupledModality should return CoupledMode.  If
        no analogous mode is found, a default base Mode will be returned.

        Returns
        -------
        Mode
        """
        return default_modality_mode(cls)

    def get_mode_name_index(self, mode_name):
        """
        Given a mode name, return its mode index.

        Parameters
        ----------
        mode_name : str
            The name of the mode.

        Returns
        -------
        index : int
        """
        for index, mode in enumerate(self.modes):
            if mode.name == mode_name:
                return index
        else:
            mode_name = self.name + ':' + mode_name
            for index, mode in enumerate(self.modes):
                if mode.name == mode_name:
                    return index
            else:
                return None

    def validate_mode_index(self, index_or_mode_name):
        """
        Return the valid index of a given mode.

        Raises an error if invalid.

        Parameters
        ----------
        index_or_mode_name : int or str
           The name of the mode, or the mode index.

        Returns
        -------
        index : int
        """
        if self.size == 0:
            raise KeyError("No modes available in Modality.")

        if isinstance(index_or_mode_name, int):
            index = index_or_mode_name
        elif isinstance(index_or_mode_name, str):
            index = self.get_mode_name_index(index_or_mode_name)
            if index is None:
                raise KeyError(f"Mode {index_or_mode_name} "
                               f"does not exist in modality.")
        else:
            raise ValueError(f"Invalid index type: {type(index_or_mode_name)}")

        if index < 0:
            reverse_index = self.size + index
            if reverse_index < 0:
                raise IndexError(f"Cannot use index {index} "
                                 f"with modality size {self.size}.")
            index = reverse_index

        if index >= self.size:
            raise IndexError(f"Mode {index_or_mode_name} out of range. "
                             f"Modality size = {self.size}.")
        return index

    def __str__(self):
        """
        Return a string representation of the modality.

        Returns
        -------
        str
        """
        name = self.__class__.__name__
        result = f"{name} (name={self.name} id={self.id}): {self.size} mode(s)"
        if self.size == 0:
            return result
        result += '\n' + '\n'.join([mode.__str__() for mode in self.modes])
        return result

    def __getitem__(self, index_or_mode_name):
        """
        Return a mode for the selected index or name from the modality.

        Parameters
        ----------
        index_or_mode_name : int or str
            The index or name of the mode.

        Returns
        -------
        Mode
        """
        index = self.validate_mode_index(index_or_mode_name)
        return self.modes[index]

    def __setitem__(self, index_or_mode_name, mode):
        """
        Set a modality index or name to the given mode.

        Parameters
        ----------
        index_or_mode_name : int or str
            The index or name of the mode.
        mode : Mode
            The mode to set in the modality.

        Returns
        -------
        None
        """
        if not isinstance(mode, Mode):
            raise ValueError(f"Mode must be of {Mode} type.")
        index = self.validate_mode_index(index_or_mode_name)
        self.modes[index] = mode

    def to_string(self):
        """
        Returns a long description of modes and associated channels.

        Returns
        -------
        description : str
        """
        description = self.__class__.__name__ + f" '{self.id}':"
        if self.size == 0:
            return description
        description += '\n' + '\n'.join([mode.to_string()
                                         for mode in self.modes])
        return description

    def set_mode_class(self, mode_class):
        """
        Set the mode class.

        Parameters
        ----------
        mode_class : Mode class

        Returns
        -------
        None
        """
        if mode_class is None:
            self.mode_class = self.default_class_mode()
        elif issubclass(mode_class, Mode):
            self.mode_class = mode_class
        else:
            raise ValueError(f"Mode class must be a {Mode}.")

    def set_channel_division(self, channel_division):
        """
        Create modes from a channel division.

        A Channel division contains a collection of channel groups.  No
        gain provider is defined at this stage, but the name of each
        mode will be set to <name>:<channel data name in group>.

        Parameters
        ----------
        channel_division : Division

        Returns
        -------
        None
        """
        if channel_division is None:
            return
        self.modes = []
        for channel_group in channel_division.groups:
            if channel_group is not None:
                mode = self.mode_class(channel_group=channel_group)
                self.modes.append(mode)
        self.set_default_names()

    def set_default_names(self):
        """
        Sets the default name for each mode in the modality.

        The default name is set to
        <modality name>:<channel data name in group>.

        Returns
        -------
        None
        """
        if self.modes is None:
            return
        for mode in self.modes:
            mode.set_name(self.name + ':' + mode.channel_group.name)

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
        if isinstance(configuration, Configuration):
            if branch is None:
                branch = self.name

            resolution = configuration.get_float(
                f'{branch}.resolution', default=0.0)
            trigger = configuration.get_string(
                f'{branch}.trigger', default=None)
            solve_gains = not configuration.get_bool(f'{branch}.nogains')
            phase_gains = configuration.get_bool(f'{branch}.phasegains')
            gain_range = configuration.get_range(f'{branch}.gainrange')
            signed = configuration.get_bool(f'{branch}.signed')
            no_gain_field = configuration.get_bool(f'{branch}.nofield')

        elif isinstance(configuration, dict):
            resolution = utils.get_float(
                configuration.get('resolution'), default=0.0)
            trigger = utils.get_string(
                configuration.get('trigger'), default=None)
            solve_gains = not utils.get_bool(
                configuration.get('nogains'))
            phase_gains = utils.get_bool(configuration.get('phasegains'))
            gain_range = utils.get_range(configuration.get('gainrange'))
            signed = utils.get_bool(configuration.get('signed'))
            no_gain_field = utils.get_bool(configuration.get('nofield'))

        elif configuration is None:
            return

        else:
            raise ValueError(
                f"Configuration must be {Configuration} or {dict}.")

        self.resolution = resolution * units.Unit('second')
        self.trigger = trigger
        self.solve_gains = solve_gains
        self.set_gain_range(gain_range)

        if signed:
            gain_direction = InstrumentFlags.flags.GAINS_SIGNED
        else:
            gain_direction = InstrumentFlags.flags.GAINS_BIDIRECTIONAL
        self.set_gain_direction(gain_direction)
        self.set_phase_gains(phase_gains)

        if self.modes is not None:
            for mode in self.modes:
                if no_gain_field:
                    mode.set_gain_provider(None)

    def set_gain_range(self, gain_range):
        """
        Sets the allowable gain range for each mode in the modality.

        The gain range defines the allowable range of gain values.  Values
        outside this range are flagged.

        Parameters
        ----------
        gain_range : Range
            A Range object with lower and upper bounds.

        Returns
        -------
        None
        """
        if self.modes is None:
            return
        for mode in self.modes:
            mode.gain_range = gain_range

    def set_gain_direction(self, gain_direction):
        """
        Sets the gain direction to signed or bi-directional for each mode.

        Gain direction is predominantly important when a mode is flagging
        gains.  The two directions are GAINS_SIGNED, or GAINS_BIDIRECTIONAL.
        If the gains are "bidirectional", the absolute value is used
        during the flagging operation.

        Parameters
        ----------
        gain_direction : InstrumentFlagTypes or str or int
            May be one of {InstrumentFlagTypes.GAINS_SIGNED} or
            {InstrumentFlagTypes.GAINS_BIDIRECTIONAL}.  Allowable string
            values are {"GAINS_SIGNED", "GAINS_BIDIRECTIONAL"} (case
            irrelevant).  If integers are used, care should be taken to
            ensure the appropriate flag is set.

        Returns
        -------
        None
        """

        gain_direction = InstrumentFlags.convert_flag(gain_direction)
        if self.modes is None:
            return
        for mode in self.modes:
            mode.gain_type = gain_direction

    def set_gain_flag(self, gain_flag):
        """
        Set the gain flag for each mode in the modality.

        Parameters
        ----------
        gain_flag : ChannelFlagTypes or str or int
            The gain flag marking channel types for gain determination.
            A string can be supplied such as 'GAIN', 'DEAD', etc., the
            flag type itself, or an integer.  Integers should be used
            with care since meaning may vary between various flag types.

        Returns
        -------
        None
        """
        if self.modes is None or self.flagspace is None:
            return

        gain_flag = self.flagspace.convert_flag(gain_flag)

        for mode in self.modes:
            mode.gain_flag = gain_flag

    def set_phase_gains(self, phase_gains=None):
        """
        Set whether phase gains are applied to each mode in modality.

        Parameters
        ----------
        phase_gains : bool, optional
            If not supplied, all mode phase gains are set to that indicated
            in the modality.  Otherwise, the new setting is applied to
            each mode and updated in the modality.

        Returns
        -------
        None
        """
        if self.modes is None:
            return
        if phase_gains is not None:
            self.phase_gains = phase_gains
        for mode in self.modes:
            mode.phase_gains = self.phase_gains

    def set_gain_provider(self, gain_provider):
        """
        Define the gain provider for each mode in the modality.

        The gain provider determines what constitutes "gain" given channel
        data for each mode.

        Parameters
        ----------
        gain_provider : str or GainProvider
            If a string is provided a `FieldGainProvider` will be set to
            operate on the given field of the channel group.  Otherwise,
            a gain provider must be specifically defined.

        Returns
        -------
        None
        """
        if self.modes is None:
            return
        for mode in self.modes:
            mode.set_gain_provider(gain_provider)

    def update_all_gains(self, integration, robust=False):
        """
        Update all gains in the integration from the modality.

        Parameters
        ----------
        integration : Integration
            The integration to update.
        robust : bool, optional
            If `True`, use the robust (median) method to calculate means.  Use
            a simple mean otherwise.

        Returns
        -------
        updated : bool
            `True` if gains were updated.
        """
        if not self.solve_gains:
            return False

        if self.modes is None:
            return False
        is_flagging = False
        for mode in self.modes:  # This is the correct order
            if mode.fixed_gains:
                continue
            try:
                gains, weights = mode.derive_gains(integration, robust=robust)
                is_flagging |= mode.set_gains(gains, flag_normalized=True)

                # Sync all gains updates frame data and the frame/channel parms
                mode.sync_all_gains(integration, weights, is_temp_ready=True)
            except Exception as err:
                log.error(f"Could not update gains for {mode}: {err}")

        return is_flagging

    def average_gains(self, integration, gains, gain_weights, robust=False):
        """
        Average gains from an integration with supplied gains for the modality.

        Gains and weights are updated in-place.

        Parameters
        ----------
        integration : Integration
        gains : numpy.ndarray (float)
            The current gains with which to average integration gains.
        gain_weights : numpy.ndarray (float)
            The current gain weights of the averaging.
        robust : bool, optional
            If `True`, use the robust (median) method to determine means.
            Otherwise, use a weighted mean.

        Returns
        -------
        None
        """
        for mode in self.modes:
            if mode.fixed_gains:
                continue
            channel_indices = mode.channel_group.indices
            new_gain, new_weight = mode.derive_gains(
                integration, robust=robust)
            update_gain = gains[channel_indices]
            update_weight = gain_weights[channel_indices]
            update_gain = ((update_weight * update_gain)
                           + (new_gain * new_weight))
            update_weight = update_weight + new_weight
            nzi = update_weight > 0
            update_gain[nzi] /= update_weight[nzi]
            gains[channel_indices] = update_gain
            gain_weights[channel_indices] = update_weight

    def apply_gains(self, integration, gains, gain_weights):
        """
        Apply gains to an integration for the modality.

        Parameters
        ----------
        integration : Integration
        gains : numpy.ndarray (float)
            The gain values to apply.
        gain_weights : numpy.ndarray (float)
            The weight of the gain values to apply.

        Returns
        -------
        flagged : bool
            Indicates whether any channels in the modality were flagged with
            out-of-range gain values.
        """
        flagged = False
        for mode in self.modes:
            if mode.fixed_gains:
                continue
            channel_indices = mode.channel_group.indices
            mode_gains = gains[channel_indices]
            mode_sum_wc2 = gain_weights[channel_indices]
            flagged |= mode.set_gains(mode_gains, flag_normalized=True)
            mode.sync_all_gains(integration, mode_sum_wc2, is_temp_ready=True)

        return flagged
