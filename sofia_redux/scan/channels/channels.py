# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log, units
from copy import deepcopy
import numpy as np

from sofia_redux.scan.utilities import utils
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.channels.channel_group.channel_group import ChannelGroup
from sofia_redux.scan.channels.division.division import ChannelDivision
from sofia_redux.scan.channels.modality.modality import Modality
from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)
from sofia_redux.scan.channels.modality.coupled_modality import (
    CoupledModality)
from sofia_redux.scan.channels.modality.non_linear_modality import (
    NonlinearModality)
from sofia_redux.scan.channels.mode.pointing_response import PointingResponse
from sofia_redux.scan.channels.mode.acceleration_response import (
    AccelerationResponse)
from sofia_redux.scan.channels.mode.chopper_response import ChopperResponse
from sofia_redux.scan.channels import channel_numba_functions as cnf
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['Channels']


class Channels(ABC):

    def __init__(self, name=None, parent=None, info=None, size=0):
        """
        Creates a Channels instance.

        The Channels class holds and operates on each "channel" of an
        instrument.  The channel information generally consists of time-
        independent characteristics for a single measuring element of the
        instrument.

        A typical example would be camera pixels.  Each pixel in the array
        will have certain characteristics such as a position that does not
        vary over time.  This will also contain flagging information such
        as whether a channel is working correctly etc.  The `data` attribute
        contains a numpy array for each "field" (characteristic) with each
        element representing a single channel.

        The Channels class also breaks down individual channels into groups
        (a collection of channels), divisions (a collection of groups), and
        modalities (a collection of modes).  Each mode in a modality is created
        from a group in a division.  A mode is an object defining the gain
        properties of a channel group.  i.e., a mode reads channel group
        information to retrieve or set gain.

        Parameters
        ----------
        parent : object, optional
            The owner of the channels such as a Reduction, Scan or Integration.
        info : Info, optional
            The channel information.
        """
        self.info = None
        self.data = None
        self.groups = None
        self.divisions = None
        self.modalities = None
        self.standard_weights = False
        self.fixed_source_gains = False
        self.overlap_point_size = np.nan * units.Unit('arcsec')
        self.is_initialized = False
        self.is_valid = False

        if info is not None:
            self.set_info(info)
        if parent is not None:
            self.set_parent(parent)

        if self.info is not None:
            if name is None:
                self.set_name(self.info.instrument.name)
            else:
                self.set_name(name)
            self.data = self.get_channel_data_instance()

        self.startup_info = info
        self.n_store_channels = size

    def copy(self):
        """
        Create and return a copy.

        Returns
        -------
        Channels
        """
        new = self.__class__(name=self.get_name(),
                             parent=self.parent,
                             info=self.info)
        for attribute, value in self.__dict__.items():
            if attribute in self.reference_attributes:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                copied_value = value.copy()
                if attribute == 'data':
                    copied_value.channels = new
                setattr(new, attribute, copied_value)
            else:
                setattr(new, attribute, deepcopy(value))
        if new.info is not None:
            new.info.unlink_configuration()
        new.set_data(new.data)
        new.groups = None
        new.divisions = None
        new.modalities = None
        if (np.isfinite(self.overlap_point_size)
                and self.overlap_point_size > 0):
            new.overlap_point_size = np.nan
            new.calculate_overlaps(self.overlap_point_size)

        if self.is_initialized:
            new.is_initialized = False
            new.initialize()

        return new

    @property
    def reference_attributes(self):
        """
        Return attributes that should be referenced rather than copied.

        Returns
        -------
        set
        """
        return {'parent'}

    @property
    def flagspace(self):
        """
        Return the appropriate channel flag space.

        Returns
        -------
        Flags
        """
        return self.data.flagspace

    @property
    def configuration(self):
        """
        Return the configuration.

        Returns
        -------
        Configuration
        """
        return self.info.configuration

    @property
    def size(self):
        """
        Return the number of channels.

        Returns
        -------
        int
        """
        return self.data.size

    @property
    def sourceless_flags(self):
        """
        Return the flag marking a channel as having no source.

        Returns
        -------
        flag : enum.Enum
        """
        return self.flagspace.sourceless_flags()

    @property
    def non_detector_flags(self):
        """
        Return the flags marking a channel as a non-detector.

        Returns
        -------
        flag : enum.Enum
        """
        return self.flagspace.non_detector_flags()

    @property
    def n_store_channels(self):
        """
        Return the number of stored channels (total) in the instrument.

        Returns
        -------
        int
        """
        if self.info is None:
            return 0
        return self.info.instrument.n_store_channels

    @n_store_channels.setter
    def n_store_channels(self, value):
        """
        Set the number of stored channels.

        Parameters
        ----------
        value : int

        Returns
        -------
        None
        """
        if self.info is None:
            return
        self.info.instrument.n_store_channels = value

    @property
    def n_mapping_channels(self):
        """
        Return the number of mapping channels in the instrument.

        Returns
        -------
        int
        """
        if self.info is None:
            return 0
        return self.info.instrument.n_mapping_channels

    @n_mapping_channels.setter
    def n_mapping_channels(self, value):
        """
        Set the number of mapping channels.

        Parameters
        ----------
        value : int

        Returns
        -------
        None
        """
        if self.info is None:
            return
        self.info.instrument.n_mapping_channels = value

    @property
    def parent(self):
        """
        Return the channel instance owner.

        Returns
        -------
        object
        """
        if self.info is None:
            return None
        return self.info.parent

    @parent.setter
    def parent(self, owner):
        """
        set the channel instance owner.

        Parameters
        ----------
        owner : object
            The owner of the info.

        Returns
        -------
        None
        """
        self.set_parent(owner)

    def has_option(self, option):
        """
        Check whether an option is set in the configuration.

        Parameters
        ----------
        option : str
            The configuration option.

        Returns
        -------
        is_configured : bool
        """
        if self.configuration is None:
            return False
        return self.configuration.is_configured(option)

    def __getitem__(self, indices):
        """
        Return a selection of the channel data.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)

        Returns
        -------
        None
        """
        return self.data[indices]

    def set_parent(self, parent):
        """
        Set the parent object for channels.

        Parameters
        ----------
        parent : object

        Returns
        -------
        None
        """
        if self.info is None:
            return
        self.info.set_parent(parent)

    def set_data(self, data):
        """
        Set the channel data object.

        Parameters
        ----------
        data : ChannelData

        Returns
        -------
        None
        """
        self.data = data
        if self.data is not None:
            self.data.set_parent(self)
        # if self.groups is not None:
        #     for group_name, group in self.groups.values():
        #         new_group = group.copy()
        #         new_group.data = data
        #         self.groups[group_name] = new_group

    def set_info(self, info):
        """
        Load the static instrument settings, which are date-independent.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        self.info = info
        self.is_valid = False
        self.is_initialized = False
        self.startup_info = None

    def get_name(self):
        """
        Return the name of the channels.

        Returns
        -------
        name : str
        """
        if self.info is None:
            return None
        return self.info.get_name()

    def set_name(self, name):
        """
        Set the name for the channels.

        Parameters
        ----------
        name : str

        Returns
        -------
        None
        """
        if self.info is None:
            return
        self.info.set_name(name)

    def initialize(self):
        """
        Initializes channel groups, divisions, and modalities.

        Returns
        -------
        None
        """
        self.init_groups()
        self.init_divisions()
        self.init_modalities()
        self.is_initialized = True

    def validate_scan(self, scan):
        """
        Validate the channel data with a scan.

        The steps are:

        1. Load the channel data
        2. Set appropriate flags
        3. Initialize channel groups, divisions, and modalities.
        4. Apply the configuration options

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        self.startup_info = self.info.copy()
        self.load_channel_data()
        self.flag_channels()
        self.initialize()
        self.info.validate_scan(self)
        self.apply_configuration()
        self.census(report=True)
        self.is_valid = True

    def apply_configuration(self):
        """
        Apply the configuration options to the channels.

        Returns
        -------
        None
        """
        if self.configuration.get_bool('scramble'):
            self.scramble()

        if self.configuration.get_bool('flatweights'):
            self.flatten_weights()

        if self.configuration.get_bool('uniform'):
            self.data.set_uniform_gains()

        gain_noise = self.configuration.get_float('gainnoise', default=np.nan)
        if np.isfinite(gain_noise):
            random_noise = np.random.normal(loc=0.0, scale=1.0, size=self.size)
            self.data.gain *= 1 + (gain_noise * random_noise)

        if self.has_option('sourcegains'):
            log.debug("Incorporating pixel coupling correlated signal gains.")
            self.data.gain *= self.data.coupling
            self.data.coupling.fill(1.0)

        self.normalize_array_gains()

        if self.configuration.get_bool('source.fixedgains'):
            self.set_fixed_source_gains()
            log.debug("Will use static source gains.")

        if self.configuration.get_bool('jackknife.channels'):
            self.jackknife()

        self.data.spikes[:] = 0
        self.data.dof[:] = 1.0
        self.data.validate_weights()

    def jackknife(self):
        """
        Randomly inverts half of the channel couplings.

        Returns
        -------
        None
        """
        log.debug("JACKKNIFE: Randomly inverted channels in source.")
        random_half_channels = np.random.random(self.size) < 0.5
        self.data.coupling[random_half_channels] *= -1

    def flag_channels(self):
        """
        Flag certain channels.

        Returns
        -------
        None
        """
        if self.configuration.has_option('blind'):
            self.set_blind_channels(
                self.configuration.get_int_list(
                    'blind', is_positive=True, default=[]))
        if self.configuration.has_option('flag'):
            self.flag_channel_list(
                self.configuration.get_int_list(
                    'flag', is_positive=True, default=[]))
        flag_branch = self.configuration.get_branch('flag', default=None)
        if flag_branch is not None:
            self.flag_fields(flag_branch)
        self.data.set_flags('BLIND', indices=self.data.weight == 0)
        self.set_channel_flag_defaults()

    def set_channel_flag_defaults(self):
        """
        Sets data values based on currently set flags.

        Returns
        -------
        None
        """
        self.data.set_flag_defaults()

    def scramble(self):
        """
        Randomly shuffle position data.

        Returns
        -------
        None
        """
        log.warning("Scrambling pixel position data (noise map only)")
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        self.data.position = self.data.position[indices]

    def normalize_array_gains(self):
        """
        Normalize the relative channel gains in observing channels.

        Returns
        -------
        average_gain : float
            The average gain prior to normalization.
        """
        log.debug("Normalizing relative channel gains.")
        array = self.modalities.get('obs-channels')[0]
        return array.normalize_gains()

    def get_channel_data_instance(self):
        """
        Return a channel data instance for these channels.

        Returns
        -------
        ChannelData
        """
        if self.info is None:
            raise ValueError("Info must be set before acquiring channel data.")
        data_class = self.info.get_channel_data_class()
        return data_class(channels=self)

    def get_scan_instance(self):
        """
        Return a scan instance for these channels.

        Returns
        -------
        Scan
        """
        scan_class = self.info.get_scan_class()
        scan = scan_class(self.copy())
        scan.info.scan = scan
        return scan

    def read_scan(self, filename, read_fully=True):
        """
        Read a file and return a scan using these channels.

        Parameters
        ----------
        filename : str
            The file path to the scan.
        read_fully : bool, optional
            If `True`, fully read the scan.

        Returns
        -------
        Scan
        """
        scan = self.get_scan_instance()
        scan.read(filename, read_fully=read_fully)

        valid_integrations = []
        if scan.integrations is not None:
            for integration in scan.integrations:
                if integration.size > 0:
                    valid_integrations.append(integration)
        scan.integrations = valid_integrations
        return scan

    def get_pixel_count(self):
        """
        Return the number of pixels in the channels.

        Returns
        -------
        pixels : int
        """
        return self.data.get_pixel_count()

    def get_pixels(self):
        """
        Return the pixels.

        Returns
        -------
        ChannelData
        """
        return self.data.get_pixels()

    def get_perimeter_pixels(self, sections=None):
        """
        Return the pixels at the perimeter positions.

        To algorithm to determine perimeter pixels divides the pixel array into
        `sections` angular slices about the mean pixel position, and then
        returns the furthest pixel from the central position for that slice.
        If no pixels exist in that slice, the slice is excluded.

        Parameters
        ----------
        sections : int, optional
            The number of perimeter pixels to return (approximate).  If not
            supplied, will be determined from the configuration.

        Returns
        -------
        perimeter_pixels : ChannelGroup
        """
        if sections is None:
            if self.configuration.get_string(
                    'perimeter', default='none').lower() == 'auto':
                sections = int(np.ceil(7 * np.sqrt(self.size)))
            else:
                sections = self.configuration.get_int('perimeter', default=0)

        mapping_pixels = self.get_mapping_pixels(
            discard_flag=self.flagspace.sourceless_flags())
        if sections <= 0:
            return mapping_pixels

        centroid = mapping_pixels.position.mean()
        pi = np.pi * units.Unit('radian')
        da = (2 * pi / sections).to('degree')
        relative = Coordinate2D(mapping_pixels.position)
        relative.subtract(centroid)
        angle = relative.angle().to('degree')
        bins = np.floor((angle + pi) / da).decompose().value.astype(int)
        distance = relative.length

        perimeter_indices = np.full(sections, -1)
        for i in range(sections):
            check = np.nonzero(bins == i)[0]
            if check.size > 0:
                perimeter_indices[i] = check[np.argmax(distance[check])]
        perimeter_indices = np.unique(
            perimeter_indices[perimeter_indices >= 0])

        return mapping_pixels.create_data_group(indices=perimeter_indices,
                                                name='perimeter_pixels')

    def find_fixed_indices(self, fixed_indices, cull=True):
        """
        Returns the actual indices given fixed indices.

        The fixed indices are those that are initially loaded.  Returned
        indices are their locations in the data arrays.

        Parameters
        ----------
        fixed_indices : int or np.ndarray (int)
            The fixed indices.
        cull : bool, optional
            If `True`, do not include fixed indices not found in the result.
            If `False`, missing indices will be replaced by -1.

        Returns
        -------
        indices : numpy.ndarray (int)
            The indices of `fixed_indices` in the data arrays.  A tuple will
            be returned, in the case where we are examining more than one
            dimension.
        """
        return self.data.find_fixed_indices(fixed_indices, cull=cull)

    def get_division(self, name, field, keep_flag=None, discard_flag=None,
                     match_flag=None):
        """
        Creates a channel division based on a given data field.

        A single group will be created for each unique field value.  The name
        of each group will be <field>-<value>.  Channels may be discarded from
        any groups using the standard flagging mechanisms (keep, discard_flag,
        match).

        Parameters
        ----------
        name : str
            The name of the channel division.
        field : str
            The data field on which to base division creation.
        keep_flag : int or str or ChannelFlagTypes, optional
            If supplied, all groups in the division will only contain channels
            flagged with `keep_flag`.
        discard_flag : int or str or ChannelFlagTypes, optional
            If supplied, all groups in the division will not contain any
            channels flagged with `discard_flag`.
        match_flag : int or str or ChannelFlagTypes, optional
            If supplied, all groups in the division will only contain channels
            containing flags exactly matching `match_flag`.  Overrides
            `keep_flag` and `discard_flag`.

        Returns
        -------
        ChannelDivision
            The newly created channel division.
        """
        values = getattr(self.data, field, None)
        if values is None:
            raise ValueError(f"Data does not contain '{field}'.")

        indices = self.data.get_flagged_indices(keep_flag=keep_flag,
                                                discard_flag=discard_flag,
                                                match_flag=match_flag)
        mask = np.full(self.size, False)
        mask[indices] = True
        unique_values = np.unique(values)
        groups = []
        for unique_value in unique_values:
            group_name = f'{field}-{unique_value}'
            indices = np.nonzero(mask & (values == unique_value))[0]
            groups.append(
                self.create_channel_group(indices=indices, name=group_name))

        division = ChannelDivision(name)
        for group in groups:
            division.groups.append(group)

        return division

    def add_group(self, channel_group, name=None):
        """
        Add a channel group to the groups dictionary.

        Parameters
        ----------
        channel_group : Group
            A channel group.
        name : str, optional
            The name of the group.  If not supplied, will be determined from
            the name of the group.  If supplied, will set the group object name
            if not previously set.

        Returns
        -------
        None
        """
        if channel_group is None:
            return
        if not isinstance(channel_group, ChannelGroup):
            raise ValueError(
                f"The groups dictionary can only contain {ChannelGroup}. "
                f"Received {type(channel_group)}.")
        if name is None:
            name = channel_group.name
        elif channel_group.name is None:
            channel_group.name = name
        self.groups[name] = channel_group

    def add_division(self, channel_division):
        """
        Adds a channel division to the divisions dictionary.

        A channel division contains sets of groups.  Will also add any groups
        to the groups dictionary.

        Parameters
        ----------
        channel_division : ChannelDivision
            The channel division to add.

        Returns
        -------
        None
        """
        if channel_division is None:
            return
        if not isinstance(channel_division, ChannelDivision):
            raise ValueError(
                f"The divisions dictionary can only "
                f"contain {ChannelDivision}. "
                f"Received {type(channel_division)}.")

        self.divisions[channel_division.name] = channel_division
        for group in channel_division.groups:
            self.add_group(group)

    def add_modality(self, modality):
        """
        Add a modality to the modalities dictionary.

        Parameters
        ----------
        modality : Modality

        Returns
        -------
        None
        """
        if modality is None:
            return
        if not isinstance(modality, Modality):
            raise ValueError(
                f"The modalities dictionary can only contain {Modality}. "
                f"Received {type(modality)}.")
        self.modalities[modality.name] = modality

    def list_divisions(self):
        """
        Return a list of all division names.

        Returns
        -------
        list (str)
        """
        if self.divisions is None:
            return []
        return sorted(self.divisions.keys())

    def list_modalities(self):
        """
        Return a list of all modality names.

        Returns
        -------
        list (str)
        """
        if self.modalities is None:
            return []
        return sorted(self.modalities.keys())

    def flatten_weights(self):
        """
        Flattens weight according to gain^2.

        Does not include hardware gains.

        Returns
        -------
        None
        """
        log.debug("Switching to flag channel weights.")
        self.data.flatten_weights()

    def uniform_gains(self, field=None):
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
        self.data.set_uniform_gains(field=field)

    def set_fixed_source_gains(self):
        """
        Multiplies coupling by gain only if this has not been done before.

        Returns
        -------
        None
        """
        if self.fixed_source_gains:
            return
        self.data.coupling *= self.data.gain
        self.fixed_source_gains = True

    def get_channel_flag_key(self, prepend=''):
        """
        Return the channel flags and values.

        Parameters
        ----------
        prepend : str
            A string to prepend to all names.

        Returns
        -------
        dict
            A key containing key=name, value=value (int).
        """
        result = []
        for flag in self.flagspace.flags:
            letter = self.flagspace.flag_to_letter(flag)
            code = flag.value
            description = self.flagspace.descriptions[flag]
            result.append(f"{prepend}'{letter}' - {code} - {description}")

        return result

    def write_channel_data(self, filename, header=None):
        """
        Write the channel data to a file.

        Parameters
        ----------
        filename : str
            The file name to write to.
        header : str, optional
            An optional header line to write to the file.

        Returns
        -------
        None
        """
        with open(filename, 'w') as f:
            print("# SOFSCAN Pixel Data File", file=f)
            print("#", file=f)

            if header is not None:
                print(header, file=f)
                print("#", file=f)

            flag_key = self.get_channel_flag_key(prepend='# Flag ')
            for key_line in flag_key:
                print(key_line, file=f)
            print("#", file=f)

            self.set_standard_weights()
            data_strings = self.data.to_string().split('\n')
            self.set_sample_weights()

            data_header = f'# {data_strings[0]}'
            data_body = '\n'.join(data_strings[1:])
            print(data_header, file=f)
            print(data_body, file=f)

        log.info(f"Written {filename}")

    def print_correlated_modalities(self):
        """
        Print all correlated modalities to the log.

        Returns
        -------
        None
        """
        if not self.is_initialized:
            self.initialize()
        log.info(f"Available pixel divisions for "
                 f"{self.info.instrument.name}: ")
        for name, modality in self.modalities.items():
            if isinstance(modality, CorrelatedModality):
                configured = self.configuration.has_option(
                    f'correlated.{name}')
                log.info(f" {'(*)' if configured else '  '} {name}")

    def print_response_modalities(self):
        """
        Print all response modalities to the log.

        Returns
        -------
        None
        """
        self.print_correlated_modalities()

    def reindex(self):
        """
        Reset the channel indices to range sequentially from 0 -> size.

        Returns
        -------
        None
        """
        for group in self.groups.values():
            group.reindex()
            group.data = self.data

        for channel_division in self.divisions.values():
            if channel_division.groups is not None:
                for group in channel_division.groups:
                    if group is not None:
                        group.reindex()
                        group.data = self.data

        for modality in self.modalities.values():
            if modality.modes is not None:
                for mode in modality.modes:
                    if mode.channel_group is not None:
                        mode.channel_group.reindex()
                        mode.channel_group.data = self.data

    def slim(self, reindex=True):
        """
        Remove all DEAD or DISCARD flagged channels.

        Will also update channel groups, divisions, and modalities.

        Parameters
        ----------
        reindex : bool, optional
            If `True`, reindex channels if slimmed.

        Returns
        -------
        slimmed : bool
            `True` if channels were discarded, `False` otherwise.
        """
        old_size = self.size
        flag = self.flagspace.flags.DEAD | self.flagspace.flags.DISCARD
        self.discard(flag=flag, criterion='DISCARD_ALL')
        self.slim_groups()
        if self.size < old_size:
            log.debug(f"Slimmed to {self.size} live pixels.")
            if reindex:
                self.reindex()
            return True
        else:
            return False

    def slim_groups(self):
        """
        Remove invalid channels from channel groups, divisions, and modalities.

        Returns
        -------
        None
        """
        for channel_group in self.groups.values():
            self.slim_group(channel_group)

        for channel_division in self.divisions.values():
            for channel_group in channel_division.groups:
                self.slim_group(channel_group)

        for modality in self.modalities.values():
            for mode in modality.modes:
                self.slim_group(mode.channel_group)

    def slim_group(self, channel_group):
        """
        Remove indices from a channel group not present in the channel data.

        Parameters
        ----------
        channel_group : ChannelGroup

        Returns
        -------
        None
        """
        fixed_indices = np.intersect1d(self.data.fixed_index,
                                       channel_group.fixed_indices)
        # Setting the fixed indices also updates standard indexing (property)
        channel_group.fixed_indices = fixed_indices

    def discard(self, flag, criterion=None):
        r"""
        Given a flag, remove channels from data and dependent structures.

        Will also remove discarded channels from channel groups, divisions,
        and modalities.

        Parameters
        ----------
        flag : int or ChannelFlagTypes, optional
            The flag to discard_flag.
        criterion : str, optional
            One of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  \*_ANY refers to any flag
            that is not zero (unflagged).  \*_ALL refers to any flag that
            contains `flag`, and \*_MATCH refers to any flag that exactly
            matches `flag`.  The default (`None`), uses DISCARD_ANY if
            `flag` is None, and DISCARD_ALL otherwise.

        Returns
        -------
        None
        """
        self.remove(self.flagspace.discard_indices(self.data.flag, flag,
                                                   criterion=criterion))

    def load_temporary_hardware_gains(self):
        """
        Load the temporary hardware gains in the channel data.

        The hardware gains are also reset from the info.

        Returns
        -------
        None
        """
        self.data.set_hardware_gain(self.info)
        self.data.temp = self.data.hardware_gain.copy()

    def get_source_gains(self, filter_corrected=True):
        """
        Return the source gains.

        The source gains are taken from the coupling data.  If gains are not
        fixed (as determined by "source.fixedgains"), gains are multiplied by
        the channel gains.  It will also be multiplied by the source filter
        if `filter_corrected` is `True`.

        Parameters
        ----------
        filter_corrected : bool, optional
            Apply source filtering.

        Returns
        -------
        gains : numpy.ndarray (float)
            The source gains.
        """
        fixed_gains = self.configuration.get_bool('source.fixedgains')
        gain_factor = 1.0 if fixed_gains else self.data.gain
        gains = self.data.coupling * gain_factor
        if filter_corrected:
            gains *= self.data.source_filtering
        return gains

    def get_min_beam_fwhm(self):
        """
        Return the minimum beam FWHM.

        Returns
        -------
        fwhm : astropy.units.Quantity
        """
        return np.nanmin(self.data.resolution)

    def get_max_beam_fwhm(self):
        """
        Return the maximum beam FWHM.

        Returns
        -------
        fwhm : astropy.units.Quantity
        """
        return np.nanmax(self.data.resolution)

    def get_average_beam_fwhm(self):
        """
        Return the average beam FWHM.

        Returns
        -------
        fwhm : astropy.units.Quantity
        """
        return np.nanmean(self.data.resolution)

    def get_average_filtering(self):
        """
        Return the average source filtering.

        Returns
        -------
        filtering : float
        """
        source_gain = self.get_source_gains(filter_corrected=False)
        valid = self.data.is_unflagged()
        weight = self.data.weight[valid] * source_gain[valid] ** 2
        phi = self.data.source_filtering[valid]
        g = np.nansum(weight * phi)
        g2 = np.nansum(weight * (phi ** 2))
        return (g2 / g) if (g > 0) else 0.0

    def flag_weights(self):
        """
        Flag channels based on channel weights and gain.

        Only detector channels are used during flagging (based on noise
        weights, not source weights).  Channels with zero degrees of
        freedom are flagged with the DOF channel flag and unflagged when
        the degrees of freedom are greater than zero.

        The 'weighting.noiserange' configuration option sets the minimum and
        maximum allowable noise ranges for a channel in standard deviations.
        The mean value of the channel gain is taken from all detector channels
        that have sufficient DOF, weight, and gain.

        Those channels that fall outside the acceptable noise ranges are
        flagged with the SENSITIVITY channel flag, whereas any others will be
        unflagged.

        Errors will be raised if there are no channels with nonzero DOF, or
        all channels are flagged as insensitive.

        Returns
        -------
        None
        """
        weight_range = Range()
        if self.configuration.has_option('weighting.noiserange'):
            noise_range = self.configuration.get_range('weighting.noiserange',
                                                       is_positive=True)
            weight_range.min = (1.0 / (noise_range.max ** 2))
            if noise_range.min > 0:
                weight_range.max = (1.0 / (noise_range.min ** 2))

        channels = self.get_detector_channels()
        valid_points, sum_wg2, channels.flag = cnf.flag_weights(
            channel_gain=channels.gain,
            channel_weight=channels.weight,
            channel_dof=channels.dof,
            channel_flags=channels.flag,
            min_weight=weight_range.min,
            max_weight=weight_range.max,
            exclude_flag=(self.flagspace.hardware_flags()
                          | self.flagspace.flags.GAIN).value,
            dof_flag=self.flagspace.convert_flag('DOF').value,
            sensitivity_flag=self.flagspace.convert_flag('SENSITIVITY').value,
            default_weight=channels.default_field_types['weight'])

        if valid_points == 0:
            log.warning("Degrees of freedom: No valid channels.")
        elif sum_wg2 == 0:
            if self.n_mapping_channels == 0:
                log.warning("No mapping channels.")
            else:
                log.warning("All channels flagged.")

        self.census()

    def get_source_nefd(self, gain=1.0):
        """
        Returns the source Noise-Equivalent-Flux-Density (NEFD).

        The NEFD is given as:

        nefd = sqrt(n * t / sum(g^2 / v)) / abs(integration_gain)

        where n are the number of mapping channels (unflagged and positive
        weights), t is the integration time, g is the source gain, and v is the
        channel variance.  Note that t will be converted to units of seconds if
        necessary.

        Parameters
        ----------
        gain : float, optional
            Normalizing gain value.  The default is 1.0.

        Returns
        -------
        source_nefd : float
        """
        return cnf.get_source_nefd(
            filtered_source_gains=self.get_source_gains(filter_corrected=True),
            weight=self.data.weight,
            variance=self.data.variance,
            flags=self.data.flag,
            integration_time=self.info.integration_time.to('second').value,
            integration_gain=gain)

    def get_stability(self):
        """
        Return the instrument stability in seconds.

        Will look in the configuration for "stability", and if not found, will
        return 10 seconds.

        Returns
        -------
        stability_time : astropy.units.Quantity
            The stability time in seconds.
        """
        return (self.configuration.get_float('stability', 10.0)
                * units.Unit('s'))

    def get_one_over_f_stat(self):
        """
        Return the 1/frequency statistic (pink noise) of the channels.

        Returns
        -------
        float
            The 1/frequency statistic.
        """
        channels = self.get_observing_channels()
        return cnf.get_one_over_f_stat(
            weights=channels.weight,
            one_over_f_stats=channels.one_over_f_stat,
            flags=channels.flag)

    def get_fits_data(self, data):
        """
        Add channel data to the FITS data.

        Parameters
        ----------
        data : dict

        Returns
        -------
        None
        """
        gains = np.full(self.n_store_channels, np.nan)
        weights = np.full(self.n_store_channels, np.nan)
        flags = np.full(self.n_store_channels, self.flagspace.flags.DEAD.value)

        self.set_standard_weights()
        gains[self.data.fixed_index] = self.data.gain
        weights[self.data.fixed_index] = self.data.weight
        flags[self.data.fixed_index] = self.data.flag
        self.set_sample_weights()

        data['Channel_Gains'] = gains
        data['Channel_Weights'] = weights
        data['Channel_Flags'] = flags

    def edit_scan_header(self, header, scans=None, configuration=None):
        """
        Edit a scan header with available information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        scans : list (Scan), optional
            A list of scans to use during editing.
        configuration : Configuration, optional
            A different configuration to use if necessary

        Returns
        -------
        None
        """
        if configuration is None:
            configuration = self.configuration
        if configuration.get_bool('write.scandata.details'):
            self.flagspace.edit_header(header, prefix='C')

    def calculate_overlaps(self, point_size=None):
        """
        Calculate channel overlaps.

        Parameters
        ----------
        point_size : astropy.units.Quantity, optional
            The overlap point size (beam FWHM for example).  The default
            is the instrument spatial resolution.

        Returns
        -------
        None
        """
        if point_size is None:
            point_size = self.info.resolution
        elif isinstance(point_size, units.Quantity):
            point_size = point_size.to(self.info.size_unit)
        else:
            point_size = point_size * self.info.size_unit

        if point_size == self.overlap_point_size:
            return  # don't need to do anything if already calculated.

        self.data.calculate_overlaps(point_size)
        self.overlap_point_size = point_size

    def get_table_entry(self, name):
        """
        Return a channel parameter for the given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        size_unit = self.info.size_unit
        if name == 'gain':
            return self.info.gain
        elif name == 'sampling':
            return self.info.sampling_interval.to('second').value
        elif name == 'rate':
            return (1.0 / self.info.sampling_interval).to('Hz').value
        elif name == 'okchannels':
            return self.n_mapping_channels
        elif name == 'maxchannels':
            return self.n_store_channels
        elif name == 'mount':
            return self.info.instrument.mount.name
        elif name == 'resolution':
            return self.info.resolution.to(size_unit).value
        elif name == 'sizeunit':
            return size_unit.name
        elif name == 'ptfilter':
            return self.get_average_filtering()
        elif name == 'FWHM':
            return self.get_average_beam_fwhm().to(size_unit).value
        elif name == 'minFWHM':
            return self.get_min_beam_fwhm().to(size_unit).value
        elif name == 'maxFWHM':
            return self.get_max_beam_fwhm().to(size_unit).value
        elif name == 'stat1f':
            return self.get_one_over_f_stat()
        else:
            return None

    def __str__(self):
        """
        Return a string representing the channels.

        Returns
        -------
        str
        """
        return f"Instrument {self.get_name()}"

    def troubleshoot_few_pixels(self):
        """
        Return suggestions as to how to remedy too few pixels in the model.

        Returns
        -------
        suggestions : list (str)
        """
        suggestions = [
            ' * Disable gain estimation for one or more modalities. E.g.:']

        for modality in self.list_modalities():
            if not self.configuration.has_option(f'correlated.{modality}'):
                continue
            if self.configuration.get_bool(f'correlated.{modality}.nogains'):
                continue
            suggestions.append(f'\t-correlated.{modality}.nogains=True')

        if self.configuration.get_bool('gains'):
            suggestions.append(" * Disable gain estimation globally with "
                               "'-forget=gains'.")
        if self.configuration.has_option('despike'):
            suggestions.append(" * Disable despiking with '-forget=despike'.")
        if self.configuration.has_option('weighting.noiseRange'):
            suggestions.append(" * Adjust noise flagging via "
                               "'weighting.noiseRange'.")
        return suggestions

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
        self.data.flag_field(field, specs)

    def flag_fields(self, fields):
        """
        Flag channel data by fields.

        Parameters
        ----------
        fields : dict
            The fields dict should have the field names for keys (attributes
            in channel data to flag) and channel ranges for values.  values
            may be in the form lower-upper, lower:upper.  Any channels with
            that field attribute in the given range will be flagged as DEAD.

        Returns
        -------
        None
        """
        if not isinstance(fields, dict):
            return
        for field, specs in fields.items():
            self.data.flag_field(field, specs)

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
        self.data.flag_channel_list(channel_list)

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
        self.data.kill_channels(flag=flag)

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
        self.data.set_blind_channels(fixed_indices)

    def load_channel_data(self):
        """
        Load the channel data.

        The pixel data and wiring data files should be defined in the
        configuration.

        Returns
        -------
        None
        """
        if 'pixeldata' in self.configuration:
            pixel_data_file = self.configuration.priority_file('pixeldata')
            if pixel_data_file is None:
                log.error(f"Pixel data file "
                          f"{self.configuration.get('pixeldata')} not found.")
                log.warning("Cannot read pixel data. "
                            "Using default gains and flags.")
            else:
                self.read_pixel_data(pixel_data_file)

        if 'wiring' in self.configuration:
            wiring_data_file = self.configuration.priority_file('wiring')
            if wiring_data_file is None:
                log.warning(f"Wiring data file "
                            f"{self.configuration.get('wiring')} not found.")
                log.warning("Cannot read wiring data. "
                            "Specific channel divisions not established.")
            else:
                self.read_wiring_data(wiring_data_file)

    def read_pixel_data(self, filename):
        """
        Read the pixel data file.

        If the instrument integration time is greater than zero, will set
        weighting accordingly.  Otherwise, standard weights are used.

        Parameters
        ----------
        filename : str
            Path to the pixel data file.

        Returns
        -------
        None
        """
        log.info(f"Loading pixel data from {filename}")

        self.data.read_pixel_data(filename)
        self.standard_weights = True
        if self.info.instrument.integration_time > 0:
            self.set_sample_weights()

    def read_wiring_data(self, filename):
        """
        Read the wiring data.

        Parameters
        ----------
        filename : str
            Path to the wiring data file.

        Returns
        -------
        None
        """
        pass

    def set_standard_weights(self):
        """
        Set standard weighting.

        If not already applied, divides the channels weights by
        sqrt(integration_time)

        Returns
        -------
        None
        """
        if self.standard_weights:
            return
        time = self.info.instrument.integration_time
        if isinstance(time, units.Quantity):
            time = time.decompose().value
        self.data.weight /= np.sqrt(time)
        self.standard_weights = True

    def set_sample_weights(self):
        """
        Set the sample weighting.

        If not already applied, multiplies the channel weighting by
        sqrt(integration_time).

        Returns
        -------
        None
        """
        if not self.standard_weights:
            return
        time = self.info.instrument.integration_time
        if isinstance(time, units.Quantity):
            time = time.decompose().value
        self.data.weight *= np.sqrt(time)
        self.standard_weights = False

    def remove(self, indices):
        """Remove channel indices and dependent data.

        In addition to removing channels from the channel data, also remove
        those channel references from channel groups, divisions, and
        modalities.

        Parameters
        ----------
        indices : numpy.ndarray of int

        Returns
        -------
        None
        """
        if len(indices) == 0:
            return
        self.data.delete_indices(indices)
        self.slim_groups()

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
        return self.data.get_typical_gain_magnitude(gains,
                                                    keep_flag=keep_flag,
                                                    discard_flag=discard_flag,
                                                    match_flag=match_flag)

    def get_observing_channels(self):
        """
        Returns the 'obs-channels' channel group.

        Returns
        -------
        obs_channels : ChannelGroup
        """
        if self.groups is None:
            return None
        return self.groups.get('obs-channels')

    def get_live_channels(self):
        """
        Returns the 'live' channel group.

        Returns
        -------
        live_channels : ChannelGroup
        """
        if self.groups is None:
            return None
        return self.groups.get('live')

    def get_detector_channels(self):
        """
        Returns the 'detectors' channel group.

        Returns
        -------
        detector_channels : ChannelGroup
        """
        if self.groups is None:
            return None
        return self.groups.get('detectors')

    def get_sensitive_channels(self):
        """
        Returns the 'sensitive' channel group.

        Returns
        -------
        sensitive_channels : ChannelGroup
        """
        if self.groups is None:
            return None
        return self.groups.get('sensitive')

    def get_blind_channels(self):
        """
        Returns the 'blinds' channel group.

        Returns
        -------
        blind_channels : ChannelGroup
        """
        if self.groups is None:
            return None
        return self.groups.get('blinds')

    def census(self, report=False):
        """
        Update the number of available mapping channels.

        Parameters
        ----------
        report : bool, optional
            If `True`, produce a log message for the number of mapping channels
            and number of mapping pixels.

        Returns
        -------
        None
        """
        if self.modalities is None:
            log.error("Modalities are not initialized.")
            obs_channels = None
        else:
            obs_channels = self.get_observing_channels()

        if obs_channels is None:
            self.n_mapping_channels = 0
        else:
            self.n_mapping_channels = np.sum(
                obs_channels.is_unflagged() & (obs_channels.weight > 0))

        if not report:
            return

        mapping_pixels = self.get_mapping_pixels(
            discard_flag=self.flagspace.sourceless_flags())

        log.debug(f"Mapping channels: {self.n_mapping_channels}")
        log.debug(f"Mapping pixels: {mapping_pixels.size}")

    def create_channel_group(self, indices=None, name=None, keep_flag=None,
                             discard_flag=None, match_flag=None):
        """
        Creates and returns a channel group.

        A channel group is a referenced subset of the channel data.  Operations
        performed on a channel group will be applied to the original channel
        data.

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
        if indices is None:
            indices = np.arange(self.size)

        for flag in [keep_flag, discard_flag, match_flag]:
            if flag is not None:
                flag_indices = self.data.get_flagged_indices(
                    keep_flag=keep_flag, discard_flag=discard_flag,
                    match_flag=match_flag)
                indices = np.intersect1d(indices, flag_indices)
                break

        group_class = self.info.get_channel_group_class()
        return group_class(self.data, indices=indices, name=name)

    def init_groups(self):
        """
        Initializes channel groups.

        Each group contains a subset of the channel data, referenced by index.

        Returns
        -------
        None
        """
        flags = self.flagspace.flags
        self.groups = dict()
        non_detector_flag = self.flagspace.non_detector_flags()
        self.add_group(self.create_channel_group(
            indices=np.arange(self.size), name='all'))

        live_indices = self.data.is_unflagged(flags.DEAD, indices=True)
        self.add_group(self.create_channel_group(
            indices=live_indices, name='live'))

        detector_indices = self.data.is_unflagged(non_detector_flag,
                                                  indices=True)
        self.add_group(
            self.create_channel_group(indices=detector_indices,
                                      name='detectors'))

        obs_indices = self.data.is_unflagged(non_detector_flag | flags.BLIND,
                                             indices=True)
        self.add_group(self.create_channel_group(indices=obs_indices,
                                                 name='obs-channels'))

        sensitive_indices = self.data.is_unflagged(
            non_detector_flag | flags.BLIND | flags.SENSITIVITY,
            indices=True)
        self.add_group(self.create_channel_group(
            indices=sensitive_indices, name='sensitive'))

        blinds = self.data.is_unflagged(non_detector_flag)
        blinds &= self.data.is_flagged(flags.BLIND)
        self.add_group(self.create_channel_group(
            indices=np.nonzero(blinds)[0], name='blinds'))

        add_groups = self.configuration.get_branch('group')
        if not isinstance(add_groups, dict):
            return

        for group_name, channel_strings in add_groups.items():
            fixed_indices = utils.get_int_list(channel_strings, default=None)
            if fixed_indices is None:
                raise ValueError(
                    f"Could not parse group: {group_name}={channel_strings}")
            indices = self.data.find_fixed_indices(fixed_indices, cull=True)
            indices = np.unique(indices)
            indices = indices[self.data.is_unflagged(flags.DEAD)[indices]]

            self.add_group(self.create_channel_group(indices=indices,
                                                     name=group_name))

    def init_divisions(self):
        """
        Initializes channel divisions.

        Divisions contain sets of channel groups.

        Returns
        -------
        None
        """
        self.divisions = dict()
        self.add_division(ChannelDivision('all', self.groups.get('all')))
        self.add_division(ChannelDivision('live', self.groups.get('live')))
        self.add_division(
            ChannelDivision('detectors', self.groups.get('detectors')))
        self.add_division(
            ChannelDivision('obs-channels', self.groups.get('obs-channels')))
        self.add_division(
            ChannelDivision('sensitive', self.groups.get('sensitive')))
        self.add_division(ChannelDivision('blinds', self.groups.get('blinds')))

        add_divisions = self.configuration.get_keys('division')
        if add_divisions is None or len(add_divisions) == 0:
            return

        for division_name in add_divisions:
            groups = []
            group_names = self.configuration.get_list(
                f'division.{division_name}')
            for group_name in group_names:
                if group_name not in self.groups:
                    log.warning(f"Channel group {group_name} is undefined for "
                                f"division {division_name}.")
                else:
                    groups.append(self.groups[group_name])
            self.add_division(ChannelDivision(division_name, groups))

    def init_modalities(self):
        """
        Initializes channel modalities.

        A modality is based of a channel division and contains a mode for each
        channel group in the channel division.

        Returns
        -------
        None
        """
        self.modalities = dict()

        # Begin with correlated modalities
        correlated_modalities_descriptions = [
            ('all', 'Ca', 'all', 'gain'),
            ('live', 'Cl', 'live', 'gain'),
            ('detectors', 'Cd', 'detectors', 'gain'),
            ('obs-channels', 'C', 'obs-channels', 'gain'),
            ('coupling', 'Cc', 'obs-channels', 'coupling'),
        ]
        for description in correlated_modalities_descriptions:
            name, identity, division_name, gain_field = description
            division = self.divisions.get(division_name)
            if division is None:
                log.warning(f"Could not create modality from {division_name}: "
                            f"division does not exist.")
                continue
            self.add_modality(CorrelatedModality(name=name,
                                                 identity=identity,
                                                 channel_division=division,
                                                 gain_provider=gain_field))

        obs_modality = self.modalities.get('obs-channels')
        if obs_modality is not None:
            self.add_modality(
                CoupledModality(modality=obs_modality,
                                name='sky',
                                identity='Cs',
                                gain_provider='coupling'))

            self.add_modality(
                NonlinearModality(modality=obs_modality,
                                  name='nonlinearity',
                                  identity='n',
                                  gain_provider='nonlinearity'))

        # Pointing and response modes
        # Note that the actual applied motion is dependent on the identifier
        # in the name after the '-'.  i.e., ...-x means the x direction.
        response_descriptions = [
            ('telescope-x', 'Tx', PointingResponse),
            ('telescope-y', 'Ty', PointingResponse),
            ('accel-x', 'ax', AccelerationResponse),
            ('accel-y', 'ay', AccelerationResponse),
            ('accel-x^2', 'axs', AccelerationResponse),
            ('accel-y^2', 'ays', AccelerationResponse),
            ('accel-|x|', 'a|x|', AccelerationResponse),
            ('accel-|y|', 'a|y|', AccelerationResponse),
            ('accel-mag', 'am', AccelerationResponse),
            ('accel-norm', 'an', AccelerationResponse),
            ('chopper-x', 'cx', ChopperResponse),
            ('chopper-y', 'cy', ChopperResponse),
            ('chopper-x^2', 'cxs', ChopperResponse),
            ('chopper-y^2', 'cys', ChopperResponse),
            ('chopper-|x|', 'c|x|', ChopperResponse),
            ('chopper-|y|', 'c|y|', ChopperResponse),
            ('chopper-mag', 'cm', ChopperResponse),
            ('chopper-norm', 'cn', ChopperResponse)
        ]

        division = self.divisions.get('detectors')
        for (name, identity, mode_class) in response_descriptions:
            self.add_modality(Modality(name=name,
                                       identity=identity,
                                       channel_division=division,
                                       mode_class=mode_class))

        # Add a blind modality if necessary
        if self.configuration.has_option('blind'):
            division = self.divisions.get('blinds')
            if division is not None:
                if 'temperature_gain' not in division[0].fields:
                    log.warning(f"{division[0]} has no 'temperature_gain' "
                                f"field for blind correction.")
                else:
                    blind_modality = CorrelatedModality(
                        name='blinds',
                        identity='Bl',
                        channel_division=division,
                        gain_provider='temperature_gain')
                    for mode in blind_modality.modes:
                        mode.skip_flags = mode.flagspace.flags.DEAD
                    self.add_modality(blind_modality)

        # Set the correct gain flags for certain modalities.
        for modality_name in ['all', 'live', 'detectors', 'obs-channels']:
            modality = self.modalities.get(modality_name)
            if modality is None:
                continue
            modality.set_gain_flag(modality.flagspace.flags.GAIN)

        division_options = self.configuration.get_branch('division')
        if division_options is not None:
            for division_name, options in division_options.items():
                division = self.divisions.get(division_name)
                if division is None:
                    log.warning(f"Configuration division {division_name} "
                                f"does not exist.")
                    continue
                elif division.size == 0:
                    log.warning(f"Configuration division {division_name} "
                                f"does not contain any channel groups.")
                    continue
                identity = options.get('id', division_name)
                gain_field = options.get('gainfield')
                if gain_field is not None:
                    gain_field = str(gain_field)
                    if gain_field not in division.fields:
                        log.warning(
                            f"Configuration division {division_name} "
                            f"does not contain {gain_field} gain field.")
                        continue
                self.add_modality(
                    CorrelatedModality(name=division_name, identity=identity,
                                       gain_provider=gain_field,
                                       channel_division=division))
                if 'gainflag' in options:
                    try:
                        gain_flag = int(options['gainflag'])
                    except ValueError:
                        gain_flag = options['gainflag']

                    self.modalities[division_name].set_gain_flag(gain_flag)

    def get_mapping_pixels(self, discard_flag=None, keep_flag=None,
                           match_flag=None):
        """
        Return the mapping pixels.

        Parameters
        ----------
        discard_flag : int or str or ChannelFlagTypes
        keep_flag : int or str or ChannelFlagTypes
        match_flag : int or str or ChannelFlagTypes

        Returns
        -------
        ChannelGroup
        """
        return self.data.get_mapping_pixels(discard_flag=discard_flag,
                                            keep_flag=keep_flag,
                                            match_flag=match_flag)

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
        self.data.remove_dependents(dependents)

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
        self.data.add_dependents(dependents)

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
        return self.data.get_filtering(integration)
