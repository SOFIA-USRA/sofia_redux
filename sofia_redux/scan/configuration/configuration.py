# Licensed under a 3-clause BSD style license - see LICENSE.rst

from configobj import ConfigObj
from copy import deepcopy
import os
from astropy import log
import fnmatch
import json

from sofia_redux import scan
from sofia_redux.scan.configuration.options import Options
from sofia_redux.scan.configuration.aliases import Aliases
from sofia_redux.scan.configuration.conditions import Conditions
from sofia_redux.scan.configuration.dates import DateRangeOptions
from sofia_redux.scan.configuration.fits import FitsOptions
from sofia_redux.scan.configuration.iterations import IterationOptions
from sofia_redux.scan.configuration.objects import ObjectOptions
from sofia_redux.scan.utilities.utils import (
    dict_difference, get_range, insert_info_in_header)

__all__ = ['Configuration']


class Configuration(Options):
    r"""
    A handler for configuration settings used during a SOFSCAN reduction.

    The configuration for a SOFSCAN reduction is highly complex and can
    involve hundreds of specific settings based on user intent or the
    contents of the data being processed.

    Notes
    -----

    *CONDITIONS*

    How a reduction should proceed are handled by "conditionals".  Conditions
    are used to perform a set of actions or update the configuration when a
    specific requirement is met.  An example of such is::

        [conditionals]
            [[fits.SIBS_X=15.5]]
                subarray = T0, R0

    This means that if the SIBS_X keyword value in the FITS header for a
    certain scan is equal to 15.5, then the value of subarray in the
    configuration should be set to ['T0', 'R0'].  The first level of
    conditional keys should always be a requirement of the form
    <key><operator><value> where <operator> may be replaced by any standard
    Python conditional operator such as =, ==, !=, <, etc.

    For SOFSCAN reductions, there are a few types of condition classes that
    specifically relate to settings for a given SOFSCAN iteration, the date of
    observation, the source being observed, and the values in the FITS
    header.  Please see :class:`DateRangeOptions`, :class:`FitsOptions`,
    :class:`IterationOptions, and :class:`ObjectOptions` for further
    details.

    Note that all processing conditions is necessarily recursive since
    triggering one condition may allow other condition requirements to be
    met and so on.  This is done during configuration validation, which
    will indefinitely check and apply conditions until no further changes
    are detected.  Therefore, care should be taken when designing a
    configuration such that no infinite loops are created.

    *ALIASES*

    A feature of the configuration is the ability to alias keys and
    reference values through the :class:`Alias` class.  A key may be aliased
    using something like::

        [aliases]
            pols = correlated.polarrays
            subs = correlated.subarrays

    which means that the "pols" key in the configuration will always refer
    to the polarrays branch of the correlated options.  Values may be
    referenced in place by using {?<key>} notation, where <key> refers to
    any other existing key/value in the configuration.  For example,
    jumpdata = {?configpath}/hawc_plus/flux_jump_FS15_v3.fits.gz would
    replace {?configpath} with the correct value when retrieved from the
    configuration.

    *COMMANDS*

    There are certain keys in the configuration that are used to perform
    specific actions or set attributes for certain key/values.  These are
    never passed into the set of key/values in the configuration and are
    processed separately.  There are:

    - blacklist :
        Never allow access to this key for any reason.  It may not
        be altered, retrieved, or made visible to the SOFSCAN reduction.  A
        blacklisted key should remain so for the entire reduction.
    - whitelist :
        Always allow access to this key and never allow any
        modification to its value for the entire reduction.
    - forget :
        Temporarily disable access to this keys value by the SOFSCAN
        reduction.  Access may be granted by using the "recall" command.
    - recall :
        Allow access to a previously forgotten key.
    - lock :
        Do not allow any further modifications to they key values or
        attributes.
    - unlock :
        Unlock a previously locked key.
    - add :
        Add a key to the configuration.  This will set the value of this
        key to "True" when retrieved by the reduction.
    - rounds :
        Set the maximum number of iterations for the SOFSCAN reduction.
        The value must be an integer (or reference an integer).
    - config :
        Read and merge the contents of another configuration whose
        file path is set as the value.

    *SYNTAX*

    Configuration files should be formatted according to
    `configobj`.  A shorthand format may be used to reference
    multiple configuration levels using a dot separator.  For example,
    correlated.sky.gains.range = 0.3:3 is the same as setting::

        [correlated]
            [[sky]]
                [[[gains]]]
                    range = 0.3:3

    The "fits" key is special and is used to refer to key/values in the
    FITS header.  For example, fits.SPECTEL1 will refer to the SPECTEL1
    value in the header, not the configuration.
    """

    # The command keys are those found in the configuration contents that
    # instruct the configuration to perform some task.  All other keys/values
    # are parsed as configuration options.
    command_keys = ('blacklist', 'whitelist', 'forget', 'recall',
                    'lock', 'unlock', 'add', 'rounds', 'config')

    # In certain circumstances, a configuration condition may be triggered that
    # issues multiple commands to the main configuration object.  The order
    # that those commands are applied is set here, and is very important to
    # fulfilling the intent of the user.  For example, if an update to a locked
    # value is required before it is locked again, it should be unlocked,
    # updated, and then locked.
    command_order = ('blacklist', 'whitelist', 'unlock', 'config', 'update',
                     'recall', 'rounds', 'add', 'forget', 'lock')

    # These are the keys that mark all value contents as belonging to a
    # particular attribute of the Configuration rather than the configuration
    # body.  For example, "aliases" should not be added as an option in the
    # configuration, but instead be passed to the Aliases object to be used
    # while processing all other settings.
    section_keys = {'aliases', 'conditionals', 'date', 'iteration', 'object'}
    handler_keys = section_keys.union(['fits'])

    def __init__(self, configuration_path=None, allow_error=False,
                 verbose=False):
        """
        Initialize a Configuration object for use with a sofia_scan Reduction.

        Parameters
        ----------
        configuration_path : str, optional
            A path to the SOFSCAN configuration directory to be used in the
            reduction.  If not supplied, defaults to
            <package_path>/data/configurations.
        allow_error : bool, optional
            If `True`, allow poorly formatted options to be skipped rather than
            raising an error.
        verbose : bool, optional
            If `True`, issues a warning when a poorly specified option is
            encountered.
        """
        super().__init__(allow_error=allow_error, verbose=verbose)
        self.instrument_name = None
        self.enabled = False
        self.config_files = []
        self.aliases = Aliases()
        self.conditions = Conditions()
        self.dates = DateRangeOptions()
        self.iterations = IterationOptions()
        self.objects = ObjectOptions()
        self.fits = FitsOptions()
        self.current_source = None
        self.current_date = None
        self.locked = set()
        self.disabled = set()
        self.whitelisted = set()
        self.forgotten = set()
        self.applied_conditions = set()
        self.work_path = None
        self.set_error_handling(allow_error)
        self.set_verbosity(verbose)

        if configuration_path is None:
            package_path = os.path.dirname(os.path.abspath(scan.__file__))
            configuration_path = os.path.join(
                package_path, 'data', 'configurations')

        self.config_path = configuration_path
        if not os.path.isdir(self.config_path):
            self.handle_error(
                f"Configuration directory not found ({self.config_path})",
                error_class=NotADirectoryError)
            self.options['configpath'] = None
            return
        self.options['configpath'] = configuration_path

    def copy(self):
        """
        Return a copy of the configuration.

        Returns
        -------
        Configuration
        """
        return super().copy()

    def clear(self):
        """
        Clear all options, handler options, and settings.

        Returns
        -------
        None
        """
        super().clear()
        # Preserve the configuration path.
        self.options['configpath'] = self.config_path
        self.locked = set()
        self.disabled = set()
        self.whitelisted = set()
        self.forgotten = set()
        self.applied_conditions = set()
        self.config_files = []
        for handler in [self.aliases, self.conditions, self.dates,
                        self.iterations, self.objects, self.fits]:
            handler.clear()

    def __contains__(self, key):
        """
        Check if a key is available in the configuration.

        Note that disabled (forgotten, blacklisted) keys will be returned as
        `False`.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        bool
        """
        uk = self.aliases(key)

        split_key = uk.split('.')
        check_string = ''
        for i, add_fork in enumerate(split_key):
            if i == 0:
                check_string = add_fork
            else:
                check_string += f'.{add_fork}'

            if check_string in self.disabled:
                return False

        return self.exists(uk, options=self.options)

    def __delitem__(self, key):
        """
        Delete a key from the options.

        Parameters
        ----------
        key : str or object

        Returns
        -------
        None
        """
        if self.options is None:
            raise KeyError("Options are not initialized.")
        self.purge(key)

    def exists(self, key, options=None):
        """
        Check if a given key exists in the configuration, even if disabled.

        Parameters
        ----------
        key : str
            The key to check.
        options : dict or ConfigObj, optional
            The configuration options to check.  If not supplied, uses the
            current configuration options.

        Returns
        -------
        bool
        """
        if options is None:
            options = self.options
        if key in options:
            return True
        paths = self.aliases(key).split('.')
        for path in paths:
            if not isinstance(options, dict):
                return False
            options = options.get(path)
            if options is None:
                return False
        return True

    def get(self, *args, default=None, unalias=True):
        """
        Retrieve a value from the configuration.

        Parameters
        ----------
        args : tuple (str)
            The configuration key levels such as
            ['correlated', 'sky', 'biasgains'].
        default : object, optional
            The value to return if not found in the configuration.
        unalias : bool, optional
            If `True`, unalias all keys provided in `args`.

        Returns
        -------
        value : str or object
            The returned string value if found in the configuration, or
            `default` if not found.
        """
        level = self.get_branch(*args, default=default, unalias=unalias)
        if isinstance(level, dict):
            result = level.get('value', default)
        else:
            result = level  # is a final value
        return result

    def get_branch(self, *args, default=None, unalias=True):
        """
        Retrieve a configuration options branch.

        Parameters
        ----------
        args : tuple (str)
            The configuration key levels such as ['correlated', 'sky'].
        default : object, optional
            The value to return if the branch was not found in the
            configuration.
        unalias : bool, optional
            If `True`, unalias all keys provided in `args` and any retrieved
            configuration values.

        Returns
        -------
        branch : dict or str or object
            The configuration branch if present or `default` if not.
        """
        key = self.aliases('.'.join(args))
        if self.is_disabled(key):
            return default

        branches = key.split('.')
        level = self.options
        for branch in branches:
            if branch in level:
                level = level[branch]
            else:
                if unalias:
                    return self.aliases.unalias_branch_values(self, default)
                else:
                    return default
        if unalias:
            return self.aliases.unalias_branch_values(self, level)
        else:
            return level

    def set_error_handling(self, allow_error):
        """
        Set the error handling for all section handlers.

        Parameters
        ----------
        allow_error : bool
            `True` if errors are permitted during configuration parsing.

        Returns
        -------
        None
        """
        self.allow_error = allow_error
        for handler in [self.aliases, self.conditions, self.dates,
                        self.iterations, self.objects, self.fits]:
            handler.allow_error = allow_error

    def set_verbosity(self, verbose):
        """
        Set verbose messages, and do the same for all section handlers.

        Parameters
        ----------
        verbose : bool
            If `True`, will emit certain log messages that may otherwise be
            slightly annoying.

        Returns
        -------
        None
        """
        self.verbose = verbose
        for handler in [self.aliases, self.conditions, self.dates,
                        self.iterations, self.objects, self.fits]:
            handler.verbose = verbose

    def get_section_handler(self, section_name):
        """
        Return the configuration handler object for a specific section.

        Note that the FitsOptions handler will not be returned since that
        configuration processing is handled in a different manner.

        Parameters
        ----------
        section_name : str
            The name of the configuration handler to retrieve

        Returns
        -------
        handler : Options or None
            Either a :class:`Aliases`, :class:`Conditions`,
            :class:`DateRangeOptions`, :class:`IterationsOptions`, or None if
            the section name does not reference a valid handler.
        """
        section_name = str(section_name).lower().strip()
        if section_name == 'aliases':
            return self.aliases
        elif section_name == 'conditionals':
            return self.conditions
        elif section_name == 'date':
            return self.dates
        elif section_name == 'iteration':
            return self.iterations
        elif section_name == 'object':
            return self.objects
        else:
            return None

    def parse_to_section(self, key, value):
        """
        Parse a key-value as a section update.

        Returns `True` if the key maps to a valid section in the configuration.

        Parameters
        ----------
        key : str
            The key to attempt to parse to a section of the configuration.
        value : dict or str
            The values to parse to the section.

        Returns
        -------
        parsed_as_section : bool
            `True` if the key-value successfully mapped to a section of the
            configuration other than the main body, and `False` otherwise.
        """
        section_handler = self.get_section_handler(key)
        if section_handler is None:
            return False
        section_handler.update({key: value})
        return True

    def set_instrument(self, instrument):
        """
        Set an instrument for the configuration.

        Parameters
        ----------
        instrument : str or object
            Should be the name of the instrument or an object with the
            instrument name in the "name" attribute.  This will have the result
            of allowing access to configuration files in the <instrument>
            directory in the configuration path.

        Returns
        -------
        None
        """
        if instrument is None:
            return
        if isinstance(instrument, str):
            self.instrument_name = instrument.strip().lower()
        elif hasattr(instrument, 'name'):
            self.instrument_name = instrument.name
        else:
            self.handle_error(
                f"Could not parse {type(instrument)} as an instrument.")

    @property
    def current_iteration(self):
        """
        Return the current iteration of the SOFSCAN reduction

        Returns
        -------
        int or None
            An integer if the iteration has been set previously, and `None`
            otherwise.
        """
        return self.iterations.current_iteration

    @current_iteration.setter
    def current_iteration(self, iteration):
        """
        Set the current iteration for the SOFSCAN reduction.

        The configuration will be validated if this value is set.

        Parameters
        ----------
        iteration : int
            The iteration to set.

        Returns
        -------
        None
        """
        self.set_iteration(iteration, validate=True)

    @property
    def max_iteration(self):
        """
        Return the maximum number of iterations for the SOFSCAN reduction.

        Returns
        -------
        int or None
            An int if the maximum number of iterations has been set and `None`
            otherwise.
        """
        return self.iterations.max_iteration

    @max_iteration.setter
    def max_iteration(self, value):
        """
        Set the maximum number of iterations for the SOFSCAN reduction.

        Parameters
        ----------
        value : int or float or str
            The maximum number of iterations.  Must be able to be parsed to an
            int.

        Returns
        -------
        None
        """
        self.parse_key_value('rounds', value)

    @property
    def blacklisted(self):
        """
        Return the current set of blacklisted keywords.

        Blacklisted keywords should not be accessible or modifiable by the
        reduction.

        Returns
        -------
        set (str)
        """
        return self.disabled & self.locked

    @property
    def preserved_cards(self):
        """
        Return the current dictionary of preserved FITS header keyword values.

        The preserved FITS header cards are those taken from the original FITS
        file and preserved for later use without fear that they will be
        modified.

        Returns
        -------
        dict
           A dictionary of the form {keyword: (value, comment)}.
        """
        if self.fits is None:
            return {}
        return self.fits.preserved_cards

    @property
    def user_path(self):
        """
        Return the path to the user configuration directory.

        Returns
        -------
        directory : str
        """
        return os.path.join(os.path.expanduser('~'), '.sofscan')

    @property
    def expected_path(self):
        """
        Return the path to the expected configuration directory.

        If the .sofscan2 directory is not found in the user home directory,
        returns the default configuration directory.

        Returns
        -------
        directory : str
        """
        user_path = self.user_path
        if not os.path.isdir(user_path):
            return self.config_path
        else:  # pragma: no cover
            return user_path

    def resolve_filepath(self, filename):
        """
        Return a full file path to the file.

        If the file name given is already a complete path, it will be returned.
        Otherwise, it will be joined with the `expected_path` and returned.
        Note that no check is performed to see if the file exists.

        Parameters
        ----------
        filename : str
            The full or partial path to a given file.

        Returns
        -------
        full_filepath : str
        """
        f = self.aliases.unalias_value(self.options, filename)
        if f.startswith(os.sep):
            return f
        else:
            return os.path.join(self.expected_path, f)

    def get_configuration_filepath(self, key):
        """
        Return a file path from the given options.

        Note that no check is performed to see if the path is valid or exists.

        Parameters
        ----------
        key : str
            The configuration key from which to extract the file path.

        Returns
        -------
        filepath : str or None
            A string if found in the configuration, and `None` otherwise.
        """
        f = self.get_string(key)
        if f is None:
            return None
        return self.resolve_filepath(f)

    def find_configuration_files(self, filename):
        """
        Find all matching files in the SOFSCAN configuration directories.

        Checks the base SOFSCAN configuration directory, the user directory,
        and the instrument directories for any file matching that which is
        given.  Not that alias references may be passed in as filenames too.
        All output files will exist and be full paths.

        Parameters
        ----------
        filename : str
            The filename to find.

        Returns
        -------
        files : list (str)
            A list of all matching files in the configuration directories.
        """
        f = self.aliases.unalias_value(self.options, filename)
        user_path = self.expected_path
        configuration_files = []

        if f.startswith(os.sep) or f.startswith('/'):
            # A full file path
            if os.path.isfile(f):
                configuration_files.append(os.path.abspath(f))
                return configuration_files
            else:
                msg = f'File not found: {filename}.'
                if self.verbose:
                    log.warning(msg)
                else:
                    log.debug(msg)

                return configuration_files

        base_file = os.path.join(self.config_path, f)
        if os.path.isfile(base_file):
            configuration_files.append(base_file)

        if user_path != self.config_path:  # pragma: no cover
            user_file = os.path.join(user_path, f)
            if (os.path.isfile(user_file)
                    and user_file not in configuration_files):
                configuration_files.append(user_file)

        if self.instrument_name is not None:
            base_instrument_file = os.path.join(
                self.config_path, self.instrument_name, f)
            if (os.path.isfile(base_instrument_file)
                    and base_instrument_file not in configuration_files):
                configuration_files.append(base_instrument_file)

            if user_path != self.config_path:  # pragma: no cover
                user_instrument_file = os.path.join(
                    user_path, self.instrument_name, f)
                if (os.path.isfile(user_instrument_file)
                        and user_instrument_file not in configuration_files):
                    configuration_files.append(user_instrument_file)

        # If not matching files found, check absolute path
        if len(configuration_files) == 0:
            if self.verbose:
                log.warning(f"No matching configuration files for {filename}.")
            else:
                log.debug(f"No matching configuration files for {filename}.")

        return configuration_files

    def priority_file(self, filename_or_key):
        """
        Returns the highest priority matching filename from configuration.

        The order of files from lowest to highest priority is:
        base_configuration -> user_configuration -> base_instrument ->
        user_instrument

        The configuration options in higher priority files overrule those in
        lower priorities.  This returns the highest priority configuration file
        available.

        Parameters
        ----------
        filename_or_key : str
            The name of the file or configuration key.  If a matching key is
            found in the configuration, it takes priority over the filename.

        Returns
        -------
        str or None
            The highest priority matching file, or `None` if not found.
        """
        if filename_or_key in self:
            filename = self.get_string(filename_or_key)
        else:
            filename = filename_or_key

        matching_files = self.find_configuration_files(filename)
        if len(matching_files) == 0:
            return None
        return matching_files[-1]

    def update(self, configuration_options):
        """
        Update the configuration options with another set of options..

        Parameters
        ----------
        configuration_options : dict or ConfigObj
            The configuration options to read and parse.

        Returns
        -------
        None
        """
        self.read_configuration(configuration_options, validate=True)

    def read_configuration_file(self, filename, validate=True):
        """
        Read and apply a given configuration file.

        Parameters
        ----------
        filename : str
            The file path to the configuration file.
        validate : bool, optional
            If `True`, validate the configuration following the read.

        Returns
        -------
        None
        """
        if not os.path.isfile(filename):
            self.handle_error(f"Not a file: {filename}")
            return

        if filename in self.config_files:
            return
        self.config_files.append(filename)
        config = ConfigObj(filename)
        log.info(f"Reading configuration from {filename}")
        self.read_configuration(config, validate=validate)

    def read_configuration(self, config, validate=True):
        """
        Read a given configuration and apply to this Configuration.

        Parameters
        ----------
        config : str or dict or ConfigObj or Configuration
            The configuration to read and apply.  A string value is assumed to
            be the file path to a configuration file, and will be read using
            :func:`Configuration.read_configuration_file`.
        validate : bool, optional
            If `True`, validate the configuration following the read.  This
            will examine and apply all conditionals.

        Returns
        -------
        None
        """
        if isinstance(config, str):  # is a filename
            filenames = self.find_configuration_files(config)
            for filename in filenames:
                self.read_configuration_file(filename, validate=validate)
            return
        elif isinstance(config, Configuration):
            if not config.enabled:
                return
            config = config.options
        elif not isinstance(config, dict):
            msg = (f"Configuration must be of type {str}, {dict}, "
                   f"or {Configuration}. Received {type(config)}.")
            self.handle_error(msg)
            return

        config = self.normalize_options(config)
        self.update_sections(config, validate=False)
        self.parse_configuration_body(config)
        self.aliases.resolve_configuration(self)
        if validate:
            self.validate()
        self.enabled = True

    def read_configurations(self, configuration_string, validate=True):
        """
        Read and apply multiple configurations.

        Parameters
        ----------
        configuration_string : str
            A list of comma separated file paths to configuration files.
        validate : bool, optional
            If `True`, validate the configuration following a read.

        Returns
        -------
        None
        """
        configuration_strings = [
            s.strip() for s in configuration_string.split(',')]
        for configuration_file in configuration_strings:
            self.read_configuration(configuration_file, validate=validate)

    def validate(self):
        """
        Apply all conditions to the configuration.

        Returns
        -------
        None
        """
        self.conditions.process_conditionals(self)

    def set_object(self, source_name, validate=True):
        """
        Set the source object in the configuration.

        Parameters
        ----------
        source_name : str
            The name of the astronomical source.
        validate : bool, optional
            If `True`, validate the configuration once the source has been set.

        Returns
        -------
        None
        """
        self.objects.set_object(self, source_name, validate=validate)

    def set_iteration(self, iteration, validate=True):
        """
        Set the current iteration of the reduction in the configuration.

        Parameters
        ----------
        iteration : int
            The iteration number.
        validate : bool, optional
            If `True`, validate the configuration once the iteration number has
            been set.

        Returns
        -------
        None
        """
        self.iterations.set_iteration(self, iteration, validate=validate)

    def set_date(self, date, validate=True):
        """
        Set the observation date in the configuration.

        Parameters
        ----------
        date : str or int or float
            The observation date.  If a string is used, it should be in ISOT
            format in UTC scale.  Integers and floats will be parsed as
            MJD times in the UTC scale.
        validate : bool, optional
            If `True`, validate the configuration once the date has been set.

        Returns
        -------
        None
        """
        self.dates.set_date(self, date, validate=validate)

    def set_serial(self, serial, validate=True):
        """
        Sets options based on a scan serial number.

        Parameters
        ----------
        serial : int
        validate : bool, optional
            If `True`, validate the whole configuration after setting the
            serial options.

        Returns
        -------
        None
        """
        serial_branch = self.get_branch('serial')
        if serial_branch is None:
            return
        for serial_key, serial_options in serial_branch.items():
            serial_range = get_range(serial_key, is_positive=True)
            if serial_range.in_range(serial):
                self.apply_configuration_options(
                    serial_options, validate=validate)

    def apply_configuration_options(self, options, validate=True):
        """
        Apply options to the configuration.

        The dictionary keys in the options are first examined and separated
        into commands, keyword=value settings, and configuration section
        updates. Sections will be updated first, followed by single
        keyword=value updates.  Finally, configuration commands will be
        processed using :func:`Configuration.apply_commands`.

        Parameters
        ----------
        options : dict or ConfigObj
            The options to apply to the configuration.
        validate : bool, optional
            If `True`, validate the configuration once the options have been
            applied.

        Returns
        -------
        None
        """
        if not isinstance(options, dict) or len(options) == 0:
            return

        commands = {}
        other = {}
        sections = {}
        for command, action in options.items():
            if command in self.command_keys:
                if isinstance(action, list):
                    commands[command] = action
                else:
                    commands[command] = [action]
            elif command in self.section_keys:
                sections[command] = action
            else:
                other[command] = action

        self.update_sections(sections, validate=False)

        for key, value in other.items():
            self.parse_key_value(key, value)

        self.apply_commands(commands)
        if validate:
            self.validate()

    def update_sections(self, options, validate=True):
        """
        Update sections in the configuration.

        The configuration aliases, conditions, dates, iterations, and objects
        will be updated if present in `options`.

        Parameters
        ----------
        options : dict or ConfigObj.
            A dictionary used to update the sections.
        validate : bool, optional
            If `True`, validate the configuration once all sections have been
            updated.

        Returns
        -------
        None
        """
        for section in [
                'aliases', 'date', 'iteration', 'object', 'conditionals']:
            self.get_section_handler(section).update(options)
        if validate:
            self.validate()

    def parse_configuration_body(self, options):
        """
        Parse configuration options and apply.

        Will filter out any sectional keys, but set and apply any other
        options. Note that configuration commands will always be processed
        last.

        Parameters
        ----------
        options : dict or ConfigObj
            The configuration options to parse.

        Returns
        -------
        None
        """
        commands = {}
        for key, value in options.items():
            if key in self.section_keys:
                continue  # Sections are handled separately
            if key in self.command_keys:
                commands[key] = value
                continue  # Command keys require special handling

            self.merge_options({key: value})

        self.apply_commands(commands)

    def apply_commands(self, commands, command_order=None):
        """
        Apply configuration commands to the configuration.

        Configuration commands are special keywords that instruct the
        configuration to perform a certain action such as setting an attribute
        for a certain key value like blacklisting it from any use in the
        reduction, or locking it's value in place.

        Parameters
        ----------
        commands : dict or ConfigObj
            The commands to apply in {command_name: actions} format, where
            command_name is a string, and actions may be a dict, list or
            comma-separated string of actions to apply for the given command
            name.
        command_order : list (str), optional
            The order in which to apply commands.  This can be very important
            since one might wish to unlock a key, set it's value, and then
            finally re-lock it's value in place.   If the lock command is
            issued first, no update may take place etc.

        Returns
        -------
        None
        """
        if command_order is None:
            command_order = self.command_order
        for command in command_order:
            if command not in commands:
                continue
            keys = commands[command]

            if command == 'update':
                if isinstance(keys, dict):
                    for key, value in keys.items():
                        self.parse_key_value(key, value)
                else:
                    for (key, value) in keys:
                        self.parse_key_value(key, value)
            else:
                if isinstance(keys, str):
                    keys = [s.strip() for s in keys.split(',')]
                for key in keys:
                    self.parse_key_value(command, key)

    def put(self, *args, check=True):
        """
        Set a value in the configuration.

        Parameters
        ----------
        args : tuple (str)
            The last argument (args[-1]) should always contain the value that
            is being set in the configuration.  All values prior to that are
            considered part of the configuration key.  For example,
            args=('correlated', 'sky', 'biasgains', '0.3:3') sets
            {'correlated': {'sky': {'biasgains': '0.3:3'}}} in the
            configuration.
        check : bool, optional
            If `True`, check that a key is not locked before attempting to set
            a value.  If it is locked, no changes will be made.

        Returns
        -------
        None
        """
        value = args[-1]
        key = self.aliases('.'.join(args[:-1]))

        if check:
            if self.is_locked(key):
                return

        self.remove_disabled_key(key)
        self.merge_options(self.key_value_to_dict(key, value))

    def remove_disabled_key(self, key):
        """
        Enable all disabled branches upto and including key.

        Parameters
        ----------
        key : str

        Returns
        -------
        None
        """
        set_key = self.matching_set_dot_key(key, self.disabled)
        if set_key is None:
            return
        self.disabled.remove(set_key)

    def read_fits(self, header_or_file, extension=0, validate=True):
        """
        Read contents of a FITS file or header and apply to the configuration.

        Parameters
        ----------
        header_or_file : fits.Header or str
            A FITS header or the file path to a FITS file to read.
        extension : int, optional
            If a file path was passed in as the argument, the extension of the
            FITS file from which to take the header.  The default is the
            primary (0) extension.
        validate : bool, optional
            If `True`, fully validate the configuration once the FITS header
            has been read.

        Returns
        -------
        None
        """
        self.fits.update_header(header_or_file, extension=extension)
        self.merge_fits_options()
        if validate:
            self.validate()
        self.preserve_header_keys()
        self.fits.enabled = True

    def merge_fits_options(self):
        """
        Merge the FITS handler options into the main configuration body.

        Once a FITS header has been read by the fits handler, this method
        creates a copy of those key/values in the main configuration body for
        easy access to the values or advanced handling methods.

        Returns
        -------
        None
        """
        fits_config = ConfigObj()
        fits_config['fits'] = deepcopy(self.fits.options)
        self.merge_options(fits_config)

    @staticmethod
    def key_value_to_dict(key, value):
        """
        Convert a string key and a given value to a dictionary.

        The configuration can use a dot (".") separator to mark dictionary
        levels using a string.  For example a.b.c = 1 represents [a][b][c] = 1.

        Examples
        --------
        >>> print(Configuration.key_value_to_dict('a.b.c', 1))
        {'a': {'b': {'c': 1}}}

        Parameters
        ----------
        key : str
            The dot key string to convert.
        value : str or object
            The final assigned value.

        Returns
        -------
        dict
        """
        branches = key.split('.')
        result = dict()
        current = result
        for branch in branches[:-1]:
            current[branch] = dict()
            current = current[branch]
        current[branches[-1]] = value
        return result

    @staticmethod
    def matching_wildcard(string_array, wildcard_pattern):
        """
        Find and return all strings in an array matching a given pattern.

        Patterns will be parsed according to :func:`fnmatch.fnmatch`.

        Parameters
        ----------
        string_array : list (str)
            A list of strings from which to find strings matching
            `wildcard_pattern`.
        wildcard_pattern : str
            The pattern to match using :func:`fnmatch.fnmatch`.

        Returns
        -------
        list (str)
            A list of strings matching `wildcard_pattern` from `string_array`.
        """
        return [s for s in string_array
                if fnmatch.fnmatch(s, wildcard_pattern)]

    def matching_wildcard_keys(self, wildcard_key, flat_dictionary=None):
        """
        Find all matching configuration keys for a given wildcard.

        Wildcards are parsed according to :func:`fnmatch.fnmatch`.  All
        configuration keys available in the main configuration body will be
        checked and returned if found.

        Parameters
        ----------
        wildcard_key : str
            The string pattern to search for using :func:`fnmatch.fnmatch`.
        flat_dictionary : dict or ConfigObj, optional
            A flat dictionary to search through for matching keys.  A flat
            dictionary refers to a single level dictionary.  If not supplied,
            the configuration body is flattened (e.g. [key1][key2] = 1 is
            converted to key1.key2 = 1).

        Returns
        -------
        list (str)
            A list of dot-separated keys that match `wildcard_key`.
        """
        if flat_dictionary is None:
            flat_dictionary = self.flatten(self.options)
        return self.matching_wildcard(
            list(flat_dictionary.keys()), wildcard_key)

    def flatten(self, options, base=None, basestring='', unalias=True,
                keep_value=False):
        """
        Transforms a nested dictionary to a single level representation.

        A nested dictionary such as {'a': {'b': {'c': 1}}} will be converted
        to a dot-separated single level version where dots (".") are used to
        mark different dictionary levels ({'a.b.c': 1}).

        Parameters
        ----------
        options : dict or ConfigObj
            The nested dictionary to flatten.
        base : ConfigObj, optional
            The base options to update during recursive calls to `flatten`.
            This should not be provided by the user during standard use.
        basestring : str, optional
            The current dot-string branch representation of a dictionary level
            during recursive calls to `flatten`.  It should not be provided by
            the user during standard use.
        unalias : bool, optional
            If `True`, unalias all keys.
        keep_value : bool, optional
            If `True`, keep .value settings in the configuration.

        Returns
        -------
        flat_dictionary : ConfigObj
            A flattened version of `options`.
        """
        new = ConfigObj()
        return_base = base is None
        if return_base:
            base = new

        for k, v in options.items():
            if unalias:
                uk = self.aliases(basestring + k)
            else:
                uk = basestring + k
            if isinstance(v, (dict, ConfigObj)):
                if len(v) == 0:
                    base[uk] = {}
                else:
                    self.flatten(v, base=base, basestring=uk + '.')
            else:
                if not keep_value:
                    if uk.endswith('.value'):
                        base[uk[:-6]] = v
                    else:
                        base[uk] = v
                else:
                    base[uk] = v

        if return_base:
            return base

    @classmethod
    def expand_options(cls, options):
        """
        Convert a potentially dot-separated dictionary to multi-level.

        Note that no aliasing, checking of values or any other configuration
        options are performed.

        Examples
        --------
        >>> options = {'write.source': False, 'foo.bar.baz': 1}
        >>> print(Configuration.expand_options(options))
        {'write': {'source': False}, 'foo': {'bar': {'baz': 1}}}

        Parameters
        ----------
        options : ConfigObj or dict
            The options to expand.

        Returns
        -------
        expanded_options : ConfigObj
            The options expanded without any dot-separators.
        """
        expanded = ConfigObj()
        for key, value in options.items():
            branch = cls.key_value_to_dict(key, value)
            expanded.merge(branch)

        return expanded

    def normalize_options(self, options):
        """
        Convert a dictionary prior to merging into the configuration.

        Converts options to a standard multi-level dictionary with all string
        values.  No aliasing is performed.

        Parameters
        ----------
        options : dict or ConfigObj
            The options to normalize.

        Returns
        -------
        normalized_options : ConfigObj
             The normalized options.
        """
        return self.expand_options(self.stringify(options))

    def merge_options(self, merge_options, options=None, chain=None,
                      keep_disabled=False):
        """
        Merge a set of options into currently existing options.

        Merges `merge_options` into `options` using rules and settings from the
        :class:`Configuration`.  Sectional keys such as 'iteration' or
        'conditionals' will be parsed by *this* Configuration and validated.
        Note that final validation on the configuration body is not performed
        and should be done so manually.

        FITS header options are *NOT* handled here, and will essentially be
        skipped over.  However, fits instructions that exist in the
        configuration will be parsed.  FITS header should instead be read in by
        :func:`Configuration.read_fits` and added via
        :func:`Configuration.merge_fits_options`.

        Parameters
        ----------
        merge_options : dict or ConfigObj
            The new options that should be merged into `options`.
        options : dict or ConfigObj, optional
            The options which to merge `merge_options` into.  Generally, this
            should not be supplied by the user and defaults to the main
            configuration body of options.  When supplied, it is usually for
            recursive calls.
        chain : list (str), optional
            A list of previously encountered dictionary levels for use during
            recursive calls.  This should generally not be supplied by the
            user.
        keep_disabled : bool, optional
            If `True`, allow a configuration value to be changed, but never
            remove it from the current set of disabled keys.

        Returns
        -------
        None
        """
        if options is None:
            options = self.options

        if chain is None:
            chain = []
            in_handler = False
        else:
            chain = chain.copy()
            # keys can sometimes relate to handlers such as 'fits.OBJECT'
            if len(chain) > 0:
                in_handler = str(chain[0]).lower().strip() in self.handler_keys
            else:  # pragma: no cover
                # Just in case
                in_handler = False

        barred_branches = self.locked | self.blacklisted

        for key, value in merge_options.items():
            new_chain = chain + [key]
            dot_key = '.'.join(new_chain)
            if self.dot_key_in_set(dot_key, barred_branches):
                continue

            if not keep_disabled:
                self.remove_disabled_key(dot_key)

            # Attempt to parse to a section of the options.
            if not in_handler and self.parse_to_section(key, value):
                continue

            # Add in the key-value if not already present.
            if key not in options:
                options.merge({key: value})
                continue

            # Merge into the existing options.
            options_value = options[key]

            if isinstance(options_value, dict):
                if isinstance(value, dict):  # Recursive
                    self.merge_options(value, options=options_value,
                                       chain=new_chain)
                else:
                    options_value['value'] = value
            else:
                if isinstance(value, dict):
                    options[key] = {'value': options_value}
                    options[key].merge(value)
                else:
                    options[key] = value

    def dot_key_in_set(self, key, test_set):
        """
        Check if a dot-separated key exists in a given set of strings.

        The `key` will be unaliased before any comparison is attempted.

        Parameters
        ----------
        key : str
            The key to test.
        test_set : iterable (str)
            A list, set, tuple, etc. of strings.  If the unaliased `key` exists
            in the set, returns `True`.

        Returns
        -------
        bool
        """
        return self.matching_set_dot_key(key, test_set) is not None

    def matching_set_dot_key(self, key, test_set):
        """
        Return the dotted key from a test set that relates to `key`.

        Parameters
        ----------
        key : str
            The key to find.
        test_set : iterable (str)
            A list, set, tuple, etc. of strings.  If the unaliased `key` exists
            in the set, returns `True`.

        Returns
        -------
        str or None
            A string if found, and `None` otherwise.
        """
        check_string = ''
        uk = self.aliases(key)
        for k in uk.split('.'):
            if check_string == '':
                check_string = k
            else:
                check_string += '.' + k
            if check_string in test_set:
                return check_string
        return None

    def parse_key_value(self, key, value):
        r"""
        Parse and apply a key-value pair into the configuration.

        This method will firstly unalias any given key.  Note that wildcards
        may be provided ('\*') to apply `value` to all matching keys found in
        the configuration.  For example, if both "key1" and "key2" exist in
        the configuration, Configuration.parse_key_value("key\*", "abc") will
        set the value of both to "abc".

        If `key` relates to a configuration command it will be applied to any
        value found in `value`.  For example,
        Configuration.parse_key_value('lock', ['key1', 'key2']) will lock
        key1 and key2.  Finally, if the `key` was not a command, it will be
        processed using :func:`Configuration.put` ->
        :func:`Configuration.merge` which will add the key-value to the
        configuration body, or process it for a certain configuration section.

        Parameters
        ----------
        key : str
            The configuration key for which `value` applies.  This may be an
            actual configuration key, configuration command, or section.
            Wildcards ('\*') may be used to apply `value` to more than one
            configuration key.
        value : str or dict or iterable
            The value to apply for `key` in the configuration.

        Returns
        -------
        None
        """

        key = self.aliases(key)

        # Recursive call if wildcard used
        flat_config = None
        if '*' in key:
            flat_config = self.flatten(self.options)
            matching_keys = self.matching_wildcard_keys(
                key, flat_dictionary=flat_config)
            for k in matching_keys:
                self.parse_key_value(k, value)  # Recursive

        elif key in self.command_keys:

            command = key
            if isinstance(value, list):
                keys = value
            else:
                # The command applied to numerous keys
                keys = str(value).split(',')

            for applied_to_key in keys:

                if '*' in applied_to_key:
                    matching_keys = self.matching_wildcard_keys(
                        applied_to_key, flat_dictionary=flat_config)
                    for k in matching_keys:
                        self.parse_key_value(command, k)

                if key == 'config':
                    self.read_configurations(value, validate=True)
                elif command == 'blacklist':
                    self.blacklist(applied_to_key)
                elif command == 'whitelist':
                    self.whitelist(applied_to_key)
                elif command == 'forget':
                    self.forget(applied_to_key)
                elif command == 'recall':
                    self.recall(applied_to_key)
                elif command == 'lock':
                    self.lock(applied_to_key)
                elif command == 'unlock':
                    self.unlock(applied_to_key)
                elif command == 'add':
                    self.add_new_branch(applied_to_key)
                elif command == 'rounds':
                    self.iterations.max_iteration = int(applied_to_key)
                else:  # pragma: no cover
                    self.handle_error(f"Command ({key}) not implemented.")

        else:
            self.put(key, value)

    def blacklist(self, key):
        """
        Blacklists a given key in the configuration.

        Once a configuration key has been blacklisted, it will be unavailable
        for retrieval or setting throughout the entire reduction via standard
        configuration functions.  This is a safe way to ensure certain settings
        are never applied.

        Parameters
        ----------
        key : str
            The configuration key to blacklist.

        Returns
        -------
        None
        """
        uk = self.aliases(key)
        if uk in self.blacklisted:
            return
        if uk in self.locked:
            if self.verbose:
                log.warning(f"Cannot blacklist locked option: {key}")
            else:
                log.debug(f"Cannot blacklist locked option: {key}")
            return
        self.disabled.add(uk)
        self.locked.add(uk)

    def whitelist(self, key):
        """
        Whitelist a configuration key.

        Whitelisting a configuration key removes it from the blacklist if
        set and unlocks it.  It will still be disabled however.  Locked options
        cannot be whitelisted.

        Parameters
        ----------
        key : str
            The configuration key to whitelist.

        Returns
        -------
        None
        """
        uk = self.aliases(key)
        if uk in self.locked:
            if uk in self.blacklisted:
                self.locked.remove(uk)
            elif self.verbose:
                log.warning(f"Cannot whitelist locked option: {key}")
            else:
                log.debug(f"Cannot whitelist locked option: {key}")

    def lock(self, key):
        """
        Lock a given configuration key.

        Once a configuration key has been locked, its current configuration
        settings cannot be altered via standard configuration functions.

        Parameters
        ----------
        key : str
            The configuration key to lock.

        Returns
        -------
        None
        """
        uk = self.aliases(key)
        if uk in self.locked:
            return
        self.locked.add(uk)

    def unlock(self, key):
        """
        Unlock a configuration key.

        If a configuration key was previously locked, unlock that key and allow
        subsequent changes to it's status or value in the configuration.  Note
        that blacklisted keys cannot be unlocked.

        Parameters
        ----------
        key : str
            The configuration key to unlock.

        Returns
        -------
        None
        """
        uk = self.aliases(key)
        if not self.is_blacklisted(uk):
            try:
                self.locked.remove(uk)
            except KeyError:
                pass

    def forget(self, key):
        """
        Forget a key in the configuration.

        Disables a given key in the configuration such that it's value or
        status may not be altered or retrieved.  If key is "blacklist" or
        "conditions", all blacklisted keywords or conditions in the
        configuration will be removed.

        Parameters
        ----------
        key : str
            The configuration key to forget.

        Returns
        -------
        None
        """
        if self.is_locked(key) and not self.is_blacklisted(key):
            if self.verbose:
                log.warning(f"Cannot forget locked option: {key}")
            else:
                log.debug(f"Cannot forget locked option: {key}")
            return

        if key == 'blacklist':
            for k in self.blacklisted:
                self.whitelist(k)
            return

        if key == 'conditions':
            self.conditions.clear()
            return

        self.disabled.add(self.aliases(key))

    def recall(self, key):
        """
        Recall (remember) a given key in the configuration.

        Performs a reverse `forget` operation on the given keyword so that it's
        value may be retrieved from the configuration or changed if applicable.

        Parameters
        ----------
        key : str
            The configuration key to recall.

        Returns
        -------
        None
        """
        if self.is_locked(key):
            if self.verbose:
                log.warning(f"Cannot recall locked option: {key}")
            else:
                log.debug(f"Cannot recall locked option: {key}")
        else:
            try:
                self.disabled.remove(self.aliases(key))
            except KeyError:
                pass

    def has_option(self, key):
        """
        Check if a key is available in the configuration.

        Parameters
        ----------
        key : str
            The configuration key to check.

        Returns
        -------
        available : bool
        """
        return self.aliases(key) in self

    def is_configured(self, key):
        """
        Check if the given key has a value in the configuration.

        Parameters
        ----------
        key : str
            The configuration key to check.

        Returns
        -------
        has_value : bool
        """
        return self.get(key) is not None

    def set_option(self, key, value=True):
        """
        Set an option in the configuration.

        The option will always be added as a branch.  I.e, a dictionary in the
        configuration options as {`key`: {value: `value`}}.  This is in
        contrast to :func:`Configuration.put` which will place the option
        as {`key`: `value`} in certain instances.

        Parameters
        ----------
        key : str
            The configuration key to set.
        value : object, optional
            The value to set in the configuration.

        Returns
        -------
        None
        """
        self.add_new_branch(self.aliases(key), value)

    def get_options(self, *args, default=None, unalias=True):
        """
        Retrieve the configuration options for a given key.

        The options refer to a specific branch of the configuration and must
        retrieve a dictionary like value.  If the key refers to a singular
        configuration value, the default will be returned instead.

        Parameters
        ----------
        args : tuple (str)
            The configuration key levels such as ['correlated', 'sky'].
        default : object, optional
            The value to return if the options could not be retrieved from the
            configuration.
        unalias : bool, optional
            If `True`, unalias all keys in `args` before attempting to retrieve
            a value.

        Returns
        -------
        options : dict or str or object
            The configuration branch or value if the options were found, or
            the `default` value otherwise.  The returned options will be a copy
            of the configuration values, so changing these will have no impact
            on the actual configuration.
        """
        level = self.get_branch(*args, default=default, unalias=unalias)
        if not isinstance(level, dict):
            return default
        result = deepcopy(level)
        if 'value' in result:
            del result['value']
        return result

    def is_locked(self, key):
        """
        Return if a configuration key is currently locked.

        Parameters
        ----------
        key : str
            The configuration key to check.

        Returns
        -------
        locked : bool
        """
        return (self.dot_key_in_set(key, self.locked) or
                self.dot_key_in_set(f'{key}.value', self.locked))

    def is_disabled(self, key):
        """
        Return if a configuration key is currently disabled.

        Parameters
        ----------
        key : str
            The configuration key to check.

        Returns
        -------
        disabled : bool
        """
        return self.dot_key_in_set(key, self.disabled)

    def is_blacklisted(self, key):
        """
        Return if a configuration key is blacklisted.

        Parameters
        ----------
        key : str
            The configuration key to check.

        Returns
        -------
        blacklisted : bool
        """
        return self.dot_key_in_set(key, self.disabled & self.locked)

    def add_new_branch(self, key, value=True):
        """
        Add a new branch to the configuration.

        Parameters
        ----------
        key : str
            The path to the configuration branch to add.  Branch levels should
            be separated by a '.'.  Note that if `key` is of the form
            "my_key_to_set=my_value", "my_value" will be used in place of
            `value`.
        value : str or object
            The value to set in the configuration.

        Returns
        -------
        None
        """
        if '=' in key:
            key, set_value = [s.strip() for s in key.split('=')]
        else:
            set_value = value

        branch_value_key = key + '.value'
        self.put(branch_value_key, set_value)

    def get_keys(self, branch_name=None):
        """
        Return a list of all keys in a given options branch.

        Parameters
        ----------
        branch_name : str, optional
            The name of the options branch in the configuration.  If not
            supplied, return all first level active configuration branches.

        Returns
        -------
        keys : list (str)
        """
        if branch_name is None:
            branch_keys = []
            for key in self.options.keys():
                keys = self.get_keys(key)
                if keys is not None:
                    branch_keys.append(key)
            return branch_keys

        branch = self.get_branch(branch_name, default=None)
        if not isinstance(branch, dict):
            return None
        return list(branch.keys())

    def get_preserved_header_keys(self):
        """
        Return a set of all preserved FITS header keys in the configuration.

        Returns
        -------
        set (str)
        """
        return set(str(s).strip().upper()
                   for s in self.get_list('fits.addkeys'))

    def preserve_header_keys(self, header=None):
        """
        Assign preserved FITS header keyword values in the configuration.

        Parameters
        ----------
        header : fits.Header, optional
            The FITS header from which to extract the keyword values.  If not
            supplied, defaults to the previously read header in the
            configuration.

        Returns
        -------
        None
        """
        self.fits.reset_preserved_cards()
        keys = self.get_preserved_header_keys()
        for key in keys:
            self.fits.set_preserved_card(key, header=header)

    def get_filepath(self, key, default=None, get_all=False):
        """
        Return the file path for a key value in the configuration.

        Parameters
        ----------
        key : str
            The name of the configuration key in which the file is referenced.
        default : str or object, optional
            The default file to look for if not found in the configuration.
        get_all : bool, optional
            If `True`, return all matching file paths for a value found in the
            configuration.  If `False`, only return a single highest priority
            file.

        Returns
        -------
        str or list (str) or None
            The highest priority file or a list of all found matching files.
            `None` will be returned if no file can be found.
        """
        value = self.get_string(key, default=default)
        if value is None:
            return None
        value = self.find_configuration_files(value)
        if len(value) == 0:
            return None
        if get_all:
            return value  # a list of all filename matches
        else:
            return value[-1]  # highest priority file

    def purge(self, key):
        """
        Completely remove a key from the configuration.

        Parameters
        ----------
        key : str

        Returns
        -------
        None
        """
        if not self.has_option(key):
            return
        key = self.aliases(key)
        branches = key.split('.')
        options = self.options
        seen = []
        for branch in branches:
            seen.append(branch)
            if '.'.join(seen) == key:
                del options[branch]
                break
            options = options[branch]

    def get_flat_alphabetical(self, options=None, unalias=True,
                              keep_value=False):
        """
        Return all configuration keys in a flattened form.

        The output from this method is a single-level dictionary containing
        flattened keys and values.  A flattened key uses dot-separators to
        distinguish dictionary levels.  E.g., a.b.c:value means {a:b:c:value}.
        Keys will be ordered alphabetically.

        Parameters
        ----------
        options : ConfigObj or dict, optional
            The options to retrieve alphabetical keys.  The default are the
            configuration options.
        unalias : bool, optional
            If `True`, unalias all keys.
        keep_value : bool, optional
            If `True`, keep .value keys in the configuration.

        Returns
        -------
        dict
        """
        if options is None:
            options = self.options

        flat_options = self.flatten(options, unalias=unalias,
                                    keep_value=keep_value)
        keys = list(sorted(flat_options.keys()))
        result = {}
        for key in keys:
            result[key] = flat_options[key]
        return result

    def get_active_options(self, options=None):
        """
        Return all options that are not disabled in the configuration.

        Parameters
        ----------
        options : ConfigObj or dict, optional
            The options from which to prune disabled values/branches.

        Returns
        -------
        enabled_options : ConfigObj
        """
        if options is None:
            options = deepcopy(self.options)
        flat_options = self.get_flat_alphabetical(options, keep_value=True)
        keep = {}
        for key, value in flat_options.items():
            if not self.is_disabled(key):
                keep[key] = value
        return self.expand_options(keep)

    def order_options(self, options=None, unalias=False):
        """
        Order the keys in options alphabetically.

        Parameters
        ----------
        options : ConfigObj or dict, options
            The options to order.  If not supplied defaults to the
            configuration options.
        unalias : bool, optional
            If `True`, unalias all keys.

        Returns
        -------
        ordered_options : ConfigObj
        """
        if options is None:
            options = deepcopy(self.options)
        return self.expand_options(
            self.get_flat_alphabetical(
                options, keep_value=True, unalias=unalias))

    def edit_header(self, header):
        """
        Add configuration settings to a FITS header.

        Parameters
        ----------
        header : astropy.fits.Header

        Returns
        -------
        None
        """
        # Prune the FITS options if in the header
        options = self.get_active_options()
        fits_options = options.get('fits')
        if fits_options is not None and self.fits.header is not None:
            keep = {}
            for key, value in fits_options.items():
                if key not in self.fits.header:
                    keep[key] = value
            options['fits'] = keep

        options['aliases'] = self.order_options(
            self.aliases.options, unalias=False)
        options['conditions'] = self.order_options(
            self.conditions.options, unalias=False)
        options['objects'] = self.order_options(
            self.objects.options, unalias=False)
        options['iterations'] = self.order_options(
            self.iterations.options, unalias=False)
        options['dates'] = self.order_options(
            self.dates.options, unalias=False)

        info = [('COMMENT', "<------ SOFSCAN Configuration ------>"),
                ('CNFGVALS', json.dumps(options), 'SOFSCAN configuration.')]
        insert_info_in_header(header, info, delete_special=True)

    def add_preserved_header_keys(self, header):
        """
        Add any preserved header keys from the original header to a new header.

        Parameters
        ----------
        header : astropy.io.fits.Header

        Returns
        -------
        None
        """
        if len(self.preserved_cards) == 0:
            return
        for key, value_and_comment in self.preserved_cards.items():
            header[key] = value_and_comment

    def set_outpath(self):
        """
        Set the output directory based on the configuration.

        If the configuration path does not exist, it will be created if the
        'outpath.create' option is set.  Otherwise, an error will be raised.

        Returns
        -------
        None
        """
        self.work_path = self.get_configuration_filepath('outpath')
        if self.work_path is None:
            self.work_path = os.getcwd()

        if not os.path.isdir(self.work_path):
            log.warning(f"The specified output path does not exist: "
                        f"{self.work_path}")
            if not self.get_bool('outpath.create'):
                log.error("Change 'outpath' to an existing directory, or "
                          "set 'outpath.create' to create a path "
                          "automatically.")
                raise ValueError(f"Specified output path does not exist: "
                                 f"{self.work_path}")

            log.info(f"Creating output directory: {self.work_path}")
            os.makedirs(self.work_path)

    def configuration_difference(self, config):
        """
        Return the difference between this configuration and another.

        Parameters
        ----------
        config : Configuration

        Returns
        -------
        difference : Configuration
        """
        difference = self.copy()
        difference.options = difference.options.__class__()
        flat = self.flatten(self.options)

        for key in flat.keys():
            if not self.is_configured(key):
                continue

            value1 = self[key]
            if not config.is_configured(key):
                difference.parse_key_value(key, value1)
                continue

            value2 = config[key]
            if value1 != value2:
                difference.parse_key_value(key, value1)

        # Now the other sections
        difference.fits.options = dict_difference(
            self.fits.options, config.fits.options)
        difference.conditions.options = dict_difference(
            self.conditions.options, config.conditions.options)
        difference.dates.options = dict_difference(
            self.dates.options, config.dates.options)
        difference.iterations.options = dict_difference(
            self.iterations.options, config.iterations.options)
        difference.objects.options = dict_difference(
            self.objects.options, config.objects.options)
        difference.aliases.options = dict_difference(
            self.aliases.options, config.aliases.options)

        return difference

    def lock_rounds(self, max_rounds=None):
        """
        Lock the number of rounds in-place for the reduction.

        Parameters
        ----------
        max_rounds : int or str or float, optional
            The new maximum number of rounds for the reduction.

        Returns
        -------
        None
        """
        self.iterations.lock_rounds(maximum_iterations=max_rounds)

    def check_trigger(self, trigger):
        """
        Check to see if the requirement for a trigger has been fulfilled.

        Parameters
        ----------
        trigger : str
            A trigger of the form <key> or <key><operator><value>.  If a single
            key is provided, the trigger is `True` so long as the bool value
            for the key evaluates as such.  Otherwise, the value for <key> will
            be evaluated using <operator> (=, !=, <, <=, >, >=) agains <value>.

        Returns
        -------
        bool
        """
        return self.conditions.check_requirement(self, trigger)
