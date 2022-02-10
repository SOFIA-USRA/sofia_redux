# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log
from configobj import ConfigObj
from copy import deepcopy
import numpy as np

from sofia_redux.scan.utilities import utils
from sofia_redux.scan.utilities.range import Range

__all__ = ['Options']


class Options(ABC):

    append_keys = ()

    def __init__(self, allow_error=False, verbose=True):
        """
        Initialize an Options object.

        The Options contain a :class:`ConfigObj` as the container for any
        given configuration.  There is also handling provided for cases when
        a configuration functionality results in an error.  These may be
        emitted as log messages if `verbose` is `True`, or raised as errors if
        `allow_error` is `False`.

        Since all configuration values are stored as strings, there are
        special handling methods to retrieve the desired value type such as
        integers, strings, angles, or time.

        Parameters
        ----------
        allow_error : bool, optional
            If `True`, allow poorly formatted options to be skipped rather than
            raising an error.
        verbose : bool, optional
            If `True`, issues a warning when a poorly option is encountered.
        """
        self.options = ConfigObj()
        self.allow_error = allow_error
        self.verbose = verbose

    def __len__(self):
        """
        Return the number of options keys.

        Returns
        -------
        int
        """
        if not hasattr(self.options, '__len__'):
            return 0
        return len(self.options)

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
        if not isinstance(self.options, dict):
            return False
        return key in self.options

    def __getitem__(self, key):
        """
        Retrieve the options value for the given key.

        Parameters
        ----------
        key : str
            The key to retrieve from the options.

        Returns
        -------
        value : str or None
            The configuration value, or `None` if it does not exist in the
            configuration.
        """
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        else:
            return result

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
        del self.options[key]

    @property
    def size(self):
        """
        Return the number of available options.

        Returns
        -------
        n_options : int
        """
        return len(self)

    @property
    def is_empty(self):
        """
        Return `True` if the options contain any key-values.

        Returns
        -------
        bool
        """
        return self.size == 0

    def copy(self):
        """
        Return a copy of the options.

        Returns
        -------
        Options
        """
        return deepcopy(self)

    def clear(self):
        """
        Clear all options.

        Returns
        -------
        None
        """
        self.options = ConfigObj()

    def get(self, key, default=None, unalias=True):
        """
        Retrieve the given key value from the options.

        Parameters
        ----------
        key : str
            The name of the value to retrieve.
        default : object, optional
            The value to return if the options do not contain the given `key`.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : str
            The retrieved value.  Note that all :class:`ConfigObj` values are
            strings.
        """
        return self.options.get(key, default=default)

    def get_string(self, key, default=None, unalias=True):
        """
        Return a string value from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : str, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : str
        """
        return utils.get_string(self.get(key, unalias=unalias),
                                default=default)

    def get_bool(self, key, default=False, unalias=True):
        """
        Return a boolean value from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : bool, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : bool
        """
        return utils.get_bool(self.get(key, unalias=unalias), default=default)

    def get_int(self, key, default=0, unalias=True):
        """
        Return an integer value from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : int, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : int
        """
        return utils.get_int(self.get(key, unalias=unalias), default=default)

    def get_float(self, key, default=np.nan, unalias=True):
        """
        Return a float value from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : float, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : float
        """
        return utils.get_float(self.get(key, unalias=unalias), default=default)

    def get_range(self, key, default=Range(), is_positive=False, unalias=True):
        """
        Return a Range value from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : Range, optional
            The default value to return if `key` does not exist in the options.
        is_positive : bool, optional
            If `True`, all values in the range are considered positive and any
            '-' character in `spec` will be treated as a delimiter rather than
            a minus sign.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : Range
        """
        return utils.get_range(self.get(key, unalias=unalias),
                               default=default, is_positive=is_positive)

    def get_list(self, key, default=None, unalias=True):
        """
        Return a list value from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : list, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : list
        """
        return utils.get_list(self.get(key, default=default, unalias=unalias))

    def get_string_list(self, key, delimiter=',', default=None, unalias=True):
        """
        Return a list of strings from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        delimiter : str, optional
            The string delimiter used to separate one element from the next.
        default : list, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : list (str)
        """
        return utils.get_string_list(
            self.get(key, default=None, unalias=unalias),
            default=default, delimiter=delimiter)

    def get_int_list(self, key, delimiter=',', default=None, unalias=True,
                     is_positive=False):
        """
        Return a list of integers from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        delimiter : str, optional
            The string delimiter used to separate one element from the next.
        default : list, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.
        is_positive : bool, optional
            If `True`, ranges may be specified using both ':' and '-'
            characters in a string.  Otherwise, the '-' character will imply
            a negative value.

        Returns
        -------
        value : list (int)
        """
        return utils.get_int_list(self.get(key, default=None, unalias=unalias),
                                  default=default, delimiter=delimiter,
                                  is_positive=is_positive)

    def get_float_list(self, key, delimiter=',', default=None, unalias=True):
        """
        Return a list of floats from the options for the given key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        delimiter : str, optional
            The string delimiter used to separate one element from the next.
        default : list, optional
            The default value to return if `key` does not exist in the options.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : list (float)
        """
        return utils.get_float_list(
            self.get(key, default=None, unalias=unalias),
            default=default, delimiter=delimiter)

    def get_dms_angle(self, key, default=np.nan, unalias=True):
        """
        Return a degree:minutes:seconds angle for the given options key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.  The value should be
            expected to be parsable as a degree:minutes:second angle according
            to :func:`utils.parse_angle`.
        default : int or float or units.Quantity, optional
            The default angle to return in cases where the `value` cannot be
            parsed correctly
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        value : units.Quantity
            The resolved angle in degrees.
        """
        return utils.get_dms_angle(self.get(key, default=None,
                                            unalias=unalias),
                                   default=default)

    def get_hms_time(self, key, angle=False, default=np.nan, unalias=True):
        """
        Return a hour:minutes:seconds angle for the given options key.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.  The value should be
            expected to be parsable as a degree:minutes:second angle according
            to :func:`utils.parse_angle`.
        angle : bool, optional
            If `True`, return an hour angle unit instead of hour unit.
        default : int or float or units.Quantity, optional
            The default angle to return in cases where the `value` cannot be
            parsed correctly
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        time : units.Quantity
            The resolved time in hours, or as an hour angle.
        """
        return utils.get_hms_time(self.get(key, default=None, unalias=unalias),
                                  default=default, angle=angle)

    def get_sign(self, key, default=0, unalias=True):
        """
        Return an integer representation of a sign value in the options.

        Parameters
        ----------
        key : str
            The name of the options value to retrieve.
        default : int, optional
            The default sign to return.
        unalias : bool, optional
            If `True`, unalias the key before attempting retrieval.

        Returns
        -------
        sign : int
            1 for a positive sign, -1 for a negative sign, and 0 for no sign.
        """
        return utils.get_sign(self.get(key, default=default, unalias=unalias))

    def handle_error(self, msg, error_class=ValueError):
        """
        Handle an error.

        If errors are allowed, will emit a log WARNING message and return if
        verbose is set.  If errors are not allowed, raises an error.

        Parameters
        ----------
        msg : str
            The message to emit by via log or in the raised error.
        error_class : class (BaseException)
            The error type to raise.

        Returns
        -------
        None

        Raises
        ------
        BaseException
            If errors are not permitted.
        """
        if self.allow_error:
            if self.verbose:
                log.warning(msg)
            else:
                log.debug(msg)
        elif not self.allow_error:
            raise error_class(msg)

    def update(self, options):
        """
        Update the stored options with another.

        Parameters
        ----------
        options : dict or ConfigObj
            The configuration options to read and parse.

        Returns
        -------
        None
        """
        opts = self.options_to_dict(options, add_singular=False)
        if opts is None:
            self.handle_error(f"Could not update with options: {options}")
            return
        self.options.merge(opts)

    @classmethod
    def stringify(cls, dictionary):
        """
        Set all values in a nested dictionary to strings.

        Parameters
        ----------
        dictionary : dict

        Returns
        -------
        dict
        """
        new_dict = {}
        for k, v in dictionary.items():
            if isinstance(v, dict):
                new_dict[k] = cls.stringify(v)
            elif isinstance(v, (list, tuple, set)):
                new_dict[k] = [str(x) for x in v]
            else:
                new_dict[k] = str(v)

        return new_dict

    @staticmethod
    def options_to_dict(options, add_singular=True):
        """
        Converts a single command to a dictionary.

        Parameters
        ----------
        options : str or dict or configobj.ConfigObj
            The options to convert to a dict.
        add_singular : bool, optional
            If `True`, and a string option just consists of the name, allow it
            to be added to the configuration as {'add': option} to be parsed
            accordingly during configuration validation.  Otherwise, `None`
            will be returned

        Returns
        -------
        dict_options : dict or None
            A dictionary or `None` if the options could not be parsed.
        """
        if isinstance(options, (dict, ConfigObj)):
            return options

        if isinstance(options, str):
            options = [x for x in [x.strip() for x in options.split('=')]
                       if x != '']
            if len(options) == 2:
                options = {options[0]: options[1]}
            elif len(options) == 1:
                # A single option will just be added to the configuration if
                # the condition is met.
                if add_singular:
                    options = {'add': options[0]}
                else:
                    return None
            else:
                return None
            return options

        return None

    @classmethod
    def merge_options(cls, current, new):
        """
        Merge new options into the current options.

        Merging new conditions into the currently existing options is slightly
        complicated.  Command key values must be appended to any currently
        existing commands keys in the current conditions as a list.  There is
        also additional handling in place to ensure conditions are updated
        appropriately since branches may also contain specific values rather
        than just a simply dictionary of values.  E.g., if we currently have
        {'sky': True} and we want to add the options {'sky': {'gain': '0.3:3'}}
        then the output options should be:
        {'sky': {'value': True, 'gain':'0.3:3'}}}.

        Parameters
        ----------
        current : ConfigObj
        new : dict or ConfigObj

        Returns
        -------
        None
        """
        for key, value in new.items():
            if key not in current:
                if isinstance(value, dict):
                    current[key] = ConfigObj()
                    current[key].merge(value)
                elif key in cls.append_keys:
                    if isinstance(value, list):
                        current[key] = value.copy()
                    else:
                        current[key] = value
                else:
                    current[key] = value
            else:
                current_value = current[key]
                if key in cls.append_keys:
                    if not isinstance(value, list):
                        value = [value]
                    if not isinstance(current_value, list):
                        current[key] = [current_value]
                        current_value = current[key]
                    for x in value:
                        if x not in current_value:
                            current_value.append(x)
                elif isinstance(current_value, (dict, ConfigObj)):
                    if isinstance(value, dict):
                        cls.merge_options(current_value, value)
                    else:
                        current_value['value'] = value
                elif isinstance(value, dict):
                    current[key] = {'value': current_value}
                    current[key].merge(value)
                else:
                    current[key] = value
