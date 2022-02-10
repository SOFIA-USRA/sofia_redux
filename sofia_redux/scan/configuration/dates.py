# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy.time import Time
from configobj import ConfigObj
from copy import deepcopy

from sofia_redux.scan.configuration.options import Options

__all__ = ['DateRangeOptions', 'DateRange']


class DateRangeOptions(Options):

    append_keys = ('blacklist', 'whitelist', 'forget', 'recall',
                   'lock', 'unlock', 'add', 'config')

    def __init__(self, allow_error=False, verbose=True):
        """
        Initialize a DatRangeOptions object.

        This class contains configuration options for specific dates and times.
        Multiple may be read from a configuration and then applied for a
        given time.
        """
        super().__init__(allow_error=allow_error, verbose=verbose)
        self.ranges = {}
        self.retrieved = None  # A copy of last retrieved options
        self.current_date = None

    def copy(self):
        """
        Return a copy of the date range options.

        Returns
        -------
        DateRangeOptions
        """
        return super().copy()

    def __setitem__(self, date, option):
        """
        Set options for a given date.

        Parameters
        ----------
        date : str or int or float
            The date for which to set options.  If a string is used, it should
            be in ISOT format in UTC scale.  Integers and floats will be
            parsed as MJD times in the UTC scale.
        option : dict or ConfigObj
            The options to set for `date`.

        Returns
        -------
        None
        """
        self.set(date, option)

    def __getitem__(self, date):
        """
        Return options for a given date.

        Parameters
        ----------
        date : str or int or float
            The date for which to retrieve options.  If a string is used, it
            should be in ISOT format in UTC scale.  Integers and floats will be
            parsed as MJD times in the UTC scale.

        Returns
        -------
        options : dict
        """
        return self.get(date)

    def __str__(self):
        """
        Return a string representation of the date range options.

        Returns
        -------
        str
        """
        s = 'Available date ranges (UTC):'
        for r in self.ranges.values():
            s += '\n' + str(r)
        return s

    def __repr__(self):
        """
        Return a string representation of the date range options.

        Returns
        -------
        str
        """
        return f'{super().__repr__()}\n{self}'

    def clear(self):
        """
        Clear all options.

        Returns
        -------
        None
        """
        super().clear()
        self.ranges = {}
        self.retrieved = None
        self.current_date = None

    def update(self, configuration_options):
        """
        Update the stored date options from a supplied configuration.

        Parameters
        ----------
        configuration_options : dict or ConfigObj
            The configuration options to read and parse for any date settings.
            Note that for any updates to occur, `configuration_options` must
            contain a "date" key.

        Returns
        -------
        None
        """
        if 'date' not in configuration_options:
            return
        opts = configuration_options['date']
        options = self.options_to_dict(opts)
        if options is None:
            self.handle_error(
                f"Supplied date options could not be parsed: {opts}.")
            return

        for date, option in options.items():
            date_options = self.options_to_dict(option)
            if date_options is None:
                msg = f"Could not parse options for date [{date}]: {option}"
                self.handle_error(msg)
                continue
            self.set(date, date_options)

    def get(self, date, default=None, unalias=True):
        """
        Retrieve configuration options for a given date.

        Parameters
        ----------
        date : str or int or float
            The observation date.  If a string is used, it should be in ISOT
            format in UTC scale.  Integers and floats will be parsed as
            MJD times in the UTC scale.
        default : dict or ConfigObj, optional
            A value to return if no results are found.  Must be of dict or
            ConfigObj type to be returned.
        unalias : bool, optional
            Not used to the DateRangeOptions.

        Returns
        -------
        options : ConfigObj
        """
        result = ConfigObj()
        date = DateRange.to_time(date)
        for key, date_range in self.ranges.items():
            if date in date_range:
                self.merge_options(result, self.options[key])
                # result.merge(self.options[key])

        if isinstance(default, dict) and len(result) == 0:
            return default

        self.retrieved = {date: result.dict()}
        return result

    def set(self, key, options):
        """
        Set options in the date range options for a certain date-time.

        Parameters
        ----------
        key : str
            A date key of the form start--stop where start and stop can be
            either * indicating no value, an ISOT str or mjd number value, both
            in the UTC scale.
        options : dict or ConfigObj
            The configuration options for the given date range `key`.

        Returns
        -------
        None
        """
        try:
            self.ranges[key] = DateRange(key)
        except Exception as error:
            self.handle_error(f"Invalid date range key: {error}")
            return

        if key in self.options:
            self.merge_options(self.options[key], options)
        else:
            self.options[key] = options

    def set_date(self, configuration, date, validate=True):
        """
        Apply options for a given date to a configuration.

        Parameters
        ----------
        configuration : Configuration
        date : str or int or float
            The date for which to retrieve and apply options.  If a string is
            used, it should be in ISOT format in UTC scale.  Integers and
            floats will be parsed as MJD times in the UTC scale.
        validate : bool, optional
            If `True`, validate the configuration once options have been
            applied.

        Returns
        -------
        None
        """
        options = self.get(date)
        self.current_date = deepcopy(date)  # could be anything
        configuration.apply_configuration_options(options, validate=validate)


class DateRange(ABC):

    def __init__(self, text):
        """
        Initialize a date range from a text string.

        The date range consists of a start and end date time.

        Parameters
        ----------
        text : str
            A date key of the form start--stop where start and stop can be
            either * indicating no value, an ISOT str or mjd number value, both
            in the UTC scale.
        """
        self.range = [None, None]
        self.parse_range(text)

    def copy(self):
        """
        Return a copy of the DateRange object.

        Returns
        -------
        DateRange
        """
        new = DateRange('*--*')
        for i in range(2):
            new.range[i] = deepcopy(self.range[i])
        return new

    def __str__(self):
        """
        Return a string representation of the DateRange.

        Returns
        -------
        str
        """
        if self.range[0] is None:
            start_string = '*' * 23
        else:
            start_string = self.range[0].isot
        if self.range[1] is None:
            end_string = '*' * 23
        else:
            end_string = self.range[1].isot

        return f'{start_string}--{end_string}'

    def __contains__(self, thing):
        """
        Return whether a date-time is contained in the DateRange.

        Parameters
        ----------
        thing : str or int or float or Time
            The date for which to set options.  If a string is used, it should
            be in ISOT format in UTC scale.  Integers and floats will be
            parsed as MJD times in the UTC scale.

        Returns
        -------
        time_in_range : bool
            `True` if the date exists in the time range, `False` otherwise.
            Note that the date range is exclusive so that `False` will be
            returned when the input date coincides exactly with the start or
            end date-time of the DateRange.
        """
        time = self.to_time(thing)
        if self.range[0] is not None:
            if time < self.range[0]:
                return False

        if self.range[1] is not None:
            if time > self.range[1]:
                return False

        return True

    def parse_range(self, text):
        """
        Parse and apply a string containing the date range.

        Parameters
        ----------
        text : str
            A date key of the form start--stop where start and stop can be
            either * indicating no value, an ISOT str or mjd number value, both
            in the UTC scale.

        Returns
        -------
        None
        """
        times = [s.strip() for s in text.split('--')]
        times = [x for x in times if x != '']
        if len(times) not in [1, 2]:
            raise ValueError(f"Cannot parse time range: {text}")
        if len(times) == 1:
            times = times * 2

        for i in range(2):
            if times[i] == '*':
                self.range[i] = None
            else:
                try:
                    self.range[i] = Time(times[i], format='isot', scale='utc')
                except ValueError:  # assume mjd's instead
                    try:
                        self.range[i] = Time(
                            float(times[i]), format='mjd', scale='utc')
                    except ValueError:
                        raise ValueError(f"Could not parse date: {times[i]}")

    @staticmethod
    def to_time(thing):
        """
        Convert a date-time like object to a consistent time object.

        Parameters
        ----------
        thing : str or int or float or Time
            The object to convert.   If a string is used, it should be in ISOT
            format in UTC scale.  Integers and floats will be parsed as MJD
            times in the UTC scale.

        Returns
        -------
        time : Time
        """
        if isinstance(thing, Time):
            return thing
        elif isinstance(thing, str):
            return Time(thing, format='isot', scale='utc')
        elif isinstance(thing, (int, float)):
            return Time(thing, format='mjd', scale='utc')
        else:
            raise ValueError(f"Input must be of type {{{str}, {int}, "
                             f"{float}, {Time}}}")
