# Licensed under a 3-clause BSD style license - see LICENSE.rst

import configobj
import numpy as np

from sofia_redux.scan.configuration.options import Options

__all__ = ['IterationOptions']


class IterationOptions(Options):

    append_keys = ('blacklist', 'whitelist', 'forget', 'recall',
                   'lock', 'unlock', 'add', 'config')

    def __init__(self, allow_error=False, verbose=True, max_iteration=None):
        """
        Initialize an IterationOptions object.

        The iterations options object contains configuration settings
        pertaining to the current SOFSCAN iteration number.

        Parameters
        ----------
        max_iteration : int, optional
            The maximum number of iterations available for a SOFSCAN reduction.
        """
        super().__init__(allow_error=allow_error, verbose=verbose)
        self._max_iteration = None
        self.current_iteration = None
        self._rounds_locked = False
        if max_iteration is not None:
            self._max_iteration = int(max_iteration)

    def copy(self):
        """
        Return a copy of the iteration options.

        Returns
        -------
        IterationOptions
        """
        return super().copy()

    def clear(self):
        """
        Clear all options.

        Returns
        -------
        None
        """
        super().clear()
        self._max_iteration = None
        self.current_iteration = None

    def __getitem__(self, iteration):
        """
        Retrieve the options for a given iteration.

        Parameters
        ----------
        iteration : int or float or str
            The iteration for which to retrieve options.

        Returns
        -------
        options : configobj.ConfigObj
        """
        result = self.get(iteration)
        return result

    def __setitem__(self, iteration, options):
        """
        Set the options for a given iteration.

        Parameters
        ----------
        iteration : int or float or str
            The iteration for which to set options.
        options : dict or configobj.ConfigObj
            The options to set.

        Returns
        -------
        None
        """
        iteration_options = self.options_to_dict(options, add_singular=True)
        if iteration_options is None:
            self.handle_error(f"Could not parse iteration {iteration} option: "
                              f"{options}")
            return

        self.set(iteration, iteration_options)

    def __str__(self):
        """
        Return a string representation of the IterationOptions.

        Returns
        -------
        str
        """
        s = 'Iteration configurations:'
        s += f'\nMaximum iterations: {self.max_iteration}'
        switches = ', '.join([str(s) for s in self.options.keys()])
        s += f'\nIteration switches: {switches}'
        return s

    def __repr__(self):
        """
        Return a string representation of the IterationOptions.

        Returns
        -------
        str
        """
        return f'{super().__repr__()}\n{self}'

    @property
    def max_iteration(self):
        """
        Return the maximum number of iterations for the current reduction.

        Returns
        -------
        int
        """
        return self._max_iteration

    @max_iteration.setter
    def max_iteration(self, iteration):
        """
        Set the maximum number of iterations for the current reduction.

        Parameters
        ----------
        iteration : int

        Returns
        -------
        None
        """
        if self.rounds_locked:
            return
        if iteration is None:
            self._max_iteration = None
        else:
            self._max_iteration = int(iteration)

    @property
    def rounds_locked(self):
        """
        Return whether the maximum number of iterations is locked.

        Returns
        -------
        bool
        """
        return self._rounds_locked

    def lock_rounds(self, maximum_iterations=None):
        """
        Lock the current maximum number of iterations in-place.

        Returns
        -------
        None
        """
        if self.rounds_locked:
            return
        if maximum_iterations is not None:
            self.max_iteration = maximum_iterations
        self._rounds_locked = True

    def unlock_rounds(self):
        """
        Allow the maximum number of iterations to be changed.

        Returns
        -------
        None
        """
        self._rounds_locked = False

    def parse_iteration(self, iteration):
        r"""
        Returns an iteration value as either an int or float value.

        Parses string iterations including "first", "last", "final" to
        integer of float representations.  Can also parse percentages
        such as 90%.  A float iteration represents a fraction of the
        maximum iteration, and must therefore be between 0 and 1.
        Raises an error if fractional iteration is out of range, or
        value could not be parsed.

        Parameters
        ----------
        iteration : str or int or float

        Returns
        -------
        iteration : int or float
            If the return value is a float, it will be between 0 and 1 and
            represents the fraction of the maximum iterations.  Negative
            integers represent maximum_iteration - iteration.
        """
        if isinstance(iteration, str):
            iteration = iteration.strip().lower()
            try:
                if iteration.endswith('%'):
                    iteration = float(iteration[:-1]) / 100.0
                elif '.' in iteration:
                    iteration = float(iteration)
                elif iteration in ['last', 'final']:
                    iteration = -1
                elif iteration == 'first':
                    iteration = 1
                else:
                    iteration = int(iteration)
            except Exception as err:
                self.handle_error(f"Could not parse iteration string: "
                                  f"{iteration}\nError: {err}")
                return None

        if not isinstance(iteration, (int, float)):
            self.handle_error(f"iteration must be {str}, {int}, or {float}.")
            return None

        if isinstance(iteration, float):
            if iteration < 0 or iteration > 1:
                msg = "Fractional iterations must be in the range [0, 1]."
                self.handle_error(msg)
                return None

        return iteration

    def get(self, iteration, default=None, unalias=True):
        """
        Retrieve configuration options for a given date.

        Parameters
        ----------
        iteration : str or int or float
            The observation date.  If a string is used, it should be in ISOT
            format in UTC scale.  Integers and floats will be parsed as
            MJD times in the UTC scale.
        default : dict or configobj.ConfigObj, optional
            A value to return if no results are found.  Must be of dict or
            configobj.ConfigObj type to be returned.
        unalias : bool, optional
            Not used by the IterationOptions.

        Returns
        -------
        options : ConfigObj
        """
        iteration = int(iteration)  # must be an integer
        options = configobj.ConfigObj()
        for check_iteration, iteration_options in self.options.items():
            relative_iteration = self.relative_iteration(check_iteration)
            if iteration != relative_iteration:
                continue

            self.merge_options(options, iteration_options)

        if len(options) == 0 and isinstance(default, dict):
            return default

        return options

    def set(self, iteration, options):
        """
        Set the options for a given iteration.

        Parameters
        ----------
        iteration : str or int or float
            The iteration number or relative iteration number.  Please see
            :func:`IterationOptions.parse_iteration` for further details.
        options : dict or configobj.ConfigObj
            The options to set for an iteration.

        Returns
        -------
        None
        """
        iteration = self.parse_iteration(iteration)
        if iteration is None:
            return
        iteration = str(iteration)

        if iteration not in self.options:
            self.options[iteration] = configobj.ConfigObj()
        options = self.options_to_dict(options, add_singular=True)
        self.merge_options(self.options[iteration], options)

    def relative_iteration(self, iteration):
        """
        Return an iteration number relative to the current maximum.

        Parameters
        ----------
        iteration : str or int or float
            The iteration number to parse.  Please see
            :func:`IterationOptions.parse_iteration` for further details.

        Returns
        -------
        int or None
            None will be returned if an iteration relative to a maximum
            iteration is required, but no maximum iteration has been set.
        """
        iteration = self.parse_iteration(iteration)
        if isinstance(iteration, float):
            if self.max_iteration is None:
                return None
            else:
                return int(np.round(self.max_iteration * iteration))
        if isinstance(iteration, int):
            if iteration < 0:
                if self.max_iteration is None:
                    return None
                else:
                    return self.max_iteration + 1 + iteration
            else:
                return iteration
        else:
            self.handle_error(f"Relative iterations must be {int}, {float} or "
                              f"can be converted to such from a {str}")
            return None

    def update(self, configuration_options):
        """
        Update the iteration options with settings from another.

        Parameters
        ----------
        configuration_options : dict or configobj.ConfigObj
            The options used to update these options.  Must contain an
            'iteration' key and values to have an effect.

        Returns
        -------
        None
        """
        if 'iteration' not in configuration_options:
            return
        opts = configuration_options['iteration']
        options = self.options_to_dict(opts, add_singular=False)
        if options is None:
            self.handle_error(f"Supplied iteration options could not be "
                              f"parsed: {opts}")
            return

        for key, value in options.items():
            self.set(key, value)

    def set_iteration(self, configuration, iteration, validate=True):
        """
        Set options for a given iteration in a configuration.

        Parameters
        ----------
        configuration : Configuration
            The configuration in which to apply the iteration options.
        iteration : int or str or float
            The iteration for which to apply options.
        validate : bool, optional
            If `True`, validate the configuration once the iteration options
            have been applied.

        Returns
        -------
        None
        """
        iteration = self.relative_iteration(iteration)
        self.current_iteration = iteration
        if iteration is None:
            return
        options = self.get(iteration)
        configuration.apply_configuration_options(options, validate=validate)
