# Licensed under a 3-clause BSD style license - see LICENSE.rst

import configobj
from copy import deepcopy

from sofia_redux.scan.configuration.options import Options

__all__ = ['ObjectOptions']


class ObjectOptions(Options):

    append_keys = ('blacklist', 'whitelist', 'forget', 'recall',
                   'lock', 'unlock', 'add', 'config')

    def __init__(self, allow_error=False, verbose=True):
        """
        Initialize an ObjectOptions object.

        The object options contain configuration settings pertaining to the
        observed source object.
        """
        super().__init__(allow_error=allow_error, verbose=verbose)
        self.applied_objects = []

    def copy(self):
        """
        Return a copy of the ObjectOptions.

        Note that any applied objects will not be included in the copy.

        Returns
        -------
        ObjectOptions
        """
        new = ObjectOptions()
        new.options = deepcopy(self.options)
        return new

    def clear(self):
        """
        Clear all options.

        Returns
        -------
        None
        """
        super().clear()
        self.applied_objects = []

    def __getitem__(self, source):
        """
        Retrieve the options for a given source.

        Parameters
        ----------
        source : str
            The name of the source for which to extract options.

        Returns
        -------
        options : configobj.ConfigObj
        """
        result = self.get(source)
        return result

    def __setitem__(self, source, options):
        """
        Set the options for a given source.

        Parameters
        ----------
        source : str
            The name of the source for which the options apply.
        options : dict or configobj.ConfigObj
            The options to set for `source`.

        Returns
        -------
        None
        """
        source_options = self.options_to_dict(options, add_singular=True)
        if source_options is None:
            self.handle_error(f"Could not parse source options for {source}: "
                              f"{options}")
            return
        self.set(source, source_options)

    def __str__(self):
        """
        Return a string representation of the SourceOptions.

        Returns
        -------
        str
        """
        s = 'Available object configurations:'
        for k in self.options.keys():
            s += f'\n{k}'
        return s

    def __repr__(self):
        """
        Return a string representation of the SourceOptions.

        Returns
        -------
        str
        """
        return f'{super().__repr__()}\n{self}'

    def get(self, source, default=None, unalias=True):
        """
        Retrieve the options for a given source.

        Parameters
        ----------
        source : str
            The name of the observed source.
        default : dict or configobj.ConfigObj, optional
            A value to return if no results are found.  Must be of dict or
            configobj.ConfigObj type to be returned.
        unalias : bool, optional
            Not used by the ObjectOptions.

        Returns
        -------
        options : configobj.ConfigObj
        """
        source = str(source).strip().lower()
        result = configobj.ConfigObj()
        if source in self.options:
            result.merge(self.options[source])
        if len(result) == 0 and isinstance(default, dict):
            return default
        return result

    def set(self, source, options):
        """
        Set the options for a given source.

        Parameters
        ----------
        source : str
            The name of the observed source.
        options : dict or configobj.ConfigObj
            The configuration options applicable to the source.

        Returns
        -------
        None
        """
        source = str(source).strip().lower()
        if source not in self.options:
            self.options[source] = configobj.ConfigObj()
        source_options = self.options_to_dict(options, add_singular=True)
        if source_options is None:
            self.handle_error(f"Could not parse options for {source} source: "
                              f"{options}")
            return

        self.merge_options(self.options[source], source_options)

    def update(self, configuration_options):
        """
        Update the object options.

        Parameters
        ----------
        configuration_options : dict or configobj.ConfigObj
            The options to apply.  These must contain an 'object' key
            and values in order to have an effect.

        Returns
        -------
        None
        """
        if 'object' not in configuration_options:
            return
        opts = configuration_options['object']
        options = self.options_to_dict(opts)
        if options is None:
            self.handle_error(f"Could not parse object options: {opts}")
            return

        for key, value in options.items():
            self.set(key, value)

    def set_object(self, configuration, source_name, validate=True):
        """
        Set the object options for a source in the supplied configuration.

        Parameters
        ----------
        configuration : Configuration
            The configuration in which to apply options for `source_name`.
        source_name : str
            The name of the source for which to apply options.
        validate : bool, optional
            If `True`, validate the configuration after the source options have
            been applied

        Returns
        -------
        None
        """
        options = self.get(source_name)
        if options is None or len(options) == 0:
            return
        if source_name not in self.applied_objects:
            self.applied_objects.append(source_name)
        configuration.apply_configuration_options(options, validate=validate)
