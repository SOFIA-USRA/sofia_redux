# Licensed under a 3-clause BSD style license - see LICENSE.rst

import configobj
import re
from copy import deepcopy

from sofia_redux.scan.configuration.options import Options
from sofia_redux.toolkit.utilities.multiprocessing import in_windows_os

__all__ = ['Aliases']


class Aliases(Options):

    substitute_pattern = r'(?<={\?).*?(?=})'

    def __init__(self, allow_error=False, verbose=True):
        super().__init__(allow_error=allow_error, verbose=verbose)

    def copy(self):
        """
        Return a copy of the aliases.

        Returns
        -------
        Aliases
        """
        return deepcopy(self)

    def unalias_dot_string(self, key):
        """
        Unalias all configuration paths in a given key.

        An alias is a name in the configuration used to represent another
        configuration keyword.  For example, if correlated.sky was aliased to
        sky, then alias.unalias_dot_string('sky.mode') would return
        'correlated.sky.mode'.

        Parameters
        ----------
        key : str

        Returns
        -------
        str
        """
        if not isinstance(key, str):
            self.handle_error(f"Unalias_dot_string: require string input. "
                              f"Received: {type(key)}")
            return

        if self.size == 0:
            return key
        branches = key.split('.')
        if branches[0] not in self.options:
            return key
        branches[0] = self.options[branches[0]]
        key = '.'.join(branches)
        return self.unalias_dot_string(key)  # Recursive

    def unalias_branch(self, options):
        """
        Unalias a single options branch of the configuration.

        Returns a potentially multi-levelled dictionary for a given single
        levelled dictionary if the key is currently aliased.  For example,
        if 'sky' was aliased to 'correlated.sky', then passing
        {'sky': {'gainrange': '0.3:3'}} into unalias_branch would return
        {'correlated': {'sky': {'gainrange': '0.3:3'}}}.

        Parameters
        ----------
        options : ConfigObj or dict
            The options branch to unalias.  Must be of length 1 (only contain
            a single key).

        Returns
        -------
        unaliased_options : dict
        """
        if not isinstance(options, dict):
            self.handle_error(f"Unalias_options: require dict input. "
                              f"Received: {type(options)}")
            return

        items = list(options.items())
        if len(items) != 1:
            self.handle_error("Multiple branches passed to unalias_branch.")
            return

        key, value = items[0]
        branch_path = self.unalias_dot_string(key).split('.')
        if branch_path[0] == key:  # No alias found
            return options
        result = dict()
        current = result
        for branch in branch_path[:-1]:
            current[branch] = dict()
            current = current[branch]
        current[branch_path[-1]] = value
        return result

    @classmethod
    def unalias_value(cls, configuration, value):
        """
        Unalias a value from the configuration.

        Aliased values in the configuration may be defined in the configuration
        by using {?key} in place of the value where key represents another
        configuration value.  For example, if the configuration contains the
        option my_value = abc, passing {?my_value} into unalias_value would
        return abc.

        Parameters
        ----------
        configuration : Configuration
            The configuration to examine when unaliasing the given value.
        value : str or object
            Only string values will be unaliased.  Anything else will be
            returned in the output.

        Returns
        -------
        unaliased_value : str
        """
        if not isinstance(value, str):
            return value
        if '{?' not in value:
            return value

        substitutions = set(re.findall(cls.substitute_pattern, value))
        if len(substitutions) == 0:
            return value

        for key in substitutions:
            s = configuration.get(key)
            if s is None:
                continue
            # workaround for Windows paths
            if in_windows_os() and '\\' in s:  # pragma: no cover
                s = re.escape(s)
            replace = r'{\?' + key + '}'
            value = re.sub(replace, s, value)

        return value

    @classmethod
    def unalias_branch_values(cls, configuration, branch, copy=True):
        """
        Unalias all values in an options branch.

        Parameters
        ----------
        configuration : Configuration
            The configuration used to unalias values.
        branch : dict or ConfigObj or str
            Either a configuration options branch to unalias, or a single
            string value to unalias.
        copy : bool, optional
            If `True`, return a copy of the branch.  Otherwise, the given
            branch will be updated in place.

        Returns
        -------
        new_branch : dict or ConfigObj or str
            The updated branch.
        """
        if copy:
            branch = deepcopy(branch)
        if not isinstance(branch, dict):
            return cls.unalias_value(configuration, branch)
        for key, value in branch.items():
            if isinstance(branch, dict):
                branch[key] = cls.unalias_branch_values(
                    configuration, value, copy=False)
        return branch

    def update(self, configuration_options):
        """
        Update the stored aliases with those from another configuration.

        Parameters
        ----------
        configuration_options : dict or ConfigObj
            The configuration options to read and parse.

        Returns
        -------
        None
        """
        if 'aliases' not in configuration_options:
            return
        super().update(configuration_options['aliases'])

    def resolve_configuration(self, configuration):
        """
        Resolve all configuration branch names.

        All branches within the Configuration options are checked such that any
        aliased branch names will be replaced by the correct unaliased branch
        paths/names.  This requires iteration due to the possibility of
        certain one alias referencing a branch that has not yet been unaliased,
        but will be.

        Parameters
        ----------
        configuration : Configuration

        Returns
        -------
        None
        """
        changed = True
        while changed:
            unaliased_options = configobj.ConfigObj()
            for key, value in configuration.options.items():
                alias = self(key)
                if alias != key:
                    unaliased_options[alias] = value
                    try:
                        del configuration.options[key]
                    except KeyError:  # pragma: no cover
                        pass

            changed = len(unaliased_options) > 0
            if changed:
                configuration.apply_configuration_options(unaliased_options)

    def __call__(self, thing):
        """
        Unalias whatever value is passed into the alias.

        Parameters
        ----------
        thing : str or dict or ConfigObj
            The object to unalias.

        Returns
        -------
        dict or str or ConfigObj
        """
        if isinstance(thing, str):
            return self.unalias_dot_string(thing)
        elif isinstance(thing, dict):
            return self.unalias_branch(thing)
        else:
            self.handle_error(f"Must supply {str} or {dict} type object.")
