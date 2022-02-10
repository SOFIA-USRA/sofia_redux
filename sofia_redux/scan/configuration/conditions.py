# Licensed under a 3-clause BSD style license - see LICENSE.rst

import configobj
import re

from sofia_redux.scan.configuration.options import Options

__all__ = ['Conditions']


class Conditions(Options):

    append_keys = ('blacklist', 'whitelist', 'forget', 'recall',
                   'lock', 'unlock', 'add', 'config')

    def __init__(self, allow_error=False, verbose=True):
        """
        Initialize a configuration conditions object.
        """
        super().__init__(allow_error=allow_error, verbose=verbose)

    def __len__(self):
        """
        Return the number of conditions available.

        Returns
        -------
        int
        """
        if not hasattr(self.options, '__len__'):
            return 0
        return len(self.options)

    def __setitem__(self, requirement, options):
        """
        Set the options for a given requirement.

        Parameters
        ----------
        requirement : str
            The requirement for the condition to be met.  Should be of the form
            <key><operator><value>.
        options : str or dict or configobj.ConfigObj
            The options to apply once the condition is met.  If a single string
            is provided, it should be of the form {<command_or_key>=<value>} or
            <value>.  If only <value> is provided, the associated command will
            be changed to add=<value>.

        Returns
        -------
        None
        """
        self.set(requirement, options)

    def __str__(self):
        """
        Return a string representation of the configuration.

        Returns
        -------
        str
        """
        size = self.size
        return f'Contains {size} condition{"s" if size != 1 else ""}.'

    def __repr__(self):
        """
        Return a string representation of the configuration.

        Returns
        -------
        str
        """
        return super().__repr__() + f' {self}'

    @property
    def size(self):
        """
        Return the number of conditions available.

        Returns
        -------
        int
        """
        return self.__len__()

    def copy(self):
        """
        Return a copy of the conditions.

        Returns
        -------
        Conditions
        """
        return super().copy()

    def set(self, requirement, options):
        """
        Set a condition in the options.

        Parameters
        ----------
        requirement : str
           Typically a requirement of the form key=value.
        options : dict or ConfigObj
           The options to apply if the condition is met.

        Returns
        -------
        None
        """
        condition_options = self.options_to_dict(options)
        if condition_options is None:
            self.handle_error(f"Could not parse condition "
                              f"[{requirement}]: {options}")
            return

        if requirement not in self.options:
            self.options[requirement] = configobj.ConfigObj()

        self.merge_options(self.options[requirement], condition_options)

    def update(self, configuration_options):
        """
        Update the stored conditions with those from another configuration.

        Parameters
        ----------
        configuration_options : dict or ConfigObj
            The configuration options to read and parse.

        Returns
        -------
        None
        """
        if 'conditionals' not in configuration_options:
            return
        options = configuration_options['conditionals']
        if not isinstance(options, dict):
            self.handle_error(f"Supplied conditionals must be {dict} type.")
            return

        for key, value in options.items():
            self.set(key, value)

    def check_requirement(self, configuration, requirement):
        r"""
        Checks requirements and returns if met

        Conditions must be of the form <thing><operator><required_value>,
        where operator must be one of '=', '!=', '<', '<=', '>', '>='.

        Alternatively, a single value may be supplied.  If set in the
        configuration, True is returned

        Parameters
        ----------
        configuration : Configuration
            The configuration in which to check the requirement.
        requirement : str
            The requirement to check.

        Returns
        -------
        bool
        """
        if re.search(r'(?<![<!>])=', requirement):  # equals
            operator = '='
        elif re.search(r'<(?!=)', requirement):  # less than
            operator = '<'
        elif re.search(r'>(?!=)', requirement):  # greater than
            operator = '>'
        elif '!=' in requirement:
            operator = '!='
        elif '<=' in requirement:
            operator = '<='
        elif '>=' in requirement:
            operator = '>='
        else:
            operator = None

        # If there is no operator, assume it is a boolean switch.
        if operator is None:
            return configuration.get_bool(requirement.strip(), default=False)

        s = [s.strip() for s in requirement.split(operator)]
        s = [x for x in s if x != '']
        if len(s) != 2:
            self.handle_error(f"Bad conditional requirement: {requirement}")
            return False

        key, test_value = s
        value = configuration.get(key, default=None)
        if value is None:
            return False

        value = str(value).strip()

        if operator == '=' or operator == '==':
            try:
                value = float(value)
                test_value = float(test_value)
                return value == test_value
            except (ValueError, TypeError):
                return value == test_value

        if operator == '!=':
            try:
                value = float(value)
                test_value = float(test_value)
                return value != test_value
            except (ValueError, TypeError):
                return value != test_value

        try:
            value = float(value)
            test_value = float(test_value)
        except (ValueError, TypeError):
            return False

        try:
            return eval('%f %s %f' % (value, operator, test_value))
        except (ValueError, TypeError):  # pragma: no cover
            return False

    def get_met_conditions(self, configuration):
        """
        Return the actions for met conditions.

        Check a configuration with all conditions and return those that are
        fulfilled.  Conditions in the options that follow the standard
        format of {requirement (str): actions (dict)} will have the requirement
        checked with the configuration.  However, condition options that are of
        the form {requirement (str): actions (str)} will always be parsed
        assuming that the requirement is met due to the complexities of the
        configuration structure and always returned in the output options.

        Parameters
        ----------
        configuration : Configuration

        Returns
        -------
        actions : dict
            A dict of form {requirement: actions}.
        """
        apply_actions = {}
        for requirement, actions in self.options.items():

            if isinstance(actions, dict):
                requirement_met = self.check_requirement(
                    configuration, requirement)
            else:
                # in rare instances there is no requirement
                requirement_met = True
                s = [s.strip() for s in actions.split('=')]
                s = [x for x in s if x != '']
                if len(s) != 2:
                    self.handle_error(f"Bad condition: {actions}")
                    continue
                actions = {s[0]: s[1]}

            if not requirement_met:
                continue

            if requirement not in apply_actions:
                apply_actions[requirement] = []

            requirement_actions = apply_actions[requirement]
            if actions not in requirement_actions:
                requirement_actions.append(actions)

        return apply_actions

    def process_conditionals(self, configuration, seen=None):
        """
        Process all conditions until no further changes are required.

        Parameters
        ----------
        configuration : Configuration
        seen : set, optional
            A set of previously applied conditions and actions.  Each member
            should be a tuple of the form (requirement, command, action).

        Returns
        -------
        None
        """
        if seen is None:
            seen = set([])
        while self.update_configuration(configuration, seen=seen):
            pass

    def update_configuration(self, configuration, seen=None):
        """
        Update the configuration with any met conditions.

        Parameters
        ----------
        configuration : Configuration
        seen : set, optional
            A set of previously applied conditions and actions

        Returns
        -------
        updated : bool
            `True` if the configuration was updated, and `False` otherwise.
        """
        if seen is None:
            seen = set([])

        apply_actions = self.get_met_conditions(configuration)
        if len(apply_actions) == 0:
            return False

        contains_update = False
        for requirement, actions in apply_actions.items():
            configuration.applied_conditions.add(requirement)

            commands = {}

            for action in actions:
                for key, value in action.items():
                    str_val = str(value)
                    check_command = (requirement, key, str_val)
                    if check_command in seen:
                        continue
                    contains_update = True
                    seen.add(check_command)

                    if key in configuration.command_keys:
                        if key in commands:
                            commands[key].append(value)  # pragma: no cover
                        else:
                            commands[key] = [value]
                    else:
                        if 'update' not in commands:
                            commands['update'] = {key: value}
                        else:
                            commands['update'][key] = value

            configuration.apply_commands(commands)

        return contains_update
