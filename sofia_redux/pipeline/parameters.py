# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Base classes for Redux parameter sets."""

from collections import OrderedDict
from copy import deepcopy
import os
import re

import configobj
from astropy import log

# Falsy string values, used for type fixing and comparison
FALSY = ['false', '0', '0.0', '', 'None']


class ParameterSet(OrderedDict):
    """
    Ordered dictionary of parameter values for a reduction step.

    Sensible defaults are defined for all parameter fields in the
    `set_param` method.
    """
    def set_param(self, key='', value=None, dtype='str',
                  wtype=None, name=None,
                  options=None, option_index=0,
                  description=None, hidden=False):
        """
        Set a parameter key-value pair.

        All input parameters are optional, although key values
        are necessary if more than one parameter is to be defined.
        The default values will create a string-valued parameter
        with an associated text-box widget.  The default name
        matches the key value.

        Parameters
        ----------
        key : str, optional
            The key for the parameter.
        value : str, int, float, bool, or list; optional
            The value of the parameter.  Should match dtype if provided.
        dtype : {'str', 'int', 'float', 'bool', 'strlist', \
            'intlist', 'floatlist', 'boollist'}; optional
            Data type of the parameter value.  Basic data types understood
            by Redux are str, int, float, and bool, and lists of the same.
            Any other data type will be treated as a string.
        wtype : {'text_box', 'check_box', 'combo_box', 'radio_button', \
            'pick_file', 'pick_directory', 'group'}; optional
            Widget type to be used for editing the parameter in a GUI
            context.  Ignored in command-line context.
        name : str, optional
            Display name or text for the parameter.
        options : `list`, optional
            Enumerated list of possible values for the parameter.  These
            values will be used as the displayed options in combo_box
            or radio_button widget types.
        option_index : int, optional
            If `options` is provided, this parameter sets the default
            selection.
        description : str, optional
            Description of the parameter.  In GUI context, the description
            will be shown in a tooltip when the parameter widget is
            hovered over.
        hidden : bool, optional
            If True, the parameter will be provided to the reduction
            step, but will not be displayed or editable by the user.
        """
        if name is None:
            name = key
        if (value is None
                and options is not None
                and option_index is not None):
            try:
                value = options[option_index]
            except IndexError:
                pass
        # if not specified
        if wtype is None:
            if options is not None:
                wtype = 'combo_box'
            elif dtype == 'bool':
                wtype = 'check_box'
            else:
                wtype = 'text_box'
        if wtype == 'check_box':
            dtype = 'bool'
        if wtype == 'group':
            hidden = True
        pdict = {'value': value,
                 'dtype': dtype,
                 'wtype': wtype,
                 'name': name,
                 'options': options,
                 'option_index': option_index,
                 'description': description,
                 'hidden': hidden}
        OrderedDict.__setitem__(self, key, pdict)

    def get_value(self, key):
        """
        Get the current value of the parameter.

        Parameters
        ----------
        key : str
            Key name for the parameter.

        Returns
        -------
        str, int, float, bool, or list
            The current value of the parameter.
        """
        return self[key]['value']

    def set_value(self, key, value=None,
                  options=None, option_index=None, hidden=None):
        """
        Set a new value for a parameter.

        If the parameter does not yet exist, it will be created
        with default values for any items not provided.

        Parameters
        ----------
        key : str
            Key name for the parameter.
        value : str, int, float, bool, or list; optional
            New value for the parameter.
        options : `list`, optional
            New enumerated value options for the parameter.
        option_index : int, optional
            New selected index for the value options.
        """
        # todo -- think about fixing type before setting
        if key not in self:
            self.set_param(key, value,
                           options=options,
                           option_index=option_index)
        if options is not None:
            self[key]['options'] = options
            if value is None and option_index is None:
                option_index = self[key]['option_index']
        if option_index is not None:
            value = self[key]['options'][option_index]
            self[key]['option_index'] = option_index
        elif value is not None and self[key]['options'] is not None:
            try:
                option_index = self[key]['options'].index(value)
                self[key]['option_index'] = option_index
            except ValueError:
                pass
        elif self[key]['dtype'] == 'bool':
            if str(value).lower().strip() in FALSY:
                value = False
            else:
                value = True
        self[key]['value'] = value
        if hidden is not None:
            self[key]['hidden'] = hidden


class Parameters(object):
    """
    Container class for all parameters needed for a reduction.

    Attributes
    ----------
    current : list of ParameterSet
        A list of parameters corresponding to a reduction recipe:
        one `ParameterSet` object per reduction step.
    stepnames : list of str
        Reduction step names corresponding to `current` parameters.
    default : dict
        Keys are reduction step names; values are default ParameterSet
        objects for the step.

    """
    def __init__(self, default=None):
        """
        Initialize the parameters.

        The ``default`` attribute is populated with `ParameterSet`
        objects based on the values provided in the `default`
        parameter.

        Parameters
        ----------
        default : dict, optional
            Default values for all known parameters for the
            reduction steps.  The keys should be the reduction step
            name, and the values a dictionary of parameter set values,
            corresponding to any desired options for
            `ParameterSet.set_param`.
        """
        # this will hold the list of current parameters,
        # corresponding to the reduction recipe
        self.current = []
        self.stepnames = []

        # this holds initial set values for each step
        self.default = {}
        if default is not None:
            for step in default:
                pdict = ParameterSet()
                for param in default[step]:
                    pdict.set_param(**param)
                self.default[step] = pdict

    def add_current_parameters(self, stepname):
        """
        Add a parameter set to the current list.

        If the step name is found in the ``self.default`` attribute,
        then the associated default ParameterSet is appended to
        the ``self.current`` list.  Otherwise, an empty ParameterSet
        is appended.  The `stepname` is stored in ``self.stapnames``.

        Parameters
        ----------
        stepname : str
            Name of the reduction step
        """
        self.stepnames.append(stepname)
        if stepname in self.default:
            self.current.append(deepcopy(self.default[stepname]))
        else:
            self.current.append(ParameterSet())

    def copy(self):
        """
        Return a copy of the parameters.

        Returns
        -------
        Parameters
        """
        cls = type(self)
        new = cls()
        new.default = deepcopy(self.default)
        new.current = deepcopy(self.current)
        new.stepnames = deepcopy(self.stepnames)
        return new

    def from_config(self, config):
        """
        Set parameter values from a configuration object.

        This function expects reduction step names as the keys of
        the configuration dictionary (the section headers in INI format).
        The step name may be recorded in either ``stepindex: stepname``
        format, or as ``stepname`` alone, if no step names are
        repeated in the reduction recipe.

        Parameters
        ----------
        config : str, dict, or ConfigObj
            Configuration file or object.  May be any type
            accepted by the `configobj.ConfigObj` constructor.

        Raises
        ------
        ValueError
            If the step names in the configuration file/object
            and the currently loaded ``stepnames`` do not match.
        """

        co = configobj.ConfigObj(config)
        try:
            if hasattr(config, 'filename') and config.filename is not None:
                if os.path.isfile(config.filename):
                    log.info("Setting parameters from configuration "
                             "file: {}".format(
                                 os.path.abspath(config.filename)))
                else:
                    log.info("Setting parameters from configuration "
                             "input: {}".format(config.filename))
        except (AttributeError, TypeError):
            pass
        for key in co:
            step = [s.strip() for s in key.split(':')]
            try:
                idx = int(step[0]) - 1
                name = step[1]
                if idx < 0 or idx >= len(self.stepnames) or \
                        self.stepnames[idx] != name:
                    step = [name]
                    raise ValueError("Parameter set and recipe do not match")
            except (ValueError, KeyError, IndexError):
                name = step[0].strip()
                try:
                    idx = self.stepnames.index(name)
                except ValueError:
                    idx = None
            if idx is not None and 0 <= idx < len(self.stepnames):
                log.debug("Modifying parameters for "
                          "step {} ({})".format(idx, name))
                pset = self.current[idx]
                for pkey, pval in co[key].items():
                    if pkey in pset:
                        pval = self.fix_param_type(pval, pset[pkey]['dtype'])
                    pset.set_value(pkey, pval)

    def to_config(self):
        """
        Read parameter values into a configuration object.

        Section names in the output object are written as
        ``stepindex: stepname`` in order to record the order of
        reduction steps, and to keep any repeated step names uniquely
        identified.  Only the current parameter values are recorded.
        Other information, such as data or widget type or default
        values, is lost.

        Returns
        -------
        ConfigObj
            The parameter values in a `configobj.ConfigObj` object.
        """
        steps = OrderedDict()
        for i, pset in enumerate(self.current):
            key_val = OrderedDict()
            for key in pset:
                if pset[key]['hidden']:
                    continue
                key_val[key] = pset.get_value(key)
            steps["{}: {}".format(i + 1, self.stepnames[i])] = key_val
        return configobj.ConfigObj(steps)

    def to_text(self):
        """
        Print the current parameters to a text list.

        Returns
        -------
        list of str
            The parameters in INI-formatted strings.
        """
        co = self.to_config()
        return co.write()

    @staticmethod
    def get_param_type(value):
        """
        Infer a parameter data type from an existing value.

        This function helps format parameters into ParameterSet
        objects when the data type of the parameters is not
        separately recorded.  It attempts to determine if the
        input data is a one of the supported simple types
        (str, float, int, or bool), or if it is a list of any of
        these simple types.  List element type is determined
        from the first element in the array.  Any value for which
        the type cannot be determined is treated as a string.

        Parameters
        ----------
        value : str, float, int, bool, list, or object
            The parameter value to be tested.

        Returns
        -------
        {'str', 'int', 'float', 'bool', 'strlist', \
        'intlist', 'floatlist', 'boollist'}
            The inferred data type of the input value.
        """
        dtype = type(value).__name__
        if dtype not in ['str', 'float', 'int', 'bool', 'list']:
            dtype = 'str'
        if dtype == 'list':
            try:
                eltype = type(value[0]).__name__
            except IndexError:
                eltype = 'str'
            if eltype not in ['str', 'float', 'int', 'bool']:
                eltype = 'str'
            dtype = eltype + dtype
        return dtype

    @staticmethod
    def fix_param_type(value, dtype):
        """
        Cast a value to its expected data type.

        This function helps update parameters in ParameterSet
        objects when the data type of the parameters is not known
        to match its expected data type.  For example, if the value
        is read from a text widget, but the data type is numerical,
        it can cast the data to its expected form.  Any problems
        with converting the value to its expected format cause
        the value to be returned as a string.

        Parameters
        ----------
        value : str, float, int, bool, or list
            The input value to cast
        dtype : {'str', 'int', 'float', 'bool', 'strlist', \
                 'intlist', 'floatlist', 'boollist'}
            The data type, as expected by a `ParameterSet` object.

        Returns
        -------
        str, float, int, bool, or list
            The data type, converted to `dtype` if possible.
        """
        if dtype == 'bool':
            if type(value) is not bool:
                sval = str(value).lower().strip()
                if sval in FALSY:
                    value = False
                else:
                    value = True
        elif dtype == 'int':
            if type(value) is not int:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    # allow it to be a non-number -- initial
                    # dtype may be not be broad enough
                    value = str(value)
        elif dtype == 'float':
            if type(value) is not float:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = str(value)
        elif dtype == 'strlist':
            if type(value) is not list:
                value = [re.sub(r'[\'"\[\]]', '', v).strip()
                         for v in str(value).split(',')]
            else:
                value = [str(v).strip() for v in value]
        elif dtype == 'intlist':
            if type(value) is not list:
                try:
                    value = [int(re.sub(r'[\s\'"\[\]]', '', v))
                             for v in str(value).split(',')]
                except (TypeError, ValueError):
                    # warn for this one -- it will likely not be
                    # interpreted correctly
                    log.warning('Found data type {}; '
                                'expected {}'.format(type(value), dtype))
                    value = str(value)
            else:
                try:
                    value = [int(v) for v in value]
                except (TypeError, ValueError):
                    # allow this case -- where some elements
                    # may be ints, some not -- since initial
                    # dtype guess may not be broad enough
                    pass
        elif dtype == 'floatlist':
            if type(value) is not list:
                try:
                    value = [float(re.sub(r'[\s\'"\[\]]', '', v))
                             for v in str(value).split(',')]
                except (TypeError, ValueError):
                    log.warning('Found data type {}; '
                                'expected {}'.format(type(value), dtype))
                    value = str(value)
            else:
                try:
                    value = [float(v) for v in value]
                except (TypeError, ValueError):
                    # allow this case -- where some elements
                    # may be floats, some not -- since initial
                    # dtype guess may not be broad enough
                    pass

        elif dtype == 'boollist':
            if type(value) is not list:
                value = [re.sub(r'[\s\'"\[\]]', '', v)
                         for v in str(value).split(',')]
            for i, v in enumerate(value):
                if type(v) is bool:
                    continue
                sval = str(v).lower().strip()
                if sval in FALSY:
                    v = False
                else:
                    v = True
                value[i] = v
        else:
            value = str(value)

        return value
