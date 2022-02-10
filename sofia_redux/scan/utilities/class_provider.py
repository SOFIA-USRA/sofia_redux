# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
import inspect
import re

from sofia_redux.scan import custom
from sofia_redux.scan.channels import mode as mode_module
from sofia_redux.scan.coordinate_systems import grid as grid_module
from sofia_redux.scan.coordinate_systems import projection as projection_module
from sofia_redux.scan.simulation \
    import source_models as simulated_source_module

custom_module_path = custom.__name__

__all__ = ['fix_instrument_name', 'get_instrument_module_path',
           'to_class_name', 'to_module_name',
           'get_instrument_path_and_class_name', 'get_class_for',
           'channel_class_for',
           'channel_data_class_for',
           'channel_group_class_for',
           'frames_class_for',
           'frames_instance_for',
           'info_class_for', 'get_integration_class',
           'get_scan_class', 'default_modality_mode', 'get_grid_class',
           'get_projection_class', 'get_simulated_source_class']


def fix_instrument_name(name):
    """
    Return a valid name for the given instrument.

    Parameters
    ----------
    name : str
        The name of the instrument.

    Returns
    -------
    name : str
    """
    if not isinstance(name, str):
        return ''
    name = name.lower().strip()
    if name == 'hawc+':
        return 'hawc_plus'
    return name


def get_instrument_module_path(instrument):
    """
    Return the module path to custom SOFSCAN instrument

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    module_path : str
        The dot-separated module path for subsequent import via
        :func:`importlib.import_module`.
    """
    return f'{custom_module_path}.{fix_instrument_name(instrument)}'


def to_class_name(name):
    """
    Return the associated class type name for a given string.

    Capitalizes the first letter of the instrument, upper-cases any character
    following an underscore, and finally remove all underscores.  For example,
    converts 'hawc_plus` to `HawcPlus`.

    Parameters
    ----------
    name : str
        The name to convert.

    Returns
    -------
    class_name : str
    """
    if name is None:
        return ''
    name = str(name)
    if len(name) == 0:
        return name
    if len(name) == 1:
        return name.upper()
    return ''.join([s[0].upper() + s[1:] for s in name.split('_')])


def to_module_name(class_name):
    """
    Convert a class name to the base module it may be found in.

    Inserts an underscore before any uppercase character or digit
    (excluding the first) and converts the string to lower case.

    Examples
    --------
    >>> print(to_module_name('TestClass1'))
    test_class_1

    Parameters
    ----------
    class_name : str

    Returns
    -------
    module_name : str
    """
    if class_name is None:
        return ''
    class_name = str(class_name)
    if len(class_name) == 0:
        return class_name
    result = class_name[0]
    for letter in class_name[1:]:
        if letter.isupper() or letter.isdigit():
            result += f'_{letter}'
        else:
            result += letter
    return result.lower()


def get_instrument_path_and_class_name(instrument):
    """
    Get the module path and class name prefix for a given instrument name.

    Parameters
    ----------
    instrument : str
       The name of the instrument.

    Returns
    -------
    instrument_module_path, class_prefix : str, str
        The instrument module path such as 'sofia_redux.scan.custom.hawc_plus'
        and the class name prefix for the instrument such as 'HawcPlus'.
    """
    instrument = fix_instrument_name(instrument)
    module_path = get_instrument_module_path(instrument)
    instrument_class_name = to_class_name(instrument)
    return module_path, instrument_class_name


def get_class_for(instrument, module_path_name, other_module=None):
    """
    Return an instrument specific class for a given module.

    The SOFSCAN custom classes for any instrument should be placed in the
    package in the following format:

    scan.custom.<instrument>.<...>

    For example, a custom channel data class for the HAWC_PLUS instrument
    should be placed at:

    scan.custom.hawc_plus.channels.channel_data.channel_data

    and contain a class called ChannelData.  Class names must always match the
    `module_path_name` final path, begin with an upper case character, and
    mark new words with an upper-case character.  For example, the class
    AbcDefGhi should be in the abc_def_ghi module.

    Examples
    --------
    >>> print(get_class_for('sofia', 'frames'))
    <class 'sofia_redux.scan.custom.sofia.frames.frames.SofiaFrames'>

    Parameters
    ----------
    instrument : str
        The name of the instrument.
    module_path_name : str
        The dot-separated module path for the required class excluding the
        custom instrument path.  For example, 'channels.channel_data'.
    other_module : str, optional
        Usually, the class will be retrieved from a module matching the last
        path level in `module_path_name`.  For example, ChannelData will
        usually be retrieved from
        <instrument_path>.channels.channel_data.channel_data.
        Set this to an empty string ('') to retrieve from
        <instrument_path>.channels.channel_data or <other> to retrieve from
        <instrument_path>.channels.channel_data.<other>.

    Returns
    -------
    class
    """
    base_path, class_prefix = get_instrument_path_and_class_name(instrument)
    class_type = module_path_name.split('.')[-1]
    class_suffix = to_class_name(class_type)
    class_name = f'{class_prefix}{class_suffix}'
    module_path = f'{base_path}.{module_path_name}'
    if other_module is None:
        full_path = f'{module_path}.{class_type}'
    elif other_module == '':
        full_path = module_path
    else:
        full_path = f'{module_path}.{other_module}'

    # Allow errors
    module = importlib.import_module(full_path)
    retrieved_class = getattr(module, class_name)
    return retrieved_class


def channel_class_for(instrument):
    """
    Returns a Channels instance for a given instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    Channels
    """
    return get_class_for(instrument, 'channels')


def channel_data_class_for(instrument):
    """
    Returns a ChannelData instance for a given instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    channel_data_class : class (ChannelData)
    """
    return get_class_for(instrument, 'channels.channel_data')


def channel_group_class_for(instrument):
    """
    Returns the appropriate ChannelGroup class for a given instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    class (ChannelGroup)
    """
    return get_class_for(instrument, 'channels.channel_group')


def frames_class_for(instrument):
    """
    Returns the appropriate ChannelGroup class for a given instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    class (ChannelGroup)
    """
    return get_class_for(instrument, 'frames')


def frames_instance_for(instrument):
    """
    Return a Frames instance for a given instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    Frames
    """
    return frames_class_for(instrument)()


def info_class_for(instrument):
    """
    Return an Info instance given an instrument name.

    Parameters
    ----------
    instrument : str
        The name of the instrument

    Returns
    -------
    Info
    """
    return get_class_for(instrument, 'info')


def get_integration_class(instrument):
    """
    Return an Integration instance given an instrument name.

    Parameters
    ----------
    instrument : str
        The name of the instrument.

    Returns
    -------
    Integration : class
    """
    return get_class_for(instrument, 'integration')


def get_scan_class(instrument):
    """
    Return the appropriate scan class for an instrument name.

    Parameters
    ----------
    instrument : str
        The instrument name.

    Returns
    -------
    Scan : class
    """
    return get_class_for(instrument, 'scan')


def default_modality_mode(modality):
    """
    Return a default mode class based on modality class.

    For example, A CoupledModality should return CoupledMode.  If
    no analogous mode is found, a default base Mode will be returned.

    Parameters
    ----------
    modality : Modality

    Returns
    -------
    class
        The correct default mode class for the given modality.
    """
    if inspect.isclass(modality):
        modality_class = modality
    else:
        modality_class = modality.__class__

    mode_module_path = mode_module.__name__
    modality_name = modality_class.__name__.split('.')[-1]
    mode_name = modality_name.replace('Modality', 'Mode')
    mode_sub_path = re.sub(r'(?<!^)(?=[A-Z])', '_', mode_name).lower()
    mode_path = f'{mode_module_path}.{mode_sub_path}'
    try:
        module = importlib.import_module(mode_path)
        mode_class = getattr(module, mode_name)
    except ModuleNotFoundError:  # Return a basic mode.
        mode_path = f'{mode_module_path}.mode'
        module = importlib.import_module(mode_path)
        mode_class = getattr(module, 'Mode')

    return mode_class


def get_grid_class(name):
    """
    Returns a Grid class of the given name

    Parameters
    ----------
    name : str
        The name of the grid.

    Returns
    -------
    Grid : class
    """
    if name in [None, '']:
        return None

    grid_path = grid_module.__name__
    module_path = f'{grid_path}.{name}'
    class_name = ''.join(
        [s[0].upper() + s[1:] for s in name.split('_')])
    if re.match(r'\dd', class_name[-2:]):
        class_name = class_name[:-1] + class_name[-1].upper()

    module = importlib.import_module(module_path)  # Allow errors
    grid_class = getattr(module, class_name)
    return grid_class


def get_projection_class(name):
    """
    Returns a Projection class of the given name

    Parameters
    ----------
    name : str
        The name of the grid omitting the "projection" suffix.

    Returns
    -------
    Projection : class
    """
    if name in [None, '']:
        return None

    projection_path = projection_module.__name__
    module_path = f'{projection_path}.{name}_projection'
    class_name = ''.join(
        [s[0].upper() + s[1:] for s in name.split('_')]) + 'Projection'
    module = importlib.import_module(module_path)  # Allow errors
    projection_class = getattr(module, class_name)
    return projection_class


def get_simulated_source_class(name):
    """
    Return a simulated source of the given name.

    Parameters
    ----------
    name : str
        The name of the simulated source model.

    Returns
    -------
    SimulatedSource
    """
    if name in [None, '']:
        return None

    path = simulated_source_module.__name__
    module_path = f'{path}.{name}'
    class_name = to_class_name(name)
    module = importlib.import_module(module_path)
    source_class = getattr(module, class_name)
    return source_class
