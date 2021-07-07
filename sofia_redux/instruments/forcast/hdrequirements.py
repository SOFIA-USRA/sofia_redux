# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re

from astropy import log
import numpy
import pandas

import sofia_redux.instruments.forcast as drip
import sofia_redux.instruments.forcast.configuration as dripconfig

__all__ = ['parse_condition', 'hdrequirements']


def parse_condition(condition):
    r"""
    Parse a condition value defined in the keyword table

    Conditions in the keyword table are usually defined as
    single condition=(KEYWORD<comparison operator>value) statements.
    However, there is a facility to provide AND or OR type logic across
    multiple conditions.  The order in which conditions are supplied
    is important, as OR conditions will contain a set of AND conditions.

    If all of the AND conditions in a single OR condition are satisfied
    then there is no need to check any of the other OR conditions and
    we can report that the full condition (the condition parameter)
    has passed.  For example if condition string passed in was of the
    form::

        <C1>&<C2>&<C3> | <C4><C5>

    If all of the conditions in the first OR condition set (C1, C2, and C3)
    were satisfied, then we do not need to check C4 or C5 since the
    condition has already been satisfied.  If one or more of the conditions
    was not met in the first OR set, we would need to check that both
    C4 and C5 in the second OR set are satisfied in order to report
    whether the overall condition has been met.

    The return value of parse_condition examines the input <condition>
    string and returns an outer list of OR conditions containing an
    inner list of AND conditions.  The above example would be converted
    to the form::

        [[c1, c2, c3], [c4, c5]]

    where the lowercase c now represents one of the AND conditions as
    a tuple of the form::

        (keyword, comparison operator, value)

    Parameters
    ----------
    condition : str
        Of the form KEYWORD<comparison operator><value>

    Returns
    -------
    list of list of tuple of str
        The outer list contains a list of 'OR' (\|, \|\|) conditions,
        which is a sublist or 'AND' (&, &&) conditions.  i.e.
        'OR' condition = [AND condition 1, AND condition 2,...]
    """
    operator_map = {'<': '<', '<<': '<',  # Less than
                    '<=': '<=', '=<': '<=',  # Less than or equal to
                    '>=': '>=', '=>': '=>',  # Greater than or equal to
                    '>': '>', '>>': '>',  # Greater than
                    '=': '==', '==': '==',  # Is equal to
                    '!=': '!=', '<>': '!=', '=!': '!='}  # Is not equal to
    regex = re.compile(r'([A-Z0-9]+)([!=<>]+)(.+)')
    or_conditions = []
    or_list = [x.strip() for x in condition.split('|') if x.strip()]
    for oreq in or_list:
        and_conditions = []
        and_list = [x.strip() for x in oreq.split('&') if x.strip()]
        for areq in and_list:
            match = regex.match(areq)
            if not match:
                log.error("Bad condition definition detected: %s" % condition)
                return
            keyword, operator, requirement = match.groups()
            if operator not in operator_map:
                log.error("Bad comparison operator detected: %s" % operator)
                return
            and_conditions.append(
                (keyword, operator_map[operator], requirement))
        or_conditions.append(and_conditions)
    return or_conditions


def hdrequirements(kwfile=None):
    """
    Returns a dataframe containing the header requirements

    If fkeydef is not defined then CALDATA will be checked in the
    configuration files (data/config_files/) providing a path
    to the input-key-definition.txt file that will be used.

    This following describes IDL data type codes mapped to Python data types:

    IDL PYTHON
    0   None (maps to string in headers)
    1   bool
    2   int
    3   int
    4   float
    5   float
    6   complex
    7   string

    Parameters
    ----------
    kwfile : str
        File path to the keyword definition file.  The default is
        input-key-definition.txt

    Returns
    -------
    pandas.DataFrame
    """
    default = pandas.DataFrame(columns=['name', 'condition', 'type',
                                        'enum', 'min', 'max'])
    if kwfile is None:
        if dripconfig.configuration is None:
            dripconfig.load()
        workpath = dripconfig.configuration.get('caldata')
        if workpath is None:
            workpath = os.path.join(os.path.dirname(drip.__file__), 'data')

        kwfile = os.path.join(workpath, 'input-key-definition.txt')

    if not os.path.isfile(kwfile):
        log.warning('file does not exist: %s' % kwfile)
        return default
    log.info('using keyword file: %s' % kwfile)

    df = pandas.read_csv(kwfile, delim_whitespace=True,
                         comment=';', index_col=0)
    if len(df) == 0:
        log.warning('keyword file does not have keyword elements')
        return default

    type_map = {0: str, 1: bool, 2: int, 3: int,
                4: float, 5: float, 6: complex, 7: str}
    df['type'] = df['type'].apply(lambda x: type_map[x])
    df['enum'] = df['enum'].apply(lambda x: x.split('|') if x != '.' else [])
    df.loc[df['format'] == '.', 'format'] = ''
    df.insert(0, 'required', True)
    df.loc[df['condition'] == 0, 'required'] = False
    zeroes = df['condition'] == '0'
    df.loc[zeroes, 'required'] = False
    df.loc[zeroes | (df['condition'] == '1'), 'condition'] = ''
    conditions = [x for x in df['condition'].unique() if x]
    condition_map = dict(map(lambda x: (x, parse_condition(x)), conditions))
    condition_map[''] = []
    df['condition'] = df['condition'].apply(lambda x: condition_map[x])
    df.loc[df['min'] == -9999, 'min'] = numpy.nan
    df.loc[df['max'] == -9999, 'max'] = numpy.nan
    return df
