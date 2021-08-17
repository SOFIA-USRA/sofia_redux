# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
import configobj
import numpy as np

from sofia_redux.instruments import forcast as drip
from sofia_redux.toolkit.utilities import date2seconds, hdinsert

__all__ = ['hdmerge', 'order_headers']


def order_headers(headers):
    """
    Order headers based on contents.

    Return the earliest header and the header list sorted by date.
    B nods are never returned as the earliest header.

    Parameters
    ----------
    headers : array_like of fits.Header

    Returns
    -------
    2-tuple
       fits.Header : earliest header
       list of fits.Header : ordered headers
    """
    nhead = len(headers)
    if nhead == 1:
        return headers[0].copy(), [headers[0]]

    dateobs, nodbeam = [], []
    for header in headers:
        dateobs.append(
            date2seconds(
                str(header.get('DATE-OBS', default='3000-01-01T00:00:00'))))
        nodbeam.append(str(header.get('NODBEAM', 'UNKNOWN')).upper())

    # get the earliest A header as the basehead
    # Otherwise, just use the earliest header
    idx = np.argsort(dateobs)
    if 'A' in nodbeam:
        a_test = np.where(np.array(nodbeam)[idx] == 'A')[0]
        earliest_a = int(a_test[0])
    else:
        earliest_a = 0

    earliest_idx = idx[earliest_a]
    basehead = headers[earliest_idx].copy()

    # sort all headers by date-obs, including the basehead
    # This is used to get 'last' values, whether
    # in A or B nod
    sorted_headers = [headers[i] for i in idx]

    return basehead, sorted_headers


def hdmerge(headers, reference_header=None):
    """
    Merge input headers.

    Keyword handling is defined in the configuration file at
    data/header_merge.cfg.

    Parameters
    ----------
    headers : list of astropy.io.fits.Header
        Headers to check.

    Returns
    -------
    astropy.io.fits.Header
        Combined header.
    """
    data_path = os.path.join(os.path.dirname(drip.__file__), 'data')
    kwfile = os.path.join(data_path, 'header_merge.cfg')
    if os.path.isfile(kwfile):
        try:
            merge_conf = configobj.ConfigObj(kwfile)
        except configobj.ConfigObjError as error:
            msg = 'Error while loading header merge configuration file.'
            log.error(msg)
            raise error
    else:
        msg = f'Config file {kwfile} is missing.'
        log.error(msg)
        raise IOError(msg)

    basehead, all_headers = order_headers(headers)

    if reference_header is not None:
        basehead = reference_header.copy()

    for key, operation in merge_conf.items():
        values = [h[key] for h in all_headers if key in h]
        if len(values) == 0:
            continue

        value = values[0]
        if operation == 'LAST':
            value = values[-1]

        elif operation == 'SUM':
            try:
                value = np.sum(values)
            except TypeError:
                log.warning(f'Key merge SUM operation is invalid for {key}')
                continue

        elif operation == 'MEAN':
            try:
                value = np.mean(values)
            except TypeError:
                log.warning(f'Key merge MEAN operation is invalid for {key}')
                continue

        elif operation == 'AND':
            try:
                value = np.all(values)
            except TypeError:
                log.warning(f'Key merge AND operation is invalid for {key}')
                continue

        elif operation == 'OR':
            try:
                value = np.any(values)
            except TypeError:
                log.warning(f'Key merge OR operation is invalid for {key}')
                continue

        elif operation == 'CONCATENATE':
            split_list = list()
            for v in values:
                split_list.extend(str(v).split(','))
            unique = set(split_list)
            value = ','.join(sorted(unique))

        elif operation == 'DEFAULT':
            test_val = values[0]
            if type(test_val) is str:
                value = 'UNKNOWN'
            elif type(test_val) is int:
                value = -9999
            elif type(test_val) is float:
                value = -9999.0

        elif operation == 'FIRST':
            value = values[0]

        else:
            log.warning(f'Invalid key merge operation {operation}')
            continue

        # comments are not handled -- it is assumed these keys
        # are already in the base header
        hdinsert(basehead, key, value)

    return basehead
