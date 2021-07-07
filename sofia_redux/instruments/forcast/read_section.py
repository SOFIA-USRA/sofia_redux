# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log

import sofia_redux.instruments.forcast.configuration as dripconfig

__all__ = ['read_section']


def read_section(xdim, ydim):
    """
    Read the section in the configuration file and check if it's correct

    Read the section to use in the configuration file.  If it is not
    correct it returns the default value [128, 128, 200, 200].

    Note that at some point in the past, the value used to be appended
    to a header.  This is no longer the case.  For reference, the comment
    for the NLINSECTION keyword is "section used to calculate bacground for
    linearity correction".  Also note that this comment is too long and has
    a typo.

    Parameters
    ----------
    xdim : int
        x-dimension size
    ydim : int
        y-dimension size

    Returns
    -------
    4-tuple
        (x0, y0, xdim, ydim)
        x0, y0: Center of section
        xdim, ydim: Dimension of section
    """
    key = 'NLINSECTION'
    default_section = 128, 128, 200, 200
    default_message = '%s default value is %s' % (key, repr(default_section))
    if dripconfig.configuration is None:
        dripconfig.load()
    configuration = dripconfig.configuration

    if key.lower() not in configuration:
        log.warning('The section has not been specified in configuration')
        log.warning(default_message)
        return default_section

    if not isinstance(xdim, int):
        log.error("xdim must be %s" % int)
        log.warning(default_message)
        return default_section
    elif not isinstance(ydim, int):
        log.error("ydim must be %s" % int)
        log.warning(default_message)
        return default_section

    section = None
    try:
        section = [int(val) for val in configuration[key.lower()]]
    except ValueError:
        log.error("The section %s has wrong format" % repr(section))
        log.warning(default_message)
        return default_section

    if not isinstance(section, list) or len(section) != 4:
        log.error("The section %s has wrong format" % repr(section))
        log.warning(default_message)
        return default_section

    # Check if the section makes sense (xsize, ysize > 10)
    if section[2] < 10 or section[3] < 10:
        log.error("The section %s has wrong size values" % repr(section))
        log.warning(default_message)
        return default_section

    # Check if the section we chose is not outside detector
    if ((section[0] - section[2] // 2) < 0
            or (section[0] + section[2] // 2) > xdim):
        log.error("wrong section size along x-dimension")
        return default_section
    if ((section[1] - section[3] // 2) < 0
            or (section[1] + section[3] // 2) > ydim):
        log.error("wrong section size along y-dimension")
        return default_section
    return section
