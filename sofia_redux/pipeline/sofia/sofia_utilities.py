# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log

__all__ = ['parse_apertures', 'parse_bg']


def parse_apertures(input_position, nfiles, error_message=None,
                    allow_empty=False):
    """
    Parse aperture parameters from input string.

    Parameters
    ----------
    input_position : str
        Input parameter string.
    nfiles : int
        Number of input files expected.
    error_message : list of str, optional
        Error message to display if parsing fails.
    allow_empty : bool
        If set, an empty list is allowed for some files.

    Returns
    -------
    list
        List of length `nfiles`, containing lists of floating point
        aperture values.
    """
    if error_message is None:
        error_message = ['Could not read input_position '
                         f"parameter: '{input_position}'",
                         'Aperture positions should be comma-separated '
                         'values, in arcsec up the slit. ',
                         'To specify different values for different '
                         'input files, provide a semi-colon separated '
                         'list matching the number of input files.']

    apertures = []
    filepos = list(str(input_position).split(';'))
    if len(filepos) == 1:
        filepos = filepos * nfiles
    elif len(filepos) != nfiles:
        for msg in error_message:
            log.error(msg)
        raise ValueError('Invalid position parameter.')
    for fp in filepos:
        if allow_empty and fp.strip() == '':
            apertures.append([])
        else:
            pos = list(fp.split(','))
            try:
                pos = [float(ap) for ap in pos]
            except (ValueError, TypeError):
                for msg in error_message:
                    log.error(msg)
                raise ValueError('Invalid position parameter.') from None
            apertures.append(pos)
    return apertures


def parse_bg(bg_string, nfiles):
    """
    Parse background parameters from input string.

    Parameters
    ----------
    bg_string : str
        Input parameter string.
    nfiles : int
        Number of input files expected.

    Returns
    -------
    list
        List of length `nfiles`, containing lists of floating point
        background start, stop values.
    """
    bad_msg = ['Could not read background region '
               f"parameter: '{bg_string}'",
               'Background regions should be comma-separated '
               'values, in arcsec up the slit, as start-stop. ',
               'To specify different values for different '
               'input files, provide a semi-colon separated '
               'list matching the number of input files.']
    bgr = []
    filepos = list(str(bg_string).split(';'))
    if len(filepos) == 1:
        filepos = filepos * nfiles
    elif len(filepos) != nfiles:
        for msg in bad_msg:
            log.error(msg)
        raise ValueError('Invalid background region parameter.')
    for fp in filepos:
        bg_set = list(fp.split(','))
        bg_list = []
        for bg_reg in bg_set:
            bg_range = bg_reg.split('-')
            if len(bg_range) == 1 and str(bg_range[0]).strip() == '':
                # allow empty set for background regions
                bg_list.append([])
            else:
                try:
                    start, stop = bg_range
                    bg_list.append((float(start), float(stop)))
                except (ValueError, TypeError):
                    for msg in bad_msg:
                        log.error(msg)
                    raise ValueError('Invalid background '
                                     'region parameter.') from None
        bgr.append(bg_list)
    return bgr
