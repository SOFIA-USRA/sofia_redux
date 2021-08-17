# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['split_input']


def split_input(infiles):
    """
    Split input into categories.

    Read the OBSTYPE keyword to sort data into sky, object, and
    standard files.  The default type for missing or unexpected OBSTYPE
    values is object.

    Parameters
    ----------
    infiles : list of fits.HDUList
        Input data.

    Returns
    -------
    manifest : dict
        Keys are 'sky', 'object', 'standard'.  Values are lists
        of HDULists in each category.  Lists are empty if no data
        of that type is present.
    """

    manifest = {'sky': [], 'object': [], 'standard': []}

    for hdul in infiles:
        obstype = hdul[0].header.get('OBSTYPE', 'OBJECT')
        obstype = str(obstype).strip().upper()
        if 'STANDARD' in obstype:
            manifest['standard'].append(hdul)
        elif obstype == 'SKY':
            manifest['sky'].append(hdul)
        else:
            manifest['object'].append(hdul)

    return manifest
