# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['clipimg']


def clipimg(hdul, datasec):
    """
    Clip image to useful portion of the detector.

    Parameters
    ----------
    hdul : fits.HDUList
        Input data.  Should have FLUX, ERROR, and BADMASK extensions.
    datasec : list or tuple
        Data section to clip to, as [xmin, xmax, ymin, ymax].

    Returns
    -------
    fits.HDUList
        Clipped data, with all extensions updated.

    Raises
    ------
    ValueError
        If the data section is invalid or the input data has
        missing or invalid WCS keywords.
    """
    clipped = hdul.copy()
    for i in range(len(hdul)):
        data = hdul[i].data.copy()
        header = hdul[i].header.copy()

        try:
            clipped[i].data = data[datasec[2]:datasec[3],
                                   datasec[0]:datasec[1]]
        except (IndexError, ValueError, TypeError):
            raise ValueError(f'Invalid data section: {datasec}')

        if clipped[i].data.size == 0:
            raise ValueError(f'Invalid data section: {datasec}')

        # correct WCS to new section
        try:
            header['CRPIX1'] -= datasec[0]
            header['CRPIX2'] -= datasec[2]
        except (ValueError, KeyError, TypeError):
            raise ValueError('Invalid CRPIX header keywords')
        clipped[i].header = header

    return clipped
