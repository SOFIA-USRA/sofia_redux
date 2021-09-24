# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.forcast.hdmerge import hdmerge
from sofia_redux.spectroscopy.readflat import readflat
from sofia_redux.toolkit.image.adjust import rotate90
from sofia_redux.toolkit.utilities import date2seconds

__all__ = ['mkspecimg']


def mkspecimg(infiles, pair_subtract=True, flatfile=None, filenum=None):
    """
    Rotate and pair-subtract spectral images.

    The process is:

    - Divide by the flat if provided.
    - If subtraction is desired, sort infiles by date-obs, then subtract
      all pairs in order.
    - Rotate the image 90 degrees counter-clockwise to align the spectral
      direction along the x-axis.
    - Propagate variance accordingly.

    Parameters
    ----------
    infiles : `list` of fits.HDUList
        Input data.  Should have FLUX, ERROR, and BADMASK extensions.
    pair_subtract : bool, optional
        If True, data will be subtracted in pairs, in order by time
        observed. If an odd number of files is specified, the last one
        will be dropped from the reduction.
    flatfile : str, optional
        Path to FITS file containing flat data to divide into the image.
        Should be in Spextool format, readable by
        `sofia_redux.spectroscopy.readflat.readflat`.
    filenum : list of int or str, optional
        List of file numbers corresponding to the input.  Will be updated
        to match the order and pairs of file numbers corresponding to
        the output, and returned as a secondary output.

    Returns
    -------
    outfiles : list of fits.HDUList
        Pair-subtracted spectral images.
    filenum : list of int or str, optional
        If an input filenum list is specified, the output is
        tuple(outfiles, filenum), where filenum is a list of file
        numbers matching the output order, if no pair subtraction is
        done. If pair subtraction was done, filenum is a list of
        lists of file numbers, where each element is the pair of
        input file numbers.
    """
    # read flat data
    flat = None
    if flatfile is not None:
        flatdata = readflat(flatfile)
        if flatdata is None:
            raise ValueError(f'Could not read flat file {flatfile}')
        flat = flatdata['image']
        log.info(f'Dividing by flat data in {flatfile}')

    # get times for sorting
    dateobs = []
    for hdul in infiles:
        dt = hdul[0].header.get('DATE-OBS', default='3000-01-01T00:00:00')
        dateobs.append(date2seconds(str(dt)))
    idx = np.argsort(dateobs)

    if pair_subtract and len(infiles) == 1:
        log.warning('Only one file; turning off pair-subtraction.')
        pair_subtract = False
    if pair_subtract:
        iter = range(0, len(infiles), 2)
    else:
        iter = range(len(infiles))

    outfiles = []
    outfilenum = []
    for i in iter:
        hdul = infiles[idx[i]]
        header = hdul[0].header
        flux = hdul['FLUX'].data
        var = hdul['ERROR'].data ** 2
        mask = hdul['BADMASK'].data
        a_nod = str(header.get('NODBEAM', 'A')).strip().upper()

        # mask saturated pixels then delete mask
        flux[mask != 0] = np.nan
        del hdul['BADMASK']

        # pair subtract if desired
        if pair_subtract:
            # drop orphaned files
            if i + 1 >= len(idx):
                log.warning(f"Mismatched pairs, dropping "
                            f"{header.get('FILENAME')}")
                continue

            # get the B data
            b_hdul = infiles[idx[i + 1]]
            b_header = b_hdul[0].header
            b_flux = b_hdul['FLUX'].data
            b_var = b_hdul['ERROR'].data ** 2
            b_mask = b_hdul['BADMASK'].data
            b_flux[b_mask != 0] = np.nan

            # subtract, propagate errors, merge headers
            flux -= b_flux
            var += b_var
            header = hdmerge([header, b_header])

            # swap sign if "A" nod is really a B, to make dithers
            # line up correctly
            if a_nod != 'A':
                flux *= -1

            # track file numbers
            if filenum is not None:
                outfilenum.append([filenum[idx[i]],
                                   filenum[idx[i + 1]]])
        elif filenum is not None:
            outfilenum.append(filenum[idx[i]])

        # rotate data to align spectral axis with x
        flux = rotate90(flux, 1)
        var = rotate90(var, 1)

        # divide flux by flat
        # don't propagate flat errors - they're systematic
        if flat is not None:
            flux = flux / flat
            var = var / flat ** 2

        hdul['FLUX'].data = flux
        hdul['ERROR'].data = np.sqrt(var)
        hdul[0].header = header

        outfiles.append(hdul)

    if filenum is None:
        return outfiles
    else:
        return outfiles, outfilenum
