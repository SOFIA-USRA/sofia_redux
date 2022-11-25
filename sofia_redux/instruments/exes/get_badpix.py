# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
import numpy as np

from sofia_redux.toolkit.utilities.fits import getdata, goodfile
from sofia_redux.instruments.exes import data as data_module
from sofia_redux.instruments.exes import utils

__all__ = ['get_badpix']


def get_badpix(header, clip_reference=False, apply_detsec=False):
    """
    Get a bad pixel mask from a reference file.

    The file name for the input mask is specified in header['BPM'].

    Input bad pixel masks are expected to be single-extension FITS files,
    containing a 1024 x 1032 image in which a value of 1 indicates a good
    pixel, 0 indicates a bad pixel, and 2 indicates the 8 columns of
    reference pixels.  These reference pixels will be clipped from the output
    to make a 1024 x 1024 mask, if `clip_reference` is set.

    The output mask has Boolean type with True=good pixel and False=bad.
    The output size may be clipped to a detector section indicated in
    the input header, if `apply_detsec` is True.

    Parameters
    ----------
    header : fits.Header
        FITS header, optionally containing BPM and DETSEC keywords.
    clip_reference : bool, optional
        If True, 8 reference columns are clipped from the output mask.
    apply_detsec : bool, optional
        If set, the output mask is clipped to match any detector section
        provided in header['DETSEC'].

    Returns
    -------
    mask : numpy.ndarray of bool
        The bad pixel mask, with True=good, False=bad.
    """
    bpm = header.get('BPM', 'UNKNOWN')
    if bpm == 'UNKNOWN':
        return

    mask = None
    if goodfile(bpm):
        mask = getdata(bpm).astype('int')
    else:
        datapath = os.path.dirname(data_module.__file__)
        bpm = os.path.join(datapath, 'bpm', bpm)
        if goodfile(bpm):
            mask = getdata(bpm).astype('int')
    if mask is None:
        return

    log.info(f'Using bad pixel mask {bpm}')
    header['BPM'] = bpm

    if clip_reference:
        # trim out reference pixels
        refidx = mask == 2
        if refidx.any():
            xrange = np.where(~refidx)[1]
            mask = mask[:, xrange.min():xrange.max() + 1]
        mask = np.asarray(mask, bool)

    if apply_detsec:
        xstart, xstop, ystart, ystop = utils.get_detsec(header)
        mask = mask[ystart:ystop, xstart:xstop]

    return mask
