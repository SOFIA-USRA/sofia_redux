# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The starting point for EOS models.

Data read in from FITS files are stored in a collection
of model object in this directory. The structure of a single
complete FITS file is as follows:

- High Level Model
   - Grism : for files with multiple images and a single spectrum
   - MultiOrder: For files with no images and multiple spectra.
   - Described in `high_model.py`, the high level models hold an
     entire FITS file. The controls and views for EOS hold
     these objects, thus all interactions happen on them. The
     user should not call anything from the lower models.

- Mid Level Models
   - Book: For holding multiple images of the same target.
   - Order: For holding multiple spectra of the same target.
   - Described in `mid_model.py`, the mid level models hold all
     data structures that would need to included together to
     be considered valid.

- Low Level Models
   - Image: For holding a single 2d data set.
   - Spectrum: For holding a single 1d data set.
   - Described in `low_model.py`, the low level models hold the
     most simple data sets. For example, a single `Order` is made
     of multiple `Spectrum` objects, one for the wavelength data,
     one for the flux data, one for the error data, etc. These are
     the most basic levels of models. All operations are done on
     this level, such as unit conversions, but the only by
     interacting through the upper models. The user should not
     interact with these directly.

This module contains the interface controlling the initialization
of a high level model. Once the high level model is created, EOS
operates on that object. The `Model` object here is not to be instantiated;
it is merely used to implement the `add_model` method.
"""

import astropy.io.fits as pf
from typing import Optional

from sofia_redux.visualization import log
from sofia_redux.visualization.models import high_model

__all__ = ['Model']


class Model(object):

    @staticmethod
    def add_model(filename: str = '',
                  hdul: Optional[pf.HDUList] = None) -> high_model.HighModel:
        """
        Parse a FITS file into the appropriate high level model.

        Either `filename` or `hdul` must be specified.

        Parameters
        ----------
        filename : str, optional
            Absolute path the FITS file to read and parse.

        hdul : `astropy.io.fits.HDUList`, optional
            An astropy HDUList to parse.

        Returns
        -------
        model : model.high_model.HighModel
            The high level model populated with the
            contents of `filename`.

        Raises
        ------
        NotImplementedError
            If the current instrument/data type is not supported
            by EOS.
        RuntimeError
            If invalid arguments are passed.
        """
        if filename and hdul is not None:
            raise RuntimeError('Model.add_model can only accept `filename` '
                               'or `hdul`, not both')
        if hdul is None:
            if filename:
                hdul = pf.open(filename, memmap=False)
            else:
                raise RuntimeError('Need to provide an hdul or filename.')

        header = hdul[0].header
        instrument = header['instrume'].lower()
        if instrument in ['forcast', 'flitecam']:
            model = high_model.Grism(hdul)
            if model.num_orders == 0:
                raise NotImplementedError('Image display is not supported.')
        elif instrument == 'exes':
            model = high_model.MultiOrder(hdul)
        else:
            raise NotImplementedError(f'Instrument {instrument} '
                                      f'is not supported')

        # if true filename was supplied, store it in the model
        if filename:
            model.filename = filename
            model.id = filename

        log.debug(f'Created model with id: {model.id}')
        hdul.close()
        return model
