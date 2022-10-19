# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The starting point for EOS models.

Data read in from FITS or non-fits files are stored in a collection
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
import re
import astropy.io.fits as pf
from typing import Optional
import numpy as np
import pandas as pd

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
                if 'fits' not in filename:
                    hdul = general(filename)
                else:
                    hdul = pf.open(filename, memmap=False)
            else:
                raise RuntimeError('Need to provide an hdul or filename.')

        header = hdul[0].header

        instrument = str(header.get('INSTRUME')).lower()

        if instrument in ['forcast', 'flitecam']:
            model = high_model.Grism(hdul)
            if model.num_orders == 0:
                raise NotImplementedError('Image display is not supported.')
        elif instrument == 'general':
            model = high_model.Grism(hdul)
        elif instrument == 'exes':
            model = high_model.MultiOrder(hdul)
        elif instrument == 'none' or instrument == '':
            # Assign the instrument to 'General' when there is no instrument
            # information has been provided.
            hdul[0].header['instrume'] = 'General'
            if not header.get('XUNIT') or not header.get('XUNITS'):
                hdul[0].header['XUNITS'] = 'um'
            if not header.get('YUNIT') or not header.get('YUNITS'):
                hdul[0].header['YUNITS'] = 'Jy'
            model = high_model.Grism(hdul)
        else:
            raise NotImplementedError('Instrument is not supported')

        # if true filename was supplied, store it in the model
        if filename:
            model.filename = filename
            model.id = filename

        log.debug(f'Created model with id: {model.id}')

        hdul.close()
        return model


def general(filename) -> pf.HDUList:
    """
    Parse a non-fits file.

    It convert the data  into `hdul` format.

    Parameters
    ----------
    filename : str
        Absolute path the non-FITS file to read and parse.

    Returns
    -------
    hdul_read : 'astropy.io.fits.HDUList'
        An astropy HDUList

    Raises
    ------
    RuntimeError
        If invalid file is passed.
        or if invalid columns in the file.
    """

    header = pf.Header()
    header['XUNITS'] = 'um'
    header['YUNITS'] = 'Jy'
    header['INSTRUME'] = 'General'
    header['FILENAME'] = filename.split('/')[-1]
    with open(filename, 'r') as f:
        skip_rows = 0
        names = []
        for line in f:
            if is_number(line.strip()):
                break
            else:
                names = line.replace('#', '').strip()
                names = re.sub(' +', ' ', names)
                delimiter = re.findall(r'[,|]|\s,', names)
                if delimiter:
                    delimiter = delimiter[0]
                else:
                    delimiter = ' '
                names = names.split(delimiter)
                skip_rows += 1
                break
    f.close()

    try:
        data = pd.read_csv(filename, sep=r'\,|\t+|\s+', skiprows=skip_rows,
                           names=names, engine='python')
    except pd.errors.ParserError:
        raise RuntimeError('Could not parse text file') from None

    try:
        n_columns = data.shape[1]
    except IndexError:  # pragma: no cover
        n_columns = 1

    if n_columns == 1:
        # assuming its flux
        wavelength = np.arange(data.shape[0])
        data.insert(0, "wavepos[pixel]", wavelength)

    cols = data.columns
    if not str(cols[1]).isdigit():
        data_new, col_wave, col_flux, col_error, col_trans, col_response = \
            None, None, None, None, None, None
        try:
            if n_columns >= 2:
                col_flux = cols[cols.str.contains('flux', flags=re.I)]
                col_wave = cols[cols.str.contains('wave', flags=re.I)]
                data_new = data.loc[:, [col_wave[0], col_flux[0]]]

            if n_columns >= 3:
                col_error = cols[cols.str.contains('err', flags=re.I,
                                                   regex=False)]
                data_new = data.loc[:, [col_wave[0], col_flux[0],
                                        col_error[0]]]

            if n_columns >= 4:
                col_trans = cols[cols.str.contains('tran', flags=re.I,
                                                   regex=False)]
                data_new = data.loc[:, [col_wave[0], col_flux[0], col_error[0],
                                        col_trans[0]]]

            if n_columns >= 5:
                col_response = cols[cols.str.contains('response', flags=re.I,
                                                      regex=False)]
                data_new = data.loc[:, [col_wave[0], col_flux[0], col_error[0],
                                        col_trans[0], col_response[0]]]

        except (IndexError, ValueError, TypeError):
            raise RuntimeError('Unexpected columns in text file') from None

        if data_new is not None:
            data = data_new
            cols = data.columns

    if '[' in str(cols[0]):
        header['XUNITS'] = cols[0][cols[0].find('[') + 1:cols[
            0].find(']')]
    elif '(' in str(cols[0]):
        header['XUNITS'] = cols[0][cols[0].find('(') + 1:cols[
            0].find(')')]

    if '[' in str(cols[1]):
        header['YUNITS'] = cols[1][cols[1].find('[') + 1:cols[
            1].find(']')]
    elif '(' in str(cols[1]):
        header['YUNITS'] = cols[1][cols[1].find('(') + 1:cols[
            1].find(')')]

    hdu_read = pf.PrimaryHDU(data.T, header)
    hdul_read = pf.HDUList(hdu_read)
    return hdul_read


def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
