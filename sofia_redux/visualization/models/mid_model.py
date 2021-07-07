# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Dict, Optional, TypeVar, Union, Any
import astropy.io.fits as pf
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.models import low_model

__all__ = ['MidModel', 'Book', 'Order']

LowModels = TypeVar('LowModels', low_model.Spectrum,
                    low_model.Image)
LowerModels = Union[LowModels, np.ndarray]


class MidModel(object):
    """
    Describe a mid-level data object.

    A mid level data object represents a single
    coherent observation structure. These can be
    either an image (consisting of the flux values,
    uncertainties, instrumental response, etc) or
    a spectrum (consisting of the flux values,
    uncertainties, atmospheric transmission, etc).

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        FITS HDU list to load.
    filename : str
        Filename associated with the HDU list.
    number : int
        Index number for the data object.
    """
    def __init__(self, hdul: pf.HDUList, filename: str,
                 number: int) -> None:
        self.number = number
        self.name = ''
        self.data = dict()
        self.enabled = True

    def populated(self) -> bool:
        """
        Test for loaded data.

        Returns
        -------
        bool
            True if data has been loaded; False otherwise.
        """
        return len(self.data) > 0

    def load_split(self, hdul: pf.HDUList, filename: str) -> None:
        """
        Load data that is split across multiple extensions.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            List of HDU objects.
        filename : str
            Name of the FITS file.
        """
        raise NotImplementedError

    def load_combined(self, hdu: pf.ImageHDU, filename: str) -> None:
        """
        Load data that is combined in a single extension.

        Parameters
        ----------
        hdu : astropy.io.fits.HDU
            An HDU object.
        filename : str
            Name of the FITS file.
        """
        raise NotImplementedError

    def retrieve(self, field: str, level: str) -> LowerModels:
        """
        Return raw data for a specified field.

        Parameters
        ----------
        field : str
            Name of field to pull from `self.data`.
        level : ['low', 'raw']
            Determines the level of the data to return.
            'Low' returns the Spectrum object and 'raw'
            will return the raw numeric data.

        Returns
        -------
        data :
            Various based on input parameters. Can be any
            subclass of `low_models.LowModel`, a numpy
            array, or None
        """
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """Describe the loaded data structure."""
        raise NotImplementedError

    def set_visibility(self, enabled: bool) -> None:
        """
        Enable or disable the model visibility.

        Parameters
        ----------
        enabled : bool
            If True, the model is visible/enabled. If False,
            the model is hidden/disabled.
        """
        log.info(f'Disabling {self.name}')
        self.enabled = enabled

    def valid_field(self, field: str) -> bool:
        """
        Test field name validity.

        Parameters
        ----------
        field : str
            Field name to test.

        Returns
        -------
        bool
            True if field has been loaded.
        """
        return field.lower() in self.data.keys()


class Book(MidModel):
    """
    Multi-image data object.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        FITS HDU list to load.
    filename : str
        Filename associated with the FITS file.
    number : int
        Index number for the data object.
    """
    def __init__(self, hdul: pf.HDUList, filename: str,
                 number: int) -> None:
        super().__init__(hdul, filename, number)

        self.name = f'Book_{self.number + 1}'
        log.debug(f'Initializing {self.name}')

        if len(hdul) > 1:
            self.load_split(hdul, filename)
        else:
            self.load_combined(hdul[0], filename)

    def load_split(self, hdul: pf.HDUList, filename: str) -> None:
        """
        Load a book that is split across multiple extensions.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            List of HDU objects.
        filename : str
            Name of the FITS file.
        """
        log.debug(f'Loading split book of length {len(hdul)}')
        for extension in hdul:
            if extension.data.ndim == 2:
                name = extension.name.lower()
                # Shouldn't be here in final version, as it
                # no image extension passes this check. Just
                # a temporary hold as the Image class is not
                # supported yet.
                if extension.data.shape[-1] == 1:  # pragma: no cover
                    self.data[name] = low_model.Image(extension, filename)

    def load_combined(self, hdu: pf.ImageHDU, filename: str) -> None:
        """
        Load a Book that is combined in a single extension.

        Parameters
        ----------
        hdu : astropy.io.fits.HDU
            An HDU object.
        filename : str
            Name of the FITS file.
        """
        raise NotImplementedError('Combined book not implemented')

    def retrieve(self, field: str, level: str) -> Optional[LowerModels]:
        """
        Return raw data for a specified field.

        Parameters
        ----------
        field : str
            Name of field to pull from `self.data`.
        level : ['low', 'raw']
            Determines the level of the data to return.
            'Low' returns the Spectrum object and 'raw'
            will return the raw numeric data. If `field`
            is not a valid options, returns None.

        Returns
        -------
        data : np.array
            Various based on input parameters. Can be any
            subclass of `low_models.LowModel`, a numpy
            array, or None
        """
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """Describe the loaded data structure."""
        raise NotImplementedError


class Order(MidModel):
    """
    Multi-spectrum data object.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        FITS HDU list to load.
    filename : str
        Filename associated with the FITS file.
    number : int
        Index number for the data object.
    """
    def __init__(self, hdul: pf.HDUList, filename: str,
                 number: int) -> None:
        super().__init__(hdul, filename, number)

        self.name = f'Order_{self.number + 1}'

        if len(hdul) > 1:
            self.load_split(hdul, filename)
        else:
            self.load_combined(hdul[0], filename)

    def load_combined(self, hdu: pf.ImageHDU, filename: str) -> None:
        """
        Load an Order that is combined in a single extension.

        The structure of the array is assumed as each row
        describes a separate property of the observation.
        The first row contains wavelength information, the
        second contains the measured flux, the third contains
        the  error on the the measured flux. The fourth row,
        if it exists, contains the transmission spectrum
        of the sky. The fifth row, if it exists, contains the
        instrument response.

        Parameters
        ----------
        hdu : astropy.io.fits.HDU
            An HDU object.
        filename : str
            Name of the FITS file.
        """
        log.info(f'Load combined order from {filename}, '
                 f'{hdu.name}')
        fields = ['wavepos', 'spectral_flux',
                  'spectral_error', 'transmission', 'response']
        kinds = ['wavelength', 'flux', 'flux', 'scale', 'scale']
        units = [hdu.header.get('XUNITS'), hdu.header.get('YUNITS'),
                 hdu.header.get('YUNITS'), None, None]
        for (i, field), kind, unit in zip(enumerate(fields), kinds, units):
            try:
                if hdu.data.ndim == 2:
                    data_ = hdu.data[i, :]
                else:
                    data_ = hdu.data[self.number, i, :]
            except IndexError:
                break
            else:
                self.data[field] = low_model.Spectrum(
                    hdu, filename, data=data_, unit=unit,
                    kind=kind, name=field)

    def load_split(self, hdul: pf.HDUList, filename: str) -> None:
        """
        Load an Order that is split across multiple extensions.

        All extensions in `hdul` whose data array only has one
        dimension are loaded into a ``Spectrum`` object.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            List of HDU objects.
        filename : str
            Name of the FITS file.
        """
        log.info(f'Load split order from {filename}')
        for extension in hdul:
            name = extension.name.lower()
            if extension.data.ndim == 1:
                self.data[name] = low_model.Spectrum(extension, filename)
            elif extension.data.ndim == 2:
                try:
                    data_ = extension.data[self.number, :]
                    self.data[name] = low_model.Spectrum(
                        extension, filename, data=data_)
                except (IndexError, RuntimeError):
                    pass

    def retrieve(self, field: str, level: str = 'raw'
                 ) -> Optional[LowerModels]:
        """
        Return raw data for a specified field.

        Parameters
        ----------
        field : str
            Name of field to pull from `self.data`.
        level : ['low', 'raw']
            Determines the level of the data to return.
            'Low' returns the Spectrum object and 'raw'
            will return the raw numeric data.

        Returns
        -------
        data :
            Various based on input parameters. Can be any
            subclass of `low_models.LowModel`, a numpy
            array, or None.
        """
        if field not in self.data:
            return None
        if level == 'low':
            return self.data[field]
        else:
            return self.data[field].retrieve()

    def describe(self) -> Dict[str, Any]:
        """Describe the loaded data structure."""
        details = dict()
        details['name'] = self.name
        details['fields'] = dict()
        for field, spectrum in self.data.items():
            details['fields'][field] = spectrum.enabled
        return details
