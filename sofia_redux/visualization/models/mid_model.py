# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Dict, Optional, TypeVar, Union, Any, List
from copy import deepcopy
import astropy.io.fits as pf
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.models import low_model

__all__ = ['MidModel', 'Book', 'Order']

LowModels = TypeVar('LowModels', low_model.Spectrum,
                    low_model.Image)
LowerModels = Union[LowModels, np.ndarray]
HL = TypeVar('HL', pf.HDUList, List[pf.FitsHDU])


class MidModel(object):
    """
    Describe a mid-level data object.

    A mid-level data object represents a single coherent observation
    structure. These can be an image (containing flux values,
    uncertainties, instrumental response, etc.) or
    a spectrum (consisting of the flux values, uncertainties,
    atmospheric transmission, etc).

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        FITS HDU list to load.
    filename : str
        Filename associated with the HDU list.
    number : int
        Index number for the data object.
    aperture : int
        Aperture number for the data object.

    Attributes
    ----------
    number : int
        Index number for the data object.
    aperture : int
        Aperture number for the data object.
    name : str
        Name for the data object.
    data : dict
        Dictionary of different types of data.
    enabled: bool
        Indicates if a model is enabled or not.
    book_extensions : list
        List of extension names known to be images.
    order_extensions : list
        List of extension names known to be spectra.
    """
    def __init__(self, hdul: pf.HDUList, filename: str,
                 number: int, aperture: Optional[int] = None) -> None:
        # Order and aperture are 0-index based
        self.number = number
        self.aperture = aperture
        self.name = ''
        self.data = dict()
        self.enabled = True
        self.book_extensions = ['flux', 'error', 'badmask', 'spatial_map',
                                'aperture_mask', 'flat', 'flat_error', 'mask'
                                'wavecal', 'spatcal', 'order_mask',
                                'flat_illumination']
        self.order_extensions = ['wavepos', 'slitpos', 'spatial_profile',
                                 'spectral_flux', 'spectral_error',
                                 'transmission', 'flux_order', 'error_order',
                                 'badmask_order', 'slitpos_order',
                                 'spatial_map_order', 'response']

    def __copy__(self):  # pragma: no cover
        """
        Shallow copy of a mid_model
        """
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memodict):
        """
        Shallow copy of a mid_model
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memodict[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memodict))
        return new

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
        data : low_model.LowModel or array-like
            Retrieved raw data. Can be any subclass of
            `low_model.LowModel`, a numpy array, or None
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
        log.debug(f'Disabling {self.name}')
        self.enabled = enabled

    def set_enabled_state(self, state: Dict[str, bool]) -> None:
        """
        Set enabled state for all contained low models.

        Parameters
        ----------
        state : dict
           Keys are 'enabled' and field names for each contained
           low model. Values are boolean flags used to set the enabled
           state for the mid-model or the low fields, respectively.
        """
        self.enabled = state['enabled']
        for field, low in self.data.items():
            low.set_visibility(state[field])

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
        checks = [key.startswith(field.lower()) for key in self.data.keys()]
        return any(checks)


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
    aperture : int
        Aperture number for the data object.
    """
    def __init__(self, hdul: HL, filename: str,
                 number: int, aperture: Optional[int] = None,
                 **kwargs) -> None:
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
                if any([n in name for n in self.book_extensions]):
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
        level : {'low', 'raw'}
            Determines the level of the data to return.
            'Low' returns the Spectrum object; 'raw' returns the raw
            numeric data array.

        Returns
        -------
        data : low_model.LowModel or array-like
            Retrieved raw data. Can be any subclass of
            `low_model.LowModel`, a numpy array, or None
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
    aperture : int, optional
        Aperture number for the data object.
    num_apertures : int, optional
        Total number of apertures expected.
    """

    def __init__(self, hdul: HL, filename: str,
                 number: int, aperture: Optional[int] = None,
                 num_apertures: Optional[int] = None) -> None:
        super().__init__(hdul, filename, number, aperture)
        if aperture is not None:
            self.name = f'Order_{self.number + 1}.{self.aperture + 1}'
        else:
            self.aperture = 0
            self.name = f'Order_{self.number + 1}'

        if len(hdul) > 1:
            self.load_split(hdul, filename, num_apertures)
        else:
            self.load_combined(hdul[0], filename)

    def load_combined(self, hdu: pf.ImageHDU, filename: str) -> None:
        """
        Load an Order that is combined in a single extension.

        Assumes each row describes a separate property of the data.
        The first row contains wavelength information, the second
        contains the measured flux, the third contains the error on
        the measured flux. The fourth row, if it exists, contains a
        fractional atmospheric transmission spectrum, for reference.
        The fifth row, if it exists, contains the instrument response.

        Parameters
        ----------
        hdu : astropy.io.fits.HDU
            An HDU object.
        filename : str
            Name of the FITS file.
        """
        log.debug(f'Load combined order from {filename}, '
                  f'{hdu.name}')
        fields = ['wavepos', 'spectral_flux',
                  'spectral_error', 'transmission', 'response']
        kinds = ['wavelength', 'flux', 'flux', 'scale', 'response']
        x_units = hdu.header.get('XUNITS')
        y_units = hdu.header.get('YUNITS')
        raw_units = hdu.header.get('RAWUNITS')
        if raw_units is not None:
            response = f'{raw_units} / {y_units}'
        else:
            response = 'Me / s / Jy'
        units = [x_units, y_units, y_units, None, response]
        for (i, field), kind, unit in zip(enumerate(fields), kinds, units):
            try:
                if hdu.data.ndim == 2:
                    data_ = hdu.data[i, :]
                else:
                    if self.number == 0:
                        index = self.aperture
                    else:
                        index = self.number

                    data_ = hdu.data[index, i, :]
            except IndexError:
                break
            else:
                self.data[field] = low_model.Spectrum(
                    hdu, filename, data=data_, unit=unit,
                    kind=kind, name=field)

    def load_split(self, hdul: HL, filename: str,
                   num_apertures: Optional[int] = None) -> None:
        """
        Load an Order that is split across multiple extensions.

        All extensions in `hdul` whose data array only has one
        dimension are loaded into a `Spectrum` object.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            List of HDU objects.
        filename : str
            Name of the FITS file.
        num_apertures : int, optional
            The number of expected apertures. If not provided,
            only 1 aperture is expected.
        """
        log.debug(f'Load split order from {filename}')
        correct_name = not any(['spectral_flux' in hdu.name.lower()
                                for hdu in hdul])
        for extension in hdul:
            name = extension.name.lower()
            if any([n in name for n in self.order_extensions]):
                if correct_name:
                    if 'spectral' not in name:
                        if 'flux_order' in name:
                            name = name.replace('flux', 'spectral_flux')
                        elif 'error_order' in name:
                            name = name.replace('error', 'spectral_error')
                else:
                    image_names = ['flux_order', 'error_order']
                    if any([name.startswith(n) for n in image_names]):
                        continue
                if np.squeeze(extension.data).ndim == 1:
                    self.data[name] = low_model.Spectrum(extension, filename)
                elif extension.data.ndim == 2:
                    try:
                        if ('transmission' in name
                                and num_apertures is not None):
                            if extension.data.shape[0] != num_apertures:
                                # Case of EXES data where each column in the
                                # transmission data is a different
                                # atmospheric species, not a different aperture
                                # Use the first column, which is the
                                # composite.
                                index = 0
                        elif self.aperture is not None:
                            index = self.aperture
                        else:
                            index = self.number
                        data_ = extension.data[index, :]
                        self.data[name] = low_model.Spectrum(
                            extension, filename, data=data_)
                    except (IndexError, RuntimeError) as err:
                        log.debug(f'Loading split order encountered error: '
                                  f'{err}')
                        pass

    def retrieve(self, field: str, level: str = 'raw'
                 ) -> Optional[LowerModels]:
        """
        Return raw data for a specified field.

        Parameters
        ----------
        field : str
            Name of field to pull from `self.data`.
        level : {'low', 'raw'}
            Determines the level of the data to return.
            'Low' returns the Spectrum object and 'raw'
            will return the raw numeric data.

        Returns
        -------
        data : low_model.LowModel or array-like
            Retrieved raw data. Can be any subclass of
            `low_model.LowModel`, a numpy array, or None
        """
        key = [k for k in self.data.keys()
               if field == k.split('_order_')[0]]
        if len(key) == 0:
            log.debug(f'Field {field} not found in Order')
            return None
        elif len(key) > 1:
            log.debug(f'Field {field} does not uniquely identify a '
                      f'field in Order')
            return None
        else:
            key = key[0]
        if level == 'low':
            return self.data[key]
        else:
            return self.data[key].retrieve()

    def describe(self) -> Dict[str, Any]:
        """
        Describe the loaded data structure.

        Returns
        -------
        details : dict
            A nested dictionary containing information for all fields.
        """
        details = dict()
        details['name'] = self.name
        details['fields'] = dict()
        for field, spectrum in self.data.items():
            details['fields'][field] = spectrum.enabled
        return details
