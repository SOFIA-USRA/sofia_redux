# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Dict, Union, Sequence, TypeVar
from copy import deepcopy
import astropy.io.fits as pf
import astropy.units as u
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.utils import unit_conversion as uc
from sofia_redux.visualization.utils.eye_error import EyeError

__all__ = ['LowModel', 'Image', 'Spectrum']

Data = TypeVar('Data', np.ndarray, Sequence)


class LowModel(object):
    """
    Describe a low-level data object.

    A low-level data object holds a single data array
    and the information describing it, such as the
    kind of data it holds and the units.

    Parameters
    ----------
    hdu : astropy.io.fits.ImageHDU
        The FITS HDU to load.
    filename : str
        Filename associated with the FITS HDU.
    kwargs : dict, optional
        Optional keywords to set `name`, `data`, and `kind`
        properties.

    Attributes
    ----------
    header : astropy.io.fits.Header
        FITS header.
    filename : str
        Filename associated with the FITS HDU.
    unit : str
        Unit associated with the data.
    unit_key : str
        Associated keyword to parse corresponding units.
        For example, 'XUNITS' for x-axis units.
    name : str
        Name of FITS extension. For example, wavepos, slitpos,
        spectral_flux, or spectral_error.
    data : np.ndarray
        Data from the FITS HDU.
    kind : str
        Determines the variable kind (e.g. 'wavelength'), to determine
        its units.
    available_units : dict
        Recognized units that can be converted from or to.
    kind_names : Dict
        Dictionary of variable kinds depending on if its Image or a spectrum.
    default_ndims : int
        Default data dimensions.
    id : str
        Full name including the filename and extension name.
    enabled : bool
        Indicates if the data set is enabled or not.
    """

    def __init__(self, hdu: pf.ImageHDU, filename: str, **kwargs) -> None:
        self.header = hdu.header.copy()  # Should this be here or not?
        self.filename = filename
        self.unit = None
        self.unit_key = ''
        self.name = kwargs.get('name', hdu.name.lower())
        self.data = kwargs.get('data', hdu.data)
        self.kind = kwargs.get('kind', '')
        self.available_units = self._define_units()
        self.kind_names = dict()

        self.default_ndims = 0
        self.id = None
        self.enabled = True

    def __copy__(self):  # pragma: no cover
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memodict):
        cls = self.__class__
        new = cls.__new__(cls)
        memodict[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memodict))
        return new

    @staticmethod
    def _define_units() -> Dict[str, Dict[str, Union[str, u.Quantity]]]:
        """Define common units and their astropy equivalent."""
        wavelength_units = {'um': u.um, 'nm': u.nm,
                            u'\u212B': u.angstrom,
                            'cm-1': u.kayser,
                            'pixel': u.pix
                            }
        flux_units = {'Jy': u.Jy,
                      'W / m2': u.W / u.m ** 2,
                      'W / (m2 Hz)': u.W / (u.m ** 2 * u.Hz),
                      'W / (m2 um)': u.W / (u.m ** 2 * u.um),
                      'erg / (s cm2 Angstrom)':
                          u.erg / (u.s * u.cm ** 2 * u.AA),
                      'erg / (s cm2 Hz)': u.erg / (u.s * u.cm ** 2 * u.Hz)
                      }
        scale_units = {'': ''}
        unitless = {'': u.dimensionless_unscaled}
        position_units = {'arcsec': u.arcsec,
                          'degree': u.deg,
                          'arcmin': u.arcmin,
                          'pixel': u.pix
                          }
        time_units = {'sec': u.s,
                      'pixel': u.pix}
        response_units = {'Me / (Jy s)': u.Mct / (u.Jy * u.s)}

        available_units = {'scale': scale_units,
                           'unitless': unitless,
                           'position': position_units,
                           'flux': flux_units,
                           'wavelength': wavelength_units,
                           'time': time_units,
                           'response': response_units}
        return available_units

    def get_unit(self) -> str:
        """Get the current data unit."""
        return self.unit_key

    def __eq__(self, other: 'LowModel') -> bool:
        """Models are equal if they share a filename."""
        return self.filename == other.filename

    def _parse_units(self, **kwargs) -> None:
        """
        Determine the initial units of the data.

        Attempt to pull the unit from the header, based
        on the BUNIT, X/YUNIT, X/YUNITS in decreasing
        priority. If a unit is provided in `kwargs` it will
        override all header values.

        Parameters
        ----------
        kwargs : dict, optional
            If `unit` keyword is present, it will be used as
            the unit directly.
        """
        if self.kind == 'wavelength':
            header_keys = ['BUNIT', 'XUNIT', 'XUNITS']
        else:
            header_keys = ['BUNIT', 'YUNIT', 'YUNITS']
        header_unit = ''
        for key in header_keys:
            try:
                header_unit = self.header[key]
            except KeyError:
                continue
            else:
                break
        unit = kwargs.get('unit', header_unit)
        if unit is None:
            unit = ''

        self.unit = uc.parse_unit(unit)
        self.unit_key = str(self.unit)

        # special handling for initial pixel units:
        # it is recorded as 'pixel' everywhere else
        if self.unit_key == 'pix':
            self.unit_key = 'pixel'

        self._verify_unit_parse()

    def _verify_unit_parse(self) -> None:
        """
        Verify the unit was successfully parsed.

        Raises
        -------
        RuntimeError
            If the resulting unit or unit_key is not
            contained in the available units.
        """
        if not isinstance(self.unit, u.UnitBase):

            message = (f'Failure to parse unit for {self.filename}, '
                       f'{self.name}: {self.kind}, {self.unit} '
                       f'({type(self.unit)}')
            log.error(message)
            raise RuntimeError(message)

        # non-standard unit found: allow it, but disable conversion
        if self.unit not in self.available_units[self.kind].values():
            log.debug(f'Non-standard unit found: {self.unit_key}. '
                      f'Disabling conversion.')
            self.available_units[self.kind] = {self.unit_key: self.unit}

    def _parse_kind(self) -> None:
        """
        Parse the kind of data from the extension name.

        Raises
        -------
        RuntimeError
            If a valid kind is not found.
        """
        name = self.name.split('_order')[0]
        for kind, names in self.kind_names.items():
            if name in names:
                self.kind = kind
        if not self.kind:
            raise RuntimeError(f'Failed to parse kind for {self.filename} '
                               f'{self.name}')

    def convert(self, target_unit: str, wavelength: Data,
                wavelength_unit: str) -> None:
        """
        Convert units.

        Parameters
        ----------
        target_unit : str
            Unit to convert to.
        wavelength : array-like
            Wavelength data array associated with current data.
        wavelength_unit : str
            Wavelength units.

        Raises
        ------
        ValueError
            If conversion cannot be completed.
        """
        raise NotImplementedError

    def retrieve(self) -> np.ndarray:
        """
        Retrieve data array from the model.

        Returns
        -------
        data: np.ndarray
            Data from FITS file in a numpy array.
        """
        return self.data

    def set_visibility(self, enabled: bool) -> None:
        """
        Set visibility for the model.

        Parameters
        ----------
        enabled : bool
            If True, the model is enabled. If False, it is disabled.
        """
        self.enabled = enabled


class Image(LowModel):
    """
    Low-level data model for an image.

    Attributes
    ----------
    default_ndims : int
        Default data dimensions.
    kind : str
        The kind of data contained, eg. transmission, flux, error.
    kind_names : dict
        Extension names corresponding to particular data kinds.
    """

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.default_ndims = 2
        self.kind = ''

        self.kind_names = {'scale': ['transmission', 'aperture_mask'],
                           'response': ['response', 'response_error'],
                           'position': ['aperture_trace'],
                           'unitless': ['badmask', 'spatial_map', 'flat',
                                        'flat_error', 'flat_illumination',
                                        'order_mask', 'mask'],
                           'flux': ['flux', 'error', 'spatcal',
                                    'spectral_flux', 'spectral_error'],
                           'time': ['exposure']}
        self._parse_kind()

    def data_mean(self) -> float:
        """
        Calculate the mean of the contained data.

        Returns
        -------
        mean: float
            Mean of the loaded array.
        """
        try:
            mean = np.nanmean(self.data)
        except (ValueError, AttributeError, TypeError):
            mean = np.nan
        return mean

    def convert(self, target_unit: str, wavelength: Data,
                wavelength_unit: str) -> None:
        """
        Convert units.

        Parameters
        ----------
        target_unit : str
            Unit to convert to.
        wavelength : array-like
            Wavelength data array associated with current data.
        wavelength_unit : str
            Wavelength units.

        Raises
        ------
        ValueError
            If conversion cannot be completed.
        """
        raise NotImplementedError


class Spectrum(LowModel):
    """
    Class for holding simple 1D spectra.

    Attributes
    ----------
    default_ndims : int
        Default data dimensions.
    id : str
        Filename associated with the FITS file.
    data : np.ndarray
        Data from the FITS HDU.
    kind_names : dict
        Extension names associated with particular data kinds.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.default_ndims = 1
        self.id = f'{self.filename}/{self.name}'
        self.data = np.squeeze(self.data)
        if self.data.ndim != 1:
            raise EyeError(f'Spectrum for {self.id} has {self.data.ndim} '
                           f'dimensions and can only have 1.')

        self.kind_names = {'scale': ['transmission'],
                           'response': ['response', 'response_error'],
                           'unitless': ['spatial_profile', 'aperture_mask',
                                        'spatial_map', 'badmask'],
                           'flux': ['spectral_flux', 'spectral_error',
                                    'flux', 'error'],
                           'wavelength': ['wavepos', 'slitpos'],
                           'position': ['slitpos'],
                           'unrecog': ['miscellaneous']}
        self._parse_kind()

        self._parse_units(**kwargs)

    def convert(self, target_unit: str, wavelength: Data,
                wavelength_unit: str) -> None:
        """
        Convert units.

        Parameters
        ----------
        target_unit : str
            Unit to convert to.
        wavelength : array-like
            Wavelength data array associated with current data.
        wavelength_unit : str
            Wavelength units.

        Raises
        ------
        ValueError
            If conversion cannot be completed.
        """
        if self.kind == 'flux':
            self.data = uc.convert_flux(
                in_flux=self.data, start_unit=self.unit_key,
                end_unit=target_unit, wavelength=wavelength,
                wave_unit=wavelength_unit)
        elif self.kind in ['wavelength', 'position']:
            # this will work for any simple conversions handled
            # by astropy
            if target_unit == 'pixel':
                self.data = np.arange(len(self.data)).astype(float)
            else:
                self.data = uc.convert_wave(
                    wavelength=self.data, start_unit=self.unit_key,
                    end_unit=target_unit)
        elif self.kind in ['scale', 'unitless']:
            # no conversions to do for these types
            return
        elif self.kind in ['response']:
            if target_unit == 'pixel':
                self.data = np.arange(len(self.data)).astype(float)
            else:
                self.data = uc.convert_flux(
                    in_flux=self.data, start_unit=self.unit_key,
                    end_unit=target_unit, wavelength=wavelength,
                    wave_unit=wavelength_unit)
        else:
            msg = f'Unknown conversion kind: {self.kind}'
            log.error(msg)
            raise ValueError(msg)

        self.unit_key = target_unit
        try:
            self.unit = u.Unit(target_unit)
            self._verify_unit_parse()
        except (ValueError, RuntimeError):
            # allow unrecognized units to propagate anyway
            self.unit = target_unit
