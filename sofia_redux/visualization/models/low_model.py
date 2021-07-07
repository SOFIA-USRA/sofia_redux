# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Dict, Union, Sequence, TypeVar
import astropy.io.fits as pf
import astropy.units as u
import numpy as np

from sofia_redux.visualization import log
from sofia_redux.visualization.utils import unit_conversion as uc

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
    hdu : astropy.io.fits.HDU
        The FITS HDU to load.
    filename : str
        Filename associated with the FITS HDU.
    kwargs : dict, optional
        Optional keywords to set `name`, `data`, and `kind`
        properties.
    """
    def __init__(self, hdu: pf.ImageHDU, filename: str, **kwargs) -> None:
        self.hdu = hdu
        self.filename = filename
        self.unit = None
        self.unit_key = ''

        self.name = kwargs.get('name', self.hdu.name.lower())
        self.data = kwargs.get('data', self.hdu.data)
        self.kind = kwargs.get('kind', '')

        self.available_units = self._define_units()
        self.kind_names = dict()

        self.default_field = None
        self.default_ndims = 0
        self.id = None
        self.enabled = True

    @staticmethod
    def _define_units() -> Dict[str, Dict[str, Union[str, u.Quantity]]]:
        """Define common units and their astropy equivalent."""
        wavelength_units = {'um': u.um, 'nm': u.nm,
                            u'\u212B': u.angstrom,
                            }
        flux_units = {'Jy': u.Jy,
                      'W / m2': u.W / u.m ** 2,
                      'W / (m2 Hz)': u.W / (u.m ** 2 * u.Hz),
                      'W / (m2 um)': u.W / (u.m ** 2 * u.um),
                      'erg / (s cm2 Angstrom)':
                          u.erg / (u.s * u.cm ** 2 * u.AA),
                      'erg / (s cm2 Hz)': u.erg / (u.s * u.cm ** 2 * u.Hz),
                      }
        scale_units = {'': ''}
        unitless = {'': u.dimensionless_unscaled}
        position_units = {'arcsec': u.arcsec,
                          'degree': u.deg,
                          'arcmin': u.arcmin,
                          }

        available_units = {'scale': scale_units,
                           'unitless': unitless,
                           'position': position_units,
                           'flux': flux_units,
                           'wavelength': wavelength_units}
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
                header_unit = self.hdu.header[key]
            except KeyError:
                continue
            else:
                break
        unit = kwargs.get('unit', header_unit)
        if unit is None:
            unit = ''

        self.unit = uc.parse_unit(unit)
        self.unit_key = str(self.unit)
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
        for kind, names in self.kind_names.items():
            if self.name in names:
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
        """Retrieve data array from the model."""
        return self.data


class Image(LowModel):
    """Low-level data model for an image."""

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.default_ndims = 2
        self.kind = ''

        self.kind_names = {'scale': ['transmission'],
                           'position': ['aperture_trace'],
                           'unitless': ['badmask', 'spatial_map'],
                           'flux': ['flux', 'error',
                                    'spectral_flux', 'spectral_error'],
                           'time': ['exposure']}
        self._parse_kind()

    def data_mean(self) -> float:
        """Calculate mean of data."""
        try:
            mean = np.nanmean(self.data)
        except (ValueError, AttributeError):
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
    """Class for holding simple, 1d spectra."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.default_ndims = 1
        self.id = f'{self.filename}/{self.name}'

        self.kind_names = {'scale': ['transmission', 'response',
                                     'response_error'],
                           'unitless': ['spatial_profile'],
                           'flux': ['spectral_flux', 'spectral_error'],
                           'wavelength': ['wavepos', 'slitpos'],
                           'position': ['slitpos']}
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
        # TODO - add pixel conversion handling.
        #  right now, would destroy calibration data
        if self.kind == 'flux':
            self.data = uc.convert_flux(
                in_flux=self.data, start_unit=self.unit_key,
                end_unit=target_unit, wavelength=wavelength,
                wave_unit=wavelength_unit)
        elif self.kind in ['wavelength', 'position']:
            # this will work for any simple conversions handled
            # by astropy
            self.data = uc.convert_wave(
                wavelength=self.data, start_unit=self.unit_key,
                end_unit=target_unit)
        elif self.kind in ['scale', 'unitless']:
            # no conversions to do for these types
            return
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
