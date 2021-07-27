# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import Dict, Any
import warnings
import numpy as np
import astropy.units as u
import astropy.modeling as am

from sofia_redux.visualization import log

__all__ = ['parse_unit', 'convert_flux', 'convert_wave']

Me = u.def_unit('Me', u.Mct)
sec = u.def_unit('sec', u.s)
cm1 = u.def_unit('cm-1', u.kayser)
u.add_enabled_units([Me, sec, cm1])


def parse_unit(unit_string):
    """
    Parse a unit string into a unit object.

    Parameters
    ----------
    unit_string : str
        Unit to be parsed.  Generally, any unit understood by
        the `astropy.units` module is supported.  Additionally,
        'cm-1' is supported as an alias for the 'kayser' unit.

    Returns
    -------
    astropy.units.Unit

    Raises
    ------
    ValueError
        If the unit is not recognized.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='astropy')
        try:
            unit = u.Unit(unit_string, parse_strict='raise')
        except ValueError:
            new_string = unit_string.replace('(cm-1)', 'kayser')
            log.debug(f'\tValue Error, try {new_string}')
            unit = u.Unit(new_string,
                          parse_strict='silent')
    log.debug(f'Parsed {unit_string} into {unit} ({type(unit)}).')
    return unit


def convert_flux(in_flux, start_unit, end_unit, wavelength,
                 wave_unit):
    """
    Convert flux data to new units.

    The wavelength array is required to convert from some flux units
    to others, via the `astropy.units.spectral_density` equivalency.

    Parameters
    ----------
    in_flux : array-like
        Input flux data to convert.
    start_unit : astropy.units.Unit or str
        Starting unit for flux data, to convert from.
    end_unit : astropy.units.Unit or str
        Desired unit for flux data, to convert to.
    wavelength : array-like
        Wavelength data associated with the flux.
    wave_unit : astropy.units.Unit or str
        Current wavelength unit.

    Returns
    -------
    out_flux : array-like
        Converted flux array.  The wavelength array is unchanged.

    Raises
    ------
    ValueError
        If the input and output units are incompatible.
    """
    # check for same input values
    if start_unit == end_unit:
        log.debug(f'Start and end units are same: {start_unit}')
        return in_flux

    if isinstance(wave_unit, str):
        wave_unit = u.Unit(wave_unit)
    if isinstance(start_unit, str):
        start_unit = u.Unit(start_unit, str)
    if isinstance(end_unit, str):
        end_unit = u.Unit(end_unit, str)

    # check again in case new unit equalities were found
    if start_unit == end_unit:
        log.debug(f'Start and end units are same: {start_unit}')
        return in_flux

    log.debug(f'Converting {len(wavelength)} of {type(wavelength)} '
              f'{type(wavelength[0])}')
    wave = wavelength * wave_unit
    in_flux = in_flux * start_unit
    try:
        out_flux = in_flux.to(end_unit, equivalencies=u.spectral_density(wave))
    except u.core.UnitConversionError:
        log.debug(f'Units not convertible: {start_unit} -> {end_unit}')
        raise ValueError('Inconvertible units') from None

    out_flux = out_flux.value
    return out_flux


def convert_wave(wavelength: np.array, start_unit: str,
                 end_unit: str) -> np.array:
    """
    Convert wavelength data to new units.

    Parameters
    ----------
    wavelength : array-like
        Input wavelength data to convert.
    start_unit : str
        Starting unit for wavelength data, to convert from.
    end_unit : str
        Desired unit for wavelength data, to convert to.

    Returns
    -------
    out_wave : array-like
        Converted wavelength array.

    Raises
    ------
    ValueError
        If the input and output units are incompatible.
    """
    start = u.Unit(start_unit)
    end = u.Unit(end_unit)
    if start == end:
        log.debug(f'Start and end units are same: {start_unit}')
        return wavelength

    log.debug(f'Converting wavelength from {start_unit} to {end_unit}')
    try:
        out_wave = (wavelength * start).to(end).value
    except u.core.UnitConversionError:
        log.debug(f'Units not convertible: {start_unit} -> {end_unit}')
        raise ValueError('Inconvertible units') from None

    return out_wave


def convert_model_fit(fit: am.Model, start_units: Dict[str, Any],
                      end_units: Dict[str, Any], wave: np.ndarray):
    input_units = _confirm_quantity(start_units)
    output_units = _confirm_quantity(end_units)
    old_fit = fit.with_units_from_data(**input_units)
    new_fit = fit.with_units_from_data(**output_units)

    changed = False
    equivs = u.spectral_density(wave * input_units['x'])
    for name in fit.param_names:
        old_param = getattr(old_fit, name)
        new_param = getattr(new_fit, name)

        if new_param.unit is None or new_param.unit == old_param.unit:
            continue
        try:
            quantity = (old_param.value * old_param.unit).to(
                new_param.unit).value
        except u.core.UnitConversionError:
            try:
                quantity = (old_param.value * old_param.unit).to(
                    new_param.unit, equivalencies=equivs).value
            except (u.core.UnitConversionError, IndexError):
                try:
                    quantity = _convert_slope(old_param, new_param, equivs,
                                              input_units, output_units)
                except (u.core.UnitConversionError, IndexError):
                    raise ValueError(f'Unable to convert units for '
                                     f'{old_param} from {old_param.unit} '
                                     f'to {new_param.unit}') from None

        setattr(fit, name, quantity)
        changed = True
    return fit, changed


def _confirm_quantity(units):
    quantities = dict()
    for key, value in units.items():
        if isinstance(value, str):
            quant = 1 * parse_unit(value)
        elif isinstance(value, u.UnitBase):
            quant = 1 * value
        elif isinstance(value, u.quantity.Quantity):
            quant = value
        else:
            quant = None
        key = key.split('_')[0]
        quantities[key] = quant
    return quantities


def _convert_slope(old_param, new_param, equivs, input_units, output_units):
    # remove denominator to get just flux
    old_top = old_param * input_units['x']
    new_top = new_param * input_units['x']

    try:
        converted_top = (old_top.value * old_top.unit).to(
            new_top.unit).value
    except u.core.UnitConversionError:
        converted_top = (old_top.value * old_top.unit).to(
            new_top.unit, equivalencies=equivs).value

    try:
        quantity = converted_top * input_units['x'].to(output_units['x'])
    except u.core.UnitConversionError:  # pragma: no cover
        # no use case for this clause yet
        quantity = converted_top * input_units['x'].to(output_units['x'],
                                                       equivalencies=equivs)
    try:
        return quantity.value
    except AttributeError:
        return quantity
