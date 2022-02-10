# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log, units
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

from sofia_redux.scan.utilities.range import Range

__all__ = ['SkyDipModel']


class SkyDipModel(ABC):

    default_initial_guess = {
        'tsky': 273.0,  # Kelvin
        'offset': np.nan,
        'kelvin': np.nan,  # Kelvin
        'tau': 1.0
    }
    default_bounds = {
        'tsky': [0.0, np.inf],
        'offset': [-np.inf, np.inf],
        'kelvin': [0.0, np.inf],
        'tau': [0.0, 10.0]
    }

    def __init__(self):
        """
        Initialize a sky dip model.
        """
        self.configuration = None
        self.initial_guess = self.default_initial_guess.copy()
        self.bounds = self.default_bounds.copy()
        self.fit_for = None
        self.has_converged = False
        self.data_unit = units.Unit("count")
        self.use_points = 0
        self.uniform_weights = False
        self.el_range = Range()
        self.parameters = None
        self.errors = None
        self.rms = np.nan
        self.fitted_values = None
        self.elevation = None
        self.data = None
        self.sigma = None
        self.p_opt = None
        self.p_cov = None

    def set_configuration(self, configuration):
        """
        Set the sky dip model configuration

        Parameters
        ----------
        configuration : Configuration

        Returns
        -------
        None
        """
        self.configuration = configuration
        if self.configuration.is_configured('skydip.elrange'):
            self.el_range = self.configuration.get_range(
                'skydip.elrange', is_positive=True)
            self.el_range.scale(units.Unit('degree'))

        self.uniform_weights = self.configuration.get_bool('skydip.uniform')
        self.fit_for = []
        if self.configuration.is_configured('skydip.fit'):
            names = self.configuration.get_string_list('skydip.fit')
            names = [x.strip().lower() for x in names]
            for name in names:
                if name in ['tau', 'offset', 'kelvin', 'tsky']:
                    self.fit_for.append(name)
                elif name == 'data2k':
                    self.fit_for.append('kelvin')
        else:
            self.fit_for.extend(['tau', 'offset', 'kelvin'])
        self.fit_for = list(np.unique(self.fit_for))

    def init_parameters(self, skydip):
        """
        Initialize the fitting parameters.

        Parameters
        ----------
        skydip : SkyDip
            The SkyDip model to fit.

        Returns
        -------
        None
        """
        if self.configuration.is_configured('skydip.tsky'):
            self.initial_guess['tsky'] = self.configuration.get_float(
                'skydip.tsky')
        elif skydip.tamb_weight > 0:
            temp = skydip.tamb
            if isinstance(temp, units.Quantity):
                temp = temp.to('Kelvin', equivalencies=units.temperature()
                               ).value
            self.initial_guess['tsky'] = temp

        signal_range = skydip.get_signal_range()
        if not np.isfinite(self.initial_guess['offset']):
            offset = signal_range.midpoint
            if np.isnan(offset):
                offset = 0.0
            self.initial_guess['offset'] = offset

        tsky = self.initial_guess['tsky']

        if not np.isfinite(self.initial_guess['kelvin']):
            kelvin = signal_range.span / tsky
            if not np.isfinite(kelvin):
                kelvin = 1.0
            self.initial_guess['kelvin'] = kelvin
            if 'kelvin' not in self.fit_for:
                self.fit_for.append('kelvin')
        else:
            kelvin = self.initial_guess['kelvin']
            am_range = skydip.get_air_mass_range()
            x = signal_range.span / (am_range.span * tsky * kelvin)
            if x < 0:
                tau = 0.1
            elif x >= 1:
                tau = 1.0
            else:
                tau = -np.log(1 - x)
            self.initial_guess['tau'] = tau

        for key, value in self.initial_guess.items():
            if isinstance(value, units.Quantity):
                self.initial_guess[key] = value.value

    def fit(self, skydip):
        """
        Fit the skydip model.

        Parameters
        ----------
        skydip : SkyDip

        Returns
        -------
        None
        """
        parameter_order = ['tau', 'offset', 'kelvin', 'tsky']
        self.parameters = {}
        self.errors = {}
        self.p_opt = None
        self.p_cov = None
        self.fitted_values = None
        self.data = None
        self.sigma = None
        self.elevation = None

        log.debug("Initial skydip values:")
        log.debug(f"    Tsky = {self.initial_guess['tsky']}")
        log.debug(f"    offset = {self.initial_guess['offset']}")
        log.debug(f"    kelvin = {self.initial_guess['kelvin']}")
        log.debug(f"    tau = {self.initial_guess['tau']}")

        if self.el_range is not None:
            from_bin = max(0, skydip.get_bin(self.el_range.min))
            to_bin = min(skydip.data.size, skydip.get_bin(self.el_range.max))
        else:
            from_bin = 0
            to_bin = skydip.data.size

        self.init_parameters(skydip)

        data = skydip.data[from_bin:to_bin]
        weight = skydip.weight[from_bin:to_bin]
        valid = weight > 0
        data = data[valid]
        weight = weight[valid]

        if self.uniform_weights:
            sigma = None
        else:
            sigma = 1 / weight

        elevation = skydip.get_elevation(
            np.nonzero(valid)[0]).to('radian').value

        self.use_points = data.size

        p0 = []
        lower_bounds = np.zeros(4, dtype=float)
        upper_bounds = np.zeros(4, dtype=float)

        for i, parameter in enumerate(parameter_order):
            value = self.initial_guess[parameter]
            p0.append(value)
            if parameter in self.fit_for:
                lower_bounds[i] = self.bounds[parameter][0]
                upper_bounds[i] = self.bounds[parameter][1]
            else:  # An attempt to fix parameters with curve_fit
                eps = abs(value - np.nextafter(value, 1))
                lower_bounds[i] = value - eps
                upper_bounds[i] = value + eps

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            p_opt, p_cov = curve_fit(self.value_at, elevation, data,
                                     p0=p0, sigma=sigma,
                                     bounds=(lower_bounds, upper_bounds))
        self.p_opt = p_opt
        self.p_cov = p_cov
        self.data = data
        self.elevation = elevation
        self.sigma = sigma

        self.has_converged = np.isfinite(p_opt).all()
        if not self.has_converged:
            log.warning("Skydip fit did not converge!")
        errors = np.sqrt(np.diag(p_cov))

        for i, parameter in enumerate(parameter_order):
            self.parameters[parameter] = p_opt[i]
            self.errors[parameter] = errors[i]

        self.fitted_values = self.fit_elevation(elevation)
        fit_weights = None if sigma is None else weight ** 2

        t_obs_rms = np.sqrt(np.average((data - self.fitted_values) ** 2,
                                       weights=fit_weights))
        self.rms = t_obs_rms / self.parameters['kelvin']

    def fit_elevation(self, elevation):
        """
        Returns a fit to elevation with the model.

        The return value Tobs is given as:

        fit = offset + (t_obs * kelvin)
        t_obs = t_sky * (-(exp(-tau / sin(el) - 1))

        where t_sky is the sky temperature, kelvin is the conversion factor
        from instrument units to kelvin, offset is the signal offset, and
        el is the elevation.

        Parameters
        ----------
        elevation : units.Quantity or float or numpy.ndarray (float)
            The elevations to fit.  If floats are provided, they should be
            in radians.

        Returns
        -------
        fit : units.Quantity
            The fit in Kelvin
        """
        if self.p_opt is None:
            return elevation * np.nan
        return self.value_at(elevation, *self.p_opt)

    @staticmethod
    def value_at(elevation, tau, offset, kelvin, tsky):
        """
        Return the result of the fitted value.

        Parameters
        ----------
        elevation : float
            The elevation in radians.
        tau : float
            The tau value.
        offset : float
            The offset in kelvins.
        kelvin : float
            The kelvin scaling factor.
        tsky : float
            The sky temperature in kelvin.

        Returns
        -------
        value : float
        """
        eps = -(np.exp(-tau / np.sin(elevation)) - 1)
        t_obs = eps * tsky
        return offset + (t_obs * kelvin)

    def get_parameter_string(self, parameter):
        """
        Return a string representation of a given parameter.

        Parameters
        ----------
        parameter : str

        Returns
        -------
        str
        """
        if not self.has_converged or self.parameters is None:
            return None
        if parameter not in self.parameters:
            return None

        fmt = self.get_parameter_format(parameter)
        unit = self.get_parameter_unit(parameter)
        value = fmt % self.parameters[parameter]

        error = self.errors[parameter]
        if np.isfinite(error):
            error = fmt % error
        else:
            error = None

        s = f"{parameter} = {value}"
        if error is not None:
            s += f' +/- {error}'
        if unit is not None:
            s += f' {unit}'

        return s

    @classmethod
    def get_parameter_format(cls, parameter_name):
        """
        Return the string format for a given parameter.

        Parameters
        ----------
        parameter_name : str

        Returns
        -------
        str
        """
        formats = {
            'tau': '%.3f',
            'tsky': '%.1f',
            'kelvin': '%.3e'
        }
        return formats.get(parameter_name, '%.3e')

    def get_parameter_unit(self, parameter_name):
        """
        Return the parameter unit for the given parameter.

        Parameters
        ----------
        parameter_name : str

        Returns
        -------
        units.Unit or None
        """
        parameter_units = {
            'tsky': units.Unit("Kelvin"),
            'kelvin': self.data_unit
        }
        return parameter_units.get(parameter_name)

    def __str__(self):
        """
        Return a string representation of the sky dip fit.

        Returns
        -------
        str
        """
        if not self.has_converged or self.parameters is None:
            log.warning("The fit has not converged. Try again!")
            return ''

        result = []
        for parameter in self.parameters.keys():
            if parameter in self.fit_for:
                parameter_string = self.get_parameter_string(parameter)
                if parameter_string is not None:
                    result.append(parameter_string)

        rms = self.get_parameter_format('kelvin') % self.rms
        result.append(f"[{rms} K rms]")
        return '\n'.join(result)
