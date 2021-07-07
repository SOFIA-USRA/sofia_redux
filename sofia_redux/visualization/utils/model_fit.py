#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
from typing import (List, Dict, Optional, Union,
                    Any, Iterable)

from astropy import modeling as am
from matplotlib.axis import Axis
import numpy as np

from sofia_redux.visualization.utils.eye_error import EyeError

__all__ = ['ModelFit']


class ModelFit(object):
    """
    Hold parameters and descriptions of generic model fits.

    Parameters
    ----------
    params : dict, optional
        A nested dictionary that describes a single fit to data.
        The top level key is the `model_id` and the corresponding
        value is a dictionary whose only key is the `order`. The
        values of the inner dictionary are the actual parameters
        of the fit. If not provided, an empty instance of `ModelFit`
        with default values is returned.

    Attributes
    ----------
    model_id : str
        The name of the data model the fit was made to. Typically
        the name of a FITS file.
    fit_type : list
        Strings describing the base model fit to the data. Typical
        values include 'gauss', 'moffat', 'const', 'linear'. Often
        composite models are fit to data, in which case all
        valid descriptions are included. The fit_type is used to
        define the parameter names to include.
    order : int
        Order number of the `model_id` data set that the model was
        fit to.
    fit : am.Model
        The actual fit to the data as described in astropy.
    axis_names : list
        Names of the axes to be used as keys for several parameters.
    units : dict
        String descriptions of the units used on each axis.
    limits : dict
        The upper and lower selection bounds of the data fit.
    axis : ma.Axes
        The axes object the fit has been plotted on.
    visible : bool
        Flag on if the fit artist is visible or not.
    param_names : dict
        Names of the parameters for each type of fitted model.

    Methods
    -------
    load_parameters
        Parses fit parameters from a nested dictionary to initialize
        the instance of `ModelFit`.
    matches
        Determines if the `ModelFit` matches with the provided arguments.
    parameters_as_string
    parameters_as_dict
    parameters_as_list
    parameters_as_html
        Formats the parameters into the desired data structure,
        for easier handling by other classes such as `View`
        or `FittingResults`.
    """

    id_iter = itertools.count()

    def __init__(self, params=None):
        self.id_tag = f'model_fit_{next(self.id_iter)}'
        self.model_id = ''
        self.feature = ''
        self.background = ''
        self.fit_type = ['gauss']
        self.order = 0
        self.status = 'pass'
        self.fit = None
        self.axis_names = ['x', 'y']
        self.units = dict.fromkeys(self.axis_names, '')
        self.limits = dict.fromkeys(['lower', 'upper'], None)
        self.fields = dict.fromkeys(self.axis_names, '')
        self.dataset = dict.fromkeys(self.axis_names, None)
        self.axis = None
        self.visible = True

        self.param_names = {'gauss': ['mean', 'stddev', 'amplitude'],
                            'moffat': ['x_0', 'gamma', 'amplitude', 'alpha'],
                            'linear': ['intercept', 'slope'],
                            'constant': ['amplitude']}

        if params:
            self.load_parameters(params)

    def get_id(self) -> str:
        return self.id_tag

    def load_parameters(self, parameters: Dict[str, Dict]) -> None:
        """
        Parse model fit parameters.

        Parameters
        ----------
        parameters : dict
            A nested dictionary that describes a single fit to data.
            The top level key is the `model_id` and the corresponding
            value is a dictionary whose only key is the `order`. The
            values of the inner dictionary are the actual parameters
            of the fit. If not provided, an empty instance of `ModelFit`
            with default values is returned.

        Returns
        -------
        None

        """
        self.model_id = list(parameters.keys())[0]
        self.order = int(list(list(parameters.values())[0].keys())[0])
        params = parameters[self.model_id][self.order]

        keys = ['fit', 'axis', 'status']
        for key in keys:
            setattr(self, key, params.get(key, None))
        self.visible = params.get('visible', True)

        for axis in self.axis_names:
            self.units[axis] = params.get(f'{axis}_unit', None)
            self.fields[axis] = params.get(f'{axis}_field', None)
        self.limits['lower'] = params.get('lower_limit', None)
        self.limits['upper'] = params.get('upper_limit', None)

        if isinstance(self.fit, am.CompoundModel):
            self.fit_type = [self._determine_fit_type(fit) for
                             fit in self.fit]
        else:
            self.fit_type = [self._determine_fit_type(self.fit)]

    def set_fit_type(self, fit_type: Optional[Union[str, List[str]]] = None,
                     feature: Optional[str] = None,
                     background: Optional[str] = None):
        """
        Set the type of fit used.

        Parameters
        ----------
        fit_type : list, optional
            If given, `self.fits` will be set to it directly.
            If not given, the fit type will be parsed from
            `feature` and `background`.
        feature : str, optional
            Name of the model used to fit the feature.
            Typical examples are 'gaussian' or 'moffat'.
        background : str, optional
            Name of the model used to fit the background.
            Typical examples are 'constant' or 'linear'.

        Returns
        -------
        None

        Raises
        ------
        EyeError
            If no arguments are given

        """
        if fit_type:
            if not isinstance(fit_type, list):
                fit_type = [fit_type]
            self.fit_type = fit_type
        elif feature or background:
            self.fit_type = list()
            if feature:
                self.fit_type.append(feature)
                self.set_feature(feature)
            if background:
                self.fit_type.append(background)
                self.set_background(background)
        else:
            raise EyeError('Need to provide fit type')

    def set_feature(self, feature: str) -> None:
        self.feature = feature

    def get_feature(self) -> str:
        return self.feature

    def set_background(self, background: str) -> None:
        self.background = background

    def get_background(self) -> str:
        return self.background

    def get_fields(self, axis: Optional[str] = None
                   ) -> Optional[Union[str, Dict[str, str]]]:
        if axis is not None:
            try:
                return self.fields[axis]
            except KeyError:
                return None
        else:
            return self.fields

    def set_fields(self, fields: Dict[str, str]) -> None:
        for key, field in fields.items():
            self.fields[key] = field

    def set_axis(self, axis: Axis) -> None:
        self.axis = axis

    def get_axis(self) -> Axis:
        return self.axis

    def set_status(self, status: str) -> None:
        self.status = status

    def get_status(self) -> str:
        return self.status

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id

    def get_model_id(self) -> str:
        return self.model_id

    def set_order(self, order: int) -> None:
        self.order = order

    def get_order(self) -> int:
        return self.order

    def set_dataset(self, dataset: Optional[Dict[str, Iterable]] = None,
                    x: Optional[Iterable] = None,
                    y: Optional[Iterable] = None) -> None:
        if dataset is not None:
            self.dataset = dataset
        elif x is not None and y is not None:
            self.dataset['x'] = x
            self.dataset['y'] = y
        else:
            raise EyeError('Must provide all axes of dataset')

    def get_dataset(self) -> Dict[str, Optional[Iterable]]:
        return self.dataset

    @staticmethod
    def _determine_fit_type(fit: am.Model) -> str:
        """
        Map the type of a model fit to a string description

        Parameters
        ----------
        fit : astropy.modeling.Model
            A simple, non-compound, astropy model.

        Returns
        -------
        fit_type : str
            String description of the type of fit. If fit type
            cannot be identified, return 'UNKNOWN'

        """
        if isinstance(fit, am.models.Gaussian1D):
            return 'gauss'
        elif isinstance(fit, am.models.Moffat1D):
            return 'moffat'
        elif isinstance(fit, am.models.Linear1D):
            return 'linear'
        elif isinstance(fit, am.models.Const1D):
            return 'constant'
        else:
            return 'UNKNOWN'

    def set_visibility(self, state: bool):
        if isinstance(state, bool):
            self.visible = state

    def get_visibility(self) -> bool:
        return self.visible

    def get_fit(self):
        if self.fit is None:
            return None
        else:
            return self.fit.copy()

    def set_fit(self, fit) -> None:
        self.fit = fit.copy()

    def get_units(self, key: Optional[str] = None
                  ) -> Optional[Union[Dict[str, str], str]]:
        if key:
            try:
                return self.units[key]
            except KeyError:
                return None
        else:
            return self.units

    def set_units(self, units: Dict[str, str]) -> None:
        self.units = units

    def get_limits(self, limit: Optional[str] = None
                   ) -> Optional[Union[float, Dict[str, Optional[float]]]]:
        if limit is not None:
            try:
                return self.limits[limit]
            except KeyError:
                return None
        else:
            return self.limits

    def set_limits(self,
                   limits: Union[float, Dict[str, float], List[List[float]]],
                   key: Optional[str] = None) -> None:
        if key:
            self.limits[key] = limits
        elif isinstance(limits, dict):
            self.limits = limits
        elif isinstance(limits, list):
            self.limits['lower'] = limits[0][0]
            self.limits['upper'] = limits[1][0]

    def get_fit_types(self) -> List[str]:
        return self.fit_type

    def parameters_as_string(self) -> Dict[str, str]:
        """
        Format all fitting details as strings for display in table

        Returns
        -------
        param : dict
            Dictionary with parameter names for keys and string
            formatted parameter values for values.

        """
        param = {'model_id': self.model_id,
                 'order': f'{self.order:d}',
                 'x_field': f"{self.fields['x']} [{self.units['x']}]",
                 'y_field': f"{self.fields['y']} [{self.units['y']}]",
                 'lower_limit': f"{self.limits['lower']:.5g}",
                 'upper_limit': f"{self.limits['upper']:.5g}",
                 'type': ', '.join(self.fit_type),
                 'axis': self.axis,
                 'visible': self.visible,
                 'baseline': f'{self.get_baseline():.5g}',
                 'mid_point': f'{self.get_mid_point():.5g}',
                 'fwhm': f'{self.get_fwhm():.5g}'}

        if self.fit is not None:
            for i, fit_type in enumerate(self.fit_type):
                for name in self.param_names[fit_type]:
                    try:
                        value = getattr(self.fit, name).value
                    except AttributeError:
                        value = getattr(self.fit, f'{name}_{i}').value
                    param[name] = f'{value:.5g}'

        return param

    def parameters_as_dict(self) -> Dict[str, Any]:
        """
        Fit parameters in dictionary form for returning to `View`.

        Returns
        -------
        param : dict
            Keys are the names of each parameter and values are
            the corresponding values.

        """
        param = {'model_id': self.model_id, 'order': self.order,
                 'x_field': self.fields['x'],
                 'y_field': self.fields['y'],
                 'x_unit': self.units['x'],
                 'y_unit': self.units['y'],
                 'fit': self.fit,
                 'lower_limit': self.limits['lower'],
                 'upper_limit': self.limits['upper'],
                 'type': self.fit_type,
                 'axis': self.axis,
                 'visible': self.visible,
                 'baseline': self.get_baseline(),
                 'mid_point': self.get_mid_point(),
                 'fwhm': self.get_fwhm()}
        for i, fit_type in enumerate(self.fit_type):
            for name in self.param_names[fit_type]:
                try:
                    value = getattr(self.fit, name).value
                except AttributeError:
                    value = getattr(self.fit, f'{name}_{i}').value
                param[name] = value
        return param

    def parameters_as_list(self) -> List[Any]:
        """
        Fit parameters in a list structure for writing to disk

        Returns
        -------
        params : list
            List of all fit parameter values. Parameter names
            are not included.

        """
        # For selection
        param = [self.model_id, self.order,
                 self.fields['x'], self.fields['y'],
                 self.units['x'], self.units['y']
                 ]
        for i, fit_type in enumerate(self.fit_type):
            for name in self.param_names[fit_type]:
                try:
                    value = getattr(self.fit, name).value
                except AttributeError:
                    value = getattr(self.fit, f'{name}_{i}').value
                param.append(f'{value:.5g}')

        param.extend([f'{self.get_baseline():.5g}',
                      f'{self.get_mid_point():.5g}',
                      f'{self.get_fwhm():.5g}',
                      self.limits['lower'], self.limits['upper'],
                      self.visible])
        return param

    def parameters_as_html(self) -> Dict[str, str]:
        """
        Format all fitting details as HTML for display in text view

        Returns
        -------
        param : dict
            Dictionary with parameter names for keys and string
            formatted parameter values for values.

        """
        param = self.parameters_as_string()
        html = ['<html>']

        status = self.get_status()
        if status == 'pass':
            style = 'color: green'
        else:
            # Fit was not successful
            style = 'color: red'
        status_string = status.replace('_', ' ').title()
        html.append(f'Last fit status: '
                    f'<span style="{style}">{status_string}</span>')
        html.append('Parameters: <pre>')
        skip = ['axis', 'visible']
        for par, value in param.items():
            if par not in skip:
                html.append(f'  {par}: {value}')
        html.append('</pre></html>')
        return '<br>'.join(html)

    def get_mid_point(self) -> float:
        """
        Determine the mid point of the feature.

        Returns
        -------
        mid_point : float
            Midpoint of the feature of the fit. If not
            found, return NaN.

        """
        if not self.fit:
            return np.nan
        if isinstance(self.fit, am.CompoundModel):
            fits = self.fit
        else:
            fits = [self.fit]

        for model in fits:
            if isinstance(model, am.models.Gaussian1D):
                return model.mean.value
            elif isinstance(model, am.models.Moffat1D):
                return model.x_0.value

        # if no feature was fit, but limits are available,
        # return the midpoint of the fit range
        try:
            return (self.limits['lower'] + self.limits['upper']) / 2
        except (TypeError, ValueError):
            return np.nan

    def get_fwhm(self) -> float:
        """
        Determine the FWHM of the feature.

        Returns
        -------
        fwhm : float
            FWHM of the feature of the fit. If not
            found, return NaN.

        """
        if not self.fit:
            return np.nan
        if isinstance(self.fit, am.CompoundModel):
            fits = self.fit
        else:
            fits = [self.fit]
        for model in fits:
            if isinstance(model, am.models.Gaussian1D):
                return model.fwhm
            elif isinstance(model, am.models.Moffat1D):
                return model.fwhm
        return np.nan

    def get_baseline(self) -> float:
        """
        Determine the baseline of the feature.

        Returns
        -------
        baseline : float
            Baseline of the feature of the fit. If not
            found, return NaN.

        """
        if not self.fit:
            return np.nan
        midpoint = self.get_mid_point()
        if np.isnan(midpoint):
            return np.nan

        if isinstance(self.fit, am.CompoundModel):
            fits = self.fit
        else:
            fits = [self.fit]

        for model in fits:
            if isinstance(model, am.models.Linear1D):
                return model.slope.value * midpoint \
                    + model.intercept.value
            elif isinstance(model, am.models.Const1D):
                return model.amplitude.value

        # if no background fit was found, model was feature only,
        # so return 0
        return 0.0

    def scale_parameters(self, x_scale: float, y_scale: float) -> None:
        """
        Scale parameters after a fit.

        This is used to restore a baseline scale, after fitting
        to normalized data.

        Parameters
        ----------
        x_scale : float
            Value to scale the x data.
        y_scale : float
            Value to scale the y data
        """
        if not self.fit:
            return
        if isinstance(self.fit, am.CompoundModel):
            fits = self.fit
        else:
            fits = [self.fit]

        for model in fits:
            if isinstance(model, am.models.Linear1D):
                model.intercept *= y_scale
                model.slope *= y_scale / x_scale
            elif isinstance(model, am.models.Const1D):
                model.amplitude *= y_scale
            elif isinstance(model, am.models.Gaussian1D):
                model.mean *= x_scale
                model.stddev *= x_scale
                model.amplitude *= y_scale
            elif isinstance(model, am.models.Moffat1D):
                model.x_0 *= x_scale
                model.gamma *= x_scale
                model.amplitude *= y_scale
