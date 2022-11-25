#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
import uuid
from typing import (List, Dict, Optional, Union,
                    Any, Iterable, TypeVar, Tuple, Sequence)

from astropy import modeling as am
from matplotlib.axis import Axis
import numpy as np

from sofia_redux.visualization.utils.eye_error import EyeError

__all__ = ['ModelFit']


ArrayLike = TypeVar('ArrayLike', List, Tuple, Sequence, np.ndarray)


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
    id_tag : str
        Represents the model fit ID number.
    model_id : uuid.UUID
        The name of the data model the fit was made to. Typically,
        the name of a FITS file.
    filename : str
        Name of the file
    feature : str
        Name of the model used to fit the feature.
    background: str
        Name of the model used to fit the background.
        Typical examples are 'constant' or 'linear'.
    fit_type : list
        Strings describing the base model fit to the data. Typical
        values include 'gauss', 'moffat', 'const', 'linear'. Often
        composite models are fit to data, in which case all
        valid descriptions are included. The fit_type is used to
        define the parameter names to include.
    order : int
        Order number of the `model_id` data set that the model was
        fit to.
    aperture : int
        Aperture number of the `model_id` data set that the model was
        fit to.
    status : str
        Status of the last attempted fit. May be 'pass' or 'fail'.
    fit : am.Model
        The actual fit to the data.
    axis_names : list
        Names for the fit axes, used as keys for several other attributes.
    units : dict
        Name of the units used for each axis.
    limits : dict
        The upper and lower selection bounds of the data fit.
    fields : dict
        Field names for each axis.
    dataset : dict
        Data arrays for each axis.
    axis : ma.Axes
        The axes object the fit has been plotted on.
    visible : bool
        Flag to indicate visibility of the fit plot.
    columns : array-like
        Column values corresponding to the x-axis data.
    color : str
        Color hex value for the fit plot.
    param_names : dict
        Names of the parameters for each type of fitted model.
    """

    id_iter = itertools.count()
    """int : ID iteration value."""

    def __init__(self, params=None):
        self.id_tag = f'model_fit_{next(self.id_iter)}'
        self.model_id = ''
        self.filename = ''
        self.feature = ''
        self.background = ''
        self.fit_type = ['gauss']
        self.order = 0
        self.aperture = 0
        self.status = 'pass'
        self.fit = None
        self.axis_names = ['x', 'y']
        self.units = dict.fromkeys(self.axis_names, '')
        self.limits = dict.fromkeys(['lower', 'upper'], None)
        self.fields = dict.fromkeys(self.axis_names, '')
        self.dataset = dict.fromkeys(self.axis_names, None)
        self.axis = None
        self.visible = True
        self.columns = None
        self.color = None

        self.param_names = {'gauss': ['mean', 'stddev', 'amplitude'],
                            'moffat': ['x_0', 'gamma', 'amplitude', 'alpha'],
                            'linear': ['intercept', 'slope'],
                            'constant': ['amplitude']}

        if params:
            self.load_parameters(params)

    def get_id(self) -> str:
        """Get the ID for the current fit model."""
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
        """
        self.model_id = list(parameters.keys())[0]
        self.order = int(list(list(parameters.values())[0].keys())[0])
        params = parameters[self.model_id][self.order]

        self.filename = params.get('filename', '')
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

        Raises
        ------
        EyeError
            If no arguments are given.
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
        """
        Set the feature name for the model fitting.

        Parameters
        ----------
        feature : str
            Name of the model used to fit the feature.
        """
        self.feature = feature

    def get_feature(self) -> str:
        """
        Get the feature name for the model fitting.

        Returns
        -------
        feature : str
            Name of the model used to fit the feature.
        """
        return self.feature

    def set_background(self, background: str) -> None:
        """
        Set the background name for the model fitting.

        Parameters
        ----------
        background: str
            Name of the model used to fit the background.
            Typical examples are 'constant' or 'linear'.
        """
        self.background = background

    def get_background(self) -> str:
        """
        Get the background name for the model fitting

        Returns
        -------
        background: str
            Name of the model used to fit the background.
            Typical examples are 'constant' or 'linear'.
        """
        return self.background

    def get_fields(self, axis: Optional[str] = None
                   ) -> Optional[Union[str, Dict[str, str]]]:
        """
        Retrieve field names for a given axis.

        Parameters
        ----------
        axis : str, optional
            Names of the axes to be used as keys for fields.

        Returns
        -------
        fields : str or dict
            If an axis has been provided, the corresponding value
            is returned. Otherwise, a dictionary containing all values
            is returned.
        """
        if axis is not None:
            try:
                return self.fields[axis]
            except KeyError:
                return None
        else:
            return self.fields

    def set_fields(self, fields: Dict[str, str]) -> None:
        """
        Set field names.

        Parameters
        ----------
        fields : dict
            Keys are axis names; values are field names.
        """
        for key, field in fields.items():
            self.fields[key] = field

    def set_axis(self, axis: Axis) -> None:
        """
        Set the fitting axis.

        Parameters
        ----------
        axis : matplotlib.Axes
            Assign the fitting axis
        """
        self.axis = axis

    def get_axis(self) -> Axis:
        """
        Get the fitting axis.

        Returns
        -------
        axis : matplotlib.Axes
            The fitting axis.
        """
        return self.axis

    def set_status(self, status: str) -> None:
        """
        Set the last fit status.

        Parameters
        ----------
        status : str
            Status to set.
        """
        self.status = status

    def get_status(self) -> str:
        """
        Get the last fit status.

        Returns
        -------
        str
            Last status set.
        """
        return self.status

    def set_filename(self, filename: str) -> None:
        """
        Set the filename.

        Parameters
        ----------
        filename : str
            Filename to set.
        """
        self.filename = str(filename)

    def get_filename(self) -> str:
        """
        Get the filename.

        Returns
        -------
        str
            The filename.
        """
        return self.filename

    def set_model_id(self, model_id: uuid.UUID) -> None:
        """
        Set a model ID.

        Parameters
        ----------
        model_id : uuid.UUID
            Unique model ID to set.
        """
        self.model_id = model_id

    def get_model_id(self) -> uuid.UUID:
        """
        Get the model ID.

        Returns
        -------
        uuid.UUID
            The model ID.
        """
        return self.model_id

    def set_order(self, order: int) -> None:
        """
        Set an order number.

        Parameters
        ----------
        order : int
            The order to set.
        """
        self.order = order

    def get_order(self) -> int:
        """
        Get the order number.

        Returns
        -------
        int
            The order number.
        """
        return self.order

    def set_aperture(self, aperture: int) -> None:
        """
        Set an aperture number.

        Parameters
        ----------
        aperture : int
            The aperture number to set.
        """
        self.aperture = aperture

    def get_aperture(self) -> int:
        """
        Get the aperture number.

        Returns
        -------
        int
            The aperture number.
        """
        return self.aperture

    def set_columns(self, columns: ArrayLike) -> None:
        """
        Set column data.

        Parameters
        ----------
        columns : array-like
            The column data to set.
        """
        self.columns = columns

    def get_columns(self) -> int:
        """
        Get the columns array.

        Returns
        -------
        array-like
            The column data.
        """
        return self.columns

    def set_dataset(self, dataset: Optional[Dict[str, Iterable]] = None,
                    x: Optional[Iterable] = None,
                    y: Optional[Iterable] = None) -> None:
        """
        Set a data set to fit to.

        Parameters
        ----------
        dataset : dict, optional
            If provided, must have axis names for keys and data
            arrays for values.
        x : array-like, optional
            Directly specifies the x-axis data values.
        y : array-like, optional
            Directly specifies the y-axis data values.

        Raises
        ------
        EyeError
            If all expected axes are not provided.
        """
        if dataset is not None:
            self.dataset = dataset
        elif x is not None and y is not None:
            self.dataset['x'] = x
            self.dataset['y'] = y
        else:
            raise EyeError('Must provide all axes of dataset')

    def get_dataset(self) -> Dict[str, Optional[Iterable]]:
        """
        Get the data set to fit to.

        Returns
        -------
        dict
            Keys are axis names, values are data arrays.
        """
        return self.dataset

    @staticmethod
    def _determine_fit_type(fit: am.Model) -> str:
        """
        Map the type of a model fit to a string description.

        Parameters
        ----------
        fit : astropy.modeling.Model
            A simple, non-compound, astropy model.

        Returns
        -------
        fit_type : str
            String description of the type of fit. If fit type
            cannot be identified, return 'UNKNOWN'.
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
        """
        Set the visibility of a fit.

        Parameters
        ----------
        state : bool
            If the fitting is going to visible or not.
        """
        if isinstance(state, bool):
            self.visible = state

    def get_visibility(self) -> bool:
        """
        Get the visibility of the fitting model.

        Returns
        -------
        visible : bool
            True, if the fitting is going to visible otherwise False.
        """
        return self.visible

    def set_color(self, color: str) -> None:
        """
        Set the plot color for the fit artist.

        Parameters
        ----------
        color : str
            The color hex value to set.
        """
        self.color = color

    def get_color(self) -> str:
        """
        Get the plot color for the fit artist.

        Returns
        -------
        str
            The plot color.
        """
        return self.color

    def get_fit(self):
        """
        Obtain a copy of a fit.

        Returns
        -------
        fit : astropy.modeling.Model
            A simple, non-compound, astropy model.
            Returns None if there is no fitting model.
        """
        if self.fit is None:
            return None
        else:
            return self.fit.copy()

    def set_fit(self, fit) -> None:
        """
        Set the fitting model to a given fit.

        Parameters
        ----------
        fit : astropy.modeling.Model
            A simple, non-compound, astropy model.
        """
        self.fit = fit.copy()

    def get_units(self, key: Optional[str] = None
                  ) -> Optional[Union[Dict[str, str], str]]:
        """
        Obtain the units.

        Parameters
        ----------
        key : str, optional
            Obtain the unit for a given key (axis).

        Returns
        -------
        units : str or Dict
            A str if valid key is provided. For an invalid key it returns
            None. If no key is provided, it returns a dict.
        """
        if key:
            try:
                return self.units[key]
            except KeyError:
                return None
        else:
            return self.units

    def set_units(self, units: Dict[str, str]) -> None:
        """
        Set the units of the fitting model

        Parameters
        ----------
        units : dict
            Keys are axis names; values are corresponding units.
        """
        self.units = units

    def get_limits(self, limit: Optional[str] = None
                   ) -> Optional[Union[float, Dict[str, Optional[float]]]]:
        """
        Obtain the limit values.

        Parameters
        ----------
        limit : str, optional
            May be 'upper' or 'lower'.

        Returns
        -------
        float or dict
            If a valid limit name is specified, only that value is
            returned. If an invalid limit is specified, None is
            returned. Otherwise, the limits dictionary is returned.
        """
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
        """
        Set the limit values.

        Parameters
        ----------
        limits : float, dict, or list of list
            If key is provided, should be specified as a float.
            Otherwise, should contain both lower and upper limits.
            If provided as a list, it is assumed the lower limit is
            the first element in the first list and the upper limit
            is the second element in the first list.
        key : str, optional
            May be 'upper' or 'lower'.
        """
        if key:
            self.limits[key] = limits
        elif isinstance(limits, dict):
            self.limits = limits
        elif isinstance(limits, list):
            self.limits['lower'] = limits[0][0]
            self.limits['upper'] = limits[1][0]

    def get_fit_types(self) -> List[str]:
        """
        Get the fit type.

        Returns
        -------
        list of str
            Fit types as [feature, background].
        """
        return self.fit_type

    def parameters_as_string(self) -> Dict[str, str]:
        """
        Format all fitting details as strings for display in table.

        Returns
        -------
        param : dict
            Dictionary with parameter names for keys and string
            formatted parameter values for values.
        """
        param = {'model_id': str(self.model_id),
                 'filename': str(self.filename),
                 'order': f'{self.order+1:d}',
                 'aperture': f'{self.aperture+1:d}',
                 'x_field': f"{self.fields['x']} [{self.units['x']}]",
                 'y_field': f"{self.fields['y']} [{self.units['y']}]",
                 'lower_limit': f"{self.limits['lower']:.10g}",
                 'upper_limit': f"{self.limits['upper']:.10g}",
                 'type': ', '.join(self.fit_type),
                 'axis': self.axis,
                 'visible': self.visible,
                 'baseline': f'{self.get_baseline():.10g}',
                 'mid_point': f'{self.get_mid_point():.10g}',
                 'mid_point_column': f'{self.get_mid_point_column():.10g}',
                 'fwhm': f'{self.get_fwhm():.10g}'}

        if self.fit is not None:
            for i, fit_type in enumerate(self.fit_type):
                for name in self.param_names[fit_type]:
                    try:
                        value = getattr(self.fit, name).value
                    except AttributeError:
                        value = getattr(self.fit, f'{name}_{i}').value
                    param[name] = f'{value:.10g}'

        return param

    def parameters_as_dict(self) -> Dict[str, Any]:
        """
        Fit parameters in dictionary form for returning to the View.

        Returns
        -------
        param : dict
            Keys are the names of each parameter and values are
            the corresponding values.
        """
        param = {'model_id': self.model_id,
                 'filename': self.filename,
                 'order': self.order,
                 'aperture': self.aperture,
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
                 'mid_point_column': self.get_mid_point_column(),
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
        Fit parameters in a list structure for writing to disk.

        Returns
        -------
        params : list
            List of all fit parameter values. Parameter names
            are not included.
        """
        # For selection
        param = [self.filename, self.order, self.aperture,
                 self.fields['x'], self.fields['y'],
                 self.units['x'], self.units['y']
                 ]
        for i, fit_type in enumerate(self.fit_type):
            for name in self.param_names[fit_type]:
                try:
                    value = getattr(self.fit, name).value
                except AttributeError:
                    value = getattr(self.fit, f'{name}_{i}').value
                param.append(f'{value:.10g}')

        param.extend([f'{self.get_baseline():.10g}',
                      f'{self.get_mid_point():.10g}',
                      f'{self.get_mid_point_column():.10g}',
                      f'{self.get_fwhm():.10g}',
                      self.limits['lower'], self.limits['upper'],
                      self.visible])
        return param

    def parameters_as_html(self) -> Dict[str, str]:
        """
        Format all fitting details as HTML for display in text view.

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
        skip = ['axis', 'visible', 'model_id']
        for par, value in param.items():
            if par not in skip:
                html.append(f'  {par}: {value}')
        html.append('</pre></html>')
        return '<br>'.join(html)

    def get_mid_point(self) -> float:
        """
        Determine the midpoint of the feature.

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

    def get_mid_point_column(self) -> float:
        """
        Determine the midpoint of the feature in column number.

        Returns
        -------
        mid_point : float
            Midpoint of the feature of the fit in pixels. If not
            found, return NaN.

        """
        midpoint = self.get_mid_point()
        if np.isnan(midpoint):
            return np.nan
        try:
            mid_col = np.interp(midpoint, self.columns,
                                np.arange(self.columns.size))
        except (TypeError, ValueError, IndexError, AttributeError):
            mid_col = np.nan
        return mid_col

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
