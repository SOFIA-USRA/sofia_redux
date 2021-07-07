# Licensed under a 3-clause BSD style license - see LICENSE.rst

from inspect import isclass
import warnings

from astropy import log
from astropy.modeling import models
from astropy.modeling import fitting, Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.utils.exceptions import AstropyWarning
import numpy as np

from sofia_redux.toolkit.utilities.func import recursive_dict_update
from sofia_redux.toolkit.stats.stats import find_outliers
from sofia_redux.toolkit.interpolate.interpolate import tabinv

__all__ = ['parse_width_arg', 'get_fitter', 'dofit', 'box_convolve',
           'get_search_model', 'initial_search', 'get_background_fit',
           'get_final_model', 'fitpeaks1d', 'medabs_baseline',
           'guess_xy_mad']

try:
    from astropy.modeling.utils import _ConstraintsDict as astropy_dict
except ImportError:
    astropy_dict = dict


def robust_masking(data, threshold=5):
    """Default outlier identification for robust fitting"""
    if not isinstance(data, np.ma.MaskedArray):
        data = np.ma.array(data)
    mask = ~find_outliers(data, threshold)
    data.mask = mask
    return data


def medabs_baseline(_, y):
    """
    Default data preparation for `initial_search` and baseline function.

    Note that the baseline function is also used to prepare an initial
    estimate of the background parameters (if a background model is
    supplied as well)

    Parameters
    ----------
    _ : unused
    y : array-like
        Dependent data values

    Returns
    -------
    baseline_removed_y, baseline : numpy.ndarray, numpy.ndarray
        The de-baselined `y` data and baseline, both of type numpy.float64.
    """
    y = np.array(y)
    baseline = np.full(y.shape, float(np.median(y)))
    return np.abs(y - baseline), baseline


def guess_xy_mad(x, y):
    """
    Default peak guess function for `initial_search`.

    This function should take in x (independent) and y (dependent) values
    and return an (x, y) tuple estimate of the most prominent peak location.

    Parameters
    ----------
    x : array_like
        Independent Values
    y : array_like
        Dependent Values

    Returns
    -------
    x_peak, y_peak : 2-tuple
        The x and y location of the most prominent peak.
    """
    xind = np.argmax(np.abs(y - np.median(y)))
    return x[xind], y[xind]


def get_model_name(model, base=False):

    if isinstance(model, str):
        return model

    if base:
        if isclass(model):
            thing = model()
        else:
            thing = model

        if get_n_submodels(thing) > 1:
            thing = model[0]
    else:
        thing = model

    if isinstance(thing, Fittable1DModel):
        name = thing.__class__.name
    elif hasattr(thing, 'name'):
        name = thing.name
        if name is None:
            name = thing.__class__.name
    else:
        name = None
        log.warning("unable to determine name of %s" % thing)
    return name


def update_model(model, kwargs):
    for k, v in kwargs.items():
        if hasattr(model, k):
            val = getattr(model, k)
            if (isinstance(val, astropy_dict) or isinstance(val, dict)) \
                    and isinstance(v, dict):
                recursive_dict_update(val, v)
            elif isinstance(val, Parameter) and isinstance(v, (int, float)):
                val.value = v
                setattr(model, k, val)
            else:
                val = v
                setattr(model, k, val)


def get_amplitude_parname(model_name, base=True):
    """The standard amplitude parameter of most common models"""
    name = get_model_name(model_name, base=base)
    return 'amplitude_L' if name == 'Voigt1D' else 'amplitude'


def get_x_parname(model_name, base=True):
    """The standard x-center parameter of most common models"""
    name = get_model_name(model_name, base=base)
    if name == 'Gaussian1D':
        x_name = 'mean'
    elif name == 'Sine1D':
        x_name = 'phase'
    else:
        x_name = 'x_0'

    return x_name


def get_width_parname(model_name, base=True):
    """The standard width parameter of most common models"""
    name = get_model_name(model_name, base=base)
    names = {'Moffat1D': 'gamma',
             'Box1D': 'width',
             'Gaussian1D': 'stddev',
             'Lorentz1D': 'fwhm',
             'MexicanHat1D': 'sigma',
             'Sersic1D': 'r_eff',
             'Trapezoid1D': 'width',
             'Voigt1D': 'fwhm_G',
             'Sine1D': 'frequency'}
    return names.get(name)


def get_n_submodels(model):
    """Determine the number of submodels in a model

    Required due to significant change to astropy.modeling in
    Astropy v4.0.
    """
    if not hasattr(model, 'n_submodels'):
        return 1
    if callable(model.n_submodels):
        return model.n_submodels()
    result = model.n_submodels
    if isinstance(result, int):
        return result
    else:
        return 1


def parse_width_arg(model_class, width_arg):
    """
    Simple convenience lookup to get width parameter name for various models.

    Parameters
    ----------
    model_class : astropy.modeling.Fittable1DModel
    width_arg : None or float or int or ((str or None), (float or int))

    Returns
    -------
    None or (int or float) or (str, (int or float))
        If None:
            Do not apply max_region
        If (int or float)
            Apply fixed max_region
        If (str, float) or (str, int)
           Apply max_region as max_region[1] multiple of model
           parameter max_region[0].

    Raises
    ------
    ValueError, TypeError
    """
    name = get_model_name(model_class)
    if name is None:
        raise TypeError("model_class is not %s" % Fittable1DModel)

    if width_arg is None:
        return
    elif isinstance(width_arg, (int, float)):
        return width_arg
    elif not hasattr(width_arg, '__len__') or len(width_arg) != 2:
        raise ValueError("invalid max_region format")

    width_param, width = width_arg
    if not isinstance(width, (int, float)):
        raise ValueError("second element of max_region is not %s" % float)
    elif isinstance(width_param, str):
        if width_param not in model_class.param_names:
            raise ValueError("%s is not a parameter of %s" %
                             (width_param, model_class))
        else:
            return width_param, width
    elif width_param is not None:
        raise ValueError("first element of max_region must be str or None")

    parname = get_width_parname(model_class)
    if parname is None:
        raise ValueError(
            'Cannot autodetect width parameter name for "%s" model' % name)

    return parname, width


def get_min_width(model_class, min_width, x=None):
    if hasattr(min_width, '__len__') and len(min_width) == 2:
        if min_width[1] is None:
            if x is None:
                raise ValueError(
                    "Require x coordinates to determine dx")
            dx = np.unique(x)
            dx = float(np.median(np.abs(dx[1:] - dx[:-1])))
            min_width = min_width[0], dx / 2

    if isinstance(min_width, (float, int)):
        min_width = None, min_width

    return parse_width_arg(model_class, min_width)


def get_fitter(fitter_class, robust=None, outlier_func=robust_masking,
               outlier_iter=None, **kwargs):
    """
    Creates the object fitting a model to data

    Parameters
    ----------
    fitter_class : class of fitting object
        Typically a "solver" from scipy.optimize or astropy.modeling
        that when instatiated will create a solver such that:

            fitted_model = solver(model, x, y)
    robust : dict or bool, optional
        If supplied, sets the outlier rejection threshold during fitting.
        The outlier function itself may be specified via `outlier_func`.
    outlier_func : function, optional
        A function of the form data_out = outlier_func(data_in).  You
        may do anything at all to the data such as clipping, but perhaps
        the simplest option is to convert the data to a np.ma.MaskedArray
        and set the mask (True=bad) while returning the original data
        untouched.
    outlier_iter : int, optional
        The maximum number of iterations if rejecting outliers.  Defaults
        to 3.
    kwargs : dict, optional
        Additional keywords to pass into the solver during initialization.

    Returns
    -------
    instance of fitting object or function
    """
    fitter = fitter_class(**kwargs)
    if robust is None:
        return fitter

    if isinstance(robust, dict):
        opts = robust.copy()
    else:
        opts = {}

    if outlier_iter is not None:
        opts.update({'niter': outlier_iter})
    return fitting.FittingWithOutlierRemoval(fitter, outlier_func, **opts)


def dofit(fitter, model, x, y, **kwargs):
    """
    A simple wrapper to fit model to the data

    This is a convenience function that throws out the masking
    array returned by `astropy.fitting.FittingWithOutlierRemoval`
    and just returns the actual fit.

    Parameters
    ----------
    fitter : fitting object
    model : astropy.modeling.Fittable1DModel
        The model that the solver will use to fit the data.
    x : array_like of float
        (N,) array of independent data to fit
    y : array_like of float
        (N,) array of dependent data to fit

    Returns
    -------
    astropy.modeling.Fittable1DModel
        A new model containing the fitted parameters.
    """
    try:
        result = fitter(model, x, y, **kwargs)
    except Exception as err:
        log.error("fitting failed: %s" % err)
        result = model.copy()
        result.parameters = np.nan

    if isinstance(fitter, fitting.FittingWithOutlierRemoval):
        result = result[0]
        fit_info = getattr(fitter.fitter, 'fit_info', {})
    else:
        fit_info = getattr(fitter, 'fit_info', {})
    result.solver = fitter

    result.fit_info = fit_info
    result.fit_info['x'] = x.copy()
    result.fit_info['y'] = y.copy()

    return result


def box_convolve(model, box_class, box_width=None, box_x=None,
                 model_x=None, box_var=None, box_params=None,
                 **kwargs):
    """
    Convolve a model with a box (or another model)

    Standard astropy models are easy to convolve (multiply).  Unfortunately,
    during fitting we want the result of this convolution to be treated
    as a single model, not have the box fitted as well as the original
    model.  `box_convolve` anchors the center of the `box_class`
    to the center of the `model`.  All other box parameters are held
    constant.  As a convenience, `box_width` is supplied so that the
    user can define a fixed width to the box (which can also be defined
    using `box_params`).  Generally, all astropy models have a fixed
    amplitude of 1, so in simple cases only the width should be
    specified.  However, if using a more complex model then additional
    parameters should be specified via `box_params`.

    In addition, the user may also easily select a single parameter of
    the `box_class` to be equal to a single parameter of the `model`
    multiplied by some scaling factor.  `fitpeaks1d` uses this by
    default to convolve a Gaussian function with a step function that
    will always be set to a multiple factor of the FWHM.  This is
    incredibly useful when trying to get a good fit on a peak by only
    using data in range of the peaks significance; not the entire
    set of data which may contain artifacts that will interfere with
    the fit.  To do this set `box_width` to (parameter, factor) of
    the model.  For example ('stddev', 6) would center a box function
    with a width of 6 times the FWHM of a Gaussian over the original
    Gaussian.

    Note though, that box_width is simply a convenience argument.  If
    you know what you are doing then you can tie or fix multiple
    parameters between the two models.  Look at the code in `box_convolve`
    for an example or how one should use the 'tied', 'bounds', and
    'fixed' keywords in the astropy.modeling documentation.  These
    may be suppled to `box_convolve` via kwargs.  kwargs is applied
    to the model at the very end of the algorithm, so anything here
    will override any previous logic.

    Parameters
    ----------
    model : astropy.modeling.Fittable1DModel (class or instance)
        model on which to apply box function
    box_class : astropy.modeling.Fittable1DModel (class)
        The box or model to convolve `model` with.
    box_width : (float or int) or (2-tuple of (str, float)), optional
        If int or float, specifies the fixed width of the box.
        Using the 2-tuple form scales the width of the box to be
        equal to box_width[1] * (the parameter box_width[0]) in
        `model`.  If set to None, the default values of the box
        model or those set in 'box_params` will define a constant
        box width.
    box_x : str, optional
        Determines the `box_class` parameter to always be set equal to
        the `model` `model_x` parameter.  If omitted, an attempt will be
        made to determine what the x-centering parameter is for the box.
    model_x : str, optional
        Determines which `model` parameter that the `box_x` parameter
        of the `box_class` will always be set equal to.  If omitted, an
        attempt will be made to determine what the x-centering parameter
        is for the `model`.
    box_var : str, optional
        Determines what the "width" parameter of the box is in relation
        to the `box_width` parameter and its uses.  If omitted, an
        attempt will be made to determine the "width" parameter name
        of the box model.
    box_params : dict, optional
        Used to set initial `box_class` parameters.  All parameters set
        here will be fixed unless overriden by other keyword options.
        Please see `astropy.modeling.models` for a list of keywords
        that can be supplied.
    kwargs : dict, optional
        keyword values to be applied that will override everything else.
        keys should be the name of the output model attributes and values
        should be the values you wish to set.  Dictionaries are applied
        recursively.

    Returns
    -------
    result : instance of astropy.modeling.Fittable1DModel
        The resulting `model` * box_model
    """
    if isclass(box_class) and not issubclass(box_class, Fittable1DModel):
        raise TypeError(
            "box_class must be %s" % Fittable1DModel)  # pragma: no cover

    is_compound = get_n_submodels(model) > 1

    if box_x is None:
        box_x = get_x_parname(box_class)
    if model_x is None:
        model_x = get_x_parname(model)
        if is_compound:
            model_x += '_0'
    if box_x not in box_class.param_names:
        raise ValueError('box_x parameter name "%s" not available for %s'
                         % (box_x, get_model_name(box_class)))
    if model_x not in model.param_names:
        raise ValueError('model_x parameter name "%s" not available for %s'
                         % (model_x, get_model_name(model)))

    if box_var is None and box_width is not None:
        box_var = get_width_parname(box_class)
        if box_var not in box_class.param_names:
            raise ValueError(
                'box_var parameter name "%s" not available for %s'
                % (box_var, get_model_name(box_class)))  # pragma: no cover

    if hasattr(box_width, '__len__') and len(box_width) == 2:
        scale_par, scale = box_width
        if is_compound:
            scale_par += '_0'
        if scale_par not in model.param_names:
            raise ValueError('"%s" is not a parameter of %s' %
                             (scale_par, get_model_name(model)))
        if not isinstance(scale, (int, float)):
            raise ValueError("scaling parameter is not int or float")
    else:
        scale_par = None
        scale = box_width if isinstance(box_width, (float, int)) else None

    if box_params is None:
        box_params = {}
    if 'fixed' not in box_params:
        box_params['fixed'] = {}
    if 'tied' not in box_params:
        box_params['tied'] = {}

    # Fix every parameter in the box model, we'll change it later
    for box_par in box_class.param_names:
        if box_par not in box_params['fixed']:
            box_params['fixed'][box_par] = True

    n = get_n_submodels(model)

    if scale_par is None and scale is not None:
        box_params[box_var] = scale
        box_params['fixed'][box_var] = True
    elif scale_par is not None:
        box_params['fixed'][box_var] = False
        box_params['tied'][box_var] = lambda m: scale * getattr(
            m[:n], scale_par)

    # tie box_x to model_x
    box_params['fixed'][box_x] = False
    box_params['tied'][box_x] = lambda m: getattr(m[:n], model_x)

    # create the model
    name = '(%s)*%s' % (get_model_name(model), get_model_name(box_class))
    m = model() if isclass(model) else model
    boxed = (m * box_class(**box_params)).rename(name)

    try:
        # in case we were passed an initialized model
        tmp = boxed.parameters
        tmp[:model.parameters.size] = model.parameters.copy()
        boxed.parameters = tmp
    except AttributeError:
        pass

    # If not fitting, ensure the box is centered over model
    setattr(boxed[-1], box_x, getattr(boxed[:n], model_x))
    # And ensure scaling rules are followed (if there are any)
    if scale_par is not None:
        setattr(boxed[-1], box_var, boxed[-1].tied[box_var](boxed))

    # finally override with kwargs
    update_model(boxed, kwargs)

    return boxed


def get_search_model(peak_model, box_class=None, box_width=None,
                     min_width=None, xrange=None, **kwargs):
    """
    Create the `initial_search` peak fitting model

    This function is used to create the peak model that will be used
    during the initial peak search.  This is a fairly simple function
    which adds an optional filtering (box) model to the peak_model
    and instantiates it with the necessary options.

    Parameters
    ----------
    peak_model : astropy.modeling.Fittable1DModel
        Class for the peak model
    box_class : astropy.modeling.Fittable1DModel, optional
        Convolve the peak model with this model, generally to reduce
        the range over which the model is fit.
    box_width : (float or int) or (2-tuple of (str, float)), optional
        Used conjunction with `box_model`.  See `box_convolve` for
        full details and usage.
    min_width : 2-tuple of (str, (int or float))
        If supplied will set the minimum bounds of the width parameter
        min_width[0] to min_width[1].  Overrides kwargs.
    xrange : 2-tuple of float, optional
        (minumum, maximum) xrange over which the model is applicable.
        Set to avoid extrapolation.
    kwargs : dict, optional
        Extra keywords to pass into either the `peak_model` or
        `box_convolve`.

    Returns
    -------
    result : instance of astropy.modeling.Fittable1DModel
        The model to be used for an initial peak-search.
    """
    if box_width is not None and box_class is not None:
        search_model = box_convolve(
            peak_model, box_class, box_width=box_width, **kwargs)
        pmodel = search_model[:][0]
    else:
        search_model = peak_model(**kwargs)
        pmodel = search_model

    if min_width is not None:
        if isinstance(min_width, (int, float)):
            min_width = get_width_parname(peak_model), min_width
            if min_width[0] is None:
                raise ValueError(
                    "could not determine width parameter for %s model" %
                    get_model_name(peak_model))

        wbounds = pmodel.bounds[min_width[0]]
        wbounds = min_width[1], wbounds[1]
        pmodel.bounds[min_width[0]] = wbounds

    model_x = get_x_parname(peak_model)
    xbounds = pmodel.bounds.get(model_x)
    if xbounds == (None, None) and xrange is not None:
        pmodel.bounds[model_x] = xrange
    return search_model


def initial_search(fitter, model, x, y, npeaks=1,
                   xpeak_parname=None, ypeak_parname=None,
                   baseline_func=medabs_baseline,
                   guess_func=guess_xy_mad, guess=None, fitopts=None):
    """
    Perform an initial search for peaks in the data

    The procedure is:

    1. Take the original (x, y) data and optionally modify it
       to some form where it is easier to fit a more accurate peak.  In
       the default case, the median is subtracted from y before taking
       the absolute value.  At this stage we do not typically fit the
       background level so instead try to center most of the data
       around zero and then take the absolute value.  This is sensible
       for stable baselines with positive and negative peaks, but
       may not be so for other flavours of data.  Different baseline
       removal and data modification functions may be specified with
       the `baseline_func` argument.

    2. Start a loop of `npeaks` iterations.  On each interation:

        a. Identify the most prominent peak in the modified y data.
           This is done via the `guess_func` function.  In the default
           case this simply finds the (x, y) position of the maximum
           `y` value.  Optionally, the user may parse in specified
           `x` values using the `guess` argument.

        b. Fit the data to the search model using the x and y guesses
           obtained in the previous step.

        c. Store the parameters obtained from the fit, then subtract
           a fit of the peak from the data and begin the next iteration.
           If successful, the most prominent peak was removed and we
           can move on to fitting the next most prominent peak.

    3. After `npeaks` iterations take the `x` values of the fit and
       use interpolation to get an estimate of `y` on the original
       unmodified data set.  This y estimate is added to the
       other stored peak parameters which are then used as a more
       accurate starting point for a refined search later.

    Parameters
    ----------
    fitter : fitting object
        Typically a "solver" from scipy.optimize or astropy.modeling
        that performs the function::

            fitted_model = solver(model, x, y)

    model : astropy.modeling.Fittable1DModel
        The model to fit.  Typically from `get_search_model`.
    x : array_like of float
        (N,) array of independent data to fit
    y : array_like of float
        (N,) array of dependent data to fit
    xpeak_parname : str, optional
        Name of the parameter in the peak model governing the location
        of the peak along the x-axis.  If None, will attempt to autodetect.
    ypeak_parname : str, optional
        Name of the parameter in the peak model governing the location
        of the peak along the y-axis.  If None, will attempt to autodetect.
    npeaks : int, optional
        The number of peaks to find
    guess_func : function
        A function of the form::

            x_guess, y_guess = guess_func(x, y_modified)

        Here x_guess and y_guess are the x (position) and y (amplitude)
        estimates of the most prominent peak in the data.  Here we use
        the output from `baseline_func` as the dependent values which
        allows the user to do things such as smoothing, filtering, or
        whatever other ideas they may have.  The default function,
        `guess_xy_mad` simply finds the x and y coordinate of the
        maximum value in y_modified.
    guess : array_like of float, optional
        An array where each element gives a guess at an initial
        `x` position for the peak.  If there are less guesses than
        `npeaks`, `guess_func` will be used after the inital guesses
        have been taken.  If there are more guesses than `npeaks`, then
        only guess[:npeaks] will be used.
    baseline_func : function, optional
        A function of the form::

            y_modified, baseline = baseline_func(x, y)

        Here y_modified may be the original `y` data, or the `y` data
        modified such that the combination of `model` and `solver` is
        able to identify peaks.  In the default case, the `medabs_baseline`
        function is used which returns y_modified = abs(y - median(y))
        and baseline = y - median(y).
    fitopts : dict, optional
        Optional arguments to pass into the solver at runtime.

    Returns
    -------
    numpy.ndarray of numpy.float64
        (npeaks, n_parameters)  array of the initial guesses for the
        model parameters of each peak.  Note that this contains the
        `peak` only portion of parameters and not the filtering
        parameters of the optional `box_model` that may have been
        appended to the peak model at `get_search_model`.
    """
    if fitopts is None:
        fitopts = {}
    pmodel = model[:][0] if hasattr(model, 'submodel_names') else model
    pnames = np.array(pmodel.param_names)
    npar = pnames.size

    if isinstance(guess, (int, float)):
        guess = np.array([guess])

    if xpeak_parname is None:
        xpeak_parname = get_x_parname(pmodel)
    if xpeak_parname not in pmodel.param_names:
        raise ValueError("xpeak_parname %s not in %s" %
                         (xpeak_parname, get_model_name(pmodel)))
    if ypeak_parname is None:
        ypeak_parname = get_amplitude_parname(pmodel)
    if ypeak_parname not in pmodel.param_names:
        raise ValueError("ypeak_parname %s not in %s" %
                         (ypeak_parname, get_model_name(pmodel)))

    xpar_ind = np.where(pnames == xpeak_parname)[0][0]
    ypar_ind = np.where(pnames == ypeak_parname)[0][0]
    params = np.empty((npeaks, npar))
    ysearch, baseline = baseline_func(x, y)

    for i in range(npeaks):
        if guess is not None and i < len(guess):
            guess_ind = int(
                np.round(np.clip(tabinv(x, guess[i]), 0, x.size - 1)))
            yguess = ysearch[guess_ind]
            xguess = x[guess_ind]
        else:
            xguess, yguess = guess_func(x, ysearch)

        tmp = pmodel.parameters
        tmp[xpar_ind] = xguess
        tmp[ypar_ind] = yguess
        pmodel.parameters = tmp

        fit = dofit(fitter, model, x, ysearch, **fitopts)

        params[i] = fit.parameters[:npar].copy()
        pmodel.parameters = fit.parameters[:npar]
        ysearch -= pmodel(x)

    # refit amplitudes as real data relative to baseline
    ycen = np.interp(params[:, xpar_ind], x, y)
    bcen = np.interp(params[:, xpar_ind], x, baseline)
    params[:, ypar_ind] = ycen - bcen

    return params


def get_background_fit(fitter, peak_model, background_class, x, y,
                       pinit, fitopts=None, bg_args=None, **bg_kwargs):
    """
    Return a background model with initialized parameters

    Subtract peaks from `y` then fit background on residual

    Parameters
    ----------
    fitter : object
    peak_model : astropy.modeling.Fittable1DModel instance
    background_class : astropy.modeling.Fittable1DModel
    x : array_like of float
    y : array_like of float
    pinit : numpy.ndarray
        (npeaks, n_peak_parameters) array giving the peak parameters
        for `peak_model`.
    fitopts : dict, optional
        Optional keywordds to pass into the `solver`.
    bg_args : tuple, optional
        If the background class requires positional arguments, they should
        be supplied here.
    bg_kwargs : dict, optional
        Optional keywords to pass into the background_class to initialize
        the background model.

    Returns
    -------
    numpy.ndarray
        (n_background_parameters,) array of background parameters based
        on the fit of (x, y - peaks).
    """
    y = y.copy()

    if not isclass(background_class):
        bmodel = background_class.copy()
        update_model(bmodel, bg_kwargs)
    else:
        if bg_args is None:
            bmodel = background_class(**bg_kwargs)
        else:
            if hasattr(bg_args, '__len__'):
                bmodel = background_class(*bg_args, **bg_kwargs)
            else:
                bmodel = background_class(bg_args, **bg_kwargs)

    init_params = peak_model.parameters.copy()
    if fitopts is None:
        fitopts = {}
    for params in pinit:
        peak_model.parameters = params
        y -= peak_model(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        bfit = dofit(fitter, bmodel, x, y, **fitopts)
    peak_model.parameters = init_params
    return bfit.parameters


def get_final_model(peak_class, pinit, background_class=None,
                    bg_args=None, binit=None,
                    min_width=None, xbounds=None, **kwargs):
    """
    Refine the initial fit and return a set of models

    Parameters
    ----------
    peak_class : astropy.modeling.Fittable1DModel instance
    pinit : array_like of float
    background_class : astropy.modeling.Fittable1DModel, optional
    binit : array_like of float, optional
        (n_background_parameters,) array of initial background parameter
        estimates.
    min_width : 2-tuple of (str, float), optional
    xbounds : 2-tuple of float, optional
    kwargs : dict, optional

    Returns
    -------
    list of astropy.modeling.Fittable1DModel
    """
    # This supports old serial implementation.  May be useful one day
    # pinit = np.array(pinit)
    # npeaks = pinit.shape[0]
    # min_width = get_min_width(peak_class, min_width)
    #
    # model_class, build = None, kwargs.copy()
    # xpar_name = get_x_parname(peak_class)
    #
    # compound = npeaks > 1 or background_class is not None
    #
    # build['bounds'] = build.get('bounds', {})
    #
    # if compound:
    #     for i, params in enumerate(pinit):
    #         if model_class is None:
    #             model_class = peak_class
    #         else:
    #             model_class += peak_class
    #         for j, parname in enumerate(peak_class.param_names):
    #             build['%s_%i' % (parname, i)] = params[j]
    #         if min_width is not None:
    #             key = '%s_%i' % (min_width[0], i)
    #             wlim = build['bounds'].get(key, (None, None))
    #             build['bounds'][key] = min_width[1], wlim[1]
    #         if xbounds is not None:
    #             build['bounds']['%s_%i' % (xpar_name, i)] = xbounds
    # else:
    #     model_class = peak_class
    #     for j, parname in enumerate(peak_class.param_names):
    #         build['%s' % parname] = pinit[0, j]
    #     if min_width is not None:
    #         wlim = build['bounds'].get(min_width[0], (None, None))
    #         build['bounds'][min_width[0]] = min_width[1], wlim[1]
    #     if xbounds is not None:
    #         build['bounds'][xpar_name] = xbounds
    #
    # if background_class is not None:
    #     model_class += background_class
    #     if binit is not None:
    #         binit = np.array(binit)
    #         for parname, parval in zip(background_class.param_names, binit):
    #             build['%s_%i' % (parname, npeaks)] = parval
    #
    # if compound:
    #     name = '%i_peak%s' % (npeaks, '' if npeaks == 1 else 's')
    #     if background_class is not None:
    #         name += '_with_background'
    #     model_class = model_class.rename(name)
    # model = model_class(**build)
    # if not compound:
    #     model.name = '1_peak'
    #
    # return model

    pinit = np.array(pinit)
    npeaks = pinit.shape[0]
    min_width = get_min_width(peak_class, min_width)

    build = {}
    xpar_name = get_x_parname(peak_class)

    compound = npeaks > 1  # or background_class is not None

    build['bounds'] = build.get('bounds', {})

    model = None
    if compound:
        for i, params in enumerate(pinit):
            if model is None:
                model = peak_class()
            else:
                model += peak_class()
            for j, parname in enumerate(peak_class.param_names):
                build['%s_%i' % (parname, i)] = params[j]
            if min_width is not None:
                key = '%s_%i' % (min_width[0], i)
                wlim = build['bounds'].get(key, (None, None))
                build['bounds'][key] = min_width[1], wlim[1]
            if xbounds is not None:
                build['bounds']['%s_%i' % (xpar_name, i)] = xbounds
    else:
        model = peak_class()
        for j, parname in enumerate(peak_class.param_names):
            build['%s' % parname] = pinit[0, j]
        if min_width is not None:
            wlim = build['bounds'].get(min_width[0], (None, None))
            build['bounds'][min_width[0]] = min_width[1], wlim[1]
        if xbounds is not None:
            build['bounds'][xpar_name] = xbounds

    update_model(model, build)
    name = '%i_peak%s' % (npeaks, '' if npeaks == 1 else 's')

    if background_class is not None:
        if isclass(background_class):
            if bg_args is None:
                background = background_class()
            elif not hasattr(bg_args, '__len__'):
                background = background_class(bg_args)
            else:
                background = background_class(*bg_args)
        else:
            background = background_class.copy()

        if binit is not None:
            background.parameters = binit

        # if binit is None:
        #     background = background_class()
        # else:
        #     binit = np.array(binit)
        #     bg_kwargs = {}
        #     for parname, parval in zip(background_class.param_names, binit):
        #         bg_kwargs[parname] = parval
        #     background = background_class(**bg_kwargs)
        model += background
        name += '_with_background'

    model = model.rename(name)
    if len(kwargs) > 0:
        update_model(model, kwargs)

    return model


def fitpeaks1d(x, y, npeaks=1, xrange=None, min_width=(None, None),
               peak_class=models.Gaussian1D,
               box_class=models.Box1D, box_width=(None, 6),
               background_class=models.Const1D, binit=None,
               bg_args=None, bg_kwargs=None,
               fitting_class=fitting.LevMarLSQFitter, maxiter=None,
               outlier_func=robust_masking, robust=None, outlier_iter=3,
               baseline_func=medabs_baseline,
               guess=None, guess_func=guess_xy_mad, optional_func=None,
               fitter_kwargs=None, fitopts=None, search_kwargs=None,
               **kwargs):
    """
    Fit peaks (and optionally background) to a 1D set of data.

    The user may fit for 1 or multiple peaks in the input data using
    using an array of different peak models and also fit for a model
    of the background.  There are a large number of standard models
    available for both, or the user may define their own models if
    necessary.  By default, we fit for Gaussian peaks and a
    constant background.

    The intention of `fitpeaks1d` is to provide a highly customizable
    peak fitting utility for one dimensional data.  The basic procedure
    is split into two phases:  the search phase and the refinement phase.

    During the search phase an attempt is made to get a good first estimate
    of where the most prominent peaks lie in the data along with
    estimates of their properties.  Initial positions can be
    supplied by the user via `guess`, but may also be derived via
    the `guess_func` algorithm which is also definable by the user.
    By default, candidate positions are determined as
    max(abs(y - median(y))).  The basic procedure is as follows:

    1. Remove a baseline from the y data via the `baseline_func`
       function.  At this early stage, we are only really interested
       in peak characteristics or even just where they kind-of are.
       The default baseline removal function is to subtract the
       median although it is completely reasonable to use many other
       methods depending on your data.  This baseline_func may also
       transform the initial data set to something a little more
       suitable for finding peaks.  Users may wish to smooth or
       filter the data, or perform some other kind of transform.
       By default the data is set to y = abs(y - median(y)) as
       we do not know if we're searching for positive or negative
       peaks.

    2. Guess where peaks may lie along the x-axis.  These estimates
       can be provided directly by the user, or derived via the
       `guess_func` algorithm.  This algorithm should take the
       (x, y) data set and return a single (x0, y0) coordinate
       giving the x position and amplitude of the most prominent
       peak candidate in the data.  In the default case this is
       simply y0 = max(y) and x0 = where(y0).

    3. Fit a peak using (x0, y0) as an initial condition.  Store
       the peak properties and then subtract a model of the peak
       from the data and go back to step 2.  Only break this loop
       once the cycle has completed `npeaks` times.  This will
       hopefully yield the `npeaks` most prominent peaks in the
       data.  After fitting we should have a more exact (x1, y1)
       position for each peak.

    4. Take the x1 position of each peak derived in step 3 and
       interpolate it back onto the original data set before we
       changed it in step 1.  Subtract the derived baseline from
       this value and we should then have a good estimate of the
       (x, y) position for each peak in relation to the real
       dataset.

    5. If the user wishes to fit a background to the data, we
       get an estimate of the background fit parameters by
       subtracting models of each peak from the original dataset
       and then fitting the desired background model to the
       residual.

    During the refinement phase, a final model is created
    from all initial peak fits and an
    optional background.  If a background is requested, an initial
    estimate of its parameters is derived by fitting the model
    on y - fit(peak_estimates).

    Parameters
    ----------
    x: array_like of float
        (N,) array of independent values.
    y : array_like of float
        (N,) array of independent values.
    npeaks : int, optional
        The number of peaks to find.  Set to None to find a single
        (most prominent) peak.  Peaks are returned in the order
        most prominent -> least prominent.
    xrange : 2-tuple of (int or float), optional
        The (minimum, maximum) range of values in x where a fitted
        peak will be valid.  Set to None to use the full range of
        valid (finite) data.
    peak_class : astropy.modeling.Fittable1DModel, optional
        Class of peak model.  The default is
        `astropy.modeling.models.Gaussian1D`.
    box_width : float or int or (2-tuple of (str, float)), optional
        If int or float, specifies the width of `box_model`. If specifed
        as a 2-tuple of (str, float), the first specifies a
        parameter of the model that will be taken and multiplied by
        the second element of the 2-tuple (float) to define the width
        of the box.  For example ('stddev', 3) for the Gaussian1D model
        would limit the range of the model to 3 * FWHM.  This is highly
        recommended when fitting continuous or wide functions.  Note
        that this will not be applied when the refined model is
        fitted in parallel, since that kind of negates the whole
        point of fitting in parallel in the first place.
    min_width : 2-tuple of (str or None, float or None), optional
        Set a lower limit on the width of each peak.  Set to None for
        no lower limit.  min_width[0] is the name of the parameter in
        the peak model governing width (str).  Set this to None to
        autodetect the parameter name if using a standard model.
        min_width[1] is the value to set.  If set to None, half of the
        median separation of sorted unique `x` samples will be used.
    box_class :  astropy.modeling.Fittable1DModel, optional
        Class for filtering the peak model.  Will be applied as
        peak_model * box_model.  This is only applied during the initial
        search stage.  The width of the box is defined by `max_region`.
        The default is astropy.modeling.models.Box1D.
    background_class : astropy.modeling.Fittable1DModel, optional
        Class for describing the background.  Will be applied as
        peak_model + background in the final refined fitting.  Note that
        it will only be fitted to the area of interest around each peak
        (`max_region`).  Disable background fitting by supplying `None`.
        The default is to fit a constant background using
        astropy.modeling.models.Const1D.  NOTE: you can supply a model
        instance rather than class in order to fit polynomial backgrounds
        as the number of polynomial parameters are dependent on the
        polynomial order.  For example, in order to fit a 3rd order
        polynomial, set background_class=Polynomial1D(3).
    binit : array_like of float, optional
        A user guess at background parameters
    bg_args : tuple, optional
        If the background class requires arguments, they should be supplied
        here as a tuple.
    bg_kwargs : dict, optional
        Optional keyword arguments to pass into the background model
        during initialization.
    fitting_class : class of fitting object
        Typically a "solver" from scipy.optimize or astropy.modeling
        that when instatiated will create a solver such that:

            fitted_model = solver(model, x, y)

    maxiter : int, optional
        Maximum number of iterations for the solver to attempt before
        quitting.  Set to None to use the default for each solver class.
    outlier_func : function, optional
        A function of the form data_out = outlier_func(data_in).  You
        may do anything at all to the data such as clipping, but perhaps
        the simplest option is to convert the data to a np.ma.MaskedArray
        and set the mask (True=bad) while returning the original data
        untouched.  The default is to use `robust_masking` which is a
        basic wrapper for `mc.find_outliers`.
    robust : dict or anything, optional
        Supplying anything other than `None` will turn on outlier rejection
        for the solver.  If `robust` is a dict, then it should contain
        optional keyword arguments that would be supplied to `outlier_func'.
        For example, the default `robust_masking` outlier rejection function
        takes the `threshold` keyword.  To turn on outlier rejection you
        could just set robust=True, but if you wanted to change threshold to
        10, then set robust={'threshold': 10}.  Setting robust=None turns
        off outlier rejection (default).
    outlier_iter : int, optional
        The maximum number of iterations for the solver when rejecting
        outlers.  Defaults to 3.
    guess : array_like of float, optional
        An array where each element gives a guess at an initial
        `x` position for the peak.  If there are less guesses than
        `npeaks`, `guess_func` will be used after the inital guesses
        have been taken.  If there are more guesses than `npeaks`, then
        only guess[:npeaks] will be used.
    baseline_func : function, optional
        A function of the form used during the initial search phase:

            ysearch, baseline = baseline_func(x, y)

        This function should remove a baseline from the data so that
        we mitigate effects from the background as much as possible
        while keeping the structure of the peaks.  In the default
        case, the baseline is taken as the median in y.  This baseline
        should be subtracted from y and then the user may transform
        y as they see fit to allow `guess_func` and the `solver` to
        get the most reliable guess possible at each peaks model
        parameters.  The default `baseline_func` returns ysearch as
        ysearch = abs(y - median(y)).  Note that the most important
        parameter to guess at during this stage is the x center
        position of the peak.  Amplitude is not important, but we
        do want to get fairly accurate representations of the
        rest of the fit parameters such as width or FWHM.
    guess_func : function
        A function of the form:

            (x0, y0) = guess_func(x, y_modified)

        (x0, y0) is the peak (x_center, y_amplitude) of
        the most prominent peak in the data after `baseline_func` has been
        applied.  These results are used to initialize peak parameters
        during the search phase.  The default function returns the
        (x, y) coordinate of the datum that satisfies
        y0 = max(abs(y - median(y))).
    optional_func : function, optional
        An optional function applied to the data prior to the final
        fit.  If fitting a background, then no attempt should be made to
        remove the background unless it is being accounted for in
        some way.  If you are not fitting a background then feel free
        to go to town on removing said background.  Preserve whatever
        properties of the peak you wish to have appear in the output
        results.  Good choices may involve filtering the data.  The
        default is to do nothing and fit the final model on the original
        data.
    fitter_kwargs : dict, optional
        Additional keywords to pass into the solver during initialization.
    fitopts : dict, optional
        Optional keywords to pass into solver at runtime.
    search_kwargs : dict, optional
        Additional keyword arguments to supply during model instantiation
        during the initial search stage (peak * optional(box)).
        Note that the model parameter names are suffixed with "_0" and
        box parameter names are suffixed with "_1", but only if a box
        is being applied.  So for example, to
        say that you want parameter "p" of the model to remain fixed:

            kwargs = {"fixed": {"p_0": True}}

        As another example, to set parameter "a" of the box equal to
        parameter "b" of the model:

            kwargs = {"tied": {"b_1": lamba x: getattr(x, "a_0")}}

        There are lots of other things you could do here, and I've tried
        to include as many hooks as possible.  You don't even need to
        treat this thing as a box function.  It could be any kind of
        filtering function etc. and you could use kwargs to supply it
        with the necessary parameters or behaviours as you wish.
    kwargs : dict, optional
        Additional keyword arguments to supply during model instantation
        during the final model refinement stage.  Exactly the same type
        of thing as `search_kwargs`.  However, the following suffix
        rules apply for parallel fitting (default):

            1. If fitting only a single peak and no background, do not
               suffix any parameters.  "amplitude" remains "amplitude".
            2. If fitting multiple peaks, begin suffixing at _0 for the
               first peak upto _(npeaks-1) for the last.  i.e. the
               amplitude of the 5th peak should be referred to as
               "amplitude_4".
            3. If fitting a single peak with a background then peak
               parameters should be suffixed with "_0" and background
               parameters with "_1".
            4. If fitting multiple peaks with a single background then
               peak parameters should be suffixed according to step 2,
               while background parameters should be suffixed with
               _(npeaks).  i.e. if there were 5 peaks, the background
               "intercept" for a linear background fit should be
               referred to as "intercept_5".

        and for serial fitting:

            1. If fitting a single peak with no background, do no
               suffix any parameters.  "amplitude" remains "amplitude".
            2. If fitting one or more peaks with a background then
               there is no distinction available between peaks.  Each
               peak parameter rule must be applied to all peaks.  For
               example, setting {'bounds': {'amplitude_0': (0, 5)} will
               limit the amplitude of all peaks to within the range
               0->5.  The background parameters are suffixed with "_1".

    Returns
    -------
    model : astropy.modeling.Fittable1DModel instance
        The final fitted model.  If more than one peak or a background was
        fitted for, then the returned model will be a compound model.
        All peaks are located in model[:npeaks].  If a background was
        included it can be found at model[npeaks].  If a single peak and
        no background was fitted, then an instance of the peak_class will
        be returned containing the fit parameters.  No indexing will be
        possible.

    Examples
    --------
    Create compound peaks on a sloped background
    >>> import numpy as np
    >>> from astropy.modeling import models
    >>> from sofia_redux.toolkit.fitting.fitpeaks1d import fitpeaks1d
    >>> x = np.linspace(0, 10, 1001)
    >>> model1 = models.Gaussian1D(amplitude=3, mean=5, stddev=0.5)
    >>> model2 = models.Gaussian1D(amplitude=4, mean=4, stddev=0.2)
    >>> y = model1(x) + model2(x) + 0.05 * x + 100
    >>> rand = np.random.RandomState(42)
    >>> y += rand.normal(0, 0.01, y.size)
    >>> fit = fitpeaks1d(x, y, background_class=models.Linear1D,
    ...                  npeaks=2, min_width=0.1)
    >>> rmse = np.sqrt(np.mean(fit.fit_info['fvec'] ** 2))
    >>> assert np.allclose(fit[0].parameters, [3, 5, 0.5], atol=0.02)  # peak 1
    >>> assert np.allclose(fit[1].parameters, [4, 4, 0.2], atol=0.02)  # peak 2
    >>> assert np.allclose(fit[2].parameters, [0.1, 100], atol=0.2)  # bg
    >>> assert np.isclose(rmse, 0.01, rtol=1)  # residuals
    """
    if not isclass(peak_class) or not issubclass(
            peak_class, Fittable1DModel):
        raise TypeError("peak model is not %s" % Fittable1DModel)
    x, y = np.array(x), np.array(y)
    shape = x.shape
    if y.shape != shape:
        raise ValueError("x and y array shape mismatch")

    if fitter_kwargs is None:
        fitter_kwargs = {}
    if search_kwargs is None:
        search_kwargs = {}
    if fitopts is None:
        fitopts = {}
        if fitting_class in [fitting.LevMarLSQFitter,
                             fitting.SLSQPLSQFitter,
                             fitting.SimplexLSQFitter]:
            fitopts['maxiter'] = 1000
    if bg_kwargs is None:
        bg_kwargs = {}
    if maxiter is not None and 'maxiter':
        fitopts['maxiter'] = maxiter

    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        raise ValueError("no finite data")
    x, y = x[mask], y[mask]
    if np.unique(x).size < 5:
        raise ValueError("not enough valid unique points")
    if xrange is None:
        xrange = x.min(), x.max()
    elif not hasattr(xrange, '__len__') or len(xrange) != 2:
        raise ValueError("xrange must be of the form (xmin, xmax)")

    box_width = parse_width_arg(peak_class, box_width)
    min_width = get_min_width(peak_class, min_width, x)

    fitter = get_fitter(fitting_class, robust=robust,
                        outlier_func=outlier_func,
                        outlier_iter=outlier_iter,
                        **fitter_kwargs)

    search_model = get_search_model(peak_class,
                                    box_class=box_class,
                                    box_width=box_width,
                                    min_width=min_width,
                                    xrange=xrange,
                                    **search_kwargs)

    pinit = initial_search(fitter, search_model, x, y, npeaks=npeaks,
                           guess=guess, guess_func=guess_func,
                           baseline_func=baseline_func,
                           fitopts=fitopts)

    y = y.copy() if optional_func is None else optional_func(x, y)
    if background_class is not None and binit is None:
        if get_n_submodels(search_model) > 1:
            search_peak = search_model[0]
        else:
            search_peak = search_model

        binit = get_background_fit(fitter, search_peak, background_class,
                                   x, y, pinit, bg_args=bg_args, **bg_kwargs)

    model = get_final_model(peak_class, pinit,
                            background_class=background_class,
                            bg_args=bg_args, binit=binit,
                            min_width=min_width, xbounds=xrange, **kwargs)

    fit = dofit(fitter, model, x, y, **fitopts)

    return fit
