# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
from scipy.optimize import curve_fit

from sofia_redux.toolkit.fitting.polynomial import polyfitnd
from sofia_redux.toolkit.fitting.fitpeaks1d import \
    fitpeaks1d, get_x_parname, get_n_submodels

__all__ = ['tracespec']


def tracespec(rectimg, positions, orders=None, fwhm=1.0,
              step=3, sumap=3, winthresh=5, fitorder=2, fitthresh=3,
              polyfit_kwargs=None, info=None, fast=False, **kwargs):
    """
    Trace spectral continua in a spatially/spectrally rectified image.

    Determines trace coefficients for the center of the aperture via fits
    to the continuum.  Within an order, `sumap` columns are added together
    to increase the total signal.  A Gaussian is then fitted around the
    guess position.  If within `winthresh` pixels of a guess, the position is
    stored.  The resulting positions are then fitted with a polynomial of
    degree `fitorder`.

    The `rectimg` value should come from sofia_redux.spectroscopy.rectify.

    Parameters
    ----------
    rectimg : dict
        Rectified image data, as returned by
        `sofia_redux.spectroscopy.rectify` with integer keys.
        Each order value is a dictionary with keys:

            ``"image"``
                Rectified image array.
                (ns, nw) array, required.
            ``"wave"``
                Wave coordinates along image axis=1.
                (nw,) array, required.
            ``"spatial"``
                Spatial coordinates along image axis=0.
                (ns,) array, required.

    positions : dict
        List of aperture positions by order, given as arcsec up the slit.
        Keys are integers, values are lists of floats.
    orders : array_like of int, optional
        Order numbers to process. If not provided, all orders in `positions`
        will be processed.
    step : int, optional
        Pixel step size in the dispersion direction used to determine
        the trace.
    sumap : int, optional
        Number of columns to add together.  Must be odd and less than
        2 * `step`.
    winthresh : float, optional
        The threshold over which an identified peak is ignored.  If the
        difference between the guess position is larger than `winthresh`,
        the fitted position is ignored.
    fitorder : int, optional
        Polynomial fit degree used to determine the trace coefficients.
    fitthresh : float, optional
        Sigma threshold used to identify outliers.
    polyfit_kwargs : dict, optional
        Optional keyword arguments for the polynomial fit to the
        trace (spatial location of peak vs wavecal).  These will
        be passed to `sofia_redux.toolkit.fitting.polyfit`.
    info : dict, optional
        If supplied will be updated as follows:
            order : int
                x : numpy.ndarray
                    (n_apertures, n_steps) array of fitted trace pixel
                    coordinates in x.
                y : numpy.ndarray
                    (n_apertures, n_steps) array of fitted trace pixel
                    coordinates in y.
                mask : numpy.ndarray
                    (n_apertures, n_steps) array of bool where True
                    indicates that the peak fit on that `step` of columns
                    was included in the final calculation of trace
                    coefficients.
                trace_model : numpy.ndarray
                    (n_apertures) array containing instances
                    of `astropy.modeling.polynomial.Polynomial1D` models
                    if aperture traces.
                fit : numpy.ndarray
                    (n_apertures, n_steps, n_parameters) array of
                    peak fit coefficients calculated by `mc.fitpeaks1d`
                    at each step.
                spatial : numpy.ndarray
                    (n_apertures, n_steps) array containing the
                    spatial coordinates at each step.
                wave : numpy.ndarray
                    (n_apertures, n_steps) array containing the
                    wave coordinates at each step.
                peak_model : `astropy.modeling.Fittable1DModel`
                    model used to fit peaks via `fitpeaks1d`
    fast : bool, optional
        If set, use `scipy.optimize.curve_fit` to calculate spatial peak
        coefficients instead of the more flexible optimization in
        `sofia_redux.toolkit.fitting.fitpeaks1d`. If NaNs may be present,
        `fast` must be False.
    kwargs : dict, optional
        Keyword arguments for `sofia_redux.toolkit.fitting.fitpeaks1d`.
        By default this will fit a single 1-D Gaussian peak and a constant
        background offset to each (median combined +/- sumap/2 @ step
        interval) column.

    Returns
    -------
    trace_coefficients : dict
        Array of polynomial fit coefficients by aperture and order.
        Coefficients are of the form:

            spatial_position = c[0] + c[1].w + c[2].w^2 + ... + c[n].w^n

        where n ranges from 0 to fit_order and w is the wave value and
        c are the trace coefficients for a given order and aperture.

        Keys and values are:
            order : int
                (n_apertures, n_coeff) array
    """
    if orders is None:
        orders = np.unique(list(positions.keys())).astype(int)
    else:
        orders = np.unique(orders).astype(int)
    polyfit_kwargs = {} if polyfit_kwargs is None else polyfit_kwargs
    do_info = isinstance(info, dict)
    dw = int(sumap / 2)

    # Gaussian width from FHWM
    sigma = gaussian_fwhm_to_sigma * fwhm

    result = {}
    for order in orders:
        rectified = rectimg.get(order)
        if rectified is None:
            log.warning(f"Order {order} is missing from rectimg.")
            continue

        image = rectified.get('image')
        if image is None:
            log.warning(f"Order {order} is missing image key.")
            continue

        wave = rectified.get('wave')
        if wave is None:
            log.warning(f"Order {order} is missing wave key.")
            continue

        spatial = rectified.get('spatial')
        if spatial is None:
            log.warning(f"Order {order} is missing spatial key.")
            continue

        n_cols = wave.size
        n_rows = spatial.size
        if image.shape != (n_rows, n_cols):
            log.warning(f"Invalid image dimensions for order {order}.")
            log.warning(f"Should be ({n_rows},{n_cols}) but is {image.shape}.")
            continue

        ap_pos = positions.get(order)
        if ap_pos is None:
            log.warning(f"Order {order} is missing from positions.")
            continue
        if not hasattr(ap_pos, '__len__'):
            ap_pos = [ap_pos]
        n_apertures = len(ap_pos)

        xpar, fitter, columns, model_fit = [None] * 4
        trace_coeffs = np.full((n_apertures, fitorder + 1), np.nan)
        y = np.arange(n_rows).astype(float)

        # start 'step' columns in, rather than at zero
        n_step = int((n_cols - step) / step)
        columns = np.arange(n_step) * step + step
        peaks_arc = np.full((n_apertures, n_step), np.nan)
        waves = np.full(n_step, np.nan)

        order_set = False
        if do_info:
            info[order] = {}
            info[order]['x'] = np.repeat([columns], n_apertures, axis=0)
            info[order]['y'] = np.full((n_apertures, n_step), np.nan)
            info[order]['mask'] = np.full((n_apertures, n_step), False)
            info[order]['trace_model'] = [None] * n_apertures

        for stepi, column in enumerate(columns):
            cr = np.clip([column - dw, column + dw], 0, n_cols - 1)
            waves[stepi] = wave[column]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                slit_image = np.nanmedian(image[:, cr[0]: cr[1] + 1], axis=1)

            for api, position in enumerate(ap_pos):

                kwargs['guess'] = position
                kwargs['npeaks'] = 1
                kwargs['stddev'] = sigma

                nn = ~np.isnan(slit_image)
                try:
                    model = fitpeaks1d(spatial[nn], slit_image[nn], **kwargs)
                except ValueError:
                    log.warning('Invalid data in initial fit.')
                    continue

                if not order_set:
                    if do_info:
                        info[order]['fit'] = np.full(
                            (n_apertures, n_step, model.parameters.size),
                            np.nan)
                        info['peak_model'] = model
                        info[order]['spatial'] = np.full(
                            (n_apertures, n_step), np.nan)
                    if fast:
                        def model_fit(x, *params):
                            model.parameters = params
                            return model(x)
                    suffix = '_0' if get_n_submodels(model) > 1 else ''
                    xpar = get_x_parname(model) + suffix
                    fitter = model.solver
                    order_set = True
                else:
                    if fast:
                        coeffs, _ = curve_fit(model_fit, spatial[nn],
                                              slit_image[nn],
                                              p0=model.parameters)
                        model.parameters = coeffs
                    else:
                        model = fitter(model, spatial[nn], slit_image[nn])

                if do_info:
                    info[order]['fit'][api, stepi] = model.parameters.copy()

                xval = getattr(model, xpar).value

                # check fit for failure if possible
                try:
                    failure = (model.fit_info['ierr'] not in [1, 2, 3, 4])
                except (AttributeError, KeyError):  # pragma: no cover
                    failure = False

                if (np.abs(xval - position) <= winthresh) and not failure:
                    peaks_arc[api, stepi] = xval
                    if do_info:
                        info[order]['y'][api, stepi] = np.interp(
                            xval, spatial, y)
                        info[order]['spatial'][api, stepi] = xval

        if do_info:
            info[order]['wave'] = np.repeat([waves], n_apertures, axis=0)

        for api, arc in enumerate(peaks_arc):
            trace_model = polyfitnd(waves, arc, fitorder, robust=fitthresh,
                                    model=True, **polyfit_kwargs)
            if trace_model.success:
                trace_coeffs[api] = trace_model.coefficients
                if do_info:
                    info[order]['trace_model'][api] = trace_model
                    info[order]['mask'][api] = trace_model.mask.copy()
        result[order] = trace_coeffs

    return result
