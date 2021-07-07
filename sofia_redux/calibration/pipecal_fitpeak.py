# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fit a 2D function to an image."""

from astropy import log
import numpy as np
from scipy.optimize import curve_fit

from sofia_redux.calibration.pipecal_error import PipeCalError

__all__ = ['elliptical_gaussian', 'elliptical_lorentzian',
           'elliptical_moffat', 'pipecal_fitpeak']


def elliptical_gaussian(coords,
                        baseline=0., dpeak=1.,
                        col_mean=0., row_mean=0.,
                        col_sigma=1., row_sigma=1.,
                        theta=0.):
    """
    Function for an elliptical Gaussian profile.

    Parameters
    ----------
    coords : tuple of arrays
        row,col coordinates of image
    baseline : float, optional
        Background flux level. Defaults to 0.
    dpeak : float, optional
        Maximum flux value. Defaults to 1.
    col_mean : float, optional
        col coordinate of star location. Defaults to 0.
    row_mean : float, optional
        row coordinate of star location. Defaults to 0
    col_sigma : float, optional
        col component of Gaussian sigma. Defaults to 1
    row_sigma : float, optional
        row component of Gaussian sigma. Defaults to 1
    theta : float, optional
        Angle the major axis of the Gaussian is rotated
        from the column axis in radians. Defaults to 0.

    Returns
    -------
    g : 1d array of floats
        Flattened 2D Gaussian array.

    """
    row, col = coords
    a = (np.cos(theta) / col_sigma)**2 + (np.sin(theta) / row_sigma)**2
    b = (np.sin(theta) / col_sigma)**2 + (np.cos(theta) / row_sigma)**2
    c = 2. * np.sin(theta) * np.cos(theta) * (1 / col_sigma**2
                                              - 1 / row_sigma**2)
    g = baseline + dpeak * np.exp(-1 / 2 * (a * (col - col_mean)**2
                                            + b * (row - row_mean)**2
                                            + c
                                            * (col - col_mean)
                                            * (row - row_mean)))
    return g.ravel()


def elliptical_lorentzian(coords,
                          baseline=0., dpeak=1.,
                          col_mean=0., row_mean=0.,
                          col_sigma=1., row_sigma=1.,
                          theta=0.):
    """
    Function for an elliptical Lorentzian profile.

    Parameters
    ----------
    coords : tuple of arrays
        row,col coordinates of image.
    baseline : float, optional
        Background flux level. Defaults to 0.
    dpeak : flat, optional
        Maximum flux value. Defaults to 1.
    col_mean : float, optional
        col coordinate of star location. Defaults to 0.
    row_mean : float, optional
        row coordinate of star location. Defaults to 0.
    col_sigma : float, optional
        col component of Lorentzian width. Defaults to 1.
    row_sigma : float, optional
        row component of Lorentzian width. Defaults to 1.
    theta : float, optional
        Angle the major axis of the Lorentzian is rotated
        from the column axis in radians. Defaults to 0.

    Returns
    -------
    g : 1d array of floats
        Flattened 2D Lorentzian array.
    """
    row, col = coords
    rwcol = (((col - col_mean) * np.cos(theta)
              + (row - row_mean) * np.sin(theta)) / col_sigma) ** 2
    rwrow = ((-(col - col_mean) * np.sin(theta)
              + (row - row_mean) * np.cos(theta)) / row_sigma) ** 2
    p = baseline + dpeak / (1 + rwcol + rwrow)
    return p.ravel()


def elliptical_moffat(coords,
                      baseline=0, dpeak=1.,
                      col_mean=0., row_mean=0.,
                      col_sigma=1., row_sigma=1.,
                      theta=0., beta=1.):
    """
    Function for an elliptical Moffat profile.

    Parameters
    ----------
    coords : tuple of floats
        row,col coordinates of image.
    baseline : float, optional
        Background flux level. Defaults to 0.
    dpeak : flat, optional
        Maximum flux value. Defaults to 1.
    col_mean : float, optional
        col coordinate of star location. Defaults to 0.
    row_mean : float, optional
        row coordinate of star location. Defaults to 0.
    col_sigma : float, optional
        col component of Moffat width. Defaults to 1.
    row_sigma : float, optional
        row component of Moffat width. Defaults to 1.
    theta : float, optional
        Angle the major axis of the Moffat profile is rotated
        from the column axis in radians. Defaults to 0.
    beta : float, optional
        Power law exponent. Defaults to 1.

    Returns
    -------
    g : 1d array of floats
        Flattened 2D Moffat array.
    """
    row, col = coords
    rwcol = (((col - col_mean) * np.cos(theta)
              + (row - row_mean) * np.sin(theta)) / col_sigma) ** 2
    rwrow = ((-(col - col_mean) * np.sin(theta)
              + (row - row_mean) * np.cos(theta)) / row_sigma) ** 2
    m = baseline + dpeak * (1 + rwcol + rwrow) ** (-beta)
    return m.ravel()


def _plot_comparison(fitdata, model, function):  # pragma: no cover
    """
    Plots difference between model and data.

    Saves the plot as model_fit_<function>.png

    Parameters
    ----------
    fitdata : numpy.ndarray
        Data that was fit
    model : numpy.ndarray
        Model of the best fit
    function : string
        Name of function used
    """
    from matplotlib.backends.backend_agg \
        import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(15, 5))
    FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    outname = 'model_fit_{0:s}.png'

    diff = fitdata - model
    vmin = np.nanmin(fitdata)
    vmax = np.nanmax(fitdata)

    ax[0].imshow(fitdata, cmap='jet', origin='bottom', vmin=vmin, vmax=vmax)
    ax[1].imshow(model, cmap='jet', origin='bottom', vmin=vmin, vmax=vmax)
    ax[2].imshow(diff, cmap='jet', origin='bottom', vmin=vmin, vmax=vmax)

    ax[0].set_title('Data')
    ax[1].set_title('Fit')
    ax[2].set_title('Difference')

    fig.tight_layout()
    fig.savefig(outname.format(function), dpi=300, bbox_inches='tight')


def pipecal_fitpeak(image, profile='moffat',
                    estimates=None, bounds=None,
                    error=None, bestnorm=True):
    """
    Fit a peak profile to a 2D image.

    Using scipy's curve_fit, fit either a 2D elliptical Gaussian,
    Lorentzian, or Moffat function to the image.

    Parameters
    ----------
    image : 2D numpy array
        The image to fit a profile to.
    profile : {'gaussian', 'lorentzian', 'moffat'}, optional
        Name of function to fit.
    estimates : dictionary, optional
        Initial estimates of fitting parameters. Keys should include:

            - *baseline* : background flux level
            - *dpeak* : peak value of the image
            - *col_mean* : col coordinate location of peak
            - *row_mean* : row coordinate location of peak
            - *col_sigma* : width of function in col direction
            - *row_sigma* : width of function in row direction
            - *theta* : angle function is rotated from column axis
            - *beta* : power law index of Moffat function. Ignored for
              Gaussian and Lorentzian fits.

        If not provided, generic estimates are generated based on `image`.
    bounds : dictionary, optional
        The limits of parameters for the fits. Must have the same keys as
        estimates. Each value should be a two element array-like containing
        the lower and upper limits of the parameter. To not impose a limit,
        set it to +/- inf.  If not provided, reasonable default bounds are
        used.
    error : 2D numpy array, optional
        Array with the 1-sigma uncertainties of each pixel in image. Must
        be the same shape as image.
    bestnorm : boolean, optional
        Set to return the summed square weighted residuals for the
        best-fit parameters. Defaults to True.

    Returns
    -------
    fit_param : dictionary
        A dictionary containing the best fit paramters. Has the same
        keys as estimates.
    fit_errs : dictionary
        A dictionary containing the 1-sigma uncertainities on the
        best fit parameters. Has the same keys as estimates.
    bestnorm : float
        The summed squared weighted residuals for the best-fit
        parameters.  Set to None if `bestnorm` is False.

    Raises
    ------
    PipeCalError
        If a provided input is not valid.
    """
    # Verify image is valid
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        msg = 'Image must be 2-dimensional numpy array'
        log.error(msg)
        raise PipeCalError(msg)

    # Verify error:
    if error is not None:
        if not isinstance(error, np.ndarray) or error.ndim != 2:
            msg = 'Error must be 2-dimensional numpy array'
            log.error(msg)
            raise PipeCalError(msg)
        if error.shape != image.shape:
            msg = 'Error must have the same shape as image'
            log.error(msg)
            raise PipeCalError(msg)

    # Verify function is a valid choice
    valid_functions = ['gaussian', 'lorentzian', 'moffat']
    if profile not in valid_functions:
        msg = 'Profile must be one of: gaussian, lorentzian, moffat'
        log.error(msg)
        raise PipeCalError(msg)

    # Coordinates for every pixel in image
    row, col = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # Verify estimates is set correctly
    fields = ['baseline', 'dpeak', 'col_mean', 'row_mean',
              'col_sigma', 'row_sigma', 'theta', 'beta']
    default_fwhm = 3.0
    if profile == 'gaussian':
        mfac = 2. * np.sqrt(2. * np.log(2.))
    else:
        mfac = 2.
    if estimates:
        if not isinstance(estimates, dict):
            msg = 'Estimates must be a dictionary'
            log.error(msg)
            raise PipeCalError(msg)

        inkeys = set(estimates.keys())
        reqkeys = set(fields)
        if profile == 'moffat':
            if inkeys != reqkeys:
                msg = 'Estimates missing required keys'
                log.error(msg)
                raise PipeCalError(msg)
        else:
            try:
                reqkeys.remove('beta')
                inkeys.remove('beta')
            except KeyError:
                pass
            if inkeys != reqkeys:
                msg = 'Estimates missing required keys'
                log.error(msg)
                raise PipeCalError(msg)
    else:
        # Initial guesses not given, come up with some
        estimates = dict()
        estimates['baseline'] = np.nanmedian(image)
        estimates['dpeak'] = np.nanmax(image)
        estimates['col_mean'] = (col.max() - col.min()) / 2.
        estimates['row_mean'] = (row.max() - row.min()) / 2.
        estimates['col_sigma'] = default_fwhm / mfac
        estimates['row_sigma'] = default_fwhm / mfac
        estimates['beta'] = 1.
        estimates['theta'] = 0.

    # Verify bounds  is set correctly
    if bounds:
        if not isinstance(bounds, dict):
            msg = 'Bounds must be a dictionary'
            log.error(msg)
            raise PipeCalError(msg)
        inkeys = set(bounds.keys())
        reqkeys = set(fields)
        if profile == 'moffat':
            if inkeys != reqkeys:
                msg = 'Bounds missing required keys'
                log.error(msg)
                raise PipeCalError(msg)
        else:
            try:
                reqkeys.remove('beta')
                inkeys.remove('beta')
            except KeyError:
                pass
            if inkeys != reqkeys:
                msg = 'Bounds missing required keys'
                log.error(msg)
                raise PipeCalError(msg)
        for key in bounds:
            if not hasattr(bounds[key], '__len__') or len(bounds[key]) != 2:
                msg = 'Elements of bounds must be 2-element list/array'
                log.error(msg)
                raise PipeCalError(msg)
    else:
        # No limits given, put up some basic ones
        bounds = dict()
        bounds['baseline'] = [0, np.inf]
        bounds['dpeak'] = [0, np.inf]
        bounds['col_mean'] = [col.min(), col.max()]
        bounds['row_mean'] = [row.min(), row.max()]
        bounds['col_sigma'] = [0, 2. * (col.max() - col.min()) / mfac]
        bounds['row_sigma'] = [0, 2. * (row.max() - row.min()) / mfac]
        bounds['beta'] = [1, 6]
        bounds['theta'] = [-np.pi / 2., np.pi / 2.]

    if profile == 'moffat':
        bounds = ([bounds['baseline'][0], bounds['dpeak'][0],
                  bounds['col_mean'][0], bounds['row_mean'][0],
                  bounds['col_sigma'][0], bounds['row_sigma'][0],
                  bounds['theta'][0], bounds['beta'][0]],
                  [bounds['baseline'][1], bounds['dpeak'][1],
                  bounds['col_mean'][1], bounds['row_mean'][1],
                  bounds['col_sigma'][1], bounds['row_sigma'][1],
                  bounds['theta'][1], bounds['beta'][1]])
    else:
        bounds = ([bounds['baseline'][0], bounds['dpeak'][0],
                  bounds['col_mean'][0], bounds['row_mean'][0],
                  bounds['col_sigma'][0], bounds['row_sigma'][0],
                  bounds['theta'][0]],
                  [bounds['baseline'][1], bounds['dpeak'][1],
                  bounds['col_mean'][1], bounds['row_mean'][1],
                  bounds['col_sigma'][1], bounds['row_sigma'][1],
                  bounds['theta'][1]])

    # Find any NaNs to ignore
    idx = np.where(np.isfinite(image))

    if error is not None:
        error = error[idx]

    if profile == 'gaussian':
        popt, pcov = curve_fit(elliptical_gaussian,
                               (row[idx], col[idx]), image[idx],
                               p0=(estimates['baseline'],
                                   estimates['dpeak'],
                                   estimates['col_mean'],
                                   estimates['row_mean'],
                                   estimates['col_sigma'],
                                   estimates['row_sigma'],
                                   estimates['theta']),
                               bounds=bounds, sigma=error,
                               absolute_sigma=True)

    elif profile == 'lorentzian':
        popt, pcov = curve_fit(elliptical_lorentzian,
                               (row[idx], col[idx]), image[idx],
                               p0=(estimates['baseline'],
                                   estimates['dpeak'],
                                   estimates['col_mean'],
                                   estimates['row_mean'],
                                   estimates['col_sigma'],
                                   estimates['row_sigma'],
                                   estimates['theta']),
                               bounds=bounds, sigma=error,
                               absolute_sigma=True)

    else:
        popt, pcov = curve_fit(elliptical_moffat,
                               (row[idx], col[idx]), image[idx],
                               p0=(estimates['baseline'],
                                   estimates['dpeak'],
                                   estimates['col_mean'],
                                   estimates['row_mean'],
                                   estimates['col_sigma'],
                                   estimates['row_sigma'],
                                   estimates['theta'],
                                   estimates['beta']),
                               bounds=bounds, sigma=error,
                               absolute_sigma=True)

    fit_param = dict()
    fit_errs = dict()
    for field, val, err in zip(fields, popt, np.sqrt(np.diag(pcov))):
        fit_param[field] = val
        fit_errs[field] = err

    if bestnorm:
        if profile == 'gaussian':
            model = elliptical_gaussian((row, col), *popt)
        elif profile == 'lorentzian':
            model = elliptical_lorentzian((row, col), *popt)
        else:
            model = elliptical_moffat((row, col), *popt)
        diff = image.ravel() - model
        bestnorm = np.nansum(diff ** 2)
        # plot_comparison(image, model.reshape(image.shape), profile)
    else:
        bestnorm = None

    return fit_param, fit_errs, bestnorm
