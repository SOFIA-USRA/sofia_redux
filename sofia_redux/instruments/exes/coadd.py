# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils, make_template
from sofia_redux.toolkit.image.combine import combine_images

__all__ = ['coadd']


def coadd(data, header, flat, variance,
          illum=None, weights=None, good_frames=None,
          weight_mode=None, std_wt=False, threshold=0):
    """
    Combine individual frames to increase signal-to-noise.

    First, a template of the spectrum is created from all frames,
    averaging in the spectral direction. By default, all input frames
    are weighted by their correlation with this spatial template,
    so that if a frame is unusually noisy had some other error, it will
    not contribute significantly to the coadded frame. If this is not
    desired, an unweighted coadd can be performed, or the user can
    directly specify the weights to use.

    The weighted frames are then summed to effect a weighted mean of the
    input data. If the threshold parameter is provided, outlier rejection
    is additionally performed before the weighted mean. If provided, the
    variance is propagated accordingly.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, ny, nx].  Input data is assumed to
        be distortion corrected and rotated as needed to align the spectral
        direction along the x-axis.
    header : fits.Header
        Header of FITS file. Will be updated in place.
    flat : numpy.ndarray
        2D processed flat, as produced by makeflat [ny, nx].
    variance : numpy.ndarray
        3D variance cube [nframe, ny, nx]. Will be updated in place.
    illum : numpy.ndarray of int, optional
        Indicates illuminated regions of the frame [ny, nx].
        1=illuminated, 0=unilluminated, -1=pixel that does not
        correspond to any region in the raw frame.
    weights : array-like, optional
        Array [nframe] of fractional weights to use for input frames. If
        a weight is set to zero the frame will be skipped. Must add
        up to 1.0. If not provided, weights will be calculated for all
        frames.
    good_frames : array-like, optional
        Array of indices of good frames, max `nframe` length. If
        provided, any frame not in `good_frames` will be skipped.
    weight_mode : {'unweighted', 'useweights', None}, optional
        If 'unweighted', all good frames will be given equal weight.
        If 'useweights', the `weights` array will be used to weight the
        frames. Otherwise, weights will be calculated from a correlation
        with the spatial template.
    std_wt : bool, optional
        If set, frames will be weighted by the square root of their
        variance planes. Otherwise, they will be weighted by the flat.
    threshold : float, optional
        If >0, will be used as an outlier rejection threshold in the
        mean combination.

    Returns
    -------
    coadded_data, coadded_variance : numpy.ndarray, numpy.ndarray
        The coadded data and variance, both [ny,nx].
    """

    params = _verify_inputs(data, header, flat, variance, illum=illum,
                            weights=weights, good_frames=good_frames,
                            weight_mode=weight_mode, std_wt=std_wt,
                            threshold=threshold)
    if params['nz'] == 1:
        log.info('Only 1 frame available; no coadd performed.')
        return data[0], variance[0]

    _determine_weighting_method(params)
    _generate_template(params)

    log.info('Coadding diffs')

    _calculate_weights(params)
    _combine_data(params)
    _update_integration_time(params)

    return params['coadded'], params['coadded_var']


def _verify_inputs(data, header, flat, variance,
                   illum, weights, good_frames, weight_mode,
                   std_wt, threshold):
    """
    Check input shape and values.

    Later functions in this step take the output params dictionary as
    input and add to it or update it as output.

    Parameters
    ----------
    data : numpy.ndarray
    header : fits.Header
    flat : numpy.ndarray
    variance : numpy.ndarray
    illum : numpy.ndarray
    weights : array-like of float
    good_frames : array-like of int
    weight_mode : str
    std_wt : bool
    threshold : float

    Returns
    -------
    params : dict
        Contains all input data, reformatted as needed.
        Keys are:

            - 'data': data array
            - 'variance': variance array
            - 'header': header object
            - 'nx': data shape, x-direction
            - 'ny': data shape, y-direction
            - 'illum': illumination array
            - 'weights': list of weights for frames
            - 'good_frames': input list of good frame indices
            - 'weight_mode': input weight mode
            - 'std_wt': input standard deviation weight flag
            - 'threshold': input robust threshold
            - 'flat': flat array
            - 'suball': list of good frames indices
            - 'zwt_sum': sum of frame weights

    """
    nx = header['NSPAT']
    ny = header['NSPEC']

    params = {'data': data, 'variance': variance, 'header': header,
              'nx': nx, 'ny': ny, 'illum': illum, 'weights': weights,
              'good_frames': good_frames}
    try:
        nz = utils.check_data_dimensions(params=params)
    except RuntimeError:
        raise ValueError(f'Data has wrong dimensions {data.shape}. '
                         f'Not coadding images.') from None
    params['nz'] = nz

    try:
        utils.check_variance_dimensions(variance, nx, ny, nz)
    except RuntimeError:
        raise ValueError(f'Variance has wrong dimensions {variance.shape}. '
                         f'Not coadding images.') from None
    if params['illum'] is None:
        params['illum'] = np.ones((ny, nx))

    # Check which frames are good
    all_frames = np.arange(nz)
    if params['good_frames'] is None:
        params['good_frames'] = np.arange(nz)
    _, suball, _ = np.intersect1d(all_frames, params['good_frames'],
                                  return_indices=True)
    if len(suball) == 0:
        raise ValueError('No good frames. Not coadding images.')

    # Weight good frames equally if not provided
    if params['weights'] is None or len(params['weights']) != nz:
        weights = np.zeros(nz)
        weights[suball] = 1
    else:
        weights = params['weights']

    zwt_sum = np.nansum(np.abs(weights))
    if zwt_sum == 0:
        raise ValueError('All weights are zero. Not coadding images.')

    params.update({'weight_mode': weight_mode, 'std_wt': std_wt,
                   'threshold': threshold, 'flat': flat,
                   'weights': weights, 'suball': suball,
                   'zwt_sum': zwt_sum})

    return params


def _determine_weighting_method(params):
    """Determine weighting method from input parameters."""
    unweighted = False
    do_weights = True
    if params['weight_mode'] is not None:
        if params['weight_mode'].lower() in ['unweighted']:
            unweighted = True
        elif params['weight_mode'].lower() in ['useweights', 'use_weights']:
            do_weights = False
        else:
            pass
    if unweighted or params['nz'] < 4:
        log.info('Doing unweighted addition of pairs.')
        unweighted = True

    # Initially weight good frames equally if not provided
    if params['weights'] is None or len(params['weights']) != params['nz']:
        params['weights'] = np.zeros(params['nz'])
        params['weights'][params['suball']] = 1
        do_weights = True
    zwt_sum = np.sum(np.abs(params['weights']))
    if zwt_sum == 0:
        raise ValueError('All weights are zero. '
                         'Not coadding images.')
    if not do_weights and not np.allclose(zwt_sum, 1):
        raise ValueError('Weights do not add up to 1. '
                         'Not coadding images.')
    params['unweighted'] = unweighted
    params['do_weights'] = do_weights
    params['zwt_sum'] = zwt_sum


def _generate_template(params):
    """Make a spatial template from the input data."""
    if params['std_wt']:
        weight_frame = np.sqrt(params['variance'])
    else:
        weight_frame = params['flat']

    if params['unweighted']:
        template = None
    else:
        template = make_template.make_template(
            params['data'], params['header'], weight_frame,
            good_frames=params['good_frames'], illum=params['illum'])
        if template is None:
            log.error('Problem making template. Using unweighted coadd.')
            params['unweighted'] = True
    params['template'] = template
    params['weight_frame'] = weight_frame


def _calculate_weights(params):
    """Calculate weights from correlation with spatial template."""

    # Weight in proportion to S/N in spectrum extracted by
    # multiplying by template

    # Get good data from illum
    weights = params['weights']
    if params['do_weights']:
        gz = (params['illum'] == 1) & (params['flat'] != 0)
        params['gz'] = gz
        params['bz'] = ~params['gz']
        if np.sum(gz) == 0:
            raise ValueError('No good data. Not coadding images.')

        for i in range(params['nz']):

            # Check if weights should be calculated or just used as is
            if params['weights'][i] == 0:
                continue

            # Check if frame should be multiplied by -1
            if params['weights'][i] < 0:
                sign = -1
            else:
                sign = 1

            if params['unweighted']:
                weights[i] = sign
                continue

            t = params['template'][gz]
            a = (params['data'][i] * sign)[gz]

            if params['std_wt']:
                s2 = (params['weight_frame'][i] ** 2)[gz]
            else:
                s2 = (params['weight_frame'] ** 2)[gz]

            sum1 = np.nansum(t * a / s2)
            sum2 = np.nansum(t ** 2 / s2)
            wti = sum1 / sum2

            if wti == 0 or not np.isfinite(wti):
                log.warning(f'Correlation zero on pair {i + 1}')
                wti = 0
            elif wti < 0.1:
                log.warning(f'Correlation negative on pair {i + 1}')
                wti = 0
            weights[i] = sign * wti

    sum_wt = np.nansum(np.abs(weights))
    sum_wt_sq = np.nansum(np.abs(weights) ** 2)
    if (params['do_weights'] or params['unweighted']) and sum_wt > 0:
        weights = weights / sum_wt
    wt_max = np.nanmax(weights)

    idx = weights == 0
    if np.sum(idx) > 0:
        log.info(f'{np.sum(idx)} pair(s) given zero weight')
    if np.sum(idx) == params['nz']:
        raise ValueError('All weights zero. Not coadding images.')
    params['sum_wt'] = sum_wt
    params['sum_wt_sq'] = sum_wt_sq
    params['weights'] = weights
    params['wt_max'] = wt_max


def _combine_data(params):
    """Combine data with weighted mean."""

    # weights are by frame at this point: expand them to match data
    log.info(f'Weights: {params["weights"]}')
    weight_image = np.zeros_like(params['data'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        weight_image[:] = 1 / params['weights'][:, None, None]

    if params['threshold'] > 0:
        robust = True
        sigma = params['threshold']
        log.info(f'Performing robust mean with threshold {sigma}.')
    else:
        robust = False
        sigma = None
    coadded = combine_images(params['data'], variance=weight_image,
                             method='mean', weighted=True,
                             robust=robust, returned=False, sigma=sigma)
    _, coadded_var = combine_images(params['data'],
                                    variance=params['variance'],
                                    method='mean', weighted=True,
                                    robust=robust, returned=True,
                                    sigma=sigma)
    params['coadded'] = coadded
    params['coadded_var'] = coadded_var


def _update_integration_time(params):
    """Update integration time for combined data."""
    int_time = (params['header']['BEAMTIME'] * params['nz']
                * params['header']['NINT'])
    instmode = str(params['header'].get('INSTMODE', 'UNKNOWN')).strip().upper()
    if instmode == 'NOD_ON_SLIT':
        int_time *= 2

    log.info(f'Total on-source integration time: {int_time}')
    params['header']['EXPTIME'] = int_time
    params['header']['NEXP'] = params['nz'] * params['header']['NINT']
