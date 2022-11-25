# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils
from sofia_redux.instruments.exes.make_template import make_template
from sofia_redux.toolkit.utilities.fits import set_log_level

__all__ = ['spatial_shift']


def spatial_shift(data, header, flat, variance,
                  illum=None, good_frames=None, sharpen=False):
    """
    Shift spectra for spatial alignment.

    A shift is derived based on correlation with average spatial profile
    from all frames. The spatial template is generated using `make_template`.
    For each input frame, the image is collapsed over the spectral dimension,
    weighting by an inverse flat (squared), shifted by a small amount,
    then added to the collapsed spatial template.  This signal is calculated
    for all spatial shifts of up to 4 pixels: the spatial shift chosen
    is the one that maximizes the signal.

    Alternately, if sharpen is not set, the value maximized is the
    signal-to-noise (the above signal, divided by the collapsed uncertainty).

    The image is then shifted by the calculated value in the spatial
    direction, i.e. along the y-axis.  Input data is assumed to be undistorted
    and rotated to orient spectra along the x-axis.

    Parameters
    ----------
    data : numpy.ndarray
        Data cube, [nframe, nspec, nspat].
    header : fits.Header
        Header from FITS file.
    flat : numpy.ndarray
        Processed flat image, [nspec, nspat].
    variance : numpy.ndarray
        Variance planes corresponding to data, [nframe, nspec, nspat].
    illum : numpy.ndarray
        Indicating illuminated regions of the frame, [nspec, nspat].
        1=illuminated, 0=unilluminated, -1=pixel that does not correspond
        to any region in the raw frame.
    good_frames : array-like, optional
        Array of indices of good frames. If provided, any frame
        not in `good_frames` will be skipped.
    sharpen : bool, optional
        If set, signal will be maximized, rather than the
        signal-to-noise ratio.

    Returns
    -------
    data, variance : 2-tuple of numpy.ndarray
        Shifted data cube [nframe, nspec, nspat]
        and updated variance data [nframe, nspec, nspat].
    """

    params = _verify_inputs(data, header, flat, variance, illum=illum,
                            good_frames=good_frames, sharpen=sharpen)

    log.info('Shifting data to match first image')

    with set_log_level('WARNING'):
        _make_all_templates(params)
    _find_shift(params)
    shifted = _shift_data(params)

    return shifted


def _verify_inputs(data, header, flat, variance, illum,
                   good_frames, sharpen):
    """Check and assemble input data and options."""
    # Retrieve data dimensions
    ny = header['NSPEC']
    nx = header['NSPAT']
    try:
        nz = utils.check_data_dimensions(data=data, nx=nx, ny=ny)
    except RuntimeError:
        raise RuntimeError(f'Data has wrong dimensions ({data.shape}).'
                           f'Not shifting images.')

    # Store the order height as well
    n_slit = header.get('SLTH_PIX', ny)

    # Check that there are at least some good frames
    all_frames = np.arange(nz)
    if good_frames is None:
        good_frames = np.arange(nz)
    _, suball, _ = np.intersect1d(all_frames, good_frames,
                                  return_indices=True)
    if len(suball) == 0:
        raise RuntimeError('No good frames. Not shifting images.')

    if illum is None:
        illum = np.ones_like(data[0])

    params = {'data': data, 'header': header, 'flat': flat,
              'variance': variance, 'illum': illum,
              'good_frames': good_frames, 'sharpen': sharpen,
              'nx': nx, 'ny': ny, 'nz': nz, 'ns': n_slit}

    return params


def _make_all_templates(params):
    """Make spatial templates for all input frames."""
    header = params['header'].copy()
    weight_frame = params['flat'] ** 2
    illum = params['illum']
    good_frames = params['good_frames']

    data_templates = []
    std_templates = []
    for i, frame in enumerate(params['data']):
        if i not in good_frames:
            data_templates.append(None)
            std_templates.append(None)
        else:
            # Make a collapsed template from the single frame
            template = make_template(frame, header, weight_frame,
                                     illum=illum, collapsed=True)
            data_templates.append(template)

            # Also one from the variance
            template = make_template(params['variance'][i], header,
                                     weight_frame,
                                     illum=illum, collapsed=True)
            std_templates.append(np.sqrt(template))

    params['data_templates'] = data_templates
    params['std_templates'] = std_templates

    # Also make a normalized unweighted template of the weight
    # frame itself, inverted
    template = make_template(weight_frame, header, np.ones_like(weight_frame),
                             collapsed=True)
    params['weight_template'] = np.nansum(template) / template


def _correlation(comparison_template, test_template, shift_array):
    """Shift and correlate template with reference."""
    n_corr = shift_array.size
    corr = np.zeros(n_corr)
    for j in range(n_corr):
        signal_shift = _shift_1d_array(test_template, shift_array[j])
        corr[j] = np.nansum((comparison_template + signal_shift) ** 2)
    return shift_array[np.argmax(corr)]


def _shift_1d_array(xs, n):
    """Shift a 1D array."""
    e = np.empty_like(xs)
    if n == 0:
        return xs
    elif n >= 0:
        e[:n] = xs[0]
        e[n:] = xs[:-n]
    else:
        e[n:] = xs[-1]
        e[:n] = xs[-n:]
    return e


def _shift_2d_array(data, n):
    """Shift a 1D array."""
    if n == 0:
        return data
    # Shift up/down
    e = np.roll(data, n, axis=0)
    if n >= 0:
        # Shift up: fill gap at bottom
        fill = data[0, :]
        e[:n, :] = np.expand_dims(fill, axis=0)
    else:
        # Shift down: fill gap at top
        fill = data[-1, :]
        e[n:, :] = np.expand_dims(fill, axis=0)
    return e


def _find_shift(params):
    """Find optimum shift."""

    # Array of shifts from -4 to 4 (usually)
    n_shift = int(np.min([4, params['ns'] / 3]))
    shift_array = np.arange(-n_shift, n_shift + 1, dtype=int)

    # How much to shift each frame (default 0)
    i_shift_arr = np.zeros(params['nz'], dtype=int)

    comparison_template = None
    for i in range(params['nz']):
        # Signal template
        test_template = params['data_templates'][i]
        if test_template is None:
            continue

        # Divide by noise template if desired
        if params['sharpen']:
            test_template /= params['std_templates'][i]

        # Weight by flat, prior to shifting,
        # to prioritize source trace
        # todo: check if intent is to correct for source shift
        #  in slit or overall shift including slit -
        #  if the latter, weighting should happen after shift
        test_template *= params['weight_template']

        # Keep the first template to compare to
        if comparison_template is None:
            comparison_template = test_template
            continue

        # Find shift which maximizes contribution to S**2 or (S/N)**2,
        # checking all integer values between -n_shift and n_shift, inclusive
        best_shift = _correlation(comparison_template, test_template,
                                  shift_array)

        # Debug plots (not threadsafe!)

        # print('Frame ', i, best_shift)
        # from matplotlib import pyplot as plt
        # plt.plot(comparison_template)
        # plt.plot(test_template)
        # plt.plot(_shift_1D_array(test_template, best_shift))
        # plt.show()

        # If best shift is pegged at either limit, skip it
        if best_shift == shift_array[0] or best_shift == shift_array[-1]:
            log.debug(f'Spatial shift out of range for pair {i}. '
                      f'Setting to 0.')
        else:
            i_shift_arr[i] = best_shift

    # Remove the mean shift if it's > 0 integer pixels
    mean_shift = int(np.mean(i_shift_arr))
    log.debug(f'Initial shifts: {i_shift_arr}')
    log.debug(f'Mean shift: {mean_shift}')
    if abs(mean_shift) > 0:
        i_shift_arr -= mean_shift

    log.info(f'Derived shifts for all frames: {i_shift_arr}')
    params['derived_shifts'] = i_shift_arr


def _shift_data(params):
    """Apply shift to data."""
    shifted_data = np.full(params['data'].shape, np.nan)
    shifted_variance = np.full(params['data'].shape, np.nan)
    shifts = params['derived_shifts']
    for i in range(params['nz']):
        data = params['data'][i]
        var = params['variance'][i]
        if i not in params['good_frames'] or shifts[i] == 0:
            shifted_data[i] = data
            shifted_variance[i] = var
            continue

        shifted_data[i] = _shift_2d_array(data, shifts[i])
        shifted_variance[i] = _shift_2d_array(var, shifts[i])

    return shifted_data, shifted_variance
