# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.instruments.exes import utils

__all__ = ['make_template']


def make_template(data, header, weighting_frame,
                  weights=None, illum=None, good_frames=None,
                  collapsed=False):
    """
    Make a spectral template for weighting and shifting spectra.

    Good frames are weighted by 1/weighting_frame^2 and averaged
    together, then the resulting frame is averaged in the spectral
    direction, and replicated over the spectral elements. The result
    is an average spatial profile that can be cross-correlated with
    individual 2D spectral frames, to determine spatial shifts
    or coadd weighting.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, ny, nx].  Input data is assumed to
        be distortion corrected and rotated as needed to align the spectral
        direction along the x-axis.
    header : Header
        FITS header associated with the data.
    weighting_frame : numpy.ndarray
        Weighting frame (eg. flat or uncertainty image). May be a single
        2D frame [ny, nx] to apply to all frames or a 3D
        cube [nframe, ny, nx].
    weights : numpy.ndarray
        1D array [nframe] of weights to use for input frames. If
        weight=0, frame will be skipped.
    illum : numpy.ndarray
        2D array [ny, nx] indicates illuminated regions of
        the frame. 1 = illuminated, 0 = unilluminated, -1 = pixel that
        does not correspond to any region in the raw frame (before
        undistortion).
    good_frames : numpy.ndarray, optional
        1D array of indices of good frames. If provided, any frame not in
        `good_frames` will be skipped.
    collapsed : bool, optional
        If set, the average template collapsed along the spectral dimension
        will be returned instead of the full 2D array.

    Returns
    -------
    template : numpy.ndarray
        2D spatial template [ny, nx], or 1D [ny] if
        `collapsed` is set.
    """
    nx = header.get('NSPAT')
    ny = header.get('NSPEC')
    shape = (ny, nx)

    try:
        nz = utils.check_data_dimensions(data=data, nx=nx, ny=ny)
    except RuntimeError:
        log.error(f'Data has wrong dimensions {data.shape}. '
                  f'Not making template')
        return None

    if weighting_frame.ndim <= 2:
        weighting_frame = np.repeat(weighting_frame[np.newaxis, :], nz, axis=0)

    if weights is None:
        weights = np.ones(nz)
    if illum is None:
        illum = np.ones((ny, nx))

    log.info('Making spatial template.')
    try:
        z_weight_sum = _weight_good_frames(good_frames, weights, nz)
    except RuntimeError as msg:
        log.error(msg)
        return None

    try:
        z_good, weight_mask = _good_data_points(illum, weighting_frame,
                                                shape, weights)
    except RuntimeError as msg:
        log.error(msg)
        return None

    # Average array in z
    if nz > 1:
        idx = np.isnan(data)
        masked_data = np.ma.MaskedArray(data, mask=idx)
        z_avg = np.ma.average(masked_data, weights=weight_mask, axis=0)
        z_avg = np.ma.filled(z_avg, fill_value=np.nan)
        z_weight_frame = np.nansum(np.abs(weight_mask), axis=0) / z_weight_sum
    else:
        z_avg = data * weight_mask / z_weight_sum
        z_weight_frame = np.abs(weight_mask) / z_weight_sum

    template = _create_template_image(z_good, z_avg, nx, ny, z_weight_frame,
                                      collapsed)

    return template


def _weight_good_frames(good_frames, weights, nz):
    """Weight frames according to input."""

    # check which frames are good
    all_frames = np.arange(nz)
    if good_frames is None or len(good_frames) == 0:
        good_frames = all_frames.copy()
    _, suball, subgood = np.intersect1d(all_frames, good_frames,
                                        return_indices=True)
    if len(suball) == 0:
        raise RuntimeError('No good frames. Not making templates.')

    # weight good frames equally if not provided
    if weights is None or len(weights) != nz:
        weights = np.zeros(nz)
        weights[suball] = 1

    # set weights to zero for any bad frames
    for i in range(nz):
        if i not in suball:
            weights[i] = 0

    z_weight_sum = np.sum(np.abs(weights))
    return z_weight_sum


def _good_data_points(illum, weighting_frame, shape, weights):
    """Get good data points from weighting frame or illumination mask."""
    good_idx = (illum == 1) & (weighting_frame > 0)
    if good_idx.sum() == 0:
        raise RuntimeError('No good data points. Not making template.')

    if np.sum(~good_idx) > 0:
        # Avoid dividing by zero
        c_weighting_frame = weighting_frame.copy()
        c_weighting_frame[~good_idx] = 1.
        weighting_frame_weight = 1 / c_weighting_frame ** 2
        weighting_frame_weight[~good_idx] = 0.
    else:
        weighting_frame_weight = 1 / weighting_frame ** 2

    z_good = np.ones(shape, dtype=bool)
    weight_mask = (weights * (good_idx.astype(float)
                              * weighting_frame_weight.astype(float)).T).T
    z_good = np.all(np.vstack((z_good[None], good_idx)), axis=0)
    if weight_mask.shape[0] == 1:
        weight_mask = weight_mask[0]

    return z_good, weight_mask


def _create_template_image(z_good, z_avg, nx, ny, z_weight_frame, collapsed):
    """Collapse and replicate template spectrally (over x)."""
    # Note: this assumes undistortion (tort) has been performed previously
    # and cross-dispersed data have been rotated to align the spectral
    # direction with the x-axis
    template = np.zeros((ny, nx))
    illsum = np.nansum(z_good, axis=1)
    ysum = np.nansum(z_avg, axis=1)

    idx = illsum > ny / 2
    if np.sum(idx) == 0:
        log.error('No good data points. Not making template.')
        return None

    flat_weight_sum = np.nansum(z_weight_frame, axis=1)
    template_col = np.zeros(ny)
    template_col[idx] = ysum[idx] / flat_weight_sum[idx]

    if collapsed:
        return template_col

    template[:] = template_col[:, None]
    template[z_good != 1] = 0

    return template
