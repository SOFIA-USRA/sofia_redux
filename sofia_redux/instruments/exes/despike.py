# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

__all__ = ['despike']


def despike(data, header, variance=None,
            abeams=None, bbeams=None, gooddata=None,
            propagate_nan=False):
    """
    Correct outlier pixels due to temporal spikes.

    All A frames are compared, and any pixels with values greater
    than a threshold factor (header['SPIKEFAC']) of standard deviations
    from the mean value across the other frames are replaced with
    that mean value. B frames are similarly compared with each
    other.

    Optionally (if header['TRASH'] is True), frames with significantly
    discrepant overall background levels ("trashed" frames) may be
    identified automatically and flagged for removal from subsequent
    reduction.

    Parameters
    ----------
    data : numpy.ndarray
        3D data cube [nframe, nspec, nspat].
    header : fits.Header
        Header of FITS file.
    variance : numpy.ndarray
        3D variance cube [nframe, nspec, nspat].
    abeams : array-like of int
        Index numbers of A frames in data cube.
    bbeams : array-like of int
        Index numbers of B frames in data cube.
    good_data : numpy.ndarray, optional
        Bad pixel array [nspec, nspat] indicating valid data
        (True=good, False=bad).
    propagate_nan : bool, optional
        If True, outlier pixels will be replaced by NaN rather
        than the average value.

    Returns
    -------
    spike_data, spike_mask, good_frames : 3-tuple of numpy.ndarray
        The corrected data, outlier pixel mask, and list of good data frames.
        Data and mask have dimensions [nframe, nspec, nspat]. In the mask,
        True=good, False=spike. The good frames list includes indices for
        all good (non-trashed) frames in the input data.
    """

    good_frames = list()
    spike_mask = np.full(data.shape, True)

    nz = _check_dimensions(data, header)
    dovar = _check_variance(variance, header, nz)
    good_index = _check_good_array(gooddata, header)

    beams = _apply_beams(data, abeams, bbeams)

    read_noise = _read_noise_contribution(header)

    spike_data = data.copy()
    for beam_name, beam in beams.items():
        if beam['data'].size == 0:
            continue
        log.info('')
        log.info(f'Despiking {beam_name} beams')

        beam_nz = _check_dimensions(beam['data'], header)
        sky = np.nanmean(beam['data'][:, good_index], axis=1)
        trash_frame, ntrash = _trash_frame_check(sky, beam_nz, header,
                                                 beam_name)

        good_beam_frames = ~trash_frame
        num_good_z = good_beam_frames.sum()
        if num_good_z < 2:
            log.warning(f"Can't despike an array with less than 2 "
                        f"good frames. Not applying despike correction to "
                        f"{beam_name} beams.")
            good_frames.extend(list(beam['beam'][good_beam_frames]))
            continue

        good_pix, scaled_data, avg_pix, avg_sky = _saturated_pixels(
            header, full_sky=sky, beam_data=beam['data'],
            good_beam_frames=good_beam_frames,
            frame_gain=read_noise['frame_gain'],
            good_index=good_index)

        calc_var = _calculate_variance(scaled_data, avg_pix,
                                       num_good_z, read_noise)

        if float(header['SPIKEFAC']) > 0:
            above_average = ((calc_var['var'] > calc_var['avgvar'])
                             & good_pix)

            # loop over each frame, comparing the
            # average without the value for this frame
            # to the average with it
            now_good = good_beam_frames
            goodbeamframes = good_beam_frames.nonzero()[0]
            spike_count_limit = 5e-3 * header['NSPEC'] * header['NSPAT']
            for i in range(num_good_z):
                frame_index = goodbeamframes[i]
                frame = beam['beam'][frame_index]
                frame_value = calc_var['value'][i]
                spike_index, averages = _find_spikes(
                    header, calc_var['value'], num_good_z,
                    above_average, frame_value)

                # mark saturated values in spike mask
                spike_mask[frame] = good_pix.copy()

                if spike_index.sum() > 0:
                    log.info(f'Frame {frame}: {spike_index.sum()} spikes')

                    # replace spikes in data
                    _replace_spikes(spike_data, frame, spike_index,
                                    averages, avg_pix,
                                    sky[frame_index],
                                    average_sky=avg_sky, do_var=dovar,
                                    variance=variance,
                                    propagate_nan=propagate_nan)

                    # also mark them in the mask
                    spike_mask[frame][spike_index] = False

                if spike_index.sum() > spike_count_limit:
                    log.warning(f'Too many spikes in frame {frame} '
                                f'(> {spike_count_limit}).')
                    if now_good.sum() >= 2:
                        log.warning('Marking frame as bad.')
                        now_good[frame_index] = False
                    else:
                        log.warning('But skipping would result '
                                    'in no good pairs.')

            good_beam_frames = now_good
        good_frames.extend(list(beam['beam'][good_beam_frames]))

    # Sort frames: numbers refer to index of original data frames
    good_frames.sort()
    log.info('')
    log.info(f'All good frames after despike: {good_frames}')

    return spike_data, spike_mask, good_frames


def _check_dimensions(data, header):
    if data.ndim <= 2:
        nz = 1
    else:
        nz = data.shape[0]
    if (data.ndim not in [2, 3]
            or data.shape[-1] != header['NSPAT']
            or data.shape[-2] != header['NSPEC']):
        raise ValueError(f'Data has wrong dimensions ({data.shape}). '
                         f'Not despiking frames')
    return nz


def _check_variance(variance, header, nz):
    if variance is not None:
        dovar = True
        if variance.ndim <= 2:
            nvz = 1
        else:
            nvz = variance.shape[0]
        if (nz != nvz
                or variance.shape[-1] != header['NSPAT']
                or variance.shape[-2] != header['NSPEC']):
            raise ValueError(f'Variance has wrong dimensions '
                             f'({variance.shape}). Not despiking frames.')
    else:
        dovar = False
    return dovar


def _check_good_array(good_data, header):
    if good_data is None:
        good_index = np.full((header['NSPAT'], header['NSPEC']), True)
    else:
        good_index = good_data
    if not good_index.any():
        raise ValueError('No good pixels in data array. Not applying despike')
    return good_index


def _apply_beams(data, abeams, bbeams):
    if data.ndim <= 2:
        nz = 1
    else:
        nz = data.shape[0]
    beam_data = dict()
    abeam_data, bbeam_data = np.empty((0, 0)), np.empty((0, 0))
    if abeams is not None and len(abeams) != 0:
        abeam_data = data[abeams]
    if bbeams is not None and len(bbeams) != 0:
        bbeam_data = data[bbeams]
    if ((abeams is None and bbeams is None)
            or (len(abeams) == 0 and len(bbeams) == 0)):
        abeams = np.arange(nz)
        abeam_data = data
    beam_data['A'] = {'beam': np.array(abeams), 'data': abeam_data}
    beam_data['B'] = {'beam': np.array(bbeams), 'data': bbeam_data}
    return beam_data


def _read_noise_contribution(header):
    nx = header['NSPAT']
    frame_time = header['FRAMETIM']
    gain = header['PAGAIN']
    beamtime = header['BEAMTIME']
    e_per_adu = header['EPERADU']
    read_noise = header['READNOIS']

    if frame_time * beamtime * gain <= 0:
        frame_time = 1
        beamtime = 1
        gain = 1

    frame_gain = frame_time * abs(gain)
    varmin = nx / (frame_time * beamtime * gain ** 2)
    e_per_adu *= beamtime
    read_var = (read_noise / e_per_adu) ** 2

    out = {'frame_gain': frame_gain, 'varmin': varmin,
           'read_var': read_var, 'frame_time': frame_time,
           'beam_time': beamtime, 'read_noise': read_noise,
           'gain': gain, 'eperadu': e_per_adu}
    return out


def _trash_frame_check(sky, nz, header, beam_name):
    trash_frame = np.zeros(nz, dtype=bool)
    if header.get('TRASH', False):
        diff = sky - np.nanmean(sky)
        sum_diff = diff.sum()
        sum_sq_diff = (diff ** 2).sum()

        # average difference without a given frame
        avg1 = (sum_diff - diff) / (nz - 1)

        # variance without a given frame
        var1 = (sum_sq_diff - diff ** 2) / (nz - 1) - avg1 ** 2

        # frames for which the diff-avg is greater than trashpar * var
        trash_frame = (diff - avg1) ** 2 > header['TRASH'] * var1
        if np.any(trash_frame):
            log.info(f'{trash_frame.sum()} frame(s) '
                     f'from {beam_name} are trashed')

    return trash_frame, sum(trash_frame)


def _saturated_pixels(header, full_sky, beam_data, good_beam_frames,
                      frame_gain, good_index):
    allowed_max = header['SATVAL'] / frame_gain
    sky = full_sky[good_beam_frames]
    avg_sky = np.nanmean(sky)
    scaled = np.zeros((len(sky), header['NSPEC'], header['NSPAT']))

    good_data = beam_data[good_beam_frames]
    for i in range(len(sky)):
        scaled[i] = good_data[i] * avg_sky / sky[i]

    avg_pix = np.nanmean(scaled, axis=0)
    if allowed_max > 0:
        saturated_mask = avg_pix > allowed_max
        good_index[saturated_mask] = False
        if saturated_mask.sum():
            log.info(f'{saturated_mask.sum()} saturated pixels found')

    return good_index, scaled, avg_pix, avg_sky


def _calculate_variance(scaled, average_pixels, num_good_z,
                        read_noise):

    broadcast = np.broadcast_to(average_pixels,
                                (num_good_z,) + average_pixels.shape)
    value = scaled - broadcast
    mean_value = np.nanmean(value, axis=0)
    mean_value_sq = np.nanmean(value**2, axis=0)
    var = (mean_value_sq - mean_value**2)

    avgvar = np.nanmean(var)
    calcvar = (np.nanmean(average_pixels) / read_noise['eperadu']
               + read_noise['read_var'])
    log.info(f'Mean measured, calculated stddev: {np.sqrt(avgvar):6f}, '
             f'{np.sqrt(calcvar):6f}')
    if avgvar < read_noise['varmin']:
        log.debug('Possible inadequate digitization')
        rms = (np.abs(read_noise['gain'])
               * np.sqrt(avgvar * read_noise['frame_time']
                         * read_noise['beam_time']))
        mean_per_frame = np.nanmean(average_pixels) * read_noise['frame_gain']
        log.debug(f'RMS std dev, mean/frame = {rms:.6f}, {mean_per_frame:.6f}')

    variance = {'avgvar': avgvar, 'calcvar': calcvar, 'var': var,
                'value': value}
    return variance


def _find_spikes(header, var_value, num_good_z,
                 above_average, frame_value):
    # sum over all variance values (includes frame_value)
    sum_value = np.sum(var_value, axis=0)
    sum_sq_value = np.sum(var_value**2, axis=0)

    # compare frame value to average without this frame
    average_1 = (sum_value - frame_value) / (num_good_z - 1)
    average_square_1 = ((sum_sq_value - frame_value**2)
                        / (num_good_z - 1))
    difference = (frame_value - average_1) ** 2

    # compute threshold, in variance
    threshold = header['SPIKEFAC']**2 * (average_square_1 - average_1**2)

    # mark discrepant pixels
    spike_index = above_average & (difference > threshold)
    averages = {'avg1': average_1, 'avgsq1': average_square_1}

    return spike_index, averages


def _replace_spikes(spike_data, frame_index, spike_index, averages,
                    average_pixels, sky, average_sky, do_var, variance,
                    propagate_nan=False):
    """
    Replace spike with average value in data frame.

    Parameters
    ----------
    spike_data
    frame_index

    Returns
    -------

    """
    d = spike_data[frame_index]
    if propagate_nan:
        d[spike_index] = np.nan
    else:
        d[spike_index] = ((averages['avg1'][spike_index]
                           + average_pixels[spike_index])
                          * sky / average_sky)
    spike_data[frame_index] = d

    if do_var:
        v = variance[frame_index]
        if propagate_nan:
            v[spike_index] = np.nan
        else:
            # Note: in previous versions of the pipeline, the variance
            # was replaced by the variance in the data over the
            # remaining frames:
            #   v[spike_index] = (averages['avgsq1'][spike_index]
            #                     - averages['avg1'][spike_index] ** 2)
            # But this results in variance values that are much too low.
            # We leave them as is here, rather than interpolating over
            # them, since the calling pipeline now typically uses
            # propagate_nan for this pipeline step anyway.

            pass

        variance[frame_index] = v
