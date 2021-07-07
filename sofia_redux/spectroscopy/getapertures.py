# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.stats import sigma_clipped_stats
import numpy as np
from scipy.optimize import curve_fit

from sofia_redux.toolkit.utilities.func import gaussian_model

__all__ = ['get_apertures']


def get_apertures(profiles, apertures, refit_fwhm=True, get_bg=True,
                  bg_threshold=3.0, min_bg_pts=3):
    """
    Determine aperture radii for extraction.

    Profiles expected are the median profiles produced by
    `sofia_redux.spectroscopy.mkspatprof`.  Aperture positions may
    be produced by `sofia_redux.spectroscopy.findapertures`.  The
    `positions` dictionary will be copied to the return value and
    updated with appropriate values for the aperture radii:

        ``""aperture_radius"``
            The aperture radius, used as the integration radius
            in optimal extraction.
        ``"psf_radius"``
            The position at which the flux drops to zero.  Used
            as the total flux weighting radius for optimal extraction.
            For standard extraction, is used as the full integration
            radius.

    Background regions, if requested, are specified for the whole order,
    rather than for an individual aperture.

    Parameters
    ----------
    profiles : dict
        order (int) -> profile (numpy.ndarray)
            (2, n_spatial) spatial profile where profile[0] = spatial
            coordinate and profile[1] = median spatial profile.
    apertures : dict
        Apertures to update.  Keys are orders, values are list of dict.
        Required keys are:

        ``"position"``
            Aperture position (float)
        ``"fwhm"``
            Aperture FWHM (float)
        ``"sign"``
            Aperture sign ({1, -1})

        The dictionaries may optionally contain 'aperture_radius'
        and 'psf_radius': if present, these values are treated as fixed.
    refit_fwhm : bool, optional
        If set, the peak will be fit to re-determine the spatial FWHM
        of the aperture.
    get_bg : bool, optional
        If set, background regions will be determined from the non-aperture
        regions.

    Returns
    -------
    aperture_regions : dict
        order (int) -> dict
            Keys and values are as follows:

            ``"apertures"``
                List of dict with keys:

                ``"position"``
                    Aperture position (float)
                ``"fwhm"``
                    Aperture FWHM (float)
                ``"sign"``
                    Aperture sign ({1, -1})
                ``"aperture_radius"``
                    Aperture radius (float)
                ``"psf_radius"``
                    PSF radius (float)
                ``"mask"``
                    Aperture mask (numpy.ndarray)

            ``"background"``
                Dict with keys:

                ``"regions"``
                    List of tuple.
                    Values are (start, stop) positions in
                    arcsec up the slit.
                ``"mask"``
                    Background mask (numpy.ndarray)

    """
    orders = np.unique(list(apertures.keys())).astype(int)
    aperture_regions = {}
    for order in orders:
        space = profiles[order][0]
        profile = profiles[order][1]

        # do clipped stats on profile to get background level, std. dev
        prof_mean, prof_med, prof_std = sigma_clipped_stats(
            profile, sigma=bg_threshold)
        bg_est = prof_med

        ds = np.mean(space[1:] - space[:-1])
        slit_max = np.max(space)
        aperture_regions[order] = {}

        # sort apertures by position: work up the slit
        aperture_set = sorted(apertures[order], key=lambda j: j['position'])
        order_aps = []
        for i, aperture in enumerate(aperture_set):
            fwhm = aperture['fwhm']
            pos = aperture['position']
            sign = aperture['sign']

            if ('psf_radius' not in aperture
                    or 'aperture_radius' not in aperture):
                if refit_fwhm:
                    # refit the aperture within a window of 8 * fwhm,
                    # with tight bounds on the center
                    limit = 8. * fwhm
                    window = (space > (pos - limit)) \
                        & (space < (pos + limit))
                    x = space[window]
                    y = profile[window]

                    # params are x0, amplitude, fwhm, y0
                    param, _ = curve_fit(
                        gaussian_model, x, y,
                        p0=[pos, sign * np.max(np.abs(y)),
                            fwhm, bg_est],
                        bounds=([pos - ds, -np.inf, 1.0, -np.inf],
                                [pos + ds, np.inf, limit, np.inf]))
                    fit_fwhm = param[2]
                else:
                    fit_fwhm = fwhm

                # set aperture radius to the fit FWHM*0.7 -- this
                # should give close to optimal S/N for a Moffat or
                # Gaussian profile.
                # set psfradius to three times the aperture radius,
                # plus 20% to make sure to catch the profile wings
                if 'aperture_radius' in aperture:
                    aprad = aperture['aperture_radius']
                else:
                    aprad = fit_fwhm * 0.7
                if 'psf_radius' in aperture:
                    psfrad = aperture['psf_radius']
                else:
                    psfrad = fit_fwhm * 2.15
            else:
                aprad = aperture['aperture_radius']
                psfrad = aperture['psf_radius']
                fit_fwhm = aperture['fwhm']

            # check psfrad
            low = pos - psfrad
            if low < 0:
                log.warning('PSF radius overlaps the low edge of the slit '
                            f'for order {order}, aperture center '
                            f'{pos}. Reducing radius.')
                psfrad = pos
            high = pos + psfrad
            if high > slit_max:
                log.warning('PSF radius overlaps the high edge of the slit '
                            f'for order {order}, aperture center '
                            f'{pos}. Reducing radius.')
                psfrad = slit_max - pos

            # modify aprad if necessary
            if aprad > psfrad:
                log.warning('Aperture radius overlaps the PSF radius '
                            f'for order {order}, aperture center '
                            f'{pos}. Reducing radius.')
                aprad = psfrad

            # test the number of pixels extracted
            extract_pix = (space > (pos - psfrad)) & (space < (pos + psfrad))
            if np.sum(extract_pix) < 1:
                msg = 'PSF radius too small. Auto-set failed ' \
                      f'for order {order}, ' \
                      f'aperture center {pos}'
                log.error(msg)
                raise ValueError(msg)

            # check for overlap with previous aperture
            if i > 0:
                prev_ap = order_aps[i - 1]
                pmask = prev_ap['mask']
                overlap = pmask & extract_pix
                if np.any(overlap):
                    log.warning('Apertures overlap. Reducing radii.')
                    opix = int(np.ceil(np.sum(overlap) / 2)) + 1.5
                    prad = prev_ap['psf_radius']
                    ppos = prev_ap['position']

                    prad -= opix * ds
                    psfrad -= opix * ds

                    # test the number of pixels extracted,
                    # for previous aperture
                    pmask = (space > (ppos - prad)) & (space < (ppos + prad))
                    if np.sum(pmask) < 1:
                        msg = f'Auto-set failed for order {order}, ' \
                              f'aperture center {ppos}'
                        log.error(msg)
                        raise ValueError(msg)

                    prev_ap['psf_radius'] = prad
                    prev_ap['mask'] = pmask
                    if prev_ap['aperture_radius'] > prad:
                        prev_ap['aperture_radius'] = prad

                    # modify aprad if necessary
                    if aprad > psfrad:
                        log.warning('Aperture radius overlaps the PSF radius '
                                    f'for order {order}, aperture center '
                                    f'{pos}. Reducing radius.')
                        aprad = psfrad

                    # test the number of pixels extracted again,
                    # for current aperture
                    extract_pix = (space > (pos - psfrad)) \
                        & (space < (pos + psfrad))
                    if np.sum(extract_pix) < 1:
                        msg = f'Auto-set failed for order {order}, ' \
                              f'aperture center {pos}'
                        log.error(msg)
                        raise ValueError(msg)

                    # verify no further overlap
                    assert not np.any(pmask & extract_pix)

            order_ap = aperture.copy()
            order_ap['fwhm'] = fit_fwhm
            order_ap['aperture_radius'] = aprad
            order_ap['psf_radius'] = psfrad
            order_ap['mask'] = extract_pix

            order_aps.append(order_ap)

        aperture_regions[order]['apertures'] = order_aps

        # identify background regions from nearby non-aperture positions
        if get_bg:
            apmask = np.full(space.shape, False)
            localmask = np.full(space.shape, False)
            local_mult = 3
            for aperture in order_aps:
                apmask |= aperture['mask']

                pos = aperture['position']
                psfrad = aperture['psf_radius']
                local_test = \
                    (space > (pos - local_mult * psfrad)) \
                    & (space < (pos + local_mult * psfrad))
                localmask |= local_test

            # find non-aperture regions within threshold of
            # median profile value
            bg_mask = ~apmask
            bg_mask &= localmask
            bg_mask &= (np.abs(profile - bg_est) <= bg_threshold * prof_std)

            # check for contiguous regions of reasonable length
            background = []
            if np.sum(bg_mask) > min_bg_pts:
                # find jumps in background regions
                bg_idx = np.where(bg_mask)[0]
                break_points = np.where(bg_idx[1:] - bg_idx[:-1] > 1)[0]
                firsts = np.hstack([[bg_idx[0]], bg_idx[break_points + 1]])
                lasts = np.hstack([bg_idx[break_points], [bg_idx[-1]]])

                # keep good size regions
                for first, last in zip(firsts, lasts):
                    if last - first > min_bg_pts:
                        background.append((space[first], space[last]))

            aperture_regions[order]['background'] = \
                {'regions': background, 'mask': bg_mask}

    return aperture_regions
