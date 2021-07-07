# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np

from sofia_redux.toolkit.image.adjust import shift
from sofia_redux.toolkit.fitting.polynomial import Polyfit
from sofia_redux.toolkit.interpolate import interpolate_nans

__all__ = ['get_wave_shift', 'fluxcal']


def get_wave_shift(flux, correction, shift_limit, model_order,
                   subsample=10):
    """
    Get pixel shift between flux and correction curve.

    Flux is directly shifted then corrected to determine shift value with
    lowest RMS, compared to a low order polynomial model.

    The correction curve is typically atmospheric transmission * response.

    Parameters
    ----------
    flux : numpy.ndarray of float
        Spectral flux (nw,).
    correction : numpy.ndarray of float
        Correction curve (nw,)
    shift_limit : float
        Maximum shift to consider.
    model_order : int
        Polynomial order of continuum model, for calculating optimum
        residuals.
    subsample : int, optional
        Subsampling for sub-pixel shifts.  Set to 1 to get whole pixel
        shifts only.

    Returns
    -------
    float
        The wavelength shift in pixels.
    """
    # subsample data to get sub-pixel lag info
    xin = np.arange(flux.size, dtype=float)
    if subsample > 1:
        xout = np.arange(flux.size * subsample, dtype=float) / subsample
        sflux = interpolate_nans(xin, flux, xout)
        scorrect = interpolate_nans(xin, correction, xout)
    else:
        xout = xin
        subsample = 1
        sflux = flux
        scorrect = correction

    good = ~np.isnan(sflux) & ~np.isnan(scorrect)
    if not np.any(good):
        return np.nan

    # lag array: distance from center wavelength
    ns = sflux.size
    lag = np.arange(ns, dtype=float) - ns / 2
    lag /= subsample

    fit_rchisq = []
    fit_shift = []
    fit_corr = []
    for sval in lag[np.abs(lag) < shift_limit]:
        test_flux = np.interp(xout, xout + sval, sflux,
                              left=np.nan, right=np.nan)
        corr_data = test_flux / scorrect

        poly = Polyfit(xout[good], corr_data[good], model_order)
        fit_rchisq.append(poly.stats.rchi2)
        fit_shift.append(sval)
        fit_corr.append(corr_data)

    # find the best shift by minimizing residuals
    min_idx = np.argmin(fit_rchisq)
    shiftval = fit_shift[min_idx]

    """
    # plotting code for debugging
    from matplotlib import pyplot as plt
    plt.plot((xin - flux.size / 2.0),
             flux / np.mean(flux), label='input flux')
    plt.plot(lag, sflux / np.mean(sflux[good]), label='subsampled flux')
    plt.plot(lag, scorrect / np.mean(scorrect[good]),
             label='correction curve')
    plt.plot(lag + shiftval, sflux / np.mean(sflux[good]),
             label='shifted flux')
    plt.plot(lag, fit_corr[min_idx] / np.nanmean(fit_corr[min_idx]),
             label='corrected flux')
    plt.legend()
    plt.show()
    plt.plot(fit_shift, fit_rchisq, label='reduced chi-squared')
    plt.axvline(shiftval)
    plt.title(f'Shift value: {shiftval}')
    plt.legend()
    plt.show()
    #"""

    return shiftval


def fluxcal(spectra, atran, response=None,
            auto_shift=False, shift_limit=5.0,
            shift_subsample=10, model_order=1):
    """
    Calibrate and telluric correct spectral flux.

    The provided atmospheric transmission and response values are
    interpolated onto the wavelength range for each order.  The correction
    curve is computed as transmission * response.

    Optionally, an optimum wavelength shift for the 1D spectra, relative
    to the correction curve, may be computed by minimizing the residuals
    in the corrected spectrum. If the calculated shift is greater than
    0.1 pixel and less than `shift_limit`, the spectrum is shifted by
    this amount, via a linear interpolation, along the wavelength dimension
    before correction.

    The 1D spectral_flux and spectral error provided are divided by the
    correction curve.  For the 2D spectral flux and error, each
    row is divided by the correction curve.

    If multiple ATRAN data sets are provided, then the optimum correction
    will be determined by fitting a low order polynomial to the corrected
    1D spectrum.  The correction curve that produces the lowest chi-squared
    residuals on the fit is selected.  The index of the ATRAN data set
    chosen is returned in the output dictionary, with key 'atran_index'.

    The calculated wavelength shift and the interpolated transmission,
    response, and response error are stored in the output dictionary,
    with keys 'wave_shift', 'transmission', 'response', and
    'response_error', respectively.

    Parameters
    ----------
    spectra : dict
        Spectra to calibrate.

        Structure is:
            order (int) -> list of dict
                flux -> numpy.ndarray (ns, nw)
                    Rectified 2D spectral flux image.
                error -> numpy.ndarray (ns, nw)
                    Rectified 2D spectral error image.
                wave -> numpy.ndarray (nw,)
                    Wavelength coordinates.
                spectral_flux -> numpy.ndarray (nw,)
                    1D spectral flux.
                spectral_error -> numpy.ndarray (nw,)
                    1D spectral error.
                wave_shift -> float
                    Manual shift to apply in the wavelength dimension
                    (pixels). If present and not None, will override
                    the `auto_shift` parameter.
    atran : numpy.ndarray or list of numpy.ndarray
        A (2, nt) array; first element is the wavelength coordinate, second
        is the fractional transmission.  If a list of arrays is provided,
        the optimum one will be selected.
    response : dict, optional
        The instrumental response for the order.  If not provided,
        only the ATRAN correction will be applied.

        Structure is:
            order (int) -> dict
                wave -> numpy.ndarray (nr,)
                    Wavelength coordinates.  Need not match the spectral
                    flux coordinates.
                response -> numpy.ndarray (nr,)
                    Instrument response, in raw units/Jy.
                error -> numpy.ndarray (nr,)
                    Error on the response.
    auto_shift : bool, optional
        If set, the spectrum cross-correlated with the
        response * transmission, and the calculated wavelength
        shift will be applied to the flux and error images and spectra.
    shift_limit : float, optional
        Maximum wavelength shift to be applied by the `auto_shift`,
        in pixels.
    shift_subsample : int, optional
        Subsampling for wavelength shifts.  Set to 1 to get whole pixel
        shifts only.
    model_order : int, optional
        Polynomial order for continuum model, used in optimizing wavelength
        shifts and ATRAN optimization.

    Returns
    -------
    spectra : dict
        Calibrated spectra.

        Structure is:
            order (int) -> list of dict
                flux -> numpy.ndarray (ns, nw)
                    Calibrated rectified 2D spectral flux image.
                error -> numpy.ndarray (ns, nw)
                    Calibrated rectified 2D spectral error image.
                wave -> numpy.ndarray (nw,)
                    Wavelength coordinates.
                spectral_flux -> numpy.ndarray (nw,)
                    Calibrated 1D spectral flux.
                spectral_error -> numpy.ndarray (nw,)
                    Calibrated 1D spectral error.
                wave_shift -> float
                    Wavelength shift applied.
                atran_index -> int
                    Index of the ATRAN data set selected.
                transmission -> numpy.ndarray (nw,)
                    Transmission correction applied to flux.
                response -> numpy.ndarray (nw,)
                    Response correction applied to flux.
                response_error -> numpy.ndarray (nw,)
                    Error on the response.
    """

    # use all orders defined in spectra
    orders = np.unique(list(spectra.keys())).astype(int)

    # if more than one atran provided, choose the best one
    if not isinstance(atran, list):
        if isinstance(atran, np.ndarray) and atran.ndim < 3:
            atran = [atran]
    if len(atran) > 1:
        optimize = True
    else:
        optimize = False

    # lower limit for wavelength shifts, in pixels --
    # below this, it's not worth the interpolation
    eps = 0.1

    # Loop through each order
    result = {}
    for orderi, order in enumerate(orders):
        result[order] = []

        for speci, spectrum in enumerate(spectra[order]):
            out_spectrum = {}

            # bail if there's no good data
            if np.all(np.isnan(spectrum['spectral_flux'])):
                log.error(f'No good flux in order {order}, spectrum {speci}')
                return

            wave = spectrum['wave']
            dw = np.mean(wave[1:] - wave[0:-1])
            out_spectrum['wave'] = wave.copy()

            # check for manual wave shift
            if 'wave_shift' in spectrum and spectrum['wave_shift'] is not None:
                check_auto = False
                wave_shift = spectrum['wave_shift']
            else:
                check_auto = auto_shift
                wave_shift = 0.0

            # interpolate response onto the wavelength range
            if response is not None:
                rwave = response[order]['wave']
                rdata = response[order]['response']
                rerr = response[order]['error']
                rmatch = np.interp(wave, rwave, rdata,
                                   left=np.nan, right=np.nan)
                ematch = np.interp(wave, rwave, rerr,
                                   left=np.nan, right=np.nan)
            else:
                rmatch = np.ones_like(wave)
                ematch = np.zeros_like(wave)

            # for all atran files, interpolate onto the
            # spectral wavelength range
            all_tran = []
            all_corr = []
            all_shift = []
            fit_chisq = []
            fit_rchisq = []
            for i, (awave, adata) in enumerate(atran):
                # interpolate atran onto input wavelengths
                amatch = np.interp(wave, awave, adata,
                                   left=np.nan, right=np.nan)

                # divide spectrum by transmission and response
                flux = spectrum['spectral_flux'].copy()
                correction = amatch * rmatch
                corr_data = flux / correction
                good = ~np.isnan(corr_data)
                all_tran.append(amatch)

                # auto shift if desired
                shiftval = wave_shift
                if check_auto:
                    shiftval = get_wave_shift(
                        flux, correction, shift_limit, model_order,
                        subsample=shift_subsample)
                    if np.isnan(shiftval):
                        log.warning('Could not calculate wave shift; '
                                    'setting to 0.')
                        shiftval = 0.0
                    log.debug('Calculated wave shift: {}'.format(shiftval))

                all_shift.append(shiftval)
                if np.abs(shiftval) > eps:
                    if not check_auto or np.abs(shiftval) < shift_limit:
                        # interpolate spectrum from shifted wavelength onto
                        # standard ATRAN wavelength
                        flux = np.interp(wave + shiftval * dw, wave, flux,
                                         left=np.nan, right=np.nan)
                        corr_data = flux / correction
                        good = ~np.isnan(corr_data)

                all_corr.append(corr_data)

                # if optimizing, fit a low order polynomial to the corrected
                # spectrum, and calculate residuals and chi-squared error
                if optimize:
                    robust = 6.0
                    poly = Polyfit(wave[good], corr_data[good], model_order,
                                   robust=robust)
                    fit_chisq.append(poly.stats.chi2)
                    fit_rchisq.append(poly.stats.rchi2)

            # find the best atran correction by minimizing residuals
            if optimize and len(fit_chisq) > 0:
                atran_index = np.argmin(fit_rchisq)
                out_spectrum['fit_chisq'] = fit_chisq
                out_spectrum['fit_rchisq'] = fit_rchisq
                out_spectrum['all_corrected'] = all_corr
            else:
                # or else just take the first one
                atran_index = 0
            best_tran = all_tran[atran_index]
            best_shift = all_shift[atran_index]

            # shift images and spectra if needed
            dnames = ['spectral_flux', 'spectral_error',
                      'flux', 'error']
            if check_auto and np.abs(best_shift) > shift_limit:
                log.warning('Calculated shift of {:.2f} pixels is too large. '
                            'Not applying auto '
                            'shift.'.format(best_shift))
                best_shift = 0.
            elif 0 < np.abs(best_shift) < eps:
                log.debug('Wave shift of {:.2f} pixels is very small. '
                          'Setting to zero.'.format(best_shift))
                best_shift = 0.
            elif np.abs(best_shift) > eps:
                for name in dnames:
                    data = spectrum[name]
                    if data.ndim > 1:
                        offsets = [0., best_shift]
                    else:
                        offsets = [best_shift]
                    out_spectrum[name] = shift(data, offsets)

            # copy over data if not already there
            for name in dnames:
                if name not in out_spectrum:
                    out_spectrum[name] = spectrum[name].copy()

            # divide spectral flux and image by transmission
            # (no error to propagate)
            out_spectrum['spectral_flux'] /= best_tran
            out_spectrum['spectral_error'] /= best_tran

            # for the image: divide each row by the correction factor
            out_spectrum['flux'] /= best_tran[None, :]
            out_spectrum['error'] /= best_tran[None, :]

            # divide by response as well, propagating statistical errors
            out_spectrum['spectral_flux'] /= rmatch
            out_spectrum['flux'] /= rmatch[None, :]

            sflx = out_spectrum['spectral_flux']
            serr = out_spectrum['spectral_error']

            out_spectrum['spectral_error'] = \
                np.sqrt(serr**2 + ematch**2 * sflx**2) / rmatch

            sflx = out_spectrum['flux']
            serr = out_spectrum['error']
            out_spectrum['error'] = \
                np.sqrt(serr**2
                        + ematch[None, :]**2 * sflx**2) / rmatch[None, :]

            # append the transmission and response for reference
            out_spectrum['wave_shift'] = best_shift
            out_spectrum['atran_index'] = atran_index
            out_spectrum['transmission'] = best_tran
            out_spectrum['response'] = rmatch
            out_spectrum['response_error'] = ematch

            result[order].append(out_spectrum)

    return result
