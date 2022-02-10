# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions that may be used by multiple pipeline steps."""

from astropy import log
import numpy as np
import numba as nb

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['calchilo', 'readchop', 'readnod', 'readhwp', 'clipped_mean']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def calchilo(chop, nsamp, choptol, good):  # pragma: no cover
    """
    Tag data as high, low, or bad.

    Parameters
    ----------
    chop : array-like of float
        Chop or nod offset values.
    nsamp : int
        Number of samples in the chop array.
    choptol : float
        Tolerance for deviation from high or low value.
    good : array-like of int
        Flag value for each sample. Must match the dimension of `chop`.
        Updated in place.

    Returns
    -------
    chophi : float
        The high chop value.
    choplo : float
        The low chop value.
    chopeff : float
        The effective chop value.
    """
    # average middle value
    hi = 0.0
    nnan = 0
    for i in range(nsamp):
        if chop[i] == chop[i]:
            hi += chop[i]
        else:
            nnan += 1
    himean = hi / (nsamp - nnan)
    lomean = himean

    # initial values for hi / lomin / max
    himax = 1.0e30
    himin = himean
    lomin = -1.0e30
    lomax = himean

    # sigma clip for 10 iterations
    nhi = 0
    nlo = 0
    for i in range(10):
        hi = 0.0
        lo = 0.0
        hi2 = 0.0
        lo2 = 0.0
        nhi = 0
        nlo = 0
        for j in range(nsamp):
            if (chop[j] >= himin) and (chop[j] <= himax):
                hi += chop[j]
                hi2 += chop[j] * chop[j]
                nhi += 1

            if (chop[j] >= lomin) and (chop[j] <= lomax):
                lo += chop[j]
                lo2 += chop[j] * chop[j]
                nlo += 1
        himean = hi / nhi
        lomean = lo / nlo
        histd = np.sqrt((hi2 - hi * hi / nhi) / (nhi - 1))
        lostd = np.sqrt((lo2 - lo * lo / nlo) / (nlo - 1))
        if histd < choptol:
            histd = choptol
        if lostd < choptol:
            lostd = choptol
        himax = himean + histd
        himin = himean - histd
        lomax = lomean + lostd
        lomin = lomean - lostd

    chophi = himean
    choplo = lomean
    chopeff = (nhi + nlo) / (1.0 * nsamp)

    for i in range(nsamp):
        if np.abs(chop[i] - chophi) <= choptol:
            good[i] = 1
        elif np.abs(chop[i] - choplo) <= choptol:
            good[i] = -1
        else:
            good[i] = 0

    return chophi, choplo, chopeff


def readchop(step, nsamp, chop_tol, chopflag=True):
    """
    Read chop state into high, low, or not used values.

    Parameters
    ----------
    step : StepParent
        The calling pipe step, containing data loaded into
        step.praw.
    nsamp: int
        The number of samples in the input data.
    chop_tol : float
        Chopper tolerance in arcseconds.
    chopflag : bool, optional
        If not set, it is assumed that no chopping was performed.

    Returns
    -------
    array-like of int
        The chop state for each sample. 1 indicates high state,
        -1 indicates low, 0 indicates not used.
    """

    if chopflag:
        log.debug('Reading Chop signal to find high/low chop values')

        chopstate = np.zeros(nsamp, dtype=np.int32)
        chophi, choplo, chopeff = calchilo(
            step.praw['Chop Offset'], nsamp, chop_tol, chopstate)

        log.info("Chopper high,low,eff = %lf, %lf, %lf" %
                 (chophi, choplo, chopeff))
    else:
        log.warning('Assuming no chopping has occurred')
        chopstate = np.ones(nsamp, dtype=np.int32)

    return chopstate


def readhwp(step, nsamp, hwp_tol, sampfreq):
    """
    Determine HWP state for all samples.

    This function determines HWP angles and moving intervals.
    It also makes sure all HWP intervals have roughly the same
    length (short intervals are removed).

    The `hwp_tol` parameter is used to determine when the HWP is moving. If
    the HWP angle changes by more than this amount between two samples,
    then the code counts this as a division between HWP angles, and starts
    a new HWP position. Furthermore, a division will only be accounted
    for if the HWP angle stays in its new position for more than 2
    seconds, in order to avoid considering noisy peaks that might
    occur as a real change in the HWP angle. Set `hwp_tol` to a number
    greater than 360 to force all the data to be in the same HWP angle
    The recommended value is between 0.2 and 0.3 for discrete HWP
    and 400 for continuous HWP movement.

    The step.praw['HWP Index'] column is updated in place by this
    function: it is filled with the angle index found (starting at 0).

    Parameters
    ----------
    step : StepParent
        The calling pipe step, containing data loaded into
        step.praw.
    nsamp: int
        The number of samples in the input data.
    hwp_tol : float
        HWP angle tolerance in degrees.
    sampfreq : float
        The sampling frequency in Hz.

    Returns
    -------
    hwpstate : array-like
        The HWP state array contains values 1 (good) and 0 (bad).
    nhwp : int
        The number of HWP angles found.
    """
    # Setup Loop and run program
    hwpangle = step.praw['HWP Angle']
    hwpstate = np.zeros(nsamp, dtype=np.int32)
    step.praw['HWP Index'][:] = -1

    # sample indices for HWP position start
    posbeg = []
    # sample indices for HWP position end
    posend = []
    # lower limit for duration of a HWP position
    mintime = 1.0
    minsamp = int(round(mintime * sampfreq))
    # Temporary variable
    hwpangval = 0.0

    # Look for regions with hwpangle within 2*hwptol of the first value

    # current sample index
    sampi = 0
    while sampi < len(hwpangle):
        # Check if we're in or out of a HWP position
        if len(posbeg) == len(posend):
            # I.e. looking for next positon
            if sampi + minsamp >= len(hwpangle):
                # stop loop if a new HWP position would have too few samples
                break

            # Check if sample minsamp ahead has hwpangle within hwptol
            if abs(hwpangle[sampi]
                   - hwpangle[sampi + minsamp]) < 1.0 * hwp_tol:
                # Found a valid HWP position start
                posbeg.append(sampi)
                hwpangval = hwpangle[sampi]
                sampi += minsamp
            else:
                # No start found, increase sampi
                sampi += 1
        else:
            # I.e. looking for end of current HWP position
            # Check if current sample is within 2*hwptol of start sample
            if abs(hwpangval - hwpangle[sampi]) > 2.0 * hwp_tol:
                posend.append(sampi)
            sampi += 1

    # Finish posend list if necessary
    if len(posbeg) > len(posend):
        posend.append(len(hwpangle))

    # Trim points to make sure all is within hwptol of median
    # Loop over HWP positions
    for posi in range(len(posbeg)):
        hwpangval = np.median(hwpangle[posbeg[posi]:posend[posi]])
        # Trim at start
        while abs(hwpangle[posbeg[posi]] - hwpangval) > hwp_tol:
            posbeg[posi] += 1
        # Trim at end
        while abs(hwpangle[posend[posi] - 1] - hwpangval) > hwp_tol:
            posend[posi] -= 1

    # Trim points at both ends to make sure all are within hwptol/2
    # of median of the first/last minsamp samples. This lowers the
    # number of bad samples at both ends.
    for posi in range(len(posbeg)):
        # Trim at start
        hwpangval = np.median(hwpangle[posbeg[posi]:posbeg[posi] + minsamp])
        while abs(hwpangle[posbeg[posi]] - hwpangval) > hwp_tol / 2.0:
            posbeg[posi] += 1
        # Trim at end
        hwpangval = np.median(hwpangle[posend[posi] - minsamp:posend[posi]])
        while abs(hwpangle[posend[posi] - 1] - hwpangval) > hwp_tol / 2.0:
            posend[posi] -= 1

    # Remove HWP angle positions that last less than half the median duration
    poslen = [posend[i] - posbeg[i] for i in range(len(posbeg))]
    lenmed = np.median(poslen)

    # countdown to avoid index messup
    for posi in range(len(posbeg) - 1, -1, -1):
        if poslen[posi] < lenmed / 2:
            del posbeg[posi]
            del posend[posi]

    # Set hwpstate
    for posi in range(len(posbeg)):
        hwpstate[posbeg[posi]:posend[posi]] = 1
        step.praw['HWP Index'][posbeg[posi]:posend[posi]] = posi

    # Return
    nhwp = len(posbeg)
    s = "Found %d HWP angles:" % nhwp
    if nhwp < 10:
        for posi in range(len(posbeg)):
            s += ' %.1f' % np.median(hwpangle[posbeg[posi]:posend[posi]])
    log.info(s)

    return hwpstate, nhwp


def readnod(step, nsamp, nod_tol, nodflag=True):
    """
    Read nod state into high, low, or not used values.

    Parameters
    ----------
    step : StepParent
        The calling pipe step, containing data loaded into
        step.praw.
    nsamp: int
        The number of samples in the input data.
    nod_tol : float
        Nodding tolerance in arcseconds.
    nodflag : bool, optional
        If not set, it is assumed that no nodding was performed.

    Returns
    -------
    array-like of int
        The nod state for each sample. 1 indicates high state,
        -1 indicates low, 0 indicates not used.
    """
    if nodflag is True:
        log.debug('Reading Nod signal to find high/low nod values')

        nodstate = np.zeros(nsamp, dtype=np.int32)
        nodhi, nodlo, nodeff = calchilo(
            step.praw['Nod Offset'], nsamp, nod_tol, nodstate)

        log.info("Nod high,low,eff = %lf, %lf, %lf" %
                 (nodhi, nodlo, nodeff))
    else:
        log.warning('Assuming no nodding has occurred')
        nodstate = np.ones(nsamp, dtype=np.int32)

    return nodstate


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def clipped_mean(data, mask, sigma=5.0):  # pragma: no cover
    """
    Compute a sigma-clipped mean of the input data.

    Mean is computed along the zeroth axis. Data is assumed
    to have three dimensions.

    Parameters
    ----------
    data : array-like of float
        Input data.  Must have three dimensions.
    mask : array-like of int
        Mask for bad values in input data.  Must match data
        dimensions.  0 indicates good value, 1 indicates bad.
        Updated in place.
    sigma : float, optional
        The sigma-clipping threshold.

    Returns
    -------
    datamean : array-like
        The mean array.
    datastd : array-like
        The standard deviation array.
    """
    nframe, nrow, ncol = data.shape
    datamean = np.zeros((nrow, ncol), dtype=np.float64)
    datastd = np.zeros((nrow, ncol), dtype=np.float64)
    for i in range(nrow):
        for j in range(ncol):
            sumx = 0.
            sumx2 = 0.
            sumn = 0
            for k in range(nframe):
                if mask[k, i, j] == 0:
                    sumx += data[k, i, j]
                    sumx2 += data[k, i, j] * data[k, i, j]
                    sumn += 1
            total = sumn + 1
            mean = 0
            std = 0
            while sumn < total:
                if sumn > 1:
                    mean = sumx / sumn
                    std = np.sqrt((sumx2
                                   - sumx * sumx
                                   / (1.0 * sumn)) / (1.0 * sumn - 1))
                elif sumn <= 1:
                    break

                total = sumn
                sumx = 0.
                sumx2 = 0.
                sumn = 0
                for k in range(nframe):
                    if np.abs(data[k, i, j] - mean) <= sigma * std:
                        sumx += data[k, i, j]
                        sumx2 += data[k, i, j] * data[k, i, j]
                        sumn += 1
                    else:
                        mask[k, i, j] = 1

            datamean[i, j] = mean
            datastd[i, j] = std / np.sqrt(total)
    return datamean, datastd
