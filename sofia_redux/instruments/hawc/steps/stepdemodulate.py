# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Chop demodulation pipeline step."""

import math
import os

from astropy import log
from astropy.io import fits
import numpy as np
from scipy.signal import fftconvolve as convolve

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc.steps.basehawc import clipped_mean
from sofia_redux.instruments.hawc.steps import basehawc

__all__ = ['StepDemodulate']


class StepDemodulate(StepParent):
    """
    Demodulate chops for chopped and nodded data.

    This step subtracts low chops from high chops by weighting
    all chopped data with a sine wave, then summing. A phase correction
    is computed from the real and imaginary parts of the chop offset
    and is applied after the demodulation. Incomplete first
    and last chops are excluded. Data exceeding defined tolerances,
    and data for which the HWP was moving, are flagged for later
    removal in the Chop Mask column. Variances on the flux are also
    computed during demodulation, and propagated throughout all pipeline
    steps.

    The input data expected is a DataFits containing one table with
    columns for R Array, T array, Chop Offset (arcsec), Nod Offset (arcsec),
    HWP Angle (degrees), Azimuth (degrees), Elevation (degrees),
    Azimuth Error (arcsec), Elevation Error (arcsec), and Array VPA
    (degrees). In addition, the Primary HDU header must contain the
    following keywords: NHWP, SMPLFREQ, CHPFREQ, NODDING, NODPATT.

    This input is typically produced by the
    `sofia_redux.instruments.hawc.steps.StepPrepare` pipeline step,
    from raw HAWC data.

    This step outputs one table containing the demodulated data. The
    columns are the same as for input data, with some additional columns.
    Each row in the table corresponds to the value for that demodulated
    chop cycle.

    The additional columns are:

    - Samples: number of raw samples that make up a chop cycle
    - R array VAR: variance on the demodulated data for the R array
    - T array VAR: variance on the demodulated data for the T array
    - R array Imag: imaginary part of the demodulated data for the R array
    - T array Imag: imaginary part the demodulated data for the T array
    - R array Imag VAR: variance on the imaginary part of the
      demodulated data for the R array
    - T array Imag VAR: variance on the imaginary part of the
      demodulated data for the T array
    - Nod Index : integer-valued (0, 1, 2 for L, R, L)
    - HWP Index : integer-valued (0, 1, 2, 3)
    - Chop Mask : integer-valued, with bits for each flag type

    The flag bits for the Chop Mask column are:

    - bit 0: incomplete chop
    - bit 1: check on azelstate
    - bit 2: HWP moving
    - bit 3: check on nodstate
    - bit 4: spare
    - bit 5: spare
    - bit 6: LOS rewind
    - bit 7: tracking errors too high
    - bit 8: CentroidExpMsec below threshold
    - bit 9: extra samples before and after tracking tolerance violation

    The EXPTIME keyword in the primary header is updated to account only
    for the on-source time, rejecting samples that
    were excluded by the algorithm (when the HWP or telescope are
    moving). If chopping on-chip (header keyword CHPONFPA = True), then
    EXPTIME is multiplied by 2.

    The Configuration HDU present in raw data will be propagated to
    the demodulated file, but will not be carried along in subsequent
    steps.

    Depending on results from tracking tolerance, the keyword
    “TRCKSTAT” will be added to the header with one of the following
    values: ‘NONE - TrackErrAoi3 and 4 not found’, ‘GOOD - less than 1/3
    of samples removed’, ‘BAD - more than 1/3 of samples removed’, or
    ‘NONE - tracking tolerance is deactivated’.
    """
    def __init__(self):
        # placeholders for additional useful attributes
        self.column_names = []
        self.extracols = []
        self.praw = {}
        self.pdemod = {}
        self.format_names = []
        self.dim_names = []
        self.unit_names = []

        # call the parent init: will call setup
        super().__init__()

    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'demodulate', and are named with
        the step abbreviation 'DMD'.

        Parameters defined for this step are:

        chop_tol : float
            Chop tolerance in arcsec. Used in determining high/low state.
        nod_tol : float
            Nod tolerance in arcsec. Used in determining high/low state.
        hwp_tol : float
            HWP tolerance in degrees. Used in determining angle
            and moving state.
        az_tol : float
            Azimuth error tolerance in arcsec.
        el_tol : float
            Elevation error tolerance in arcsec.
        track_tol : float or str
            Tracking error tolerance in arcsec, as recorded by AOI 3
            or AOI 4. Set negative to turn off flagging for tracking
            errors. Set to the string 'beam' to use a default beam-size
            value appropriate to the observed filter band. Set to
            'centroidexp' to use the the CentroidExpMsec column to
            flag bad data instead of TrackErrAoi3 and TrackErrAoi4.
        track_extra : list of float
            Extra samples removed (in seconds) before and
            after samples flagged by track_tol. Set to [0, 0] to
            disable.
        chopphase : bool
            If not set, chop phases are not corrected.
        checkhwp : bool
            If set, the NHWP keyword is not corrected from the number
            of angles actually found in the data, and a warning is
            issued. If not set, NHWP is corrected to the actual number
            of HWP angles.
        phasefile : str or float
            If a string, this should be set to a path to a file containing
            phase corrections for each pixel. If a float, then the value
            assigned is applied as the phase correction to every pixel. The
            value should be the phase delay in seconds.
        phaseoffset : float
            Phase offset to apply in addition to the per-pixel offsets
            assigned by the phasefile. This is specified in degrees, and
            may be determined from the plots produced by the
            StepDmdPlot pipeline step for CALMODE = INT_CAL files. Not
            applied if the phasefile parameter is not a file.
        l0method : {'RE', 'IM', 'ABS'}
            Method to normalize data: real component only, imaginary
            component only, or absolute value.
        boxfilter : float
            Time constant for a box filter to convolve with the data
            before demodulation. Set to -1 to use 1/CHPFREQ. Set to
            0 to disable.
        chopavg : bool
            If set, R and T arrays before demodulation will be averaged
            over the chop period and stored in 'R array AVG' and
            'T array AVG' columns in the output table.
        tracksampcut : float
            If the fraction of all samples removed due to tracking
            is larger than this number, then tracking status
            (TRCKSTAT) is set to BAD.
        data_sigma : float
            Value for sigma-clipping of detector data, used in determining
            variance values for the demodulated flux.
        """
        # Name of the pipeline reduction step
        self.name = 'demodulate'
        self.description = 'Demodulate All Chops'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'dmd'

        # Clear Parameter list
        self.paramlist = []

        # Column names in output table used for demodulation.
        # Additional columns in the input file are kept
        # track of in self.extracols[]
        # and treated by regular averaging
        self.column_names = ['Samples', 'R array', 'T array',
                             'Chop Offset', 'Nod Offset', 'HWP Angle',
                             'Azimuth', 'Azimuth Error', 'Elevation',
                             'Elevation Error', 'Array VPA',
                             'Nod Index', 'HWP Index']

        # List of columns not used by demodulation
        self.extracols = []

        # A dictionary with raw input data columns
        self.praw = {}

        # Append parameters
        self.paramlist.append(['chop_tol', 4.0,
                               'chopper tolerance in '
                               'arcseconds'])
        self.paramlist.append(['nod_tol', 2.0,
                               'nod tolerance in arcseconds'])
        self.paramlist.append(['hwp_tol', 2.0,
                               'hwp angle tolerance in degrees'])
        self.paramlist.append(['az_tol', 5.0,
                               'Azimuth error tolerance in arcseconds'])
        self.paramlist.append(['el_tol', 5.0,
                               'Elevation error tolerance in arcseconds'])
        self.paramlist.append(['track_tol', 'beam',
                               'Track error tolerance in '
                               'arcseconds (AOIs 3 and 4) - '
                               'set negative to deactivate'])
        self.paramlist.append(['track_extra', [0., 0.],
                               'Extra samples removed (in seconds) before '
                               'and after samples flagged by track_tol'])
        self.paramlist.append(['chopphase', True,
                               'Flag requiring chop phase correction'])
        self.paramlist.append(['checkhwp', True,
                               'Set to FALSE to avoid checking the expected '
                               'number of HWP angles'])
        self.paramlist.append(['phasefile', '0.0',
                               'Phase file information file'])
        self.paramlist.append(['phaseoffset', 0.0,
                               'Phase offset to apply to phasefile (degrees)'])
        self.paramlist.append(['l0method', 'RE',
                               'Method to normalize data: REal, '
                               'IMag, and ABSolute'])
        self.paramlist.append(['boxfilter', 0.0,
                               'Time constant for box hipass filter '
                               '(0.0=no filter, -1 for 1/CHPFREQ)'])
        self.paramlist.append(['chopavg', False,
                               'Flag to save chop averaged raw data'])
        self.paramlist.append(['tracksampcut', 0.333,
                               'If fraction of all samples removed due '
                               'to tracking is larger than this number, '
                               'then tracking status is BAD'])
        self.paramlist.append(['data_sigma', 5.0,
                               'Value for sigma-clipping of detector data'])

    def read_beam(self):
        """
        Read the beam size for the observed filter.

        Returns
        -------
        beam : float
            The beam size.
        waveband : str
            The filter band name.
        """
        beam = [5.0, 5.5, 9.0, 14.0, 19.0]
        waveband = self.datain.getheadval('spectel1')
        bands = ['A', 'B', 'C', 'D', 'E']
        try:
            idx = bands.index(waveband[-1])
        except (ValueError, IndexError):
            # waveband not in list
            msg = 'Cannot parse waveband: %s' % waveband
            log.error(msg)
            raise ValueError(msg)

        beam = beam[idx]
        return beam, waveband[-1]

    def makedemod(self, nchop):
        """
        Construct arrays to hold demodulated data.

        Arrays are stored in self.pdemod, a dictionary with keys
        as specified by self.column_names.

        Parameters
        ----------
        nchop : int
            Number of chop cycles.
        """
        self.pdemod = {}

        for c in self.column_names:
            if c in ['R array', 'T array']:
                self.pdemod[c] = np.zeros(
                    (nchop, self.praw['nrow'], self.praw['ncol']),
                    dtype=np.float64)
                for name in [' Imag', ' VAR', ' Imag VAR']:
                    cstr = c + name
                    self.pdemod[cstr] = np.zeros((nchop,
                                                  self.praw['nrow'],
                                                  self.praw['ncol']),
                                                 dtype=np.float64)
            elif c in ['Samples', 'Nod Index', 'HWP Index']:
                self.pdemod[c] = np.zeros(nchop, dtype=np.int32)
            else:
                self.pdemod[c] = np.zeros(nchop, dtype=np.float64)
        self.pdemod['Chop Offset Imag'] = np.zeros(nchop, dtype=np.float64)

        self.pdemod['nsamp'] = nchop
        self.pdemod['nrow'] = self.praw['nrow']
        self.pdemod['ncol'] = self.praw['ncol']

    def readdata(self):
        """
        Read raw data into memory.

        Data is stored in the self.praw dictionary, with keys set
        by self.column_names.
        """
        self.praw = {}

        # The following three are filled from the raw data, each index will
        # correspond to the name in column_names
        self.format_names = []
        self.dim_names = []
        self.unit_names = []
        rawnames = self.datain.table.names

        # Fill the format_ dim_ unit_names lists,
        # fill raw data into self.praw

        # Read raw table columns
        for colnam in self.column_names:
            if colnam in ['Samples', 'Nod Index', 'HWP Index']:
                # only exist in output data
                self.format_names.append('1J')
                self.dim_names.append(None)
                self.unit_names.append('integer')
            else:
                try:
                    idx = rawnames.index(colnam)
                    self.praw[colnam] = \
                        self.datain.table.field(colnam).astype(np.float64)
                    self.format_names.append(
                        self.datain.table.columns[idx].format)
                    self.dim_names.append(self.datain.table.columns[idx].dim)
                    self.unit_names.append(self.datain.table.columns[idx].unit)
                except ValueError:
                    msg = "Column '%s' not found in raw data!" % colnam
                    log.error(msg)
                    raise ValueError(msg)

        # add columns in raw data not used for demodulation.
        for ind, colnam in enumerate(rawnames):
            if colnam in self.column_names:
                # skip column names that already exist
                continue
            self.praw[colnam] = \
                self.datain.table.field(colnam).astype(np.float64)
            col = self.datain.table.columns[ind]
            self.extracols.append(colnam)
            self.column_names.append(colnam)
            self.format_names.append(col.format)
            self.dim_names.append(col.dim)
            self.unit_names.append(col.unit)

        # Add additional info to praw
        rarr = self.praw['R array']
        nsamp, nrow, ncol = rarr.shape
        self.praw['nsamp'] = nsamp
        self.praw['nrow'] = nrow
        self.praw['ncol'] = ncol
        self.praw['Samples'] = np.ones(nsamp, dtype=np.int32)
        self.praw['Nod Index'] = np.zeros(nsamp, dtype=np.int32)
        self.praw['HWP Index'] = np.zeros(nsamp, dtype=np.int32)

    def make_chop_phase_tags(self, chopstate, chopsamp, hwpstate, nodstate):
        """
        Create the choptag and phasetag arrays from chopstate.

        This function is intended for sine-wave demodulation and the
        outputs are to be used by avgdemodsin.

        Parameters
        ----------
        chopstate: array-like
            Array of -1, 0, +1 derived from chop signal.
        chopsamp: int
            Number of samples per chop.
        hwpstate: array-like
            Array of 0 (HWP moving), 1 (HWP stable) from HWP Angle.
        nodstate: array-like
            Array of -1, 0, +1 derived from Nod Offset signal.

        Returns
        -------
        nchops: int
            Number of chops.
        choptag: array-like
            Array of chop tags which number the chops.
        phasetag: array-like
            Array of integer phase, numbering samples within a chop.

        """
        ntot = len(chopstate)
        nchops = ntot // chopsamp

        chopcheck = np.where(chopstate == 0)[0]
        if len(chopcheck > 0):
            first_zero = chopcheck[0]
        else:
            zcheck = np.where((nodstate == 0) | (hwpstate == 0))[0]
            if len(zcheck) > 0:
                first_zero = zcheck[0]
            else:
                first_zero = 0
                nchops += 1
        first_one = np.where(chopstate[first_zero:] == 1)[0][0] + first_zero

        log.debug('initial nchops=%d, n_praw=%d' % (nchops, ntot))

        choptag = np.concatenate([i * np.ones(chopsamp, dtype=np.int32)
                                  for i in range(nchops + 1)])
        choptag = np.concatenate([-1 * np.ones(first_one, dtype=np.int32),
                                  choptag])
        choptag = choptag[:ntot]

        phasetag = np.concatenate([np.arange(chopsamp, dtype=np.int32)
                                   for _ in range(nchops + 1)])
        phasetag = np.concatenate([-1 * np.ones(first_one, dtype=np.int32),
                                   phasetag])
        phasetag = phasetag[:ntot]

        # Find where hwpstate transitions
        hwp_on = np.where(hwpstate == 1)[0][0]
        nod_on = np.where(nodstate != 0)[0][0]
        nodlast = nodstate[nod_on]
        nodval = -1
        pos = max(hwp_on, nod_on)
        self.praw['Nod Index'][:pos] = 0
        while pos < (ntot - 1):
            nodcheck = np.where(nodstate[pos:] == 0)[0]
            if len(nodcheck > 0):
                nod_off = nodcheck[0] + pos
            else:
                nod_off = ntot - 1
            hwpcheck = np.where(hwpstate[pos:] == 0)[0]
            if len(hwpcheck > 0):
                hwp_off = hwpcheck[0] + pos
            else:
                hwp_off = ntot - 1
            nodnext = np.where(nodstate[nod_off:] == -nodlast)[0]
            if len(nodnext > 0):
                next_nod = nodnext[0] + nod_off
            else:
                next_nod = ntot - 1
            hwpnext = np.where(hwpstate[hwp_off:] == 1)[0]
            if len(hwpnext > 0):
                next_hwp = hwpnext[0] + hwp_off
            else:
                next_hwp = ntot - 1
            if next_nod < next_hwp:
                nodval += 1
                nodlast = -1 * nodlast
                self.praw['Nod Index'][pos:next_nod] = nodval
                pos = next_nod
                log.debug(
                    'pos={}, nodval incre, nodval={}, '
                    'nodlast={}, next_nod={}, next_hwp={}'.format(
                        pos, nodval, nodlast, next_nod, next_hwp))
            else:
                nodval += 1
                self.praw['Nod Index'][pos:next_hwp] = nodval
                pos = next_hwp
                nodval = -1
                log.debug(
                    'pos={}, nodval reset, nodval={}, '
                    'nodlast={}, next_nod={}, next_hwp={}'.format(
                        pos, nodval, nodlast, next_nod, next_hwp))

        # Set nod index to -1 where nodstate or hwpstate indicate not settled
        self.praw['Nod Index'][(nodstate == 0) | (hwpstate == 0)] = -1

        lastphase = phasetag[-1]
        if lastphase != chopsamp - 1:
            # incomplete last chop
            choptag[-(lastphase + 1):] = -1
            phasetag[-(lastphase + 1):] = -1
            nchops = choptag[-(lastphase + 2)] + 1
            log.debug('incomplete last chop for %d samples, '
                      'nchops is %d' % (-(lastphase + 1), nchops))
        else:
            nchops = choptag[-1]
            log.debug('complete last chop, lastphase=%d, '
                      'nchops=%d' % (lastphase, nchops))

        # Make sure the arrays are contiguous
        choptag = np.ascontiguousarray(choptag, dtype=np.int32)
        phasetag = np.ascontiguousarray(phasetag, dtype=np.int32)
        self.praw['Nod Index'] = np.ascontiguousarray(self.praw['Nod Index'])
        self.praw['HWP Index'] = np.ascontiguousarray(self.praw['HWP Index'])

        return nchops, choptag, phasetag

    def avgdemodsin(self, choptag, phasetag, chopsamp, nchop, data_sigma):
        """
        Average data with sine-wave demodulation.

        Demodulated data is stored in self.pdemod.

        Parameters
        ----------
        choptag : array-like
             Array values specifying which chop each sample belongs to.
        phasetag : array-like
             Array of values specifying sample number in
             current nod and hwp angle.
        chopsamp : array-like
             Number of samples per chop cycle.
        nchop : int
             Number of chops.
        data_sigma : float
             Value for sigma-clipping of detector data, used in determining
             variance values for the demodulated flux.
        """

        # get raw data
        r_real = self.praw['R array'].astype(np.float64)
        t_real = self.praw['T array'].astype(np.float64)

        # first group by phasetag, nod index, and hwp index
        # for variance across chop position
        nodtag = self.praw["Nod Index"]
        hwptag = self.praw["HWP Index"]

        # get all unique phase values and the indices
        # to them in the phasetag array
        phase_vals, full_idx = \
            np.unique(np.array(list(zip(nodtag, hwptag, phasetag))),
                      axis=0, return_inverse=True)

        # invert the index for each phase value
        phase_idx = [[] for _ in range(max(full_idx) + 1)]
        for i, k in enumerate(full_idx):
            phase_idx[k].append(i)
        phase_idx = [np.array(idx) for idx in phase_idx]

        # average over each chop phase value within a
        # nod and HWP position
        rvar = np.zeros_like(r_real)
        tvar = np.zeros_like(t_real)
        for i, phase in enumerate(phase_vals):
            mask = np.zeros_like(r_real[phase_idx[i], :, :], dtype=np.uint8)
            rmeans, _ = clipped_mean(r_real[phase_idx[i], :, :],
                                     mask, data_sigma)
            tmeans, _ = clipped_mean(t_real[phase_idx[i], :, :],
                                     mask, data_sigma)

            rvar[phase_idx[i], :, :] = (r_real[phase_idx[i], :, :]
                                        - rmeans[None, :, :])**2
            tvar[phase_idx[i], :, :] = (t_real[phase_idx[i], :, :]
                                        - tmeans[None, :, :])**2

        # now demodulate with sine wave

        freq = 2 * np.pi / float(chopsamp)
        w_real = np.cos(phasetag * freq)
        w_imag = np.sin(phasetag * freq)

        # multiply data by sin and cosine weights with chop frequency
        r_imag = w_imag[:, None, None] * r_real
        r_real *= w_real[:, None, None]
        t_imag = w_imag[:, None, None] * t_real
        t_real *= w_real[:, None, None]

        # do the same for the variance arrays
        rivar = w_imag[:, None, None]**2 * rvar
        rvar *= w_real[:, None, None]**2
        tivar = w_imag[:, None, None]**2 * tvar
        tvar *= w_real[:, None, None]**2

        # group by choptag

        # get all unique chop values and the indices
        # to them in the choptag array, as well as the
        # count for each chop cycle
        chop_vals, full_idx, chop_cts = \
            np.unique(choptag, return_inverse=True, return_counts=True)

        # invert the index for each chop value
        chop_idx = [[] for _ in range(max(full_idx) + 1)]
        for i, k in enumerate(full_idx):
            chop_idx[k].append(i)
        chop_idx = [np.array(idx) for idx in chop_idx]

        # columns to directly chop average
        colnames = ['Nod Offset', 'HWP Angle', 'Azimuth',
                    'Azimuth Error', 'Elevation', 'Elevation Error',
                    'Array VPA', 'Nod Index', 'HWP Index']
        # real and imaginary averages
        realchop = w_real * self.praw['Chop Offset']
        imagchop = w_imag * self.praw['Chop Offset']

        # average over each chop cycle
        for i, chop in enumerate(chop_vals):
            # skip negative and too large values
            if chop < 0 or chop >= nchop:
                continue

            nsamp = chop_cts[i]

            # calculate and store the average values in the
            # demod arrays by chop
            self.pdemod['R array'][chop, :] = \
                np.sum(r_real[chop_idx[i], :, :], axis=0) / nsamp
            self.pdemod['R array Imag'][chop, :] = \
                np.sum(r_imag[chop_idx[i], :, :], axis=0) / nsamp
            self.pdemod['T array'][chop, :] = \
                np.sum(t_real[chop_idx[i], :, :], axis=0) / nsamp
            self.pdemod['T array Imag'][chop, :] = \
                np.sum(t_imag[chop_idx[i], :, :], axis=0) / nsamp

            self.pdemod['R array VAR'][chop, :] = \
                np.sum(rvar[chop_idx[i], :, :], axis=0) / nsamp**2
            self.pdemod['R array Imag VAR'][chop, :] = \
                np.sum(rivar[chop_idx[i], :, :], axis=0) / nsamp**2
            self.pdemod['T array VAR'][chop, :] = \
                np.sum(tvar[chop_idx[i], :, :], axis=0) / nsamp**2
            self.pdemod['T array Imag VAR'][chop, :] = \
                np.sum(tivar[chop_idx[i], :, :], axis=0) / nsamp**2

            # also store some other chop-averaged values
            self.pdemod['Samples'][chop] = chop_cts[i]
            self.pdemod['Chop Offset'][chop] = \
                np.sum(realchop[chop_idx[i]]) / nsamp
            self.pdemod['Chop Offset Imag'][chop] = \
                np.sum(imagchop[chop_idx[i]]) / nsamp
            for col in colnames:
                self.pdemod[col][chop] = \
                    np.sum(self.praw[col][chop_idx[i]]) / nsamp

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a single-in single-out (SISO) step:
        self.datain should be a DataFits object, and output will also
        be a single DataFits, stored in self.dataout.

        The process is:

        1. Read data and tag chop, nod, and HWP state for all
           samples.
        2. Convolve data with a box filter.
        3. Flag data for bad tracking states.
        4. Weight the chop data with a sine wave and average to
           demodulate the chops. Compute the variance across samples
           at the same time.
        5. Apply a phase correction to every pixel to correct for
           readout delays.
        6. Compute the magnitude of the demodulated signal (from
           real, imaginary, or absolute value).
        7. Store demodulated data in a new DataFits.

        """

        # Set up variables
        chop_tol = self.getarg('chop_tol')
        nod_tol = self.getarg('nod_tol')
        hwp_tol = self.getarg('hwp_tol')
        az_tol = self.getarg('az_tol')  # arcseconds
        el_tol = self.getarg('el_tol')  # arcseconds
        track_tol = self.getarg('track_tol')
        track_extra = self.getarg('track_extra')
        checkhwp = self.getarg('checkhwp')
        boxfilter = self.getarg('boxfilter')
        nhwpexpect = self.datain.getheadval("nhwp")
        sampfreq = self.datain.getheadval("smplfreq")
        chopfreq = self.datain.getheadval("chpfreq")
        choponfpa = self.datain.getheadval("chponfpa")  # T or F
        nodflag = self.datain.getheadval("nodding")
        tracksampcut = float(self.getarg('tracksampcut'))
        data_sigma = self.getarg('data_sigma')
        l0method = self.getarg('l0method')

        # get a user frequency for a couple default purposes
        try:
            user_freq = float(self.config['dmdplot']['user_freq'])
            log.debug('User frequency from dmdplot '
                      'params: %f' % user_freq)
        except (KeyError, ValueError):
            user_freq = 10.2
            log.debug('Default user frequency: %f' % user_freq)

        # Check chop frequency for legal values
        if chopfreq <= 0.:
            if l0method[:3] == 'ABS':
                # set to user value
                log.debug("Invalid chop frequency: {}. Using user "
                          "frequency instead: {}".format(chopfreq, user_freq))
                chopfreq = user_freq
            else:
                msg = "Invalid chop frequency: %f" % chopfreq
                log.error(msg)
                raise ValueError(msg)

        # If needed, get chop frequency for boxfilter
        if boxfilter < 0:
            boxfilter = chopfreq

        # Box highpass filter: subtract highpass filtered data
        if boxfilter > 0:
            datar = self.datain.table['R array']
            datat = self.datain.table['T array']

            # make convolution function size and shape
            dtime = 1.0 / self.datain.getheadval('SMPLFREQ')
            winwidth = int(round(1.0 / boxfilter / dtime))
            winsize = [winwidth]
            for i in range(1, datar.ndim):
                winsize.append(1)

            # make convolution function
            window = np.ones(winwidth) / winwidth
            window.shape = winsize

            # convolve data
            log.debug('Convolving with box filter')
            flatr = convolve(datar, window, mode='same')
            flatt = convolve(datat, window, mode='same')

            # subtract and save result
            self.datain.table['R array'] = datar - flatr
            self.datain.table['T array'] = datat - flatt
        else:
            flatr = None
            flatt = None

        # Read raw data and set up python (self.praw) arrays
        log.debug('Reading data into Python structures')
        self.readdata()
        nsamp = self.praw['nsamp']

        # Define the array for premask
        premask = np.zeros(nsamp, dtype=np.int32)
        # bit 0: incomplete chop
        # bit 1: check on azelstate
        # bit 2: HWP moving
        # bit 3: check on nodstate
        # bit 4: spare
        # bit 5: spare
        # bit 6: LOS rewind
        # bit 7: tracking errors too high
        # bit 8: CentroidExpMsec below threshold
        # bit 9: extra samples "before" and "after"
        #        tracking tolerance violation

        # Read, parse and check HWP angles
        hwpstate, nhwp = basehawc.readhwp(self, nsamp, hwp_tol, sampfreq)

        # Use checkhwp to determine if should
        # check the expected number of HWP angles
        # with the number actually found in the timestream (true).
        # If false, will ignore the checking. This is useful
        # if continuously rotating the HWP (in lab mode) or if
        # initial keyword NHWP is expected to be incorrect.
        if checkhwp:
            if nhwpexpect != nhwp:
                msg = "Expected %d HWP angles; found %d" % (nhwpexpect, nhwp)
                log.warning(msg)
        else:
            log.warning('Assigning NHWP keyword with the actual '
                        'number of HWP angles (%d) found' % nhwp)
            self.datain.setheadval('NHWP', nhwp, 'Number of HWP Angles')

        # Read and parse chop and nod signals
        chopstate = basehawc.readchop(self, nsamp, chop_tol)
        nodstate = basehawc.readnod(self, nsamp, nod_tol, nodflag)
        chopstaten = sum([1 for s in chopstate if s != 0])
        nodstaten = sum([1 for s in nodstate if s != 0])
        log.debug('Chop State: #good samples=%d' % chopstaten)
        log.debug('Nod State: #good samples=%d' % nodstaten)

        # Read az and el errors to determine good/bad
        log.debug('Reading Az/El errors to find when telescope is moving')
        azelstate = np.zeros(nsamp, dtype=np.int32)
        c1 = np.abs(self.praw['Azimuth Error']) <= az_tol
        c2 = np.abs(self.praw['Elevation Error']) <= el_tol
        azelstate[c1 & c2] = 1
        premask[azelstate == 0] += 2**1
        log.debug('Az/El State: #good samples=%d' % sum(azelstate))

        # Removing samples that occur during LOS rewinds (flag = 1)
        if 'Flag' in self.praw:
            # to avoid failing with old data where 'Flag' was not present
            flagstate = self.praw['Flag'].astype(np.int32)
            ind_losrewind = np.where(flagstate != 0)
            premask[ind_losrewind] += 2**6
            azelstate[ind_losrewind] = 0

        log.debug('Az/El after flagstate: #good samples=%d' %
                  sum(azelstate))

        # Removing samples if tracking errors are too high
        log.debug('Number of good samples before tracking '
                  'error correction: %d' % sum(azelstate))
        ngoodsampbefore = sum(azelstate)

        trcerr3 = None
        trcerr4 = None
        try:
            trcerr3 = self.praw['TrackErrAoi3'].astype(np.float64)
            trcerr4 = self.praw['TrackErrAoi4'].astype(np.float64)
            if np.isnan(trcerr3).any():
                log.warning('TrackErrAoi3 signal contains NaNs - '
                            'Assigning 999999 so these samples '
                            'can be flagged by track_tol')
                trcerr3[trcerr3 != trcerr3] = 999999.
            if np.isnan(trcerr4).any():
                log.warning('TrackErrAoi4 signal contains NaNs - '
                            'Assigning 999999 so these samples '
                            'can be flagged by track_tol')
                trcerr4[trcerr4 != trcerr4] = 999999.
        except (ValueError, KeyError):
            msg = 'TrackErrAoi3 and 4 tables not found. ' \
                  'Deactivating sample removal due to tracking errors.'
            log.warning(msg)
            # assigning negative track_tol will
            # deactivate sample removal (see below)
            track_tol = -1.

        if track_tol == 'beam':
            # Labeling bad samples based on trcerr3 and trcerr4
            beamsize, waveband = self.read_beam()
            indtrc = np.where((trcerr3 > beamsize) | (trcerr4 > beamsize))
            azelstate[indtrc] = np.int32(0)
            premask[indtrc] += 2**7
            log.debug('Removing bad samples, tracking issues - '
                      'using beam size of %s arcsec (Band %s)' %
                      (beamsize, waveband))
            log.debug('Number of good samples after removing samples '
                      'due to bad tracking = %d' % sum(azelstate))

            ngoodsampafter = sum(azelstate)
            if float(ngoodsampafter) / float(ngoodsampbefore) > \
                    (1.0 - tracksampcut):
                self.datain.setheadval('TRCKSTAT',
                                       'GOOD - less than %.3f of '
                                       'samples removed' % tracksampcut,
                                       'Tracking status (stepDemod)')
            elif float(ngoodsampafter) / float(ngoodsampbefore) <= \
                    (1.0 - tracksampcut):
                self.datain.setheadval('TRCKSTAT',
                                       'BAD - more than %.3f of samples '
                                       'removed' % tracksampcut,
                                       'Tracking status (stepDemod)')
                log.warning('Tracking status is BAD: more '
                            'than %.3f of samples removed' % tracksampcut)
        elif track_tol == 'centroidexp':
            # Look for centroidexp
            centroidexp = None
            try:
                centroidexp = self.praw['CentroidExpMsec'].astype(np.float64)
            except (KeyError, ValueError):
                msg = 'CentroidExpMsec not found. Deactivating sample ' \
                      'removal due to tracking errors.'
                log.warning(msg)

                # assigning negative track_tol will deactivate
                # sample removal (see below)
                track_tol = -1.
                self.datain.setheadval('TRCKSTAT',
                                       'NONE - CentroidExpMsec not found',
                                       'Tracking status (stepDemod)')

            if track_tol == 'centroidexp':
                # Labeling bad samples based on centroidexp
                badtrack = np.where(centroidexp < 10)
                azelstate[badtrack] = np.int32(0)
                premask[badtrack] += 2**8
                log.debug('Removing bad samples, tracking '
                          'issues - using centroidexp)' % centroidexp)
                log.debug('Number of good samples after removing '
                          'samples due to bad tracking = %d' %
                          sum(azelstate))

                ngoodsampafter = sum(azelstate)
                if float(ngoodsampafter) / float(ngoodsampbefore) > \
                        (1.0 - tracksampcut):
                    self.datain.setheadval('TRCKSTAT',
                                           'GOOD - less than %.3f of '
                                           'samples removed' % tracksampcut,
                                           'Tracking status (stepDemod)')
                elif float(ngoodsampafter) / float(ngoodsampbefore) <= \
                        (1.0 - tracksampcut):
                    self.datain.setheadval('TRCKSTAT',
                                           'BAD - more than %.3f of samples '
                                           'removed' % tracksampcut,
                                           'Tracking status (stepDemod)')
                    log.warning('Tracking status is BAD: more '
                                'than %.3f of samples removed' %
                                tracksampcut)

        else:
            try:
                track_tol = float(track_tol)
            except ValueError:
                msg = 'track_tol value is undefined: %s; ' \
                      'specify either "beam", a positive number (arcsec) ' \
                      'or a negative number (to deactivate)' % track_tol
                log.error(msg)
                raise ValueError(msg)
            if track_tol <= 0.:
                log.debug('Removal bad samples due to tracking '
                          'issues is deactivated (negative track_tol '
                          'equal to %s)' % track_tol)
                self.datain.setheadval('TRCKSTAT',
                                       'NONE - tracking tolerance is '
                                       'deactivated',
                                       'Tracking status (stepDemod)')
            else:
                indtrc = np.where(
                    (trcerr3 > track_tol) | (trcerr4 > track_tol))
                azelstate[indtrc] = np.int32(0)
                premask[indtrc] += 2**7
                log.debug('Removing bad samples, tracking issues - '
                          'using track_tol equal to %s arcsec' %
                          track_tol)
                log.debug('Number of good samples after removing '
                          'samples due to bad tracking = %d' %
                          sum(azelstate))
                ngoodsampafter = sum(azelstate)

                if float(ngoodsampafter) / float(ngoodsampbefore) > \
                        (1.0 - tracksampcut):
                    self.datain.setheadval('TRCKSTAT',
                                           'GOOD - less than %.3f of samples '
                                           'removed' % tracksampcut,
                                           'Tracking status (stepDemod)')
                else:
                    self.datain.setheadval('TRCKSTAT',
                                           'BAD - more than %.3f of samples '
                                           'removed' % tracksampcut,
                                           'Tracking status (stepDemod)')
                    log.warning('Tracking status is BAD: more than %.3f of '
                                'samples removed' % tracksampcut)

        # Extra tracking error correction
        if track_tol == 'beam' or track_tol == 'centroidexp' or track_tol > 0.:
            track_extra_before = int(sampfreq * track_extra[0])
            track_extra_after = int(sampfreq * track_extra[1])
            # at least one positive, neither negative
            if (track_extra_before > 0 or track_extra_after > 0) and not \
                    (track_extra_before < 0 or track_extra_after < 0):
                ref_azelstate = np.empty_like(azelstate)
                np.copyto(ref_azelstate, azelstate)
                for i in range(nsamp - 1):
                    samp1 = ref_azelstate[i]
                    samp2 = ref_azelstate[i + 1]
                    if (samp1 - samp2) == 1:
                        beg = max(0, i - track_extra_before)
                        end = i
                        azelstate[beg:end] = np.int32(0)
                        premask[beg:end] += 2**9
                    elif (samp1 - samp2) == -1:
                        beg = i
                        end = min(i + track_extra_after, nsamp - 1)
                        azelstate[beg:end] = np.int32(0)
                        premask[beg:end] += 2**9
                    else:
                        pass

                log.debug('Number of good samples after EXTRA removal '
                          'of bad samples due to bad tracking '
                          '(track_extra) = %d' % sum(azelstate))

                ngoodsampafter = sum(azelstate)
                if float(ngoodsampafter) / float(ngoodsampbefore) > \
                        (1.0 - tracksampcut):
                    self.datain.setheadval('TRCKSTAT',
                                           'GOOD - less than %.3f of samples '
                                           'removed' % tracksampcut,
                                           'Tracking status (stepDemod)')
                elif float(ngoodsampafter) / float(ngoodsampbefore) \
                        <= (1.0 - tracksampcut):
                    self.datain.setheadval('TRCKSTAT',
                                           'BAD - more than %.3f of samples '
                                           'removed' % tracksampcut,
                                           'Tracking status (stepDemod)')
                    log.warning('Tracking status is BAD: more '
                                'than %.3f of samples removed' %
                                tracksampcut)
            else:
                log.debug('Extra removal of bad samples is '
                          'deactivated because values in '
                          'track_extra are <= 0.')
        else:
            log.debug('Extra removal of bad samples is deactivated '
                      'because track_tol is not activated')

        # Read Phase file: Set arrays, get angles, reformat
        # - FLOAT ARR rphases = np.arange(0, 1, 0.1).astype(np.float64)
        rphase = np.ones((self.praw['nrow'],
                          self.praw['ncol'])).astype(np.float64)
        tphase = np.ones((self.praw['nrow'],
                          self.praw['ncol'])).astype(np.float64)
        phasefile = os.path.expandvars(str(self.getarg('phasefile')))
        phaseoffset = self.getarg('phaseoffset')
        aux = False
        if os.path.isfile(phasefile):
            # If it's a file, read it as such
            aux = True
            phasedata = DataFits(config=self.config)
            phasedata.load(phasefile)
            rphase = phasedata.imgdata[0]
            tphase = phasedata.imgdata[1]

            # add in phaseoffset
            log.debug('Applying phase offset: %f / 360. / %f sec' %
                      (phaseoffset, user_freq))
            offset_sec = phaseoffset / 360. / user_freq
            rphase += offset_sec
            tphase += offset_sec
        else:
            # Try read as number
            log.debug('Could not load <%s> as a phasefile' % phasefile)
            try:
                phasefloat = float(phasefile)
            except ValueError:
                log.warning("Could not set phasefile <%s> "
                            "setting phase=0" % phasefile)
                phasefile = "Invalid Phase File"
                phasefloat = 0.0
            log.debug('Applying a phase correction of <%s>' % phasefloat)
            rphase *= phasefloat
            tphase *= phasefloat

        # Finally, demodulate chop
        log.info('Running Sine-wave demodulation...')

        # number of samples in chop period is chopsamp
        chopsamp = int(round(sampfreq / (1. * chopfreq)))
        log.debug("Number of samples in chop "
                  "period = %d" % chopsamp)
        log.debug('Demodulating chop signal to count '
                  'number of chop cycles')

        nchops, choptag, phasetag = \
            self.make_chop_phase_tags(chopstate, chopsamp,
                                      hwpstate, nodstate)
        premask[hwpstate != 1] += 2**2
        premask[azelstate != 1] += 2**1
        premask[nodstate == 0] += 2**3
        premask[phasetag == -1] += 2**0

        log.info('Found %d chop cycles' % nchops)
        if nchops <= 0:
            msg = "Invalid number of chops (%d) found" % nchops
            log.error(msg)
            raise ValueError(msg)

        # allocate space
        self.makedemod(nchops)

        # do sine demodulation
        log.debug('Averaging values in each chop using '
                  'sine/cosine waves')
        log.debug('Calculating variance across chops at '
                  'common nod/HWP positions')
        self.avgdemodsin(choptag, phasetag, chopsamp, nchops,
                         data_sigma)

        # compute chopmask (may need speeding up)
        log.debug('Making chopmask, averaging columns')
        chopmask = np.zeros(nchops, dtype=np.int32)
        for i in range(nchops):
            chopmask[i] = np.bitwise_or.reduce(premask[choptag == i])

            # also average extracols, while we're iterating over nchop
            for name in self.extracols:
                self.pdemod[name][i] = np.mean(self.praw[name][choptag == i])

        # make chop phase correction
        if self.getarg('chopphase'):
            log.debug('Doing Chop Phase Correction')
            mag = np.sqrt(self.pdemod['Chop Offset'] ** 2
                          + self.pdemod['Chop Offset Imag'] ** 2)

            # to avoid Nan
            mag[np.where(mag == 0.0)] = 1.0
            phasereal = self.pdemod['Chop Offset'] / mag
            phaseimag = -self.pdemod['Chop Offset Imag'] / mag
            phasereal = phasereal.reshape((phasereal.shape[0], 1, 1))
            phaseimag = phaseimag.reshape((phaseimag.shape[0], 1, 1))
            self.pdemod['Chop Offset'] = mag
            tmp = \
                self.pdemod['R array'] * phasereal \
                - self.pdemod['R array Imag'] * phaseimag
            self.pdemod['R array Imag'] = \
                self.pdemod['R array'] * phaseimag \
                + self.pdemod['R array Imag'] * phasereal
            self.pdemod['R array'] = tmp

            tmp = \
                self.pdemod['T array'] * phasereal \
                - self.pdemod['T array Imag'] * phaseimag
            self.pdemod['T array Imag'] = \
                self.pdemod['T array'] * phaseimag \
                + self.pdemod['T array Imag'] * phasereal
            self.pdemod['T array'] = tmp

            phase = np.arctan2(phaseimag, phasereal)

            # propagate variance
            # (normalized correlation coefficient for r and i is 1)
            # qr = r * pr - i * pi
            # Vqr = pr^2 * Vr + pi^2 * Vi - 2 * pr * pi sqrt(Vr * Vi)
            # qi = r * pi + i * pr
            # Vqi = pi^2 * Vr + pr^2 * Vi + 2 * pr * pi sqrt(Vr * Vi)
            tmp = \
                self.pdemod['R array VAR'] * phasereal**2 \
                + self.pdemod['R array Imag VAR'] * phaseimag**2 \
                - 2 * phasereal * phaseimag \
                * np.sqrt(self.pdemod['R array VAR']
                          * self.pdemod['R array Imag VAR'])
            self.pdemod['R array Imag VAR'] = \
                self.pdemod['R array VAR'] * phaseimag ** 2 \
                + self.pdemod['R array Imag VAR'] * phasereal ** 2 \
                + 2 * phasereal * phaseimag \
                * np.sqrt(self.pdemod['R array VAR']
                          * self.pdemod['R array Imag VAR'])
            self.pdemod['R array VAR'] = tmp

            tmp = \
                self.pdemod['T array VAR'] * phasereal**2 \
                + self.pdemod['T array Imag VAR'] * phaseimag**2 \
                - 2 * phasereal * phaseimag \
                * np.sqrt(self.pdemod['T array VAR']
                          * self.pdemod['T array Imag VAR'])
            self.pdemod['T array Imag VAR'] = \
                self.pdemod['T array VAR'] * phaseimag ** 2 \
                + self.pdemod['T array Imag VAR'] * phasereal ** 2 \
                + 2 * phasereal * phaseimag \
                * np.sqrt(self.pdemod['T array VAR']
                          * self.pdemod['T array Imag VAR'])
            self.pdemod['T array VAR'] = tmp
        else:
            phase = np.zeros(nchops, dtype=np.float64)

        # Do pixel phase correction

        # R phase values
        rphase = 2. * math.pi * rphase * chopfreq
        phasereal = np.cos(rphase).reshape(
            1, self.praw['nrow'], self.praw['ncol'])
        phaseimag = np.sin(rphase).reshape(
            1, self.praw['nrow'], self.praw['ncol'])

        # correct R data
        tmp = \
            self.pdemod['R array'] * phasereal - \
            self.pdemod['R array Imag'] * phaseimag
        self.pdemod['R array Imag'] = \
            self.pdemod['R array'] * phaseimag + \
            self.pdemod['R array Imag'] * phasereal
        self.pdemod['R array'] = tmp

        # propagate R variance
        # (normalized correlation coefficient for R and I is still 1)
        tmp = \
            self.pdemod['R array VAR'] * phasereal ** 2 \
            + self.pdemod['R array Imag VAR'] * phaseimag ** 2 \
            - 2 * phasereal * phaseimag \
            * np.sqrt(self.pdemod['R array VAR']
                      * self.pdemod['R array Imag VAR'])
        self.pdemod['R array Imag VAR'] = \
            self.pdemod['R array VAR'] * phaseimag ** 2 \
            + self.pdemod['R array Imag VAR'] * phasereal ** 2 \
            + 2 * phasereal * phaseimag \
            * np.sqrt(self.pdemod['R array VAR']
                      * self.pdemod['R array Imag VAR'])
        self.pdemod['R array VAR'] = tmp

        # now T phase values
        tphase = 2. * math.pi * tphase * chopfreq
        phasereal = np.cos(tphase).reshape(
            1, self.praw['nrow'], self.praw['ncol'])
        phaseimag = np.sin(tphase).reshape(
            1, self.praw['nrow'], self.praw['ncol'])

        # correct T data
        tmp = \
            self.pdemod['T array'] * phasereal \
            - self.pdemod['T array Imag'] * phaseimag
        self.pdemod['T array Imag'] = \
            self.pdemod['T array'] * phaseimag \
            + self.pdemod['T array Imag'] * phasereal
        self.pdemod['T array'] = tmp

        # propagate T variance
        tmp = \
            self.pdemod['T array VAR'] * phasereal ** 2 \
            + self.pdemod['T array Imag VAR'] * phaseimag ** 2 \
            - 2 * phasereal * phaseimag \
            * np.sqrt(self.pdemod['T array VAR']
                      * self.pdemod['T array Imag VAR'])
        self.pdemod['T array Imag VAR'] = \
            self.pdemod['T array VAR'] * phaseimag ** 2 \
            + self.pdemod['T array Imag VAR'] * phasereal ** 2 \
            + 2 * phasereal * phaseimag \
            * np.sqrt(self.pdemod['T array VAR']
                      * self.pdemod['T array Imag VAR'])
        self.pdemod['T array VAR'] = tmp

        # Do L0 method correction -> Result in R array and T array
        if l0method[:3] == 'ABS':
            rr = self.pdemod['R array']
            ri = self.pdemod['R array Imag']
            tr = self.pdemod['T array']
            ti = self.pdemod['T array Imag']
            self.pdemod['R array'] = np.sqrt(rr**2 + ri**2)
            self.pdemod['T array'] = np.sqrt(tr**2 + ti**2)

            # propagate variance
            # q = (r^2 + i^2)^(1/2)
            # Vq = 1/q^2 * (r^2 Vr + i^2 Vi + 2 r i sqrt(Vr Vi))
            qr = self.pdemod['R array']
            qt = self.pdemod['T array']
            rrv = self.pdemod['R array VAR']
            riv = self.pdemod['R array Imag VAR']
            trv = self.pdemod['T array VAR']
            tiv = self.pdemod['T array Imag VAR']
            self.pdemod['R array VAR'] = \
                (1 / qr**2) * (rrv * rr**2 + riv * ri**2
                               + 2 * rr * ri * np.sqrt(rrv * riv))
            self.pdemod['T array VAR'] = \
                (1 / qt**2) * (trv * tr**2 + tiv * ti**2
                               + 2 * tr * ti * np.sqrt(trv * tiv))
        elif l0method[:2] == 'IM':
            self.pdemod['R array'] = self.pdemod['R array Imag']
            self.pdemod['T array'] = self.pdemod['T array Imag']
            self.pdemod['R array VAR'] = self.pdemod['R array Imag VAR']
            self.pdemod['T array VAR'] = self.pdemod['T array Imag VAR']
        elif l0method[:2] != 'RE':
            log.warning('Sine demod: unknown l0method=%s' % l0method)

        # Calculating the effective EXPTIME (on-source) based
        # on the samples actually used
        # (the samples for which choptag <= 0 where excluded)
        ind_goodsamp = np.where(choptag >= 0)
        goodsamp = choptag[ind_goodsamp]
        ngoodsamp = len(goodsamp)
        if choponfpa:
            # chopping on-chip (double the exposure time)
            exptime = 2.0 * (ngoodsamp / 2.0) / sampfreq
        else:
            # chopping off-chip
            exptime = (ngoodsamp / 2.0) / sampfreq

        # make output columns of demodulated data
        cols = []
        for c, f, d, u in zip(self.column_names, self.format_names,
                              self.dim_names, self.unit_names):
            log.debug("Appending column with name=%s" % c)
            junk = fits.Column(name=c, format=f, dim=d,
                               unit=u, array=self.pdemod[c])
            cols.append(junk)

        for name in ['R array', 'T array', 'Chop Offset']:
            # add in Imag columns
            log.debug("Appending column with name=%s Imag" % name)
            idx = self.column_names.index(name)
            col_to_add = fits.Column(name="%s Imag" % name,
                                     format=self.format_names[idx],
                                     dim=self.dim_names[idx],
                                     unit=self.unit_names[idx],
                                     array=self.pdemod['%s Imag' % name])
            cols.append(col_to_add)

        # add phase and chopmask column for demod mode
        log.debug("Appending column with name=Phase Corr")
        col_to_add = fits.Column(name="Phase Corr", format='1E',
                                 unit='Radians', array=phase)
        cols.append(col_to_add)
        log.debug("Appending column with name=Chop Mask")
        col_to_add = fits.Column(name="Chop Mask", format='1J',
                                 unit='', array=chopmask)
        cols.append(col_to_add)

        for name in ['R array', 'T array']:
            # add in VAR columns
            log.debug("Appending column with name=%s VAR" % name)
            idx = self.column_names.index(name)
            col_to_add = fits.Column(name="%s VAR" % name,
                                     format=self.format_names[idx],
                                     dim=self.dim_names[idx],
                                     unit=self.unit_names[idx] + '^2',
                                     array=self.pdemod['%s VAR' % name])
            cols.append(col_to_add)
            col_to_add = fits.Column(name="%s Imag VAR" % name,
                                     format=self.format_names[idx],
                                     dim=self.dim_names[idx],
                                     unit=self.unit_names[idx] + '^2',
                                     array=self.pdemod['%s Imag VAR' % name])
            cols.append(col_to_add)

        # Add column with chop averaged raw data (if requested)
        if self.getarg("chopavg") and boxfilter:
            # Make Arrays
            log.debug("Making Chop Averaged Arrays")
            chopavgr = np.empty((nchops,
                                 self.praw['nrow'], self.praw['ncol']))
            chopavgt = np.empty((nchops,
                                 self.praw['nrow'], self.praw['ncol']))

            # Fill Arrays (choptag is int array, increases from 0 to nchop-1)
            # limits of chop in raw samples
            indbeg, indend = 0, 0
            # index of current chop
            chopind = 0
            while chopind < nchops - 1 and indbeg < nsamp:
                # search for next chop start
                indbeg = indend
                while indbeg < nsamp and choptag[indbeg] <= chopind:
                    indbeg += 1
                if indbeg < nsamp:
                    # search for next chop end
                    indend = indbeg
                    # safer than chopind+=1 in case chops are missing
                    chopind = choptag[indbeg]
                    while indend < nsamp and choptag[indend] == chopind:
                        indend += 1

                    # average data
                    chopavgr[chopind, ...] = np.average(
                        flatr[indbeg:indend, ...], axis=0)
                    chopavgt[chopind, ...] = np.average(
                        flatt[indbeg:indend, ...], axis=0)

            # Add to columns
            newcol = fits.Column(name='R array AVG',
                                 format='%dE' % (self.praw['nrow']
                                                 * self.praw['ncol']),
                                 dim='(%d,%d)' % (self.praw['ncol'],
                                                  self.praw['nrow']),
                                 unit='counts', array=chopavgr)
            cols.append(newcol)
            newcol = fits.Column(name='T array AVG',
                                 format='%dE' % (self.praw['nrow']
                                                 * self.praw['ncol']),
                                 dim='(%d,%d)' % (self.praw['ncol'],
                                                  self.praw['nrow']),
                                 unit='counts', array=chopavgt)
            cols.append(newcol)

        # Make output data
        log.debug("Storing output data")
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))

        self.dataout = DataFits(config=self.datain.config)
        self.dataout.filename = self.datain.filename
        self.dataout.imageset(None, self.datain.imgnames[0],
                              self.datain.header)
        self.dataout.tableset(tbhdu.data, 'DEMODULATED DATA', tbhdu.header)

        # Updating the EXPTIME keyword value in the header
        self.dataout.setheadval('EXPTIME', exptime,
                                'On-source exposure time [s]')

        # Update SOFIA mandated keywords (since this is first pipe step)
        obsid = 'P_' + self.dataout.getheadval('OBS_ID')
        self.dataout.setheadval('OBS_ID', obsid)
        self.dataout.setheadval('PROCSTAT', 'LEVEL_1')
        self.dataout.setheadval('PIPELINE', 'HAWC_DRP')
        try:
            self.dataout.setheadval('ASSC_AOR',
                                    self.dataout.getheadval('AOR_ID'),
                                    'Associated AORs')
        except KeyError:
            pass
        try:
            self.dataout.setheadval('ASSC_MSN',
                                    self.dataout.getheadval('MISSN-ID'),
                                    'Associated Mission IDs')
        except KeyError:
            pass

        # Add auxiliary file to history
        if aux:
            phasebasename = phasefile.split(str(self.dataout.data_path))[-1]
            self.dataout.setheadval('HISTORY', 'PHASE: %s' %
                                    phasebasename)

        # If an Instrumental Configuration HDU is present
        # in datain, copy it to dataout (in HDU1)
        if 'CONFIGURATION' in self.datain.imgnames:
            configdata = self.datain.imgdata[
                self.datain.imageindex(imagename='CONFIGURATION')]
            confighead = self.datain.imgheads[
                self.datain.imageindex(imagename='CONFIGURATION')]
            self.dataout.imageset(configdata,
                                  'CONFIGURATION',
                                  imageheader=confighead, index=1)
