# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Flat creation pipeline step."""

import os
import re

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.stepmoparent import StepMOParent
from sofia_redux.instruments.hawc.steploadaux import StepLoadAux
from sofia_redux.instruments.hawc.datafits import DataFits

__all__ = ['StepMkflat']


class StepMkflat(StepMOParent, StepLoadAux):
    """
    Create a flat file from internal calibrator data.

    This step combines the internal calibrator observations
    (INTCALs) taken adjacent to a science observation, then multiplies
    by a master flat (skycal) to generate an observation flat
    (OFT) file for each input group of INTCALs.  The skycal
    file is generally a flat generated from a Chop-Scan observation
    of a bright source, divided by its own associated INTCAL.
    The INTCAL files are used to correct for temporal variations
    in detector responsivity, caused by changes in bias, background
    loading, and/or ADR control temperature; the skycal corrects for
    more static detector pixel response variations.

    Variances from the INTCAL files are also propagated into the
    output flat files.

    The input for this step is multiple demodulated files from
    internal calibrator observations (CALMODE = 'INT_CAL' files),
    demodulated using the ‘mode_intcal’ pipeline mode.  Thus mode
    must set the following parameters for StepDemodulate:

    -  *l0method* = RE
    -  *chopavg* = True
    -  *phasefile* = 0.0

    The output for this step is a normalized INTCAL, one per input
    file, before combination and multiplication by the skycal.
    As a side effect, the observation flat files are also written
    to disk with product identifier 'OFT', in a folder designated
    by the 'flatoutfolder' parameter. OFT files have R ARRAY GAIN,
    T ARRAY GAIN, R ARRAY GAIN VAR, T ARRAY GAIN VAR, R BAD PIXEL
    MASK, and T BAD PIXEL MASK image extensions.

    The mode_intcal pipeline ending in StepMkFlat should be run before
    the regular chop/nod pipeline to make the observation flats. Then, the
    regular pipeline can be run, making sure that StepFlat is configured
    to look in flatoutfolder for flatfiles, with the appropriate
    fitkeys settings to make sure the correct flat is paired with the
    observation.
    """
    def setup(self):
        r"""
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'mkflat', and are named with
        the step abbreviation 'DCL'.

        Parameters defined for this step are:

        flatoutfolder : str
            Path for the folder to write flat files to.  May be
            relative or absolute.  Set to an empty string to write to
            the same folder as the input file.
        groupkey : str
            Header keyword that must match across INTCAL files in
            order to group them together.  Typically set to FILEGPID.
        skip_start : int
            Chops to exclude from the beginning of the file.
        skip_end : int
            Chops to exclude from the end of the file.
        bad_dead : float
            Raw data threshold for dead pixels.
        bad_ramping : float
            Raw data threshold for ramping pixels.
        normstd : float
            Threshold to exclude pixels with high standard deviation.
        ynormlowlim : list of float
            Threshold to exclude pixels with low normalized signal, given
            for [R0, R1, T0].
        ynormhighlim : list of float
            Threshold to exclude pixels with high normalized signal, given
            for [R0, R1, T0].
        ttor : float
            Scale factor for the T array to the R array.
        scalfile : str
            Filename for auxiliary file(s). Can contain \* and ?
            wildcards to match multiple files to be selected using fitkeys.
        bkupscal : str
            Back up filename for auxiliary file(s). Can contain \*
            and ? wildcards to match multiple files to be selected using
            fitkeys.
        scalfitkeys : list of str
            List of header keys that need to match the scal
            data file.  These are only used if multiple files match
            the input INTCAL file.
        daterange : float
            If DATE-OBS is in scalfitkeys, files are matched within
            this many days.
        """
        # Name of the pipeline reduction step
        self.name = 'mkflat'
        self.description = 'Make INTCAL Flats'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'dcl'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['flatoutfolder', '',
                               "Path for the folder to write flat "
                               "files to ('' means folder of "
                               "input file"])
        self.paramlist.append(['groupkey', 'FILEGPID',
                               'Header keyword to match input '
                               'files to the same observation'])
        self.paramlist.append(['skip_start', 1,
                               'Chops to exclude from the '
                               'beginning of the file'])
        self.paramlist.append(['skip_end', 1,
                               'Chops to exclude from the '
                               'end of the file'])
        self.paramlist.append(['bad_dead', 10.,
                               'Raw data threshold for dead pixels'])
        self.paramlist.append(['bad_ramping', 2e6,
                               'Raw data threshold for ramping pixels'])
        self.paramlist.append(['normstd', 10.,
                               'Threshold for HIGH STD of DMD '
                               'SIGNAL to exclude pixels'])
        self.paramlist.append(['ynormlowlim', [.5, .5, .5],
                               'Threshold to eliminate pixels '
                               'with LOW SIGNAL'])
        self.paramlist.append(['ynormhighlim', [10., 10., 10.],
                               'Threshold to eliminate pixels with '
                               'HIGH NORMALIZED SIGNAL'])
        self.paramlist.append(['TtoR', 2.0,
                               'Scale factor for T/R flatfield'])
        self.paramlist.append(['dcl_only', False,
                               'Make DCL output only (no OFT), and save '
                               'to flatoutfolder'])

        # Get parameters for StepLoadAux, replace auxfile with scal
        self.loadauxsetup(auxpar='scal')

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is multi-in, multi-out (MIMO),
        self.datain must be a list of DataFits objects.  The output
        is also a list of DataFits objects, stored in self.dataout.

        The process is:

        1. Read data from the INTCAL files.
        2. For each file, flag outliers and normalize by the median
           signal in the R and T arrays.
        3. Identify groups of data by matching across the 'groupkey'
           parameter.
        4. Identify and read the data from an auxiliary skycal file.
        5. Mean-combine the normalized gain for each image in group.
        6. Multiply the mean INTCAL images by the skycal master
           flat.
        7. Save resulting OFT files to disk, in 'flatoutfolder'.
        """
        # Set Options and get parameters
        # Chops to exclude from the beginning of the file
        skip_start = self.getarg('skip_start')
        # Chops to exclude from the end of the file
        skip_end = self.getarg('skip_end')
        # Raw data threshold for dead pixles
        bad_dead = self.getarg('bad_dead')
        # Raw data threshold for ramping pixels. Set ramping as low as
        # possible, but if < 5e5 you may start to lose some good pixels
        bad_ramping = self.getarg('bad_ramping')
        # threshold for HIGH STD of DMD SIGNAL to exclude pixels
        normstd = self.getarg('normstd')
        # threshold to elliminate pixels with LOW SIGNAL
        ynormlowlim = self.getarg('ynormlowlim')
        # threshold for pixels with HIGH NORMALIZED SIGNAL
        ynormhighlim = self.getarg('ynormhighlim')
        # Scale factor for T/R flatfield
        t_to_r = self.getarg('TtoR')
        # option to skip flat generation entirely and make the INTCAL
        # product only
        skip_flat = self.getarg('dcl_only')

        # Loop through input datasets
        self.dataout = []
        self.auxout = []
        for pd in self.datain:
            # Store R/T Real/Imag in indata
            end = len(pd.table) - skip_end
            signals = ['R array', 'T array', 'R array Imag', 'T array Imag']
            indata = np.zeros((4, end - skip_start, 41, 64))
            invar = np.zeros((4, end - skip_start, 41, 64))
            for i in range(4):
                indata[i] = pd.table[signals[i]][skip_start:end, :, :]
                invar[i] = pd.table[signals[i] + ' VAR'][skip_start:end, :, :]

            # Store R/T Raw averages in rawavg
            rawavg = np.zeros((2, end - skip_start, 41, 64))
            signals = ['R array AVG', 'T array AVG']
            for i in [0, 1]:
                rawavg[i] = pd.table[signals[i]][skip_start:end, :, :]

            # Make raw stdevs
            # 2, 41, 64 array
            rawstd = np.nanstd(rawavg, axis=1)

            # Set Dead / Ramping pixels as NaN in indata
            # This next section finds and masks dead and
            # ramping pixels, so they won't bias the medians
            # calculated below. This may be redundant if they have
            # already been eliminated by the pipeline in StepNoah.
            # If so, it won't hurt to do it again, and it
            # allows tightening the criteria beyond those
            # set in StepNoah, if desired.
            nbad = np.ones((4, 41, 64))

            # Identify bad and ramping pixels by looking low and high rawstd
            # Dead pixels
            nbad[np.where(rawstd < bad_dead)] = np.NaN
            # Ramping pixels
            nbad[np.where(rawstd > bad_ramping)] = np.NaN

            # Mark all bad pixels with NaN in indata
            # This sets the T1 array signal values to nan.
            nbad[1, :, 32:64] = np.NaN
            nbad.shape = (4, 1, 41, 64)
            # multiply a (4, n, 41, 64) array by (4, 1, 41, 64) array
            indata *= nbad
            invar *= nbad

            # Calculate Modulus, phase and the medians for R and T array

            # initialize arrays
            modulus = np.zeros((2, indata.shape[1], 41, 64))
            phase = np.zeros((2, indata.shape[1], 41, 64))
            modvar = np.zeros((2, invar.shape[1], 41, 64))
            # phasevar = np.zeros((2, invar.shape[1], 41, 64))

            # Loop over R and T array
            for i in range(2):
                rld = indata[i]
                imd = indata[i + 2]
                rlv = invar[i]
                imv = invar[i + 2]

                # m = sqrt(r^2 + i^2)
                # Vm = (1/m^2)(r^2 Vr + i^2 Vi + 2 r i sqrt(Vr Vi))
                msq = (rld**2 + imd**2)
                modulus[i] = np.sqrt(msq)
                modvar[i] = (1 / msq) * (rld**2 * rlv
                                         + imd**2 * imv
                                         + 2 * rld * imd * np.sqrt(rlv * imv))

                # p = arctan(i / r)
                # Vp = (r^2 + i^2)^(-2) * (i^2 Vr + Vi - 2 sqrt(Vr Vi))
                # (phase variance not needed at this time)
                phase[i] = np.arctan(imd / rld)
                # phasevar[i] = (rld**2 + imd**2)^(-2) * \
                #               (imd**2 * rlv + imv - 2 * np.sqrt(rlv * imv))

            modmap = np.nanmedian(modulus, axis=1)
            phasemap = np.nanmedian(phase, axis=1)

            # variance of median is approximately pi/2 * variance of mean
            with np.errstate(invalid='ignore'):
                modmapvar = (np.pi / 2.) \
                    * (np.nansum(modvar, axis=1)
                       / np.count_nonzero(~np.isnan(modulus), axis=1) ** 2)

            # Generate temporary subarrays to iterate over data
            subs = np.zeros((3, 41, 32))
            subs[0] = modmap[0, :, 0:32]
            subs[1] = modmap[0, :, 32:64]
            subs[2] = modmap[1, :, 0:32]

            # Set limits then iterate over them
            lowlim = (np.nanmedian(modmap[0, :, 0:32]) / 5,
                      np.nanmedian(modmap[1, :, 0:32]) / 5,
                      np.nanmedian(modmap[0, :, 32:64]) / 5)
            highlim = (200000., 200000., 200000.)
            outs = []
            for i in range(4):
                # Operates on the real parts of the signals
                outs = self._histogram3d(subs, lowlim, highlim)
                lowlim = outs[0] / 2
                highlim = 2 * outs[0]
            ymeds = outs[0]

            # Normalized modulus map and its stdev -> ynorm, ynormstd

            # Make normalized signal maps
            # Alias modmap to ymean to make it easier to re-use
            # code for making flats
            ymean = modmap
            ynorm = np.zeros((2, 41, 64))
            ynorm[0, :, 0:32] = ymean[0, :, 0:32] / ymeds[0]
            ynorm[0, :, 32:64] = ymean[0, :, 32:64] / ymeds[1]
            ynorm[1, :, 0:32] = ymean[1, :, 0:32] / ymeds[2]
            ynorm[1, :, 32:64] = np.NaN

            # Make normalized std maps
            ystd = np.nanstd(modulus, axis=1)
            ynormstd = np.zeros((2, 41, 64))
            ynormstd[0, :, 0:32] = ystd[0, :, 0:32] / ymean[0, :, 0:32]
            ynormstd[0, :, 32:64] = ystd[0, :, 32:64] / ymean[0, :, 32:64]
            ynormstd[1, :, 0:32] = ystd[1, :, 0:32] / ymean[1, :, 0:32]

            # Create bad pixel map and flatfield
            # using various selection criteria

            # Badstack datacube holds information about
            # which bad-pixel criteria have been detected.
            badstack = np.zeros((6, 2, 41, 64))

            # Badstack will be flattened into badsum
            # below (by addition). In badsum, bits in
            # a binary number indicate which criteria
            # have been triggered.

            # Criteria (1/1) This criterion identifies DEAD
            # pixels by looking for those with essentially zero rawstd.
            ybad = np.zeros((2, 41, 64))
            # dead has already been set above in section creating nbad.
            with np.errstate(invalid='ignore'):
                ybad[np.where(rawstd < bad_dead)] = 1
            badstack[0] = ybad.copy()

            # Criteria (2/2) This criterion identifies RAMPING
            # pixels by looking for very high rawstd
            ybad = np.zeros((2, 41, 64))
            # ramping has already been set above in section creating nbad.
            ybad[np.where(rawstd > bad_ramping)] = 2
            badstack[1] = ybad.copy()

            # Criteria (3/4) This criterion looks for very HIGH STD
            # of the demodulated signal.
            # May not be completely redundant with lower limit
            # on rawstd above. Need to check this with more data.
            ybad = np.zeros((2, 41, 64))
            with np.errstate(invalid='ignore'):
                ybad[np.where(ynormstd > normstd)] = 4
            badstack[2] = ybad.copy()

            # One could define a criterion that looks for low rawstd
            # not to identify dead pixels but to identify pixels
            # with very low signal
            # (e.g., those on the "normal" resistance part of the IV curve).
            # For example, Rbad[where(rstd < 1e1)]=xx. However this
            # may be redundant with rnorm < 0.x below. May want to take
            # a closer look at this later.

            # Criteria (4/8) This criterion attempts to identify
            # and eliminate pixels with very LOW SIGNAL
            # (e.g., those on the "normal" part of the IV curve).
            ybad = np.zeros((2, 41, 64))
            with np.errstate(invalid='ignore'):
                ybad[0, :, 0:32][np.where(ynorm[0, :, 0:32]
                                          < ynormlowlim[0])] = 8
                ybad[0, :, 32:64][np.where(ynorm[0, :, 32:64]
                                           < ynormlowlim[1])] = 8
                ybad[1, :, 0:32][np.where(ynorm[1, :, 0:32]
                                          < ynormlowlim[2])] = 8
            badstack[3] = ybad.copy()

            # Criteria (5/16) This criterion identifies pixels
            # with very HIGH NORMALIZED SIGNAL.
            # Need to check whether this might eliminate some "good" pixels.
            # Might not be needed. Might be redundant with
            # (and more prone to eliminate "good" pixels than)
            # the criterion on ynormstd above.
            ybad = np.zeros((2, 41, 64))
            with np.errstate(invalid='ignore'):
                ybad[0, :, 0:32][np.where(ynorm[0, :, 0:32]
                                          > ynormhighlim[0])] = 16
                ybad[0, :, 32:64][np.where(ynorm[0, :, 32:64]
                                           > ynormhighlim[1])] = 16
                ybad[1, :, 0:32][np.where(ynorm[1, :, 0:32]
                                          > ynormhighlim[2])] = 16
            badstack[4] = ybad.copy()

            # Criteria (6/32) This one ELIMINATES ROW 40
            # (the row with multiplexer but no active IR pixels).
            # This one may not be needed if the bad pixel mask ingested
            # at the beginning of this algorithm has already set
            # these rows to "bad". On the other hand, this doesn't
            # take long to do here, so we could leave it
            # for now, just in case we change approaches later.
            ybad = np.zeros((2, 41, 64))
            ybad[:, 40, :] = 32
            badstack[5] = ybad.copy()

            # Create bad pixel map (value labels how the pixel is bad)
            # This step combines all the bad pixel identifications
            # into a single map with binary values for which the bits
            # each indicate that a particular selection criterion
            # has been triggered.
            badsum = np.sum(badstack, axis=0)

            # Create bad pixel mask (with nans) to apply
            # to flatfields. Add this to the target map
            # to mask the bad pixels.
            bad = np.zeros((2, 41, 64))
            with np.errstate(invalid='ignore'):
                bad[np.where(badsum != 0)] = np.NaN

            # Make flatfields (Rcal, Tcal) and medians array

            # Make DCAL type flat fields, based on time-averaged
            # modulus (bad pixels set to NaN)
            rcal = 10000. / ymean[0] + bad[0]
            tcal = 10000. * t_to_r / ymean[1] + bad[1]

            # Propagate variance
            rvar = (10000. / (ymean[0]**2))**2 * modmapvar[0] + bad[0]
            tvar = (10000. * t_to_r / (ymean[1]**2))**2 * modmapvar[1] + bad[1]

            # Make an array to hold information on the
            # median intensities and masked map medians of
            # normalized std's, rawavg, and rawavg std's.
            medians = np.zeros((4, 3))
            medians[0] = ymeds

            ynormstdm = ynormstd + bad
            ynormstdmmed = np.zeros(3)
            ynormstdmmed[0] = np.nanmedian(ynormstdm[0, :, 0:32])
            ynormstdmmed[1] = np.nanmedian(ynormstdm[0, :, 32:64])
            ynormstdmmed[2] = np.nanmedian(ynormstdm[1, :, 0:32])
            medians[1] = ynormstdmmed

            rawmedianm = np.nanmedian(rawavg, axis=1) + bad
            rawmedianmmed = np.zeros(3)
            rawmedianmmed[0] = np.nanmedian(rawmedianm[0, :, 0:32])
            rawmedianmmed[1] = np.nanmedian(rawmedianm[0, :, 32:64])
            rawmedianmmed[2] = np.nanmedian(rawmedianm[1, :, 0:32])
            medians[2] = rawmedianmmed

            rawstdm = rawstd + bad
            rawstdmmed = np.zeros(3)
            rawstdmmed[0] = np.nanmedian(rawstdm[0, :, 0:32])
            rawstdmmed[1] = np.nanmedian(rawstdm[0, :, 32:64])
            rawstdmmed[2] = np.nanmedian(rawstdm[1, :, 0:32])
            medians[3] = rawstdmmed

            # Save to output object.

            # This will save a DCAL named after input file
            # with 'DCAL' in file-type identifier field.
            # Rcal and Tcal correct detector gains to the
            # values that would give a signal of 10000 ADU
            # when exposed to the IR-50 internal calibrator source.
            # The factor TorR is a first-order correction for the
            # difference in the IR-50 illumination of the R and T
            # detectors. Higher-order corrections for the
            # detector to detector differences when exposed to
            # radiation from the sky must be derived from direct
            # measurements of astronomical sources.
            # t_to_r = 2.0 is a first-order guess based on
            # limited observations of astronomical source
            # signals. It could be refined by additional
            # observations, or the remaining differences could
            # be subsumed in the intcal-to-source-flat
            # transformation coefficients. The number 10000
            # was chosen to make the median R0 subarray signal
            # approximately equal to what it would be if no
            # corrections were applied (based on observations
            # with "typical" biases used in commissioning flights).

            # Get Configuration image from DMD file
            imgconfig = pd.imageget('CONFIGURATION')

            # Convert phases to time delay
            chopfreq = pd.getheadval('CHPFREQ')
            rphase = phasemap[0] * (-1 / (2 * np.pi * chopfreq)) + bad[0]
            tphase = phasemap[1] * (-1 / (2 * np.pi * chopfreq)) + bad[1]

            # Shift the phases to center around zero
            rphase = rphase - (-0.055)
            tphase = tphase - (-0.055)

            # Set NaN pixels to median
            # (not used at this time)

            # R0medianm = np.nanmedian(rphase[:, 0:32])
            # R1medianm = np.nanmedian(rphase[:, 32:64])
            # T0medianm = np.nanmedian(tphase[:, 0:32])
            # nanR0 = np.where(np.isnan(rphase[:, 0:32]))
            # nanT = np.where(np.isnan(tphase))
            # rphase[nanR0]=R0medianm
            # tphase[nanT]=T0medianm
            # nanR1 = np.where(np.isnan(rphase))
            # rphase[nanR1] = R1medianm

            # Make and fill output data
            outd = DataFits(config=pd.config)
            outd.filename = pd.filename
            outd.imageset(rcal, 'R ARRAY GAIN')
            outd.imageset(tcal, 'T ARRAY GAIN')
            outd.imageset(rvar, 'R ARRAY GAIN VAR')
            outd.imageset(tvar, 'T ARRAY GAIN VAR')
            outd.imageset(badsum[0], 'R BAD PIXEL MASK')
            outd.imageset(badsum[1], 'T BAD PIXEL MASK')
            outd.imageset(medians, 'SUBARRAY MEDIANS')
            outd.imageset(imgconfig, 'CONFIGURATION')
            outd.imageset(rphase, 'RPHASE')
            outd.imageset(tphase, 'TPHASE')
            outd.imageset(rawmedianm[0], 'R RAWAVG')
            outd.imageset(rawmedianm[1], 'T RAWAVG')
            outd.header = pd.header

            # Write the configuration header from pd into
            # the corresponding HDU in outd.
            outd.imgheads[5] = pd.imgheads[1]

            # Append to dataout
            self.dataout.append(outd)

        # END of loop over input datasets

        # Make groups of output files -> datagroups
        groupkey = self.getarg('groupkey')
        datagroups = []

        # Make output folder if it doesn't exist
        outfolder = self.getarg('flatoutfolder')
        if len(outfolder) > 0:
            if not os.path.isabs(outfolder):
                # if outfolder is not an absolute path, make it
                # relative to the output data location
                outfolder = os.path.join(
                    os.path.dirname(self.dataout[0].filename),
                    outfolder)
            try:
                os.makedirs(outfolder)
            except OSError:
                if not os.path.isdir(outfolder):
                    log.error('Run: Failed to make flatoutfolder = %s' %
                              outfolder)
                    raise

        # if desired, just save DCL files to flats folder and quit
        if skip_flat:
            for data in self.dataout:
                dclname = data.filenamebegin + 'DCL' + data.filenameend
                if len(outfolder) > 0:
                    dclname = os.path.join(outfolder,
                                           os.path.basename(dclname))
                data.save(dclname)
                self.auxout.append(dclname)
                log.info('Saved result %s' % dclname)
            self.dataout = []
            return

        # Loop over datasets in datain
        for data in self.dataout:
            groupind = 0
            # Loop over groups until group match found or end reached
            while groupind < len(datagroups):
                # Check if data fits group: Get first group element
                gdata = datagroups[groupind][0]

                # Get key from group and new data - format if needed
                dkey = data.getheadval(groupkey)
                gkey = gdata.getheadval(groupkey)

                # Match -> add to group
                if dkey == gkey:
                    datagroups[groupind].append(data)
                    break

                # Not found -> increase group index
                groupind += 1
            # If not in any group -> make new group
            if groupind == len(datagroups):
                datagroups.append([data, ])

        # info messages
        log.debug(" Found %d data groups" % len(datagroups))
        for groupind in range(len(datagroups)):
            group = datagroups[groupind]
            msg = "  Group %d len=%d" % (groupind, len(group))
            msg += " %s = %s" % (groupkey, group[0].getheadval(groupkey))
            log.debug(msg)

        # Make and save groupflats

        # loop over datagroups
        for groupind in range(len(datagroups)):
            data0 = datagroups[groupind][0]

            # Load masterflat for this datagroup -> mflat
            mflat = self.loadauxfile(data=data0)

            # Average intcal R/T Array Gains
            filenum = []
            rlist = []
            tlist = []
            rvlist = []
            tvlist = []
            for data in datagroups[groupind]:
                rg = data.imageget('R ARRAY GAIN')
                if np.any(~np.isnan(rg)):
                    rlist.append(rg)
                    rvlist.append(data.imageget('R ARRAY GAIN VAR'))
                    found_r = True
                else:
                    found_r = False

                tg = data.imageget('T ARRAY GAIN')
                if np.any(~np.isnan(tg)):
                    tlist.append(tg)
                    tvlist.append(data.imageget('T ARRAY GAIN VAR'))
                    found_t = True
                else:
                    found_t = False

                if found_r or found_t:
                    if data.filenum is not None:
                        filenum.extend(data.filenum.split('-'))
                else:
                    log.warning('Excluding bad file %s' % data.filename)

            if not rlist or not tlist:
                msg = 'No good flat files found.'
                log.error(msg)
                raise ValueError(msg)

            rgainavg = np.nanmean(np.array(rlist), axis=0)
            tgainavg = np.nanmean(np.array(tlist), axis=0)
            rgainvar = np.nansum(np.array(rvlist), axis=0) / len(rvlist) ** 2
            tgainvar = np.nansum(np.array(tvlist), axis=0) / len(tvlist) ** 2

            # Multiply masterflat -> groupflat
            gflatr = mflat.imageget('R ARRAY GAIN') * rgainavg
            gflatt = mflat.imageget('T ARRAY GAIN') * tgainavg
            gflatrvar = (mflat.imageget('R ARRAY GAIN'))**2 * rgainvar
            gflattvar = (mflat.imageget('T ARRAY GAIN'))**2 * tgainvar

            # Groupflat bad pixel mask: see where there are NaNs
            gbadr = np.zeros(gflatr.shape)
            gbadt = np.zeros(gflatt.shape)
            gbadr[np.where(np.isnan(gflatr))] = 1.0
            gbadt[np.where(np.isnan(gflatt))] = 2.0

            # Make groupflat filename: Filename of
            # first file with OBSFLAT (OFT)
            gflatname = data0.filenamebegin + 'OFT' + data0.filenameend

            # Split file, make sure outfolder is valid
            if len(outfolder) > 0:
                gflatname = os.path.split(gflatname)[1]
            else:
                outfolder, gflatname = os.path.split(gflatname)

            # Update file number to be a range
            if len(filenum) > 1:
                fn = sorted(filenum)
                filenums = fn[0] + '-' + fn[-1]
                match = re.search(self.config['data']['filenum'], gflatname)
                if match is not None:
                    # regex may contain multiple possible matches --
                    # for middle or end of filename
                    for i, g in enumerate(match.groups()):
                        if g is not None:
                            gbegin = gflatname[:match.start(i + 1)]
                            gend = gflatname[match.end(i + 1):]
                            gflatname = gbegin + filenums + gend
                            break
            gflatname = os.path.join(outfolder, gflatname)

            # Make groupflat data
            gflat = DataFits(config=self.datain[0].config)
            gflat.header = data0.header.copy()

            # Make header
            for data in datagroups[groupind][1:]:
                gflat.mergehead(data)
            for data in datagroups[groupind]:
                gflat.setheadval('PRODTYPE', 'obsflat')
                gflat.setheadval('PROCSTAT', 'LEVEL_2')
                gflat.setheadval('HISTORY', 'MakeFlat Source: %s' %
                                 (os.path.split(data.filename)[1], ))
            # Fill data
            gflat.imageset(gflatr, 'R ARRAY GAIN')
            gflat.imageset(gflatt, 'T ARRAY GAIN')
            gflat.imageset(gflatrvar, 'R ARRAY GAIN VAR')
            gflat.imageset(gflattvar, 'T ARRAY GAIN VAR')
            gflat.imageset(gbadr, 'R BAD PIXEL MASK')
            gflat.imageset(gbadt, 'T BAD PIXEL MASK')

            # Save groupflat data
            gflat.save(gflatname)
            self.auxout.append(gflatname)
            log.info('Saved result %s' % gflatname)

    # fn 'histogram3d' operates on a three-dimensional
    # image with multiple planes
    # (e.g., one containing both R and T array data).
    # Inputs are the following:
    #   cube = a three dimensional image
    #   hmin and hmax = lower and upper limits for the histogram
    #                   (one-dimensional arrays with
    #                    number of elements = "planes").
    def _histogram3d(self, cube, hmin, hmax):
        """
        Compute median and standard deviation of image planes.

        Parameters
        ----------
        cube : array-like
            A 3-dimensional image array, with distinct image planes.
        hmin : array-like of float
            Lower limit of the histogram, one value per plane.
        hmax : array-like of float
            Upper limit of the histogram, one value per plane

        Returns
        -------
        list of array-like
            The first element is the median value for each plane;
            second element is the standard deviation for each plane.
        """
        planes = len(cube)
        medians = np.zeros(planes)
        stdof = np.zeros(planes)
        for i in range(planes):
            # Make a copy because we want to change the image
            # into a list without changing the original
            subarr = cube[i, ].copy()
            # Change the shape of subarr
            subarr.shape = (subarr.shape[0] * subarr.shape[1], )
            medians[i] = np.nanmedian([f for f in subarr if
                                       hmin[i] <= f < hmax[i]])
            stdof[i] = np.nanstd([f for f in subarr if
                                  hmin[i] <= f < hmax[i]])
        outdata = [medians, stdof]
        return outdata
