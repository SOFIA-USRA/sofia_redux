# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Diagnostic plots pipeline step."""

import os

from astropy import log
import astropy.units as u
from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.instruments.hawc.datafits import DataFits

__all__ = ['StepDmdPlot']


class StepDmdPlot(StepParent):
    """
    Produce diagnostic plots for demodulated data.

    This pipeline step produces a set of diagnostic plots from demodulated
    chop/nod data. For data that are not from the internal calibrator,
    the data is checked for possible door vignetting. For
    data that are from the internal calibrator, images of phase
    differences from expected values are displayed.

    This step requires demodulated data as input, as produced by
    the `sofia_redux.instruments.hawc.steps.StepDemodulate` step.
    The input table should include columns for Chop Mask and
    Samples, RA, Dec, TrackErrAoi3, TrackErrAoi4,
    and either CentroidAoi or SofiaHkTrkAoi; R array, R array Imag,
    T array, T array Imag; CentroidExpMsec; HWP Angle, HWP Index and
    Nod Index.

    The output from this step is identical to the input. As a side
    effect, a PNG file is saved to disk to the same base name as the input
    file, with 'DPL' replacing the product type indicator.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'dmdplot', and are named with
        the step abbreviation 'DPL'.

        Parameters defined for this step are:

        door_threshold : float
            Ratio of real to imaginary median stds, for door
            vignetting warnings.
        detector_i : int
            i-location of detector pixel to plot.
        detector_j : int
            j-location of detector pixel to plot.
        data_sigma : float
            Value for sigma-clipping of detector data in computing
            averages.
        data_iters : int
            Number of iterations for sigma-clipping.
        ref_phase_file : str
            Path to a FITS file containing reference phase values,
            for comparison with CALMODE = INT_CAL data. Set to
            '0.0' to skip the comparison.
        user_freq : float
            Expected chop frequency for science data. Used to compute
            phases in degrees from CALMODE = INT_CAL data.
        phase_thresh : float
            Threshold for phase outliers. Values above this number
            are ignored.
        save_phase : bool
            If set, phase images are saved as FITS files to the
            input base name with 'PHS' replacing the product type
            indicator.
        savefolder : str
            Folder to save plots to. An empty string indicates the
            same folder as the input file.
        """
        # Name of the pipeline reduction step
        self.name = 'dmdplot'
        self.description = 'Make Demod Plots'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        # the step gets the procname from datain
        self.procname = 'dpl'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['door_threshold', 2.0,
                               "ratio of real to imaginary median "
                               "stds for door vignetting warnings"])
        self.paramlist.append(['detector_i', 14,
                               "i-location of detector pixel to plot"])
        self.paramlist.append(['detector_j', 24,
                               "j-location of detector pixel to plot"])
        self.paramlist.append(['data_sigma', 5.0,
                               "value for sigma-clipping of detector data"])
        self.paramlist.append(['data_iters', 3,
                               "number of iterations for data clipping"])
        self.paramlist.append(['ref_phase_file', '0.0',
                               "INT_CAL: path to reference phase file "
                               "in degrees (0.0 is none)"])
        self.paramlist.append(['user_freq', 10.2,
                               "INT_CAL: user frequency in Hz"])
        self.paramlist.append(['phase_thresh', 50.0,
                               'threshold in phase uncertainty (deg)'])
        self.paramlist.append(['save_phase', False,
                               'Save phase images to PHS suffix'])
        self.paramlist.append(['savefolder', '',
                               "folder to save plots to. '' means same "
                               "as input file."])

    def get_gapindex(self, timeseries, thresh=0.12):
        """
        Find indices of time gaps above a threshhold.

        Parameters
        ----------
        timeseries : array-like
            Sorted array of times.

        thresh : float, optional
            Threshold for determining a gap.

        Returns
        -------
        gapindex : array-like of int
            index at which to insert None values to indicate gaps
        """
        timedelt = timeseries[1:] - timeseries[:-1]
        with np.errstate(invalid='ignore'):
            gapindex = np.where(timedelt > thresh)[0] + 1
        return gapindex

    def calc_phase(self, real_part, imag_part,
                   chop_freq, user_freq, phaseref=None):
        """
        Calculate a phase image.

        Parameters
        ----------
        real_part : array-like
            Array of real values.
        imag_part : array-like
            Array of imaginary values.
        chop_freq : float
            Chopper frequency of the data.
        user_freq : float
            User-specified frequency.
        phaseref : astropy.units.Quantity with units of degrees, optional
            If specified, subtract this reference phase.

        Returns
        -------
        phase : array of astropy.units.Quantity with units of degrees
            Phase in degrees, in range -180 deg to 180 deg.
        """
        if phaseref is None:
            phaseref = np.zeros_like(real_part) * u.rad
        phase = -np.arctan2(imag_part, real_part) * u.rad - \
            np.deg2rad(phaseref)
        phase *= (user_freq / chop_freq)
        phase = (phase + np.pi * u.rad) % (2 * np.pi * u.rad) - \
            np.pi * u.rad
        phase = np.rad2deg(phase)
        return phase

    def sigma_mask(self, array, sigma=5, iters=3):
        """
        Mask array along axis 0 using sigma-clipping and median.

        Parameters
        ----------
        array : array-like
            Data array.

        sigma : float, optional
            Threshold for sigma-clipping, with median.

        iters : int, optional
            Number of iterations.

        Returns
        -------
        msk_array : masked array
            Masked array with mask from sigma-clipping.
        """
        from numpy.ma import masked_where
        with np.errstate(invalid='ignore'):
            if iters > 0:
                msk_array = masked_where(
                    np.ma.abs(array
                              - np.ma.median(array, axis=0))
                    > sigma * np.ma.std(array, axis=0),
                    array, copy=True)
                if iters > 1:
                    for i in range(iters - 1):
                        msk_array = masked_where(
                            np.ma.abs(msk_array
                                      - np.ma.median(msk_array, axis=0))
                            > sigma * np.ma.std(msk_array, axis=0), msk_array)
            else:
                msk_array = np.ma.asarray(array)

        return msk_array

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a single-in single-out (SISO) step:
        self.datain should be a DataFits object, and output will also
        be a single DataFits, stored in self.dataout.

        The process is:

        1. For all data types, plot the RA and Dec in one panel; TrackErrAoi3
           and TrackErrAoi4 in another panel; HWP angle in yet another panel;
           and the sigma-clipped data values of the user-specified detector
           in the final panel. If present, CentroidAoi or SofiaHkTrkAoi are
           used to indicate which of TrackErrAoi3 and TrackErrAoi4 are
           active. Green shading indicates which samples will be passed by
           StepDmdCut; for this purpose, this step reads the StepDmdCut
           parameters from the configuration.

        2. For all data not taken with the internal calibrator, calculate the
           ratio of real to imaginary data in a 5x5 pixel box centered on the
           user-specified pixel used for plotting. The signals are
           sigma-clipped before the ratios of medians are computed. The
           values are output in a table at the upper left corner of the plot.
           If either the R ratio or T ratio exceeds the parameter
           door_threshold, warnings are output to the plot and to the
           logger that possible door vignetting may have occurred. In
           separate panels, CentroidExpMsec, and both Nod Index and HWP
           Index, are plotted.

        3. For data taken with the internal calibrator, images of the
           detector phase are plotted. If the ref_phase_file is
           specified, it is subtracted from the calculated phases. The
           algorithm includes these steps:

           - Discard the first and last chops and any NaN values, then
             calculate the mean along the first (time) axis of the R and T
             arrays, for both real and imaginary signals.

           - For each of R and T, compute -arctan2(imaginary, real) and
             convert to degrees.

           - Subtract the reference phase if it is provided.

           - Multiply by the ratio of 10.2 Hz to the value of ‘CHPFREQ’ in
             the file header (about 3 Hz).

           - Add or subtract multiples of 360 degrees to put values in the
             branch -180 degrees to +180 degrees

        """
        # Copy dataout
        self.dataout = self.datain

        # Get parameters
        door_threshold = self.getarg('door_threshold')
        detector_i = self.getarg('detector_i')
        detector_j = self.getarg('detector_j')
        data_sigma = self.getarg('data_sigma')
        data_iters = self.getarg('data_iters')
        mask_bits = int(self.config['dmdcut']['mask_bits'])
        min_samples = int(self.config['dmdcut']['min_samples'])

        table = self.datain.table

        # Apply mask
        if 'Chop Mask' in table.names:
            chopmask = table['Chop Mask']
        else:
            log.info('Chop mask column not found, using zeros')
            chopmask = np.zeros(len(self.datain.table), dtype=np.int32)
        nsamples = table['Samples']

        with np.errstate(invalid='ignore'):
            keep_where = ((chopmask & mask_bits) == 0) & \
                         (nsamples >= min_samples)

        reltime = table['Timestamp'] - table['Timestamp'][0]

        try:
            calmode = self.datain.getheadval('calmode', errmsg=False)
        except KeyError:
            calmode = 'UNKNOWN'

        if calmode == 'INT_CAL':

            chpfreq = self.datain.getheadval('chpfreq')

            user_freq = float(self.getarg('user_freq'))

            phase_thresh = float(self.getarg('phase_thresh'))

            save_phase = self.getarg('save_phase')

            array_shape = table['R array'].shape[1:]
            r_phaseref = np.zeros(array_shape, dtype=np.float64) * u.deg
            t_phaseref = np.zeros(array_shape, dtype=np.float64) * u.deg
            ref_phase_file = os.path.expandvars(self.getarg('ref_phase_file'))
            auxphase = False
            phmin = -180
            phmax = 180
            if os.path.isfile(ref_phase_file):
                # If it's a file, read it as such
                phasedata = DataFits(config=self.config)
                phasedata.load(ref_phase_file)
                r_phaseref = phasedata.imgdata[0] * u.deg
                t_phaseref = phasedata.imgdata[1] * u.deg
                auxphase = True
                phmin = -40
                phmax = 40

            mean_r = np.nanmean(table['R array'][1:-1, :, :], axis=0)
            std_r = np.nanstd(table['R array'][1:-1, :, :], axis=0)
            mean_r_imag = np.nanmean(table['R array Imag'][1:-1, :, :], axis=0)
            std_r_imag = np.nanstd(table['R array Imag'][1:-1, :, :], axis=0)
            mean_t = np.nanmean(table['T array'][1:-1, :, :], axis=0)
            std_t = np.nanstd(table['T array'][1:-1, :, :], axis=0)
            mean_t_imag = np.nanmean(table['T array Imag'][1:-1, :, :], axis=0)
            std_t_imag = np.nanstd(table['T array Imag'][1:-1, :, :], axis=0)

            r_phase = self.calc_phase(mean_r, mean_r_imag, chpfreq,
                                      user_freq, phaseref=r_phaseref)
            t_phase = self.calc_phase(mean_t, mean_t_imag, chpfreq,
                                      user_freq, phaseref=t_phaseref)
            r_plus = self.calc_phase(mean_r + std_r,
                                     mean_r_imag + std_r_imag,
                                     chpfreq, user_freq,
                                     phaseref=r_phaseref)
            r_minus = self.calc_phase(mean_r - std_r,
                                      mean_r_imag - std_r_imag,
                                      chpfreq, user_freq,
                                      phaseref=r_phaseref)
            t_plus = self.calc_phase(mean_t + std_t,
                                     mean_t_imag + std_t_imag,
                                     chpfreq, user_freq, phaseref=r_phaseref)
            t_minus = self.calc_phase(mean_t - std_t,
                                      mean_t_imag - std_t_imag,
                                      chpfreq, user_freq,
                                      phaseref=r_phaseref)

            cutval = phase_thresh * u.deg
            with np.errstate(invalid='ignore'):
                r_phase[(np.abs(r_plus - r_phase) > cutval)
                        | (np.abs(r_minus - r_phase) > cutval)] = np.nan
                t_phase[(np.abs(t_plus - t_phase) > cutval)
                        | (np.abs(t_minus - t_phase) > cutval)] = np.nan
            median_r_phase = np.nanmedian(r_phase.value)
            median_t_phase = np.nanmedian(t_phase.value)
            log.info('')
            if auxphase:
                log.info('median R phase offset = %.2f deg' %
                         float(median_r_phase))
                log.info('median T phase offset = %.2f deg' %
                         float(median_t_phase))
            else:
                log.info('median R phase = %.2f deg' %
                         float(median_r_phase))
                log.info('median T phase = %.2f deg' %
                         float(median_t_phase))
            log.info('')

            if save_phase:
                phasename = self.datain.filenamebegin + 'PHS' + \
                    self.datain.filenameend
                phaseproduct = DataFits(config=self.datain.config)

                phaseproduct.imageset(r_phase.value, 'RPHASE')
                phaseproduct.imageset(t_phase.value, 'TPHASE')

                phaseproduct.header = self.datain.header.copy()
                phaseproduct.setheadval('BUNIT', 'deg')
                phaseproduct.setheadval('CHPFREQ', chpfreq,
                                        'Input chop freq [Hz]')
                phaseproduct.setheadval('USERFREQ', user_freq,
                                        'User-specified chop freq [Hz]')
                phaseproduct.setheadval('PHASEREF', ref_phase_file,
                                        'Phase reference')
                phaseproduct.setheadval('PRODTYPE', 'phaseoffset')
                phaseproduct.setheadval('PROCSTAT', 'LEVEL_2')
                phaseproduct.save(phasename)

            fig = Figure(figsize=(20, 24))
            FigureCanvas(fig)
            outer = gridspec.GridSpec(5, 1, wspace=0.2, hspace=0.2)

            # Plot RA and Dec first
            ax1 = fig.add_subplot(outer[0])
            ax1.plot(reltime, table['RA'], 'b-', label='RA')
            ax1.set_xlabel('Time [sec]', fontsize=15)

            # Make the y-axis label, ticks and tick labels
            # match the line color
            ax1.set_ylabel('RA [deg]', color='b', fontsize=15)
            ax1.tick_params('y', colors='b')

            ax2 = ax1.twinx()
            ax2.plot(reltime, table['DEC'], 'r-', label='Dec')
            ax2.set_ylabel('Dec [deg]', color='r', fontsize=15)
            ax2.tick_params('y', colors='r')
            with np.errstate(invalid='ignore'):
                idx = (((chopmask & mask_bits) == 0)
                       & (nsamples >= min_samples))
            ax2.fill_between(reltime, ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=idx, facecolor='green', alpha=0.3)

            # Plot title
            ax1.set_title('Plots for %s' %
                          os.path.basename(self.datain.filename),
                          fontsize=20, fontweight='bold')

            # Plot TrackErrAoi3 and 4 next
            ax = fig.add_subplot(outer[1])
            aoikey = None
            if 'SofHkTrkaoi' in table.names:
                aoikey = 'SofHkTrkaoi'
            elif 'CentroidAoi' in table.names:
                aoikey = 'CentroidAoi'
            if aoikey is not None:
                log.debug('Found AOI key: {}'.format(aoikey))
                centroidaoi = table[aoikey]
                on3 = centroidaoi == 3.0
                on4 = centroidaoi == 4.0
                off3 = np.not_equal(centroidaoi, 3.0)
                off4 = np.not_equal(centroidaoi, 4.0)
                gidxon3 = self.get_gapindex(reltime[on3])
                gidxon4 = self.get_gapindex(reltime[on4])
                gidxoff3 = self.get_gapindex(reltime[off3])
                gidxoff4 = self.get_gapindex(reltime[off4])
                ontime3 = np.insert(reltime[on3], gidxon3, None)
                onaoi3 = np.insert(table['TrackErrAoi3'][on3],
                                   gidxon3, None)
                offtime3 = np.insert(reltime[off3], gidxoff3, None)
                offaoi3 = np.insert(table['TrackErrAoi3'][off3],
                                    gidxoff3, None)
                ontime4 = np.insert(reltime[on4], gidxon4, None)
                onaoi4 = np.insert(table['TrackErrAoi4'][on4],
                                   gidxon4, None)
                offtime4 = np.insert(reltime[off4], gidxoff4, None)
                offaoi4 = np.insert(table['TrackErrAoi4'][off4],
                                    gidxoff4, None)
                ax.plot(ontime3, onaoi3, 'b-',
                        linewidth=2, alpha=1.0, label='Aoi3 Active')
                ax.plot(ontime4, onaoi4, 'r-',
                        linewidth=2, alpha=1.0, label='Aoi4 Active')
                ax.plot(offtime3, offaoi3, 'b-', alpha=0.2,
                        linewidth=5, label='Aoi3 Off')
                ax.plot(offtime4, offaoi4, 'r-', alpha=0.2,
                        linewidth=5, label='Aoi4 Off')
            else:
                try:
                    ax.plot(reltime, table['TrackErrAoi3'], 'b-',
                            linewidth=2, alpha=1.0, label='Aoi3')
                    ax.plot(reltime, table['TrackErrAoi4'], 'r-',
                            linewidth=2, alpha=1.0, label='Aoi4')
                    log.debug('Found AOI keys: TrackErrAoi3, TrackErrAoi4')
                except KeyError:
                    log.debug('Found no AOI keys')

            ax.set_yscale('symlog', linthresh=10)
            ax.legend()
            ax.set_ylabel('TrackErr [arcsec]', fontsize=15)
            ax.fill_between(reltime, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=idx, facecolor='green', alpha=0.3)

            # Plot phases
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[2], wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            im = ax.imshow(r_phase.value, cmap='coolwarm', vmin=phmin,
                           vmax=phmax, origin='lower')
            cb = fig.colorbar(im, ax=ax)
            if auxphase:
                ax.set_title('R phase offset from %s, median=%.2f deg' %
                             (os.path.basename(ref_phase_file),
                              float(median_r_phase)))
                cb.set_label('Phase offset [degrees at %.2f Hz]' % user_freq,
                             fontsize=12)
            else:
                ax.set_title('R phase, median=%.2f deg' % median_r_phase)
                cb.set_label('Phase [degrees at %.2f Hz]' % user_freq,
                             fontsize=12)

            ax = fig.add_subplot(inner[1])
            im = ax.imshow(t_phase.value, cmap='coolwarm', vmin=phmin,
                           vmax=phmax, origin='lower')
            cb = fig.colorbar(im, ax=ax)
            if auxphase:
                ax.set_title('T phase offset from %s, median=%.2f deg' %
                             (os.path.basename(ref_phase_file),
                              float(median_t_phase)))
                cb.set_label('Phase offset [degrees at %.2f Hz]' % user_freq,
                             fontsize=12)
            else:
                ax.set_title('T phase, median=%.2f deg' % median_t_phase)
                cb.set_label('Phase [degrees at %.2f Hz]' % user_freq,
                             fontsize=12)

            # Plot HWP Angle
            ax = fig.add_subplot(outer[3])
            ax.plot(reltime, table['HWP Angle'], 'b-', label='HWP Angle')
            # Make the y-axis label, ticks and tick labels
            # match the line color
            ax.set_ylabel('HWP Angle', fontsize=15)
            ax.fill_between(reltime, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=idx, facecolor='green', alpha=0.3)

            # Plot R and T signal at specified location
            log.debug('Plotting R0 and T0 at '
                      'i={:d}, j={:d}'.format(detector_i, detector_j))
            ax1 = fig.add_subplot(outer[4])
            r_real = np.array(table['R array'])
            r_masked = self.sigma_mask(r_real[:, detector_j, detector_i],
                                       data_sigma, iters=data_iters)

            ax1.plot(reltime, r_masked, 'b-', label='R0')
            ax1.set_xlabel('Time [sec]', fontsize=15)
            ax1.set_ylabel('R0', color='b', fontsize=15)
            if data_iters > 0:
                ax1.set_title('Detector signals at i={:d}, j={:d}, '
                              '{:.1f}-sigma clipped w/ {:d} '
                              'iterations'.format(detector_i, detector_j,
                                                  data_sigma, data_iters))
            else:
                ax1.set_title('Detector signals at i={:d}, '
                              'j={:d}'.format(detector_j, detector_i))
            ax1.tick_params('y', colors='b')

            t_real = np.array(table['T array'])
            ax2 = ax1.twinx()
            t_masked = self.sigma_mask(t_real[:, detector_j, detector_i],
                                       data_sigma, iters=data_iters)

            ax2.plot(reltime, t_masked, 'r-', label='T0')
            ax2.set_ylabel('T0', color='r', fontsize=15)
            ax2.tick_params('y', colors='r')
            ax2.fill_between(reltime, ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=idx, facecolor='green', alpha=0.3)
        else:
            # Set up plot
            fig = Figure(figsize=(20, 24))
            FigureCanvas(fig)

            # Plot RA and Dec
            ax1 = fig.add_subplot(6, 1, 1)
            ax1.plot(reltime, table['RA'], 'b-', label='RA')
            ax1.set_xlabel('Time [sec]', fontsize=15)

            # Make the y-axis label, ticks and tick labels
            # match the line color
            ax1.set_ylabel('RA [deg]', color='b', fontsize=15)
            ax1.tick_params('y', colors='b')

            ax2 = ax1.twinx()
            ax2.plot(reltime, table['DEC'], 'r-', label='Dec')
            ax2.set_ylabel('Dec [deg]', color='r', fontsize=15)
            ax2.tick_params('y', colors='r')
            ax2.fill_between(reltime, ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=keep_where,
                             facecolor='green', alpha=0.3)

            # Plot title
            ax1.set_title('Plots for %s' %
                          os.path.basename(self.datain.filename),
                          fontsize=20, fontweight='bold')

            # Check for a door vignetting event
            # D.A. Harper's algorithm: for each nod index
            # and hwp index, compute standard
            # deviations for each pixel and then take array median
            # If ratio of imaginary part to real part exceeds
            # door threshold, mark on plot
            n_door_events = 0
            msg_counter = 0

            with np.errstate(invalid='ignore'):
                hwplist = sorted(np.unique(
                    table['HWP Index'][table['HWP Index'] >= 0]))
                nodlist = sorted(np.unique(
                    table['Nod Index'][table['Nod Index'] >= 0]))
            ax1.annotate('hwp ' + 'nod R_ratio T_ratio  ' * len(nodlist),
                         xy=(.10, .98),
                         xycoords='figure fraction',
                         horizontalalignment='left',
                         verticalalignment='top',
                         fontsize=14, color='black')
            for hwpind in hwplist:
                nomstr = ' {:3d} '.format(hwpind)
                for nodind in nodlist:
                    with np.errstate(invalid='ignore'):
                        index = (table['Nod Index'] == nodind) & \
                                (table['HWP Index'] == hwpind)
                    r_real = self.sigma_mask(
                        np.array(table['R array'][index,
                                 detector_j - 2:detector_j + 3,
                                 detector_i - 2:detector_j + 3]),
                        data_sigma, iters=data_iters)
                    r_imag = self.sigma_mask(
                        np.array(table['R array Imag'][index,
                                 detector_j - 2:detector_j + 3,
                                 detector_i - 2:detector_j + 3]),
                        data_sigma, iters=data_iters)
                    t_real = self.sigma_mask(
                        np.array(table['T array'][index,
                                 detector_j - 2:detector_j + 3,
                                 detector_i - 2:detector_j + 3]),
                        data_sigma, iters=data_iters)
                    t_imag = self.sigma_mask(
                        np.array(table['T array Imag'][index,
                                 detector_j - 2:detector_j + 3,
                                 detector_i - 2:detector_j + 3]),
                        data_sigma, iters=data_iters)
                    r_real_medstd = np.ma.median(np.ma.std(r_real, axis=0))
                    r_imag_medstd = np.ma.median(np.ma.std(r_imag, axis=0))
                    t_real_medstd = np.ma.median(np.ma.std(t_real, axis=0))
                    t_imag_medstd = np.ma.median(np.ma.std(t_imag, axis=0))
                    r_ratio = r_real_medstd / r_imag_medstd
                    t_ratio = t_real_medstd / t_imag_medstd
                    log.debug('hwp {:3d}, nod {:3d}, '
                              'r_ratio {:.3f}, t_ratio '
                              '{:.3f}'.format(hwpind, nodind,
                                              float(r_ratio),
                                              float(t_ratio)))
                    if (r_ratio > door_threshold) or \
                            (t_ratio > door_threshold):
                        n_door_events += 1
                        msg = ('Possible door vignetting:hwp {:d}, '
                               'nod {:d}. R_ratio {:.3f}, T_ratio '
                               '{:.3f}'.format(hwpind, nodind,
                                               float(r_ratio), float(t_ratio)))
                        ax1.annotate(msg,
                                     xy=(.90, .99 - n_door_events * 0.011),
                                     xycoords='figure fraction',
                                     horizontalalignment='right',
                                     verticalalignment='top',
                                     fontsize=15, color='red',
                                     fontweight='bold')
                        log.warning(msg)

                    nomstr += ' {:3d} {:8.3f} {:8.3f} '.format(
                        nodind, float(r_ratio), float(t_ratio))
                msg_counter += 1
                ax1.annotate(nomstr,
                             xy=(.10, .976 - msg_counter * 0.006),
                             xycoords='figure fraction',
                             horizontalalignment='left',
                             verticalalignment='top',
                             fontsize=14, color='blue')
                msg_counter += 1

            # Plot TrackErrAoi3 and 4
            ax = fig.add_subplot(6, 1, 2)
            aoikey = None
            if 'SofHkTrkaoi' in table.names:
                aoikey = 'SofHkTrkaoi'
            elif 'CentroidAoi' in table.names:
                aoikey = 'CentroidAoi'
            if aoikey is not None:
                log.debug('Found AOI key: {}'.format(aoikey))
                centroidaoi = table[aoikey]
                on3 = centroidaoi == 3.0
                on4 = centroidaoi == 4.0
                off3 = np.not_equal(centroidaoi, 3.0)
                off4 = np.not_equal(centroidaoi, 4.0)
                gidxon3 = self.get_gapindex(reltime[on3])
                gidxon4 = self.get_gapindex(reltime[on4])
                gidxoff3 = self.get_gapindex(reltime[off3])
                gidxoff4 = self.get_gapindex(reltime[off4])
                ontime3 = np.insert(reltime[on3], gidxon3, None)
                onaoi3 = np.insert(table['TrackErrAoi3'][on3], gidxon3, None)
                offtime3 = np.insert(reltime[off3], gidxoff3, None)
                offaoi3 = np.insert(table['TrackErrAoi3'][off3],
                                    gidxoff3, None)
                ontime4 = np.insert(reltime[on4], gidxon4, None)
                onaoi4 = np.insert(table['TrackErrAoi4'][on4], gidxon4, None)
                offtime4 = np.insert(reltime[off4], gidxoff4, None)
                offaoi4 = np.insert(table['TrackErrAoi4'][off4],
                                    gidxoff4, None)
                ax.plot(ontime3, onaoi3, 'b-',
                        linewidth=2, alpha=1.0, label='Aoi3 Active')
                ax.plot(ontime4, onaoi4, 'r-',
                        linewidth=2, alpha=1.0, label='Aoi4 Active')
                ax.plot(offtime3, offaoi3, 'b-',
                        linewidth=5, alpha=0.2, label='Aoi3 Off')
                ax.plot(offtime4, offaoi4, 'r-',
                        linewidth=5, alpha=0.2, label='Aoi4 Off')
            else:
                try:
                    ax.plot(reltime, table['TrackErrAoi3'],
                            'b-', linewidth=2, alpha=1.0,
                            label='Aoi3')
                    ax.plot(reltime, table['TrackErrAoi4'],
                            'r-', linewidth=2, alpha=1.0,
                            label='Aoi4')
                    log.debug('Found AOI keys: TrackErrAoi3, TrackErrAoi4')
                except KeyError:
                    log.debug('Found no AOI keys')
            ax.set_xlabel('Time [sec]', fontsize=15)
            ax.set_yscale('symlog', linthresh=10)
            ax.legend()
            ax.set_ylabel('TrackErr [arcsec]', fontsize=15)
            with np.errstate(invalid='ignore'):
                idx = (((chopmask & mask_bits) == 0)
                       & (nsamples >= min_samples))
            ax.fill_between(reltime, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=idx, facecolor='green', alpha=0.3)

            # Plot CentroidExpMsec
            ax = fig.add_subplot(6, 1, 3)
            try:
                ax.plot(reltime, table['CentroidExpMsec'],
                        'b-', label='CentroidExpMsec')
            except KeyError:
                log.debug('CentroidExpMsec not found')
            ax.set_xlabel('Time [sec]', fontsize=15)

            # Make the y-axis label, ticks and tick
            # labels match the line color
            ax.set_ylabel('CentroidExpMsec', fontsize=15)
            ax.fill_between(reltime, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=idx, facecolor='green', alpha=0.3)

            # Plot HWP Angle
            ax = fig.add_subplot(6, 1, 4)
            ax.plot(reltime, table['HWP Angle'], 'b-', label='HWP Angle')
            ax.set_xlabel('Time [sec]', fontsize=15)
            # Make the y-axis label, ticks and tick labels match the line color
            ax.set_ylabel('HWP Angle', fontsize=15)
            ax.fill_between(reltime, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=idx, facecolor='green', alpha=0.3)

            # Plot Nod Index and HWP Index together
            ax1 = fig.add_subplot(6, 1, 5)
            ax1.plot(reltime, table['Nod Index'], 'b-', label='Nod Index')
            # Make the y-axis label, ticks and tick labels match the line color
            ax1.set_ylabel('Nod Index', color='b', fontsize=15)
            ax1.tick_params('y', colors='b')

            ax2 = ax1.twinx()
            ax2.plot(reltime, table['HWP Index'], 'r-', label='HWP Index')
            ax2.set_ylabel('HWP Index', color='r', fontsize=15)
            ax2.tick_params('y', colors='r')

            ax2.fill_between(reltime, ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=idx, facecolor='green', alpha=0.3)

            # Plot R and T signal at specified location
            log.debug('Plotting R0 and T0 at i={:d}, '
                      'j={:d}'.format(detector_i, detector_j))
            ax1 = fig.add_subplot(6, 1, 6)
            r_real = np.array(table['R array'])
            r_masked = self.sigma_mask(
                r_real[:, detector_j, detector_i],
                data_sigma, iters=data_iters)

            ax1.plot(reltime, r_masked, 'b-', label='R0')
            ax1.set_xlabel('Time [sec]', fontsize=15)
            ax1.set_ylabel('R0', color='b', fontsize=15)
            if data_iters > 0:
                ax1.set_title('Detector signals at i={:d}, j={:d}, '
                              '{:.1f}-sigma clipped w/ {:d} '
                              'iterations'.format(detector_i, detector_j,
                                                  data_sigma, data_iters))
            else:
                ax1.set_title('Detector signals at i={:d}, '
                              'j={:d}'.format(detector_j, detector_i))
            ax1.tick_params('y', colors='b')

            t_real = np.array(table['T array'])
            ax2 = ax1.twinx()
            t_masked = self.sigma_mask(
                t_real[:, detector_j, detector_i],
                data_sigma, iters=data_iters)

            ax2.plot(reltime, t_masked, 'r-', label='T0')
            ax2.set_ylabel('T0', color='r', fontsize=15)
            ax2.tick_params('y', colors='r')
            ax2.fill_between(reltime, ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=idx, facecolor='green', alpha=0.3)

        # Save the image
        newfname = self.datain.filenamebegin + self.procname.upper() + \
            self.datain.filenameend
        pngname = os.path.splitext(newfname)[0] + '.png'
        if len(self.getarg('savefolder')):
            pngname = os.path.join(self.getarg('savefolder'),
                                   os.path.split(pngname)[1])
        with np.errstate(invalid='ignore'):
            fig.savefig(pngname)

        # record output filename
        self.auxout = [pngname]
        log.info('Saved result %s' % pngname)
