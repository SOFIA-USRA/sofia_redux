# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sky calibration pipeline step."""

import os

from astropy import log
from astropy.stats.sigma_clipping import sigma_clip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steploadaux import StepLoadAux
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent


__all__ = ['StepSkycal']


class StepSkycal(StepMIParent, StepLoadAux):
    """
    Generate a reference sky calibration file.

    This step divides scan mode sky flats by processed INT_CAL data,
    then combines and normalizes the sky flats to generate a
    master reference sky cal.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'skycal', and are named with
        the step abbreviation 'SCL'.

        Parameters defined for this step are:

        normalize : bool
            If set, sky cals are divided by their median values to
            normalize them before sigma-clipping.
        sigma_lower : float
            Lower sigma value to use for clipping.
        sigma_upper : float
            Upper sigma value to use for clipping.
        ttor : float
            Scale factor for the T array to the R array
        scalfitkeys : list of str
            Keys that need to match between data file reference SCAL file,
            for comparison.
        binning : str
            Binning method for comparison histogram.
        dclfile : str
            File name glob to match DCL files.  Default is
            'intcals/*DCL*.fits' to match DCL files in a folder
            named intcals, in the same directory as the input file.
        dclfitkeys : list of str
            Keys that need to match between DCL and data file.

        """
        # Name of the pipeline reduction step
        self.name = 'skycal'
        self.description = 'Make SkyCal'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'scl'

        # Clear Parameter list
        self.paramlist = []

        self.paramlist.append(['normalize', False,
                               'Normalize skycal for each array by '
                               'its median value.'])
        self.paramlist.append(['sigma_lower', 3.0,
                               'Lower sigma value to use for clipping.'])
        self.paramlist.append(['sigma_upper', 3.0,
                               'Upper sigma value to use for clipping.'])
        self.paramlist.append(['ttor', 1.275,
                               'Scale factor for T/R flatfield'])
        self.paramlist.append(['bins', 'fd',
                               'Binning method for comparison histogram.'])
        self.paramlist.append(['scalfitkeys', ['SPECTEL1'],
                               'Match keys for SCAL comparison.'])
        self.paramlist.append(['pixfile', 'pixel*.dat',
                               'File name pattern for pixeldata files.'])
        self.paramlist.append(['pixfitkeys', [],
                               'Match keys for pixeldata files.'])
        self.paramlist.append(['ref_pixpath', '',
                               'Path to reference pixel files.'])
        self.paramlist.append(['ref_pixfile', [],
                               'Reference pixel files, one per band.'])

        # Get parameters for StepLoadAux, replace auxfile with dclfile
        self.loadauxsetup('dcl')

    def loadauxname(self, auxpar='', data=None, multi=False):
        """
        Search for files matching auxfile.

        Overrides the default function in order to make flat path
        relative to data output directory if necessary.

        Parameters
        ----------
        auxpar : str, optional
            A name for the aux file parameter to use. This
            allows loadauxfiles to be used multiple times
            in a given pipe step (for example for darks and
            flats). Default value is self.auxpar which is set
            by loadauxsetup().
        data : DataFits or DataText, optional
            A data object to match the auxiliary file to.
            If no data is specified, self.datain is used (for
            Multi Input steps self.datain[0]).
        multi : bool, optional
            If set, a list of file names is returned instead of a single
            file name.

        Returns
        -------
        str or list of str
            The matching auxiliary file(s).
        """
        # override loadauxname to make aux path relative to
        # data output directory if necessary

        # Set auxpar
        if len(auxpar) == 0:
            auxpar = self.auxpar

        # Get parameters
        auxfile = os.path.expandvars(self.getarg(auxpar + 'file'))
        if not os.path.isabs(auxfile):
            # if input folder is not an absolute path, make it
            # relative to the data location
            auxfile = os.path.join(
                os.path.dirname(self.datain[0].filename),
                auxfile)
        return self._loadauxnamefile(auxfile, auxpar, data, multi,
                                     backup=False)

    def read_refpix(self):
        """
        Read a reference pixel file from the parameters.

        The parameter is expected to be defined as a list, with
        one entry for each HAWC band.  The correct value for the
        input data is selected from the list.

        Returns
        -------
        pixfile : str
            Path to the reference file.
        """
        pixfiles = self.getarg('ref_pixfile')
        pixpath = os.path.expandvars(self.getarg('ref_pixpath'))
        waveband = self.datain[0].getheadval('spectel1')
        bands = ['A', 'B', 'C', 'D', 'E']
        try:
            idx = bands.index(waveband[-1])
        except (ValueError, IndexError):
            # waveband not in list
            msg = f'Cannot parse waveband: {waveband}'
            log.error(msg)
            raise ValueError(msg)
        try:
            pixfile = os.path.expandvars(pixfiles[idx])
            if pixfile == '':
                log.warning(f'No reference pixel file '
                            f'specified for {waveband}.')
                pixfile = None
            else:
                pixfile = os.path.join(pixpath, pixfile)
                if not os.path.isfile(pixfile):
                    msg = f'Pixel file {pixfile} not found'
                    log.error(msg)
                    raise ValueError(msg)
                else:
                    log.info(f'Found default pixel file: {pixfile}')
        except IndexError:
            log.warning(f'No reference pixel file specified for {waveband}.')
            pixfile = None

        return pixfile

    @staticmethod
    def _log_stats(rgain, tgain, rbad, tbad):
        log.info('')
        log.info("     R Array Stats:")
        log.info("         Min       : {:4.2f}".format(np.nanmin(rgain)))
        log.info("         Max       : {:4.2f}".format(np.nanmax(rgain)))
        log.info("         Median    : {:4.2f} \u00B1 {:4.2f}".
                 format(np.nanmedian(rgain), np.nanstd(rgain)))
        log.info("         Bad Pixel : {:<6d}".format(np.sum(rbad == 1)))

        log.info("     T Array Stats:")
        log.info("         Min       : {:4.2f}".format(np.nanmin(tgain)))
        log.info("         Max       : {:4.2f}".format(np.nanmax(tgain)))
        log.info("         Median    : {:4.2f} \u00B1 {:4.2f}".
                 format(np.nanmedian(tgain), np.nanstd(tgain)))
        log.info("         Bad Pixel : {:<6d}".format(np.sum(tbad == 2)))

        log.info("     T/R Ratio: {:5.3f}".format(
                 np.nanmedian(tgain) / np.nanmedian(rgain)))
        log.info('')

    def _make_flat_plot(self, auxname, rgain, tgain, basename):
        scal = DataFits(auxname)
        prgain = scal.imageget('R ARRAY GAIN')
        ptgain = scal.imageget('T ARRAY GAIN')
        try:
            prbad = scal.imageget('R BAD PIXEL MASK')
        except ValueError:  # pragma: no cover
            prbad = np.zeros_like(prgain)
        try:
            ptbad = scal.imageget('T BAD PIXEL MASK')
        except ValueError:  # pragma: no cover
            ptbad = np.zeros_like(ptgain)

        log.info('')
        log.info('Default flat statistics:')
        self._log_stats(prgain, ptgain, prbad, ptbad)

        fig = Figure()
        FigureCanvas(fig)
        bins = self.getarg('bins')

        ax = fig.add_subplot(2, 2, 1)
        ax.hist(prgain.flatten(), bins=bins, label='R Gain')
        ax.set_xlim([0, 2.0])
        ax.legend()
        ax.set_title('Default %.2f $\\pm$ %.2f' %
                     (float(np.nanmedian(prgain)),
                      float(np.nanstd(prgain))))

        ax = fig.add_subplot(2, 2, 2)
        ax.hist(ptgain.flatten(), bins=bins, label='T Gain')
        ax.set_xlim([0.5, 2.5])
        ax.legend()
        ax.set_title('Default %.2f $\\pm$ %.2f' %
                     (float(np.nanmedian(ptgain)),
                      float(np.nanstd(ptgain))))

        ax = fig.add_subplot(2, 2, 3)
        try:
            ax.hist(rgain.flatten(), bins=bins, label='R Gain')
        except ValueError:  # pragma: no cover
            log.warning('Could not plot R histogram')
        ax.set_xlim([0, 2.0])
        ax.legend()
        ax.set_title('Scan map %.2f $\\pm$ %.2f' %
                     (float(np.nanmedian(rgain)),
                      float(np.nanstd(rgain))))

        ax = fig.add_subplot(2, 2, 4)
        try:
            ax.hist(tgain.flatten(), bins=bins, label='T Gain')
        except ValueError:  # pragma: no cover
            log.warning('Could not plot T histogram')
        ax.set_xlim([0.5, 2.5])
        ax.legend()
        ax.set_title('Scan map %.2f $\\pm$ %.2f' %
                     (float(np.nanmedian(tgain)),
                      float(np.nanstd(tgain))))

        fig.tight_layout()

        # output file name
        outfile = os.path.splitext(basename)[0] + '_comparison.png'

        # save figure
        fig.savefig(outfile, dpi=200)
        fig.clear()
        log.info('Saved result %s' % outfile)
        return outfile

    @staticmethod
    def _average_pixel_data(pixfiles):
        f_tbl = {}
        f_cols = ['ch', 'gain', 'weight', 'flag', 'eff',
                  'Gmux1', 'Gmux2', 'idx', 'sub', 'row', 'col']
        a_cols = ['ch', 'Gmux1', 'Gmux2', 'idx', 'sub', 'row', 'col']

        new_df = {}
        for i in range(len(pixfiles)):
            f_id = os.path.splitext(
                os.path.basename(pixfiles[i]))[0].split('-')[-1]
            f_tbl[f_id] = pd.read_table(pixfiles[i], names=f_cols, comment='#')

            m_cols = ['ch', f'gain_{i}', f'weight_{i}',
                      f'flag_{i}', f'eff_{i}',
                      'Gmux1', 'Gmux2', 'idx', 'sub',
                      'row', 'col']
            if i == 0:
                new_df = pd.read_table(pixfiles[i], names=m_cols, comment='#')
            else:
                tmp_df = pd.read_table(pixfiles[i], names=m_cols, comment='#')
                new_df = pd.merge(new_df, tmp_df, how='outer', left_on=a_cols,
                                  right_on=a_cols).sort_values(['idx'])

        # keep gain, weight, and eff for averaging;
        # directly or-combine character flags
        gain = []
        weight = []
        eff = []
        flags = None
        for key in new_df.keys():
            if key.find('gain') != -1:
                gain.append(new_df[key])
            elif key.find('weight') != -1:
                weight.append(new_df[key])
            elif key.find('eff') != -1:
                eff.append(new_df[key])
            elif key.find('flag') != -1:
                new_flags = list(new_df[key].fillna('-'))
                if flags is None:
                    flags = new_flags
                else:
                    # join unique characters
                    for i in range(len(flags)):
                        join_flags = set(flags[i]) | set(new_flags[i])
                        all_flags = ''.join(sorted(join_flags))
                        # should contain - only if there are no other flags
                        if all_flags != '-':
                            all_flags = all_flags.replace('-', '')
                        flags[i] = all_flags

        new_df['avg_gain'] = np.nanmean(gain, axis=0)
        new_df['avg_weight'] = np.nanmean(weight, axis=0)
        new_df['avg_eff'] = np.nanmean(eff, axis=0)
        new_df['flag'] = flags

        n_cols = ['ch', 'avg_gain', 'avg_weight', 'flag', 'avg_eff',
                  'Gmux1', 'Gmux2', 'idx', 'sub', 'row', 'col']
        new_df = new_df[n_cols]

        return f_tbl, new_df

    @staticmethod
    def _get_pix_header(pixfiles):
        hdr = []
        scan_line = 3
        for i, pixfile in enumerate(pixfiles):
            with open(pixfile, 'r') as fh:
                for line in fh.readlines():
                    if line.startswith('#'):
                        if i == 0:
                            hdr.append(line)
                        if 'Scan' in line:
                            # keep the index for the "Scan" line
                            # from the first file
                            if i == 0:
                                scan_line = len(hdr) - 1
                            else:
                                # for all other files, concat in the
                                # scan number to this description
                                nbr = line.split('-')[-1]
                                hdr[scan_line] = hdr[scan_line].rstrip() \
                                    + ',' + nbr
        return hdr

    @staticmethod
    def _gain_stat(tag, df, idx=None, key='gain'):
        gain = df[key]
        x_med = np.median(gain)
        x_std = gain.std()
        x_max = gain.max()
        x_min = gain.min()

        if idx is not None:
            gain = df[df.idx.isin(idx)][key]

        x_gain = sigma_clip(gain, sigma=3.0)

        log.info(f"    {tag} Median: {x_med:.3f} \u00B1 {x_std:.3f}, "
                 f"Max: {x_max:.3f}, Min: {x_min:.3f}")
        x_gs = {'ID': tag, 'Gain': x_gain, 'Median': x_med,
                'StdDev': x_std, 'Max': x_max, 'Min': x_min}
        return x_gs

    def _plot_gains(self, data, average, default, basename):
        # first plot: histograms
        fig = Figure(figsize=(12, 8))
        FigureCanvas(fig)
        fig.suptitle('Histogram of pixel*.dat file')

        # layout from number of files + average + default
        n = len(data) + 2
        ncol = int(np.ceil(np.sqrt(n)))
        nrow = int(np.ceil(float(n) / ncol))

        # put each data set in a new subplot
        ax0 = None
        bins = self.getarg('bins')
        for i, dataset in enumerate(data + [average, default]):
            # create an axis
            if i == 0:
                ax = fig.add_subplot(nrow, ncol, i + 1)
                ax0 = ax
            else:
                ax = fig.add_subplot(nrow, ncol, i + 1,
                                     sharex=ax0, sharey=ax0)
            # handle missing default
            if dataset is None:
                ax.text(0.5, 0.5, '(No default provided)',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
                continue

            # histogram the gain and label with stats
            ax.hist(dataset['Gain'], bins=bins,
                    label=f"    {dataset['ID']}\n"
                          f"{dataset['Median']:.3f} \u00B1 "
                          f"{dataset['StdDev']:.3f}")
            ax.set(xlabel='Gain', ylabel='Counts')
            ax.legend()
        fig.tight_layout()

        # save figure
        outhist = os.path.splitext(basename)[0] + '_histogram.png'
        fig.savefig(outhist, dpi=200)
        log.info(f'Saved result {outhist}')

        # no second plot if default is None
        if default is None or len(default['Gain']) == 0:
            return outhist,

        # second plot: pixel-to-pixel comparison
        fig2 = Figure(figsize=(12, 8))
        FigureCanvas(fig2)
        fig2.suptitle('Pixel-to-pixel comparison of pixel*.dat file')

        # layout from number of files + average
        ncol = 1
        nrow = len(data) + 1

        ax0 = None
        minval = np.nanmin(default['Gain'])
        maxval = np.nanmax(default['Gain'])
        for i, dataset in enumerate(data + [average]):
            if i == 0:
                ax = fig2.add_subplot(nrow, ncol, i + 1)
                ax0 = ax
            else:
                ax = fig2.add_subplot(nrow, ncol, i + 1,
                                      sharex=ax0, sharey=ax0)

            # plot individual and average pixel gains
            # against the default
            ax.plot(default['Gain'], dataset['Gain'], '.')
            ax.set(xlabel=default['ID'], ylabel=dataset['ID'])

            # plot default line
            ax.plot([minval, maxval], [minval, maxval], '-r')

        fig2.tight_layout()

        # save figure
        outpix = os.path.splitext(basename)[0] + '_pix2pix.png'
        fig2.savefig(outpix, dpi=200)
        log.info(f'Saved result {outpix}')

        return outhist, outpix

    @staticmethod
    def _write_pixfile(outfile, header, av_df):
        with open(outfile, 'w') as fh:
            for line in header:
                fh.write(line)

        # format data for text file
        av_df['avg_gain'] = av_df['avg_gain'].map('{:.3f}'.format)
        av_df['avg_weight'] = av_df['avg_weight'].map('{:.3E}'.format)
        av_df['avg_eff'] = av_df['avg_eff'].map('{:.3f}'.format)
        av_df['Gmux1'] = av_df['Gmux1'].map('{:.3f}'.format)
        av_df['Gmux2'] = av_df['Gmux2'].map('{:.3f}'.format)
        av_df.to_csv(outfile, sep='\t', index=None, header=0, mode='a')
        log.info(f'Saved result {outfile}')

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a multiple-in single-out (MISO) step:
        self.datain should be a list of DataFits, and output
        will also be a single DataFits, stored in self.dataout.

        The process is to make a skyflat for the DRP pipeline:

        1. Divide R and T gains by the INTCAL DCL files closest in
           time, to derive master sky gains.
        2. Mean combine all sky gains and optionally normalize by
           the median value.
        3. Sigma clip the gains arrays to identify additional bad pixels.
        4. Scale the T array to the R array.
        5. Compare the produced skyflat with the current default, saving
           a comparison image to disk with the filename as
           basename + '_comparison.png'.

        Then, to make a pixeldata (flat/bad pixel table) for the scan
        mode pipeline:

        1. Read in pixeldata files.
        2. Mean-combine gains for each pixel.
        3. Or-combine flags for each pixel.
        4. Generate plots comparing the individual and averaged pixel
           data files to an existing default file.  Plot file names
           are basename + '_histogram.png' and basename + '_pix2pix.png'.
        5. Save the combined table to disk, with the filename as
           basename + '.dat'.

        """
        # loop over input data, dividing by intcal
        rgains = []
        tgains = []
        for data in self.datain:
            # get matching DCL, closest in time to the input file
            try:
                # this will log an error message if no DCL is found
                dcl = self.loadauxfile(data=data)
            except ValueError:
                dcl = None

            # get DCL images
            if dcl is not None:
                dcl_rgain = dcl.imageget('R ARRAY GAIN')
                dcl_tgain = dcl.imageget('T ARRAY GAIN')
                dcl_rbad = dcl.imageget('R BAD PIXEL MASK')
                dcl_tbad = dcl.imageget('T BAD PIXEL MASK')
                dcl_rgain[dcl_rbad != 0] = np.nan
                dcl_tgain[dcl_tbad != 0] = np.nan
            else:
                # allow flat to be saved, even if no DCL is
                # available
                dcl_rgain = 1.0
                dcl_tgain = 1.0

            # get sky flat images
            sky_rgain = data.imageget('R ARRAY GAIN')
            sky_tgain = data.imageget('T ARRAY GAIN')
            sky_rbad = data.imageget('R BAD PIXEL MASK')
            sky_tbad = data.imageget('T BAD PIXEL MASK')

            # set NaNs from bad masks
            sky_rgain[sky_rbad != 0] = np.nan
            sky_tgain[sky_tbad != 0] = np.nan

            # divide sky flat by intcal
            rgains.append(sky_rgain / dcl_rgain)
            tgains.append(sky_tgain / dcl_tgain)

        # mean combine the flats, allowing NaNs to propagate
        rgain = np.mean(rgains, axis=0)
        tgain = np.mean(tgains, axis=0)

        # normalize if desired
        if self.getarg('normalize'):
            log.info('')
            log.info('Normalizing gain arrays by median value.')
            log.info(f'    R median: {np.nanmedian(rgain)}')
            log.info(f'    T median: {np.nanmedian(tgain)}')
            rgain /= np.nanmedian(rgain)
            tgain /= np.nanmedian(tgain)

        # make bad pixel masks from the new gains
        rbad = np.zeros(rgain.shape, dtype=int)
        rbad[np.isnan(rgain)] = 1
        tbad = np.zeros(tgain.shape, dtype=int)
        tbad[np.isnan(tgain)] = 2

        # log flat statistics
        log.info('')
        log.info('Flat statistics before sigma clip:')
        self._log_stats(rgain, tgain, rbad, tbad)

        # sigma clip both arrays for additional bad pixels
        sigma_lower = self.getarg('sigma_lower')
        sigma_upper = self.getarg('sigma_upper')
        rgain = sigma_clip(rgain, masked=True, sigma_upper=sigma_upper,
                           sigma_lower=sigma_lower, copy=False)
        rgain = rgain.filled(np.nan)
        tgain = sigma_clip(tgain, masked=True, sigma_upper=sigma_upper,
                           sigma_lower=sigma_lower, copy=False)
        tgain = tgain.filled(np.nan)

        # propagate to bad mask
        rbad[np.isnan(rgain)] = 1
        tbad[np.isnan(tgain)] = 2

        # scale T to R
        ttor = self.getarg('ttor')
        tgain *= ttor * np.nanmedian(rgain) / np.nanmedian(tgain)

        # log flat statistics again
        log.info('Flat statistics after sigma clip and T scaling:')
        self._log_stats(rgain, tgain, rbad, tbad)

        # make output data by copying first input
        self.dataout = self.datain[0].copy()
        for data in self.datain[1:]:
            self.dataout.mergehead(data)

        # update gain and mask images
        self.dataout.imageset(rgain, 'R ARRAY GAIN')
        self.dataout.imageset(tgain, 'T ARRAY GAIN')
        self.dataout.imageset(rbad, 'R BAD PIXEL MASK')
        self.dataout.imageset(tbad, 'T BAD PIXEL MASK')

        # get the basename for side effect output files
        tmp = self.dataout.copy()
        self.updateheader(tmp)
        basename = tmp.filename

        # compare to existing pipeline flat
        log.info('')
        log.info('Comparison to default flat:')
        try:
            auxfile = os.path.expandvars(
                self.dataout.config['mkflat']['scalfile'])
            auxname = self._loadauxnamefile(auxfile, 'scal',
                                            self.dataout, False,
                                            backup=False)
        except (KeyError, ValueError):
            auxname = None
            log.warning('No default pipeline flat found; '
                        'comparison plot will not be generated.')

        # make SCL comparison plot
        self.auxout = []
        if auxname is not None:
            flat_plot = self._make_flat_plot(auxname, rgain, tgain, basename)
            self.auxout.append(flat_plot)

        # get pixeldata files for combination: any files that match the
        # 'pixfile' glob parameter
        log.info('')
        log.info('Get pixel data files:')
        try:
            pixfiles = self.loadauxname(auxpar='pix', data=None, multi=True)
        except ValueError:
            # just return if no pixeldata found
            log.warning("No pixel data files found; skipping pixel file "
                        "combination.")
            return

        # also get a default pixel file, if available
        de_file = self.read_refpix()
        log.info('')

        # average all input pixel data files
        f_df, av_df = self._average_pixel_data(pixfiles)

        # compare to existing pixeldata
        if de_file is not None:
            col_names = ('ch', 'gain', 'weight', 'flag', 'eff',
                         'Gmux1', 'Gmux2', 'idx', 'sub', 'row', 'col')
            de_df = pd.read_table(de_file, names=col_names, comment='#')

            # get matching pixels for comparison purposes
            idx = set(de_df['idx'])
            for key in f_df:
                idx &= set(f_df[key]['idx'])
        else:
            de_df = None
            idx = None

        log.info('Pixel data stats:')
        log.info('')
        stats = []
        for key in f_df:
            stats.append(self._gain_stat(f'File {key}', f_df[key], idx=idx))

        log.info('    --')
        avg_stats = self._gain_stat('Average', av_df, idx=idx, key='avg_gain')
        if de_df is not None:
            default_stats = self._gain_stat('Default', de_df, idx=idx)
        else:
            default_stats = None
        log.info('')

        plots = self._plot_gains(stats, avg_stats, default_stats, basename)
        self.auxout.extend(list(plots))

        # compile output header from input pixel files
        header = self._get_pix_header(pixfiles)

        # write output file
        outfile = os.path.splitext(basename)[0] + '.dat'
        self._write_pixfile(outfile, header, av_df)
        self.auxout.append(outfile)
        log.info('')
