# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Scanning polarimetry image reconstruction pipeline step."""

from datetime import datetime
import os
import warnings

from astropy import log
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepmoparent import StepMOParent
from sofia_redux.instruments.hawc.steps.stepscanmap import StepScanMap
from sofia_redux.scan.reduction.reduction import Reduction

__all__ = ['StepScanMapPol']


class StepScanMapPol(StepMOParent):
    """
    Reconstruct an image from scanning polarimetry data.

    This step requires that scanning polarimetry data are taken with
    four HWP angles, one per input file. Sets of data are identified
    by the SCRIPTID keyword, and are passed to the scan map algorithm
    one group at a  time. R0 and T0 output maps are created for each
    input file to.  This step assembles them into a single file with
    R0 and T0 extensions (DATA, ERROR, and EXPOSURE) for each HWP angle.
    One output file is produced for each set of 4 HWP angles.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'scanmappol', and are named with
        the step abbreviation 'SMP'.

        Parameters defined for this step are:

        vpa_tol : float
            If differences between telescope angles (VPA) are more
            than this value, this step will issue a warning.
        use_frames : str
            Frames to use from the reduction. Specify a particular
            range, as '400:-400', or '400:1000'.
        deep : bool
            If set, faint point-like emission is prioritized.
        faint : bool
            If set, faint emission (point-like or extended) is prioritized.
        extended : bool
            If set, extended emission is prioritized.
            This may increase noise on large angular scales.
        options : str
            Additional options to pass to the scan map algorithm.
        """
        # Name of the pipeline reduction step
        self.name = 'scanmappol'
        self.description = 'Construct Scan Map'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'smp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['save_intermediate', False,
                               'Save individual scanmap frames'])
        self.paramlist.append(['vpa_tol', 5.0,
                               'Tolerance for matching VPA angle'])
        self.paramlist.append(['use_frames', '',
                               "Frames to use from reduction. "
                               "Specify a particular range, as "
                               "'400:-400', or '400:1000'."])
        self.paramlist.append(['deep', False,
                               'Attempt to recover faint point-like '
                               'emission'])
        self.paramlist.append(['faint', False,
                               'Attempt to recover faint emission '
                               '(point-like or extended)'])
        self.paramlist.append(['extended', False,
                               'Attempt to recover extended emission '
                               '(may increase noise)'])
        self.paramlist.append(['options', '',
                               'Additional options for scan reconstruction'])

    def assemble_scanpol(self, dataout, basehead):
        """
        Assemble scan maps into a single file.

        HWP angles are determined from the HWPINIT key in the output
        file headers. If 4 unique angles with two subarray images each
        are not found, an error is raised.

        Parameters
        ----------
        dataout : `list` of DataFits
            Output scan maps to assemble.
        basehead : fits.Header
            Baseline header to update for output file.

        Returns
        -------
        DataFits
            The assembled data file.

        Raises
        ------
        ValueError
            If output is missing keywords HWPINIT, SUBARRAY, or TELVPA,
            or if 4 unique HWP angles are not found, or if the wrong
            number of subarray images are found (should be one R and one T).
        """
        errmsg = 'Unexpected scan map output for SCANPOL mode. ' \
                 'Check SUBARRAY, HWPINIT, SCRIPTID, TELVPA.'

        # sort by HWP and subarray
        hwp = {}
        dates = []
        actual_hwp = {}
        actual_vpa = {}
        for outfile in dataout:
            # get the date, for determining header order
            try:
                dateobs_str = outfile.getheadval('DATE-OBS')
                dateobs = datetime.strptime(dateobs_str.split('.')[0],
                                            "%Y-%m-%dT%H:%M:%S")
                dates.append(dateobs)
            except (KeyError, ValueError):
                dates.append(datetime.now())

            try:
                hwpinit = float(outfile.getheadval('HWPINIT'))
                subarray = outfile.getheadval('SUBARRAY')
                telvpa = float(outfile.getheadval('TELVPA'))
            except KeyError:
                log.error('Scan map output missing required keywords: '
                          'HWPINIT, SUBARRAY, or TELVPA')
                raise ValueError(errmsg)
            hwpround = int(np.round(hwpinit / 10, decimals=0) * 10)

            # telvpa + 180 for storage
            rotvpa = telvpa + 180.
            if rotvpa > 360:
                rotvpa = rotvpa - 360.

            if hwpround in hwp:
                if subarray in hwp[hwpround]:
                    log.error('Too many subarray={} images for HWP '
                              'angle={}.'.format(subarray, hwpround))
                    raise ValueError(errmsg)
                hwp[hwpround][subarray] = outfile
                actual_hwp[hwpround].append(hwpinit)
                actual_vpa[hwpround].append(rotvpa)
            else:
                hwp[hwpround] = {subarray: outfile}
                actual_hwp[hwpround] = [hwpinit]
                actual_vpa[hwpround] = [rotvpa]

        # check that there are exactly 4 angles
        # (NOTE: DRP supports multiples of 4, but we don't currently
        # use more than 4, so keep it simple here.)
        hwps = sorted(hwp.keys())
        nhwp = len(hwps)
        if nhwp != 4:
            log.error('Must be exactly 4 HWP '
                      'angles. Found: {}.'.format(hwps))
            raise ValueError(errmsg)

        # now store R and T at each angle

        df = DataFits(config=self.config)

        # merge in all the input headers
        sort_order = np.argsort(dates)
        log.debug(f'Setting header from '
                  f'{dataout[sort_order[0]].filename}')
        df.setheader(dataout[sort_order[0]].header)
        exptime = df.getheadval('EXPTIME')
        ref_arr = df.getheadval('SUBARRAY')
        for idx in sort_order[1:]:
            outdf = dataout[idx]

            # for time accounting, sum exposure time only over one
            # simultaneously observed subarray
            if outdf.getheadval('SUBARRAY') == ref_arr:
                exptime += outdf.getheadval('EXPTIME')
            log.debug('Merging header for '
                      '{}'.format(outdf.filename))
            df.mergehead(outdf)

        # set the summed exptime
        df.setheadval('EXPTIME', exptime, 'Total on-source exposure time [s]')

        # add a few keywords needed for polarimetry processing
        df.setheadval('NHWP', nhwp, 'Number of HWP Angles')
        df.setheadval('PIXSCAL', df.getheadval('CDELT2') * 3600.,
                      'Pixel scale [arcsec]')

        wcs0 = {}
        for i, hwp_angle in enumerate(hwps):

            if 'R0' not in hwp[hwp_angle]:
                log.error('Missing R0 at HWP angle {}'.format(hwp_angle))
                raise ValueError(errmsg)
            if 'T0' not in hwp[hwp_angle]:
                log.error('Missing T0 at HWP angle {}'.format(hwp_angle))
                raise ValueError(errmsg)

            rfile = hwp[hwp_angle]['R0']
            rdata = rfile.imageget('PRIMARY IMAGE')
            rerr = rfile.imageget('NOISE')
            rexp = rfile.imageget('EXPOSURE')
            rwcs = WCS(rfile.header)

            tfile = hwp[hwp_angle]['T0']
            tdata = tfile.imageget('PRIMARY IMAGE')
            terr = tfile.imageget('NOISE')
            texp = tfile.imageget('EXPOSURE')
            twcs = WCS(tfile.header)

            # divide by 2 for normalization with
            # standard scan map calibration factors
            rdata /= 2.0
            rerr /= 2.0
            tdata /= 2.0
            terr /= 2.0

            # set image extensions, with appropriate WCS keys
            rd_ext = 'DATA R HWP%d' % i
            td_ext = 'DATA T HWP%d' % i
            re_ext = 'ERROR R HWP%d' % i
            te_ext = 'ERROR T HWP%d' % i
            rx_ext = 'EXPOSURE R HWP%d' % i
            tx_ext = 'EXPOSURE T HWP%d' % i
            if i == 0:
                # don't overwrite primary header
                df.imageset(rdata, rd_ext)
                wcs0 = rwcs.to_header()
            else:
                df.imageset(rdata, rd_ext, rwcs.to_header())
            df.imageset(tdata, td_ext, twcs.to_header())
            df.imageset(rerr, re_ext, rwcs.to_header())
            df.imageset(terr, te_ext, twcs.to_header())
            df.imageset(rexp, rx_ext, rwcs.to_header())
            df.imageset(texp, tx_ext, twcs.to_header())

            # set a few more header values as needed
            d_ext = [rd_ext, td_ext]
            e_ext = [re_ext, te_ext]
            x_ext = [rx_ext, tx_ext]
            r_ext = [rd_ext, re_ext, rx_ext]
            t_ext = [td_ext, te_ext, tx_ext]

            # set BUNIT in data and error extensions (usually counts)
            try:
                bunit = rfile.header['BUNIT']
            except KeyError:
                log.warning('BUNIT not found in scan map output')
                bunit = 'UNKNOWN'
            for x in d_ext + e_ext:
                df.setheadval('BUNIT', bunit,
                              comment='Data units', dataname=x)
            # set BUNIT in exposure extensions (seconds)
            for x in x_ext:
                df.setheadval('BUNIT', 's',
                              comment='Data units', dataname=x)
            # set HWP angle in all extensions
            for x in r_ext + t_ext:
                hwpinit = float(np.mean(actual_hwp[hwp_angle]))
                df.setheadval('HWPINIT', hwpinit,
                              comment='HWP angle [deg]',
                              dataname=x)
            # set SUBARRAY=R0 in R extensions
            for x in r_ext:
                df.setheadval('SUBARRAY', 'R0',
                              comment='Subarrays in image',
                              dataname=x)
            # set SUBARRAY=T0 in T extensions
            for x in t_ext:
                df.setheadval('SUBARRAY', 'T0',
                              comment='Subarrays in image',
                              dataname=x)
            # set average VPA angle
            for x in r_ext + t_ext:
                vpos = float(np.mean(actual_vpa[hwp_angle]))
                df.setheadval('VPOS_ANG', vpos,
                              comment='Average Array VPA [deg]',
                              dataname=x)

        # Copy and save headers
        scnhead = df.header.copy()
        df.header = basehead.copy()

        # Merge scanmap headers
        StepScanMap.merge_scan_hdr(df, scnhead, self.datain)

        # Update primary header with proper WCS
        for key, value in wcs0.items():
            df.setheadval(key, value)

        # Update SOFIA mandated keywords (since this is first pipe step)
        obsid = 'P_' + basehead['OBS_ID']
        df.setheadval('OBS_ID', obsid)
        df.setheadval('PIPELINE', 'HAWC_DRP')

        return df

    def _run_scanmap(self, argset):
        """
        Run a scan reduction on a data set.

        Helper function to create a map from a group and assemble
        the output.

        Parameters
        ----------
        argset : tuple
            This tuple should have 4 elements: a list of strings, a dict,
            a fits.Header, and a string. These are the input
            files, the keyword parameters, a baseline FITS header,
            and the group identification string.

        Returns
        -------
        df : DataFits
            The output data object.
        group : str
            The group identifier.
        """
        infiles, kwargs, basehead, group = argset

        # log input
        log.debug(f'All provided options: {kwargs}')
        log.debug(f'Input files: {infiles}')

        # run map reduction
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            reduction = Reduction('hawc_plus')
            output_list = reduction.run(infiles, **kwargs)

        # read output file(s)
        if (output_list is None) or (None in output_list):
            log.error('No output created.')
            return None, group
        elif isinstance(output_list, fits.HDUList):
            log.error('Unexpected output for scan pol mode. '
                      'Check INSTCFG.')
            return None, group

        dataout = []
        for hdul in output_list:
            df = DataFits(config=self.config)
            df.load(hdul=hdul)
            dataout.append(df)

        try:
            df = self.assemble_scanpol(dataout, basehead)
        except ValueError as err:
            log.error('Encountered error in assembly:')
            log.error(err)
            df = None
        return df, group

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a multi-in multi-out (MIMO) step:
        self.datain should be a list of DataFits, and output
        will also be a list of DataFits, stored in self.dataout.

        The process is:

        1. Group input data by the header keyword SCRIPTID.
        2. Assemble the scan map options from input parameters.
        3. Call the iterative scan map reconstructor. Internally,
           the process will run multiple times, with appropriate arguments
           for each HWP angle. It will create multiple output maps:
           one for each of the R0 and T0 subarrays at each HWP angle.
           For the standard four-angle case, 8 maps are created.
        4. Assemble the output maps into a single output file per
           group, with a separate extension for flux, error, and
           exposure at each HWP angle and subarray.
        5. Merge header keywords from the input FITS files to generate
           an appropriate output header for each output file.

        """
        vtol = self.getarg('vpa_tol')

        # group data by scriptid
        groups = {}
        vpa = {}
        fnum = {}
        for df in self.datain:
            try:
                scriptid = df.getheadval('SCRIPTID')
            except KeyError:
                scriptid = 'UNKNOWN'
            if scriptid in groups:
                groups[scriptid].append(df)
                vpa[scriptid].append(df.getheadval('TELVPA'))
                if df.filenum is not None:
                    fnum[scriptid].append(df.filenum)
                else:
                    fnum[scriptid].append('UNKNOWN')
            else:
                groups[scriptid] = [df]
                vpa[scriptid] = [df.getheadval('TELVPA')]
                if df.filenum is not None:
                    fnum[scriptid] = [df.filenum]
                else:
                    fnum[scriptid] = ['UNKNOWN']

        # if scriptids are all different (old data), just run
        # the data together
        counts = np.array([len(groups[s]) for s in groups])
        if len(self.datain) > 1 and not np.any(counts > 1):
            log.warning('No matching SCRIPTIDs. '
                        'Running all data together.')
            groups = {'ALL': self.datain}
            vpa = {'ALL': [v[0] for v in vpa.values()]}
            fnum = {'ALL': [v[0] for v in fnum.values()]}

        # collect input options in dict
        kwargs = {}
        options = {}
        if not self.getarg('save_intermediate'):
            kwargs['write'] = {'source': False}

        # output path
        outpath = os.path.dirname(self.datain[0].filename)
        kwargs['outpath'] = outpath

        # add scanpol defaults
        kwargs['scanpol'] = True

        # add additional top-level parameters
        for arg in ['deep', 'faint', 'extended']:
            if self.getarg(arg):
                kwargs[arg] = True

        # add frame clipping if necessary
        use_frames = str(self.getarg('use_frames')).strip()
        use_frames = StepScanMap.check_use_frames(self.datain, use_frames)
        if use_frames != '':
            kwargs['frames'] = use_frames

        # add additional options from parameters at end,
        # so they can override any defaults set by the above
        additional = str(self.getarg('options')).strip()
        if additional != '':
            all_val = additional.split()
            for val in all_val:
                try:
                    k, v = val.split('=')
                except (IndexError, ValueError, TypeError):
                    pass
                else:
                    options[k] = v
        kwargs['options'] = options

        # loop over groups, reducing together
        self.dataout = []
        for i, group in enumerate(groups):
            kw = kwargs.copy()
            log.info('Group {}/{}: '
                     'SCRIPTID = {}'.format(i + 1, len(groups), group))
            log.info('  Mean, std VPA: '
                     '{:.2f} +/- {:.2f}'.format(np.mean(vpa[group]),
                                                np.std(vpa[group])))

            # warn if vpa is outside tolerance
            if np.any(np.array(vpa[group]) - np.min(vpa[group]) > vtol):
                log.warning('VPA is outside tolerance for '
                            'group {}: {}'.format(i + 1, vpa[group]))
                log.warning('Should be within {} degrees'.format(vtol))

            # output base name and header
            d1 = groups[group][0]
            basehead = d1.header
            outname = os.path.basename(d1.filenamebegin
                                       + self.procname.upper()
                                       + d1.filenameend)
            if not self.getarg('save_intermediate'):
                kw['name'] = outname

            # add input file names
            infiles = []
            for datain in groups[group]:
                # get filename
                if os.path.exists(datain.filename):
                    fname = datain.filename
                else:
                    rawname = datain.rawname
                    if os.path.exists(rawname):
                        fname = rawname
                    else:
                        msg = 'File {} not found'.format(datain.filename)
                        log.error(msg)
                        raise ValueError(msg)
                infiles.append(fname)

            # argument set to run the reduction on the group
            argset = (infiles, kw, basehead, group)

            # run the group
            df, group = self._run_scanmap(argset)

            # check for errored reduction
            if df is None:
                msg = 'Problem in assembling scanpol ' \
                      'data for group {}'.format(group)
                log.error(msg)
                log.warning('Dropping files from reduction:')
                for bad_df in groups[group]:
                    log.warning('  {}'.format(
                        os.path.basename(bad_df.filename)))
                continue

            # update header, filename with MISO-style handling
            # for file numbers
            df.filename = groups[group][0].filename
            self.filenum = fnum[group]
            self.iomode = 'MISO'
            self.updateheader(df)
            self.iomode = 'MIMO'

            self.dataout.append(df)
