# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Mapping pipeline step."""

from astropy import log
from astropy.io import fits
import numpy as np
import psutil

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent
from sofia_redux.instruments.hawc.steps.basemap import BaseMap
from sofia_redux.toolkit.resampling.resample import Resample

__all__ = ['StepMerge']


class StepMerge(StepMIParent, BaseMap):
    """
    Create a map from multiple input images.

    This step resamples all input data into a common output grid.
    Resampling is performed via a distance-weighted, low-order
    polynomial surface fit to the input data within a window around
    each output grid point.  It is assumed that all input files
    already contain the correct WCS.

    Input files for this step must contain STOKES and ERROR frames for
    I, Q and U each, as well as COVAR Q I, COVAR U I, and COVAR Q U
    images.  The output is a single file with the same extensions.
    In addition, an image mask is created, which gives some information
    on how much data went into each output pixel in the map.  The
    values are sums over the Gaussian weight of each input data point.
    The EXPTIME keyword in the merged header is the sum of all exposure
    times from the individual files. The table values for each input
    image are stored in the rows of the MERGED DATA table in the output
    file.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'merge', and are named with
        the step abbreviation 'MRG'.

        Parameters defined for this step are:

        beamsize : list of float
            Beam FWHM size (arcsec) to write into BMAJ/BMIN header
            keywords.  One value for each HAWC filter band.
        cdelt : list of float
            Pixel size in arcseconds of output map. One value for each
            HAWC filter band.
        proj : str
            Projection of output map.
        sizelimit : int
            Upper limit on output map size (either axis, in pixels).
        widowstokesi : bool
            Use widow pixels (flagged 1 or 2) to compute Stokes I map.
        conserveflux : bool
            If set, a flux conservation factor (due to the change
            in pixel size) will be applied to all output images.
        fwhm : list of float
            FWHM of gaussian smoothing kernel in arcseconds (per band).
        radius : list of float
            Integration radius for local fits, in arcseconds (per band).
        fit_order : int
            Polynomial fit order for local regression.
        errflag : bool
            If set, use uncertainties when computing averages.
        edge_threshold : float
            Threshold to set edge pixels to NaN.  Range is 0-1; 0 means
            keep all edge pixels. Higher values keep fewer pixels.
        adaptive_algorithm : {'scaled', 'shaped', 'none'}
            If 'shaped' or 'scaled', adaptive smoothing will be used,
            varying the kernel size according to the data. If 'shaped',
            the kernel shape and rotation angle may also vary.
        fit_threshold : float
            Deviation from weighted mean to allow for higher order
            fit. Set to 0 to turn off.  Positive values replace bad
            values with the mean value in the window; negative values
            replace bad values with NaN.
        bin_cdelt : bool
            If set, and data was previously binned via StepBinPixels,
            then the input cdelt will be multiplied by the binning factor.
            If not set, the provided cdelt will be used directly. This
            allows useful default behavior for binned data, but still
            allows for tunable output pixel sizes.
        """

        # Name of the pipeline reduction step
        self.name = 'merge'
        self.description = 'Merge Maps'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'mrg'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['beamsize', [4.84, 7.80, 7.80, 13.6, 18.2],
                               "Beam FWHM size (arcsec) to write into "
                               "BMAJ/BMIN header keywords"])
        self.paramlist.append(['cdelt', [1.21, 1.95, 1.95, 3.40, 4.55],
                               'Pixel size in arcseconds of output map'])
        self.paramlist.append(['proj', 'TAN',
                               'Projection of output map'])
        self.paramlist.append(['sizelimit', 3000,
                               "Upper limit on output map size "
                               "(either axis, in pixels)."])
        self.paramlist.append(['widowstokesi', True,
                               "Use widow pixels (flagged 1 or 2) to "
                               "compute Stokes I map"])
        self.paramlist.append(['conserveflux', True,
                               "Apply flux conservation factor (due to "
                               "change in pixel size) to all output images"])
        self.paramlist.append(['fwhm', [2.57, 4.02, 4.02, 6.93, 9.43],
                               'FWHM of gaussian smoothing kernel, '
                               'in arcseconds (per band)'])
        self.paramlist.append(['radius', [4.85, 7.8, 7.8, 13.6, 18.2],
                               'Integration radius for smoothing, in '
                               'arcseconds (per band)'])
        self.paramlist.append(['fit_order', 0,
                               "Polynomial fit order for local regression."])
        self.paramlist.append(['errflag', True,
                               "Use uncertainties to weight fits."])
        self.paramlist.append(['edge_threshold', 0.5,
                               "Set edge pixels to NaN.  Range 0-1; 0 means "
                               "keep all edge pixels. Higher values "
                               "keep fewer pixels."])
        self.paramlist.append(['adaptive_algorithm', 'None',
                               "If 'shaped' or 'scaled', adaptive "
                               "smoothing will be used, varying the kernel "
                               "size according to the data. If 'shaped', "
                               "the kernel shape and rotation angle "
                               "may also vary."])
        self.paramlist.append(['fit_threshold', -10.0,
                               "Deviation from weighted mean to allow, "
                               "for higher order fit. Set to 0 to turn "
                               "off, < 0 replaces with NaN."])
        self.paramlist.append(['bin_cdelt', True,
                               "Multiply provided cdelt by binning factor "
                               "for the input data."])

    def read_fwhm_radius_cdelt_beam(self):
        """
        Read a fwhm, radius, and cdelt value from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band.  The correct value for the
        input data is selected from the list.

        Returns
        -------
        fwhm : float
            FWHM value for the input data.
        radius : float, float, float
            Radius value for the input data.
        cdelt : float
            Pixel scale value for the input data.
        beamsize : float
            Beam size value for the input data.
        """
        fwhm = self.getarg('fwhm')
        radius = self.getarg('radius')
        cdelt = self.getarg('cdelt')
        beamsize = self.getarg('beamsize')
        waveband = self.datain[0].getheadval('spectel1')
        bands = ['A', 'B', 'C', 'D', 'E']
        try:
            idx = bands.index(waveband[-1])
        except (ValueError, IndexError):
            # waveband not in list
            msg = 'Cannot parse waveband: %s' % waveband
            log.error(msg)
            raise ValueError(msg)
        try:
            fwhm = fwhm[idx]
            radius = radius[idx]
            cdelt = cdelt[idx]
            beamsize = beamsize[idx]
        except IndexError:
            msg = 'Missing radius/fwhm values for all wavebands'
            log.error(msg)
            raise IndexError(msg)

        return fwhm, radius, cdelt, beamsize

    def resample_images(self, radius, fit_order, smoothing,
                        edge, adaptive_threshold, adaptive_algorithm,
                        fit_threshold, errflag, max_cores):
        """
        Resample input images into a common grid.

        Resampling is performed via a distance-weighted, low-order
        polynomial surface fit to the input data within a window
        around each output grid point.  Output images are stored in
        self.pmap.

        Parameters
        ----------
        radius : float
            Fit window to consider for local fits.
        fit_order : int
            Polynomial surface order to fit.
        smoothing : float
            Smoothing radius for distance weighting, expressed
            as a fraction of the fit window.
        edge : float
            Threshold for setting edge pixels to NaN.  Higher
            values block more pixels, a zero values allows all
            edge pixels through.
        adaptive_threshold : float
            Threshold for adaptive smoothing.  Range is 0-2; 1.0 is
            optimal.  Set lower for smaller scale adaptive kernel; higher
            for larger scale.
        adaptive_algorithm : {'scaled', 'shaped', None}
            Algorithm for adaptive smoothing kernel.  If scaled, only the
            size is allowed to vary.  If shaped, the kernel shape and
            rotation may also vary.
        fit_threshold : float or None
            Threshold for fit rejection, specified in sigma.  Set to
            None to turn off.
        errflag : bool
            If True, errors on the flux values will be used to weight
            the fits to the data.  If False, only distance weights
            will be used.
        max_cores : int, or None
            If a number larger than 1, the data processing will
            proceed in parallel on max_cores CPUs.  Multiprocessing
            requires that joblib is installed.
        """

        if self.nhwp == 1:
            stokes_vals = ['I']
        else:
            stokes_vals = ['I', 'Q', 'U']

        # loop over stokes for fluxes and errors
        flxvals = []
        errvals = []
        for stokes in stokes_vals:
            flxvals.append(np.hstack([d[stokes].ravel()
                           for d in self.pdata]))
            errvals.append(np.hstack([d['d{}'.format(stokes)].ravel()
                           for d in self.pdata]))
        resampler = Resample(
            self.pmap['coordinates'], flxvals, error=errvals,
            window=radius, order=fit_order,
            robust=None, negthresh=None)

        flux, std, weights = resampler(
            *self.pmap['grid'], smoothing=smoothing,
            fit_threshold=fit_threshold, edge_threshold=edge,
            adaptive_threshold=adaptive_threshold,
            adaptive_algorithm=adaptive_algorithm,
            edge_algorithm='distribution',
            get_error=True, get_distance_weights=True,
            error_weighting=errflag, jobs=max_cores)

        for i, stokes in enumerate(stokes_vals):
            self.pmap[stokes] = flux[i]
            self.pmap['d{}'.format(stokes)] = std[i]
            if stokes == 'I':
                self.pmap['mask'] = weights[i]

        # stop here for imaging data
        if self.nhwp == 1:
            return

        # otherwise do covariances too
        covvals = []
        errvals = []
        covnames = []
        for stokes in [('Q', 'I'), ('U', 'I'), ('Q', 'U')]:
            cov = '%s%scov' % stokes
            err1 = 'd%s' % stokes[0]
            err2 = 'd%s' % stokes[1]
            covnames.append(cov)

            covvals.append(np.hstack([d[cov].ravel()
                                      for d in self.pdata]))
            errvals.append(np.hstack([np.sqrt(d[err1].ravel()
                                              * d[err2].ravel())
                                      for d in self.pdata]))

        # handle covariances as flux values, weighted by
        # the appropriate stokes combinations, with a flag
        # to tell the resampler how to propagate them
        resampler = Resample(
            self.pmap['coordinates'], covvals, error=errvals,
            window=radius, order=0,
            robust=None, negthresh=None)
        flux = resampler(
            *self.pmap['grid'], smoothing=smoothing,
            fit_threshold=None, edge_threshold=edge,
            edge_algorithm='distribution', is_covar=True,
            error_weighting=True, jobs=max_cores)

        for i, cov in enumerate(covnames):
            self.pmap[cov] = flux[i]

    def run(self):
        """
        Run the data reduction algorithm.

        This step is run as a multi-in single-out (MISO) step:
        self.datain should be a list of DataFits, and output
        will be a single DataFits, stored in self.dataout.

        The process is:

        1. Read in all good pixels from the input data.
        2. Resample all input data into a common map.
        3. Store output images in a DataFits object.
        4. Merge headers and table data from all input files.
        """
        # check validity, set self.nhwp
        self.checkvalid()

        # self.datain must be a list/tuple
        self.nfiles = len(self.datain)
        log.debug('Merging %d files' % self.nfiles)

        # set up for parallel processing via joblib
        max_cores = psutil.cpu_count() - 1
        if max_cores < 2:  # pragma: no cover
            max_cores = 1

        # read parameter values
        fwhm, radius, cdelt, beamsize = self.read_fwhm_radius_cdelt_beam()
        errflag = self.getarg('errflag')
        widowstokesi = self.getarg('widowstokesi')
        conserveflux = self.getarg('conserveflux')
        edge = self.getarg('edge_threshold')
        adaptive_algorithm = self.getarg('adaptive_algorithm')
        fit_order = self.getarg('fit_order')
        fit_threshold = self.getarg('fit_threshold')
        bin_cdelt = self.getarg('bin_cdelt')

        # get pixel scale from header
        pixscal = self.datain[0].getheadval('PIXSCAL')

        # get binning factor from first header, if applicable
        if bin_cdelt:
            bin_factor = self.datain[0].header.get('PIXELBIN', 1)
            if bin_factor > 1:
                log.info(f'Multiplying cdelt and radius by '
                         f'binning factor {bin_factor}')
                cdelt *= bin_factor
                radius *= bin_factor
                log.info(f'Output pixel size is {cdelt} arcsec')
                log.info(f'Fit radius is {radius} arcsec')
                if fit_order > 1:
                    log.info('Reducing fit order to 1.')
                    fit_order = 1

        # if adaptive is on, smoothing should be set to true
        # Gaussian sigma
        adaptive_algorithm = str(adaptive_algorithm).strip().lower()
        if adaptive_algorithm in ['shaped', 'scaled']:
            adaptive = 1.0
        else:
            adaptive = 0.0
            adaptive_algorithm = None
        if adaptive > 0 and not np.allclose(fwhm, beamsize):
            log.warning('Setting smoothing FWHM to beam size '
                        'for adaptive case.')
            fwhm = beamsize
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        if conserveflux:
            # get pixel scale from first image and cdelt from params
            flux_factor = cdelt**2 / pixscal**2
            log.debug('')
            log.debug('Flux factor: %.2f' % flux_factor)
            log.debug('')
        else:
            flux_factor = 1.

        # If widowstokesi = True, then widow pixels (flagged as
        # 1 or 2) will also be accounted in the smoothed
        # Stokes I image. Stokes Q and U will continue
        # using only strictly good pixels (flagged as 0)
        if widowstokesi:
            maxgoodpix = 2
        else:
            maxgoodpix = 0

        # read data, allocate space for maps
        self.read_resample_data(maxgoodpix)
        self.make_resample_map(cdelt)

        if errflag:
            log.debug('Uncertainties used for weighting')
        else:
            log.warning('Uncertainties NOT used for weighting')

        # Resample all the input files together to generate the output images
        log.debug("Starting resample: all files")

        # Output data is stored in self.pmap
        self.resample_images(radius, fit_order, sigma, edge,
                             adaptive, adaptive_algorithm,
                             fit_threshold, errflag,
                             max_cores)

        # Initialize pipedata, add input OBS_IDs
        self.dataout = DataFits(config=self.datain[0].config)
        self.allhead.add_history('Merged Files:')
        for i in range(self.nfiles):
            self.allhead.add_history('OBS_ID: %s' %
                                     self.datain[i].getheadval('OBS_ID'))

        # Put output image data from self.pmap into dataout
        # for I (and if present Q and U)
        if self.nhwp == 1:
            stokes = ['I']
        else:
            stokes = ['I', 'Q', 'U']
        for var in stokes:
            name = 'STOKES ' + var
            tmp = self.pmap[var]

            # correct for flux conservation
            tmp *= flux_factor

            tmp.shape = (self.allhead['naxis2'], self.allhead['naxis1'])

            # Set I, Q and U maps - copy same astrometry
            # info from header to these HDUs
            self.dataout.imageset(tmp, name, self.allhead)

            # Set dI, dQ, dU - also copy astrometry into these headers
            name = 'ERROR ' + var
            tmp = self.pmap['d' + var]
            tmp *= flux_factor
            tmp.shape = (self.allhead['naxis2'], self.allhead['naxis1'])
            self.dataout.imageset(tmp, name, self.allhead)

        # also propagate the covariance extensions
        if self.nhwp != 1:
            stokes = [('Q', 'I'), ('U', 'I'), ('Q', 'U')]
            for var in stokes:
                name = 'COVAR %s %s' % var
                tmp = flux_factor**2 * self.pmap['%s%scov' % var]
                tmp.shape = (self.allhead['naxis2'], self.allhead['naxis1'])
                self.dataout.imageset(tmp, name, self.allhead)

        # Set image mask
        tmp = self.pmap['mask']
        tmp.shape = (self.allhead['naxis2'], self.allhead['naxis1'])
        self.dataout.imageset(tmp, 'IMAGE MASK', self.allhead)
        self.dataout.filename = self.datain[-1].filename
        self.dataout.setheadval('BUNIT', 'pixel',
                                comment='Data units',
                                dataname='IMAGE MASK')

        # add headers from first input file (to get keywords)
        for name in self.datain[0].imgnames:
            if name in self.dataout.imgnames:
                self.dataout.copyhead(self.datain[0],
                                      overwrite=False, name=name)

        # add relevant header values from other input files
        for other in self.datain[1:]:
            self.dataout.mergehead(other)

        # remove CROTA2, if present because output image is already rotated
        self.dataout.delheadval("CROTA2")

        # remove source position keywords if present, since their
        # values are no longer correct for the resampled map
        pos_keys = ['STCENTX', 'STCENTY',
                    'STCENTXE', 'STCENTYE',
                    'SRCPOSX', 'SRCPOSY']
        for key in pos_keys:
            self.dataout.delheadval(key)

        # Create table with combined data from the input files
        #     One table row for each input file
        iterfiles = range(self.nfiles)
        tablist = []
        for i in iterfiles:
            if 'TABLE DATA' in self.datain[i].tabnames:
                tablist.append(self.datain[i].tableget('TABLE DATA'))

        if len(tablist) > 0:

            # for each name, loop through all tables,
            # make sure name, format, dim
            # and unit is the same among them. A column with a
            # mismatch is thrown out
            names = []
            formats = []
            dims = []
            units = []
            for n in tablist[0].names:
                format0 = tablist[0].columns[n].format
                dim = tablist[0].columns[n].dim
                unit = tablist[0].columns[n].unit
                column_matches = True
                for inf in iterfiles:
                    if n not in tablist[inf].names:
                        log.warning("name not found in file %d, "
                                    "removing %s" % (inf, n))
                        column_matches = False
                        break
                    if format0 != tablist[inf].columns[n].format:
                        log.warning("%s has different format in "
                                    "file %d, removing" % (n, inf))
                        column_matches = False
                        break
                    if dim != tablist[inf].columns[n].dim:
                        log.warning("%s has different dimension in "
                                    "file %d, removing" % (n, inf))
                        column_matches = False
                        break
                    if unit != tablist[inf].columns[n].unit:
                        log.warning("%s has different units in "
                                    "file %d, removing" % (n, inf))
                        column_matches = False
                        break
                if column_matches:
                    names.append(n)
                    formats.append(format0)
                    dims.append(dim)
                    units.append(unit)

            # Fill up fits columns to make table
            cols = []
            for n, f, d, u in zip(names, formats, dims, units):
                data = [a[n] for a in tablist]
                cols.append(fits.Column(name=n, format=f, dim=d,
                                        unit=u, array=data))

            # add some super-important columns to table, if not present
            if 'Right Ascension' not in names:
                ra = [self.datain[i].getheadval('crval1') for i in iterfiles]
                cols.append(fits.Column(name='Right Ascension',
                                        format='D', unit='deg',
                                        array=np.array(ra)))
            if 'Declination' not in names:
                dec = [self.datain[i].getheadval('crval2') for i in iterfiles]
                cols.append(fits.Column(name='Declination',
                                        format='D', unit='deg',
                                        array=np.array(dec)))
            if 'Filename' not in names:
                fname = [self.datain[i].filename for i in iterfiles]
                maxlen = max(map(len, fname))
                cols.append(fits.Column(name="Filename", array=np.array(fname),
                                        format='%dA' % maxlen, unit='str'))

            tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
            self.dataout.tableset(tbhdu.data, "MERGED DATA", tbhdu.header)

        # Update the EXPTIME header keyword with the summed value of
        # all individual exposure times of the input files
        self.sumexptime()

        # Merge all file headers with the first header

        # Write beam size (from beamsize config param) to header keywords
        self.dataout.setheadval('BMAJ', beamsize / 3600., 'Beam major axis')
        self.dataout.setheadval('BMIN', beamsize / 3600., 'Beam minor axis')
        self.dataout.setheadval('BPA', 0., 'Beam position angle')

        # Write list of input OBS_IDs in LSTOBSID keyword
        lstobsid = [self.datain[j].getheadval('obs_id')
                    for j in range(0, self.nfiles, 1)]
        self.dataout.setheadval("LSTOBSID", ','.join(lstobsid),
                                'List of OBS_IDs for merged files')

        # Delete lat/lonpole from header if present
        if 'LATPOLE' in self.dataout.header:
            del self.dataout.header['LATPOLE']
        if 'LONPOLE' in self.dataout.header:
            del self.dataout.header['LONPOLE']
