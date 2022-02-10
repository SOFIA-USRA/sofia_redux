# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background subtraction pipeline step."""

from astropy import log
from astropy.stats import sigma_clip
import numpy as np
import psutil

from sofia_redux.instruments.hawc.steps.basemap import BaseMap
from sofia_redux.instruments.hawc.stepmoparent import StepMOParent
from sofia_redux.toolkit.resampling.resample import Resample
from sofia_redux.toolkit.utilities.func import stack

__all__ = ['StepBgSubtract']


class StepBgSubtract(StepMOParent, BaseMap):
    """
    Subtract residual background across multiple input files.

    This step iteratively solves for additive offsets between the input
    files, then subtracts the offset from each flux image.

    The input data expected is a DataFits with STOKES and ERROR frames
    for I, Q and U each, as well as COVAR Q I, COVAR U I, and COVAR Q U
    images. For total intensity data, only STOKES I and ERROR I are
    expected. Input is typically produced by the
    `sofia_redux.instruments.hawc.steps.StepCalibrate` pipeline step.

    The output image from this step contains the same image frames as the
    input image. The STOKES frames have been background corrected.
    """
    def __init__(self):
        # placeholder for calculated offsets
        self.offsets = []
        super().__init__()

    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'bgsubtract', and are named with
        the step abbreviation 'BGS'.

        Parameters defined for this step are:

        cdelt : list of float
            Pixel size in arcseconds of output map. One value for each
            HAWC filter band.
        proj : str
            Projection of output map.
        sizelimit : int
            Upper limit on output map size (either axis, in pixels).
        fwhm : list of float
            FWHM of gaussian smoothing kernel in arcseconds (per band).
        radius : list of float
            Integration radius for local fits, in arcseconds (per band).
        errflag : bool
            If set, use uncertainties when computing averages.
        widowstokesi : bool
            Use widow pixels (flagged 1 or 2) to compute Stokes I map.
        edge_threshold : float
            Threshold to set edge pixels to NaN. Range is 0-1; 0 means
            keep all edge pixels. Higher values keep fewer pixels.
        fit_order : int
            Polynomial fit order for local regression.
        bgoffset : int
            Maximum number of iterations of background subtraction.
        chauvenet : bool
            If set, use Chauvenet's criterion (sigma clipping for
            outlier rejection) in background averages.
        fitflag : bool
            If set, use errors in intensity for weighting background
            averages.
        qubgsubtract : bool
            If set, apply background correction to Stokes Q and U
            images as well as Stokes I.
        """

        # Name of the pipeline reduction step
        self.name = 'bgsubtract'
        self.description = 'Subtract Background'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'bgs'

        # Clear Parameter list
        self.paramlist = []

        # resampling parameters
        self.paramlist.append(['cdelt', [2.57, 4.02, 4.02, 6.93, 9.43],
                               'Pixel size in arcseconds of output map'])
        self.paramlist.append(['proj', 'TAN',
                               'Projection of output map'])
        self.paramlist.append(['sizelimit', 3000,
                               "Upper limit on output map size "
                               "(either axis, in pixels)."])
        self.paramlist.append(['fwhm', [2.57, 4.02, 4.02, 6.93, 9.43],
                               'FWHM of gaussian smoothing kernel, '
                               'in arcseconds (per band)'])
        self.paramlist.append(['radius', [7.71, 12.06, 12.06, 20.79, 28.29],
                               'Integration radius for smoothing, in '
                               'arcseconds (per band)'])
        self.paramlist.append(['errflag', True,
                               "Use uncertainties when computing averages"])
        self.paramlist.append(['widowstokesi', True,
                               "Use widow pixels (flagged 1 or 2) to "
                               "compute Stokes I map"])
        self.paramlist.append(['edge_threshold', 0.5,
                               "Set edge pixels to NaN. Range 0-1; 0 means "
                               "keep all edge pixels. Higher values "
                               "keep fewer pixels."])
        self.paramlist.append(['fit_order', 0,
                               "Polynomial fit order for local regression."])

        # background parameters
        self.paramlist.append(['bgoffset', 10,
                               'Number of iterations of background '
                               'subtraction'])
        self.paramlist.append(['chauvenet', True,
                               "Use Chauvenet's criterion in "
                               "background subtraction"])
        self.paramlist.append(['fitflag', False,
                               "Use errors in intensity for offset "
                               "calculation"])
        self.paramlist.append(['qubgsubtract', True,
                               'Apply linear background correction '
                               'to Stokes Q and U images'])

    def read_fwhm_radius_cdelt(self):
        """
        Read a fwhm, radius, and cdelt value from the parameters.

        The parameters are expected to be defined as a list, with
        one entry for each HAWC band. The correct value for the
        input data is selected from the list.

        Returns
        -------
        fwhm : float
            FWHM value for the input data.
        radius : float, float, float
            Radius value for the input data.
        cdelt : float
            Pixel scale value for the input data.
        """
        fwhm = self.getarg('fwhm')
        radius = self.getarg('radius')
        cdelt = self.getarg('cdelt')
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
        except IndexError:
            msg = 'Missing radius/fwhm values for all wavebands'
            log.error(msg)
            raise IndexError(msg)

        return fwhm, radius, cdelt

    def resample_images(self, radius, fit_order, smoothing,
                        edge, errflag, max_cores, separate=False):
        """
        Resample input images into a common grid.

        Resampling is performed via a distance-weighted, low-order
        polynomial surface fit to the input data within a window
        around each output grid point.

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
            Threshold for setting edge pixels to NaN. Higher
            values block more pixels, a zero values allows all
            edge pixels through.
        errflag : bool
            If True, errors on the flux values will be used to weight
            the fits to the data. If False, only distance weights
            will be used.
        max_cores : int, or None
            If a number larger than 1, the data processing will
            proceed in parallel on max_cores CPUs. Multiprocessing
            requires that joblib is installed.
        separate : bool, optional
            If True, separate maps will be made for each input file
            (all on the same coordinate grid), and returned in a
            list of dictionaries. If False, a single map will be made,
            and stored in self.pmap

        Returns
        -------
        list of dict
            The map(s) generated from the input.   Keys for the
            dictionary are the Stokes values and associated errors:
            I, dI for all; and Q, dQ, U, dU if more than one HWP is
            present.
        """

        if self.nhwp == 1:
            stokes_vals = ['I']
        else:
            stokes_vals = ['I', 'Q', 'U']

        if separate:
            nmap = self.nfiles
        else:
            nmap = 1

        # loop over maps and stokes for fluxes and errors
        maps = []
        for i in range(nmap):
            flxvals = []
            errvals = []

            for stokes in stokes_vals:
                if separate:
                    flx = self.pdata[i][stokes].ravel()
                    err = self.pdata[i]['d{}'.format(stokes)].ravel()
                else:
                    flx = np.hstack([d[stokes].ravel() for d in self.pdata])
                    err = np.hstack([d['d{}'.format(stokes)].ravel()
                                     for d in self.pdata])
                flxvals.append(flx)
                errvals.append(err)

            if separate:
                ra = self.pdata[i]['ra'].ravel()
                dec = self.pdata[i]['dec'].ravel()
                base_ra = self.pmap['base_ra']
                base_dec = self.pmap['base_dec']

                # offsets in arcsec, with RA reversed
                xs = -1 * (ra - base_ra) * \
                    np.cos(np.radians(base_dec)) * 3600.
                ys = (dec - base_dec) * 3600.

                coordvals = stack(xs, ys)
                pmap = {}
            else:
                coordvals = self.pmap['coordinates']
                pmap = self.pmap

            resampler = Resample(
                coordvals, flxvals, error=errvals,
                window=radius, order=fit_order,
                robust=None, negthresh=None)

            flux, std = resampler(
                *self.pmap['grid'], smoothing=smoothing,
                fit_threshold=None, edge_threshold=edge,
                edge_algorithm='distribution', get_error=True,
                error_weighting=errflag, jobs=max_cores)

            for j, stokes in enumerate(stokes_vals):
                pmap[stokes] = flux[j]
                pmap['d{}'.format(stokes)] = std[j]

            maps.append(pmap)
        return maps

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is multi-in, multi-out (MIMO),
        self.datain must be a list of DataFits objects. The output
        is also a list of DataFits objects, stored in self.dataout.

        The process is:

        1. Read in all good pixels from the input data.
        2. Make a map out of all input data for reference.
        3. Make a map out of each input data file to compare to
           the reference.
        4. Compute and subtract the average offset for all corresponding
           pixels, from the individual map to the reference map.
        5. Repeat steps 2-4 until convergence or the maximum number of
           iterations is reached.
        6. Store offset-subtracted images in self.dataout.
        """
        # make sure it is possible to combine input files
        # and set self.nhwp
        self.checkvalid()

        # self.datain must be a list/tuple
        self.nfiles = len(self.datain)
        self.dataout = [d.copy() for d in self.datain]

        # if only one input file, just return
        if self.nfiles < 2:
            log.debug("One file only. No background subtracted.")
            return

        fwhm, radius, cdelt = self.read_fwhm_radius_cdelt()
        errflag = self.getarg('errflag')
        widowstokesi = self.getarg('widowstokesi')
        edge = self.getarg('edge_threshold')
        fit_order = self.getarg('fit_order')

        bgoffset = self.getarg('bgoffset')
        chflag = int(self.getarg('chauvenet'))
        fitflag = int(self.getarg('fitflag'))
        qubgsubtract = self.getarg('qubgsubtract')

        if qubgsubtract and self.nhwp > 1:
            stokes = ['I', 'Q', 'U']
        else:
            stokes = ['I']

        # set up for parallel processing via joblib
        max_cores = psutil.cpu_count() - 1
        if max_cores < 2:  # pragma: no cover
            max_cores = 1

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # If widowstokesi = True, then widow pixels (flagged
        # as 1 or 2) will also be accounted in the smoothed
        # Stokes I image. Stokes Q and U will continue
        # using only strictly good pixels (flagged as 0)
        if widowstokesi:
            maxgoodpix = 2
        else:
            maxgoodpix = 0

        # read data, allocate space for maps
        self.read_resample_data(maxgoodpix)
        self.make_resample_map(cdelt)

        # iteratively remove offsets from maps
        self.offsets = {'I': np.zeros(self.nfiles),
                        'Q': np.zeros(self.nfiles),
                        'U': np.zeros(self.nfiles)}
        epsilon = 1e-3
        last_offsets = None
        while bgoffset > 0:
            log.debug("iteration {} of bgoffset".format(bgoffset))

            # smoothed map of all files together
            # output data is stored in self.pmap
            self.resample_images(radius, fit_order, sigma, edge,
                                 errflag, max_cores)

            # create new individual smoothed maps on the same grid
            new_maps = self.resample_images(radius, fit_order, sigma,
                                            edge, errflag, max_cores,
                                            separate=True)

            new_offsets = {'I': np.zeros(self.nfiles),
                           'Q': np.zeros(self.nfiles),
                           'U': np.zeros(self.nfiles)}
            for i, map_img in enumerate(new_maps):
                for s in stokes:
                    # offset from combined map
                    diff = map_img[s] - self.pmap[s]
                    var = self.pmap['d{}'.format(s)] ** 2

                    # flatten and pull out nans
                    idx = (~np.isnan(diff)) & (~np.isnan(var))
                    diff = diff[idx]
                    var = var[idx]

                    # weighted, sigma-clipped averaged offset
                    if chflag:
                        diff = sigma_clip(diff, sigma=3, masked=True,
                                          copy=False, cenfunc='mean')
                        var = var[~diff.mask]
                        diff = diff.compressed()
                    if fitflag:
                        offset = np.average(diff, weights=1 / var)
                    else:
                        offset = np.mean(diff)

                    # subtract offset from data
                    self.pdata[i][s] -= offset
                    new_offsets[s][i] = offset

            # stop iterating if offsets are not changing
            all_offsets = np.hstack([np.abs(new_offsets[s])
                                     for s in new_offsets])
            if np.all(all_offsets < epsilon):  # pragma: no cover
                bgoffset = 0
            if last_offsets is not None:
                diff = np.abs(all_offsets - last_offsets)
                if np.all(diff < epsilon):  # pragma: no cover
                    bgoffset = 0

            for s in self.offsets:
                self.offsets[s] += new_offsets[s]
            last_offsets = all_offsets
            bgoffset -= 1

        # Now assign background subtracted images back to each pipedata object
        # Errors and covariances are untouched
        for i, f in enumerate(self.dataout):
            f.imageset(f.imageget('STOKES I')
                       - self.offsets['I'][i], 'STOKES I')
            log.debug('Image {} Stokes I '
                      'offset: {:.2f}'.format(i + 1, self.offsets['I'][i]))
            if self.nhwp > 1:
                # polarization
                f.imageset(f.imageget('STOKES Q') - self.offsets['Q'][i],
                           'STOKES Q')
                f.imageset(f.imageget('STOKES U') - self.offsets['U'][i],
                           'STOKES U')
                log.debug('Image {} Stokes Q '
                          'offset: {:.2f}'.format(i + 1, self.offsets['Q'][i]))
                log.debug('Image {} Stokes U '
                          'offset: {:.2f}'.format(i + 1, self.offsets['U'][i]))
