# Licensed under a 3-clause BSD style license - see LICENSE.rst

from warnings import simplefilter, catch_warnings

from astropy import log
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import numpy as np
from pandas import DataFrame
from photutils import DAOStarFinder
from scipy.signal import correlate2d

__all__ = ['peakfind', 'PeakFinder']


class PeakFinder(object):
    """Configure and run peak finding algorithm."""
    def __init__(self, image, reference=None,
                 npeaks=4, fwhm=4.5,
                 sharplo=0.2, sharphi=1.0,
                 roundlo=-0.75, roundhi=0.75,
                 silent=False, maxiter=1000,
                 epsilon=5, eps=1e-7, ncut=30,
                 chopnoddist=None, refine=True,
                 positive=False, smooth=True):
        self.image = image
        self.reference = reference
        self.npeaks = npeaks
        self.fwhm = fwhm
        self.gfit_image = None
        self.gfit_reference = None
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.iteration = 0
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.eps = eps
        self.ncut = ncut
        self.positive = positive
        self.smooth = smooth
        self.refine = refine
        self.silent = silent
        self.image_table = Table()
        self.reference_table = Table()
        self.current = None
        self.frame = DataFrame()
        self.threshold = None
        try:
            self.chopdist = chopnoddist[0]
            self.noddist = chopnoddist[1]
        except TypeError:
            self.chopdist = None
            self.noddist = None

    @property
    def sigma(self):
        return self.fwhm * gaussian_fwhm_to_sigma

    def print(self, message):
        if not self.silent:
            log.info(message)

    def chopnod_sort(self, table):
        """
        Select peaks that follow a particular pattern for merge

        The table passed in will be cut in place.

        Parameters
        ----------
        table : astropy.table.Table
        """
        if not isinstance(table, Table):
            return
        elif None in [self.chopdist, self.noddist]:
            return
        elif 'xcentroid' not in table.columns or \
                'ycentroid' not in table.columns:
            return
        dist = np.sqrt((self.chopdist ** 2) + (self.noddist ** 2))
        x0, y0 = table['xcentroid'], table['ycentroid']
        valid = [False] * len(table)
        for idx, row in enumerate(table):
            dx = x0 - row['xcentroid']
            dy = y0 - row['ycentroid']
            dr = np.sqrt((dx ** 2) + (dy ** 2))
            dchop = abs(dr - self.chopdist)
            dnod = abs(dr - self.noddist)
            dchopnod = abs(dr - dist)
            ok = (np.array([dchop, dnod, dchopnod]) < self.epsilon)
            if ok.astype(int).sum() >= 2:
                valid[idx] = True
        table = table[valid]

    def search_peaks(self, image):
        """Initial search for peaks in the image

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        astropy.table.Table
        """
        table = Table()
        if not isinstance(image, np.ndarray):
            return table

        search_image = image
        if self.smooth:
            with catch_warnings():
                simplefilter('ignore')
                search_image = convolve_fft(
                    search_image, Gaussian2DKernel(self.sigma),
                    normalize_kernel=True, preserve_nan=True)

        # always take absolute value for fitting purposes
        search_image = abs(search_image)

        threshold = np.array([np.nanmin(search_image),
                              np.nanmax(search_image)])
        threshold *= [0.9, 1.1]
        if threshold[0] < 0:  # pragma: no cover
            threshold[0] = 0.

        self.iteration = 0
        while self.iteration < self.maxiter:
            self.iteration += 1
            with catch_warnings():
                simplefilter('ignore', AstropyWarning)
                self.threshold = threshold.mean()
                finder = DAOStarFinder(
                    self.threshold, self.fwhm,
                    sharplo=self.sharplo, sharphi=self.sharphi,
                    roundlo=self.roundlo, roundhi=self.roundhi)
                table = finder.find_stars(search_image)
            self.chopnod_sort(table)

            if self.refine and self.positive:
                self.refine_table(image, table)
            if not table:
                nfound = 0
            else:
                nfound = len(table)

            if abs(threshold[0] - threshold[1]) < self.eps and (
                    nfound != self.npeaks):
                self.print('Min/max interval is null, breaking loop at '
                           'iteration #%s' % self.iteration)
                return table
            elif nfound < self.npeaks:
                threshold[1] = self.threshold
            elif nfound > self.npeaks:
                threshold[0] = self.threshold
            else:
                return table
        else:
            return table

    def refine_table(self, image, table):
        """
        Refine table contents with a fit on the image

        The table is modified in place.

        Parameters
        ----------
        image : np.ndarray
        table : astropy.table.Table
        """
        if table is None or len(table) == 0 or \
                not isinstance(image, np.ndarray):
            return

        x0, y0 = np.mgrid[-self.ncut:self.ncut, -self.ncut:self.ncut]
        model_g = models.Gaussian2D(
            1.0, self.ncut, self.ncut,
            x_stddev=self.sigma, y_stddev=self.sigma)
        psf = np.zeros(x0.shape)
        psf = model_g.render(psf)

        ymax = image.shape[0] - self.ncut
        xmax = image.shape[1] - self.ncut

        remove_rows = []
        for idx, row in enumerate(table):
            x, y = row['xcentroid'], row['ycentroid']
            if (self.ncut < x < xmax) and (self.ncut < y < ymax):
                bxmin = int(np.floor(x - self.ncut))
                bxmax = int(np.ceil(x + self.ncut))
                bymin = int(np.floor(y - self.ncut))
                bymax = int(np.ceil(y + self.ncut))
                box = image[bymin:bymax, bxmin:bxmax].copy()
                g_init = models.Gaussian2D(row['flux'], x, y,
                                           x_stddev=self.sigma,
                                           y_stddev=self.sigma)

                flip = np.nanmean(correlate2d(box, psf)) < 0
                box *= -1 if flip else 1

                fitter = fitting.LevMarLSQFitter()
                by, bx = np.mgrid[bymin:bymax, bxmin:bxmax]
                nn = ~np.isnan(box)
                g = fitter(g_init, bx[nn], by[nn], box[nn])

                # catch fit failures - just skip refinement in this case
                try:
                    failure = (g.fit_info['ierr'] not in [1, 2, 3, 4])
                except (AttributeError, KeyError):  # pragma: no cover
                    failure = False
                bad = not np.all(
                    np.isfinite([g.x_mean.value, g.y_mean.value,
                                 g.amplitude.value]))
                if failure or bad:  # pragma: no cover
                    # ignore bad fits
                    continue

                if flip:
                    g.amplitude *= -1

                row['xcentroid'] = g.x_mean.value
                row['ycentroid'] = g.y_mean.value
                row['flux'] = g.amplitude.value

                if self.positive:
                    if row['flux'] < 0 or row['peak'] < 0:
                        remove_rows.append(idx)

        for idx in sorted(remove_rows, reverse=True):
            table.remove_row(idx)

    def findpeaks(self):
        columns = ['id', 'xcentroid', 'ycentroid', 'sharpness',
                   'roundness1', 'roundness2', 'npix', 'sky',
                   'peak', 'flux', 'mag']
        self.frame = DataFrame(index=range(self.npeaks), columns=columns)
        self.frame['id'] = self.frame['id'].fillna(-1)
        self.frame = self.frame.fillna(np.nan)

        tables = []
        for attr in ['reference', 'image']:
            image = getattr(self, attr)
            table_name = attr + '_table'
            table = Table()
            setattr(self, table_name, table)
            if not isinstance(image, np.ndarray) or (np.isnan(image)).all():
                continue
            table = self.search_peaks(image)

            if not isinstance(table, Table) or len(table) == 0:
                self.print("Could not create %s table" % attr)
                return

            if self.refine and not self.positive:
                self.refine_table(image, table)

            tables.append(table)

            self.print('Number of loops run: %s/%s' %
                       (self.iteration, self.maxiter))
            self.print('Peaks found: %s/%s' % (len(table), self.npeaks))
            self.print("Threshold used: %s" % self.threshold)

        if len(tables) == 0:
            log.error("No peaks could be found")
            return

        if len(tables) == 2:
            if len(tables[0]) != len(tables[1]):  # pragma: no cover
                # edge case of peak finding failure
                log.error("Matching peaks could not be found")
                return
            dtable = tables[1].copy()
            dtable['xcentroid'] -= tables[0]['xcentroid']
            dtable['ycentroid'] -= tables[0]['ycentroid']
        else:
            dtable = tables[0].copy()

        self.frame.update(dtable.to_pandas())
        fluxcol = 'fit_amplitude' if self.refine else 'flux'
        self.frame = self.frame.rename(columns={
            'xcentroid': 'x', 'ycentroid': 'y', 'flux': fluxcol})
        cols = ['x', 'y', 'sharpness', 'roundness2',
                'peak', fluxcol]
        self.print('\n' + self.frame[cols].to_string(
            header=[c.upper() for c in cols]))


def peakfind(coadded, newimage=None,
             refine=True, npeaks=4, fwhm=4.5,
             sharplo=0.2, sharphi=1.0,
             roundlo=-0.75, roundhi=0.75,
             silent=False, maxiter=1000,
             epsilon=5, eps=1e-7, ncut=30,
             chopnoddist=None,
             positive=False,
             smooth=True,
             return_object=False,
             coordinates=False):
    """
    Find peaks (stars) in FORCAST images

    Identifies the desired number of peaks in the image using the DAO
    search algorithm followed by optional (default=True) Levenberg-
    Marquardt least squares fitting of a gaussian psf to refine the
    fit.

    Parameters
    ----------
    coadded : numpy.ndarray
        Reference image to find peak positions
    newimage : numpy.ndarray, optional
        Secondary image to compare to coadded.  If provided, the return
        value will be the shifts required to move the sources in newimage
        onto the sources in coadded.  If not provided, the return value
        is the positions of the peaks in coadded
    refine : bool, optional
        If set, the x, y coordinates are fine tuned by fitting a
        Gaussian to the profile of each found center
    npeaks : int, optional
        Number of peaks to find; usually determined from the chop/nod
        mode
    fwhm : float, optional
        Expected FWHM in pixels
    roundlo : float, optional
        The lower bound on roundness for object detection
    roundhi : float, optional
        The upper bound on roundness for object detection
    sharplo : float, optional
        The lower bound on sharpness for object detection
    sharphi : float, optional
        The upper bound on sharpness for object detection
    silent : bool, optional
        If set, output will be suppressed
    maxiter : int, optional
        Maximum number of iterations for iteratefind
    epsilon : int or float, optional
        Maximum nod/chop deviation for exclusion in chopnod_sort
    eps : float, optional
        Precision to terminate iteration
    ncut : int, optional
        Positive integer cutout region used for Gaussian fitting
        (pixels)
    chopnoddist : (list or tuple) of float, optional
        If provided, will be used to identify the expected positions of the
        peaks and prioritize those that are nearest to where they should be.
        Format is [chop distance, nod distance]
    positive : bool, optional
        If set, only positive peaks will be returned.
    return_object : bool
        If set, return the PeakFinder object rather than the DataFrame
    coordinates : bool
        If set, return tuples of (x, y) coordinates as a list

    Returns
    -------
    pandas.DataFrame or PeakFinder or list
        The default DataFrame represents either shift_image distances to
        move newimage on top of coadded, or positions of the peaks
        in coadded if newimage was not supplied as an input parameter.
        The PeakFinder object contains lots of useful little attributes
        such as the fits, tables and models of coadded and newimage.

        If coordinates=True, a list of (x, y) coordinates will be
        returned.
    """

    # If newimage is input then we calculate the peaks for that image.
    # If not, we set x and y to 0.  If newimage is input then we will
    # return the shift_image, otherwise we return the peak positions.
    findpeak = PeakFinder(coadded, reference=newimage,
                          npeaks=npeaks, fwhm=fwhm,
                          sharplo=sharplo, sharphi=sharphi,
                          roundlo=roundlo, roundhi=roundhi,
                          silent=silent, maxiter=maxiter,
                          epsilon=epsilon, eps=eps, ncut=ncut,
                          chopnoddist=chopnoddist, refine=refine,
                          positive=positive, smooth=smooth)
    findpeak.findpeaks()
    if return_object:
        return findpeak
    else:
        if coordinates:
            if isinstance(findpeak.frame, DataFrame):
                result = []
                if 'x' not in findpeak.frame:
                    return result
                for _, row in findpeak.frame.iterrows():
                    result.append((row['x'], row['y']))
                return result
        else:
            return findpeak.frame
