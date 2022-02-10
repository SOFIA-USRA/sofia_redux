# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Focus analysis pipeline step."""

from astropy import log
from astropy import wcs as astwcs
from matplotlib.backends.backend_agg \
    import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import numpy as np
from scipy import ndimage, optimize

from sofia_redux.instruments.hawc.stepmoparent import StepMOParent

__all__ = ['StepFocus']


class StepFocus(StepMOParent):
    """
    Calculate an optimal focus value from short calibration scans.

    This step fits and reports the best focus offset from a
    set of image with varying focus values.

    Input for this step is a set calibrated scan maps.  This step is
    typically run after
    `sofia_redux.instruments.hawc.steps.StepStdPhotCal`. The output from
    this step is identical to the input.  It is not typically saved.
    As a side effect, this step produces several PNG images of focus plots,
    written to the same directory and basename as the input file.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'focus', and are named with
        the step abbreviation 'FCS'.

        Parameters defined for this step are:

        widowisgood : bool
            Include widow pixels in the analysis if set.
        medianaverage : bool
            If set, replace missing pixels with a local median
            value.
        boxaverage : int
            Size of the box used in medianaverage.
        autocrop : bool
            If set, the image will be automatcally be cropped to
            center the target.
        cropimage : bool
            If set, the image will be cropped, using 'xyboxcent'
            and 'boxsizecrop' parameters; 'autocrop' overrides
            this option if set.
        xyboxcent : list
            Central [x, y] pixel to crop around, if cropimage
            is set; 'autocrop' overrides this option if set.
        boxsizecrop : int
            Box size to crop to, if cropimage is set; 'autocrop'
            overrides this option if set.
        primaryimg : str
            Image extension name to use for the fit.  If blank,
            the first image extension is used.
        """

        # Name of the pipeline reduction step
        self.name = 'focus'
        self.description = 'Make Focus Plots'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'fcs'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['widowisgood', True,
                               'Include widow pixels in the analysis (T) '
                               'or only good, non-widow pixels (F)'])
        self.paramlist.append(['medianaverage', True,
                               'Run a median average box through the array '
                               'to fill bad pixels (T) or not (F)'])
        self.paramlist.append(['boxaverage', 3,
                               'Size of the median average box '
                               '(if medianaverage is True) in pixels'])
        self.paramlist.append(['autocrop', True,
                               'Crop image automatically around the target '
                               '(w/ boxsize = 1/3 of image size)'])
        self.paramlist.append(['cropimage', False,
                               'Crop portion (box) of the image '
                               'for analysis?'])
        self.paramlist.append(['xyboxcent', [32, 20],
                               'If cropimage = True, central X/Y pixel '
                               'position of the box to be cropped'])
        self.paramlist.append(['boxsizecrop', 20,
                               'If cropimage = True, size of the box '
                               'to be cropped (in pixels)'])
        self.paramlist.append(['primaryimg', '',
                               'Specifies which image will be used for '
                               'the Gaussian fit. If left blank, the '
                               'first image will be used.'])

    def gaussian(self, height, center_x, center_y, width_x, width_y, bgoffset):
        """
        Return a Gaussian function with the given parameters.

        Parameters
        ----------
        height : float
            Gaussian amplitude.
        center_x : float
            Center x pixel.
        center_y : float
            Center y pixel.
        width_x : float
            Gaussian width, x-direction.
        width_y : float
            Gaussian width, y-direction.
        bgoffset : float
            Background level.

        Returns
        -------
        function
            The Gaussian function.  Arguments are x, y.
        """
        width_x = float(width_x)
        width_y = float(width_y)

        def gauss(x, y):
            g = height * np.exp(
                -(((center_x - x) / width_x)**2
                  + ((center_y - y) / width_y)**2) / 2.) + bgoffset
            return g
        return gauss

    def moments(self, data):
        """
        Compute Gaussian parameters from moments.

        Parameters
        ----------
        data : array-like
            The image to fit.
        Returns
        -------
        tuple of float
            Elements are the Gaussian parameters for the 2D distribution:
            height, x, y, width_x, width_y, bgoffset.

        """
        total = np.nansum(data)
        bgoffset = np.nanmedian(data)
        big_y, big_x = np.indices(data.shape)
        ysize, xsize = data.shape
        if abs(total) == 0:
            total = 1e-7

        x = np.nansum(big_x * data) / total
        y = np.nansum(big_y * data) / total

        log.debug("Moments: x=%f y=%f tot=%f bgoff=%f" %
                  (x, y, total, float(bgoffset)))

        # If the initial guess for x and y is outside the array,
        # will assume a guess in the center of the image
        # (this situation might happen in case there is no source
        # in the image. The assumption is used to avoid an index
        # problem in the col and row definitions below
        if x >= xsize or x <= 0 or np.isnan(x):
            x = xsize / 2.
        if y >= ysize or y <= 0 or np.isnan(y):
            y = ysize / 2.
        col = data[int(y), :]
        width_x = np.sqrt(np.nansum(abs((np.arange(col.size) - y)**2 * col))
                          / abs(np.nansum(col)))
        row = data[:, int(x)]
        width_y = np.sqrt(np.nansum(abs((np.arange(row.size) - x)**2 * row))
                          / abs(np.nansum(row)))
        height = np.nanmax(data)
        log.debug("Moments: returning h=%f x=%f y=%f wx=%f wy=%f bg=%f" %
                  (float(height), x, y, width_x, width_y, float(bgoffset)))
        return height, x, y, width_x, width_y, bgoffset

    def fitgaussian(self, data, nanpix, medianaverage):
        """
        Fit a Gaussian function to an image.

        Parameters
        ----------
        data : array-like
            The image to fit.
        nanpix : array-like
            A mask or index array indicating the positions of NaN pixels.
        medianaverage : bool
            If not set, NaN pixels will be replaced with model values.

        Returns
        -------
        height, x, y, width_x, width_y : tuple of float
            Parameters for a Gaussian fit to the data.
        """
        params = list(self.moments(data))
        data_img = data.copy()
        # If medianaverage is False,
        # assume that for bad pixels (NaNs), the value
        # on the array is equal to the model
        # (so that the difference -- errorfunction -- will be zero)
        if not medianaverage:
            model = self.gaussian(*params)(*np.indices(data.shape))
            data_img[nanpix] = model[nanpix]

        # function to fit
        def errorfunction(par):
            return np.ravel(self.gaussian(*par)(*np.indices(data_img.shape))
                            - data_img)

        result = optimize.leastsq(errorfunction, np.array(params))
        amp, centy, centx, widy, widx, offset = result[0]
        success = result[1]

        log.debug("Fit Values: returning h=%f, x=%f y=%f "
                  "wx=%f wy=%f bg=%f" %
                  (amp, centx, centy, widx, widy, offset))
        return amp, centy, centx, widy, widx, offset, success

    def focusplot(self, focus, values, difftotfoc,
                  label, lbl, sign, units=''):
        """
        Find and plot the best fit focus value.

        The best focus value is at either the maximum or minimum of
        the `values`, as fit by a 2nd order polynomial.  Plots
        showing the fit and best value are written to disk.

        Parameters
        ----------
        focus : `list` of float
            Focus values (independent variable).
        values : list of float
            Fit values (dependent variable).
        difftotfoc : float
            Mean difference from the total focus offset.
        label : str
            Long label for the plot.
        lbl : str
            Short label for the plot.
        sign : {-1, 1}
            If -1, best fit is at a maximum.  If 1, best fit is at
            a minimum.
        units : str, optional
            Units for the `values`.
        """
        # Fit focus curve, find best focus
        #   fit(x) = fit[0]*x**2. + fit[1]*x + fit[2]
        fit = np.polyfit(focus, values, 2)
        xax = np.asarray(np.linspace(min(focus), max(focus), 1000))
        yax = fit[0] * xax**2. + fit[1] * xax + fit[2]
        bestfocx = -fit[1] / (2. * fit[0])

        # Set up plot
        fig = Figure(figsize=(6, 5))
        FigureCanvas(fig)
        ax = fig.add_subplot()
        ax.plot(xax, yax, 'r--', linewidth=3)
        ax.plot(focus, values, 'mo', markersize=10)
        ax.set_xlabel(r'Focus TOTAL offsets ($\mu m$)')
        ax.set_ylabel(label)

        # bestfocx is a local extremum, but could be a
        # maximum (ideally it would be a minimum).
        # Test to see if there are points in the fit below bestfocx

        # Create a polynomial using the coefficients from fit
        fittest = np.poly1d(fit)

        # Add DEBUG records to the log for the FWHM values
        log.debug(' %s at bestfoc: %.3f %s' %
                  (lbl, fittest(bestfocx), units))

        # Check if extremum is correct
        if fit[0] * sign < 0 or bestfocx < min(focus) or bestfocx > max(focus):
            # Use np.where to find the index
            # of the minimum fwhmx (creates a tuple)
            # Pull the first value from the tuple
            # and use it as an index for focus
            # This raises a warning in the log, but we can ignore it
            valsign = [sign * v for v in values]
            try:
                # It's possible for
                # np.where(valsign == min(valsign)) to return []
                guessfocx = focus[np.where(valsign == min(valsign))[0]]
            except (IndexError, TypeError):
                log.warning('Error in focus guess - '
                            'just using first value')
                guessfocx = focus[0]

            if sign < 0:
                fig.suptitle('No local maximum found!', fontsize=12)
                log.warning('No local maximum found! Best focus '
                            'may be out of analyzed range.')
            else:
                fig.suptitle('No local minimum found!', fontsize=12)
                log.warning('No local minimum found! Best focus '
                            'may be out of analyzed range.')

            ax.set_title('Suggested best focus position '
                         r'(%s): %.1f $\mu m$' %
                         (lbl, guessfocx))
        else:
            # Do this if bestfocx is at the minimum
            ax.set_title(r'Best TOTAL Offset (%s): %.1f $\mu m$ '
                         r'(Absolute Position $\sim$ %.1f $\mu m$)' %
                         (lbl, bestfocx, bestfocx + difftotfoc),
                         fontsize=11)
            log.info('')
            log.info('Best focus position (%s): %.1f um ' %
                     (lbl, bestfocx))
            log.info('  (Look at images and graphs '
                     'to make sure it is a valid minimum!)')
            log.info('')

            # Plot a black 'x' on the minimum (just for convenience)
            ax.plot(bestfocx, fittest(bestfocx), 'kx', markersize=10, mew=3)

        # Save image
        pngname = self.datain[-1].filename.replace(
            '.fits', '_autofocus_%s.png' % lbl.replace(' ', '_'))
        fig.savefig(pngname)
        self.auxout.append(pngname)
        log.info('Saved result %s' % pngname)

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is multi-in, multi-out (MIMO),
        self.datain must be a list of DataFits objects.  The output
        is also a list of DataFits objects, stored in self.dataout.

        The process is:

        1. Read in each file, extract an image stamp, and fit
           a Gaussian to it.
        2. From the fit Gaussian parameters for all files, calculate
           the best focus value from: the minimum FWHM (x, y, and
           total), and the maximum Gaussian height.
        """
        # Get parameters
        widowisgood = self.getarg('widowisgood')
        medianaverage = self.getarg('medianaverage')
        boxaverage = self.getarg('boxaverage')
        autocrop = self.getarg('autocrop')
        cropimage = self.getarg('cropimage')
        xyboxcent = list(self.getarg('xyboxcent'))
        boxsizecrop = self.getarg('boxsizecrop')
        primaryimg = self.getarg('primaryimg')

        # self.datain must be a list/tuple
        self.nfiles = len(self.datain)
        log.debug('Analysing %d files' % self.nfiles)

        # counter for the number of images
        nfigs = 1

        # input is set to output, unmodified
        self.dataout = self.datain

        # list to hold output .png names
        self.auxout = []

        amplitude = []
        fwhmx = []
        fwhmy = []
        focus = []
        focus_totoff = []
        srcpeak = []
        srcfwhm = []
        image = None
        for i in range(self.nfiles):
            log.debug("ANALYZING FILE %d" % (i + 1))

            # If the primary image HDU isn't specified,
            # use the first image in the file
            if primaryimg == '':
                try:
                    image = self.datain[i].image.copy()
                except AttributeError:
                    msg = 'No image data in file.'
                    log.error(msg)
                    raise ValueError(msg)
                log.debug("No HDU specified. Using first "
                          "image: %s" % self.datain[i].imgnames[0])
            else:
                # If the HDU is specified, use that image
                image = self.datain[i].imageget(primaryimg).copy()
                log.debug("Using specified image: %s" % primaryimg)

            # Determine what to use for the bad pixel mask
            if 'BAD PIXEL MASK' in self.datain[i].imgnames:
                # Read the bad pixel mask
                badpix = self.datain[i].imageget('BAD PIXEL MASK')
                log.debug("Bad Pixel Mask found")
            elif 'IMAGE MASK' in self.datain[i].imgnames:
                # Read the image mask
                imgmask = self.datain[i].imageget('IMAGE MASK')
                log.debug("Didn't find Bad Pixel Mask - "
                          "Using Image Mask")

                # get bad pixel mask, as zero, set to 3 (bad)
                # wherever imgmask==Nan
                badpix = np.zeros_like(imgmask)
                badpix[np.isnan(imgmask)] = 3
            else:
                # If there's no bad pixel or image mask, use a zero array
                badpix = np.zeros_like(image)
                log.debug("No Bad Pixel or Image Mask - "
                          "Using zero array")

            # Autocrop
            if autocrop:
                header = self.datain[i].header
                tempwcs = astwcs.WCS(header)
                ratarget = self.datain[i].getheadval('CRVAL1')
                dectarget = self.datain[i].getheadval('CRVAL2')
                pix1, pix2 = tempwcs.wcs_world2pix(ratarget, dectarget, 1)
                cropimage = True
                xyboxcent[0] = pix1
                xyboxcent[1] = pix2
                boxsizecrop = np.mean(image.shape) / 3.

            # Option to crop part of the image around a central pixel
            if cropimage:
                x1 = int(xyboxcent[0] - boxsizecrop / 2.)
                x2 = int(xyboxcent[0] + boxsizecrop / 2.)
                y1 = int(xyboxcent[1] - boxsizecrop / 2.)
                y2 = int(xyboxcent[1] + boxsizecrop / 2.)
                if x1 < 0 or x1 > image.shape[1]:
                    x1 = 0
                    log.warning('Crop box x1 invalid, was set to 0')
                if y1 < 0 or y1 > image.shape[0]:
                    y1 = 0
                    log.warning('Crop box y1 invalid, was set to 0')
                if x2 <= x1 or x2 > image.shape[1]:
                    x2 = image.shape[1]
                    log.warning('Crop box x2 invalid, was set to width')
                if y2 <= y1 or y2 > image.shape[0]:
                    y2 = image.shape[0]
                    log.warning('Crop box y2 invalid, was set to height')
                image = image[y1:y2, x1:x2]
                badpix = badpix[y1:y2, x1:x2]

            # Select between use widow pixels or use only good pixels
            if widowisgood:
                nanpix = np.where(badpix > 2)
            else:
                nanpix = np.where(badpix != 0)

            # Choose between median average the image
            #  (to get rid of bad pixels)
            # or keep the bad pixels as NaNs in the 2D gaussian fit
            if medianaverage:
                # Assign bad pixels to 0 to allow the box median averaging
                image[nanpix] = 0.
                # Added to make sure there's no NANs
                image[np.where(image != image)] = 0.
                image = ndimage.filters.uniform_filter(image, size=boxaverage)
            else:
                image[nanpix] = np.nan

            # Fit 2d Gaussian (offset is not used in later code)
            amp, centy, centx, widy, widx, _offset, success = \
                self.fitgaussian(image, nanpix, medianaverage)

            # Append values only if gaussian fit was successful
            if success in [1, 2, 3, 4]:
                fwhmx_img = 2.355 * np.abs(widx)
                fwhmy_img = 2.355 * np.abs(widy)
                amplitude.append(amp)
                fwhmx.append(fwhmx_img)
                fwhmy.append(fwhmy_img)

                # Compute average focus
                focval_st = self.datain[i].getheadval('FOCUS_ST')
                focval_en = self.datain[i].getheadval('FOCUS_EN')
                focval = np.mean([focval_st, focval_en])
                focus.append(focval)
                if 'FCSTOFF' in self.datain[i].header:
                    fcstoff = self.datain[i].getheadval('FCSTOFF')
                    focus_totoff.append(fcstoff)
                else:
                    # if no FoCus Total OFFset in the header, copy focval
                    fcstoff = focval
                    focus_totoff.append(focval)
                    log.debug('FCSTOFF not found; using focus value')

                # Plotting figure and ellipse around object
                fig = Figure()
                FigureCanvas(fig)
                ax = fig.add_subplot()
                nfigs += 1
                ymax, xmax = image.shape
                ax.imshow(image, cmap='gray',
                          extent=[0, xmax - 1, 0, ymax - 1],
                          interpolation='none')
                ax.plot(centx, ymax - 1 - centy,
                        'm+', markersize=15, mew=2)

                ellipse = Ellipse(xy=(centx, ymax - 1 - centy),
                                  width=2.355 * np.abs(widx),
                                  height=2.355 * np.abs(widy),
                                  edgecolor='b', fc='None', lw=2)
                ax.add_patch(ellipse)
                ax.annotate('Img %s; Focus: %.1f microns, '
                            'Tot. Off: %.1f microns' %
                            (nfigs - 1, float(focval), fcstoff),
                            xy=(1, 0.95 * ymax), color='.5')
                ax.annotate('Gaussian FWHM X / Y: %.1f / %.1f pixels' %
                            (fwhmx_img, fwhmy_img),
                            xy=(1, 0.90 * ymax), color='.5')
                ax.annotate('Gaussian center X / Y: %.1f / %.1f pixels' %
                            (centx, centy), xy=(1, 1), color='.5')
                ax.annotate('Gaussian height: %1f' % amp,
                            xy=(1, 0.85 * ymax), color='.5')

                # Get scanmap estimates (from header of first table)
                dat = self.datain[i]
                try:
                    srcfwhm.append(dat.getheadval('SRCFWHM', dat.tabnames[0],
                                                  errmsg=False))
                except (KeyError, IndexError):
                    srcfwhm.append(0)
                try:
                    srcpeak.append(dat.getheadval('SRCPEAK', dat.tabnames[0],
                                                  errmsg=False))
                except (KeyError, IndexError):
                    srcpeak.append(0)
            else:
                # unsuccessful gaussian fit
                # Plotting figure and state that the fit was unsuccessful
                fig = Figure()
                FigureCanvas(fig)
                ax = fig.add_subplot()
                nfigs += 1
                ymax, xmax = image.shape
                ax.imshow(image, cmap='gray',
                          extent=[0, xmax - 1, 0, ymax - 1],
                          interpolation='none')
                ax.annotate('Gaussian fit was unsuccessful',
                            xy=(1, ymax - 3), color='k')
                log.debug('Gaussian fit was unsuccessful '
                          'for image {}'.format(i + 1))

            # Save the Plot
            pngname = self.datain[-1].filename.replace(
                '.fits', '_autofocus_image%s.png' % (nfigs - 1))
            fig.savefig(pngname)
            self.auxout.append(pngname)
            log.info('Saved result %s' % pngname)

        # Calculate difference between focus and focus_totoff
        difftotfoc = float(np.mean(np.asarray(focus)[:]
                                   - np.asarray(focus_totoff)[:]))

        # At least 3 successful gaussian fits are required
        # to attempt a parabolic fit of FWHM x Focus
        # Also attempt a fit for Gaussian height
        if len(focus_totoff) >= 3:
            # FWHM x Focus for X
            self.focusplot(focus_totoff, fwhmx, difftotfoc,
                           'FWHM Along X Axis (pix)',
                           'FWHM X', 1, units='pix')

            # FWHM x Focus for Y
            self.focusplot(focus_totoff, fwhmy, difftotfoc,
                           'FWHM Along Y Axis (pix)',
                           'FWHM Y', 1, units='pix')

            # FWHM x Focus for XY
            self.focusplot(focus_totoff + focus_totoff,
                           fwhmx + fwhmy, difftotfoc,
                           'FWHM Along X and Y Axis (pix)',
                           'FWHM XY', 1, units='pix')

            # Focus for Gaussian Height
            try:
                imgunit = self.datain[0].header['BUNIT']
            except KeyError:
                imgunit = 'Img Units'
            self.focusplot(focus_totoff, amplitude, difftotfoc,
                           'Gaussian Height ({})'.format(imgunit),
                           'Amplitude', -1, units=imgunit)

            # Select only data with valid inputs
            # (i.e. srcpeak not zero)
            srcn = len(srcpeak)
            srcfoc = [focus_totoff[i]
                      for i in range(srcn) if srcpeak[i] != 0]
            srcp = [srcpeak[i] for i in range(srcn) if srcpeak[i] != 0]
            srcselfwhm = [srcfwhm[i]
                          for i in range(srcn) if srcpeak[i] != 0]
            if len(srcfoc) > 0:
                # Focus from Scanmap Peak/Int
                self.focusplot(srcfoc, srcp, difftotfoc,
                               'Peak from scan map fit ({})'.format(imgunit),
                               'Peak', -1, units=imgunit)
                # Focus from Scanmap FWHM
                self.focusplot(srcfoc, srcselfwhm, difftotfoc,
                               'FWHM from scan map fit (arcsec)',
                               'FWHM-C', 1, units='arcsec')
            else:
                log.debug('Scan map fit keys not found.')

        elif image is not None:
            # Plotting last figure again and state that
            # there are not enough points for parabolic fit
            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot()
            ymax, xmax = image.shape
            ax.imshow(image, cmap='gray', extent=[0, xmax - 1, 0, ymax - 1],
                      interpolation='none')
            ax.annotate('There are not enough successful',
                        xy=(1, ymax - 5), color='w')
            ax.annotate('Gaussian fits to fit a parabola (minimum 3)',
                        xy=(1, ymax - 9), color='w')
            log.info('There are not enough successful Gaussian '
                     'fits to fit a parabola (minimum 3)')
            pngname = self.datain[-1].filename.replace(
                '.fits', '_autofocus_xyaxis.png')
            fig.savefig(pngname)
            self.auxout.append(pngname)
            log.info('Saved result %s' % pngname)
