# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Polarization map pipeline step."""

import os

from astropy import log
from astropy.wcs import WCS
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.visualization.quicklook import make_image


__all__ = ['StepPolMap']


class StepPolMap(StepParent):
    """
    Generate a polarization map image.

    This pipeline step calls
    `sofia_redux.visualization.quicklook.make_image` to generate a PNG
    image for quick-look purposes from polarization data in the input.
    This step must be run after
    `sofia_redux.instruments.hawc.steps.StepRegion`.  It expects a
    'FINAL POL DATA' table extension in the input DataFits.

    The output from this step is identical to the input.  As a side
    effect, a PNG file is saved to disk to the same base name as the input
    file, with 'PMP' replacing the product type indicator.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'polmap', and are named with
        the step abbreviation 'PMP'.

        Parameters defined for this step are:

        maphdu : str
            Extension name to use as the background image.
        lowhighscale : list of float
            Specify a low and high percentile value for the image scale,
            e.g. [0,99].
        scalevec : float
            Scale factor for sizing polarization vectors.
        scale : bool
            If set, vector lengths are scaled by their magnitude.
            If not, all vectors will be the same length.
        rotate : bool
            If set, vectors are rotated to display B-field directions.
        debias : bool
            If set, the debiased polarizations are used to determine
            the vectors.
        colorvec : str
            Vector color.
        colorcontour : str
            Contour color.
        colormap : str
            Image colormap.
        ncontours : int
            Number of contour levels.
        fillcontours : bool
            If set, the contours will be filled, rather than just
            overlaid on the image.
        grid : bool
            If set, a grid will be overlaid on the image.
        title : str
            A title string. If set to 'info', a title will be
            automatically generated.
        centercrop : bool
            If set, the image will be cropped, using the values in
            the 'centercropparams' parameter.
        centercropparams : list of float
            Cropping area to use if centercrop = True.  Should be
            a 4-element list of [RA center (deg), Dec center (deg,
            box width (deg), box height (deg)].
        watermark : str
            Text to add to the plot as a watermark.
        """
        # Name of the pipeline reduction step
        self.name = 'polmap'
        self.description = 'Make Polarization Map'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'pmp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['maphdu', 'STOKES I',
                               'HDU name to be used in the mapfile'])
        self.paramlist.append(['lowhighscale', [0., 99.],
                               'Low/High percentile values for '
                               'image scaling.'])
        self.paramlist.append(['scalevec', 0.0005,
                               'Scale factor for vector sizes'])
        self.paramlist.append(['scale', True,
                               'Set to False to make all vectors '
                               'the same length'])
        self.paramlist.append(['rotate', False,
                               'Use rotated (B-Field) vectors'])
        self.paramlist.append(['debias', True,
                               'Use debiased polarizations'])
        self.paramlist.append(['colorvec', 'lime',
                               'Vector color'])
        self.paramlist.append(['colorcontour', 'blue',
                               'Contour color'])
        self.paramlist.append(['colormap', 'inferno',
                               'Color scheme of the background map'])
        self.paramlist.append(['ncontours', 30,
                               'Number of contour levels'])
        self.paramlist.append(['fillcontours', False,
                               'Fill in contours'])
        self.paramlist.append(['grid', False,
                               'Show grid'])
        self.paramlist.append(['title', 'info',
                               'Title for the map.'
                               'Input text or "info"'])
        self.paramlist.append(['centercrop', False,
                               'Define a center and cropping '
                               'parameters (True) or use entire '
                               'image size (False)?'])
        self.paramlist.append(['centercropparams',
                               [266.41721, -29.006936, 0.05, 0.08],
                               'If centercrop = True, this is a list of: '
                               'RA center (deg), DEC center (deg), '
                               'Area width (deg), Area height (deg)'])
        self.paramlist.append(['watermark', '',
                               'Text to add to the plot as a watermark'])

    def run(self):
        """
        Run the data reduction algorithm.

        Because this step is single-in, single-out (SISO),
        self.datain must be a DataFits object.  The output
        is also a DataFits object, stored in self.dataout.

        The process is:

        1. Read image data and polarization table data.
        2. Generate a plot with vectors overlaid on a flux image.
        """

        self.auxout = []
        self.dataout = self.datain.copy()
        nhwp = self.dataout.getheadval('nhwp')

        if nhwp == 1:
            log.info('No polarization data, so skipping step %s' %
                     self.name)
            return

        maphdu = self.getarg('maphdu')
        scalevec = self.getarg('scalevec')
        rotflag = self.getarg('rotate')
        scaleflag = self.getarg('scale')
        debiasflag = self.getarg('debias')
        colorvec = self.getarg('colorvec')
        colorcon = self.getarg('colorcontour')
        colormap = self.getarg('colormap')
        ncontours = self.getarg('ncontours')
        fillcontours = self.getarg('fillcontours')
        grid = self.getarg('grid')
        title = self.getarg('title')
        centercrop = self.getarg('centercrop')
        centercropparams = self.getarg('centercropparams')
        scale = self.getarg('lowhighscale')
        watermark = self.getarg('watermark')

        # Check if any pol. vect. was found after data cuts
        poldataexist = True
        poldata = None
        header = self.datain.header
        if 'FINAL POL DATA' in self.dataout.tabnames:
            poldata = self.dataout.tableget('FINAL POL DATA')
            nvec = header.get("NVECCUT", 0)
        else:
            poldataexist = False
            nvec = 0

        # Set text for title and subtitle
        # Read data cuts from stepregion
        mini = header.get("CUTMINI", 0)
        minp = header.get("CUTMINP", 0)
        sigma = header.get("CUTPSIGP", 0)
        minisigi = header.get("CUTISIGI", 0)
        maxp = header.get("CUTMAXP", 0)

        obj = header.get('OBJECT', 'UNKNOWN')
        band = header.get('SPECTEL1', 'UNKNOWN')

        # Title
        if title == 'info':
            if rotflag:
                eorb = "B"
                log.debug('Plotting B vectors')
            else:
                eorb = "E"
                log.debug('Plotting E vectors')
            title = "Object: %s, Band: %s, Polarization %s vectors " % \
                    (obj, band[-1], eorb)

        # Subtitle
        fname = os.path.basename(self.datain.filenamebegin) + \
            self.procname.upper() + self.datain.filenameend
        subtitle1 = "Filename: %s" % fname
        if nvec > 0.5:
            subtitle2 = r"Pol. data selection: $p/\sigma p >$ %s ; " \
                        r"%s $< p (%s) <$ %s ; $I/peak(I) >$ %s ; " \
                        r"$I/\sigma I >$ %s ; " \
                        r"N. vectors = %s" % \
                        (str(sigma), str(minp), str(r"\%"), str(maxp),
                         str(mini), str(minisigi), str(nvec))
        else:
            subtitle2 = r"Pol. data selection: $p/\sigma p >$ %s ; " \
                        r"%s $< p (%s) <$ %s ; $I/peak(I) >$ %s ; " \
                        r"$I/\sigma I >$ %s ; " \
                        r"No vectors found after cuts" % \
                        (str(sigma), str(minp), str(r"\%"),
                         str(maxp), str(mini), str(minisigi))

        # Set parameters for Pol vectors
        if poldataexist:
            ra = poldata['Right Ascension']
            dec = poldata['Declination']
            if debiasflag:
                pol = poldata['debiased percent pol']
            else:
                pol = poldata['percent pol']
            if rotflag:
                theta = poldata['rotated theta']
            else:
                theta = poldata['theta']
            if scaleflag is False:
                polplot = scalevec * np.ones(theta.shape[0]) * 5.0
            else:
                polplot = scalevec * pol
        else:
            ra = None
            dec = None
            pol = None
            theta = None
            polplot = None

        # get cropping parameters
        if centercrop:
            crop_region = centercropparams
        else:
            crop_region = None

        # make image
        hdul = self.datain.to_hdulist(save_tables=False)
        fig = make_image(hdul, extension=maphdu, colormap=colormap,
                         scale=scale, n_contour=ncontours,
                         contour_color=colorcon,
                         fill_contours=fillcontours, title=title,
                         subtitle=subtitle1, subsubtitle=subtitle2,
                         crop_region=crop_region,
                         grid=grid, beam=True, watermark=watermark)

        # get axes to add vectors to plot
        ax = fig.get_axes()[0]

        # pixel scale from header WCS
        hwcs = WCS(hdul[maphdu])
        try:
            pixscale = np.max(np.abs(hwcs.celestial.pixel_scale_matrix))
        except ValueError:  # pragma: no cover
            log.warning('Could not read pixel scale from header')
            pixscale = 1.0

        # from aplpy.overlays.Scalebar.show
        length = scalevec * 5.0 / pixscale
        scalebar = AnchoredSizeBar(ax.transData, length,
                                   "p = 5%", 1,
                                   pad=0.5, borderpad=0.4, sep=5,
                                   color=colorvec, frameon=False)
        ax.add_artist(scalebar)

        # Plot vectors
        if poldataexist:
            line_list = []
            for i in range(0, len(ra)):
                if pol[i] >= 0.0:
                    ra1 = ra[i] - 0.5 * polplot[i] \
                        * (np.sin(theta[i] * np.pi / 180.)
                           / np.cos(dec[i] * np.pi / 180.))
                    dec1 = dec[i] - 0.5 * polplot[i] \
                        * np.cos(theta[i] * np.pi / 180.)
                    ra2 = ra[i] + 0.5 * polplot[i] \
                        * (np.sin(theta[i] * np.pi / 180.)
                           / np.cos(dec[i] * np.pi / 180.))
                    dec2 = dec[i] + 0.5 * polplot[i] \
                        * np.cos(theta[i] * np.pi / 180.)
                    line_list.append(
                        np.column_stack([[ra1, ra2], [dec1, dec2]]))

            line_col = LineCollection(line_list, color=colorvec,
                                      linewidth=1, alpha=0.5,
                                      transform=ax.get_transform('world'))
            ax.add_collection(line_col)

        else:
            log.debug('No vectors found.')

        # save the figure
        basename = self.datain.filenamebegin + \
            self.procname.upper() + self.datain.filenameend
        fname = os.path.splitext(basename)[0] + '_polmap.png'
        with np.errstate(invalid='ignore'):
            fig.savefig(fname, dpi=300)

        log.info(f'Saved image to {fname}')
        self.auxout.append(fname)
