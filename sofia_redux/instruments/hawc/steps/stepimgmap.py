# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image map pipeline step."""

import os

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.stepparent import StepParent
from sofia_redux.visualization.quicklook import make_image

__all__ = ['StepImgMap']


class StepImgMap(StepParent):
    """
    Generate a map image.

    This pipeline step calls `sofia_redux.visualization.quicklook.make_image`
    to generate a PNG image for quick-look purposes from imaging data in the
    input.  This step should be run after calibration, as a final step in
    the pipeline.

    The output from this step is identical to the input.  As a side
    effect, a PNG file is saved to disk to the same base name as the input
    file, with a '.png' extension.
    """
    def setup(self):
        """
        Set parameters and metadata for the pipeline step.

        Output files have PRODTYPE = 'imgmap', and are named with
        the step abbreviation 'IMP'.

        Parameters defined for this step are:

        maphdu : str
            Extension name to use as the image.
        lowhighscale : list of float
            Specify a low and high percentile value for the image scale,
            e.g. [0,99].
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
        self.name = 'imgmap'
        self.description = 'Make Image Map'

        # Shortcut for pipeline reduction step and identifier for
        # saved file names.
        self.procname = 'imp'

        # Clear Parameter list
        self.paramlist = []

        # Append parameters
        self.paramlist.append(['maphdu', 'STOKES I',
                               'HDU name to be used in the mapfile'])
        self.paramlist.append(['lowhighscale', [0., 99.],
                               'Low/High percentile values for '
                               'image scaling.'])
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
                               'Title for the map. '
                               'Input text or "info".'])
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

        1. Read image data.
        2. Generate a plot with contours overlaid on a flux image.
        """

        self.auxout = []
        self.dataout = self.datain.copy()

        maphdu = self.getarg('maphdu')
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

        # Set text for title and subtitle
        header = self.datain.header
        obj = header.get('OBJECT', 'UNKNOWN')
        band = header.get('SPECTEL1', 'UNKNOWN')
        if title == 'info':
            title = "Object: %s, Band: %s" % (obj, band[-1])
        fname = os.path.basename(self.datain.filename)
        subtitle = "Filename: %s" % fname

        # get extension name
        maphdu = str(maphdu).upper().strip()
        if maphdu not in self.datain.imgnames:
            # translate names for scan data
            if maphdu == 'STOKES I' \
                    and 'PRIMARY IMAGE' in self.datain.imgnames:
                maphdu = 'PRIMARY IMAGE'
            elif maphdu == 'ERROR I' and 'NOISE' in self.datain.imgnames:
                maphdu = 'NOISE'
        log.debug(f'Mapping extension {maphdu}')

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
                         subtitle=subtitle, crop_region=crop_region,
                         grid=grid, beam=True, watermark=watermark)

        # save the figure
        basename = self.datain.filename
        fname = os.path.splitext(basename)[0] + '.png'
        with np.errstate(invalid='ignore'):
            fig.savefig(fname, dpi=300)

        log.info(f'Saved image to {fname}')
        self.auxout.append(fname)
