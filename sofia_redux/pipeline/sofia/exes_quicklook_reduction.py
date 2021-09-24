# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""EXES Quicklook pipeline steps"""

import os

from astropy import log
import numpy as np

from sofia_redux.pipeline.sofia.forcast_spectroscopy_reduction \
    import FORCASTSpectroscopyReduction
from sofia_redux.pipeline.sofia.parameters.exes_quicklook_parameters \
    import EXESQuicklookParameters


class EXESQuicklookReduction(FORCASTSpectroscopyReduction):
    """
    EXES quicklook reduction steps.

    This reduction object borrows from the FORCAST pipeline to
    make a spectral map for final EXES data products.  It is not
    a full EXES reduction pipeline.

    See `FORCASTSpectroscopyReduction` for more information.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to EXES
        self.instrument = 'EXES'

    def load(self, data, param_class=None):
        """Call parent load, with EXES parameters."""
        FORCASTSpectroscopyReduction.load(
            self, data, param_class=EXESQuicklookParameters)

        # override recipe for last step only
        self.recipe = ['specmap']

    def specmap(self):
        """
        Generate a quick-look spectral plot.

        Calls `sofia_redux.visualization.quicklook.make_image`.

        The output from this step is identical to the input, so is
        not saved.  As a side effect, a PNG file is saved to disk to the
        same base name as the input file, with a '.png' extension.
        """
        from matplotlib.backends.backend_agg \
            import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from sofia_redux.visualization.quicklook import make_spectral_plot

        # get parameters
        param = self.get_parameter_set()
        normalize = param.get_value('normalize')
        scale = param.get_value('scale')
        ignore_outer = param.get_value('ignore_outer')
        atran_plot = param.get_value('atran_plot')
        error_plot = param.get_value('error_plot')
        colormap = param.get_value('colormap')
        overplot_color = param.get_value('overplot_color')
        watermark = param.get_value('watermark')

        if scale[0] <= 0 and scale[1] >= 100:
            scale = None

        for i, hdul in enumerate(self.input):
            header = hdul[0].header
            xunit = hdul[0].header.get('XUNITS', 'UNKNOWN')
            yunit = hdul[0].header.get('YUNITS', 'UNKNOWN')

            # 1D spectrum: check for multiple orders
            labels = None
            aplot = None
            if hdul[0].data.ndim > 2:
                # multi-order or multi-ap
                wave = hdul[0].data[:, 0]
                nspec = hdul[0].data.shape[0]
                labels = [f'Spectrum {j + 1}' for j in range(nspec)]
                spec_flux = hdul[0].data[:, 1]
                spec_err = hdul[0].data[:, 2]
                try:
                    atran = hdul[0].data[:, 3]
                except IndexError:  # pragma: no cover
                    # may be missing for old data
                    atran = np.full_like(spec_flux, np.nan)
                if normalize:
                    norm = np.nanmedian(spec_flux, axis=1)[:, None]
                    spec_flux /= norm
                    spec_err /= norm
                    yunit = 'normalized'

                if ignore_outer > 0:
                    # set the outer N% of frames to NaN
                    wstart = int(ignore_outer * wave.shape[1])
                    wend = int((1 - ignore_outer) * wave.shape[1])
                    spec_flux[:, :wstart] = np.nan
                    spec_flux[:, wend:] = np.nan
                    atran[:wstart] = np.nan
                    atran[wend:] = np.nan
                    log.debug(f'Plotting between w={wstart} and w={wend}')
            else:
                # single order/ap
                wave = hdul[0].data[0]
                spec_flux = hdul[0].data[1]
                spec_err = hdul[0].data[2]
                try:
                    atran = hdul[0].data[3]
                except IndexError:  # pragma: no cover
                    atran = np.full_like(spec_flux, np.nan)
                if normalize:
                    norm = np.nanmedian(spec_flux)
                    spec_flux /= norm
                    spec_err /= norm
                    yunit = 'normalized'
                if ignore_outer > 0:
                    wstart = int(ignore_outer * len(wave))
                    wend = int((1 - ignore_outer) * len(wave))
                    spec_flux[:wstart] = np.nan
                    spec_flux[wend:] = np.nan
                    atran[:wstart] = np.nan
                    atran[wend:] = np.nan
                    log.debug(f'Plotting between w={wstart} and w={wend}')

            # set text for title in plot
            obj = header.get('OBJECT', 'UNKNOWN')
            mode = header.get('INSTCFG', 'UNKNOWN')
            basename = os.path.basename(header.get('FILENAME', 'UNKNOWN'))
            title = f'Object: {obj}, Mode: {mode}\nFilename: {basename}'

            # make the figure for the spectral plot
            fig = Figure(figsize=(8, 5))
            FigureCanvas(fig)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(title)

            if atran_plot and not np.all(np.isnan(atran)):
                if wave.ndim > 1:
                    aplot = [wave.T, atran.T]
                else:
                    aplot = [wave, atran]

            if not error_plot:
                spec_err = None

            # plot spectral flux
            make_spectral_plot(ax, wave, spec_flux, spectral_error=spec_err,
                               scale=scale, labels=labels, colormap=colormap,
                               xunit=xunit, yunit=yunit,
                               title=title, overplot=aplot,
                               overplot_label='Atmospheric Transmission',
                               overplot_color=overplot_color,
                               watermark=watermark)

            # output filename for image
            fname = os.path.splitext(basename)[0] + '.png'
            outname = os.path.join(self.output_directory, fname)

            fig.savefig(outname, dpi=300)
            log.info(f'Saved image to {outname}')
