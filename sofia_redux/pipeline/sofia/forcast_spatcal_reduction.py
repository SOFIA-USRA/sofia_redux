# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Grism Calibration Reduction pipeline steps"""

import os
import warnings

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.pipeline.sofia.sofia_exception import SOFIAImportError
try:
    import sofia_redux.instruments.forcast
    assert sofia_redux.instruments.forcast
except ImportError:
    raise SOFIAImportError('FORCAST modules not installed')

from sofia_redux.instruments.forcast.hdmerge import hdmerge
from sofia_redux.pipeline.sofia.forcast_reduction import FORCASTReduction
from sofia_redux.pipeline.sofia.parameters.forcast_spatcal_parameters \
    import FORCASTSpatcalParameters
from sofia_redux.pipeline.sofia.forcast_wavecal_reduction \
    import FORCASTWavecalReduction
from sofia_redux.spectroscopy.readwavecal import readwavecal
from sofia_redux.toolkit.utilities.fits import hdinsert, getheader
from sofia_redux.toolkit.fitting.polynomial import polyfitnd
from sofia_redux.toolkit.image.adjust import unrotate90
from sofia_redux.toolkit.interpolate import tabinv


class FORCASTSpatcalReduction(FORCASTWavecalReduction):
    r"""
    FORCAST spectroscopic spatial calibration reduction steps.

    This reduction object defines specialized reduction steps
    for generating spatial calibration data from spectroscopic input
    files.  It is selected by the SOFIA chooser only if a top-level
    configuration flag is supplied (spatcal=True).  The final
    output product from this reduction is a FITS file (\*SCL\*.fits)
    with PRODTYPE = 'spatcal'.  This file can be supplied to the
    standard spectroscopic pipeline, at the make_profiles step,
    to specify a new spatial calibration.
    """
    def __init__(self):
        """Initialize the reduction object."""
        super().__init__()

        # descriptive attributes specific to calibration
        self.name = 'Spatcal'

        # product type definitions for spectral steps
        self.prodtype_map.update(
            {'make_profiles': 'spatial_profile',
             'fit_traces': 'traces_fit'})
        self.prodnames.update(
            {'spatial_profile': 'PRF',
             'traces_fit': 'TFT'})

        # invert the map for quick lookup of step from type
        self.step_map = {v: k for k, v in self.prodtype_map.items()}

        # default recipe and step names
        self.recipe = ['checkhead', 'clean', 'droop', 'nonlin',
                       'stack', 'stack_dithers', 'make_profiles',
                       'locate_apertures', 'trace_continuum',
                       'fit_traces', 'rectify']
        self.processing_steps.update({'fit_traces': 'Fit Trace Positions'})

    def load(self, data, param_class=None):
        """Call parent load, with spatcal parameters."""
        FORCASTReduction.load(self, data,
                              param_class=FORCASTSpatcalParameters)

    def fit_traces(self):
        """Fit a 2D surface to traced positions."""
        param = self.get_parameter_set()
        x_order = param.get_value('x_fit_order')
        y_order = param.get_value('y_fit_order')
        weighted = param.get_value('weighted')
        wavefile = param.get_value('wavefile')
        rotation = param.get_value('rotation')

        xpos, ypos, expected, height = [], [], [], []
        data_shape = None
        appos_arcs = []
        hdr_list = []
        for i, hdul in enumerate(self.input):

            if data_shape is None:
                data_shape = hdul['FLUX'].data.shape

            # traces from previous step
            header = hdul[0].header
            hdr_list.append(header)
            trace_x = hdul['APERTURE_XPOS'].data
            trace_y = hdul['APERTURE_YPOS'].data

            # expected spatial cal from order mask
            sim_spatcal = self._sim_spatcal(data_shape)
            arcsec = sim_spatcal[:, data_shape[1] // 2]

            # calibrated aperture positions
            appos = np.array(self._parse_apertures(header['APPOSO01'], 1)[0])
            appos_arc = np.interp(appos, np.arange(arcsec.size), arcsec)
            appos_arcs.extend(appos_arc)

            # get profile heights from spatial map
            # and expected value from assumed slit height
            smap = hdul['SPATIAL_MAP'].data
            for x, y in zip(trace_x, trace_y):
                height.append(np.abs(smap[int(np.round(y)), int(np.round(x))]))
                # nearest aperture position gives expected spatial pos
                ap = appos_arc[np.argmin(np.abs(appos - y))]
                expected.append(ap)
                xpos.append(x)
                ypos.append(y)

        # 2D surface fit to all positions
        if weighted:
            log.info('Weighting fit by spatial profile height.')
            error = 1 / np.array(height)
        else:
            log.info('Fit is unweighted.')
            error = None
        log.info('')
        sfit_model = polyfitnd(ypos, xpos, expected,
                               [y_order, x_order], error=error,
                               robust=5.0, model=True)
        log.info(sfit_model)

        idx = np.arange(data_shape[0], dtype=float)
        space = np.tile(np.expand_dims(idx, 1), (1, data_shape[1]))
        idx = np.arange(data_shape[1], dtype=float)
        wave = np.tile(np.expand_dims(idx, 0), (data_shape[0], 1))
        sfit_full = sfit_model(space, wave)
        sfit = sfit_full.copy()

        # wavelength calibration from input or pixel positions
        if os.path.isfile(wavefile):
            log.info(f'Using {wavefile} for wavelength calibration.')
            wavecal, _ = readwavecal(wavefile, rotate=rotation)
            wave_hdr = getheader(wavefile)
        else:
            log.info('Using pixel positions for wavelength calibration.')
            wavecal = wave.copy()
            wave_hdr = fits.Header()

            # apply the order mask from the first file
            mask = self.input[0]['BADMASK'].data
            wavecal[mask != 0] = np.nan

        # match spatcal nans to wavecal
        sfit[np.isnan(wavecal)] = np.nan

        results = []
        regions = []
        residuals = []
        save_names = []
        for i, hdul in enumerate(self.input):
            # record data
            del hdul[0].header['APPOSO01']
            hdul.append(fits.ImageHDU(wavecal, name='WAVECAL'))
            hdul.append(fits.ImageHDU(sfit, name='SPATCAL'))

            # update output name
            outname = self.update_output(hdul, self.filenum[i],
                                         self.prodtypes[self.step_index])

            # save if desired
            if param.get_value('save'):
                self.write_output(hdul, outname)

            results.append(hdul)

        # also write final spatcal file
        header = hdmerge(hdr_list, hdr_list[0])

        # rotate if needed before saving
        rs = unrotate90(sfit.copy(), rotation)
        rw = unrotate90(wavecal.copy(), rotation)
        hdinsert(header, 'ROTATION', rotation, 'Rotate 90deg value')

        calfile = fits.HDUList(
            fits.PrimaryHDU(header=header,
                            data=np.array([rw, rs])))
        hdinsert(calfile[0].header, 'WCTYPE', '2D',
                 comment='Wavelength calibration type')
        hdinsert(calfile[0].header, 'WXDEG', wave_hdr.get('WXDEG', 1),
                 comment='X polynomial degree for 2D wavecal')
        hdinsert(calfile[0].header, 'WYDEG', wave_hdr.get('WYDEG', 0),
                 comment='Y polynomial degree for 2D wavecal')
        hdinsert(calfile[0].header, 'WCOEFF', wave_hdr.get('WCOEFF', ''),
                 comment='Wavelength fit coefficients')
        hdinsert(calfile[0].header, 'SXDEG', x_order,
                 comment='X polynomial degree for 2D spatcal')
        hdinsert(calfile[0].header, 'SYDEG', y_order,
                 comment='Y polynomial degree for 2D spatcal')
        hdinsert(calfile[0].header, 'SCOEFF',
                 ','.join(str(c) for c in sfit_model.coefficients),
                 comment='Spatial fit coefficients')
        hdinsert(calfile[0].header, 'NORDERS', 1,
                 comment='Number of orders')
        hdinsert(calfile[0].header, 'ORDERS', '1',
                 comment='Order numbers')

        outname = self.getfilename(header, update=True,
                                   prodtype='SCL', filenum=self.filenum)
        calfile[0].header['FILENAME'] = os.path.basename(outname)
        calfile[0].header['PRODTYPE'] = 'spatcal'
        self.write_output(calfile, outname)

        # make a region file to display
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            min_line = np.nanmax(np.nanmin(sfit, axis=0))
            max_line = np.nanmin(np.nanmax(sfit, axis=0))
        slines = sorted([min_line, max_line] + appos_arcs)
        trace_fit = []
        for xval in idx:
            aptab = []
            for line in slines:
                aptab.append(tabinv(sfit_full[:, int(xval)],
                                    line, missing=np.nan))
            trace_fit.append(aptab)
        trace_fit = np.array(trace_fit).T

        region = self._trace_region(header, self.filenum, 'TFT',
                                    xpos, ypos, idx, trace_fit,
                                    fit_direction='x')
        regions.append(region)

        # keep residuals for plotting
        ds = np.nanmean(sfit[1:, :] - sfit[:-1, :])
        residuals_data = [xpos, ypos,
                          sfit_model.stats.residuals / ds]
        residuals.append(residuals_data)
        pngname = outname.replace('SCL', 'RSD')
        pngname = os.path.join(self.output_directory,
                               os.path.splitext(pngname)[0] + '.png')
        save_names.append(pngname)

        self.input = results
        self.set_display_data(regions=regions, residuals=residuals)

        # save residual plot to disk, after assembled in display data
        self._save_residual_plot(save_names)

        log.info('')
