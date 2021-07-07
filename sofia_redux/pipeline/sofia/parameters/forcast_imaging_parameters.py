# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FORCAST Imaging parameter sets."""

from copy import deepcopy

from astropy.io import fits

from sofia_redux.pipeline.sofia.parameters.forcast_parameters import \
    FORCASTParameters, DEFAULT


# Store default values for all parameters here.
# They could equivalently be read from a file, or
# constructed programmatically.  All keys are optional;
# defaults are specified in the ParameterSet object.
# All 'key' values should be unique.
IMAGE_DEFAULT = {
    'stack': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'add_frames',
         'name': "Add all frames instead of subtracting",
         'value': False,
         'description': 'Generates a sky image, for diagnostic purposes.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'jbclean',
         'name': "Apply 'jailbar' correction",
         'value': True,
         'description': 'If set, the jailbar pattern will be '
                        'removed after stacking.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bgscale',
         'name': 'Scale frames to common level',
         'value': False,
         'description': 'If set, a multiplicative scaling will be applied.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bgsub',
         'name': 'Subtract residual background',
         'value': False,
         'description': 'If set, an additive background level '
                        'will be removed.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'secctr',
         'name': 'Background section center',
         'value': '',
         'description': "Specify the center point in integers as 'x,y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'secsize',
         'name': 'Background section size',
         'value': '',
         'description': "Specify in integers as 'size_x,size_y'.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'bgstat',
         'name': 'Residual background statistic',
         'wtype': 'combo_box',
         'options': ['median', 'mode'],
         'option_index': 0,
         'description': 'Select the statistic to use to calculate '
                        'the residual background.'},
    ],
    'undistort': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'pinfile',
         'name': 'Pinhole locations',
         'value': '',
         'description': 'Text file containing x,y locations to model',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'transform_type',
         'name': 'Transform type',
         'wtype': 'combo_box',
         'options': ['piecewise-affine', 'polynomial'],
         'option_index': 1,
         'description': 'Select the warping algorithm used to '
                        'apply the distortion correction. \nThe '
                        'polynomial algorithm is more traditional; \n'
                        'the piecewise-affine algorithm may '
                        'be more accurate for central regions.'},
        {'key': 'extrapolate',
         'name': 'Extrapolate solution',
         'value': True,
         'description': 'If not set, edges of the image beyond known '
                        'model inputs will be set to NaN.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'merge': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'cormerge',
         'name': 'Merging algorithm',
         'wtype': 'combo_box',
         'options': ['Header shifts', 'Centroid', 'Cross-correlation',
                     'No shift'],
         'option_index': 3,
         'description': 'Select the merging style. \nCentroid '
                        'is default for flux standards; no shift '
                        'is recommended for most other data.'},
    ],
    'register': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'corcoadd',
         'name': 'Registration algorithm',
         'wtype': 'combo_box',
         'options': ['Header shifts', 'Centroid', 'Cross-correlation',
                     'Use first WCS (no image shift)', 'Use WCS as is'],
         'option_index': 4,
         'description': 'Select the registration style.'},
        {'key': 'offsets',
         'name': 'Override offsets for all images',
         'value': '',
         'description': "Specify semi-colon separated offsets, as x,y.\n"
                        "For example, for three input images, "
                        "specify '0,0;2,0;0,2'\nto leave the first as is,"
                        "shift the second two pixels to the right\nin x, "
                        "and shift the third two pixels up in y.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'mfwhm',
         'name': 'Expected FWHM for centroiding (pix)',
         'value': '',
         'description': "Specify the expected FWHM in pixels, for "
                        "the centroiding algorithm.",
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'xyshift',
         'name': 'Maximum shift for cross-correlation',
         'value': '',
         'description': "Specify the maximum allowed shift in x and y "
                        "for the cross-correlation algorithm.",
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
    'tellcor': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'coadd': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_coadd',
         'name': 'Skip coaddition',
         'value': False,
         'description': 'Set to skip coadd of input files '
                        'and propagate separate images instead.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'reference',
         'name': 'Reference coordinate system',
         'wtype': 'combo_box',
         'options': ['First image', 'Target position'],
         'option_index': 1,
         'description': 'Select the reference coordinate system.'},
        {'key': 'method',
         'name': 'Combination method',
         'wtype': 'combo_box',
         'options': ['mean', 'median', 'resample'],
         'option_index': 1,
         'description': 'Select the combination method.'},
        {'key': 'weighted',
         'name': 'Use weighted mean',
         'value': True,
         'description': 'If set, the average of the data will be '
                        'weighted by the variance.  '
                        'Ignored for method=median.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'robust',
         'name': 'Robust combination',
         'value': True,
         'description': 'If set, data will be sigma-clipped '
                        'before combination for mean or median '
                        'methods.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'threshold',
         'name': 'Outlier rejection threshold (sigma)',
         'value': 8.0,
         'description': 'Specify the number of sigma to use in '
                        'sigma clip for robust algorithms.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'maxiters',
         'name': 'Maximum sigma-clipping iterations',
         'value': 5,
         'description': 'Specify the maximum number of outlier '
                        'rejection iterations to use if robust=True.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'smoothing',
         'name': 'Gaussian width for smoothing (pixels)',
         'value': 1.0,
         'description': 'Specify the width of the smoothing kernel '
                        '(resample method only).',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'fluxcal': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'rerun_phot',
         'name': 'Re-run photometry for standards',
         'value': False,
         'description': 'If set, photometry will be re-calculated on the\n'
                        'input image, using below parameters.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'srcpos',
         'name': 'Source position (x,y in pix)',
         'value': '',
         'description': 'Initial guess position for source photometry.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'fitsize',
         'name': 'Photometry fit size (pix)',
         'description': 'Subimage size for profile fits.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'fwhm',
         'name': 'Initial FWHM (pix)',
         'description': 'Starting value for FWHM in profile fits.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'profile',
         'name': 'Profile type',
         'wtype': 'combo_box',
         'options': ['Moffat', 'Gaussian'],
         'option_index': 0,
         'description': 'Select the profile type to fit.'},
    ],
    'imgmap': [
        {'key': 'colormap',
         'name': 'Color map',
         'value': 'plasma',
         'description': 'Matplotlib color map name.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'scale',
         'name': 'Flux scale for image',
         'value': [0.25, 99.9],
         'description': 'Specify a low and high percentile value for '
                        'the image scale, e.g. [0,99].',
         'dtype': 'floatlist',
         'wtype': 'text_box'},
        {'key': 'n_contour',
         'name': 'Number of contours',
         'value': 0,
         'description': 'Set to 0 to turn off countours.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'contour_color',
         'name': 'Contour color',
         'value': 'gray',
         'description': 'Matplotlib color name.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'fill_contours',
         'name': 'Filled contours',
         'value': False,
         'description': 'If set, contours will be filled instead of overlaid.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'grid',
         'name': 'Overlay grid',
         'value': False,
         'description': 'If set, a coordinate grid will be overlaid.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'beam',
         'name': 'Beam marker',
         'value': True,
         'description': 'If set, a beam marker will be added to the plot.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'watermark',
         'name': 'Watermark text',
         'value': '',
         'description': 'Text to add to image as a watermark.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'crop_border',
         'name': 'Crop NaN border',
         'value': True,
         'description': 'If set, any remaining NaN or zero-valued border '
                        'will be cropped out of plot.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
}


class FORCASTImagingParameters(FORCASTParameters):
    """Reduction parameters for the FORCAST Imaging pipeline."""
    def __init__(self, default=None, drip_cal_config=None,
                 drip_config=None, pipecal_config=None):
        """
        Initialize parameters with default values.

        The various config files are used to override certain
        parameter defaults for particular observation modes,
        or dates, etc.

        Parameters
        ----------
        drip_cal_config : dict-like, optional
            Reduction mode and auxiliary file configuration mapping,
            as returned from the sofia_redux.instruments.forcast
            `getcalpath` function.
        drip_config : dict-like, optional
            DRIP configuration, as loaded by the
            sofia_redux.instruments.forcast `configuration` function.
        pipecal_config : dict-like, optional
            Flux calibration and atmospheric correction configuration,
            as returned from the pipecal `pipecal_config` function.
        """
        if default is None:
            default = DEFAULT.copy()
            default.update(IMAGE_DEFAULT)

        super().__init__(default=default,
                         drip_cal_config=drip_cal_config,
                         drip_config=drip_config)

        self.pipecal_config = pipecal_config

        # merge/register option strings as expected
        # by sofia_redux.instruments.forcast configuration
        self.merge_opt = ['HEADER', 'CENTROID', 'XCOR',
                          'NOSHIFT', 'WCS', 'USER']

    def copy(self):
        """
        Return a copy of the parameters.

        Overrides default copy to add in config attributes.

        Returns
        -------
        Parameters
        """
        new = super().copy()
        new.drip_cal_config = deepcopy(self.drip_cal_config)
        new.drip_config = deepcopy(self.drip_config)
        new.pipecal_config = deepcopy(self.pipecal_config)
        return new

    def undistort(self, step_index):
        """
        Modify parameters for the undistort step.

        Sets default pinfile, using `drip_cal_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if (self.drip_cal_config is not None
                and 'pinfile' in self.drip_cal_config):
            self.current[step_index].set_value(
                'pinfile', self.drip_cal_config['pinfile'])

    def merge(self, step_index):
        """
        Modify parameters for the merge step.

        Sets the merging algorithm default (cormerge) using
        `drip_config` and `drip_cal_config`.

        If the observation is a flux standard, the merging
        algorithm is set to centroiding, by default. Otherwise,
        the default is read from the DRIP config file.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if (self.drip_config is not None
                and self.drip_cal_config is not None):
            from sofia_redux.instruments.forcast.getpar import getpar
            header = fits.Header()

            cormerge = getpar(header, 'CORMERGE', dtype=str,
                              default='NOSHIFT')

            # modify by obstype or chop/nod mode
            if self.drip_cal_config['obstype'] == 'STANDARD_FLUX' \
                    or 'npc' in (self.drip_cal_config['cnmode'].lower()):
                cormerge = 'CENTROID'

            # set parameter values in current set
            if cormerge.upper() in self.merge_opt:
                idx = self.merge_opt.index(cormerge)
                self.current[step_index].set_value('cormerge',
                                                   option_index=idx)

    def register(self, step_index):
        """
        Modify parameters for the register step.

        Sets the cross-correlation parameter xyshift and
        the centroiding parameter mfwhm from `drip_config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.drip_config is not None:
            from sofia_redux.instruments.forcast.getpar import getpar
            header = fits.Header()
            xyshift = getpar(header, 'XYSHIFT', dtype=float)
            mfwhm = getpar(header, 'MFWHM', dtype=float)

            # set parameter values in current set
            self.current[step_index].set_value('xyshift', xyshift)
            self.current[step_index].set_value('mfwhm', mfwhm)

    def coadd(self, step_index):
        """
        Modify parameters for the coadd step.

        Skips coadd for spectroscopic acquisition images.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.drip_config is not None:
            if ('boresight' in self.drip_cal_config
                    and 'SLIT' in self.drip_cal_config['boresight']):
                self.current[step_index].set_value('skip_coadd', True)

    def fluxcal(self, step_index):
        """
        Modify parameters for the fluxcal step.

        Sets the photometry parameters fitsize and fwhm
        from `pipecal_config`.

        Default values for these parameters, by instrument and mode,
        are defined in pipecal configuration files.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.pipecal_config is not None:
            self.current[step_index].set_value(
                'fitsize', self.pipecal_config['fitsize'])
            self.current[step_index].set_value(
                'fwhm', self.pipecal_config['fwhm'])
