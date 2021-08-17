# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM Imaging parameter sets."""

from sofia_redux.pipeline.sofia.parameters.flitecam_parameters import \
    FLITECAMParameters, DEFAULT


# Store default values for all parameters here.
# They could equivalently be read from a file, or
# constructed programmatically.  All keys are optional;
# defaults are specified in the ParameterSet object.
# All 'key' values should be unique.
IMAGE_DEFAULT = {
    'clip_image': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_clean',
         'name': 'Skip clean',
         'value': False,
         'description': 'Skip bad pixel identification and cleaning',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'datasec',
         'name': 'Data section to clip to',
         'value': [186, 838, 186, 838],
         'description': 'Specify as [xmin, xmax, ymin, ymax]. \n'
                        'Max values are not included.',
         'dtype': 'intlist',
         'wtype': 'text_box'},
    ],
    'make_flat': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'flatfile',
         'name': 'Override flat file',
         'value': '',
         'description': 'FITS file containing a flat to use, \n'
                        'in place of generating one from the data.',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'skip_flat',
         'name': 'Skip gain correction',
         'value': False,
         'description': 'Set to skip making a flat from the input data. \n'
                        'Data will not be gain-corrected.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'correct_gain': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'subtract_sky': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skyfile',
         'name': 'Override sky file',
         'value': '',
         'description': 'FITS file containing a sky image to subtract, \n'
                        'in place of using values derived from the data.',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'skip_sky',
         'name': 'Skip sky subtraction',
         'value': False,
         'description': 'Set to skip sky subtraction.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'sky_method',
         'name': 'Method for deriving sky value',
         'wtype': 'combo_box',
         'options': ['Use flat normalization value', 'Use image median'],
         'option_index': 1,
         'description': 'If the flat was derived from the input data, \n'
                        'its normalization value represents the median \n'
                        'background across all data. If not, the image \n'
                        'median is likely more appropriate.'},
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
         'options': ['Centroid', 'Cross-correlation',
                     'Use first WCS (no image shift)', 'Use WCS as is'],
         'option_index': 3,
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
         'value': 6,
         'description': "Specify the expected FWHM in pixels, for "
                        "the centroiding algorithm.",
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'xyshift',
         'name': 'Maximum shift for cross-correlation',
         'value': 100,
         'description': "Specify the maximum allowed shift in x and y "
                        "for the cross-correlation algorithm.",
         'dtype': 'float',
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
         'value': True,
         'description': 'If set, a coordinate grid will be overlaid.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'beam',
         'name': 'Beam marker',
         'value': False,
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
         'hidden': True,
         'value': False,
         'description': 'If set, any remaining NaN border will be '
                        'cropped out of plot.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
}


class FLITECAMImagingParameters(FLITECAMParameters):
    """Reduction parameters for the FLITECAM Imaging pipeline."""
    def __init__(self, default=None, config=None, pipecal_config=None):
        """
        Initialize parameters with default values.

        The various config files are used to override certain
        parameter defaults for particular observation modes,
        or dates, etc.

        Parameters
        ----------
        config : dict-like, optional
            Reduction mode and auxiliary file configuration mapping,
            as returned from the sofia_redux.instruments.flitecam
            `getcalpath` function.
        pipecal_config : dict-like, optional
            Flux calibration and atmospheric correction configuration,
            as returned from the pipecal `pipecal_config` function.
        """
        if default is None:
            default = DEFAULT.copy()
            default.update(IMAGE_DEFAULT)

        super().__init__(default=default,
                         config=config,
                         pipecal_config=pipecal_config)

        # merge/register option strings as expected
        # by sofia_redux.instruments.forcast configuration
        self.merge_opt = ['CENTROID', 'XCOR',
                          'NOSHIFT', 'WCS', 'USER']

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
                'fitsize', self.pipecal_config.get('fitsize', 138))
            self.current[step_index].set_value(
                'fwhm', self.pipecal_config.get('fwhm', 6.0))
