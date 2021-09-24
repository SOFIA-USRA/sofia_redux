# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM spectroscopy parameter sets."""

from sofia_redux.pipeline.sofia.parameters.flitecam_parameters import \
    FLITECAMParameters, DEFAULT

SPECTRAL_DEFAULT = {
    'make_image': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'pair_sub',
         'name': 'Subtract pairs',
         'value': True,
         'description': 'If set, pairs of files are subtracted.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'flatfile',
         'name': 'Flat file',
         'value': '',
         'description': 'FITS file containing a flat to divide into the '
                        'data.  Set to empty string to skip flat division.',
         'dtype': 'str',
         'wtype': 'pick_file'},
    ],
    'stack_dithers': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_stack',
         'name': 'Skip dither stacking',
         'value': True,
         'description': 'Set to skip stacking input files '
                        'and propagate separate images instead.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'ignore_dither',
         'name': 'Ignore dither information from header',
         'value': False,
         'description': 'Set to ignore dither information and '
                        'stack all input.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Combination method',
         'wtype': 'combo_box',
         'options': ['mean', 'median', 'sum'],
         'option_index': 0,
         'description': 'Select the combination method.'},
        {'key': 'weighted',
         'name': 'Use weighted mean',
         'value': True,
         'description': 'If set, the average of the data will be '
                        'weighted by the variance.\n'
                        'Ignored for method=median.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'robust',
         'name': 'Robust combination',
         'value': True,
         'description': 'If set, data will be sigma-clipped '
                        'before combination',
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
    ],
    'make_profiles': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'wavefile',
         'name': 'Wave/space calibration file',
         'value': '',
         'description': 'FITS file containing coordinate calibration data',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'slitfile',
         'name': 'Slit correction file',
         'value': '',
         'description': 'FITS file containing slit correction data',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'fit_order',
         'name': 'Row fit order',
         'value': 3,
         'description': 'Polynomial fit order for rows '
                        '(along spectral dimension).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'bg_sub',
         'name': 'Subtract median background',
         'value': True,
         'description': 'If set, the median value along columns will '
                        'be subtracted from the profile.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'atmosthresh',
         'name': 'Atmospheric transmission threshold',
         'value': 0.0,
         'description': 'Transmission values below this threshold are not '
                        'considered when making the spatial profile.\n'
                        'Values are 0-1.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'simwavecal',
         'name': 'Simulate calibrations',
         'value': False,
         'description': 'If set, the data will not be rectified or '
                        'calibrated.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'testwavecal',
         'hidden': True,
         'name': 'Test new calibrations',
         'value': False,
         'description': 'If set, WAVECAL and SPATCAL extensions will be used '
                        'for rectification.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'locate_apertures': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Aperture location method',
         'wtype': 'combo_box',
         'options': ['auto', 'fix to input', 'fix to center'],
         'option_index': 0,
         'description': 'Select the aperture location method.'},
        {'key': 'num_aps',
         'name': 'Number of auto apertures',
         'value': 1,
         'description': 'Number of apertures to look for '
                        'if aperture is not fixed',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'input_position',
         'name': 'Aperture position',
         'value': '',
         'description': 'Starting position(s) for aperture detection, '
                        'comma-separated for apertures, semi-colon '
                        'separated for files.\n'
                        'If method is "fix to input", will be used '
                        'directly.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'fwhm',
         'name': 'Expected aperture FWHM (arcsec)',
         'value': 3.0,
         'description': 'Gaussian FWHM estimate for fit to profile.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'trace_continuum': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'attach_trace_xy',
         'name': 'Attach trace positions table',
         'hidden': True,
         'value': False,
         'description': 'If set, trace x/y positions will be attached\n'
                        'in an additional extension.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Trace method',
         'wtype': 'combo_box',
         'options': ['fix to aperture position', 'fit to continuum'],
         'option_index': 1,
         'description': 'Select the trace method.'},
        {'key': 'fit_order',
         'name': 'Trace fit order',
         'value': 2,
         'description': 'Polynomial fit order for aperture center '
                        '(along spectral dimension).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'fit_thresh',
         'name': 'Trace fit threshold',
         'value': 4.0,
         'description': 'Robust rejection threshold, in sigma.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'step_size',
         'name': 'Fit position step size (pixels)',
         'value': 9,
         'description': 'Step size along trace for fitting locations.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'set_apertures': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'full_slit',
         'name': 'Extract the full slit',
         'value': False,
         'description': 'If set, all other parameters will be ignored.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'refit',
         'name': 'Refit apertures for FWHM',
         'value': True,
         'description': 'If set, the peak position will be fit '
                        'with a Gaussian to determine the aperture '
                        'radii.  If not, the assumed value will '
                        'be used.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'apsign',
         'name': 'Aperture sign',
         'value': '',
         'description': 'If specified, will be used directly. Separate '
                        'files with semi-colon, apertures with comma.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'aprad',
         'name': 'Aperture radius',
         'value': '',
         'description': 'If specified, will be used directly. Separate '
                        'files with semi-colon, apertures with comma.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'psfrad',
         'name': 'PSF radius',
         'value': '',
         'description': 'If specified, will be used directly. Separate '
                        'files with semi-colon, apertures with comma.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'bgr',
         'name': 'Background regions',
         'value': '',
         'description': 'If specified, will be used directly. Separate '
                        'files with semi-colon, regions with comma, '
                        'start and stop with dash.',
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
    'subtract_background': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_bg',
         'name': 'Skip background subtraction',
         'value': False,
         'description': 'Set to skip subtracting background.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bg_fit_order',
         'name': 'Background fit order',
         'value': 0,
         'description': 'Polynomial fit order for background level.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'extract_spectra': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_1d',
         'name': 'Save extracted 1D spectra',
         'value': True,
         'description': 'Save spectra to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Extraction method',
         'wtype': 'combo_box',
         'options': ['optimal', 'standard'],
         'option_index': 0,
         'description': 'Select the extraction method.'},
        {'key': 'use_profile',
         'name': 'Use median profile instead of spatial map',
         'value': False,
         'description': 'If true, the same profile will be used to '
                        'scale all wavelengths.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'fix_bad',
         'name': 'Use spatial profile to fix bad pixels',
         'value': True,
         'description': 'Ignored for optimal extraction.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'threshold',
         'name': 'Bad pixel threshold (sigma)',
         'value': 4.0,
         'description': 'Specify the number of sigma to use in '
                        'sigma clip for robust algorithms.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'flux_calibrate': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_1d',
         'name': 'Save calibrated 1D spectra',
         'value': False,
         'description': 'Save spectra to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_cal',
         'name': 'Skip flux calibration',
         'value': False,
         'description': 'Set to skip calibrating data.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'making_response',
         'name': 'Making response file',
         'value': False,
         'hidden': True,
         'description': 'For use when calling flux_calibrate '
                        'for standards.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'respfile',
         'name': 'Response file',
         'value': '',
         'description': 'FITS file containing instrument response data',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'resolution',
         'name': 'Spectral resolution (l/dl)',
         'value': -9999.,
         'description': 'Specify the resolution of the mode.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'atran_params',
         'name': 'Telluric Correction Parameters',
         'wtype': 'group'},
        {'key': 'atrandir',
         'name': 'ATRAN directory',
         'value': '',
         'description': 'Directory containing ATRAN data.',
         'dtype': 'str',
         'wtype': 'pick_directory'},
        {'key': 'atranfile',
         'name': 'ATRAN file',
         'value': '',
         'description': 'FITS file containing atmospheric transmission data, '
                        'if not default.',
         'dtype': 'str',
         'wtype': 'pick_file'},
        {'key': 'waveshift_params',
         'name': 'Wavelength Shift Parameters',
         'wtype': 'group'},
        {'key': 'auto_shift',
         'name': 'Auto-shift wavelength to telluric spectrum',
         'value': True,
         'description': 'Set to cross-correlate spectrum and telluric '
                        'spectrum to determine wavelength shift.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'auto_shift_limit',
         'name': 'Maximum auto wavelength shift to apply (pix)',
         'value': 2.0,
         'description': 'Higher calculated shifts will not be applied.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'waveshift',
         'name': 'Wavelength shift to apply (pix)',
         'value': 0.0,
         'description': 'Manually specify a wavelength shift correction.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'model_order',
         'name': 'Polynomial order for continuum',
         'value': 1,
         'description': 'Used in optimizing ATRAN and waveshift.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'sn_threshold',
         'name': 'S/N threshold for auto-shift',
         'value': 10.0,
         'description': 'Below this mean S/N value, auto-shift will '
                        'not be attempted.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'combine_spectra': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'registration',
         'name': 'Registration method',
         'wtype': 'combo_box',
         'options': ['Use WCS as is', 'Correct to target position',
                     'Use header offsets'],
         'option_index': 1,
         'description': 'Select the registration method.'},
        {'key': 'method',
         'name': 'Combination method',
         'wtype': 'combo_box',
         'options': ['mean', 'median', 'spectral cube'],
         'option_index': 0,
         'description': 'Select the combination method.'},
        {'key': 'weighted',
         'name': 'Weight by errors',
         'value': True,
         'description': 'If set, flux values are inversely weighted '
                        'by the error values before combination. '
                        'Ignored for method=median.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'combination_params',
         'name': '1-2D Combination Parameters',
         'wtype': 'group'},
        {'key': 'combine_aps',
         'name': 'Combine apertures',
         'value': True,
         'description': 'If set, apertures will be combined.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'robust',
         'name': 'Robust combination',
         'value': True,
         'description': 'If set, data will be sigma-clipped '
                        'before combination',
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
        {'key': 'resample_params',
         'name': '3D Resample Parameters',
         'wtype': 'group'},
        {'key': 'fit_order',
         'name': 'Spatial surface fit order',
         'value': 2,
         'description': 'Set lower for more stable fits.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'fit_window',
         'name': 'Spatial fit window (pixels)',
         'value': 7.0,
         'description': 'Set higher to fit more pixels.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'smoothing',
         'name': 'Spatial smoothing radius (pixels)',
         'value': 2.0,
         'description': 'Gaussian sigma. Set higher to '
                        'smooth over more pixels.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'edge_threshold',
         'name': 'Spatial edge threshold (0-1)',
         'value': 0.7,
         'description': 'Set higher to set more edge pixels to NaN.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'adaptive_algorithm',
         'name': 'Adaptive smoothing algorithm',
         'wtype': 'combo_box',
         'options': ['scaled', 'shaped', 'none'],
         'option_index': 2,
         'description': 'If scaled, only the size is allowed to vary.\n'
                        'If shaped, the kernel shape and rotation may \n'
                        'also vary. If none, the kernel will not vary.'},
    ],
    'make_response': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'model_file',
         'name': 'Standard model file',
         'value': '',
         'description': 'FITS file containing standard model spectrum. '
                        'Leave blank to determine from header.',
         'dtype': 'str',
         'wtype': 'pick_file'},
    ],
    'combine_response': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'scale_params',
         'name': 'Scaling Parameters',
         'wtype': 'group'},
        {'key': 'scale_method',
         'name': 'Scaling method',
         'wtype': 'combo_box',
         'options': ['median', 'highest', 'lowest', 'index', 'none'],
         'option_index': 0,
         'description': 'Select the scaling method.'},
        {'key': 'scale_index',
         'name': 'Index of spectrum to scale to',
         'value': 0,
         'description': 'If scale_method=index, will scale to this spectrum.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'combo_params',
         'name': 'Combination Parameters',
         'wtype': 'group'},
        {'key': 'combine_aps',
         'name': 'Combine apertures',
         'value': False,
         'description': 'If set, apertures will be combined.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'method',
         'name': 'Combination method',
         'wtype': 'combo_box',
         'options': ['mean', 'median'],
         'option_index': 0,
         'description': 'Select the combination method.'},
        {'key': 'weighted',
         'name': 'Weight by errors',
         'value': True,
         'description': 'If set, flux values are inversely weighted '
                        'by the error values before combination. '
                        'Ignored for method=median.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'robust',
         'name': 'Robust combination',
         'value': True,
         'description': 'If set, data will be sigma-clipped '
                        'before combination',
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
        {'key': 'smooth_params',
         'name': 'Smoothing Parameters',
         'wtype': 'group'},
        {'key': 'fwhm',
         'name': 'Smoothing Gaussian FWHM (pixels)',
         'value': 2.0,
         'description': 'FWHM of Gaussian to convolve with.\n'
                        'Set to zero to turn off smoothing.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'specmap': [
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
        {'key': 'watermark',
         'name': 'Watermark text',
         'value': '',
         'description': 'Text to add to image as a watermark.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'ignore_outer',
         'name': 'Fraction of outer wavelengths to ignore',
         'value': 0.0,
         'description': 'Used to block edge effects for noisy '
                        'spectral orders. \n'
                        'Set to 0 to include all wavelengths in the plot.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'atran_plot',
         'name': 'Overplot transmission',
         'value': True,
         'description': 'If set, the atmospheric transmission spectrum will\n '
                        'be displayed in the spectral plot.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'error_plot',
         'name': 'Overplot error range',
         'value': True,
         'description': 'If set, the error range will\n '
                        'be overlaid on the spectral plot.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'spec_scale',
         'name': 'Flux scale for spectral plot',
         'value': [0.25, 99.75],
         'description': 'Specify a low and high percentile value for '
                        'the spectral flux scale, e.g. [0,99].',
         'dtype': 'floatlist',
         'wtype': 'text_box'},
        {'key': 'override_slice',
         'name': 'Override wavelength slice for spectral cube',
         'hidden': True,
         'value': '',
         'description': 'Manually specify the wavelength slice '
                        '(zero-indexed) for the image.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'override_point',
         'name': 'Override spatial point for spectral cube',
         'hidden': True,
         'value': '',
         'description': "Manually specify the spatial "
                        "index for the spectrum, as 'x,y', "
                        "zero-indexed.",
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
}


class FLITECAMSpectroscopyParameters(FLITECAMParameters):
    """Reduction parameters for the FLITECAM spectroscopy pipeline."""
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
            as returned from the pipecal `pipecal_config` function.on.
        """
        if default is None:
            default = DEFAULT.copy()
            default.update(SPECTRAL_DEFAULT)
        super().__init__(default=default,
                         config=config,
                         pipecal_config=pipecal_config)

    def make_image(self, step_index):
        """
        Modify parameters for the make_image step.

        Sets default flat file.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set default flat file from cal config
            if 'maskfile' in self.config:
                self.current[step_index].set_value(
                    'flatfile', self.config['maskfile'])

    def make_profiles(self, step_index):
        """
        Modify parameters for the profile step.

        Sets default wavefile and sets background subtraction
        according to `config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set default wave and slit file from cal config
            if 'wavefile' in self.config:
                self.current[step_index].set_value(
                    'wavefile', self.config['wavefile'])
            if 'slitfile' in self.config:
                self.current[step_index].set_value(
                    'slitfile', self.config['slitfile'])

            # set background subtraction to false for extended sources
            unfittable = 'extended' in str(self.config['srctype']).lower()
            if unfittable:
                self.current[step_index].set_value('bg_sub', False)

    def locate_apertures(self, step_index):
        """
        Modify parameters for the locate apertures step.

        Sets defaults according to observation parameters in
        `config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set detection method: fix to center for slitscan
            # and extended, otherwise auto
            unfittable = 'extended' in str(self.config['srctype']).lower()
            if unfittable:
                self.current[step_index].set_value('method', 'fix to center')
            nod_along = 'along' in str(self.config['cnmode']).lower()
            if nod_along:
                self.current[step_index].set_value('num_aps', 2)

    def trace_continuum(self, step_index):
        """
        Modify parameters for the trace continuum step.

        Sets defaults according to observation parameters in
        `config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set trace method: fix to aperture position for
            # extended, otherwise auto
            unfittable = 'extended' in str(self.config['srctype']).lower()
            if unfittable:
                self.current[step_index].set_value('method',
                                                   'fix to aperture position')

    def set_apertures(self, step_index):
        """
        Modify parameters for the set apertures step.

        Sets defaults according to observation parameters in
        `config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set to full slit for extended or slitscan
            unfittable = 'extended' in str(self.config['srctype']).lower()
            if unfittable:
                self.current[step_index].set_value('full_slit', True)

    def extract_spectra(self, step_index):
        """
        Modify parameters for the set apertures step.

        Sets defaults according to observation parameters in
        `config`.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set to full slit for extended
            unfittable = 'extended' in str(self.config['srctype']).lower()
            if unfittable:
                self.current[step_index].set_value('method', 'standard')

    def flux_calibrate(self, step_index):
        """
        Modify parameters for the flux calibrate step.

        Sets default response file and resolution.
        Turns off auto optimization for G111 grism.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.config is not None:
            # set default response file from cal config
            if 'respfile' in self.config:
                self.current[step_index].set_value(
                    'respfile', self.config['respfile'])
            if 'resolution' in self.config:
                self.current[step_index].set_value(
                    'resolution', self.config['resolution'])

            # set 'making_response' if flux standard
            rsp = 'STANDARD' in self.config['obstype']
            if rsp:
                self.current[step_index].set_value('making_response', True)
                self.current[step_index].set_value('save', False)
