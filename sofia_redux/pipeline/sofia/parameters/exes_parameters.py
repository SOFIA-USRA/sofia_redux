# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""EXES parameter sets."""

from copy import deepcopy
from sofia_redux.pipeline.parameters import Parameters

__all__ = ['EXESParameters']


# Store default values for all parameters here.
EXES_DEFAULT = {
    'load_data': [
        {'key': 'abort',
         'name': 'Abort reduction for invalid headers',
         'value': True,
         'description': 'If set, the reduction will be '
                        'aborted if the input headers '
                        'do not meet requirements',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'sky_spec',
         'name': 'Extract sky spectrum',
         'value': False,
         'description': 'Set to extract a sky spectrum',
         'dtype': 'bool',
         'wtype': 'check_box'
         },
        {'key': 'cent_wave',
         'name': 'Central wavenumber',
         'value': '',
         'description': 'If the wavenumber is known more precisely than the \n'
                        'value in the WAVENO0 key- word, enter it here to \n'
                        'use it in the distortion correction calculations \n'
                        'and wavelength calibration.',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'hrfl',
         'name': 'HR focal length',
         'value': '',
         'description': 'This distortion parameter may be adjusted \n'
                        'to tune the dispersion solution for \n'
                        'cross-dispersed spectra.',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'xdfl',
         'name': 'XD focal length',
         'value': '',
         'description': 'This distortion parameter may be adjusted to \n'
                        'tune the dispersion solution for long-slit spectra.',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'slit_rot',
         'name': 'Slit rotation',
         'value': '',
         'description': 'This distortion parameter may be adjusted to tune \n'
                        'the slit skewing, to correct for spectral \n'
                        'features that appear tilted across an order.',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'det_rot',
         'name': 'Detector rotation',
         'value': '',
         'description': 'Adjust the detector rotation.',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'hrr',
         'name': 'HRR',
         'value': '',
         'description': 'Adjust the R number for the echelon grating',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'flattamb',
         'name': 'Ambient temperature for the flat mirror',
         'value': '',
         'description': 'Set to override the default ambient temperature '
                        'for the flat mirror. \n'
                        'Typical default is 290K.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'flatemis',
         'name': 'Emissivity for the flat mirror',
         'value': '',
         'description': 'Set to override the default emissivity fraction '
                        'for the flat mirror. \n'
                        'Typical default is 0.1.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'coadd_readouts': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save coadded readouts',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'algorithm',
         'name': 'Readout algorithm',
         'options': ['Default for read mode',
                     'Last destructive only',
                     'First/last frame only',
                     'Second/penultimate frame only'],
         'option_index': 1,
         'description': 'Set the readout mode. The default is to determine '
                        'the best option from the readout pattern. \n'
                        'Last destructive will use a separate dark frame as '
                        'the reset frame. \nFirst/last will use the first '
                        'non-destructive read as the reset.\n '
                        'Second/penultimate will use the second read as the '
                        'reset and the second-to-last as the signal.',
         'wtype': 'combo_box'
         },
        {'key': 'lin_corr',
         'name': 'Apply linear correction',
         'value': False,
         'description': 'Apply linear correction',
         'dtype': 'bool',
         'wtype': 'check_box'
         },
        {'key': 'fix_row_gains',
         'name': 'Correct odd/even row gains',
         'value': False,
         'description': 'Fit and remove gain offsets by odd/even row value.',
         'dtype': 'bool',
         'wtype': 'check_box'
         },
        {'key': 'int_params',
         'name': 'Integration Handling Parameters',
         'wtype': 'group'},
        {'key': 'toss_int_sci',
         'name': 'Science files: toss first integrations (0, 1, or 2)',
         'value': 0,
         'description': 'If set to 0, all integration sets will be used. \n'
                        'If set to 1 or 2, that many integrations will be \n'
                        'discarded from the beginning of the input '
                        'science files.',
         'dtype': 'int',
         'wtype': 'text_box'
         },
        {'key': 'toss_int_flat',
         'name': 'Flat files: toss first integrations (0, 1, or 2)',
         'value': 0,
         'description': 'If set to 0, all integration sets will be used. \n'
                        'If set to 1 or 2, that many integrations will be \n'
                        'discarded from the beginning of the input '
                        'flat files.',
         'dtype': 'int',
         'wtype': 'text_box'
         },
        {'key': 'toss_int_dark',
         'name': 'Dark files: toss first integrations (0, 1, or 2)',
         'value': 0,
         'description': 'If set to 0, all integration sets will be used. \n'
                        'If set to 1 or 2, that many integrations will be \n'
                        'discarded from the beginning of the input '
                        'dark files.',
         'dtype': 'int',
         'wtype': 'text_box'
         },
        {'key': 'copy_integrations',
         'name': 'Copy integrations instead of tossing',
         'value': False,
         'description': 'If set, "tossed" integrations will be replaced with'
                        'frames from the next B nod instead of discarded, '
                        'if possible.',
         'dtype': 'bool',
         'wtype': 'check_box'
         }
    ],
    'make_flat': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_flat',
         'name': 'Save flat as separate file',
         'value': True,
         'description': 'Save flat (FLT) file',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'order_params',
         'name': 'Order Edge Determination Parameters',
         'wtype': 'group'},
        {'key': 'threshold',
         'name': 'Threshold factor (fraction)',
         'value': 0.15,
         'description': 'This value defines the illumination threshold '
                        'for a flat, which determines the edges '
                        'of cross-dispersed orders. \nEnter a '
                        'number between 0 and 1.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'opt_rot',
         'name': 'Optimize rotation angle',
         'value': True,
         'description': 'Unset to prevent the pipeline from '
                        'using a 2D FFT to determine the best rotation '
                        'angle (krot).',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'edge_method',
         'name': 'Edge enhancement method',
         'options': ['Derivative', 'Squared derivative', 'Sobel'],
         'option_index': 0,
         'description': 'Sets the algorithm for enhancing order edges '
                        'prior to FFT.',
         'wtype': 'combo_box'},
        {'key': 'start_rot',
         'name': 'Starting rotation angle',
         'value': '',
         'description': 'Starting value for krot. \nIf not set, the '
                        'value will be taken from a default '
                        'value in configuration.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'predict_spacing',
         'name': 'Predicted spacing',
         'value': '',
         'description': 'Expected order spacing in pixels for cross-dispersed '
                        'orders. \nIf provided, it is used as a first '
                        'guess for the spacing parameter, overriding the '
                        'calculated value.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'order_override_params',
         'name': 'Order Edge Override Parameters',
         'wtype': 'group'},
        {'key': 'bottom_pix',
         'name': 'Bottom pixel for undistorted order (Med/Low)',
         'value': '',
         'description': 'Directly set the bottom edge of the order '
                        'to this value (medium and low modes only).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'top_pix',
         'name': 'Top pixel for undistorted order (Med/Low)',
         'value': '',
         'description': 'Directly set the top edge of the order '
                        'to this value (medium and low modes only).',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'start_pix',
         'name': 'Starting pixel for undistorted order',
         'value': '',
         'description': 'Directly set the left edge of all orders '
                        'to this value.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'end_pix',
         'name': 'Ending pixel for undistorted order',
         'value': '',
         'description': 'Directly set the right edge of all orders '
                        'to this value.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'custom_wavemap',
         'name': 'Custom cross-dispersed order mask',
         'value': '',
         'description': 'This file should contain bottom, '
                        'top, start, and end edges to set '
                        'in the output order mask for cross-dispersed modes. '
                        '\nOne line for each order with '
                        'edge values specified as white-space separated '
                        'integers (B T S E), starting with the top-most '
                        'order (largest B/T). \nValues are post-distortion '
                        'correction and rotation, so that B/T are y values '
                        'and S/E are x values, zero-indexed.',
         'dtype': 'str',
         'wtype': 'pick_file'},
    ],
    'despike': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save generated files',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'combine_all',
         'name': 'Combine all files before despike',
         'value': False,
         'description': 'If set, all frames will be combined into a single '
                        'file before continuing processing.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'ignore_beams',
         'name': 'Ignore beam designation',
         'value': False,
         'description': 'If set, all frames will be compared, '
                        'including A and B nods.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'spike_fac',
         'name': 'Spike factor (sigma)',
         'value': 20.0,
         'description': 'Enter a value for the threshold for a pixel to be '
                        'considered a spike.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'mark_trash',
         'name': 'Mark trashed frames for exclusion',
         'value': False,
         'description': 'If checked, frames with more '
                        'noise than the rest will be identified and '
                        'excluded from future processing.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'propagate_nan',
         'name': 'Propagate NaN values',
         'value': False,
         'description': 'If set, spikes will be set to NaN. \n'
                        'Otherwise, spikes will be replaced with '
                        'average values.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'debounce': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bounce_fac',
         'name': 'Bounce factor',
         'value': 0.0,
         'description': 'Enter a number to use as the amplitude of '
                        'the bounce correction. \nA value of 0 means '
                        'no bounce correction will be applied. \n'
                        'Values greater than 0 apply the first '
                        'derivative bounce (shifting) only. \nValues '
                        'less than 0 also apply the second derivative '
                        'bounce (smoothing). \nTypical nonzero values '
                        'are 0.1 or -0.1.',
         'dtype': 'float',
         'wtype': 'text_box'
         },
        {'key': 'spec_direction',
         'name': 'Spectral direction',
         'value': False,
         'description': 'If set, the bounce correction will be '
                        'applied in the spectral direction instead '
                        'of the spatial direction.',
         'dtype': 'bool',
         'wtype': 'check_box'
         }
    ],
    'subtract_nods': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save generated files',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_nod',
         'name': 'Skip nod subtraction',
         'value': False,
         'description': 'If set, no nod subtraction is performed.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'subtract_sky',
         'name': 'Subtract residual sky (nod-off-slit only)',
         'value': False,
         'description': 'If set, residual sky signal is removed before '
                        'subtracting pairs',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'subtract_dark',
         'name': 'Dark subtraction',
         'value': False,
         'description': 'If set, a dark frame will be '
                        'subtracted from all planes instead of '
                        'subtracting nods.',
         'dtype': 'bool',
         'wtype': 'check_box'}
    ],
    'flat_correct': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_flat',
         'name': 'Skip flat correction',
         'value': False,
         'description': 'If set, the data will not be multiplied '
                        'by the flat.',
         'dtype': 'bool',
         'wtype': 'check_box'}
    ],
    'clean_badpix': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'bp_threshold',
         'name': 'Bad pixel threshold',
         'value': 20.0,
         'description': 'Threshold for a pixel to be considered a bad pixel. '
                        '\nThis value is multiplied by the standard deviation '
                        'of all good pixels in the frame.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'propagate_nan',
         'name': 'Propagate NaN values',
         'value': False,
         'description': 'If set, bad pixels will be set to NaN. \n'
                        'Otherwise, bad pixels will be interpolated over.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'undistort': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'interpolation_method',
         'name': 'Interpolation method',
         'options': ['Piecewise spline', 'Cubic convolution'],
         'option_index': 1,
         'description': 'Choose spline interpolation with specified order '
                        'or parametric cubic convolution. \n'
                        'Cubic convolution replicates the behavior in the '
                        'original IDL pipeline.',
         'dtype': 'str',
         'wtype': 'combo_box'},
        {'key': 'spline_order',
         'name': 'Spline order',
         'value': 3,
         'description': 'Can be 0-5. Ignored if interpolation method is '
                        'cubic convolution.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'block_unilluminated',
         'name': 'Set unilluminated pixels to NaN',
         'value': False,
         'description': 'If set, unilluminated regions will be set to NaN.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'correct_calibration': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_correction',
         'name': 'Skip calibration correction',
         'value': False,
         'description': 'If set, no calibration correction will be performed.',
         'dtype': 'bool',
         'wtype': 'check_box'},
    ],
    'coadd_pairs': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'save_intermediate',
         'name': 'Save modified frames before coadd',
         'value': False,
         'description': 'If set, input frames that have been optionally '
                        'sky-subtracted or shifted will be saved to disk '
                        'for inspection.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'subtract_sky',
         'name': 'Subtract residual sky (nod-on-slit only)',
         'value': False,
         'description': 'If checked, the mean value at each '
                        'wavelength bin will be subtracted to remove '
                        'any residual sky noise.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'shift',
         'name': 'Shift before coadd',
         'value': False,
         'description': 'Set to attempt to shift spectra in the '
                        'spatial direction to align with each other '
                        'before coadding.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'shift_method',
         'name': 'Shift method',
         'options': ['Maximize signal-to-noise', 'Maximize signal (sharpen)'],
         'option_index': 0,
         'description': 'Select whether to maximize the signal '
                        'or the signal-to-noise when determining the '
                        'spatial shift.',
         'dtype': 'str',
         'wtype': 'combo_box'},
        {'key': 'skip_coadd',
         'name': 'Skip coadd',
         'value': False,
         'description': 'If set, coadd is skipped and all input frames are '
                        'treated as separate files.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'coadd_all_files',
         'name': 'Coadd all files',
         'value': False,
         'description': 'If set, all input files will be '
                        'coadded together, as if they were in a single file.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'exclude_pairs',
         'name': 'Exclude pairs',
         'value': '',
         'description': 'Enter comma-separated numbers to '
                        'identify specific nod-subtracted frames to '
                        'exclude from the coadd. \n'
                        'Separate lists for multiple files with a semi-colon.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'threshold',
         'name': 'Robust threshold',
         'value': 8.0,
         'description': 'Set to a number greater than zero to enable'
                        'outlier rejection before coadding. \nSpecified '
                        'as a factor times the standard deviation in '
                        'the data.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'weight_method',
         'name': 'Weighting method',
         'options': ['Uniform weights', 'Weight by flat',
                     'Weight by variance'],
         'option_index': 1,
         'description': 'If not set to uniform weights, this step will '
                        'weight nod pairs by their correlation with a '
                        'spatial template. \nEither the flat or the '
                        'variance on the flux can be used to weight the '
                        'template average.',
         'dtype': 'str',
         'wtype': 'combo_box'},
        {'key': 'override_weights',
         'name': 'Override weights',
         'value': '',
         'description': 'Enter comma-separated numbers to directly '
                        'specify weight values for each input frame. \n'
                        'Separate lists for multiple files with a semi-colon.',
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
    'convert_units': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'skip_conversion',
         'name': 'Skip unit conversion',
         'value': False,
         'description': 'If set, no unit conversion will be performed.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'cal_factor',
         'name': 'Additional calibration scaling',
         'value': 1.0,
         'description': 'If provided, this factor is multiplied '
                        'into all flux extensions, \n'
                        'for additional a priori calibration correction '
                        'to the flux scale.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'zero_level',
         'name': 'Additional calibration offset',
         'value': 0.0,
         'description': 'If provided, this factor is added '
                        'into all flux extensions, \n'
                        'for additional a priori calibration correction '
                        'to the zero level.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'make_profiles': [
        {'key': 'save',
         'name': 'Save output',
         'value': False,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'fit_order',
         'name': 'Row fit order',
         'value': 4,
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
        {'key': 'exclude_orders',
         'name': 'Exclude orders',
         'value': '',
         'description': 'Enter comma-separated numbers to '
                        'identify specific orders to exclude from '
                        'spectral extraction.',
         'dtype': 'str',
         'wtype': 'text_box'}
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
                        'radii. \nIf not, the assumed value will '
                        'be used.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'apsign',
         'name': 'Aperture sign',
         'value': '',
         'description': 'If specified, will be used directly. \nSeparate '
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
                        'start and stop with dash.  \n '
                        'Leave blank to calculate automatically; set to '
                        'None to leave out.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'ap_start',
         'name': 'Override aperture start',
         'value': '',
         'description': 'Set the aperture start. Values should be '
                        'given in arcseconds across the slit. \n'
                        'Separate multiple apertures by commas; '
                        'separate values for multiple files by semi-colons.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'ap_end',
         'name': 'Override aperture end',
         'value': '',
         'description': 'Set the aperture end. Values should be '
                        'given in arcseconds across the slit. \n'
                        'Separate multiple apertures by commas; '
                        'separate values for multiple files by semi-colons.',
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
        {'key': 'threshold',
         'name': 'Bad pixel threshold (sigma)',
         'value': 4.0,
         'description': 'Specify the number of sigma to use in '
                        'for rejecting bad pixels in the fit.',
         'dtype': 'float',
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
         'description': 'If set, the average profile will be used to '
                        'scale all wavelengths.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'fix_bad',
         'name': 'Use spatial profile to fix bad pixels',
         'value': True,
         'description': 'If set, bad pixels are replaced in the spectral '
                        'image.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'threshold',
         'name': 'Bad pixel threshold (sigma)',
         'value': 5.0,
         'description': 'Specify the number of sigma to use in '
                        'sigma clip for robust algorithms.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'atrandir',
         'name': 'Transmission directory',
         'value': '$DPS_SHARE/calibrations/PSG/fits',
         'description': 'Directory containing transmission data. \n'
                        'Ignored if model file is specified.',
         'dtype': 'str',
         'wtype': 'pick_directory'},
        {'key': 'atranfile',
         'name': 'Transmission model file (PSG)',
         'value': '',
         'description': 'FITS file containing atmospheric transmission data.',
         'dtype': 'str',
         'wtype': 'pick_file'},
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
    ],
    'refine_wavecal': [
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'interactive',
         'name': 'Select line interactively',
         'hidden': True,
         'value': False,
         'description': 'If unchecked, no wavelength calibration '
                        'refinement will be done. If checked, a '
                        'GUI window will pop up, allowing selection '
                        'and editing of a spectral feature.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'identify_order',
         'name': 'Order number for identified line',
         'value': '',
         'description': 'Enter the order number of the line to identify.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'identify_line',
         'name': 'Identify line (pixel position)',
         'value': '',
         'description': 'Enter the pixel position of the spectral '
                        'line to identify.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'identify_waveno',
         'name': 'Identify wavenumber',
         'value': '',
         'description': 'Enter the calibrated wavenumber for the '
                        'identified line.',
         'dtype': 'float',
         'wtype': 'text_box'},
    ],
    'merge_orders': [
        {'key': 'general_params',
         'name': 'General Parameters',
         'wtype': 'group'},
        {'key': 'save',
         'name': 'Save output',
         'value': True,
         'description': 'Save output data to disk',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'trim_params',
         'name': 'Trimming Parameters',
         'wtype': 'group'},
        {'key': 'trim',
         'name': 'Interactive trim before merge',
         'hidden': True,
         'value': False,
         'description': 'If set, the spectra from each order '
                        'can be interactively trimmed before merging.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'trim_regions',
         'name': 'Regions to trim before merge',
         'value': '',
         'description': 'Specified regions will be set to NaN before '
                        'merging. \nSpecify as semi-colon separated lists '
                        'of order number : wavenumber regions. \n'
                        'For example, "1:780-785,786-787;2:801.5-802.3" will '
                        'trim 2 regions from order 1 and 1 region from '
                        'order 2.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'merge_params',
         'name': 'Merging Parameters',
         'wtype': 'group'},
        {'key': 'threshold',
         'name': 'Selection threshold (fraction of S/N)',
         'value': 0.10,
         'description': 'Pixels are used if their signal-to-noise ratio is '
                        'greater than this value times the median \n'
                        'signal-to-noise in the order. Set <= 0 to turn off.',
         'dtype': 'float',
         'wtype': 'text_box'},
        {'key': 'statistic',
         'name': 'S/N statistic to compare',
         'options': ['median', 'mean', 'max'],
         'option_index': 2,
         'description': 'S/N threshold refers to values above this statistic, '
                        'for each order.',
         'dtype': 'str',
         'wtype': 'combo_box'},
        {'key': 'noise_test',
         'name': 'Apply selection threshold to noise only',
         'value': False,
         'description': 'Use 1/noise for the S/N check, ignoring '
                        'signal values.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'local_noise',
         'name': 'Use local standard deviation for noise thresholding',
         'value': False,
         'description': 'If set, the input error spectrum is ignored for '
                        'the S/N thresholding.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'local_radius',
         'name': 'Radius for local standard deviation (pixels)',
         'value': 5,
         'description': 'Sets the window for the noise calculation if '
                        'local noise is desired.',
         'dtype': 'int',
         'wtype': 'text_box'},
    ],
    'specmap': [
        {'key': 'normalize',
         'name': 'Normalize spectrum before plotting',
         'value': True,
         'description': 'If set, the spectrum will be divided by '
                        'its median value.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'scale',
         'name': 'Flux scale for plot',
         'value': [0, 100],
         'description': 'Specify a low and high percentile value for '
                        'the image scale, e.g. [0,99].',
         'dtype': 'floatlist',
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
        {'key': 'colormap',
         'name': 'Color map',
         'value': 'plasma',
         'description': 'Matplotlib color map name.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'overplot_color',
         'name': 'Overplot color',
         'value': 'gray',
         'description': 'Matplotlib color name.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'watermark',
         'name': 'Watermark text',
         'value': '',
         'description': 'Text to add to image as a watermark.',
         'dtype': 'str',
         'wtype': 'text_box'},
    ],
}


class EXESParameters(Parameters):
    """Reduction parameters for the EXES pipeline."""

    def __init__(self, default=None, base_header=None):
        """
        Initialize parameters with default values.
        """
        if default is None:
            default = EXES_DEFAULT.copy()
        super().__init__(default=default)

        self.base_header = base_header
        self.sky_spec_set = self.is_sky_spec()

    def copy(self):
        """
        Return a copy of the parameters.

        Overrides default copy to add in configuration attributes.

        Returns
        -------
        Parameters
        """
        new = super().copy()
        new.base_header = deepcopy(self.base_header)
        new.sky_spec_set = self.sky_spec_set
        return new

    def unfittable(self):
        """
        Determine if the observation mode is unfittable.

        "Fittable" modes are compact sources, with spatial size
        significantly less than the slit height. Unfittable modes
        are those for which the SRCTYPE is extended, INSTMODE is
        map, or INSTCFG is high-low and INSTMODE is not nod-on-slit.

        Returns
        -------
        bool
            True if data is not a compact, fittable observation mode;
            False otherwise.
        """
        test = False
        if self.base_header is not None:
            srctype = str(self.base_header.get('SRCTYPE')).lower()
            instcfg = str(self.base_header.get('INSTCFG')).lower()
            instmode = str(self.base_header.get('INSTMODE')).lower()
            test = (('extended' in srctype)
                    or ('map' in instmode)
                    or (('high_low' in instcfg)
                        and ('on_slit' not in instmode)))
        return test

    def is_sky_spec(self):
        """
        Determine if the reduction should be considered a sky spectrum.

        Tests for the value of the SKYSPEC keyword in the base_header
        attribute.

        Returns
        -------
        bool
            True if SKYSPEC is True, False otherwise.
        """
        test = False
        if self.base_header is not None:
            test = self.base_header.get('SKYSPEC', False)
        return test

    def load_data(self, step_index):
        """
        Modify parameters for the load data step.

        Sets sky spectrum defaults if indicated in the base header.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            self.current[step_index].set_value('sky_spec', self.is_sky_spec())

    def subtract_nods(self, step_index):
        """
        Modify parameters for the nod subtraction step.

        Sets default to subtract dark if sky spectrum is indicated.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            if self.is_sky_spec():
                self.current[step_index].set_value('subtract_dark', True)
            else:
                self.current[step_index].set_value('subtract_dark', False)

    def coadd_pairs(self, step_index):
        """
        Modify parameters for the coadd pairs step.

        Turns off coadd by default for INSTMODE = MAP.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            # turn off coadd for mapping mode
            instmode = str(self.base_header.get('INSTMODE')).lower()
            if 'map' in instmode:  # pragma: no cover
                self.current[step_index].set_value('skip_coadd', True)

    def make_profiles(self, step_index):
        """
        Modify parameters for the profile step.

        Sets default background subtraction option according to
        observation mode.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            # set background subtraction to false for extended sources
            # and short slits and sky spectra
            if self.unfittable() or self.is_sky_spec():
                self.current[step_index].set_value('bg_sub', False)
            else:
                # restore to default value
                restore = self.default['make_profiles'].get_value('bg_sub')
                self.current[step_index].set_value('bg_sub', restore)

    def locate_apertures(self, step_index):
        """
        Modify parameters for the aperture location step.

        Sets defaults according to observation parameters in
        base header.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            # set detection method: fix to center for high-low
            # and extended, otherwise auto
            if self.unfittable() or self.is_sky_spec():
                self.current[step_index].set_value('method', 'fix to center')
            else:
                restore = self.default['locate_apertures'].get_value('method')
                self.current[step_index].set_value('method', restore)

                # set 2 aps by default for nod-on-slit
                instmode = str(self.base_header.get('INSTMODE')).lower()
                if 'on_slit' in instmode:
                    self.current[step_index].set_value('num_aps', 2)

    def set_apertures(self, step_index):
        """
        Modify parameters for the aperture setting step.

        Sets defaults according to observation parameters in
        base header.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            # set to full slit for extended or high-low
            if self.unfittable() or self.is_sky_spec():
                self.current[step_index].set_value('full_slit', True)
            else:
                self.current[step_index].set_value('full_slit', False)

    def subtract_background(self, step_index):
        """
        Modify parameters for the background subtraction step.

        Sets defaults according to observation parameters in
        base header.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            if self.unfittable() or self.is_sky_spec():
                self.current[step_index].set_value('skip_bg', True)
            else:
                restore = self.default[
                    'subtract_background'].get_value('skip_bg')
                self.current[step_index].set_value('skip_bg', restore)

    def extract_spectra(self, step_index):
        """
        Modify parameters for the spectral extraction step.

        Sets defaults according to observation parameters in
        base header.

        Parameters
        ----------
        step_index : int
            Reduction recipe index for the step.
        """
        if self.base_header is not None:
            # set to standard for extended or high-low
            if self.unfittable() or self.is_sky_spec():
                self.current[step_index].set_value('method', 'standard')
            else:
                restore = self.default['extract_spectra'].get_value('method')
                self.current[step_index].set_value('method', restore)
