# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FLITECAM parameter sets."""

from sofia_redux.pipeline.parameters import Parameters


FLITECAM_DEFAULT = {
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
    'specmap': [
        {'key': 'colormap',
         'name': 'Color map',
         'value': 'plasma',
         'description': 'Matplotlib color map name.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'scale',
         'name': 'Flux scale for image',
         'hidden': True,
         'value': [0.25, 99.9],
         'description': 'Specify a low and high percentile value for '
                        'the image scale, e.g. [0,99].',
         'dtype': 'floatlist',
         'wtype': 'text_box'},
        {'key': 'n_contour',
         'name': 'Number of contours',
         'hidden': True,
         'value': 0,
         'description': 'Set to 0 to turn off countours.',
         'dtype': 'int',
         'wtype': 'text_box'},
        {'key': 'contour_color',
         'name': 'Marker color',
         'value': 'gray',
         'description': 'Matplotlib color name.',
         'dtype': 'str',
         'wtype': 'text_box'},
        {'key': 'fill_contours',
         'name': 'Filled contours',
         'hidden': True,
         'value': False,
         'description': 'If set, contours will be filled instead of overlaid.',
         'dtype': 'bool',
         'wtype': 'check_box'},
        {'key': 'grid',
         'name': 'Overlay grid',
         'hidden': True,
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
        {'key': 'spec_scale',
         'name': 'Flux scale for spectral plot',
         'value': [0.25, 99.75],
         'description': 'Specify a low and high percentile value for '
                        'the image scale, e.g. [0,99].',
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


class FLITECAMQuicklookParameters(Parameters):
    """Reduction parameters for the FLITECAM quicklook pipeline."""

    def __init__(self, default=None, **kwargs):
        """
        Initialize parameters with default values.

        The additional parameters are not used here, but are needed for
        compatibility with FORCAST pipeline.
        """
        if default is None:
            default = FLITECAM_DEFAULT.copy()
        super().__init__(default=default)
