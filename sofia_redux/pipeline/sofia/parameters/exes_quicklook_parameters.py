# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""EXES parameter sets."""

from sofia_redux.pipeline.parameters import Parameters


EXES_DEFAULT = {
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


class EXESQuicklookParameters(Parameters):
    """Reduction parameters for the EXES quicklook pipeline."""
    def __init__(self, default=None, **kwargs):
        """
        Initialize parameters with default values.

        The additional parameters are not used here, but are needed for
        compatibility with FORCAST pipeline.
        """
        if default is None:
            default = EXES_DEFAULT.copy()
        super().__init__(default=default)
