# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pandas as pd

from sofia_redux.scan.custom.example.flags.channel_flags import (
    ExampleChannelFlags)
from sofia_redux.scan.channels.channel_data.single_color_channel_data import (
    SingleColorChannelData)

__all__ = ['ExampleChannelData']


class ExampleChannelData(SingleColorChannelData):

    flagspace = ExampleChannelFlags

    def __init__(self, channels=None):
        super().__init__(channels=channels)
        self.bias_line = None
        self.mux_gain = None
        self.bias_gain = None
        self.default_info = None

    @property
    def default_field_types(self):
        result = super().default_field_types
        result.update({'mux_gain': 1.0,
                       'bias_gain': 1.0})
        return result

    @property
    def info(self):
        """
        Return the instrument information object.

        Returns
        -------
        ExampleInfo
        """
        info = super().info
        if info is not None:
            return info
        return self.default_info

    def calculate_sibs_position(self):
        """
        Calculate the SIBS position for each pixel.

        Returns
        -------
        None
        """
        self.position = self.info.detector_array.get_sibs_position(
            self.row, self.col)
        self.position.nan(self.is_flagged('BLIND'))

    def set_uniform_gains(self, field=None):
        """
        Sets the gain and coupling to 1.0 for all channels.

        Parameters
        ----------
        field : str, optional
            If supplied, sets all values of the requested field to unity.

        Returns
        -------
        None
        """
        super().set_uniform_gains(field=field)
        if field is None:
            self.mux_gain = np.full(self.size, 1.0)
            self.bias_gain = np.full(self.size, 1.0)

    def to_string(self, indices=None, frame=False):
        """
        Return a string representation of channels.

        Parameters
        ----------
        indices : numpy.ndarray or slice, optional
            The channel indices (not fixed) to return.  The default is all
            channels.
        frame : bool, optional
            If `True`, returns a :class:`pd.DataFrame` instead of a string
            representation.

        Returns
        -------
        str or pd.DataFrame
        """
        df = super().to_string(indices=indices, frame=True)
        if indices is None:
            indices = slice(None)

        df['eff'] = list(map(lambda x: "%.3f" % x, self.coupling[indices]))
        df['Gmux'] = list(map(lambda x: "%.3f" % x, self.mux_gain[indices]))
        df['Gbias'] = list(map(lambda x: "%.3f" % x, self.bias_gain[indices]))
        df['idx'] = list(map(lambda x: str(x), self.fixed_index[indices]))
        df['row'] = list(map(lambda x: str(x), self.row[indices]))
        df['col'] = list(map(lambda x: str(x), self.col[indices]))
        if frame:
            return df
        else:
            return df.to_csv(sep='\t', index=False)

    def initialize_from_detector(self, detector):
        """
        Apply this information to create and populate the channel data.

        Parameters
        ----------
        detector : ExampleDetectorArrayInfo

        Returns
        -------
        None
        """
        detector.initialize_channel_data(self)

    def apply_info(self, info):
        """
        Apply information to the channel data.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        detector_array = info.detector_array
        self.initialize_from_detector(detector_array)
        self.set_sibs_positions(detector_array)
        super().apply_info(info)

    def set_sibs_positions(self, detector_array):
        """
        Set the pixel positions based on detector array information.

        BLIND channels will have NaN pixel positions.  The spatial units will
        be those defined by the detector array for `pixel_xy_size`.  The
        result will be to populate the `position` attribute with an (N, 2)
        array of (x, y) positions.

        Parameters
        ----------
        detector_array : ExampleDetectorArrayInfo

        Returns
        -------
        None
        """
        self.position = detector_array.get_sibs_position(self.row, self.col)
        self.position.nan(self.is_flagged(self.flagspace.flags.BLIND))

    @classmethod
    def read_channel_data_file(cls, filename):
        """
        Read a channel data file and return the information within.

        Returns a `pandas` DataFrame with the following columns:
        {gain, weight, flag, coupling, mux_gain, idx, sub, row, col, unknown}.

        Parameters
        ----------
        filename : str
            The path to a channel data file.

        Returns
        -------
        channel_info : pandas.DataFrame
        """
        column_names = ['gain', 'weight', 'flag', 'coupling', 'mux_gain',
                        'bias_gain', 'fixed_id', 'row', 'col']
        data_types = {'gain': float, 'weight': np.float64,
                      'coupling': np.float64, 'mux_gain': np.float64,
                      'bias_gain': np.float64}
        converters = {'flag': lambda x: cls.flagspace.parse_string(x).value}
        pixel_info = pd.read_csv(filename, delim_whitespace=True, comment='#',
                                 names=column_names, dtype=data_types,
                                 converters=converters).to_dict('index')
        return pixel_info

    def set_channel_data(self, index, channel_info):
        """
        Set the channel info for a selected index.

        Parameters
        ----------
        index : int
            The channel index for which to set new data.
        channel_info : dict
            A dictionary of the form {field: value} where.  The attribute
            field at 'index' will be set to value.

        Returns
        -------
        None
        """
        super().set_channel_data(index, channel_info)
        if channel_info is None:
            return
        self.coupling[index] = channel_info['coupling']
        self.mux_gain[index] = channel_info['mux_gain']
        self.bias_gain[index] = channel_info['bias_gain']

    def geometric_rows(self):
        """
        Return the number of geometric rows in the detector array.

        Returns
        -------
        rows : int
        """
        return self.info.detector_array.ROWS

    def geometric_cols(self):
        """
        Return the number of geometric columns in the detector array.

        Returns
        -------
        cols : int
        """
        return self.info.detector_array.COLS

    def get_geometric_overlap_indices(self, radius):
        """
        Return a cross-array indicating overlapping indices from data.

        Overlaps are calculated based on the geometric properties of channels
        (rows and columns).  A maximum radius must be supplied as well as the
        pixel size indicating the separation between pixels in the x and y
        directions.

        For HAWC_PLUS, the subrow attribute is used instead of row for the
        y-coordinate.

        Parameters
        ----------
        radius : astropy.units.Quantity
            The maximum radius about which to include overlaps.

        Returns
        -------
        overlap_indices : scipy.sparse.csr.csr_matrix (bool)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where a `True` value for overlap_indices[i, j] signals that
            channel `i` overlaps with the channel `j`.
        """
        return self.find_row_col_overlap_indices(radius, self.row, self.col)
