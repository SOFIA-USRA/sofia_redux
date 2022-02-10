# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pandas as pd

from sofia_redux.scan.custom.hawc_plus.flags.channel_flags import (
    HawcPlusChannelFlags)
from sofia_redux.scan.custom.sofia.channels.channel_data.channel_data import (
    SofiaChannelData)
from sofia_redux.scan.channels.channel_data.single_color_channel_data import (
    SingleColorChannelData)
from sofia_redux.scan.utilities.range import Range

__all__ = ['HawcPlusChannelData']


class HawcPlusChannelData(SingleColorChannelData, SofiaChannelData):

    flagspace = HawcPlusChannelFlags

    def __init__(self, channels=None):
        super().__init__(channels=channels)
        self.sub = None
        self.pol = None
        self.subrow = None
        self.bias_line = None
        self.series_array = None
        self.mux = None
        self.fits_row = None
        self.fits_col = None
        self.fits_index = None
        self.jump = None
        self.has_jumps = None
        self.sub_gain = None
        self.mux_gain = None
        self.pin_gain = None
        self.bias_gain = None
        self.series_gain = None
        self.los_gain = None
        self.roll_gain = None

    @property
    def default_field_types(self):
        result = super().default_field_types
        result.update({'jump': 0.0,
                       'has_jumps': False,
                       'sub_gain': 1.0,
                       'mux_gain': 1.0,
                       'pin_gain': 1.0,
                       'bias_gain': 1.0,
                       'series_gain': 1.0,
                       'los_gain': 1.0,
                       'roll_gain': 1.0})
        return result

    @property
    def info(self):
        """
        Return the instrument information object.

        Returns
        -------
        HawcPlusInfo
        """
        return super().info

    def calculate_sibs_position(self):
        """
        Calculate the SIBS position for each pixel.

        Returns
        -------
        None
        """
        self.position = self.info.detector_array.get_sibs_position(
            self.sub, self.subrow, self.col)
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
        df['Gsub'] = list(map(
            lambda x: "%.3f" % x, self.channels.subarray_gain_renorm[
                self.sub[indices]]))
        df['Gmux'] = list(map(lambda x: "%.3f" % x, self.mux_gain[indices]))
        df['idx'] = list(map(lambda x: str(x), self.fixed_index[indices]))
        df['sub'] = list(map(lambda x: str(x), self.sub[indices]))
        df['row'] = list(map(lambda x: str(x), self.subrow[indices]))
        df['col'] = list(map(lambda x: str(x), self.col[indices]))
        if frame:
            return df
        else:
            return df.to_csv(sep='\t', index=False)

    def validate_pixel_data(self):
        """
        Validates data read from the pixel data file.

        Returns
        -------
        None
        """
        super().validate_pixel_data()

        gain_range = self.configuration.get_range(
            'pixels.gain.range', default=Range(0.3, 3.0))
        coupling_range = self.configuration.get_range(
            'pixels.coupling.range', default=Range(0.3, 3.0))
        exclude_gain = self.configuration.get_float_list(
            'pixels.gain.exclude', default=None)
        exclude_coupling = self.configuration.get_float_list(
            'pixels.coupling.exclude', default=[1.0])

        bad_channels = ~gain_range.in_range(self.gain)
        bad_channels |= ~coupling_range.in_range(self.coupling)
        if exclude_gain is not None:
            for gain_value in exclude_gain:
                bad_channels |= self.gain == gain_value
        if exclude_coupling is not None:
            for coupling_value in exclude_coupling:
                bad_channels |= self.coupling == coupling_value

        self.coupling[bad_channels] = 0.0
        self.set_flags('DEAD', indices=bad_channels)

    def initialize_from_detector(self, detector):
        """
        Apply this information to create and populate the channel data.

        Parameters
        ----------
        detector : HawcPlusDetectorArrayInfo

        Returns
        -------
        None
        """
        index = np.arange(detector.pixels)
        sub = index // detector.subarray_pixels
        keep = detector.has_subarray[sub]
        index = index[keep]

        self.fixed_index = index
        self.set_default_values()

        self.col = index % detector.subarray_cols
        self.row = index // detector.subarray_cols
        self.sub = index // detector.subarray_pixels
        self.pol = self.sub >> 1

        self.fits_row = self.subrow = self.row % detector.rows
        self.bias_line = np.right_shift(self.row, 1)

        self.mux = (self.sub * detector.subarray_cols) + self.col
        self.fits_col = self.mux
        self.series_array = np.right_shift(self.mux, 2)

        self.fits_index = (self.fits_row * detector.FITS_COLS) + self.fits_col

        self.flag = np.zeros(self.size, dtype=int)
        blind_flag = self.flagspace.flags.BLIND.value
        self.flag[self.subrow == detector.DARK_SQUID_ROW] = blind_flag

        self.channel_id = np.array(
            [detector.POL_ID[pol]
             + str(sub & 1) + f'[{subrow},{col}]'
             for (pol, sub, subrow, col)
             in zip(self.pol, self.sub, self.subrow, self.col)])

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

        center = detector_array.get_sibs_position(
            0, 39.0 - detector_array.boresight_index.y,
            detector_array.boresight_index.x)

        self.set_sibs_positions(detector_array)

        self.position = detector_array.get_sibs_position(
            self.sub, self.subrow, self.col)

        self.set_reference_position(center)
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
        detector_array : HawcPlusDetectorArrayInfo

        Returns
        -------
        None
        """
        self.position = detector_array.get_sibs_position(
            self.sub, self.subrow, self.col)
        self.position.nan(self.is_flagged(self.flagspace.flags.BLIND))

    def set_reference_position(self, reference_position):
        """
        Sets the reference position by subtracting from the position field.

        Parameters
        ----------
        reference_position : Coordinate2D
            The reference position to subtract

        Returns
        -------
        None
        """
        self.position.subtract(reference_position)

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
        column_names = ['gain', 'weight', 'flag', 'coupling', 'sub_gain',
                        'mux_gain', 'fixed_id', 'sub', 'subrow', 'col']
        data_types = {'gain': float, 'weight': np.float64,
                      'coupling': np.float64, 'mux_gain': np.float64}
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

    def geometric_rows(self):
        """
        Return the number of geometric rows in the detector array.

        Returns
        -------
        rows : int
        """
        return self.info.detector_array.rows

    def geometric_cols(self):
        """
        Return the number of geometric columns in the detector array.

        Returns
        -------
        cols : int
        """
        return self.info.detector_array.subarray_cols

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
        return self.find_row_col_overlap_indices(radius, self.subrow, self.col)

    def read_jump_hdu(self, hdu):
        """
        Read the jump levels from a given FITS HDU.

        Parameters
        ----------
        hdu : fits.PrimaryHDU

        Returns
        -------
        None
        """
        self.jump = hdu.data[self.col, self.row].astype(int)
