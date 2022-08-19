# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import SparseEfficiencyWarning
from sklearn.neighbors import radius_neighbors_graph
import warnings

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.fifi_ls.flags.channel_flags import (
    FifiLsChannelFlags)
from sofia_redux.scan.custom.sofia.channels.channel_data.channel_data import (
    SofiaChannelData)
from sofia_redux.scan.channels.channel_data.single_color_channel_data import (
    SingleColorChannelData)
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.custom.fifi_ls.channels.channel_numba_functions import\
    get_relative_channel_weights

__all__ = ['FifiLsChannelData']

um = units.Unit('um')


class FifiLsChannelData(SingleColorChannelData, SofiaChannelData):

    flagspace = FifiLsChannelFlags

    def __init__(self, channels=None):
        """
        Initialize the channel data for the FIFI-LS instrument.

        Parameters
        ----------
        channels : FifiLsChannels, optional
            The full channel object on which to base the data.
        """
        super().__init__(channels=channels)
        self.spexel = None
        self.spaxel = None
        self.wavelength = None
        self.uncorrected_wavelength = None
        self.response = None
        self.atran = None
        self.spexel_gain = None
        self.spaxel_gain = None
        self.col_gain = None
        self.row_gain = None

    def copy(self):
        """
        Return a copy of the channel data.

        Returns
        -------
        FifiLsChannelData
        """
        return super().copy()

    @property
    def default_field_types(self):
        """
        Return the defaults for the various channel data parameters.

        Returns
        -------
        defaults : dict
        """
        result = super().default_field_types
        result.update({'spexel': -1,
                       'spaxel': -1,
                       'spexel_gain': 1.0,
                       'spaxel_gain': 1.0,
                       'col_gain': 1.0,
                       'row_gain': 1.0,
                       'independent': False,
                       })
        return result

    @property
    def info(self):
        """
        Return the instrument information object.

        Returns
        -------
        sofia_redux.scan.custom.fifi_ls.info.info.FifiLsInfo
        """
        return super().info

    @property
    def central_wavelength(self):
        """
        Return the central wavelength in the channel data.

        Returns
        -------
        wave_center : units.Quantity
        """
        if self.size == 0 or self.wavelength is None:
            return np.nan * um

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(self.wavelength)

    @property
    def spectral_fwhm(self):
        """
        Return the spectral FWHM for the wavelength.

        Returns
        -------
        fwhm : units.Quantity
        """
        wave = self.central_wavelength
        return wave / self.info.instrument.spectral_resolution

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
            self.spexel_gain = np.full(self.size, 1.0)
            self.spaxel_gain = np.full(self.size, 1.0)
            self.col_gain = np.full(self.size, 1.0)
            self.row_gain = np.full(self.size, 1.0)

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
        df['Gspex'] = list(
            map(lambda x: "%.3f" % x, self.spexel_gain[indices]))
        df['Gspax'] = list(
            map(lambda x: "%.3f" % x, self.spaxel_gain[indices]))
        df['idx'] = list(map(lambda x: str(x), self.fixed_index[indices]))
        df['spex'] = list(map(lambda x: str(x), self.spexel[indices]))
        df['spax'] = list(map(lambda x: str(x), self.spaxel[indices]))
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
            'pixels.coupling.exclude', default=None)

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
        Apply detector information to create and populate the channel data.

        The following attributes are determined from the detector::

          - col: The column on the array
          - row: The row on the array
          - spexel: The spexel index
          - spaxel: The spaxel index
          - position: The spaxel offset on the array (arcseconds)
          - wavelength : The spexel wavelength (um). Set to NaN.

        Additionally, the channel string ID is set to::

          <CHANNEL>[<spexel>,<spaxel>]

        where spexel and spaxel are described above and CHANNEL may be one of
        {B, R} for the RED and BLUE channels respectively.

        Parameters
        ----------
        detector : FifiLsDetectorArrayInfo

        Returns
        -------
        None
        """
        detector.initialize_channel_data(self)

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
        column_names = ['gain', 'weight', 'flag', 'coupling', 'spexel_gain',
                        'spaxel_gain', 'row_gain', 'col_gain', 'fixed_id',
                        'spexel', 'spaxel']
        data_types = {'gain': float, 'weight': np.float64,
                      'coupling': np.float64, 'spexel_gain': np.float64,
                      'spaxel_gain': np.float64, 'row_gain': np.float64,
                      'col_gain': np.float64}
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
        self.spexel_gain[index] = channel_info['spexel_gain']
        self.spaxel_gain[index] = channel_info['spaxel_gain']
        self.row_gain[index] = channel_info['row_gain']
        self.col_gain[index] = channel_info['col_gain']

    def calculate_overlaps(self, point_size, maximum_radius=2.0):
        """
        Calculates the overlaps between channels.

        The overlap array (in the `overlaps` attribute) is a csr_sparse array
        of shape (n_channels, n_channels) where overlaps[i, j] givens the
        overlap value of channel j from the channel i.

        Parameters
        ----------
        point_size : Coordinate2D1
            The point size for calculating the overlaps.  Typically, the beam
            FWHM.
        maximum_radius : float, optional
            The maximum radius in units of `point_size` to search for channel
            overlaps.  Will be overwritten by any values present in the
            overlaps.radius configuration section.

        Returns
        -------
        overlap_indices : scipy.sparse.csr.csr_matrix (bool)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where a `True` value for overlap_indices[i, j] signals that
            channel `i` overlaps with the channel `j`.
        """
        maximum_radius = self.configuration.get_float(
            'overlaps.radius.spatial', default=maximum_radius)
        radius = point_size.x * maximum_radius
        spatial_position = self.position.coordinates / radius
        px, py = spatial_position.decompose().value

        scalings = np.full(3, maximum_radius, dtype=float)

        wave_fwhm = point_size.z
        wave_center = self.central_wavelength
        coordinates = [px, py]
        if not np.isnan(wave_fwhm):
            pz = ((self.wavelength - wave_center) /
                  wave_fwhm).decompose().value
            n_fwhms = self.configuration.get_float(
                'overlaps.radius.spectral', default=maximum_radius)
            pz /= n_fwhms
            coordinates.append(pz)
            scalings[2] = n_fwhms

        coordinates = np.asarray(coordinates)
        overlap_indices = radius_neighbors_graph(coordinates.T, 1,
                                                 include_self=False,
                                                 mode='connectivity')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)
            overlap_indices.eliminate_zeros()

        matrix_rows, matrix_cols = overlap_indices.nonzero()

        fwhms = coordinates * scalings[:len(coordinates), None]

        dx = fwhms[0][matrix_rows] - fwhms[0][matrix_cols]
        dy = fwhms[1][matrix_rows] - fwhms[1][matrix_cols]
        deltas = [dx, dy]
        if fwhms.shape[0] == 3:
            deltas.append(fwhms[2][matrix_rows] - fwhms[2][matrix_cols])
        deltas = np.asarray(deltas)
        distances = np.linalg.norm(deltas, axis=0)

        good_positions = np.isfinite(distances)
        overlap_distances = csr_matrix((distances[good_positions],
                                       (matrix_rows[good_positions],
                                        matrix_cols[good_positions])),
                                       shape=overlap_indices.shape)

        # The distances are FWHM normalized
        point_size = 1 * units.dimensionless_unscaled
        self.calculate_overlap_values(overlap_distances, point_size)

    def read_hdul(self, hdul):
        """
        Read an HDU list and apply to the channel data.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList

        Returns
        -------
        None
        """
        self.info.detector_array.initialize_channel_data(self)
        idx = self.spexel, self.spaxel
        do_uncorrected = self.configuration.get_bool('fifi_ls.uncorrected')
        if do_uncorrected and 'UNCORRECTED_LAMBDA' in hdul:
            self.wavelength = hdul['UNCORRECTED_LAMBDA'].data[idx] * um
        else:
            self.wavelength = hdul['LAMBDA'].data[idx] * um

        if 'XS' in hdul and 'YS' in hdul:
            self.populate_positions(hdul['XS'].data, hdul['YS'].data)

        self.apply_hdul_weights(hdul['STDDEV'].data)

        for hdu in hdul:
            extname = hdu.header.get('EXTNAME', 'UNKNOWN').strip().upper()
            if extname == 'UNCORRECTED_LAMBDA':
                self.uncorrected_wavelength = hdu.data[idx] * um
            elif extname == 'ATRAN':
                self.atran = hdu.data[idx]
            elif extname == 'RESPONSE':
                self.response = hdu.data[idx]

    def apply_hdul_weights(self, stddev):
        """
        Read in the weights from the HDU list.

        Parameters
        ----------
        stddev : numpy.ndarray (float)
            The error values from the HDU list of shape
            (n_frames, n_spexels, n_spaxels).

        Returns
        -------
        None
        """
        variance = (stddev ** 2)[:, self.spexel, self.spaxel]
        self.weight = get_relative_channel_weights(variance)

    def populate_positions(self, xs, ys):
        """
        Populate the pixel positions from HDUList data.

        Parameters
        ----------
        xs : numpy.ndarray (float)
            The pixel x-positions of shape (n_frames, n_spexel, n_spaxel) in
            arcseconds.
        ys : numpy.ndarray (float)
            The pixel y-positions of shape (n_frames, n_spexel, n_spaxel) in
            arcseconds.

        Returns
        -------
        None
        """
        # All spexel positions should contain the same coordinates
        xy = Coordinate2D([xs, ys], unit='arcsec')
        self.position = self.info.detector_array.find_pixel_positions(xy)

    def geometric_rows(self):
        """
        Return the number of geometric rows in the detector array.

        Returns
        -------
        rows : int
        """
        return self.info.detector_array.spaxel_rows

    def geometric_cols(self):
        """
        Return the number of geometric columns in the detector array.

        Returns
        -------
        cols : int
        """
        return self.info.detector_array.spaxel_cols
