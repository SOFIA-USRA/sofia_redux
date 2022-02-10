# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import SparseEfficiencyWarning
from sklearn.neighbors import radius_neighbors_graph
import warnings

from sofia_redux.scan.channels.channel_data.color_arrangement_data import (
    ColorArrangementData)
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['SingleColorChannelData']


class SingleColorChannelData(ColorArrangementData):

    def __init__(self, channels=None):
        super().__init__(channels=channels)
        self.row = None
        self.col = None
        self.position = None
        self.independent = None

    @property
    def default_field_types(self):
        result = super().default_field_types
        result.update({'position': (Coordinate2D, 'arcsec'),
                       'independent': True,
                       'row': -1,
                       'col': -1})
        return result

    def get_overlap_distances(self, overlap_indices):
        """
        Calculates the overlap distances.

        The overlap distances are stored in the `overlaps` attribute values.
        This should be a csr_sparse matrix of shape (n_channels, n_channels)
        where overlaps[i, j] gives the distance between channel i and
        channel j.

        Parameters
        ----------
        overlap_indices : scipy.sparse.csr.csr_matrix (bool)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where a `True` value for overlap_indices[i, j] signals that
            channel `i` overlaps with the channel `j`.

        Returns
        -------
        distances, unit : scipy.sparse.csr.csr_matrix, astropy.units.Unit
            `distances` is a Compressed Sparse Row (CSR) matrix of shape
            (channels, channels) and of float type where distances[i, j] gives
            the distance between channels i and j.  `unit` gives the distance
            unit.
        """
        matrix_rows, matrix_cols = overlap_indices.nonzero()

        distances = np.hypot(
            self.position.x[matrix_rows] - self.position.x[matrix_cols],
            self.position.y[matrix_rows] - self.position.y[matrix_cols])

        good_positions = np.isfinite(distances)
        overlap_distances = csr_matrix((distances[good_positions],
                                       (matrix_rows[good_positions],
                                        matrix_cols[good_positions])),
                                       shape=overlap_indices.shape)

        return overlap_distances, self.position.unit

    def calculate_overlap_values(self, overlap_distances, point_size):
        """
        Calculates the overlap values based on overlap distances.

        The overlap values are stored in the `overlaps` attribute values.
        This should be a csr_sparse matrix of shape (n_channels, n_channels)
        where overlaps[i, j] gives the overlap value between channels i and j.

        Parameters
        ----------
        overlap_distances : scipy.sparse.csr.csr_matrix (float)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where distances[i, j] gives the distance between channels i and j.
            Although these are float values, they should be converted be in
            units of `point_size`.
        point_size : astropy.units.Quantity
            The point size for calculating the overlaps.  Typically, the beam
            FWHM.

        Returns
        -------
        None
        """
        sigma = gaussian_fwhm_to_sigma * point_size.value
        matrix_rows, matrix_cols = m_ind = overlap_distances.nonzero()

        if matrix_rows.size > 0:
            values = np.exp(
                -0.5 * (np.asarray(overlap_distances[m_ind])[0] / sigma) ** 2)
        else:
            values = np.zeros(0, dtype=float)

        flagged_channels = np.nonzero(
            self.is_flagged(self.flagspace.flags.BLIND
                            | self.flagspace.flags.DEAD)
            | self.independent)[0]

        valid = np.isin(matrix_rows, flagged_channels, invert=True)
        valid &= np.isin(matrix_cols, flagged_channels, invert=True)
        valid &= np.isfinite(values)

        overlap_values = csr_matrix((values[valid],
                                     (matrix_rows[valid], matrix_cols[valid])),
                                    shape=overlap_distances.shape)

        self.overlaps = overlap_values

    def get_pixel_count(self):
        """
        Return the number of pixels in the channels.

        Returns
        -------
        pixels : int
        """
        return self.channels.size

    def get_pixels(self):
        """
        Return the pixels in the arrangement.

        Returns
        -------
        ChannelData
        """
        return self.channels.data

    def get_mapping_pixels(self, indices=None, name=None, keep_flag=None,
                           discard_flag=None, match_flag=None):
        """
        Creates and returns mapping pixels.

        Parameters
        ----------
        indices : numpy.ndarray (int), optional
            The indices to reference.  If not supplied, defaults to all
            channels.
        name : str, optional
            The name of the channel group.  If not supplied, defaults to the
            name of the channel data.
        discard_flag : int or str or ChannelFlagTypes, optional
            Flags to discard_flag from the new group.
        keep_flag : int or str or ChannelFlagTypes, optional
            Keep channels with these matching flags.
        match_flag : int or str or ChannelFlagTypes, optional
            Keep only channels with a flag exactly matching this flag.

        Returns
        -------
        ChannelGroup
            A newly created channel group.
        """
        return self.channels.get_observing_channels().create_group(
            name=name, keep_flag=keep_flag, discard_flag=discard_flag,
            match_flag=match_flag)

    def get_rcp_string(self, indices=None):
        """
        Return a string representation for the RCP of ALL channels.

        Parameters
        ----------
        indices : numpy.ndarray or slice, optional
            The channel indices (not fixed) to return.  The default is all
            channels.

        Returns
        -------
        str
        """
        if indices is None:
            indices = slice(None)
        x_pos = self.position.x[indices].to('arcsec').value
        y_pos = self.position.y[indices].to('arcsec').value

        df = pd.DataFrame(
            {'ch': list(map(lambda x: "%i" % x, self.fixed_index[indices])),
             '[Gpnt]': list(map(lambda x: "%.3f" % x,
                                self.gain[indices] * self.coupling[indices])),
             '[Gsky]ch': list(map(lambda x: "%.3f" % x, self.gain[indices])),
             'dX': list(map(lambda x: "%.3e" % x, x_pos)),
             'dY': list(map(lambda x: "%.3e" % x, y_pos))})
        return df.to_csv(sep='\t', index=False)

    def get_overlap_indices(self, radius):
        """
        Return a cross-array indicating overlapping indices.

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
        return self.get_geometric_overlap_indices(radius)

    @abstractmethod
    def geometric_rows(self):
        """
        Return the number of geometric rows in the detector array.

        Returns
        -------
        rows : int
        """
        pass

    @abstractmethod
    def geometric_cols(self):
        """
        Return the number of geometric columns in the detector array.

        Returns
        -------
        cols : int
        """
        pass

    def get_geometric_overlap_indices(self, radius):
        """
        Return a cross-array indicating overlapping indices from data.

        Overlaps are calculated based on the geometric properties of channels
        (rows and columns).  A maximum radius must be supplied as well as the
        pixel size indicating the separation between pixels in the x and y
        directions.

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
        rows = self.fixed_index // self.geometric_cols()
        cols = self.fixed_index % self.geometric_cols()
        return self.find_row_col_overlap_indices(radius, rows, cols)

    def find_row_col_overlap_indices(self, radius, rows, cols):
        pixel_sizes = self.channels.get_si_pixel_size()
        x = cols * pixel_sizes.x.value
        y = rows * pixel_sizes.y.value
        position = np.stack((x, y), axis=1)

        max_row = self.geometric_rows() - 1
        max_col = self.geometric_cols() - 1

        keep = (cols <= max_col) & (rows <= max_row)
        keep_inds = np.nonzero(keep)[0]
        position = position[keep]

        short_overlap_indices = self.get_positional_overlap_indices(
            position, radius.value).nonzero()

        matrix_cols = keep_inds[short_overlap_indices[0]]
        matrix_rows = keep_inds[short_overlap_indices[1]]
        overlap_matrix = csr_matrix((np.full(matrix_cols.shape, True),
                                     (matrix_rows, matrix_cols)),
                                    shape=(self.size, self.size))
        return overlap_matrix

    @staticmethod
    def get_positional_overlap_indices(position, radius):
        """
        Return a cross-array indicating overlapping indices from positions.

        Given a search radius, find all overlapping positions.

        Parameters
        ----------
        position : numpy.ndarray (float or int)
            An array of orthogonal 2-dimensional coordinates of shape (n, 2).
            Note that all negative positions will not be included in the
            overlaps (this is a way of flagging default values).
        radius : float
            The maximum radius about which to include overlaps.

        Returns
        -------
        overlap_indices : scipy.sparse.csr.csr_matrix (bool)
            A Compressed Sparse Row (CSR) matrix of shape (channels, channels)
            where a `True` value for overlap_indices[i, j] signals that
            channel `i` overlaps with the channel `j`.
        """
        invalid = np.any(position < 0, axis=1)

        overlap_indices = radius_neighbors_graph(position, radius,
                                                 include_self=False,
                                                 mode='connectivity')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)
            overlap_indices[invalid, :] = 0.0
            overlap_indices[:, invalid] = 0.0
            overlap_indices.eliminate_zeros()

        return overlap_indices.astype(bool)
