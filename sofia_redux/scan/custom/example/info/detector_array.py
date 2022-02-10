# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['ExampleDetectorArrayInfo']


class ExampleDetectorArrayInfo(InfoBase):

    # An 11x11 pixel array
    COLS = 11
    ROWS = 11

    boresight_index = Coordinate2D(
        (np.array([ROWS, COLS]) / 2) - 0.5, unit=None)

    pixel_size = 2 * units.Unit('arcsec')  # Nyquist ish

    def __init__(self):
        super().__init__()
        self.pixels = self.ROWS * self.COLS
        self.pixel_sizes = Coordinate2D([self.pixel_size, self.pixel_size])

    def get_sibs_position(self, row, col):
        """
        Given a subarray, row, and column, return the pixel position.

        The SIBS position are in tEl, tXel coordinates in units of the
        `pixel_size` attribute.

        Parameters
        ----------
        row : int or float or numpy.ndarray (int or float)
            The channel/pixel detector row.
        col : int or float or numpy.ndarray (int or float)
            The channel/pixel detector column.

        Returns
        -------
        position : Coordinate2D
            The pixel (x, y) pixel positions.
        """
        position = Coordinate2D()
        position.set([self.boresight_index.x - col,
                      self.boresight_index.y - row])
        position.scale(self.pixel_size)
        return position

    def initialize_channel_data(self, data):
        """
        Apply this information to create and populate the channel data.

        Parameters
        ----------
        data : ExampleChannelData

        Returns
        -------
        None
        """
        index = np.arange(self.pixels)
        data.fixed_index = index
        data.set_default_values()
        data.col = index % self.COLS
        data.row = data.mux = index // self.COLS
        data.bias_line = np.right_shift(data.row, 1)
        data.flag = np.zeros(data.size, dtype=int)
        data.channel_id = np.array(
            [f'{x[0]},{x[1]}' for x in zip(data.row, data.col)])
        data.calculate_sibs_position()
