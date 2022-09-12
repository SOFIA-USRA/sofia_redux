# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.source_models.beams.asymmetry_2d import Asymmetry2D


__all__ = ['Image2D1']


class Image2D1(Image2D):

    def __init__(self, x_size=None, y_size=None, z_size=None, dtype=float,
                 data=None, blanking_value=np.nan, unit=None):
        """
        Create an Image2D1 instance.

        The Image2D1 is an extension of the :class:`Image2D` class that
        consists of 3 (x, y, z) dimensions.  The (x, y) coordinate plane is
        replicated at each z coordinate.  Thus, we only need to store the
        (x, y) plane, and z values rather than each (x, y, z) coordinate.

        Parameters
        ----------
        x_size : int, optional
            The number of pixels for the image in the x-direction.  Will only
            be applied if `data` is `None` and `y_size` and `z_size` are also
            set.
        y_size : int, optional
            The number of pixels for the image in the y-direction.  Will only
            be applied if `data` is `None` and `x_size` and `z_size` are also
            set.
        z_size : int, optional
            The number of pixels for the image in the z-direction.  Will only
            be applied if `data` is `None` and `x_size` and `y_size` are also
            set.
        dtype : type, optional
            The data type of the data array.
        data : numpy.ndarray, optional
            Data to initialize the flagged array with.  If supplied, sets the
            shape of the array.  Note that the data type will be set to that
            defined by the `dtype` parameter.  Data should be of shape
            (z_size, y_size, x_size).
        blanking_value : int or float, optional
            The blanking value defines invalid values in the data array.  This
            is the equivalent of defining a NaN value.
        unit : str or units.Unit or units.Quantity, optional
            The data unit.
        """
        super().__init__(dtype=dtype, blanking_value=blanking_value, unit=unit)
        if data is not None:
            self.set_data(data)
        elif x_size is not None and y_size is not None and z_size is not None:
            self.set_data_size(x_size, y_size, z_size)

    @property
    def ndim(self):
        """
        Return the number of dimensions in the map image data.

        Returns
        -------
        int
        """
        return 3

    def size_x(self):
        """
        Return the size of the image in the x-direction.

        Returns
        -------
        int
        """
        return int(self.shape[2])

    def size_y(self):
        """
        Return the size of the image in the y-direction.

        Returns
        -------
        int
        """
        return int(self.shape[1])

    def size_z(self):
        """
        Return the size of the image in the z-direction.

        Returns
        -------
        z : int
        """
        return int(self.shape[0])

    def copy(self, with_contents=True):
        """
        Return a copy of the Image2D1.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, copy the image data to the output image.  Otherwise,
            the new copy will be of the same shape, but contain zeroed values.

        Returns
        -------
        image : Image2D1
        """
        return super().copy(with_contents=with_contents)

    def set_data_size(self, x_size, y_size, z_size):
        """
        Set the data shape.

        Parameters
        ----------
        x_size : int
        y_size : int
        z_size : int

        Returns
        -------
        None
        """
        # self.set_data_shape((y_size, x_size, z_size))
        self.set_data_shape((z_size, y_size, x_size))

    def set_data(self, data, change_type=False):
        """
        Set the image data array.

        Parameters
        ----------
        data : numpy.ndarray
            A 3-dimensional image array.
        change_type : bool, optional
            If `True`, change the data type to that of the data.

        Returns
        -------
        None
        """
        data = data.astype(self.dtype)
        self.log_new_data = False  # To not add additional message during super
        super().set_data(data, change_type=change_type)
        self.record_new_data(detail=f'2D1 {self.dtype} (no copy)')

    def new_image(self):
        """
        Return a new image.

        Returns
        -------
        image : Image2D1
        """
        return super().new_image()

    def get_image(self, dtype=None, blanking_value=None):
        """
        Return an image copy, optionally changing type and blanking value.

        Parameters
        ----------
        dtype : type, optional
            The image data type.
        blanking_value : int or float, optional
            The new image blanking value.

        Returns
        -------
        image : Image2D1
        """
        return super().get_image(dtype=dtype, blanking_value=blanking_value)

    def get_asymmetry(self, grid, center_index, angle, radial_range):
        """
        Return the Asymmetry.

        The asymmetry and rms are calculated via for a single plane by::

            asymmetry = sum(d * c) / sum(|d|)
            rms = sum(|c|) / sum(|d|)

        where::

            c = cos(arctan2(dy, dx) - angle)
            dx = x - x_0
            dy = y - y_0

        The sum occurs over all points within the provided `radial_range`
        about the `center_index`.  The coordinates (x, y) are the projected
        coordinates of the image map indices onto the grid, and (x_0, y_0) is
        the `center_index` projected onto the same grid.  The data values d
        are the image data values.

        Parameters
        ----------
        grid : Grid2D
            The grid used to convert image pixel locations to the coordinate
            system in which to calculate the asymmetry.
        center_index : Coordinate2D
            The center index on the image (x_pix, y_pix) from which to
            calculate the asymmetry.
        angle : units.Quantity
            The rotation of the image with respect to the asymmetry,
        radial_range : Range
            The range (in `grid` units) about `center_index` on which to base
            the asymmetry calculation.

        Returns
        -------
        asymmetry, asymmetry_rms : numpy.ndarray, numpy.ndarray
            The asymmetry and asymmetry RMS both of shape (z_size,).
        """
        n_planes = self.size_z()
        asymmetry = np.zeros(n_planes, dtype=float)
        rms = np.zeros(n_planes, dtype=float)
        for z in range(n_planes):
            valid = self.valid[z]
            data = self.data[z]
            asymmetry[z], rms[z] = self.get_data_asymmetry(
                data, valid, grid, center_index, angle, radial_range)

        return asymmetry, rms

    def get_asymmetry_2d(self, grid, center_index, angle, radial_range):
        """
        Return the 2-D asymmetry in each z-plane.

        Calculates the asymmetry in both the x and y directions.  The y
        asymmetry values are calculated by applying an addition rotation of
        90 degrees to supplied angle during the asymmetry derivation.

        Parameters
        ----------
        grid : Grid2D
            The grid used to convert image pixel locations to the coordinate
            system in which to calculate the asymmetry.
        center_index : Coordinate2D
            The center index on the image (x_pix, y_pix) from which to
            calculate the asymmetry.
        angle : units.Quantity
            The rotation of the image with respect to the asymmetry,
        radial_range : Range
            The range (in `grid` units) about `center_index` on which to base
            the asymmetry calculation.

        Returns
        -------
        asymmetry : Asymmetry2D
        """
        right_angle = 90 * units.Unit('degree')
        asymmetry_x, x_rms = self.get_asymmetry(
            grid, center_index, angle, radial_range)
        asymmetry_y, y_rms = self.get_asymmetry(
            grid, center_index, angle + right_angle, radial_range)

        zi = x_rms == 0
        x_weight = np.zeros(x_rms.size, dtype=float)
        x_weight[zi] = np.inf
        x_weight[~zi] = (1 / x_rms[~zi]) ** 2

        zi = y_rms == 0
        y_weight = np.zeros(y_rms.size, dtype=float)
        y_weight[zi] = np.inf
        y_weight[~zi] = (1 / y_rms[~zi]) ** 2

        return Asymmetry2D(x=asymmetry_x, y=asymmetry_y,
                           x_weight=x_weight, y_weight=y_weight)
