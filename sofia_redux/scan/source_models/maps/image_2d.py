# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.source_models.maps.image import Image
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['Image2D']


class Image2D(Image):

    def __init__(self, x_size=None, y_size=None, dtype=float,
                 data=None, blanking_value=np.nan, unit=None):
        """
        Create an Image2D instance.

        The Image2D is an extension of the :class:`Image` class that restricts
        the number of image dimensions to 2.  Generally, the two dimensions
        will be referred to as x and y where x refers to data columns, and y
        refers to data rows.

        Parameters
        ----------
        x_size : int, optional
            The number of pixels for the image in the x-direction.  Will only
            be applied if `data` is `None` and `y_size` is also set to an
            integer.
        y_size : int, optional
            The number of pixels for the image in the y-direction.  Will only
            be applied if `data` is `None` and `x_size` is also set to an
            integer.
        dtype : type, optional
            The data type of the data array.
        data : numpy.ndarray, optional
            Data to initialize the flagged array with.  If supplied, sets the
            shape of the array.  Note that the data type will be set to that
            defined by the `dtype` parameter.
        blanking_value : int or float, optional
            The blanking value defines invalid values in the data array.  This
            is the equivalent of defining a NaN value.
        unit : str or units.Unit or units.Quantity, optional
            The data unit.
        """
        super().__init__(dtype=dtype, blanking_value=blanking_value, unit=unit)
        if data is not None:
            self.set_data(data)
        elif x_size is not None and y_size is not None:
            self.set_data_size(x_size, y_size)

    @property
    def core(self):
        """
        Return the core data.

        Returns
        -------
        numpy.ndarray or None
        """
        return self.data

    @property
    def ndim(self):
        """
        Return the number of dimensions in the map image data.

        Returns
        -------
        int
        """
        return 2

    def add_proprietary_unit(self):
        """
        Add proprietary units to the local units.

        Returns
        -------
        None
        """
        pass

    def size_x(self):
        """
        Return the size of the image in the x-direction.

        Returns
        -------
        int
        """
        return self.shape[1]

    def size_y(self):
        """
        Return the size of the image in the y-direction.

        Returns
        -------
        int
        """
        return self.shape[0]

    def copy(self, with_contents=True):
        """
        Return a copy of the Image2D.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, copy the image data to the output image.  Otherwise,
            the new copy will be of the same shape, but contain zeroed values.

        Returns
        -------
        image : Image2D
        """
        new = super().copy(with_contents=False)
        if self.size > 0:
            new.set_data_shape(self.shape)
            if with_contents:
                new.paste(self, report=True)
        return new

    def set_data_size(self, x_size, y_size):
        """
        Set the data shape.

        Parameters
        ----------
        x_size : int
        y_size : int

        Returns
        -------
        None
        """
        self.set_data_shape((y_size, x_size))

    def set_data(self, data, change_type=False):
        """
        Set the image data array.

        Parameters
        ----------
        data : numpy.ndarray
            A 2-dimensional image array.
        change_type : bool, optional
            If `True`, change the data type to that of the data.

        Returns
        -------
        None
        """
        data = data.astype(self.dtype)
        self.log_new_data = False  # To not add additional message during super
        super().set_data(data, change_type=change_type)
        self.record_new_data(detail=f'2D {self.dtype} (no copy)')

    def new_image(self):
        """
        Return a new image.

        Returns
        -------
        image : Image2D
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
        image : Image2D
        """
        return super().get_image(dtype=dtype, blanking_value=blanking_value)

    @classmethod
    def read_hdul(cls, hdul, hdu_index):
        """
        Read a specific HDU image from an HDU list, and return an image.

        Parameters
        ----------
        hdul : astropy.io.fits.hdu.hdulist.HDUList
            The HDU list to read.
        hdu_index : int
            The index of the HDU in the provided `hdul` to read.

        Returns
        -------
        image : Image2D
        """
        return super().read_hdul(hdul, hdu_index)

    @classmethod
    def numpy_to_fits(cls, coordinates):
        """
        Convert numpy based (x, y) coordinates/indices to FITS coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray

        Returns
        -------
        Coordinate2D
        """
        coordinates = super().numpy_to_fits(coordinates)
        return Coordinate2D(coordinates)
