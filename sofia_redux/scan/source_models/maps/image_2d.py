# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.source_models.maps.image import Image
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['Image2D']


class Image2D(Image):

    def __init__(self, x_size=None, y_size=None, dtype=float,
                 data=None, blanking_value=np.nan, unit=None):
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

    # TODO: Re-examine this wrt flagged_array and map_data.

    def copy(self, with_content=True):
        """
        Return a copy of the Image2D.

        Returns
        -------
        Image2D
        """
        new = super().copy()
        if self.size > 0:
            new.set_data_shape(self.shape)
            if with_content:
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

    def destroy(self):
        """
        Destroy the image data.

        Returns
        -------
        None
        """
        super().destroy()
        self.clear_history()

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
        super().set_data(data, change_type=change_type)
        self.record_new_data(detail=f'2D {self.dtype} (no copy)')

    def transpose(self):
        """
        Transpose the data array.

        Returns
        -------
        None
        """
        if self.data is None:
            return
        self.data = self.data.T
        self.add_history('transposed')

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
        Image2D
        """
        change_type = not((dtype is None) or (dtype == self.dtype))
        change_level = not((blanking_value is None)
                           or (blanking_value is self.blanking_value))

        image = self.copy(with_content=True)

        if not change_type and not change_level:
            return image

        if change_type:
            image.dtype = dtype
            image.data = image.data.astype(dtype)

        if change_level:
            image.blanking_value = blanking_value
            image.paste(self, report=True)

    def get_valid_data(self, default=None):
        """
        Return a copy of the image data, optionally replacing invalid points.

        Parameters
        ----------
        default : int or float, optional
            The value to replace blanked values with.

        Returns
        -------
        numpy.ndarray or None
        """
        if self.data is None:
            return None
        data = self.data.copy()
        if default is None:
            return data
        data[~self.valid] = default
        return data

    def auto_crop(self):
        """
        Auto crop the image data.

        The data is cropped to the extent of valid data point indices.

        Returns
        -------
        None
        """
        if self.data is None:
            return
        ranges = self.get_index_range()
        if len(ranges) != 2 or None in ranges:
            return
        self.crop(ranges)

    def edit_header(self, header):
        """
        Edit a FITS header using information in the current map.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        if self.id not in ['', None]:
            header['EXTNAME'] = self.id, 'Content identifier.'
        super().parse_header(header)

    def parse_header(self, header):
        """
        Parse a FITS header and apply to the image.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to parse.

        Returns
        -------
        None
        """
        self.id = header.get('EXTNAME', '')
        if 'BUNIT' in header:
            self.set_unit(header['BUNIT'])
        self.parse_history(header)

    def read_hdu(self, image_hdu):
        """
        Read and apply an HDU.

        Parameters
        ----------
        image_hdu : astropy.io.fits.hdu.image.ImageHDU

        Returns
        -------
        None
        """
        self.parse_header(image_hdu.header)
        self.set_data(image_hdu.data)
        self.data *= self.unit.value

    @staticmethod
    def read_hdul(hdul, hdu_index):
        """
        Read a specific HDU image from an HDU list, and return an image.

        Parameters
        ----------
        hdul : astropy.io.fits.hdu.hdulist.HDUList
        hdu_index : int

        Returns
        -------
        Image2D
        """
        hdu = hdul[hdu_index]
        image = Image2D(dtype=hdu.data.dtype)
        image.read_hdu(hdu)
        return image

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

    def renew(self):
        """
        Renew the image by clearing all data.

        Returns
        -------
        None
        """
        self.clear()
