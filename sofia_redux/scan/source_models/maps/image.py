# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.source_models.maps.fits_data import FitsData
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.source_models.beams.asymmetry_2d import Asymmetry2D

__all__ = ['Image']


class Image(FitsData):

    def __init__(self, data=None, blanking_value=None, dtype=float,
                 shape=None, unit=None):

        self.id = ''
        super().__init__(data=data, blanking_value=blanking_value, dtype=dtype,
                         shape=shape, unit=unit)

    def __eq__(self, other):
        """
        Check whether this

        Parameters
        ----------
        other : Image

        Returns
        -------
        equal : bool
        """
        if other is self:
            return True
        if not isinstance(other, Image):
            return False
        if self.id != other.id:
            return False
        return super().__eq__(other)

    def copy(self, with_contents=True):
        """
        Return a copy of the image.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, paste the contents of this image onto the new one.

        Returns
        -------
        Image
        """
        new = super().copy()
        if with_contents and self.size > 0:
            new.paste(self, report=True)
            return new
        elif not with_contents:
            new.shape = self.shape
        return new

    def new_image(self):
        """
        Return a new image.

        Returns
        -------
        Image
        """
        return self.copy(with_contents=False)

    def set_id(self, image_id):
        """
        Set the image ID.

        Parameters
        ----------
        image_id : str

        Returns
        -------
        None
        """
        self.id = str(image_id)

    def get_id(self):
        """
        Return the image ID.

        Returns
        -------
        image_id : str
        """
        return self.id

    def set_data(self, data, change_type=False):
        """
        Set the image data array.

        Parameters
        ----------
        data : numpy.ndarray or FlaggedArray
            A 2-dimensional image array.
        change_type : bool, optional
            If `True`, change the data type to that of the data.

        Returns
        -------
        None
        """
        super().set_data(data, change_type=change_type)
        self.record_new_data(detail=f'{self.ndim}D {self.dtype}')

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

        image = self.copy(with_contents=True)

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

    def crop(self, ranges):
        """
        Crop the image to the required dimensions.

        Parameters
        ----------
        ranges : numpy.ndarray (int,)
            The ranges to set crop the data to.  Should be of shape
            (n_dimensions, 2) where ranges[0, 0] would give the minimum crop
            limit for the first dimension and ranges[0, 1] would give the
            maximum crop limit for the first dimension.  In this case, the
            'first' dimension is in numpy format.  i.e., (y, x) for a
            2-D array. Also note that the upper crop limit is not inclusive
            so a range of (0, 3) includes indices [0, 1, 2] but not 3.

        Returns
        -------
        None
        """
        if self.data is None:
            return
        if (not isinstance(ranges, np.ndarray)
                or ranges.shape != (self.ndim, 2)):
            raise ValueError(f"The crop range should be of shape "
                             f"({self.ndim}, 2). Received {ranges}")
        self.add_history(f"Cropped {ranges[:, 0]} : {ranges[:, 1]}")
        super().crop(ranges)

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
        if ranges.shape != (self.ndim, 2) or None in ranges:
            return  # invalid ranges
        elif (ranges[:, 0] == 0).all() and np.allclose(
                ranges[:, 1], self.shape[::-1]):  # ::-1 because FITS to numpy
            return  # no change

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
        if self.get_id() not in ['', None]:
            header['EXTNAME'] = self.get_id(), 'Content identifier.'
        super().edit_header(header)

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
        self.set_id(header.get('EXTNAME', ''))
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
        image = Image(dtype=hdu.data.dtype)
        image.read_hdu(hdu)
        return image

    def get_asymmetry_2d(self, grid, center_index, angle, radial_range):
        """
        Return the 2-D asymmetry.

        Parameters
        ----------
        grid : Grid2D
        center_index : Coordinate2D
        angle : units.Quantity
        radial_range : Range

        Returns
        -------
        asymmetry : Asymmetry2D
        """
        right_angle = 90 * units.Unit('degree')
        asymmetry_x, x_rms = self.get_asymmetry(
            grid, center_index, angle, radial_range)
        asymmetry_y, y_rms = self.get_asymmetry(
            grid, center_index, angle + right_angle, radial_range)

        x_weight = np.inf if x_rms == 0 else (1 / x_rms) ** 2
        y_weight = np.inf if y_rms == 0 else (1 / y_rms) ** 2

        return Asymmetry2D(x=asymmetry_x, y=asymmetry_y,
                           x_weight=x_weight, y_weight=y_weight)

    def get_asymmetry(self, grid, center_index, angle, radial_range):
        """
        Return the Asymmetry.

        Parameters
        ----------
        grid : Grid2D
        center_index : Coordinate2D
        angle : units.Quantity
        radial_range : Range

        Returns
        -------
        asymmetry, asymmetry_rms
        """
        center_offset = grid.index_to_offset(center_index)
        j, i = np.nonzero(self.valid)
        data = self.data[j, i]
        indices = Coordinate2D([i, j])
        grid_offsets = grid.index_to_offset(indices)
        grid_offsets.subtract(center_offset)
        r = grid_offsets.length

        in_range = r <= radial_range.max
        m0 = np.nansum(np.abs(data[in_range]))

        in_range &= r >= radial_range.min

        d = data[in_range]
        c = np.cos(grid_offsets[in_range].angle() - angle)

        mc = np.sum(d * c)
        c2 = np.sum(c * c)

        if m0 > 0:
            value = mc / m0
            rms = np.sqrt(c2) / m0
        else:
            value = 0.0
            rms = 0.0

        if isinstance(value, units.Quantity):
            value = value.value
            rms = rms.value

        return value, rms
