# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.source_models.maps.map_2d1 import Map2D1
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D
from sofia_redux.scan.source_models.maps.image_2d1 import Image2D1
from sofia_redux.scan.flags.flagged_array import FlaggedArray

__all__ = ['Observation2D1']


class Observation2D1(Map2D1, Observation2D):

    def __init__(self, data=None, blanking_value=np.nan, dtype=float,
                 shape=None, unit=None, weight_dtype=float,
                 weight_blanking_value=None):
        """
        Initialize an Observation2D1 object.

        The 2-D + 1-D observation is an extension of the :class:`Map2D1` class
        that includes weights and exposure times in addition to the
        observation data values.  It also includes these for an additional
        spectral dimension along the z-axis.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to initialize the flagged array with.  If supplied, sets the
            shape of the array.  Note that the data type will be set to that
            defined by the `dtype` parameter.
        blanking_value : int or float, optional
            The blanking value defines invalid values in the data array.  This
            is the equivalent of defining a NaN value.
        dtype : type, optional
            The data type of the data array.
        shape : tuple (int), optional
            The shape of the data array.  This will only be relevant if
            `data` is not defined.
        unit : str or units.Unit or units.Quantity, optional
            The data unit.
        weight_dtype : type, optional
            Similar the `dtype`, except defines the data type for the
            observation weights and exposure times.
        weight_blanking_value : int or float, optional
            The blanking value for the weight and exposure maps.  If `None`,
            will be set to np.nan if `weight_dtype` is a float, and 0 if
            `weight_dtype` is an integer.
        """
        super().__init__(data=data, blanking_value=blanking_value, dtype=dtype,
                         shape=shape, unit=unit)
        self.weight = Image2D1(dtype=weight_dtype,
                               blanking_value=weight_blanking_value,)
        self.exposure = Image2D1(dtype=weight_dtype,
                                 blanking_value=weight_blanking_value)
        shape = self.shape
        if shape != ():
            self.weight.shape = shape
            self.exposure.shape = shape

    def copy(self, with_contents=True):
        """
        Return a copy of the map.

        Returns
        -------
        Observation2D1
        """
        return super().copy(with_contents=with_contents)

    def to_weight_image(self, data):
        """
        Convert data to a weight image.

        Parameters
        ----------
        data : FlaggedArray or FitsData or numpy.ndarray or None

        Returns
        -------
        Image2D1
        """
        if data is None:
            data = Image2D1(x_size=self.shape[2],
                            y_size=self.shape[1],
                            z_size=self.shape[0],
                            blanking_value=self.weight.blanking_value,
                            dtype=self.weight_dtype)
        elif isinstance(data, np.ndarray):
            data = Image2D1(data=data,
                            blanking_value=self.weight.blanking_value,
                            dtype=self.weight_dtype)
        return data

    def crop(self, ranges):
        """
        Crop the image data.

        Parameters
        ----------
        ranges : numpy.ndarray (int,) or Coordinate2D1
            The ranges to set crop the data to.  Should be of shape
            (n_dimensions, 2) where ranges[0, 0] would give the minimum crop
            limit for the first dimension and ranges[0, 1] would give the
            maximum crop limit for the first dimension.  In this case, the
            'first' dimension is in FITS format.  i.e., (x, y) for a 2-D image.
            If a Quantity is supplied this should contain the min and max
            grid values to clip to in each dimension.

        Returns
        -------
        None
        """
        super().crop(ranges)

    def filter_correct(self, underlying_fwhm, reference=None, valid=None):
        """
        Apply filter correction.

        Parameters
        ----------
        underlying_fwhm : Coordinate2D1
        reference : FlaggedArray or numpy.ndarray, optional
        valid : numpy.ndarray (bool), optional

        Returns
        -------
        None
        """
        super().filter_correct(underlying_fwhm, reference=reference,
                               valid=valid)

    def fft_filter_above(self, fwhm, valid=None, weight=None):
        """
        Apply FFT filtering above a given FWHM.

        Parameters
        ----------
        fwhm : Coordinate2D1
        valid : numpy.ndarray (bool), optional
        weight : FlaggedArray or numpy.ndarray (float)

        Returns
        -------
        None
        """
        super().fft_filter_above(fwhm, valid=valid, weight=weight)
