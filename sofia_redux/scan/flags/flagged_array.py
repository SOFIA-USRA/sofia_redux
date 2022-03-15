# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy import signal

from sofia_redux.scan.flags.array_flags import ArrayFlags
from sofia_redux.scan.flags.flagged_data import FlaggedData
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.flags import flag_numba_functions as fnf

from sofia_redux.toolkit.splines.spline import Spline

__all__ = ['FlaggedArray']


class FlaggedArray(FlaggedData):

    flagspace = ArrayFlags

    def __init__(self, data=None, blanking_value=None, dtype=None,
                 shape=None):
        """
        Creates a flagged array instance.

        The flagged array provides a wrapper around a numpy array allowing
        for each element to have an associated flag value.  There are also
        a number of arithmetic operators provided that automatically discount
        certain flags and invalid values from any processing.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to initialize the flagged array with.  If supplied, sets the
            shape of the array.  Note that the data type will be set to that
            defined by the `dtype` parameter.
        blanking_value : int or float or bool or complex, optional
            The blanking value defines invalid values in the data array.  This
            is the equivalent of defining a NaN value.
        dtype : type, optional
            The data type of the data array.
        shape : tuple (int), optional
            The shape of the data array.  This will only be relevant if
            `data` is not defined.
        """
        super().__init__()
        # These are all arrays of equal size
        self._data = None
        self._flag = None
        self.dtype = dtype
        self.validating_flags = None
        self._blanking_value = None

        if data is not None:
            self.set_data(data)  # May set dtype and other stuff
        elif shape is not None:
            self.set_data_shape(shape)

        if self.dtype is None:
            if blanking_value is not None:
                self.dtype = type(blanking_value)
            else:
                self.dtype = float

        if self.blanking_value is None:
            if blanking_value is None:
                self.blanking_value = self.default_blanking_value()
            elif isinstance(self.dtype, np.dtype):
                self.blanking_value = self.dtype.type(blanking_value)
            else:
                self.blanking_value = self.dtype(blanking_value)

    def copy(self):
        """
        Return a copy of the FlaggedArray.

        Returns
        -------
        FlaggedArray
        """
        return super().copy()

    def __eq__(self, other):
        """
        Check whether this flagged array is equal to another.

        Parameters
        ----------
        other : FlaggedArray

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, FlaggedArray):
            return False
        if self.validating_flags != other.validating_flags:
            return False

        if (self.blanking_value is not None
                and other.blanking_value is not None):
            try:
                not_equal_nan = np.isnan(self.blanking_value) is not np.isnan(
                    other.blanking_value)
                if not_equal_nan:
                    return False
            except (TypeError, ValueError):  # pragma: no cover
                pass

        elif self.blanking_value != other.blanking_value:
            return False

        return self.check_equal_contents(other)

    def check_equal_contents(self, other):
        """
        Check the data contents of this data are equal with another.

        Parameters
        ----------
        other : FitsData

        Returns
        -------
        equal : bool
        """
        if self.shape != other.shape:
            return False

        valid, other_valid = self.valid, other.valid
        if valid is not None or other_valid is not None:
            if valid is None or other_valid is None:
                return False
            elif not np.allclose(valid, other_valid):
                return False

        data, other_data = self.data, other.data
        if data is not None or other_data is not None:
            if data is None or other_data is None:
                return False
            elif not np.allclose(data, other_data, equal_nan=True):
                return False

        flag, other_flag = self.flag, other.flag
        if flag is not None or other_flag is not None:
            if flag is None or other_flag is None:
                return False
            elif not np.allclose(flag, other_flag, equal_nan=True):
                return False
        return True

    @property
    def nan_blanking(self):
        """
        Return whether the blanking value is NaN.

        Returns
        -------
        bool
        """
        try:
            return np.isnan(self.blanking_value)
        except (TypeError, ValueError):
            return False

    @property
    def data(self):
        """
        Return the data array.

        Returns
        -------
        data : np.ndarray or None
        """
        return self._data

    @data.setter
    def data(self, values):
        """
        Set the data array.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        None
        """
        self.set_data(values)

    @property
    def shape(self):
        """
        Return the shape of the data image array.

        Returns
        -------
        tuple (int)
        """
        data = self.data
        if data is None:
            return ()
        return data.shape

    @shape.setter
    def shape(self, new_shape):
        """
        Set the data shape.

        Parameters
        ----------
        new_shape : tuple (int)

        Returns
        -------
        None
        """
        self.set_data_shape(new_shape)

    @property
    def blanking_value(self):
        """
        Return the blanking value for the array.

        The blanking value is that which defines an invalid value.

        Returns
        -------
        blanking_value : int or float or None
        """
        return self._blanking_value

    @blanking_value.setter
    def blanking_value(self, value):
        """
        Set the blanking value for the array.

        Parameters
        ----------
        value : int or float or None

        Returns
        -------
        None
        """
        self.set_blanking_level(value)

    @property
    def size(self):
        """
        Return the size of the data image array.

        Returns
        -------
        int
        """
        data = self.data
        if data is None:
            return 0
        return data.size

    @property
    def ndim(self):
        """
        Return the number of dimensions in the map image data.

        Returns
        -------
        int
        """
        data = self.data
        if data is None:
            return 0
        return data.ndim

    @property
    def valid(self):
        """
        Return a boolean mask array of valid data elements.

        Valid elements are neither NaN, set to the blanking value, or
        flagged as the validating_flags.

        Returns
        -------
        numpy.ndarray (bool)
           A boolean mask where `True` indicates a valid element.
        """
        return self.is_valid()

    def is_valid(self):
        """
        Return a boolean mask array of valid data elements.

        Valid elements are neither NaN, set to the blanking value, or
        flagged as the validating_flags.

        Returns
        -------
        numpy.ndarray (bool)
           A boolean mask where `True` indicates a valid element.
        """
        if self.size == 0:
            return np.full(self.shape, False)

        mask = np.logical_not(np.isnan(self.data))
        if self.blanking_value is not None and not self.nan_blanking:
            mask &= self.data != self.blanking_value

        if self.validating_flags is None:
            return mask

        elif self.validating_flags.value == 0:
            return mask & self.is_unflagged()

        else:
            return mask & self.is_unflagged(self.validating_flags)

    @property
    def valid_data(self):
        """
        Return valid data.

        Returns
        -------
        valid_data : numpy.ndarray (self.dtype)
            The valid data of shape (n_valid,).
        """
        if self.data is None:
            return np.empty(0, dtype=self.dtype)
        return np.array(self.data[self.valid])

    @property
    def data_range(self):
        """
        Get the data range.

        Returns
        -------
        Range
        """
        if self.data is None:
            return Range(min_val=np.inf, max_val=-np.inf)

        valid_data = self.valid_data
        if valid_data.size == 0:
            return Range(min_val=np.inf, max_val=-np.inf)

        return Range(min_val=np.min(valid_data), max_val=np.max(valid_data))

    def default_blanking_value(self):
        """
        Return a default blanking value type for the current data type.

        Returns
        -------
        value : float or int or bool or complex
            The default blanking value.
        """
        dtype = self.dtype
        if dtype is None:
            dtype = self.dtype = float
        if dtype == float or np.issubdtype(self.dtype, float):
            return np.nan
        elif dtype == int or np.issubdtype(dtype, int):
            return -9999
        elif dtype == bool or np.issubdtype(dtype, bool):
            return False
        elif dtype == complex or np.issubdtype(dtype, complex):
            return complex(np.nan, np.nan)
        else:
            raise ValueError(f"Invalid dtype: {dtype}.")

    def set_data(self, data, change_type=False):
        """
        Set the data of the flagged array.

        All flags are set to zero.

        Parameters
        ----------
        data : numpy.ndarray or FlaggedArray
        change_type : bool, optional
            If `True`, change the data type to that of the data.

        Returns
        -------
        None
        """
        if not isinstance(data, (FlaggedArray, np.ndarray)):
            raise ValueError(f"Must supply data as a {FlaggedArray} "
                             f"or {np.ndarray}.")

        if change_type or self.dtype is None:
            self.dtype = data.dtype
        self.shape = data.shape  # this resets data and flag
        self.set_flags('DEFAULT')

        if isinstance(data, FlaggedArray):
            self.data[...] = data.data
            self.discard(np.logical_not(data.valid))
            self.unflag('DEFAULT', indices=data.valid)
        else:  # numpy.ndarray
            self.data[...] = data
            self.unflag('DEFAULT')

    def set_data_shape(self, shape):
        """
        Set the shape of the data array.

        Parameters
        ----------
        shape : tuple (int)

        Returns
        -------
        None
        """
        self._data = np.zeros(shape, dtype=self.dtype)
        self.dtype = self._data.dtype
        self.flag = np.zeros(shape, dtype=int)
        self.fixed_index = np.arange(self.size).reshape(self.shape)
        if self.size > 0:
            self.set_flags('DEFAULT')

    def set_blanking_level(self, value):
        """
        Set the blanking level for the map data.

        The blanking level is the value for all null values.  If a new blanking
        level is set, and the data exists, all previously set blanked values
        will be set to the new level.

        Parameters
        ----------
        value : float or int or None

        Returns
        -------
        None
        """
        if value is None:
            self._blanking_value = None
            return

        try:
            both_nan = np.isnan(value) & np.isnan(self.blanking_value)
            if both_nan:
                return
        except (ValueError, TypeError):
            pass

        if value == self.blanking_value:
            return

        if self.data is not None and self.blanking_value is not None:
            fnf.set_new_blank_value(self.data, self.blanking_value, value)

        self._blanking_value = value

    def get_size_string(self):
        """
        Return the shape of the data array as a string.

        Returns
        -------
        str
        """
        if self.data is None:
            return '0'
        return 'x'.join([str(x) for x in self.shape])

    def discard_flag(self, flag, criterion=None):
        r"""
        Clear all data flagged with the given flag.

        All data matching the given flag and criteria will be cleared according
        to :func:`FlaggedArray.clear` method.  This generally should result in
        all matching elements having their flag and data values set to zero.

        Parameters
        ----------
        flag : int or str or ChannelFlagTypes
            The flag to discard_flag.
        criterion : str, optional
            One of {'DISCARD_ANY', 'DISCARD_ALL', 'DISCARD_MATCH',
            'KEEP_ANY', 'KEEP_ALL', 'KEEP_MATCH'}.  \*_ANY refers to any flag
            that is not zero (unflagged).  \*_ALL refers to any flag that
            contains `flag`, and \*_MATCH refers to any flag that exactly
            matches `flag`.  The default (`None`), uses DISCARD_ANY if
            `flag` is None, and DISCARD_ALL otherwise.

        Returns
        -------
        None
        """
        self.discard(self.flagspace.discard_indices(
            self.flag, flag, criterion=criterion))

    def discard(self, indices=None):
        """
        Set the flags for discarded indices to DISCARD and data to zero.

        Parameters
        ----------
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).
        Returns
        -------
        None
        """
        if self.data is None:
            return
        if indices is None:
            self.data.fill(self.blanking_value)
        else:
            self.clear(indices)
            # self.data[indices] = self.blanking_value
        self.set_flags('DISCARD', indices=indices)

    def clear(self, indices=None):
        """
        Clear flags and set data to zero.

        Parameters
        ----------
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        if self.data is None:
            return
        if indices is None:
            self.data.fill(0)
        else:
            data = self.data
            data[indices] = 0
            self.data[...] = data
        self.unflag(indices=indices)

    def fill(self, value, indices=None):
        """
        Fill the data array with a given value.

        Parameters
        ----------
        value : int or float
            The value to fill.
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        if self.data is None:
            return
        if indices is None:
            self.data.fill(value)
        else:
            data = self.data
            data[indices] = value
            self.data[...] = data
        self.unflag(indices=indices)

    def add(self, values, indices=None, factor=None):
        """
        Add a value to the data array.

        Parameters
        ----------
        values : int or float or FlaggedArray
            The value to add.
        indices : numpy.ndarray (int or bool), optional
            The indices on self.data for which to add `value`.  If a boolean
            mask is supplied, this assumes `value` is the same shape as
            self.data and indicates valid elements to add.
        factor : int or float, optional
            An optional factor to scale the data by.

        Returns
        -------
        None
        """
        if self.data is None or values is None:
            return

        if isinstance(values, FlaggedArray):
            add_values = values.data
            valid_indices = values.valid
        elif isinstance(values, np.ndarray) and values.shape != ():
            add_values = values
            valid_indices = None
        else:  # single value
            if indices is None:
                add_values = np.full(self.shape, values)
            elif indices.dtype == bool:
                # indices are a mask array
                add_values = np.full(indices.shape, values)
            else:
                # Indices are of shape (n_dimensions,...)
                add_values = np.full(indices.shape[1:], values)
            valid_indices = None

        if add_values is self.data:  # just in case
            add_values = add_values.copy()

        if indices is None:
            indices = np.indices(self.shape)
        elif indices.dtype == bool:
            # Get indices in (n_dimensions, index) integer format.
            keep = np.nonzero(indices)
            indices = np.stack([x for x in keep])
            if valid_indices is not None:
                valid_indices = valid_indices[keep]
            # added values are in the form of a regular array e.g., (ny, nx)
            # but need to be flattened with invalid points removed.
            add_values = add_values[keep]
        else:
            indices = np.asarray(indices)

        if factor is not None:
            add_values = add_values * factor

        data = self.data

        added = numba_functions.sequential_array_add(
            data, add_values, indices, valid_indices=valid_indices)

        self.data[...] = data
        self.unflag(self.flagspace.flags.DEFAULT, indices=added)

    def subtract(self, value, indices=None, factor=None):
        """
        Subtract a value or FlaggedData from the data array.

        Parameters
        ----------
        value : int or float or FlaggedArray
            The value to add.
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).
        factor : int or float, optional
            An optional factor to scale the data by.

        Returns
        -------
        None
        """
        if factor is None:
            factor = 1
        self.add(value, indices=indices, factor=-factor)

    def scale(self, factor, indices=None):
        """
        Scale the data by a given factor.

        Parameters
        ----------
        factor : int or float
            The factor to scale by.
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        if self.data is None:
            return

        data = self.data
        if indices is None:
            data *= factor
        else:
            data[indices] *= factor
        self.data[...] = data

    def destroy(self):
        """
        Destroy the image data.

        Returns
        -------
        None
        """
        self.shape = (0,) * self.ndim

    def validate(self, validator=None):
        """
        Discard all invalid data.

        Parameters
        ----------
        validator : Validator, optional
            An optional object or function that can take FlaggedArray as an
            argument and perform the validation.

        Returns
        -------
        None
        """
        if validator is None:
            self.discard(~self.valid)
        else:
            validator(self)

    def find_fixed_indices(self, fixed_indices, cull=True):
        """
        Returns the actual indices given fixed indices.

        The fixed indices are those that are initially loaded.  Returned
        indices are their locations in the data arrays.

        Parameters
        ----------
        fixed_indices : int or np.ndarray (int)
            The fixed indices.
        cull : bool, optional
            If `True`, do not include fixed indices not found in the result.
            If `False`, missing indices will be replaced by -1.

        Returns
        -------
        indices : numpy.ndarray (int) or tuple (numpy.ndarray (int))
            The indices of `fixed_indices` in the data arrays.  A tuple will
            be returned, in the case where we are examining more than one
            dimension.
        """
        if self.ndim == 1:
            return super().find_fixed_indices(fixed_indices, cull=cull)

        values = np.asarray(fixed_indices, dtype=int)
        singular = values.ndim == 0
        if singular:
            indices = np.nonzero(self.fixed_index == values)
            if not cull and indices[0].size == 0:
                return (-1,) * self.ndim
            elif indices[0].size == 1:
                return tuple([int(index[0]) for index in indices])
            else:
                return indices

        mask = values[:, None] == self.fixed_index.ravel()
        value_indices, flat_indices = np.nonzero(mask)
        indices = np.unravel_index(flat_indices, self.shape)

        if cull or value_indices.size == values.size:
            # Nothing to cull
            return indices

        found = tuple([np.full(values.size, -1) for _ in range(self.ndim)])
        for dimension in range(self.ndim):
            found[dimension][value_indices] = indices[dimension]
        return found

    def count_flags(self, flag=None):
        """
        Count the number of flagged elements.

        Parameters
        ----------
        flag : str or int or enum.Enum
            The flag to count.

        Returns
        -------
        int
        """
        return np.sum(self.is_flagged(flag))

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : list or int or numpy.ndarray (bool or int)
            The indices to extract.

        Returns
        -------
        FlaggedData
        """
        raise NotImplementedError("Not available for a Flagged Array.")

    def delete_indices(self, indices_or_mask):
        """
        Completely deletes data elements.

        Actual indices should be passed in.  To delete based on fixed index
        values, please convert first using `find_fixed_indices`.

        Parameters
        ----------
        indices_or_mask : int or list or numpy.ndarray of (bool or int)
            The indices to delete, or a boolean mask where `True` marks an
            element for deletion.

        Returns
        -------
        None
        """
        raise NotImplementedError("Cannot delete indices for a shaped array.")

    def insert_blanks(self, insert_indices):
        """
        Inserts blank frame data.

        Actual indices should be passed in.  To delete based on fixed index
        values, please convert first using `find_fixed_indices`.

        Blank data are set to 0 in whatever unit is applicable.

        Parameters
        ----------
        insert_indices : int or list or numpy.ndarray of (bool or int)
            The index locations to insert.

        Returns
        -------
        None
        """
        raise NotImplementedError("Cannot insert blanks in a shaped array.")

    def merge(self, data):
        """
        Add additional data onto the end of this data.

        Parameters
        ----------
        data : FlaggedData

        Returns
        -------
        None
        """
        raise NotImplementedError("Cannot merge a shaped array.")

    def discard_range(self, discard_range):
        """
        Discard data values in a certain range.

        Parameters
        ----------
        discard_range : Range

        Returns
        -------
        None
        """
        if self.data is None:
            return
        discard = discard_range.in_range(self.data)
        self.discard(discard)

    def restrict_range(self, keep_range):
        """
        Discard data values outside of a certain range.

        Parameters
        ----------
        keep_range : Range

        Returns
        -------
        None
        """
        if self.data is None:
            return
        discard = ~keep_range.in_range(self.data)
        self.discard(discard)

    def min(self):
        """
        Return the minimum value of the valid data.

        Returns
        -------
        float or int
        """
        if self.data is None:
            return self.blanking_value
        valid_data = self.valid_data
        if valid_data.size == 0:
            return self.blanking_value
        return np.min(valid_data)

    def argmin(self):
        """
        Return the index of the minimum value.

        Returns
        -------
        int or tuple (int)
        """
        if self.data is None:
            return ()
        indices = np.nonzero(self.data == self.min())
        if indices[0].size == 0:
            return ()
        if self.ndim == 1:
            return indices[0][0]
        return tuple([x[0] for x in indices])

    def max(self):
        """
        Return the maximum value of the valid data.

        Returns
        -------
        float or int
        """
        if self.data is None:
            return self.blanking_value
        valid_data = self.valid_data
        if valid_data.size == 0:
            return self.blanking_value
        return np.max(valid_data)

    def argmax(self):
        """
        Return the index of the maximum value.

        Returns
        -------
        int or tuple (int)
        """
        if self.data is None:
            return ()
        indices = np.nonzero(self.data == self.max())
        if indices[0].size == 0:
            return ()
        if self.ndim == 1:
            return indices[0][0]
        return tuple([x[0] for x in indices])

    def argmaxdev(self):
        """
        Return the index of maximum (absolute) deviation from zero.

        Returns
        -------
        int or tuple (int)
        """
        if self.data is None:
            return ()
        min_val = self.min()
        if self.nan_blanking and np.isnan(min_val):
            return ()
        elif min_val == self.blanking_value:
            return ()

        max_val = self.max()
        if abs(min_val) > abs(max_val):
            return self.argmin()
        else:
            return self.argmax()

    def mean(self, weights=None):
        """
        Return the mean of the data.

        Parameters
        ----------
        weights : numpy.ndarray (float), optional
            If provided, perform a weighted mean.  Must be an array of shape
            (self.shape).  If not provided, the weight assigned to each datum
            is 1.

        Returns
        -------
        mean, mean_weight : float, float
        """
        if self.data is None:
            return np.nan
        if weights is not None:
            valid = self.valid
            weights = weights[valid]
            data = self.data[valid]
        else:
            data = self.valid_data

        mean_value, mean_weight = numba_functions.mean(data, weights=weights)
        return mean_value, mean_weight

    def median(self, weights=None):
        """
        Return the median of the data.

        Parameters
        ----------
        weights : numpy.ndarray (float), optional
            If provided, perform a weighted mean.  Must be an array of shape
            (self.shape).

        Returns
        -------
        median, median_weight : float, float
        """
        if self.data is None:
            return np.nan
        if weights is not None:
            valid = self.valid
            weights = weights[valid]
            data = self.data[valid]
        else:
            data = self.valid_data

        if weights is None:
            return np.median(data), float(data.size)
        else:
            return numba_functions.smart_median(
                data, weights=weights, max_dependence=1.0)

    def select(self, fraction):
        """
        Return the value representative of the fraction of the sorted data.

        Parameters
        ----------
        fraction : float
            The fraction of the sorted data array to return.  Must be between
            0 and 1.

        Returns
        -------
        value : int or float
        """
        if self.data is None:
            return self.blanking_value
        if fraction == 0:
            return self.min()
        elif fraction == 1.0:
            return self.max()
        elif not 0 < fraction < 1:
            raise ValueError("Fraction must be between 0 and 1.")

        values = self.valid_data
        if values.size == 0:
            return self.blanking_value
        values.sort()
        index = np.clip(int(np.round(fraction * (values.size - 1))),
                        0, values.size - 1)
        return values[index]

    def level(self, robust=True, weights=None):
        """
        Remove the mean or median value from the data.

        Parameters
        ----------
        robust : bool, optional
            If `True`, remove the median value.  Otherwise, remove the
            mean value.
        weights : numpy.ndarray (float), optional
            If provided, perform a weighted mean.  Must be an array of shape
            (self.shape).

        Returns
        -------
        None
        """
        if self.data is None:
            return
        if robust:
            level, level_weight = self.median(weights=weights)
        else:
            level, level_weight = self.mean(weights=weights)

        if np.isfinite(level):
            self.subtract(level, indices=self.valid)

    def variance(self, robust=True):
        """
        Return the variance of the data.

        The variance is given as::

           var = func(d[valid]^2)

        where d is the data and func is median(x)/0.454937 if `robust`
        is `True` and mean(x) otherwise.

        Parameters
        ----------
        robust : bool, optional
            If `True`, use the robust (median) method to determine the
            variance.  Otherwise, use the mean.

        Returns
        -------
        float
        """
        if self.data is None:
            return np.nan
        values = self.valid_data
        if values.size == 0:
            return np.nan
        values *= values
        if robust:
            return np.median(values) / 0.454937
        else:
            return np.mean(values)

    def rms(self, robust=True):
        """
        Return the RMS of the data.

        Parameters
        ----------
        robust : bool, optional
            If `True`, use the robust (median) method to determine the RMS.
            Otherwise, use the mean.

        Returns
        -------
        float
        """
        return np.sqrt(self.variance(robust=robust))

    def sum(self):
        """
        Return the sum of the data.

        Returns
        -------
        float
        """
        if self.data is None:
            return np.nan
        values = self.valid_data
        if values.size == 0:
            return np.nan
        return np.sum(values)

    def abs_sum(self):
        """
        Return the sum of the absolute data values.

        Returns
        -------
        float
        """
        if self.data is None:
            return np.nan
        values = self.valid_data
        if values.size == 0:
            return np.nan
        return np.sum(np.abs(values))

    def mem_correct(self, model, noise, lg_multiplier):
        """
        Apply a maximum entropy correction given a model.

        Parameters
        ----------
        model : numpy.ndarray or None
            The model from which to base MEM correction.  Should be of shape
            (self.shape).
        noise : numpy.ndarray
            The associated noise values.  Should be of shape (self.shape).
        lg_multiplier : float
            The Lagrange multiplier (lambda) for the MEM correction.

        Returns
        -------
        None
        """
        if self.data is None:
            return

        valid = self.valid
        mem_correction = fnf.get_mem_correction(
            data=self.data,
            noise=noise,
            multiplier=lg_multiplier,
            valid=self.valid,
            model=model)
        self.subtract(mem_correction, indices=valid)

    def paste(self, source):
        """
        Paste data from another FitsData onto the data array.

        Parameters
        ----------
        source : FlaggedArray

        Returns
        -------
        None
        """
        if source is self:
            return
        if source.data is None:
            return
        elif self.data is None or self.data.size == 0:
            self.data = source.data

        valid = self.valid & source.valid
        data = self.data
        data[valid] = source.data[valid]
        self.data = data
        if not valid.all():
            invalid = np.logical_not(valid)
            self.discard(invalid)

    def count_valid_points(self):
        """
        Return the number of valid data points in the array.

        Returns
        -------
        int
        """
        if self.valid is None:  # pragma: no cover
            return 0
        return np.sum(self.valid)

    def get_index_range(self):
        """
        Return the index ranges of valid points.

        Returns
        -------
        ranges : numpy.ndarray (int)
            A range for each dimension or shape (n_dimensions, 2) giving the
            minimum and maximum range in each dimension.  Note that this is
            numpy dimension ordering (y-range = ranges[0],
            x-range = ranges[1]). Also note that the upper range is returned
            such that the real upper index is included in any slice
            operation. i.e., max = real max index + 1.
        """
        if self.data is None:
            return np.full((self.ndim, 2), -1, dtype=int)

        valid_indices = np.nonzero(self.valid)
        ranges = np.full((self.ndim, 2), -1, dtype=int)
        for dimension, indices in enumerate(valid_indices):
            if indices.size != 0:
                ranges[dimension] = indices.min(), indices.max() + 1
        return ranges

    def fast_smooth(self, beam_map, steps, reference_index=None, weights=None):
        """
        Smooth using the fast method.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The kernel to convolve with.
        steps : numpy.ndarray (int)
            The size of the steps in each dimension.
        reference_index : numpy.ndarray (float)
            The reference index of the beam map center.  The default is
            (beam_map.shape - 1) / 2.0.
        weights : numpy.ndarray (float)
            Weights the same shape as beam map.

        Returns
        -------
        None
        """
        convolved = self.get_fast_smoothed(
            beam_map, steps, reference_index=reference_index, weights=weights,
            get_weights=False)
        convolved[~self.valid] = np.nan
        new = self.__class__(data=convolved)
        self.paste(new)

    def get_valid_smoothed(self, beam_map, reference_index=None,
                           weights=None, get_weights=False):
        """
        Return smoothed data and weights, where invalid entries are zeroed.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The kernel to convolve with.
        reference_index : numpy.ndarray (float)
            The reference index of the beam map center.  The default is
            (beam_map.shape - 1) / 2.0.
        weights : numpy.ndarray (float)
            Weights the same shape as beam map.
        get_weights : bool, optional
            If `True`, calculate the smoothed weights in addition to the
            smoothed data.

        Returns
        -------
        smoothed_data, [smoothed_weights] : numpy.ndarray, [numpy.ndarray]
            The smoothed data and weights.  Will only return smoothed data if
            `get_weights` is `False`.
        """
        if reference_index is None:
            reference_index = (np.asarray(beam_map.shape) - 1) / 2.0
            reference_index = reference_index.astype(int)

        smoothed, smoothed_weights = self.get_smoothed(
            beam_map, reference_index=reference_index, weights=weights)

        invalid = ~self.valid
        invalid |= ~np.isfinite(smoothed)
        smoothed[invalid] = self.data[invalid]

        if get_weights:
            smoothed_weights[invalid] = 0.0
            smoothed_weights[np.isnan(smoothed_weights)] = 0.0
            return smoothed, smoothed_weights
        else:
            return smoothed

    def get_fast_smoothed(self, beam_map, steps, reference_index=None,
                          weights=None, get_weights=False):
        """
        Return smoothed values using the fast method.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The kernel to convolve with.
        steps : numpy.ndarray (int)
            The size of the steps in each dimension.
        reference_index : numpy.ndarray (float)
            The reference index of the beam map center.  The default is
            (beam_map.shape - 1) / 2.0.
        weights : numpy.ndarray (float)
            Weights the same shape as beam map.
        get_weights : bool, optional
            If `True`, calculate the smoothed weights in addition to the
            smoothed data.

        Returns
        -------
        smoothed_data, [smoothed_weights] : numpy.ndarray, [numpy.ndarray]
            The smoothed data and weights.  Will only return smoothed data if
            `get_weights` is `False`.
        """
        if reference_index is None:
            reference_index = (np.asarray(beam_map.shape) - 1) / 2.0
            reference_index = reference_index.astype(int)

        if np.prod(steps) <= 1:
            return self.get_valid_smoothed(
                beam_map, reference_index=reference_index, weights=weights,
                get_weights=get_weights)

        # Perform the convolution on a coarse grid.
        course_signal, course_weight, ratio = (
            numba_functions.regular_coarse_kernel_convolve(
                self.data, beam_map, steps,
                kernel_reference_index=reference_index,
                weight=weights, valid=self.valid))

        course_signal = course_signal.reshape(ratio)
        course_weight = course_weight.reshape(ratio)
        course_weight[course_weight <= 0] = 0.0
        course_weight[np.isnan(course_signal)] = 0.0

        try:
            spline = Spline(course_signal, exact=True, weights=course_weight,
                            reduce_degrees=True)
        except ValueError:
            return self.get_valid_smoothed(
                beam_map, reference_index=reference_index, weights=weights,
                get_weights=get_weights)

        # Interpolate onto the original grid.
        args = []
        for i in range(self.ndim):
            x = np.arange(self.shape[i]) / steps[i]
            args.append(x)
        # Reverse for (x, y) order into spline
        args = args[::-1]

        convolved = spline(*args)
        convolved[~self.valid] = self.data[~self.valid]

        if not get_weights:
            return convolved

        try:
            spline = Spline(course_weight, exact=True, weights=None,
                            reduce_degrees=True)
        except ValueError:
            return self.get_valid_smoothed(
                beam_map, reference_index=reference_index, weights=weights,
                get_weights=get_weights)

        convolved_weights = spline(*args)

        convolved_weights[~self.valid] = 0.0
        convolved_weights[convolved_weights < 0] = 0.0
        convolved_weights[np.isnan(convolved)] = 0.0
        convolved_weights[np.isnan(convolved_weights)] = 0.0

        return convolved, convolved_weights

    def smooth(self, beam_map, reference_index=None, weights=None):
        """
        Smooth the data gy a given kernel.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
        reference_index : numpy.ndarray (float or int), optional
        weights : numpy.ndarray (float), optional

        Returns
        -------
        None
        """
        smoothed, smoothed_weight = self.get_smoothed(
            beam_map, reference_index=reference_index, weights=weights)
        new = self.__class__(data=smoothed)
        self.paste(new)

    def get_smoothed(self, beam_map, reference_index=None, weights=None):
        """
        Return smoothed data and weights.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
        reference_index : numpy.ndarray (float or int), optional
        weights : numpy.ndarray (float), optional

        Returns
        -------
        smoothed_data, smoothed_weights : numpy.ndarray, numpy.ndarray
        """
        return numba_functions.regular_kernel_convolve(
            self.data, beam_map,
            weight=weights,
            kernel_reference_index=reference_index,
            valid=self.valid)

    def resample_from(self, image, to_indices, kernel=None,
                      kernel_reference_index=None, weights=None):
        """
        Resample an image onto given indices of this map.

        Parameters
        ----------
        image : FlaggedArray or numpy.ndarray (float)
            The image to resample.
        to_indices : numpy.ndarray (float or int)
            An array of shape (n_dimensions, self.shape or self.size)
            specifying which image pixels belong to the resampled map.
            I.e., if this were a 2-D array and the pixel at (x, y) = (2, 2)
            corresponds to the image at pixel (3.3, 4.4) then
            to_indices[:, 2, 2] = [4.4, 3.3] (reversed because numpy).
        kernel : numpy.ndarray (float), optional
            The kernel used to perform the resampling.  If supplied, the result
            will be smoothed accordingly.
        kernel_reference_index : numpy.ndarray (int), optional
            If a kernel is supplied, specifies the center pixel of the kernel
            to be used during kernel convolution.  The default is
            (kernel.shape - 1) / 2.
        weights : numpy.ndarray (int or float), optional
            An optional weighting array used for kernel convolution.

        Returns
        -------
        None
        """
        if kernel is not None:
            self.kernel_resample_from(
                image, kernel, to_indices,
                kernel_reference_index=kernel_reference_index,
                weights=weights)
        else:
            self.direct_resample_from(image, to_indices)

    def direct_resample_from(self, image, to_indices):
        """
        Resample an image onto the current flagged array.

        Parameters
        ----------
        image : FlaggedArray or numpy.ndarray (float)
        to_indices : numpy.ndarray (float or int)
            An array of shape (n_dimensions, self.shape).  Indices should be
            supplied using numpy ordering (z, y, x,...), not (x, y, z...) FITS
            style.

        Returns
        -------
        None
        """
        if isinstance(image, np.ndarray):
            data = image
        elif isinstance(image, FlaggedArray):
            data = image.data
        else:
            raise ValueError("Image must be an array or FlaggedArray.")

        spline = Spline(data, exact=True, reduce_degrees=True)
        indices = np.stack([x.ravel() for x in to_indices])

        # The spline takes data in (x, y, z) FITS order for dimensions.
        self.data = spline(indices[::-1]).reshape(self.shape)

        invalid = indices < 0
        invalid |= indices >= np.asarray(self.shape)[:, None]
        invalid = np.any(invalid, axis=0)
        invalid = invalid.reshape(self.shape)
        self.discard(invalid)

    def value_at(self, index, degree=3):
        """
        Return the data value at a given index.

        Parameters
        ----------
        index : numpy.ndarray (int or float)
            An array of shape (n_dimensions,).  Should be supplied in (y, x)
            order (numpy).
        degree : int
            The spline degree to fit.

        Returns
        -------
        float
        """
        index = np.asarray(index)
        if not np.any(index % 1):
            index = tuple(index.astype(int))
            return self.data[index]

        slicer = []
        from_index = np.empty(self.ndim, dtype=int)
        for dimension in range(self.ndim):
            ind = index[dimension]
            min_ind = max(0, int(np.floor(ind)) - degree - 1)
            from_index[dimension] = min_ind
            max_ind = min(self.shape[dimension],
                          int(np.ceil(ind)) + degree + 1)
            slicer.append(slice(min_ind, max_ind))

        slicer = tuple(slicer)
        data = self.data[slicer].astype(float)
        valid = self.valid[slicer]
        if not valid.all():
            valid = valid.astype(float)
        else:
            valid = None

        spline = Spline(data, weights=valid, degrees=degree, smoothing=0.0)

        # From numpy to (x, y) ordering for spline
        xy_index = (index - from_index)[::-1]
        result = spline(*xy_index)
        return result

    def kernel_resample_from(self, image, kernel, to_indices,
                             kernel_reference_index=None, weights=None):
        """
        Resample an image onto the current flagged array via a kernel.

        Parameters
        ----------
        image : FlaggedArray or numpy.ndarray (float)
        kernel : numpy.ndarray (float)
        to_indices : numpy.ndarray (float or int)
            An array of shape (n_dimensions, self.shape or self.size)
        kernel_reference_index : numpy.ndarray (int or float)
            The reference index of the kernel defining center of the
            convolution operation.  The default is (kernel.shape - 1) / 2.
        weights : numpy.ndarray (float or int)
            The data weights for resampling.  Should be the same shape
            as image.

        Returns
        -------
        None
        """
        if kernel_reference_index is None:
            kernel_reference_index = (np.asarray(kernel.shape) - 1) / 2.0
            kernel_reference_index = kernel_reference_index.astype(int)

        kernel_spline = Spline(kernel, exact=True, reduce_degrees=True)
        indices = np.stack([x.ravel() for x in to_indices])

        if isinstance(image, FlaggedArray):
            data = image.data
        else:
            data = image

        if isinstance(weights, FlaggedArray):
            weights = weights.data

        smooth_values, smooth_weights = numba_functions.smooth_values_at(
            data=data,
            kernel=kernel,
            indices=indices,
            kernel_reference_index=kernel_reference_index,
            knots=kernel_spline.knots,
            coefficients=kernel_spline.coefficients,
            degrees=kernel_spline.degrees,
            panel_mapping=kernel_spline.panel_mapping,
            panel_steps=kernel_spline.panel_steps,
            knot_steps=kernel_spline.knot_steps,
            nk1=kernel_spline.nk1,
            spline_mapping=kernel_spline.spline_mapping,
            weight=weights,
            valid=None)

        smooth_values = smooth_values.reshape(self.shape)
        smooth_weights = smooth_weights.reshape(self.shape)

        invalid = indices < 0
        invalid |= indices >= np.asarray(self.shape)[:, None]
        invalid = np.any(invalid, axis=0)
        invalid = invalid.reshape(self.shape)
        invalid |= np.isnan(smooth_values)
        invalid |= smooth_weights <= 0
        self.data = smooth_values
        self.discard(invalid)

    def despike(self, threshold, noise_weight=None):
        """
        Discard spikes whose significance is above a given threshold.

        Parameters
        ----------
        threshold : float
        noise_weight : numpy.ndarray (float), optional
            Optional noise weights.

        Returns
        -------
        None
        """
        neighbor_kernel = self.get_neighbor_kernel()

        smoothed, smoothed_weight = numba_functions.regular_kernel_convolve(
            self.data, neighbor_kernel,
            weight=noise_weight,
            valid=self.valid)

        if noise_weight is None:
            noise_weight = self.valid.astype(float)

        difference_value = self.data - smoothed
        difference_weight = noise_weight * smoothed_weight
        nzi = difference_weight > 0
        difference_weight[nzi] /= (noise_weight[nzi] + smoothed_weight[nzi])
        difference_weight[~nzi] = 0
        significance = np.abs(difference_value * np.sqrt(difference_weight))
        self.discard(significance > threshold)

    def get_neighbor_kernel(self):
        """
        Return a neighbor kernel.

        The neighbor kernel contains the inverse square distance to the center
        pixel.  The center pixel will always be set to zero.  E.g., in 2D:

        kernel = [[0.5, 1. , 0.5],
                  [1. , 0. , 1. ],
                  [0.5, 1. , 0.5]]

        Returns
        -------
        numpy.ndarray (float)
        """
        shape = np.full(self.ndim, 3)
        indices = np.indices(shape)
        center = 1
        indices -= center
        kernel = np.zeros(indices.shape[1:], dtype=float)
        for dimension in range(self.ndim):
            kernel += indices[dimension] ** 2
        nzi = kernel != 0
        kernel[nzi] = 1 / kernel[nzi]
        return kernel

    def get_neighbors(self):
        """
        Return the number of valid neighbors for each point, including itself.

        Partial implementation of GetNeighborValidator.

        Returns
        -------
        numpy.ndarray (int)
        """
        kernel_shape = np.full(self.ndim, 3)
        center = np.full(self.ndim, 1)
        kernel = np.ones(kernel_shape)
        kernel.ravel()[np.ravel_multi_index(center, kernel_shape)] = 0
        valid = self.valid.astype(float)

        kernel = np.ones(np.full(self.ndim, 3), dtype=float)
        kernel.ravel()
        return np.round(signal.convolve(
            valid, kernel, mode='same')).astype(int)

    def discard_min_neighbors(self, min_neighbors):
        """
        Discard points with insufficient neighbors.

        Parameters
        ----------
        min_neighbors : int

        Returns
        -------
        None
        """
        self.discard(self.get_neighbors() < min_neighbors)

    def set_validating_flags(self, flag):
        """
        Set the validating flags (flags that are considered invalid).

        Parameters
        ----------
        flag : int or str of enum.Enum

        Returns
        -------
        None
        """
        self.validating_flags = self.flagspace.convert_flag(flag)

    def get_info(self):
        """
        Return a string descriptors of the array.

        Returns
        -------
        list of str
        """
        return [f"Image Size: {self.get_size_string()} pixels."]

    def count_points(self):
        """
        Return the number of valid points.

        Returns
        -------
        int
        """
        if self.data is None:
            return 0
        return np.sum(self.valid)

    def index_of_max(self, sign=1, data=None):
        """
        Return the maximum value and index of maximum value.

        Parameters
        ----------
        sign : int or float, optional
            If positive, find the maximum value in the array.  If negative,
            find the minimum value in the array.  If zero, find the maximum
            magnitude in the array.
        data : numpy.ndarray (float), optional
            The data array to examine.  Default is self.data.

        Returns
        -------
        maximum_value, maximum_index : float, int
        """
        if data is None:
            data = self.data

        value, index = numba_functions.index_of_max(
            data, valid=self.valid, sign=sign)
        return value, tuple(index)

    def get_refined_peak_index(self, peak_index):
        """
        Get the peak index given a local peak.

        Basically performs a quadratic fit on local neighborhood to determine
        the maximum.

        Parameters
        ----------
        peak_index : tuple (int)
            The peak index of the data array in (y, x) numpy format.

        Returns
        -------
        refined_peak_index : numpy.ndarray (float)
        """
        data = self.data
        valid = self.valid
        increment = np.zeros(self.ndim, dtype=float)
        for dimension in range(self.ndim):
            di = peak_index[dimension]
            from_index = di - 1
            if from_index < 0:
                continue
            to_index = di + 1
            if to_index >= self.shape[dimension]:
                continue

            slice_index = []
            for dim in range(self.ndim):
                if dim == dimension:
                    slice_index.append(slice(from_index, to_index + 1))
                else:
                    slice_index.append(peak_index[dim])

            slice_index = tuple(slice_index)
            if not np.all(valid[slice_index]):
                continue
            d0, d1, d2 = data[slice_index]
            v1 = 0.5 * (d2 + d0) - d1
            v2 = 0.5 * (d2 - d0)
            if v1 == 0:
                continue
            v = -0.5 * v2 / v1
            if np.abs(v) <= 0.5:
                increment[dimension] = v

        return np.asarray(peak_index) + increment

    def crop(self, ranges):
        """
        Crop the array to the required dimensions.

        Parameters
        ----------
        ranges : numpy.ndarray (int,)
            The ranges to set crop the data to.  Should be of shape
            (n_dimensions, 2) where ranges[0, 0] would give the minimum crop
            limit for the first dimension and ranges[0, 1] would give the
            maximum crop limit for the first dimension.  In this case, the
            'first' dimension is in numpy format.  i.e., (y, x) for a 2-D
            array. Also note that the upper crop limit is not inclusive so
            a range of (0, 3) includes indices [0, 1, 2] but not 3.

        Returns
        -------
        None
        """
        if self.size == 0:
            return
        slicer = []
        for dimension in range(self.ndim):
            from_index, to_index = ranges[dimension]
            slicer.append(slice(from_index, to_index))

        slicer = tuple(slicer)
        data = self.data[slicer].copy()
        flags = self.flag[slicer].copy()
        fixed_indices = self.fixed_index[slicer].copy()
        self.set_data_shape(data.shape)
        self.set_data(data)
        self.flag = flags
        self.fixed_index = fixed_indices

    def get_cropped(self, ranges):
        """
        Return a copy of the cropped Flagged Array.

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
        FlaggedArray
        """
        if self.size == 0:
            return self.copy()

        new = self.copy()
        new.crop(ranges)
        return new
