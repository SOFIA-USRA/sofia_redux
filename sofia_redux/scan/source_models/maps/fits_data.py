# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import log, units
from astropy.io import fits

from sofia_redux.scan.flags.flagged_array import FlaggedArray
from sofia_redux.scan.coordinate_systems.coordinate import Coordinate

__all__ = ['FitsData']


class FitsData(FlaggedArray):

    DEFAULT_UNIT = 1.0 * units.dimensionless_unscaled

    def __init__(self, data=None, blanking_value=None, dtype=float,
                 shape=None, unit=None):
        """
        Creates a FitsData instance.

        The FitsData class is an extension of `FlaggedArray` that allows
        additional functionality for FITS data.  This includes specifying
        data units (astropy.units), history messages and FITS handling.

        Notably, the dimensional ordering will now be in FITS (x, y) order
        rather than the `FlaggedArray` Numpy (y, x) ordering for the various
        methods.

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
        """
        self.history = []
        self.verbose = False
        self._unit = None
        self.local_units = {}
        self.alternate_unit_names = {}

        self.log_new_data = True
        self.parallelism = 0
        self.executor = None
        if unit is not None:
            self.unit = unit

        super().__init__(data=data, blanking_value=blanking_value, dtype=dtype,
                         shape=shape)

        if self.unit is None:
            self.unit = self.DEFAULT_UNIT

    def copy(self):
        """
        Return a copy of the FITS data.

        Returns
        -------
        FitsData
        """
        return super().copy()

    def __eq__(self, other):
        """
        Check whether this data is equal to another.

        Parameters
        ----------
        other : FitsData

        Returns
        -------
        equal : bool
        """
        if self is other:
            return True
        if not isinstance(other, FitsData):
            return False
        if self.unit != other.unit:
            return False
        return super().__eq__(other)

    @property
    def referenced_attributes(self):
        """
        Return attribute names that should be referenced during a copy.

        Returns
        -------
        set (str)
        """
        referenced = super().referenced_attributes
        referenced.add('local_units')
        referenced.add('alternate_unit_names')
        return referenced

    @property
    def unit(self):
        """
        Return the current data unit.

        Returns
        -------
        astropy.units.Quantity
        """
        return self._unit

    @unit.setter
    def unit(self, value):
        """
        Set the unit value.

        Parameters
        ----------
        value : str or astropy.units.Unit or astropy.units.Quantity.
            The unit to set.  If a string or Unit is supplied, the quantity
            value will be set to unity.

        Returns
        -------
        None
        """
        self.set_unit(value)

    @classmethod
    def fits_to_numpy(cls, coordinates):
        """
        Convert FITS based (x, y) coordinates/indices to numpy (y, x).

        Reverses the dimensional ordering so that (x, y) coordinates are
        returned as (y, x) coordinates.  Note that boolean arrays remain
        unaltered, since these usually indicate masking arrays.

        Parameters
        ----------
        coordinates : numpy.ndarray or Coordinate or iterable

        Returns
        -------
        numpy_coordinates : numpy.ndarray (int or float)
        """
        if isinstance(coordinates, Coordinate):
            new = coordinates.coordinates[::-1]
        elif isinstance(coordinates, np.ndarray):
            if coordinates.dtype != bool:  # To avoid messing up masks
                new = coordinates[::-1]
            else:
                new = coordinates
        elif hasattr(coordinates, '__len__'):
            new = coordinates[::-1]
        else:
            new = coordinates
        return new

    @classmethod
    def numpy_to_fits(cls, coordinates):
        """
        Convert numpy based (x, y) coordinates/indices to FITS coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray or Coordinate or iterable

        Returns
        -------
        numpy.ndarray
        """
        return cls.fits_to_numpy(coordinates)

    def get_size_string(self):
        """
        Return the shape of the data array as a string.

        Returns
        -------
        str
        """
        if self.data is None:
            return '0'
        return 'x'.join([str(x) for x in self.shape[::-1]])

    def set_data_shape(self, shape):
        """
        Set the shape of the data image array.

        Note that array shapes should still be provided in Numpy (y, x) format.
        In addition to settings the data shape, adds a new history message
        indicating that this has been done.

        Parameters
        ----------
        shape : tuple (int)

        Returns
        -------
        None
        """
        super().set_data_shape(shape)
        if self.log_new_data:
            self.clear_history()
        self.add_history(f'new size {self.get_size_string()}')

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
        if isinstance(data, FitsData):
            if data.unit != self.unit:
                self.unit = data.unit
        super().set_data(data, change_type=change_type)

    @staticmethod
    def unit_to_quantity(unit):
        """
        Return an astropy Quantity from a given unit.

        If a string or astropy.units.Unit is supplied, the resultant output
        will be a quantity with value 1.

        Parameters
        ----------
        unit : astropy.units.Quantity or astropy.units.Unit or str
            The unit to convert to a quantity.

        Returns
        -------
        astropy.units.Quantity
        """
        if isinstance(unit, units.Quantity):
            return unit
        elif isinstance(unit, str):
            return 1.0 * units.Unit(unit)
        elif isinstance(
                unit, (units.Unit, units.Quantity, units.IrreducibleUnit)):
            return 1.0 * unit
        elif isinstance(unit, units.UnitBase):  # dimensionless
            return 1.0 * unit
        else:
            raise ValueError(
                f"Unit must be a {str}, {units.Unit}, or {units.Quantity}.")

    def add_local_unit(self, unit, alternate_names=None):
        """
        Add a unit to the dictionary of local units.

        Parameters
        ----------
        unit : astropy.units.Quantity or units.Unit or str
            The unit to add to the unit dictionary.  Should be a quantity
            with both value and unit type.  If a simple unit is supplied, the
            value is assumed to be 1.0.
        alternate_names : list, optional
            A list of alternate names for the unit.

        Returns
        -------
        None
        """
        unit = self.unit_to_quantity(unit)
        unit_key = unit.unit.to_string()
        self.local_units[unit_key] = unit

        if hasattr(unit.unit, 'names'):
            self.add_alternate_unit_names(unit_key, unit.unit.names)
        else:
            self.add_alternate_unit_names(unit_key, unit_key)

        if alternate_names is not None:
            self.add_alternate_unit_names(unit_key, alternate_names)

    def add_alternate_unit_names(self, unit_name, alternate_names):
        """
        Add alternative names for a unit.

        Parameters
        ----------
        unit_name : str
            The name of the unit.
        alternate_names : str or list (str)
            Alternate names for the unit.

        Returns
        -------
        None
        """
        if isinstance(alternate_names, str):
            alternate_names = [alternate_names]
        for alternate_name in alternate_names:
            self.alternate_unit_names[alternate_name] = unit_name
        self.alternate_unit_names[unit_name] = unit_name

    def get_unit(self, unit):
        """
        Return the unit quantity for a given unit.

        Parameters
        ----------
        unit : str or units.Unit or units.Quantity

        Returns
        -------
        units.Quantity
        """
        if isinstance(unit, units.Quantity):
            unit_key = str(unit.unit)
        else:
            unit_key = str(unit)

        if unit_key in self.local_units:
            return self.local_units.get(unit_key)
        elif unit_key in self.alternate_unit_names:
            return self.local_units.get(
                self.alternate_unit_names.get(unit_key))
        else:
            return self.unit_to_quantity(unit)

    def set_unit(self, unit):
        """
        Set the map data unit.

        Parameters
        ----------
        unit : astropy.units.Quantity or str or astropy.units.Unit
            The unit to set as the map data unit.  Should be a quantity
            (value and unit type).  If a string or Unit is supplied, the
            map data unit will be set to the value located in the local_units
            dictionary.  If no such value exists, a KeyError will be raised.

        Returns
        -------
        None
        """
        unit = self.unit_to_quantity(unit)
        self.add_local_unit(unit)
        self._unit = unit

    def set_default_unit(self):
        """
        Set the default unit for the map data.

        Returns
        -------
        None
        """
        self.set_unit(self.DEFAULT_UNIT)

    def clear_history(self):
        """
        Clear the history messages.

        Returns
        -------
        None
        """
        self.history = []

    def add_history(self, message):
        """
        Add a history message.

        Will also result in a log message if verbose is True.

        Parameters
        ----------
        message : str or list(str)

        Returns
        -------
        None
        """
        if isinstance(message, str):
            if self.verbose:
                log.info(message)
            if self.history is None:
                self.history = []
            self.history.append(message)
        elif isinstance(message, list):
            for msg in message:
                self.add_history(msg)

    def set_history(self, messages):
        """
        Set the history to a given list of history messages.

        Parameters
        ----------
        messages : str or list (str)

        Returns
        -------
        None
        """
        if isinstance(messages, str):
            messages = [messages]
        else:
            messages = list(messages)
        self.history = messages

    def add_history_to_header(self, header):
        """
        Add history messages to a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        if self.history is None:
            return
        for message in self.history:
            header['HISTORY'] = message

    def record_new_data(self, detail=None):
        """
        Start recording new data.

        Parameters
        ----------
        detail : str, optional
            An optional message to append to the start of the new data log.

        Returns
        -------
        None
        """
        if not self.log_new_data:
            self.log_new_data = True
            return
        self.clear_history()
        message = f"set new image {self.get_size_string()}"
        if isinstance(detail, str):
            message += f' {detail}'
        self.add_history(message)

    def set_parallel(self, threads):
        """
        Set the number of parallel threads.

        Parameters
        ----------
        threads : int

        Returns
        -------
        None
        """
        self.parallelism = int(threads)

    def set_executor(self, executor):
        """
        Set the executor?

        Parameters
        ----------
        executor : Executor

        Returns
        -------
        None
        """
        self.executor = executor

    def clear(self, indices=None):
        """
        Clear flags and set data to zero.  Clear history.

        Parameters
        ----------
        indices : tuple (numpy.ndarray (int)) or numpy.ndarray (bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).  Note that if an integer array or tuple
            of integer arrays are supplied, they should be in (x, y) FITS
            order.  Boolean masks should be of the same shape as the data.

        Returns
        -------
        None
        """
        super().clear(indices=self.fits_to_numpy(indices))
        self.clear_history()
        self.add_history(f'clear {self.get_size_string()}')

    def destroy(self):
        """
        Destroy the image data.

        Returns
        -------
        None
        """
        super().destroy()
        self.clear_history()

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
        super().fill(value, indices=self.fits_to_numpy(indices))
        self.clear_history()
        self.add_history(f"fill {self.get_size_string()} with {value}")

    def add(self, value, indices=None, factor=None):
        """
        Add a value or FlaggedData to the data array.

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
        super().add(value, indices=self.fits_to_numpy(indices), factor=factor)
        is_array = isinstance(value, FlaggedArray)
        if factor is None:
            if is_array:
                message = f'added {value.__class__.__name__}'
            else:
                message = f'add {value}'
        else:
            if is_array:
                message = f'added scaled ' \
                          f'{value.__class__.__name__} ({factor}x)'
            else:
                message = f'add {value * factor}'
        self.add_history(message)

    def scale(self, factor, indices=None):
        """
        Scale the data by a given factor.

        Parameters
        ----------
        factor : int or float
            The factor to scale by.
        indices : tuple (np.ndarray) or np.ndarray (int or bool), optional
            The indices to discard.  Either supplied as a boolean mask of
            shape (self.data.shape).

        Returns
        -------
        None
        """
        np_indices = self.fits_to_numpy(indices)
        if isinstance(np_indices, np.ndarray) and np_indices.dtype != bool:
            if 1 < self.ndim == np_indices.shape[0]:
                np_indices = tuple(np_indices)
        super().scale(factor, indices=np_indices)
        self.add_history(f'scale by {factor}')

    def validate(self, validator=None):
        """
        Discard all invalid data.

        Parameters
        ----------
        validator : Validator, optional
            An optional object or function that can take FitsData as an
            argument and perform the validation.

        Returns
        -------
        None
        """
        super().validate(validator=validator)
        if validator is None:
            self.add_history('validate')
        else:
            self.add_history(f'validate via {validator}')

    def paste(self, source, report=True):
        """
        Paste data from another MapData onto the data array.

        Parameters
        ----------
        source : FitsData
        report : bool, optional
            If `True`, add a history message.

        Returns
        -------
        None
        """
        super().paste(source)
        if report:
            self.add_history(f"pasted new content: {source.get_size_string()}")

    def smooth(self, beam_map, reference_index=None, weights=None):
        """
        Smooth the map with a beam map kernel.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The beam map with which to smooth the map of shape (ny, nx, ...).
        reference_index : numpy.ndarray (int) or Coordinate, optional
            The index specifying the center pixel of the kernel.  Should be
            provided in (x, y) ordering.  The default is the center of the
            kernel.
        weights : numpy.ndarray (float), optional
            The map weights to apply.  Should be the same shape as the primary
            map image.  The default is to apply no weighting.

        Returns
        -------
        None
        """
        super().smooth(beam_map,
                       reference_index=self.fits_to_numpy(reference_index),
                       weights=weights)
        self.add_history('smoothed')

    def get_smoothed(self, beam_map, reference_index=None, weights=None):
        """
        Return smoothed data and weights.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The kernel with which to smooth the map.  Should be of shape
            (ny, nx).
        reference_index : numpy.ndarray (float or int) or Coordinate, optional
            The reference index specifying the center pixel of the beam map.
            The default is the center pixel position, eg., (ny - 1) / 2.
        weights : numpy.ndarray (float), optional
            The weights to apply during smoothing.  Should be the same shape
            as the primary map image.  The default is no weighting.

        Returns
        -------
        smoothed_data, smoothed_weights : numpy.ndarray, numpy.ndarray
        """
        return super().get_smoothed(
            beam_map, reference_index=self.fits_to_numpy(reference_index),
            weights=weights)

    def fast_smooth(self, beam_map, steps, reference_index=None, weights=None):
        """
        Smooth using the fast method.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The beam map image kernel with which to smooth the map of shape
            (ky, kx).
        steps : Index2D
            The fast step skips in (x, y) FITS dimensional order.
        reference_index : Coordinate2D, optional
            The reference pixel index of the kernel in (x, y) FITS
            dimensional order.  The default is the beam map center at
            (kx-1)/2, (ky-1)/2.
        weights : numpy.ndarray (float), optional
            The map weights of shape (ny, nx).  The default is no weighting.

        Returns
        -------
        None
        """
        super().fast_smooth(
            beam_map,
            self.fits_to_numpy(steps),
            reference_index=self.fits_to_numpy(reference_index),
            weights=weights)

        self.add_history('smoothed (fast method)')

    def get_fast_smoothed(self, beam_map, steps, reference_index=None,
                          weights=None, get_weights=False):
        """
        Return smoothed values using the fast method.

        Parameters
        ----------
        beam_map : numpy.ndarray (float)
            The kernel to convolve with.
        steps : numpy.ndarray (float)
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
        return super().get_fast_smoothed(
            beam_map,
            self.fits_to_numpy(steps),
            reference_index=self.fits_to_numpy(reference_index),
            weights=weights,
            get_weights=get_weights)

    def create_fits(self):
        """
        Create and return a FITS HDU list from data content.

        Returns
        -------
        astropy.io.fits.HDUList
        """
        hdu_list = fits.HDUList()
        for hdu in self.get_hdus():
            hdu_list.append(hdu)
        return hdu_list

    def get_hdus(self):
        """
        Create and return a list if FITS HDUs.

        Returns
        -------
        hdus: list (astropy.io.fits.hdu.base.ExtensionHDU)
        """
        return [self.create_hdu()]

    def create_hdu(self):
        """
        Create a FITS HDU from map data content.

        Returns
        -------
        astropy.io.fits.ImageHDU
        """
        data = self.get_fits_data().copy()
        data[~self.valid] = np.nan
        hdu = fits.ImageHDU(data=data)
        self.edit_header(hdu.header)
        return hdu

    def get_fits_data(self):
        """
        Return the FITS data array.

        Returns
        -------
        array
        """
        return self.data

    def edit_header(self, header):
        """
        Edit a FITS header using information in the current map.

        The information keywords added to the header are::

          - DATAMIN: The lowest value in the data (float)
          - DATAMAX: The highesst value in the data (float)
          - BZERO: Zeroing level of the data (float)
          - BSCALE: Scaling of the data (float)
          - BUNIT: The data unit (str)

        Note that BUNIT will default to 'count' if no unit has been set.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        data_range = self.data_range
        unit_value = 1.0 if self.unit is None else self.unit.value

        if data_range.min < data_range.max:

            if np.isfinite(data_range.min):
                header['DATAMIN'] = (data_range.min / unit_value,
                                     'The lowest value in the image')
            if np.isfinite(data_range.max):
                header['DATAMAX'] = (data_range.max / unit_value,
                                     'The highest value in the image')

        header['BZERO'] = 0.0, 'Zeroing level of the image data'
        header['BSCALE'] = 1.0, 'Scaling of the image data'

        if (not isinstance(self.unit, units.Quantity) or
                self.unit.unit == units.dimensionless_unscaled):
            header['BUNIT'] = (units.Unit("count").to_string(),
                               'Data unit specification.')
        else:
            header['BUNIT'] = (self.unit.unit.to_string(),
                               'Data unit specification.')

        self.add_history_to_header(header)

    def parse_header(self, header):
        """
        Parse and apply information from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        if 'BUNIT' in header:
            self.set_unit(header['BUNIT'])
        else:
            self.set_unit(units.dimensionless_unscaled)

    def parse_history(self, header):
        """
        Set the history from a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header

        Returns
        -------
        None
        """
        history = list(header['HISTORY']) if 'HISTORY' in header else []
        self.set_history(history)

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : list or int or numpy.ndarray (bool or int)
            The indices to extract.

        Raises
        ------
        NotImplementedError
        """
        super().get_indices(indices)

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

        Raises
        ------
        NotImplementedError
        """
        super().delete_indices(indices_or_mask)

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

        Raises
        ------
        NotImplementedError
        """
        super().insert_blanks(insert_indices)

    def merge(self, data):
        """
        Add additional data onto the end of this data.

        Parameters
        ----------
        data : FlaggedData

        Raises
        ------
        NotImplementedError
        """
        super().merge(data)

    def resample_from(self, image, to_indices, kernel=None,
                      kernel_reference_index=None, weights=None):
        """
        Resample an image onto given indices of this map.

        Parameters
        ----------
        image : FlaggedArray or numpy.ndarray (float)
            The image to resample of shape (shape,) and have n_dimensions.
        to_indices : numpy.ndarray (float or int)
            An array of shape (n_dimensions, self.shape or self.size)
            specifying which image pixels belong to the resampled map.
            Dimensions should be ordered using the FITS (x, y) convention.
            For example, if pixel i is at (x, y) = (2, 3) then
            `to_indices[:, i] = [2, 3]`.
        kernel : numpy.ndarray (float), optional
            The kernel used to perform the resampling.  If supplied, the result
            will be smoothed accordingly.
        kernel_reference_index : numpy.ndarray (int), optional
            If a kernel is supplied, specifies the center pixel of the kernel
            to be used during kernel convolution.  The default is
            (kernel.shape - 1) / 2.  Should be an array of shape
            (n_dimensions,) using FITS dimensional ordering (x, y).
        weights : numpy.ndarray (int or float), optional
            An optional weighting array used for kernel convolution.  Should
            be an array of shape (shape,).

        Returns
        -------
        None
        """
        # Reverse for FITS to numpy dimensional order.
        if kernel_reference_index is not None:
            kernel_reference_index = self.fits_to_numpy(kernel_reference_index)
        indices = self.fits_to_numpy(to_indices)

        super().resample_from(image, indices, kernel=kernel,
                              kernel_reference_index=kernel_reference_index,
                              weights=weights)

        if isinstance(image, np.ndarray):
            # Just to get the size string
            image = self.__class__(data=image)

        self.add_history(f'resampled {self.get_size_string()} '
                         f'from {image.get_size_string()}')

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
        super().despike(threshold, noise_weight=noise_weight)
        self.add_history(f'despiked at {threshold:.3f}')

    def get_index_range(self):
        """
        Return the index ranges of valid points.

        Returns
        -------
        ranges : numpy.ndarray (int) or Coordinate
            A range for each dimension or shape (n_dimensions, 2) giving the
            minimum and maximum range in each dimension.  Note that this is
            FITS dimension ordering (x-range = ranges[0], y-range = ranges[1]).
            Also note that the upper range is returned such that the real
            upper index is included in any slice operation. i.e., max = real
            max index + 1.
        """
        return self.numpy_to_fits(super().get_index_range())

    def value_at(self, index, degree=3):
        """
        Return the data value at a given index.

        Parameters
        ----------
        index : numpy.ndarray (int or float)
            An array of shape (n_dimensions,).  Should be supplied in (x, y)
            order (FITS).
        degree : int, optional
            The degree of spline to fit.

        Returns
        -------
        float
        """
        return super().value_at(
            np.atleast_1d(self.fits_to_numpy(index)), degree=degree)

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
        maximum_value, maximum_index : float, numpy.ndarray or Coordinate
        """
        value, index = super().index_of_max(sign=sign, data=data)
        return value, self.numpy_to_fits(index)

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
        refined_peak_index : numpy.ndarray or Coordinate
        """
        return self.numpy_to_fits(super().get_refined_peak_index(
            self.fits_to_numpy(peak_index)))

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
            'first' dimension is in FITS format.  i.e., (x, y) for a 2-D array.
            Also note that the upper crop limit is not inclusive so a range
            of (0, 3) includes indices [0, 1, 2] but not 3.

        Returns
        -------
        None
        """
        super().crop(self.fits_to_numpy(ranges))
