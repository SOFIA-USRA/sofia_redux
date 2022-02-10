# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
from astropy.coordinates import (
    BaseCoordinateFrame, SkyCoord, SkyOffsetFrame, EarthLocation)
from astropy.coordinates import concatenate as astro_concat
from copy import deepcopy
import inspect
import numpy as np
from scipy.sparse.csr import csr_matrix

from sofia_redux.scan.flags.flags import Flags
from sofia_redux.scan.utilities.utils import skycoord_insert_blanks
from sofia_redux.scan.flags import flag_numba_functions
from sofia_redux.scan.coordinate_systems.coordinate import Coordinate

__all__ = ['FlaggedData']


class FlaggedData(ABC):

    flagspace = Flags

    def __init__(self):
        """
        Initialize a FlaggedData object.

        The FlaggedData object contains a set of associated flags and original
        data indices and allows operations that include those attributes.
        """
        # These are all arrays of equal size
        self._fixed_index = None
        self._flag = None

    @property
    def is_singular(self):
        """
        Return whether the data contains only a single scalar measurement.

        Returns
        -------
        singular : bool
        """
        return self.fixed_index.shape == ()

    @property
    def fixed_index(self):
        """
        Return the fixed index.

        The fixed indices are designed to be constant irrespective of any
        operations, deletions, etc and provide a reverse lookup onto the
        original data set.

        Returns
        -------
        fixed_index : numpy.ndarray (int)
        """
        return self._fixed_index

    @fixed_index.setter
    def fixed_index(self, values):
        """
        Set the fixed index values of the FlaggedData.

        Parameters
        ----------
        values : numpy.ndarray (int)

        Returns
        -------
        None
        """
        self._fixed_index = np.asarray(values)

    @property
    def flag(self):
        """
        Return the associated data flags.

        Returns
        -------
        flags : numpy.ndarray (int)
        """
        return self._flag

    @flag.setter
    def flag(self, values):
        """
        Set the data flags.

        Parameters
        ----------
        values : numpy.ndarray (int)

        Returns
        -------
        None
        """
        self._flag = values

    @property
    def default_field_types(self):
        """
        Return the default values for FlaggedData attributes.

        Returns
        -------
        dict
            Keys contain the name of the attribute and values contain the
            value.
        """
        return {'flag': 0}

    @property
    def referenced_attributes(self):
        """
        Returns attribute names that should be referenced during a copy.

        Returns
        -------
        set (str)
        """
        return set([])

    @property
    def internal_attributes(self):
        """
        Returns attribute names that are internal to the data for get actions.

        These attributes should always be returned as-is regardless
        of indexing.

        Returns
        -------
        set (str)
        """
        return set([])

    @property
    def special_fields(self):
        """
        Return fields that do not comply with the shape of other data.

        This is of particular importance for `delete_indices`.  Although all
        arrays must have shape[0] = self.size, special handling may be required
        in certain cases.

        Returns
        -------
        fields : set (str)
        """
        return set([])

    @property
    def fields(self):
        """
        Return the available attributes of the FlaggedData object.

        Returns
        -------
        set (str)
        """
        return set([key for key in self.__dict__.keys()])

    @property
    def size(self):
        """
        Return the size of the FlaggedData.

        Returns
        -------
        int
        """
        if self.fixed_index is None:
            return 0
        return self.fixed_index.size

    @property
    def shape(self):
        """
        Return the shape of the FlaggedData.

        This is only 1-D.

        Returns
        -------
        tuple (int)
        """
        if self.fixed_index is None:
            return ()
        if self.is_singular:
            return ()
        return self.size,

    @shape.setter
    def shape(self, _):
        """
        Set the shape of the FlaggedData

        Parameters
        ----------
        _ : unused for FlaggedData

        Returns
        -------
        None
        """
        pass  # pragma: no cover

    def __getitem__(self, indices):
        """
        Return a selection of the data.

        Parameters
        ----------
        indices : int or slice or numpy.ndarray (int or bool)

        Returns
        -------
        None
        """
        return self.get_indices(indices)

    def set_default_values(self):
        """
        Populate data fields with default values.

        The default values are loaded from the `default_field_types` property
        which returns a dictionary of the form {field_name: value}.  If the
        value is a type, the default values will be empty numpy arrays.  Other
        valid values can be astropy quantities or standard python types such
        as int, float, str, etc.  All fields will be set to numpy arrays of the
        same type and filled with the same value.

        Returns
        -------
        None
        """
        for key, value in self.default_field_types.items():
            if isinstance(value, type):
                setattr(self, key, np.empty(self.shape, dtype=value))
            elif isinstance(value, units.Quantity):
                setattr(self, key,
                        np.full(self.shape, value.value) * value.unit)
            elif isinstance(value, units.UnitBase):
                setattr(self, key, np.empty(self.shape, dtype=float) * value)

            elif isinstance(value, tuple):
                shape = self.shape + value[1:]
                fill_value = value[0]
                if inspect.isclass(fill_value) and issubclass(
                        fill_value, Coordinate):
                    coordinate_class, fill_value = value[0], value[1]
                    coordinate_shape = (
                        (coordinate_class.default_dimensions,)
                        + self.shape + value[2:])
                    if isinstance(fill_value, units.Quantity):
                        unit = fill_value.unit
                        fill_values = np.full(
                            coordinate_shape, fill_value.value) * unit
                    elif isinstance(fill_value, units.UnitBase):
                        unit = fill_value
                        fill_values = np.empty(
                            coordinate_shape, dtype=float) * unit
                    elif isinstance(fill_value, str):
                        unit = units.Unit(fill_value)
                        fill_values = np.empty(
                            coordinate_shape, dtype=float) * unit
                    else:
                        fill_values = np.full(coordinate_shape, fill_value)
                    setattr(
                        self, key, coordinate_class(fill_values, copy=False))

                elif isinstance(fill_value, units.Quantity):
                    setattr(self, key, np.full(
                        shape, fill_value.value) * fill_value.unit)

                elif isinstance(fill_value, units.UnitBase):
                    setattr(self, key, np.empty(shape, dtype=float)
                            * fill_value)

                elif isinstance(fill_value, type):
                    setattr(self, key, np.empty(shape, dtype=fill_value))

                else:
                    setattr(self, key, np.full(shape, fill_value))
            else:
                setattr(self, key, np.full(self.shape, value))

    def copy(self):
        """
        Return a copy of the data.

        Returns
        -------
        FlaggedData
        """
        new = self.__class__()
        for key, value in self.__dict__.items():
            if key in self.referenced_attributes:
                setattr(new, key, value)
            elif hasattr(value, 'copy'):
                setattr(new, key, value.copy())
            else:
                setattr(new, key, deepcopy(value))
        return new

    def is_flagged(self, flag=None, indices=False):
        """
        Find flagged data indices.

        Parameters
        ----------
        flag : int or str or Flag.flags, optional
            The flag to check.  If not supplied, returns any indices that have
            non-zero flags.
        indices : bool, optional
            If `False` the return value will be a boolean mask.  If `True`,
            the actual indices will be returned.

        Returns
        -------
        flagged : numpy.ndarray (bool or int)
            A boolean mask if `indices` is `False` or channel
            indices otherwise.
        """
        return self.flagspace.is_flagged(self.flag, flag, indices=indices)

    def is_unflagged(self, flag=None, indices=False):
        """
        Find data indices that are not flagged.

        Parameters
        ----------
        flag : int or str or Flags.flags, optional
            The flag to check.  If not supplied, returns any indices that have
            zero flags.
        indices : bool, optional
            If `False` the return value will be a boolean mask.  If `True`, the
            actual indices will be returned.

        Returns
        -------
        unflagged : numpy.ndarray (bool or int)
            A boolean mask if `indices` is `False` or indices otherwise.
        """
        return self.flagspace.is_unflagged(self.flag, flag, indices=indices)

    def set_flags(self, flag, indices=None):
        """
        Flag data with the supplied flag.

        Parameters
        ----------
        flag : int or str or Flags.flags
            The flag to set.
        indices : numpy.ndarray (int), optional
            The indices to flag.  If not supplied, all data are flagged.

        Returns
        -------
        None
        """
        indices = self.to_indices(indices)
        if indices is not None:
            if self.fixed_index.ndim == 1 and indices.size == 0:
                return
            elif self.fixed_index.ndim > 1 and indices[0].size == 0:
                return

        if not isinstance(flag, int):
            flag = self.flagspace.convert_flag(flag).value

        flag_numba_functions.set_flags(self.flag, flag, indices=indices)

    def unflag(self, flag=None, indices=None):
        """
        Remove data flags.

        Parameters
        ----------
        flag : int or str or Flags.flags
            The flag to remove.  If not supplied, all data are unflagged.
        indices : numpy.ndarray (int), optional
            The indices to flag.  If not supplied, all data are flagged.

        Returns
        -------
        None
        """
        indices = self.to_indices(indices)
        if indices is not None:
            if self.fixed_index.ndim == 1 and indices.size == 0:
                return
            elif self.fixed_index.ndim > 1 and indices[0].size == 0:
                return

        if flag is not None and not isinstance(flag, int):
            flag = self.flagspace.convert_flag(flag).value

        flag_numba_functions.unflag(self.flag, flag, indices=indices)

    def discard_flag(self, flag, criterion=None):
        r"""
        Remove all data flagged with the given flag.

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
        self.delete_indices(self.flagspace.discard_indices(
            self.flag, flag, criterion=criterion))

    def get_flagged_indices(self, keep_flag=None, discard_flag=None,
                            match_flag=None, indices=True):
        """
        Return flagged indices based on a number of criteria.

        Parameters
        ----------
        keep_flag : int or str or Flags.flags
            Keep flags that contain this/these flags.
        discard_flag : int or str or Flags.flags
            Discard any flags that contain this/these flags.
        match_flag : int or str or Flags.flags
            Keep only those flags that exactly match this/these flags.
            Overrides all other criteria.
        indices : bool, optional
            If `True`, return an array of integer indices.  Otherwise, return
            a boolean mask where `True` indicates flagged data.

        Returns
        -------
        indices : numpy.ndarray (int or bool)
             The indices or mask matching the flag criteria(s).
        """
        if match_flag is not None:
            mask = self.flagspace.is_flagged(
                self.flag, flag=match_flag, exact=True)
        else:
            mask = None
            if keep_flag is not None:
                m = self.flagspace.is_flagged(self.flag, flag=keep_flag)
                if mask is None:
                    mask = m

            if discard_flag is not None:
                m = self.flagspace.is_unflagged(self.flag, flag=discard_flag)
                if mask is None:
                    mask = m
                else:
                    mask &= m

            if mask is None:
                mask = np.full(self.flag.shape, True)

        if not indices:
            return mask

        result = np.nonzero(mask)
        return result[0] if len(result) == 1 else result

    def find_fixed_indices(self, fixed_indices, cull=True):
        """
        Returns the actual indices given fixed indices.

        The fixed indices are those that are initially loaded.
        Returned indices are their locations in the data arrays.

        Parameters
        ----------
        fixed_indices : int or np.ndarray (int)
            The fixed indices.
        cull : bool, optional
            If `True`, do not include fixed indices not found in the result.
            If `False`, missing indices will be replaced by -1.

        Returns
        -------
        indices : numpy.ndarray (int)
            The indices of `fixed_indices` in the data arrays.  A tuple will
            be returned, in the case where we are examining more than one
            dimension.
        """
        values = np.asarray(fixed_indices, dtype=int)
        singular = values.ndim == 0
        if singular:
            indices = np.nonzero(self.fixed_index == values)[0]
            if not cull and indices.size == 0:
                return -1
            if indices.size == 1:
                return int(indices[0])
            else:
                return indices

        if cull:
            return np.nonzero(values[:, None] == self.fixed_index)[1]
        else:
            mask = values[:, None] == self.fixed_index
            valid = np.any(mask, axis=1)
            indices = np.full(values.size, -1)
            indices[valid] = np.nonzero(mask)[1]
            return indices

    def to_indices(self, indices_or_mask, discard=False):
        """
        Convert an array of indices, or an array mask to the correct format.

        Parameters
        ----------
        indices_or_mask : numpy.ndarray (bool or int) or slice
            An array of indices or an array mask where each `True` element
            indicates an index that should be included.
        discard : bool, optional
            If `True`, return all indices that are not included in the indices,
            or the indices of `False` values in a mask.

        Returns
        -------
        indices : numpy.ndarray (int)
        """
        if indices_or_mask is None:
            return None

        if isinstance(indices_or_mask, slice):
            mask = np.full(self.shape, False)
            mask[indices_or_mask] = True
            indices_or_mask = mask

        indices_or_mask = np.atleast_1d(indices_or_mask)

        if indices_or_mask.dtype == bool:
            if discard:
                indices = np.nonzero(np.logical_not(indices_or_mask))
            else:
                indices = np.nonzero(indices_or_mask)
        elif discard:
            keep = np.full(self.shape, False)
            keep[indices_or_mask] = True
            keep = np.logical_not(keep)
            indices = np.nonzero(keep)
        else:
            indices = indices_or_mask

        if isinstance(indices, tuple) and len(indices) == 1:
            indices = indices[0]

        return indices

    def get_indices(self, indices):
        """
        Return selected data for given indices.

        Parameters
        ----------
        indices : list or int or numpy.ndarray (int)
            The indices to extract.

        Returns
        -------
        FlaggedData
        """
        if isinstance(indices, np.ndarray) and indices.shape == ():
            indices = int(indices)
        new = self.__class__()
        if self.size == 0:
            return new

        internal = self.internal_attributes

        for attribute, value in self.__dict__.items():
            setattr(new, attribute, self.get_attribute_indices(
                internal, attribute, value, indices))
        return new

    @staticmethod
    def get_attribute_indices(internal, attribute, value, indices):
        """
        Return selected indices of a given value.

        Parameters
        ----------
        internal : set (str)
            If `attribute` is found in this set, the original value will be
            returned, as internal attributes should remain constant regardless
            of indexing.
        attribute : str
            The name of the attribute in the FlaggedData to which `value`
            belongs.
        value : numpy.ndarray or Coordinate or object or csr_matrix
            The value for which to retrieve `indices`.
        indices : numpy.ndarray (bool or int) or slice or tuple (int)
            The indices to select and return from `value`.

        Returns
        -------
        selected_values : numpy.ndarray or Coordinate or object or csr_matrix
            In most cases the result will be `value[indices]`.
        """
        if attribute in internal:
            return value
        elif isinstance(value, csr_matrix):
            return value[indices][:, indices]
        elif isinstance(value, (np.ndarray, BaseCoordinateFrame,
                                SkyCoord, SkyOffsetFrame, EarthLocation,
                                Coordinate)):
            return value[indices]
        else:
            return value

    def delete_indices(self, indices_or_mask):
        """
        Completely deletes data elements.

        Actual indices should be passed in.  To delete based on fixed index
        values, please convert first using `find_fixed_indices`.

        Parameters
        ----------
        indices_or_mask : numpy.ndarray of (bool or int)
            The indices to delete, or a boolean mask where `True` marks an
            element for deletion.

        Returns
        -------
        None
        """
        keep_indices = self.to_indices(indices_or_mask, discard=True)
        keep_indices = np.unique(keep_indices)
        if ((keep_indices.size == self.size)
                and np.allclose(keep_indices, np.arange(self.size))):
            return

        special_fields = self.special_fields
        internal = self.internal_attributes

        for key, value in self.__dict__.items():
            if key in special_fields or key in internal:
                continue  # pragma: no cover
            if isinstance(value, csr_matrix):
                setattr(self, key, value[keep_indices][:, keep_indices])
            elif isinstance(value, (np.ndarray, BaseCoordinateFrame,
                                    SkyCoord, SkyOffsetFrame, EarthLocation,
                                    Coordinate)):
                setattr(self, key, value[keep_indices])

    def insert_blanks(self, insert_indices):
        """
        Inserts blank frame data.

        Actual indices should be passed in.  To delete based on fixed index
        values, please convert first using `find_fixed_indices`.

        Blank data are set to 0 in whatever unit is applicable.

        Parameters
        ----------
        insert_indices : numpy.ndarray of (int)
            The index locations to insert.

        Returns
        -------
        None
        """
        special = self.special_fields
        internal = self.internal_attributes

        for key, value in self.__dict__.items():
            if key in special or key in internal:
                continue  # pragma: no cover
            if isinstance(value, EarthLocation):
                lat = np.insert(value.lat, insert_indices, 0.0)
                lon = np.insert(value.lon, insert_indices, 0.0)
                height = np.insert(value.height, insert_indices, 0.0)
                setattr(self, key,
                        EarthLocation(lat=lat, lon=lon, height=height))
            elif isinstance(value,
                            (BaseCoordinateFrame, SkyCoord, SkyOffsetFrame)):
                setattr(
                    self, key, skycoord_insert_blanks(value, insert_indices))
            elif isinstance(value, Coordinate):
                value.insert_blanks(insert_indices)
            elif isinstance(value, csr_matrix):
                value = value.toarray()
                value = np.insert(value, insert_indices, 0.0, axis=0)
                value = np.insert(value, insert_indices, 0.0, axis=1)
                setattr(self, key, csr_matrix(value))
            elif isinstance(value, np.ndarray):
                if key == 'fixed_index' or key == '_fixed_index':
                    blank = -1
                else:
                    blank = 0
                setattr(self, key,
                        np.insert(value, insert_indices, blank, axis=0))

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
        special_fields = self.special_fields
        internal = self.internal_attributes

        for attribute, value in self.__dict__.items():
            if attribute in special_fields or attribute in internal:
                continue  # pragma: no cover
            if value is None:
                continue
            new_value = getattr(data, attribute, None)
            if new_value is None:
                continue
            if isinstance(value, EarthLocation):
                lat = np.concatenate((value.lat, new_value.lat))
                lon = np.concatenate((value.lon, new_value.lon))
                height = np.concatenate((value.height, new_value.height))
                setattr(self, attribute,
                        EarthLocation(lat=lat, lon=lon, height=height))
            elif isinstance(value,
                            (BaseCoordinateFrame, SkyCoord, SkyOffsetFrame)):
                setattr(self, attribute, astro_concat((value, new_value)))
            elif isinstance(value, Coordinate):
                value.merge(new_value)
            elif isinstance(value, csr_matrix):
                nzi1 = value.nonzero()
                nzi2 = new_value.nonzero()
                full_data = np.concatenate((value.data, new_value.data))
                rows = np.concatenate((nzi1[0], nzi2[0] + value.shape[0]))
                cols = np.concatenate((nzi1[1], nzi2[1] + value.shape[1]))
                new_shape = (value.shape[0] + new_value.shape[0],
                             value.shape[1] + new_value.shape[1])
                setattr(self, attribute,
                        csr_matrix((full_data, (rows, cols)), shape=new_shape))
            elif isinstance(value, np.ndarray):
                setattr(self, attribute,
                        np.concatenate((value, new_value), axis=0))

    def get_index_size(self, indices=None):
        """
        Get indices and the size of indices covered.

        Parameters
        ----------
        indices : int or numpy.ndarray (int) or slice

        Returns
        -------
        indices, size : slice or int or numpy.ndarray (int), int
        """
        if indices is None:
            indices = slice(None)
            size = self.size
        elif isinstance(indices, slice):
            size = len(range(*indices.indices(self.size)))
        elif isinstance(indices, np.ndarray):
            size = indices.size
        elif isinstance(indices, int):
            size = 0
        else:
            raise ValueError("Incorrect indices format")
        return indices, size
